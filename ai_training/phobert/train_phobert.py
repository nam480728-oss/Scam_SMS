"""Fine-tune PhoBERT-base-v2 de phan loai tin nhan lua dao (0 = binh thuong, 1 = lua dao).

Chay (tu thu muc ai_training/phobert):
    python train_phobert.py                      # dung config.yaml
    python train_phobert.py --smoke-test         # chay thu rat nhanh (64 mau, 1 epoch) de kiem tra moi truong
    python train_phobert.py --set train.epochs=5 --set train.learning_rate=3e-5

Toan bo ket qua (tham so, log, so lieu, bieu do, mo hinh tot nhat, bao cao) nam trong
outputs/<run_name>_<thoi gian>/. Xem README.md de biet y nghia tung file.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import eval_utils as eu  # noqa: E402

ROOT = Path(__file__).resolve().parent
MONITOR_KEYS = {"macro_f1": "val_macro_f1", "accuracy": "val_accuracy", "f1_scam": "val_f1_scam",
                "val_loss": "val_loss"}


# =============================================================================
# 1. CAU HINH
# =============================================================================
def load_config(path: Path, sets: list[str]) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in sets or []:  # --set train.epochs=5
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Khong co tham so '{key}' trong config")
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# =============================================================================
# 2. TIEN XU LY VAN BAN
# =============================================================================
def normalize_text(text: str, cfg_text: dict) -> str:
    if cfg_text.get("normalize_whitespace", True):
        text = re.sub(r"\s+", " ", text).strip()
    if cfg_text.get("lowercase", False):
        text = text.lower()
    return text


def build_segmenter(cfg_text: dict):
    if not cfg_text.get("word_segment", True):
        return lambda t: t
    if cfg_text.get("segmenter", "pyvi") != "pyvi":
        raise ValueError("Hien chi ho tro segmenter: pyvi")
    try:
        from pyvi import ViTokenizer
    except ImportError as e:  # noqa: BLE001
        raise SystemExit("Thieu thu vien pyvi. Cai bang: pip install pyvi  (xem README.md)") from e
    return ViTokenizer.tokenize


def prepare_texts(df: pd.DataFrame, cfg_text: dict, segment) -> tuple[list[str], list[str]]:
    norm = [normalize_text(t, cfg_text) for t in df["content"]]
    seg = [segment(t) for t in norm]
    return norm, seg


class EncodedDataset(Dataset):
    def __init__(self, input_ids: list[list[int]], labels: list[int]):
        self.input_ids, self.labels = input_ids, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.input_ids[i], self.labels[i]


def make_collate(pad_id: int):
    def collate(batch):
        longest = max(len(ids) for ids, _ in batch)  # padding dong: chi pad den do dai lon nhat batch
        ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, (x, _) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        return ids, mask, torch.tensor([y for _, y in batch], dtype=torch.long)

    return collate


# =============================================================================
# 3. DANH GIA / DU DOAN
# =============================================================================
def make_amp(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext, None, False
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return (lambda: torch.autocast(device_type="cuda", dtype=dtype)), dtype, True


@torch.no_grad()
def predict(model, loader, device, amp_ctx, class_w):
    """Tra ve (loss co trong so lop, xac suat lop 1, nhan that)."""
    model.eval()
    logits_all, labels_all = [], []
    for ids, mask, y in loader:
        with amp_ctx():
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        logits_all.append(logits.float().cpu())
        labels_all.append(y)
    logits, labels = torch.cat(logits_all), torch.cat(labels_all)
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    w = class_w[labels] if class_w is not None else torch.ones_like(per_sample)
    loss = float((per_sample * w).sum() / w.sum())
    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return loss, probs, labels.numpy()


# =============================================================================
# 4. VE BIEU DO QUA TRINH TRAIN
# =============================================================================
def plot_history(hist: pd.DataFrame, steps: pd.DataFrame, best_epoch: int, out: Path, monitor_name: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(hist["epoch"], hist["train_loss"], "o-", label="Train loss")
    ax.plot(hist["epoch"], hist["val_loss"], "s-", label="Validation loss")
    ax.axvline(best_epoch, color="gray", ls=":", label=f"Epoch tốt nhất ({best_epoch})")
    ax.set(xlabel="Epoch", ylabel="Loss (có trọng số lớp)", title="Đường cong loss theo epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    eu._save(fig, out / "plots" / "01_loss_curve.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, lab in (("val_macro_f1", "macro-F1"), ("val_accuracy", "Accuracy"), ("val_f1_scam", "F1 (lừa đảo)"),
                     ("val_precision_scam", "Precision (lừa đảo)"), ("val_recall_scam", "Recall (lừa đảo)")):
        ax.plot(hist["epoch"], hist[col], "o-", label=lab)
    ax.axvline(best_epoch, color="gray", ls=":")
    ax.set(xlabel="Epoch", ylabel="Giá trị", title=f"Chỉ số trên tập validation theo epoch (chọn theo {monitor_name})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    eu._save(fig, out / "plots" / "02_val_metrics_curve.png")

    if len(steps):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        axes[0].plot(steps["step"], steps["lr"])
        axes[0].set(xlabel="Bước tối ưu", ylabel="Learning rate", title="Lịch learning rate (warmup + giảm dần)")
        axes[1].plot(steps["step"], steps["loss"], alpha=0.35, label="loss")
        axes[1].plot(steps["step"], steps["loss"].rolling(10, min_periods=1).mean(), label="trung bình trượt (10)")
        axes[1].set(xlabel="Bước tối ưu", ylabel="Train loss", title="Train loss theo bước")
        axes[1].legend()
        axes[2].plot(steps["step"], steps["grad_norm"])
        axes[2].axhline(1.0, color="r", ls="--", lw=0.8, label="ngưỡng cắt (max_grad_norm)")
        axes[2].set(xlabel="Bước tối ưu", ylabel="Norm gradient (trước khi cắt)", title="Độ lớn gradient")
        axes[2].legend()
        for a in axes:
            a.grid(alpha=0.3)
        eu._save(fig, out / "plots" / "03_lr_loss_gradnorm_steps.png")


# =============================================================================
# 5. NHAN XET TU DONG + BAO CAO
# =============================================================================
def auto_comments(hist: pd.DataFrame, best_epoch: int, stopped_early: bool, metrics: dict,
                  group_tbl: pd.DataFrame | None, trunc_rate: float, thr_info: dict, cfg: dict) -> list[str]:
    c = []
    n_ep, max_ep = len(hist), cfg["train"]["epochs"]
    best = hist[hist["epoch"] == best_epoch].iloc[0]
    c.append(f"Mô hình huấn luyện {n_ep}/{max_ep} epoch; epoch tốt nhất là **{best_epoch}** "
             f"(val macro-F1 = {best['val_macro_f1']:.4f}, val loss = {best['val_loss']:.4f})."
             + (" Dừng sớm vì không còn cải thiện." if stopped_early else " Chạy hết số epoch cấu hình."))
    gap = float(best["val_loss"] - best["train_loss"])
    if gap > 0.15:
        c.append(f"Val loss cao hơn train loss {gap:.3f} tại epoch tốt nhất → có dấu hiệu **overfitting**. "
                 "Cân nhắc tăng dropout/weight_decay, giảm epoch hoặc thêm dữ liệu.")
    else:
        c.append(f"Chênh lệch val loss − train loss tại epoch tốt nhất chỉ {gap:.3f} → mô hình khớp tương đối cân bằng.")
    if n_ep > best_epoch and hist["val_loss"].iloc[-1] > best["val_loss"] * 1.2:
        c.append("Val loss tăng rõ sau epoch tốt nhất trong khi chỉ số phân loại ít đổi: dấu hiệu quá tự tin (over-confident).")
    tr, va, te = metrics["train"], metrics["val"], metrics["test"]
    c.append(f"Kết quả cuối (ngưỡng 0.5): train macro-F1 {tr['macro_f1']:.4f}, val {va['macro_f1']:.4f}, test {te['macro_f1']:.4f}.")
    if tr["macro_f1"] - te["macro_f1"] > 0.05:
        c.append(f"Train cao hơn test {tr['macro_f1'] - te['macro_f1']:.3f} macro-F1 → khoảng cách tổng quát hoá đáng kể.")
    if abs(va["macro_f1"] - te["macro_f1"]) > 0.03:
        c.append(f"Val và test lệch nhau {abs(va['macro_f1'] - te['macro_f1']):.3f} macro-F1 (tập nhỏ nên dao động là bình thường).")
    c.append(f"Trên test: precision lừa đảo {te['precision_scam']:.4f}, recall lừa đảo {te['recall_scam']:.4f}; "
             f"bỏ sót {te['fn']} tin lừa đảo (FN) và báo nhầm {te['fp']} tin thường (FP).")
    if te["recall_scam"] + 0.03 < te["precision_scam"]:
        c.append("Mô hình thiên về **an toàn cho tin thường** (precision > recall): có thể hạ ngưỡng để bắt thêm tin lừa đảo.")
    elif te["precision_scam"] + 0.03 < te["recall_scam"]:
        c.append("Mô hình thiên về **bắt lừa đảo** (recall > precision): nhiều cảnh báo nhầm, có thể nâng ngưỡng.")
    if "roc_auc" in te:
        c.append(f"ROC-AUC test = {te['roc_auc']:.4f}, PR-AUC = {te['pr_auc']:.4f}.")
    c.append(f"Ngưỡng tối ưu F1 (chọn trên validation) = {thr_info['threshold']:.2f}; áp dụng cho test cho macro-F1 "
             f"= {thr_info['test_macro_f1']:.4f} (so với {te['macro_f1']:.4f} ở ngưỡng 0.5).")
    if group_tbl is not None and not group_tbl.empty:
        g = group_tbl[group_tbl["loai_nhom"] == "nhom_dau"]
        if len(g) == 2:
            a, b = g.sort_values("macro_f1").iloc[0], g.sort_values("macro_f1").iloc[-1]
            c.append(f"Theo nhóm trên test: tin '{a['nhom']}' macro-F1 {a['macro_f1']:.4f} (n={int(a['n'])}) so với "
                     f"'{b['nhom']}' {b['macro_f1']:.4f} (n={int(b['n'])}). "
                     + ("PhoBERT đọc tin không dấu kém hơn do được pretrain trên tiếng Việt có dấu."
                        if a["nhom"] == "không dấu" and b["macro_f1"] - a["macro_f1"] > 0.02 else ""))
    if trunc_rate > 0.01:
        c.append(f"{trunc_rate:.1%} tin trong tập train bị cắt do vượt max_length — cân nhắc tăng max_length (tối đa 256).")
    else:
        c.append(f"Chỉ {trunc_rate:.2%} tin train bị cắt bởi max_length → độ dài {cfg['model']['max_length']} là đủ.")
    c.append("Lưu ý: dữ liệu có nhiều tin do máy sinh theo khuôn nên điểm có thể cao hơn khi gặp tin thật; "
             "cần kiểm tra thêm trên tập tin thật độc lập.")
    return c


def write_report(path: Path, ctx: dict):
    cfg, run = ctx["cfg"], ctx["run_dir"]
    h = ctx["hist"]
    lines = [f"# Báo cáo huấn luyện PhoBERT-base-v2 — {ctx['run_name']}", "",
             f"- Thời điểm: {ctx['started']}  |  Tổng thời gian: {ctx['total_seconds'] / 60:.1f} phút",
             f"- Thiết bị: {ctx['device_name']}  |  Độ chính xác số: {cfg['train']['precision']}",
             f"- Checkpoint tốt nhất: epoch {ctx['best_epoch']} → `best_model/`", "", "## 1. Dữ liệu", "",
             ctx["data_table"], "", "![Phân bố nhãn](plots/00_class_distribution.png)",
             "![Phân bố độ dài](plots/00_length_distribution.png)", "",
             f"Kiểm tra trùng nội dung giữa các tập (phải = 0): `{ctx['overlap']}`", "",
             "## 2. Tiền xử lý & token hoá", "",
             "- Chuẩn hoá khoảng trắng → tách từ bằng pyvi → BPE của PhoBERT (xem `data/segmentation_samples.xlsx`).",
             f"- Độ dài token (train): {ctx['token_stats']}",
             f"- Tỉ lệ tin bị cắt bởi max_length={cfg['model']['max_length']}: {ctx['trunc_rate']:.2%}",
             "![Phân bố độ dài token](plots/00_token_length.png)", "",
             "## 3. Mô hình & tham số", "", "![Sơ đồ pipeline](plots/00_pipeline.png)", "",
             ctx["param_table"], "", "Toàn bộ tham số huấn luyện: xem `config_used.yaml`.", "",
             "## 4. Quá trình huấn luyện", "", eu.md_table(h.round(4)), "",
             "![Loss](plots/01_loss_curve.png)", "![Chỉ số val](plots/02_val_metrics_curve.png)",
             "![Bước](plots/03_lr_loss_gradnorm_steps.png)", "", "## 5. Kết quả cuối cùng (ngưỡng 0.5)", "",
             ctx["metrics_table"], "",
             "![ROC/PR](plots/10_roc_pr.png)", "![CM val](plots/11_confusion_val.png)",
             "![CM test](plots/12_confusion_test.png)", "![Điểm test](plots/13_score_hist_test.png)", "",
             "### Phân tích ngưỡng quyết định", "", "![Ngưỡng val](plots/14_threshold_val.png)",
             f"Ngưỡng tối ưu trên val = **{ctx['thr']['threshold']:.2f}** → macro-F1 test = {ctx['thr']['test_macro_f1']:.4f}.", "",
             "### Phân tích theo nhóm (tập test)", "", ctx["group_table"], "",
             "![Nhóm](plots/15_group_test.png)", "", "## 6. Báo cáo phân loại chi tiết (test)", "", "```",
             ctx["class_report_test"], "```", "", "## 7. Phân tích lỗi", "",
             "Danh sách dòng dự đoán sai: `errors_val.xlsx`, `errors_test.xlsx`; toàn bộ dự đoán: `predictions_*.csv`.", "",
             "## 8. Nhận xét tự động", ""]
    lines += [f"{i}. {t}" for i, t in enumerate(ctx["comments"], 1)]
    lines += ["", "## 9. Danh sách file kết quả", "",
              "| File | Nội dung |", "|---|---|",
              "| config_used.yaml | Toàn bộ tham số đã dùng |", "| environment.json | Phiên bản thư viện, GPU |",
              "| history.csv / step_log.csv | Log theo epoch / theo bước tối ưu |",
              "| metrics.json / metrics_summary.csv | Chỉ số mọi tập |",
              "| model_architecture.txt, model_params.json | Kiến trúc và số tham số |",
              "| best_model/ | Mô hình + tokenizer tốt nhất (dùng với predict_phobert.py) |",
              "| plots/ | Toàn bộ biểu đồ |", "| timing.json | Thời gian từng giai đoạn |"]
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 6. CHUONG TRINH CHINH
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Fine-tune PhoBERT-base-v2 phan loai tin nhan lua dao")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="ghi de tham so, vi du --set train.epochs=5 (co the lap lai)")
    ap.add_argument("--smoke-test", action="store_true", help="chay thu nhanh: 64/32/32 mau, 1 epoch")
    ap.add_argument("--output-dir", default=None, help="thu muc ket qua (mac dinh outputs/<run>_<time>)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config), args.set)
    if args.smoke_test:
        cfg["train"]["epochs"] = 1
        cfg["run_name"] += "_smoke"
    timer = eu.Timer()
    t_start = time.perf_counter()
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_dir = Path(args.output_dir) if args.output_dir else \
        ROOT / cfg["output"]["dir"] / f"{cfg['run_name']}_{datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    print(f"[+] Ket qua se luu tai: {run_dir}")

    set_seed(cfg["seed"], cfg["train"]["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    if device.type == "cpu":
        print("[!] KHONG phat hien GPU CUDA -> chay bang CPU se rat cham. Xem README.md de cai torch CUDA.")
    amp_ctx, amp_dtype, use_amp = make_amp(cfg["train"]["precision"], device)
    print(f"[+] Thiet bi: {device_name} | precision={cfg['train']['precision'] if use_amp else 'fp32'}")

    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    eu.save_json(eu.environment_info(), run_dir / "environment.json")

    # ---------------------------------------------------------------- du lieu
    data_dir = (ROOT / cfg["data"]["data_dir"]).resolve()
    dfs = {"train": eu.load_split(data_dir / cfg["data"]["train_file"]),
           "val": eu.load_split(data_dir / cfg["data"]["val_file"]),
           "test": eu.load_split(data_dir / cfg["data"]["test_file"])}
    if args.smoke_test:
        dfs = {k: v.groupby("label").sample(n=(32 if k == "train" else 16), random_state=1).reset_index(drop=True)
               for k, v in dfs.items()}
    overlap = eu.check_no_overlap(dfs)
    print(f"[+] Kich thuoc: " + ", ".join(f"{k}={len(v)}" for k, v in dfs.items()) + f" | trung giua cac tap: {overlap}")
    if any(overlap.values()):
        print("[!] CANH BAO: co noi dung trung giua cac tap -> ket qua bi thoi phong.")
    eu.plot_class_distribution(dfs, run_dir / "plots" / "00_class_distribution.png")
    eu.plot_length_distribution(dfs, run_dir / "plots" / "00_length_distribution.png")
    data_rows = [{"tap": k, "so_dong": len(v), "thuong_0": int((v.label == 0).sum()), "lua_dao_1": int((v.label == 1).sum()),
                  "ty_le_lua_dao": round(float(v.label.mean()), 4),
                  "ty_le_khong_dau": round(float(1 - v.content.map(eu.has_diacritics).mean()), 4),
                  "do_dai_trung_vi": int(v.content.str.len().median())} for k, v in dfs.items()]
    data_df = pd.DataFrame(data_rows)
    data_df.to_csv(run_dir / "data" / "data_summary.csv", index=False)

    # ----------------------------------------------------------- tien xu ly
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
        get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    with timer("tien_xu_ly_va_token_hoa"):
        segment = build_segmenter(cfg["text"])
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
        max_len = cfg["model"]["max_length"]
        texts, encoded, labels = {}, {}, {}
        for k, df in dfs.items():
            norm, seg = prepare_texts(df, cfg["text"], segment)
            texts[k] = (norm, seg)
            encoded[k] = tokenizer(seg, truncation=True, max_length=max_len)["input_ids"]
            labels[k] = df["label"].tolist()
        untrunc = [len(x) for x in tokenizer(texts["train"][1], truncation=False)["input_ids"]]
    lens = np.asarray(untrunc)
    trunc_rate = float((lens > max_len).mean())
    token_stats = {"min": int(lens.min()), "trung_vi": int(np.median(lens)), "p90": int(np.percentile(lens, 90)),
                   "p95": int(np.percentile(lens, 95)), "p99": int(np.percentile(lens, 99)), "max": int(lens.max())}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(np.clip(lens, 0, 400), bins=40, color="#4C78A8")
    ax.axvline(max_len, color="r", ls="--", label=f"max_length={max_len}")
    ax.set(xlabel="Số token / tin (cắt ở 400)", ylabel="Số tin", title="Phân bố độ dài token (train)")
    ax.legend()
    eu._save(fig, run_dir / "plots" / "00_token_length.png")
    n_show = min(30, len(dfs["train"]))
    pd.DataFrame({"goc": dfs["train"]["content"][:n_show], "chuan_hoa": texts["train"][0][:n_show],
                  "tach_tu": texts["train"][1][:n_show],
                  "token_bpe": [" ".join(tokenizer.convert_ids_to_tokens(x)) for x in encoded["train"][:n_show]],
                  "nhan": labels["train"][:n_show]}).to_excel(run_dir / "data" / "segmentation_samples.xlsx", index=False)

    collate = make_collate(tokenizer.pad_token_id)
    tcfg = cfg["train"]
    loaders = {
        "train": DataLoader(EncodedDataset(encoded["train"], labels["train"]), batch_size=tcfg["batch_size"],
                            shuffle=True, collate_fn=collate, num_workers=tcfg["num_workers"]),
        "train_eval": DataLoader(EncodedDataset(encoded["train"], labels["train"]), batch_size=tcfg["eval_batch_size"],
                                 shuffle=False, collate_fn=collate, num_workers=tcfg["num_workers"]),
    }
    for k in ("val", "test"):
        loaders[k] = DataLoader(EncodedDataset(encoded[k], labels[k]), batch_size=tcfg["eval_batch_size"],
                                shuffle=False, collate_fn=collate, num_workers=tcfg["num_workers"])

    # ------------------------------------------------------------- mo hinh
    with timer("tai_mo_hinh"):
        mkw = {"num_labels": cfg["model"]["num_labels"],
               "hidden_dropout_prob": cfg["model"]["hidden_dropout_prob"],
               "attention_probs_dropout_prob": cfg["model"]["attention_probs_dropout_prob"]}
        if cfg["model"]["classifier_dropout"] is not None:
            mkw["classifier_dropout"] = cfg["model"]["classifier_dropout"]
        model = AutoModelForSequenceClassification.from_pretrained(cfg["model"]["name"], **mkw).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parts = {"embeddings": sum(p.numel() for p in model.roberta.embeddings.parameters()),
             "encoder (12 lop transformer)": sum(p.numel() for p in model.roberta.encoder.parameters()),
             "classifier head": sum(p.numel() for p in model.classifier.parameters())}
    eu.save_json({"tong_tham_so": n_total, "tham_so_huan_luyen": n_train, "theo_khoi": parts,
                  "model_config": model.config.to_dict()}, run_dir / "model_params.json")
    (run_dir / "model_architecture.txt").write_text(str(model), encoding="utf-8")
    param_table = eu.md_table(pd.DataFrame([{"khoi": k, "so_tham_so": v} for k, v in {"Tổng": n_total, **parts}.items()]))
    print(f"[+] Tong tham so: {n_total / 1e6:.1f}M (huan luyen {n_train / 1e6:.1f}M)")
    eu.draw_pipeline_diagram(
        [("Tin nhắn", "văn bản thô\n(có dấu / không dấu)"), ("Tiền xử lý", "chuẩn hoá khoảng trắng\ntách từ (pyvi)"),
         ("Tokenizer", f"BPE PhoBERT\nmax_length={max_len}"), ("PhoBERT-base-v2", "12 lớp Transformer\nhidden=768"),
         ("Classifier", "Linear 768→2\nsoftmax"), ("Kết quả", "0 = thường\n1 = lừa đảo")],
        run_dir / "plots" / "00_pipeline.png", "Sơ đồ pipeline phân loại tin nhắn lừa đảo với PhoBERT")

    # ----------------------------------------- toi uu hoa / loss / scheduler
    no_decay = ("bias", "LayerNorm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
               "weight_decay": tcfg["weight_decay"]},
              {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
               "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=tcfg["learning_rate"], eps=tcfg["adam_epsilon"],
                                  betas=(tcfg["adam_beta1"], tcfg["adam_beta2"]))
    accum = tcfg["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(loaders["train"]) / accum)
    total_steps = steps_per_epoch * tcfg["epochs"]
    warmup = int(total_steps * tcfg["warmup_ratio"])
    sched_fn = get_cosine_schedule_with_warmup if tcfg["lr_scheduler"] == "cosine" else get_linear_schedule_with_warmup
    scheduler = sched_fn(optimizer, warmup, total_steps)
    if tcfg["class_weight"] == "balanced":
        cnt = np.bincount(labels["train"], minlength=2)
        class_w = torch.tensor(len(labels["train"]) / (2.0 * cnt), dtype=torch.float)
    else:
        class_w = None
    print(f"[+] Trong so lop: {None if class_w is None else class_w.numpy().round(3).tolist()} | "
          f"buoc/epoch={steps_per_epoch}, tong buoc={total_steps}, warmup={warmup}")
    train_loss_fn = torch.nn.CrossEntropyLoss(weight=None if class_w is None else class_w.to(device),
                                              label_smoothing=tcfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # ------------------------------------------------------------ vong train
    es = cfg["early_stopping"]
    mon_key = MONITOR_KEYS[es["monitor"]]
    maximize = es["monitor"] != "val_loss"
    best_score, best_epoch, bad_epochs, best_state = None, 0, 0, None
    history, step_log, stopped_early = [], [], False
    global_step = 0
    best_dir = run_dir / "best_model"
    with timer("huan_luyen"):
        for epoch in range(1, tcfg["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0, loss_sum, n_seen = time.perf_counter(), 0.0, 0
            try:
                from tqdm import tqdm
                it = tqdm(loaders["train"], desc=f"Epoch {epoch}/{tcfg['epochs']}", leave=False)
            except ImportError:
                it = loaders["train"]
            for i, (ids, mask, y) in enumerate(it, 1):
                with amp_ctx():
                    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                loss = train_loss_fn(logits.float(), y.to(device))
                scaler.scale(loss / accum).backward()
                loss_sum += loss.item() * len(y)
                n_seen += len(y)
                if i % accum == 0 or i == len(loaders["train"]):
                    scaler.unscale_(optimizer)
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % tcfg["log_every_steps"] == 0 or global_step == 1:
                        step_log.append({"step": global_step, "epoch": epoch, "loss": loss.item(),
                                         "lr": scheduler.get_last_lr()[0], "grad_norm": gn})
            secs = time.perf_counter() - t0
            v_loss, v_prob, v_true = predict(model, loaders["val"], device, amp_ctx, class_w)
            vm = eu.compute_metrics(v_true, (v_prob >= 0.5).astype(int), v_prob)
            row = {"epoch": epoch, "train_loss": loss_sum / n_seen, "val_loss": v_loss,
                   "val_accuracy": vm["accuracy"], "val_precision_scam": vm["precision_scam"],
                   "val_recall_scam": vm["recall_scam"], "val_f1_scam": vm["f1_scam"],
                   "val_macro_f1": vm["macro_f1"], "val_roc_auc": vm.get("roc_auc"),
                   "lr_cuoi_epoch": scheduler.get_last_lr()[0], "giay_epoch": round(secs, 1),
                   "mau_moi_giay": round(n_seen / secs, 1),
                   "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None}
            history.append(row)
            score = row[mon_key]
            improved = best_score is None or (score > best_score + es["min_delta"] if maximize
                                              else score < best_score - es["min_delta"])
            print(f"[Epoch {epoch}] train_loss={row['train_loss']:.4f} val_loss={v_loss:.4f} "
                  f"val_macroF1={vm['macro_f1']:.4f} acc={vm['accuracy']:.4f} ({secs:.0f}s)"
                  + ("  <-- tot nhat" if improved else ""))
            if improved:
                best_score, best_epoch, bad_epochs = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if cfg["output"]["save_best_model"]:
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    eu.save_json({"model_name": cfg["model"]["name"], "max_length": max_len,
                                  "text": cfg["text"], "best_epoch": epoch, "threshold_default": 0.5},
                                 best_dir / "inference_config.json")
            else:
                bad_epochs += 1
                if bad_epochs >= es["patience"]:
                    print(f"[+] Dung som: {es['monitor']} khong cai thien sau {es['patience']} epoch.")
                    stopped_early = True
                    break
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            pd.DataFrame(step_log).to_csv(run_dir / "step_log.csv", index=False)

    hist, steps = pd.DataFrame(history), pd.DataFrame(step_log)
    hist.to_csv(run_dir / "history.csv", index=False)
    steps.to_csv(run_dir / "step_log.csv", index=False)
    plot_history(hist, steps, best_epoch, run_dir, es["monitor"])

    # ------------------------------------------------- danh gia cuoi cung
    model.load_state_dict(best_state)
    results, probs_out = {}, {}
    with timer("danh_gia_cuoi"):
        for name, key in (("train", "train_eval"), ("val", "val"), ("test", "test")):
            loss, prob, ytrue = predict(model, loaders[key], device, amp_ctx, class_w)
            pred = (prob >= 0.5).astype(int)
            results[name] = eu.compute_metrics(ytrue, pred, prob)
            results[name]["loss"] = loss
            probs_out[name] = (ytrue, prob, pred)
            out_df = dfs[name].copy()
            out_df["xac_suat_lua_dao"], out_df["du_doan"] = prob.round(5), pred
            out_df["so_token"] = [len(x) for x in encoded[name]]
            out_df.to_csv(run_dir / f"predictions_{name}.csv", index=False, encoding="utf-8-sig")
    eu.plot_roc_pr({k: (v[0], v[1]) for k, v in probs_out.items()}, run_dir / "plots" / "10_roc_pr.png")
    eu.plot_confusion_matrix(results["val"]["confusion_matrix"], run_dir / "plots" / "11_confusion_val.png",
                             "Ma trận nhầm lẫn — Validation")
    eu.plot_confusion_matrix(results["test"]["confusion_matrix"], run_dir / "plots" / "12_confusion_test.png",
                             "Ma trận nhầm lẫn — Test")
    eu.plot_score_hist(probs_out["test"][0], probs_out["test"][1], run_dir / "plots" / "13_score_hist_test.png",
                       "Phân bố xác suất lừa đảo — Test")
    eu.plot_score_hist(probs_out["val"][0], probs_out["val"][1], run_dir / "plots" / "13_score_hist_val.png",
                       "Phân bố xác suất lừa đảo — Validation")
    thr_val = eu.threshold_analysis(probs_out["val"][0], probs_out["val"][1])
    thr_val.to_csv(run_dir / "threshold_val.csv", index=False)
    eu.plot_threshold_curve(thr_val, run_dir / "plots" / "14_threshold_val.png", "Precision/Recall/F1 theo ngưỡng — Validation")
    best_t = float(thr_val.loc[thr_val["macro_f1"].idxmax(), "threshold"])
    tm = eu.compute_metrics(probs_out["test"][0], (probs_out["test"][1] >= best_t).astype(int), probs_out["test"][1])
    thr_info = {"threshold": best_t, "test_macro_f1": tm["macro_f1"], "test_metrics": tm}
    results["test_at_val_threshold"] = tm
    group_val = eu.group_metrics(dfs["val"], probs_out["val"][2], probs_out["val"][1])
    group_test = eu.group_metrics(dfs["test"], probs_out["test"][2], probs_out["test"][1])
    group_val.to_csv(run_dir / "group_metrics_val.csv", index=False)
    group_test.to_csv(run_dir / "group_metrics_test.csv", index=False)
    eu.plot_group_metrics(group_test, run_dir / "plots" / "15_group_test.png", "Chỉ số theo nhóm — Test")
    n_err = cfg["output"]["n_error_samples"]
    with pd.ExcelWriter(run_dir / "errors_val.xlsx") as w1, pd.ExcelWriter(run_dir / "errors_test.xlsx") as w2:
        eu.error_table(dfs["val"], probs_out["val"][2], probs_out["val"][1]).head(n_err).to_excel(w1, index=False)
        eu.error_table(dfs["test"], probs_out["test"][2], probs_out["test"][1]).head(n_err).to_excel(w2, index=False)
    eu.save_json(results, run_dir / "metrics.json")
    eu.metrics_table({k: v for k, v in results.items() if k != "test_at_val_threshold"}).round(4).to_csv(
        run_dir / "metrics_summary.csv")
    class_report = eu.text_report(probs_out["test"][0], probs_out["test"][2])
    (run_dir / "classification_report_test.txt").write_text(class_report, encoding="utf-8")

    # ---------------------------------------------------------- bao cao
    total_s = time.perf_counter() - t_start
    timer.times["tong"] = round(total_s, 1)
    eu.save_json(timer.times, run_dir / "timing.json")
    comments = auto_comments(hist, best_epoch, stopped_early, results, group_test, trunc_rate, thr_info, cfg)
    (run_dir / "nhan_xet_tu_dong.md").write_text("\n".join(f"{i}. {t}" for i, t in enumerate(comments, 1)), encoding="utf-8")
    write_report(run_dir / "report.md", {
        "cfg": cfg, "run_dir": run_dir, "run_name": cfg["run_name"], "started": started, "total_seconds": total_s,
        "device_name": device_name, "best_epoch": best_epoch, "hist": hist, "data_table": eu.md_table(data_df),
        "overlap": overlap, "token_stats": token_stats, "trunc_rate": trunc_rate, "param_table": param_table,
        "metrics_table": eu.md_table(eu.metrics_table({k: v for k, v in results.items() if k != "test_at_val_threshold"}).round(4), index=True),
        "thr": thr_info, "group_table": eu.md_table(group_test.round(4)),
        "class_report_test": class_report, "comments": comments})
    print("\n" + "=" * 70)
    for k in ("train", "val", "test"):
        r = results[k]
        print(f"{k:5s} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} F1_scam={r['f1_scam']:.4f} "
              f"P={r['precision_scam']:.4f} R={r['recall_scam']:.4f} AUC={r.get('roc_auc', float('nan')):.4f}")
    print(f"Bao cao day du: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
