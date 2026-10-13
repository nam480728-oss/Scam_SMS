"""Tien ich danh gia + ve bieu do dung chung cho bai toan phan loai tin nhan lua dao.

LUU Y: file nay giong het nhau o hai thu muc `phobert/` va `IDF_SVM/` de moi thu muc
tu chua du moi thu (de quan ly, de dong goi). Neu sua o day thi sua ca ben kia.

Quy uoc: nhan 0 = tin binh thuong, nhan 1 = tin lua dao. `y_score` la diem/xac suat
cua lop 1 (cang cao cang giong lua dao).
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ve ra file, khong can man hinh
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

CLASS_NAMES = ["Bình thường (0)", "Lừa đảo (1)"]
DIACRITIC_CHARS = "àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
SPLIT_COLORS = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}


# --------------------------------------------------------------------------- #
# Doc du lieu
# --------------------------------------------------------------------------- #
def load_split(path: str | Path) -> pd.DataFrame:
    """Doc 1 file xlsx (cot `content`, `label`) va kiem tra tinh hop le."""
    df = pd.read_excel(path)
    missing = {"content", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: thieu cot {missing}")
    df = df[["content", "label"]].copy()
    if df["label"].isna().any():
        raise ValueError(f"{path}: con dong khong co nhan")
    df["content"] = df["content"].astype(str)
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError(f"{path}: nhan phai la 0/1")
    return df.reset_index(drop=True)


def has_diacritics(text: str) -> bool:
    return any(c in DIACRITIC_CHARS for c in text.lower())


def check_no_overlap(dfs: dict[str, pd.DataFrame]) -> dict:
    """Kiem tra ro ri: noi dung giong het nhau giua cac tap."""
    names = list(dfs)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}&{b}"] = len(set(dfs[a]["content"]) & set(dfs[b]["content"]))
    return out


# --------------------------------------------------------------------------- #
# Chi so
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    m = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_scam": float(p[1]),
        "recall_scam": float(r[1]),
        "f1_scam": float(f[1]),
        "precision_normal": float(p[0]),
        "recall_normal": float(r[0]),
        "f1_normal": float(f[0]),
        "macro_f1": float(np.mean(f)),
        "weighted_f1": float(np.average(f, weights=s)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,  # tin thuong bi bao lua dao
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,  # lua dao bi bo sot
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }
    if y_score is not None and len(set(y_true.tolist())) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, y_score))
        m["pr_auc"] = float(average_precision_score(y_true, y_score))
    return m


def text_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)


def metrics_table(all_metrics: dict[str, dict]) -> pd.DataFrame:
    cols = ["n", "accuracy", "precision_scam", "recall_scam", "f1_scam", "macro_f1",
            "mcc", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate"]
    rows = {k: {c: v.get(c) for c in cols} for k, v in all_metrics.items()}
    return pd.DataFrame(rows).T


def group_metrics(df: pd.DataFrame, y_pred, y_score=None) -> pd.DataFrame:
    """Chi so tach theo nhom: co dau / khong dau va ngan / dai."""
    d = df.copy()
    d["pred"] = np.asarray(y_pred)
    if y_score is not None:
        d["score"] = np.asarray(y_score)
    d["nhom_dau"] = np.where(d["content"].map(has_diacritics), "có dấu", "không dấu")
    d["nhom_do_dai"] = pd.cut(d["content"].str.len(), [0, 50, 150, 10**6],
                              labels=["≤50 ký tự", "51-150", ">150"])
    rows = []
    for col in ("nhom_dau", "nhom_do_dai"):
        for g, sub in d.groupby(col, observed=True):
            if len(sub) == 0:
                continue
            m = compute_metrics(sub["label"], sub["pred"])
            rows.append({"loai_nhom": col, "nhom": str(g), "n": m["n"],
                         "ty_le_lua_dao": float(sub["label"].mean()), "accuracy": m["accuracy"],
                         "f1_scam": m["f1_scam"], "macro_f1": m["macro_f1"],
                         "fpr": m["false_positive_rate"], "fnr": m["false_negative_rate"]})
    return pd.DataFrame(rows)


def error_table(df: pd.DataFrame, y_pred, y_score) -> pd.DataFrame:
    d = df.copy()
    d["du_doan"] = np.asarray(y_pred)
    d["diem_lua_dao"] = np.round(np.asarray(y_score), 4)
    d["loai_loi"] = np.where((d["label"] == 0) & (d["du_doan"] == 1), "FP (thường bị báo lừa đảo)",
                    np.where((d["label"] == 1) & (d["du_doan"] == 0), "FN (lừa đảo bị bỏ sót)", "dung"))
    d["do_tin_cay_sai"] = np.where(d["loai_loi"] == "dung", 0.0,
                                   np.abs(d["diem_lua_dao"] - d["label"]))
    return d[d["loai_loi"] != "dung"].sort_values("do_tin_cay_sai", ascending=False).reset_index(drop=True)


def threshold_analysis(y_true, y_score) -> pd.DataFrame:
    rows = []
    y_true = np.asarray(y_true)
    for t in np.linspace(0.05, 0.95, 91):
        pred = (np.asarray(y_score) >= t).astype(int)
        m = compute_metrics(y_true, pred)
        rows.append({"threshold": round(float(t), 3), "precision_scam": m["precision_scam"],
                     "recall_scam": m["recall_scam"], "f1_scam": m["f1_scam"],
                     "macro_f1": m["macro_f1"], "fpr": m["false_positive_rate"],
                     "fnr": m["false_negative_rate"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Ve bieu do
# --------------------------------------------------------------------------- #
def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cm, path, title):
    cm = np.asarray(cm)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "Số lượng"),
                               (axes[1], cm / cm.sum(axis=1, keepdims=True), ".1%", "Tỷ lệ theo hàng (recall)")):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1], ["Thường", "Lừa đảo"])
        ax.set_yticks([0, 1], ["Thường", "Lừa đảo"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(sub)
        for i in range(2):
            for j in range(2):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v > data.max() / 2 else "black", fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    _save(fig, path)


def plot_roc_pr(curves: dict[str, tuple], path):
    """curves = {ten_tap: (y_true, y_score)}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, (yt, ys) in curves.items():
        color = SPLIT_COLORS.get(name)
        fpr, tpr, _ = roc_curve(yt, ys)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yt, ys):.4f})", color=color)
        pr, rc, _ = precision_recall_curve(yt, ys)
        axes[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(yt, ys):.4f})", color=color)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Đường ROC")
    axes[1].set(xlabel="Recall (lừa đảo)", ylabel="Precision (lừa đảo)", title="Đường Precision-Recall")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    _save(fig, path)


def plot_score_hist(y_true, y_score, path, title, threshold=0.5):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 26)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.65, label="Thực tế: thường", color="#4C78A8")
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.65, label="Thực tế: lừa đảo", color="#E45756")
    ax.axvline(threshold, color="k", ls="--", lw=1, label=f"ngưỡng {threshold}")
    ax.set(xlabel="Điểm/xác suất lừa đảo", ylabel="Số tin nhắn", title=title)
    ax.legend()
    _save(fig, path)


def plot_threshold_curve(table: pd.DataFrame, path, title):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, lab in (("precision_scam", "Precision"), ("recall_scam", "Recall"), ("f1_scam", "F1 (lừa đảo)")):
        ax.plot(table["threshold"], table[col], label=lab)
    best = table.loc[table["f1_scam"].idxmax()]
    ax.axvline(best["threshold"], color="gray", ls=":", label=f"F1 tốt nhất @ {best['threshold']:.2f}")
    ax.set(xlabel="Ngưỡng quyết định", ylabel="Giá trị", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_class_distribution(dfs: dict[str, pd.DataFrame], path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(dfs)
    x = np.arange(len(names))
    c0 = [int((dfs[n]["label"] == 0).sum()) for n in names]
    c1 = [int((dfs[n]["label"] == 1).sum()) for n in names]
    axes[0].bar(x - 0.2, c0, 0.4, label="Thường (0)", color="#4C78A8")
    axes[0].bar(x + 0.2, c1, 0.4, label="Lừa đảo (1)", color="#E45756")
    for i, (a, b) in enumerate(zip(c0, c1)):
        axes[0].text(i - 0.2, a, a, ha="center", va="bottom")
        axes[0].text(i + 0.2, b, b, ha="center", va="bottom")
    axes[0].set_xticks(x, names)
    axes[0].set(ylabel="Số tin nhắn", title="Phân bố nhãn theo tập")
    axes[0].legend()
    nodia = [float(1 - dfs[n]["content"].map(has_diacritics).mean()) for n in names]
    axes[1].bar(names, nodia, color=[SPLIT_COLORS.get(n, "gray") for n in names])
    axes[1].set(ylabel="Tỷ lệ tin KHÔNG DẤU", title="Tỷ lệ tin không dấu theo tập", ylim=(0, 1))
    for i, v in enumerate(nodia):
        axes[1].text(i, v, f"{v:.0%}", ha="center", va="bottom")
    _save(fig, path)


def plot_length_distribution(dfs: dict[str, pd.DataFrame], path):
    fig, ax = plt.subplots(figsize=(8, 4))
    all_df = pd.concat(dfs.values())
    bins = np.linspace(0, 400, 41)
    ax.hist(all_df.loc[all_df.label == 0, "content"].str.len().clip(upper=400), bins=bins, alpha=0.6,
            label="Thường", color="#4C78A8")
    ax.hist(all_df.loc[all_df.label == 1, "content"].str.len().clip(upper=400), bins=bins, alpha=0.6,
            label="Lừa đảo", color="#E45756")
    ax.set(xlabel="Độ dài (ký tự, cắt ở 400)", ylabel="Số tin nhắn", title="Phân bố độ dài tin nhắn theo nhãn")
    ax.legend()
    _save(fig, path)


def plot_group_metrics(table: pd.DataFrame, path, title):
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = table["loai_nhom"].str.replace("nhom_", "") + ": " + table["nhom"] + "\n(n=" + table["n"].astype(str) + ")"
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["macro_f1"], 0.4, label="macro-F1", color="#4C78A8")
    ax.bar(x + 0.2, table["accuracy"], 0.4, label="accuracy", color="#72B7B2")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set(ylim=(0, 1.05), title=title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def draw_pipeline_diagram(steps: list[tuple[str, str]], path, title):
    """Ve so do khoi noi tiep: steps = [(tieu de, mo ta ngan), ...]."""
    n = len(steps)
    fig, ax = plt.subplots(figsize=(2.6 * n, 3.2))
    ax.set_xlim(0, n * 2.6)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i, (head, body) in enumerate(steps):
        x = i * 2.6 + 0.2
        ax.add_patch(FancyBboxPatch((x, 0.5), 2.0, 1.9, boxstyle="round,pad=0.05",
                                    fc="#EAF1FB", ec="#4C78A8", lw=1.4))
        ax.text(x + 1.0, 2.05, head, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 1.0, 1.25, body, ha="center", va="center", fontsize=8)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + 2.06, 1.45), (x + 2.56, 1.45), arrowstyle="-|>",
                                         mutation_scale=16, lw=1.6, color="#222"))
    ax.set_title(title, fontsize=12)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# Tien ich khac
# --------------------------------------------------------------------------- #
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "torch", "transformers", "pyvi", "joblib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            info["cuda_version"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return info


class Timer:
    """Do thoi gian tung giai doan: with timer('ten'): ..."""

    def __init__(self):
        self.times: dict[str, float] = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self_):
                self_.t0 = time.perf_counter()

            def __exit__(self_, *exc):
                timer.times[name] = round(time.perf_counter() - self_.t0, 2)

        return _Ctx()


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """DataFrame -> bang Markdown (khong can thu vien tabulate)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) or v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def fmt_pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}%"
