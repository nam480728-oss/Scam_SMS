"""Train TF-IDF + Linear SVM phan loai tin nhan lua dao (0 = binh thuong, 1 = lua dao).

Chay (tu thu muc ai_training/IDF_SVM):
    python train_tfidf_svm.py                    # dung config.yaml (co tim sieu tham so)
    python train_tfidf_svm.py --smoke-test       # chay thu rat nhanh de kiem tra moi truong
    python train_tfidf_svm.py --set search.enabled=false --set model.C=3

Ket qua nam trong outputs/<run_name>_<thoi gian>/ (xem README.md).
"""
from __future__ import annotations

import argparse
import itertools
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, train_test_split  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

import eval_utils as eu  # noqa: E402
from preprocess import TextPreprocessor, strip_accents  # noqa: E402

ROOT = Path(__file__).resolve().parent


# =============================================================================
# 1. CAU HINH / TIEN ICH
# =============================================================================
def load_config(path: Path, sets: list[str]) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in sets or []:
        key, _, val = item.partition("=")
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Khong co tham so '{key}' trong config")
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def make_groups(texts: list[str], n_words: int = 4) -> np.ndarray:
    """Nhom tin cung khuon (4 tu dau giong nhau) de CV khong bi ro ri bien the."""
    ids, out = {}, []
    for i, t in enumerate(texts):
        words = strip_accents(t).lower().split()
        key = " ".join(words[:n_words]) if len(words) >= n_words else f"__unique_{i}"
        out.append(ids.setdefault(key, len(ids)))
    return np.asarray(out)


def make_prep(cfg_pre: dict, strip: bool) -> TextPreprocessor:
    p = dict(cfg_pre)
    p["strip_accents"] = strip
    return TextPreprocessor(**p)


def make_features(cfg_feat: dict, word_ng, char_ng, use_word=None, use_char=None) -> FeatureUnion:
    use_word = cfg_feat["use_word"] if use_word is None else use_word
    use_char = cfg_feat["use_char"] if use_char is None else use_char
    parts = []
    common = {"sublinear_tf": cfg_feat["sublinear_tf"], "norm": cfg_feat["norm"], "lowercase": False,
              "dtype": np.float32}
    if use_word:
        w = cfg_feat["word"]
        parts.append(("word", TfidfVectorizer(analyzer="word", ngram_range=tuple(word_ng), min_df=w["min_df"],
                                              max_df=w["max_df"], max_features=w["max_features"],
                                              token_pattern=r"(?u)\b\w+\b", **common)))
    if use_char:
        c = cfg_feat["char"]
        parts.append(("char", TfidfVectorizer(analyzer=c["analyzer"], ngram_range=tuple(char_ng), min_df=c["min_df"],
                                              max_df=c["max_df"], max_features=c["max_features"], **common)))
    if not parts:
        raise ValueError("Phai bat it nhat 1 trong use_word / use_char")
    return FeatureUnion(parts)


def fit_svm(X, y, C, class_weight, cfg_model, seed):
    """Huan luyen LinearSVC; tra ve (model, da_hoi_tu)."""
    svm = LinearSVC(C=C, class_weight=class_weight, loss=cfg_model["loss"], max_iter=cfg_model["max_iter"],
                    tol=cfg_model["tol"], dual=True, random_state=seed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        svm.fit(X, y)
    return svm, not any(issubclass(x.category, ConvergenceWarning) for x in w)


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


# =============================================================================
# 2. TIM SIEU THAM SO
# =============================================================================
def run_search(cfg, raw, y, groups, timer_cb=print) -> pd.DataFrame:
    """Thu moi to hop trong grid: huan luyen tren train, do tren val + CV(train, theo nhom)."""
    g = cfg["search"]["grid"]
    cv = StratifiedGroupKFold(n_splits=cfg["search"]["cv_folds"], shuffle=True, random_state=cfg["seed"])
    folds = list(cv.split(raw["train"], y["train"], groups))
    rows, text_cache = [], {}
    combos = list(itertools.product(g["strip_accents"], g["word_ngram"], g["char_ngram"]))
    total = len(combos) * len(g["class_weight"]) * len(g["C"])
    done = 0
    for strip, wng, cng in combos:
        if strip not in text_cache:
            prep = make_prep(cfg["preprocess"], strip)
            text_cache[strip] = {k: prep.transform(v) for k, v in raw.items() if k in ("train", "val")}
        tr_txt, va_txt = text_cache[strip]["train"], text_cache[strip]["val"]
        feats = make_features(cfg["features"], wng, cng)
        Xtr = feats.fit_transform(tr_txt)  # IDF/tu vung hoc tu train (khong dung val/test)
        Xva = feats.transform(va_txt)
        for cw, C in itertools.product(g["class_weight"], g["C"]):
            svm, ok = fit_svm(Xtr, y["train"], C, cw, cfg["model"], cfg["seed"])
            cv_scores = []
            for tr_i, te_i in folds:
                m, _ = fit_svm(Xtr[tr_i], y["train"][tr_i], C, cw, cfg["model"], cfg["seed"])
                cv_scores.append(macro_f1(y["train"][te_i], m.predict(Xtr[te_i])))
            rows.append({"strip_accents": strip, "word_ngram": str(list(wng)), "char_ngram": str(list(cng)),
                         "class_weight": cw if cw is not None else "none", "C": C,
                         "n_features": Xtr.shape[1], "train_macro_f1": macro_f1(y["train"], svm.predict(Xtr)),
                         "cv_macro_f1_mean": float(np.mean(cv_scores)), "cv_macro_f1_std": float(np.std(cv_scores)),
                         "val_macro_f1": macro_f1(y["val"], svm.predict(Xva)), "converged": ok})
            done += 1
        timer_cb(f"    [search] {done}/{total} to hop xong")
    return pd.DataFrame(rows)


def pick_best(results: pd.DataFrame, selection: str) -> pd.Series:
    primary, secondary = ("val_macro_f1", "cv_macro_f1_mean") if selection == "val" else ("cv_macro_f1_mean", "val_macro_f1")
    return results.sort_values([primary, secondary, "C"], ascending=[False, False, True]).iloc[0]


# =============================================================================
# 3. BIEU DO RIENG CUA SVM
# =============================================================================
def plot_search_summary(res: pd.DataFrame, best: pd.Series, out: Path):
    top = res.sort_values("val_macro_f1", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    lab = [f"C={r.C} cw={r.class_weight} strip={r.strip_accents} w{r.word_ngram} c{r.char_ngram}" for r in top.itertuples()]
    y = np.arange(len(top))
    ax.barh(y + 0.2, top["val_macro_f1"], 0.4, label="val macro-F1", color="#F58518")
    ax.barh(y - 0.2, top["cv_macro_f1_mean"], 0.4, xerr=top["cv_macro_f1_std"], label="CV macro-F1 (train)", color="#4C78A8")
    ax.set_yticks(y, lab, fontsize=7)
    lo = min(top["val_macro_f1"].min(), top["cv_macro_f1_mean"].min())
    ax.set(xlim=(max(0, lo - 0.02), 1.0), xlabel="macro-F1", title="15 tổ hợp tham số tốt nhất (theo val)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    eu._save(fig, out / "plots" / "20_search_top15.png")

    same = res[(res["strip_accents"] == best["strip_accents"]) & (res["word_ngram"] == best["word_ngram"])
               & (res["char_ngram"] == best["char_ngram"]) & (res["class_weight"] == best["class_weight"])].sort_values("C")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogx(same["C"], same["train_macro_f1"], "o-", label="train")
    ax.semilogx(same["C"], same["cv_macro_f1_mean"], "s-", label="CV (train)")
    ax.fill_between(same["C"], same["cv_macro_f1_mean"] - same["cv_macro_f1_std"],
                    same["cv_macro_f1_mean"] + same["cv_macro_f1_std"], alpha=0.2)
    ax.semilogx(same["C"], same["val_macro_f1"], "^-", label="validation")
    ax.axvline(best["C"], color="gray", ls=":", label=f"C chọn = {best['C']}")
    ax.set(xlabel="C (log)", ylabel="macro-F1", title="Ảnh hưởng của C (các tham số khác cố định như bộ tốt nhất)")
    ax.grid(alpha=0.3)
    ax.legend()
    eu._save(fig, out / "plots" / "21_C_curve.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, ttl in ((axes[0], "strip_accents", "Bỏ dấu"), (axes[1], "class_weight", "Trọng số lớp")):
        data = [res.loc[res[col] == v, "val_macro_f1"].values for v in sorted(res[col].unique(), key=str)]
        ax.boxplot(data, tick_labels=[str(v) for v in sorted(res[col].unique(), key=str)])
        ax.set(ylabel="val macro-F1", title=f"{ttl}: phân bố val macro-F1 trên toàn grid")
        ax.grid(alpha=0.3)
    eu._save(fig, out / "plots" / "22_param_effect.png")


def plot_top_features(feat_df: pd.DataFrame, k: int, out: Path):
    for scope, sub in (("tat_ca", feat_df), ("chi_tu", feat_df[feat_df["loai"] == "word"])):
        if sub.empty:
            continue
        pos = sub.sort_values("trong_so", ascending=False).head(k).iloc[::-1]
        neg = sub.sort_values("trong_so").head(k).iloc[::-1]
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, k * 0.28)))
        axes[0].barh(pos["dac_trung"], pos["trong_so"], color="#E45756")
        axes[0].set(title=f"Đặc trưng đẩy về LỪA ĐẢO ({scope})", xlabel="Trọng số SVM")
        axes[1].barh(neg["dac_trung"], neg["trong_so"], color="#4C78A8")
        axes[1].set(title=f"Đặc trưng đẩy về BÌNH THƯỜNG ({scope})", xlabel="Trọng số SVM")
        for a in axes:
            a.tick_params(axis="y", labelsize=8)
            a.grid(axis="x", alpha=0.3)
        eu._save(fig, out / "plots" / f"23_top_features_{scope}.png")


def learning_curve(cfg, tr_txt, y_tr, va_txt, y_va, best, fractions, seed, repeats=3):
    rows = []
    wng, cng = eval(best["word_ngram"]), eval(best["char_ngram"])  # noqa: S307 - chuoi do chinh script tao
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    idx_all = np.arange(len(tr_txt))
    for f in fractions:
        for r in range(repeats if f < 1.0 else 1):
            idx = idx_all if f >= 1.0 else train_test_split(idx_all, train_size=f, stratify=y_tr, random_state=seed + r)[0]
            sub = [tr_txt[i] for i in idx]
            feats = make_features(cfg["features"], wng, cng)
            X = feats.fit_transform(sub)
            svm, _ = fit_svm(X, y_tr[idx], float(best["C"]), cw, cfg["model"], seed)
            rows.append({"fraction": f, "n_train": len(idx), "train_macro_f1": macro_f1(y_tr[idx], svm.predict(X)),
                         "val_macro_f1": macro_f1(y_va, svm.predict(feats.transform(va_txt)))})
    return pd.DataFrame(rows)


def plot_learning_curve(lc: pd.DataFrame, out: Path):
    g = lc.groupby("n_train").agg(["mean", "std"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, lab in (("train_macro_f1", "train"), ("val_macro_f1", "validation")):
        ax.errorbar(g.index, g[(col, "mean")], yerr=g[(col, "std")], fmt="o-", capsize=3, label=lab)
    ax.set(xlabel="Số mẫu train", ylabel="macro-F1", title="Learning curve (bộ tham số tốt nhất)")
    ax.grid(alpha=0.3)
    ax.legend()
    eu._save(fig, out / "plots" / "24_learning_curve.png")


def run_ablation(cfg, tr_txt, y_tr, texts, y, best, seed) -> pd.DataFrame:
    wng, cng = eval(best["word_ngram"]), eval(best["char_ngram"])  # noqa: S307
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    rows = []
    for name, uw, uc in (("Chỉ TF-IDF từ", True, False), ("Chỉ TF-IDF ký tự", False, True), ("Từ + ký tự", True, True)):
        feats = make_features(cfg["features"], wng, cng, uw, uc)
        X = feats.fit_transform(tr_txt)
        svm, _ = fit_svm(X, y_tr, float(best["C"]), cw, cfg["model"], seed)
        rows.append({"cau_hinh": name, "n_features": X.shape[1],
                     "val_macro_f1": macro_f1(y["val"], svm.predict(feats.transform(texts["val"]))),
                     "test_macro_f1": macro_f1(y["test"], svm.predict(feats.transform(texts["test"])))})
    return pd.DataFrame(rows)


def plot_ablation(ab: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(ab))
    ax.bar(x - 0.2, ab["val_macro_f1"], 0.4, label="val", color="#F58518")
    ax.bar(x + 0.2, ab["test_macro_f1"], 0.4, label="test", color="#54A24B")
    ax.set_xticks(x, [f"{r.cau_hinh}\n({r.n_features:,} đặc trưng)" for r in ab.itertuples()], fontsize=8)
    ax.set(ylim=(max(0, ab[["val_macro_f1", "test_macro_f1"]].min().min() - 0.05), 1.0), ylabel="macro-F1",
           title="So sánh loại đặc trưng (bộ tham số tốt nhất)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    eu._save(fig, out / "plots" / "25_ablation.png")


# =============================================================================
# 4. NHAN XET TU DONG + BAO CAO
# =============================================================================
def auto_comments(ctx: dict) -> list[str]:
    res, best, m, cfg = ctx["search"], ctx["best"], ctx["metrics"], ctx["cfg"]
    tr, va, te = m["train"], m["val"], m["test"]
    c = []
    if res is not None:
        top5 = res.sort_values("val_macro_f1", ascending=False).head(5)
        c.append(f"Đã thử {len(res)} tổ hợp tham số. Bộ tốt nhất (chọn theo {cfg['search']['selection']}): C={best['C']}, "
                 f"class_weight={best['class_weight']}, bỏ dấu={best['strip_accents']}, n-gram từ {best['word_ngram']}, "
                 f"n-gram ký tự {best['char_ngram']} → val macro-F1 {best['val_macro_f1']:.4f}, CV {best['cv_macro_f1_mean']:.4f} "
                 f"± {best['cv_macro_f1_std']:.4f}.")
        spread = top5["val_macro_f1"].max() - top5["val_macro_f1"].min()
        c.append(f"5 tổ hợp tốt nhất chỉ chênh nhau {spread:.4f} val macro-F1 → "
                 + ("mô hình ít nhạy với siêu tham số." if spread < 0.01 else "kết quả khá nhạy với siêu tham số, nên kiểm tra kỹ."))
        if abs(best["cv_macro_f1_mean"] - best["val_macro_f1"]) > 0.03:
            c.append("CV trên train và val lệch nhau > 0.03: val nhỏ nên dao động, hoặc phân bố train/val không hoàn toàn giống nhau.")
        bad = int((~res["converged"]).sum())
        if bad:
            c.append(f"{bad} tổ hợp không hội tụ trong max_iter (thường C lớn); tăng model.max_iter nếu cần.")
        no_dia = res[res.strip_accents == False]["val_macro_f1"].max()  # noqa: E712
        dia = res[res.strip_accents == True]["val_macro_f1"].max()  # noqa: E712
        if not (np.isnan(no_dia) or np.isnan(dia)):
            c.append(f"Bỏ dấu: tốt nhất {dia:.4f} (bỏ dấu) so với {no_dia:.4f} (giữ dấu) → "
                     + ("nên bỏ dấu để tin có dấu/không dấu dùng chung đặc trưng." if dia >= no_dia else "giữ dấu cho kết quả tốt hơn."))
    c.append(f"Kết quả cuối (decision>0): train macro-F1 {tr['macro_f1']:.4f}, val {va['macro_f1']:.4f}, test {te['macro_f1']:.4f}.")
    if tr["macro_f1"] - va["macro_f1"] > 0.05:
        c.append(f"Train cao hơn val {tr['macro_f1'] - va['macro_f1']:.3f} → có overfitting; thử giảm C hoặc tăng min_df.")
    c.append(f"Trên test: precision lừa đảo {te['precision_scam']:.4f}, recall {te['recall_scam']:.4f}; bỏ sót {te['fn']} tin lừa đảo (FN), "
             f"báo nhầm {te['fp']} tin thường (FP). ROC-AUC {te.get('roc_auc', float('nan')):.4f}, PR-AUC {te.get('pr_auc', float('nan')):.4f}.")
    c.append(f"Ngưỡng tối ưu trên val (xác suất Platt) = {ctx['thr']['threshold']:.2f} → macro-F1 test {ctx['thr']['test_macro_f1']:.4f}.")
    c.append(f"Số đặc trưng sau khi fit: {ctx['n_features']:,}; mật độ ma trận train {ctx['density']:.3%}.")
    if ctx.get("top_pos"):
        c.append("Đặc trưng từ mạnh nhất về LỪA ĐẢO: " + ", ".join(ctx["top_pos"]) + ". Về BÌNH THƯỜNG: " + ", ".join(ctx["top_neg"]) + ".")
    lc = ctx.get("lc")
    if lc is not None:
        g = lc.groupby("n_train")["val_macro_f1"].mean()
        if len(g) >= 2:
            k = -3 if len(g) >= 3 else -2
            gain = g.iloc[-1] - g.iloc[k]
            c.append(f"Learning curve: tăng từ {g.index[k]} lên {g.index[-1]} mẫu cải thiện val macro-F1 {gain:+.4f} → "
                     + ("thêm dữ liệu vẫn có ích." if gain > 0.01 else "đường cong gần bão hoà, thêm dữ liệu cùng kiểu ít giúp."))
    ab = ctx.get("ablation")
    if ab is not None:
        b = ab.sort_values("val_macro_f1", ascending=False).iloc[0]
        c.append(f"So sánh đặc trưng: '{b['cau_hinh']}' tốt nhất trên val ({b['val_macro_f1']:.4f}).")
    gt = ctx.get("group_test")
    if gt is not None and not gt.empty:
        g = gt[gt["loai_nhom"] == "nhom_dau"]
        if len(g) == 2:
            a, b = g.sort_values("macro_f1").iloc[0], g.sort_values("macro_f1").iloc[-1]
            c.append(f"Theo nhóm trên test: tin '{a['nhom']}' macro-F1 {a['macro_f1']:.4f} (n={int(a['n'])}) so với '{b['nhom']}' "
                     f"{b['macro_f1']:.4f} (n={int(b['n'])}).")
    c.append("Lưu ý: dữ liệu có nhiều tin do máy sinh theo khuôn nên điểm có thể cao hơn khi gặp tin thật; "
             "cần kiểm tra thêm trên tập tin thật độc lập.")
    return c


def write_report(path: Path, ctx: dict):
    cfg = ctx["cfg"]
    best = ctx["best"]
    L = [f"# Báo cáo huấn luyện TF-IDF + Linear SVM — {cfg['run_name']}", "",
         f"- Thời điểm: {ctx['started']}  |  Tổng thời gian: {ctx['total_seconds'] / 60:.1f} phút",
         f"- Mô hình lưu tại `model/tfidf_svm.joblib`", "", "## 1. Dữ liệu", "", ctx["data_table"], "",
         "![Phân bố nhãn](plots/00_class_distribution.png)", "![Độ dài](plots/00_length_distribution.png)", "",
         f"Trùng nội dung giữa các tập (phải = 0): `{ctx['overlap']}`", "",
         "## 2. Pipeline", "", "![Pipeline](plots/00_pipeline.png)", "",
         "Ví dụ tiền xử lý (xem `data/preprocess_samples.xlsx`):", "", ctx["prep_examples"], "",
         "## 3. Tham số", "", "### Bộ tham số được chọn", "", ctx["best_table"], "",
         "Toàn bộ cấu hình: `config_used.yaml`; tham số mô hình cuối: `model_params.json`.", ""]
    if ctx["search"] is not None:
        L += ["## 4. Tìm siêu tham số", "",
              f"Số tổ hợp: {len(ctx['search'])} (chi tiết: `search_results.csv`). Chọn theo: **{cfg['search']['selection']}**.", "",
              "![Top15](plots/20_search_top15.png)", "![Đường cong C](plots/21_C_curve.png)",
              "![Ảnh hưởng tham số](plots/22_param_effect.png)", ""]
    L += ["## 5. Kết quả cuối cùng", "", ctx["metrics_table"], "", "![ROC/PR](plots/10_roc_pr.png)",
          "![CM val](plots/11_confusion_val.png)", "![CM test](plots/12_confusion_test.png)",
          "![Điểm test](plots/13_score_hist_test.png)", "", "### Phân tích ngưỡng", "", "![Ngưỡng](plots/14_threshold_val.png)", "",
          f"Ngưỡng tối ưu trên val = **{ctx['thr']['threshold']:.2f}** → macro-F1 test = {ctx['thr']['test_macro_f1']:.4f}.", "",
          "### Theo nhóm (test)", "", ctx["group_table"], "", "![Nhóm](plots/15_group_test.png)", "",
          "### Báo cáo phân loại (test)", "", "```", ctx["class_report_test"], "```", "",
          "## 6. Giải thích mô hình & phân tích bổ sung", "", "![Đặc trưng](plots/23_top_features_tat_ca.png)",
          "![Đặc trưng từ](plots/23_top_features_chi_tu.png)", "", "![Learning curve](plots/24_learning_curve.png)", ""]
    if ctx.get("ablation") is not None:
        L += ["![Ablation](plots/25_ablation.png)", "", eu.md_table(ctx["ablation"].round(4)), ""]
    L += ["## 7. Nhận xét tự động", ""] + [f"{i}. {t}" for i, t in enumerate(ctx["comments"], 1)]
    L += ["", "## 8. Danh sách file kết quả", "", "| File | Nội dung |", "|---|---|",
          "| config_used.yaml, environment.json | Tham số, phiên bản thư viện |",
          "| search_results.csv | Kết quả mọi tổ hợp tham số |", "| model/tfidf_svm.joblib | Pipeline + SVM + Platt |",
          "| top_features.csv | Trọng số đặc trưng |", "| learning_curve.csv, ablation.csv | Phân tích bổ sung |",
          "| metrics.json, metrics_summary.csv | Chỉ số |", "| predictions_*.csv, errors_*.xlsx | Dự đoán và lỗi |",
          "| plots/ | Biểu đồ |"]
    path.write_text("\n".join(L), encoding="utf-8")


# =============================================================================
# 5. CHUONG TRINH CHINH
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Train TF-IDF + Linear SVM phan loai tin nhan lua dao")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--smoke-test", action="store_true", help="chay thu nhanh voi du lieu nho + grid nho")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config), args.set)
    if args.smoke_test:
        cfg["run_name"] += "_smoke"
        g = cfg["search"]["grid"]
        for k in g:
            g[k] = g[k][:1]
        g["class_weight"] = ["balanced"]
        cfg["search"]["cv_folds"] = 2
        cfg["analysis"]["learning_curve_fractions"] = [0.5, 1.0]
    timer = eu.Timer()
    t_start = time.perf_counter()
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_dir = Path(args.output_dir) if args.output_dir else \
        ROOT / cfg["output"]["dir"] / f"{cfg['run_name']}_{datetime.now():%Y%m%d_%H%M%S}"
    for sub in ("plots", "data", "model"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    print(f"[+] Ket qua se luu tai: {run_dir}")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    eu.save_json(eu.environment_info(), run_dir / "environment.json")

    # ---------------------------------------------------------------- du lieu
    data_dir = (ROOT / cfg["data"]["data_dir"]).resolve()
    dfs = {"train": eu.load_split(data_dir / cfg["data"]["train_file"]),
           "val": eu.load_split(data_dir / cfg["data"]["val_file"]),
           "test": eu.load_split(data_dir / cfg["data"]["test_file"])}
    if args.smoke_test:
        dfs = {k: v.groupby("label").sample(n=(60 if k == "train" else 20), random_state=1).reset_index(drop=True)
               for k, v in dfs.items()}
    overlap = eu.check_no_overlap(dfs)
    print("[+] Kich thuoc: " + ", ".join(f"{k}={len(v)}" for k, v in dfs.items()) + f" | trung giua cac tap: {overlap}")
    if any(overlap.values()):
        print("[!] CANH BAO: co noi dung trung giua cac tap -> ket qua bi thoi phong.")
    raw = {k: v["content"].tolist() for k, v in dfs.items()}
    y = {k: v["label"].to_numpy() for k, v in dfs.items()}
    eu.plot_class_distribution(dfs, run_dir / "plots" / "00_class_distribution.png")
    eu.plot_length_distribution(dfs, run_dir / "plots" / "00_length_distribution.png")
    data_df = pd.DataFrame([{"tap": k, "so_dong": len(v), "thuong_0": int((v.label == 0).sum()),
                             "lua_dao_1": int((v.label == 1).sum()), "ty_le_lua_dao": round(float(v.label.mean()), 4),
                             "ty_le_khong_dau": round(float(1 - v.content.map(eu.has_diacritics).mean()), 4),
                             "do_dai_trung_vi": int(v.content.str.len().median())} for k, v in dfs.items()])
    data_df.to_csv(run_dir / "data" / "data_summary.csv", index=False)
    groups = make_groups(raw["train"])
    print(f"[+] So nhom mau trong train (dung cho CV): {len(set(groups))}")

    # ---------------------------------------------------- tim sieu tham so
    search_df = None
    if cfg["search"]["enabled"]:
        n_combo = int(np.prod([len(v) for v in cfg["search"]["grid"].values()]))
        print(f"[+] Tim sieu tham so: {n_combo} to hop x ({cfg['search']['cv_folds']} fold CV + 1 val) ...")
        with timer("tim_sieu_tham_so"):
            search_df = run_search(cfg, raw, y, groups)
        search_df.to_csv(run_dir / "search_results.csv", index=False)
        best = pick_best(search_df, cfg["search"]["selection"])
        plot_search_summary(search_df, best, run_dir)
    else:
        m, f_ = cfg["model"], cfg["features"]
        best = pd.Series({"strip_accents": cfg["preprocess"]["strip_accents"], "word_ngram": str(f_["word"]["ngram_range"]),
                          "char_ngram": str(f_["char"]["ngram_range"]),
                          "class_weight": m["class_weight"] if m["class_weight"] is not None else "none", "C": m["C"],
                          "train_macro_f1": np.nan, "cv_macro_f1_mean": np.nan, "cv_macro_f1_std": np.nan,
                          "val_macro_f1": np.nan, "n_features": np.nan, "converged": True})
    print(f"[+] Bo tham so duoc chon: {best[['C', 'class_weight', 'strip_accents', 'word_ngram', 'char_ngram']].to_dict()}")
    cw_best = None if best["class_weight"] == "none" else best["class_weight"]
    wng, cng = eval(best["word_ngram"]), eval(best["char_ngram"])  # noqa: S307 - chuoi do chinh script tao

    # ------------------------------------------------ huan luyen mo hinh cuoi
    with timer("huan_luyen_mo_hinh_cuoi"):
        pipe = Pipeline([("prep", make_prep(cfg["preprocess"], bool(best["strip_accents"]))),
                         ("features", make_features(cfg["features"], wng, cng)),
                         ("svm", LinearSVC(C=float(best["C"]), class_weight=cw_best, loss=cfg["model"]["loss"],
                                           max_iter=cfg["model"]["max_iter"], tol=cfg["model"]["tol"], dual=True,
                                           random_state=cfg["seed"]))])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            pipe.fit(raw["train"], y["train"])
        converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    if not converged:
        print("[!] SVM khong hoi tu trong max_iter -> tang model.max_iter trong config.yaml")
    dec = {k: pipe.decision_function(raw[k]) for k in raw}
    if cfg["calibration"]["platt_on_val"]:
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(dec["val"].reshape(-1, 1), y["val"])
        to_prob = lambda d: platt.predict_proba(d.reshape(-1, 1))[:, 1]  # noqa: E731
    else:
        platt = None
        to_prob = lambda d: 1.0 / (1.0 + np.exp(-d))  # noqa: E731
    prob = {k: to_prob(dec[k]) for k in dec}
    pred = {k: (dec[k] > 0).astype(int) for k in dec}  # nhan quyet dinh goc cua SVM

    # ---------------------------------------------------------- danh gia
    results = {}
    for k in ("train", "val", "test"):
        results[k] = eu.compute_metrics(y[k], pred[k], prob[k])
        out_df = dfs[k].copy()
        out_df["decision_function"], out_df["xac_suat_lua_dao"], out_df["du_doan"] = dec[k].round(5), prob[k].round(5), pred[k]
        out_df.to_csv(run_dir / f"predictions_{k}.csv", index=False, encoding="utf-8-sig")
    eu.plot_roc_pr({k: (y[k], prob[k]) for k in prob}, run_dir / "plots" / "10_roc_pr.png")
    eu.plot_confusion_matrix(results["val"]["confusion_matrix"], run_dir / "plots" / "11_confusion_val.png", "Ma trận nhầm lẫn — Validation")
    eu.plot_confusion_matrix(results["test"]["confusion_matrix"], run_dir / "plots" / "12_confusion_test.png", "Ma trận nhầm lẫn — Test")
    eu.plot_score_hist(y["test"], prob["test"], run_dir / "plots" / "13_score_hist_test.png", "Phân bố xác suất lừa đảo — Test")
    eu.plot_score_hist(y["val"], prob["val"], run_dir / "plots" / "13_score_hist_val.png", "Phân bố xác suất lừa đảo — Validation")
    thr_val = eu.threshold_analysis(y["val"], prob["val"])
    thr_val.to_csv(run_dir / "threshold_val.csv", index=False)
    eu.plot_threshold_curve(thr_val, run_dir / "plots" / "14_threshold_val.png", "Precision/Recall/F1 theo ngưỡng — Validation")
    best_t = float(thr_val.loc[thr_val["macro_f1"].idxmax(), "threshold"])
    tm = eu.compute_metrics(y["test"], (prob["test"] >= best_t).astype(int), prob["test"])
    thr_info = {"threshold": best_t, "test_macro_f1": tm["macro_f1"]}
    results["test_at_val_threshold"] = tm
    group_val = eu.group_metrics(dfs["val"], pred["val"], prob["val"])
    group_test = eu.group_metrics(dfs["test"], pred["test"], prob["test"])
    group_val.to_csv(run_dir / "group_metrics_val.csv", index=False)
    group_test.to_csv(run_dir / "group_metrics_test.csv", index=False)
    eu.plot_group_metrics(group_test, run_dir / "plots" / "15_group_test.png", "Chỉ số theo nhóm — Test")
    n_err = cfg["output"]["n_error_samples"]
    with pd.ExcelWriter(run_dir / "errors_val.xlsx") as w1, pd.ExcelWriter(run_dir / "errors_test.xlsx") as w2:
        eu.error_table(dfs["val"], pred["val"], prob["val"]).head(n_err).to_excel(w1, index=False)
        eu.error_table(dfs["test"], pred["test"], prob["test"]).head(n_err).to_excel(w2, index=False)
    eu.save_json(results, run_dir / "metrics.json")
    table_src = {k: v for k, v in results.items() if k != "test_at_val_threshold"}
    eu.metrics_table(table_src).round(4).to_csv(run_dir / "metrics_summary.csv")
    class_report = eu.text_report(y["test"], pred["test"])
    (run_dir / "classification_report_test.txt").write_text(class_report, encoding="utf-8")

    # ------------------------------------------- dac trung + phan tich them
    names = pipe.named_steps["features"].get_feature_names_out()
    svm = pipe.named_steps["svm"]
    coef = svm.coef_[0]
    Xtr = pipe.named_steps["features"].transform(pipe.named_steps["prep"].transform(raw["train"]))
    df_count = np.asarray((Xtr > 0).sum(axis=0)).ravel()
    feat_df = pd.DataFrame({"dac_trung": [n.split("__", 1)[1] for n in names], "loai": [n.split("__", 1)[0] for n in names],
                            "trong_so": coef, "so_tin_chua": df_count})
    feat_df.reindex(feat_df["trong_so"].abs().sort_values(ascending=False).index).head(500).to_csv(
        run_dir / "top_features.csv", index=False, encoding="utf-8-sig")
    plot_top_features(feat_df, cfg["analysis"]["top_features"], run_dir)
    words = feat_df[feat_df["loai"] == "word"] if (feat_df["loai"] == "word").any() else feat_df
    top_pos = words.sort_values("trong_so", ascending=False).head(8)["dac_trung"].tolist()
    top_neg = words.sort_values("trong_so").head(8)["dac_trung"].tolist()
    eu.save_json({"C": float(best["C"]), "class_weight": cw_best, "loss": cfg["model"]["loss"], "n_features": int(len(coef)),
                  "intercept": float(svm.intercept_[0]), "coef_min": float(coef.min()), "coef_max": float(coef.max()),
                  "coef_l2_norm": float(np.linalg.norm(coef)), "n_nonzero_coef": int((coef != 0).sum()),
                  "n_iter": int(svm.n_iter_), "converged": converged,
                  "platt_slope": None if platt is None else float(platt.coef_[0][0]),
                  "platt_intercept": None if platt is None else float(platt.intercept_[0]),
                  "preprocess": pipe.named_steps["prep"].get_params(),
                  "features": {k: str(v) for k, v in pipe.named_steps["features"].get_params().items() if "__" not in k}},
                 run_dir / "model_params.json")

    pre = pipe.named_steps["prep"]
    pre_samples = pd.DataFrame({"goc": dfs["train"]["content"][:30], "sau_tien_xu_ly": pre.transform(raw["train"][:30]),
                                "nhan": y["train"][:30]})
    pre_samples.to_excel(run_dir / "data" / "preprocess_samples.xlsx", index=False)
    prep_examples = eu.md_table(pre_samples.head(5).assign(goc=lambda d: d["goc"].str.slice(0, 90),
                                                           sau_tien_xu_ly=lambda d: d["sau_tien_xu_ly"].str.slice(0, 90)))

    tr_txt = pre.transform(raw["train"])
    txt = {"val": pre.transform(raw["val"]), "test": pre.transform(raw["test"])}
    with timer("learning_curve"):
        lc = learning_curve(cfg, tr_txt, y["train"], txt["val"], y["val"], best, cfg["analysis"]["learning_curve_fractions"], cfg["seed"])
    lc.to_csv(run_dir / "learning_curve.csv", index=False)
    plot_learning_curve(lc, run_dir)
    ablation = None
    if cfg["analysis"]["ablation"]:
        with timer("ablation"):
            ablation = run_ablation(cfg, tr_txt, y["train"], txt, y, best, cfg["seed"])
        ablation.to_csv(run_dir / "ablation.csv", index=False)
        plot_ablation(ablation, run_dir)

    eu.draw_pipeline_diagram(
        [("Tin nhắn", "văn bản thô"),
         ("Tiền xử lý", f"hạ chữ thường\nbỏ dấu={bool(best['strip_accents'])}\nmask link/SĐT/STK/tiền"),
         ("TF-IDF", f"từ {wng} + ký tự {cng}\nsublinear, L2"), ("Linear SVM", f"C={best['C']}\nclass_weight={best['class_weight']}"),
         ("Platt", "decision → xác suất\n(hiệu chỉnh trên val)"), ("Kết quả", "0 = thường\n1 = lừa đảo")],
        run_dir / "plots" / "00_pipeline.png", "Sơ đồ pipeline TF-IDF + Linear SVM")

    joblib.dump({"pipeline": pipe, "platt": platt, "threshold_default": 0.5, "decision_threshold": 0.0,
                 "config": cfg, "best_params": best.to_dict()}, run_dir / "model" / "tfidf_svm.joblib")

    # ---------------------------------------------------------- bao cao
    total_s = time.perf_counter() - t_start
    timer.times["tong"] = round(total_s, 1)
    eu.save_json(timer.times, run_dir / "timing.json")
    ctx = {"cfg": cfg, "search": search_df, "best": best, "metrics": results, "thr": thr_info, "n_features": len(coef),
           "density": float(Xtr.nnz / (Xtr.shape[0] * Xtr.shape[1])), "top_pos": top_pos, "top_neg": top_neg, "lc": lc,
           "ablation": ablation, "group_test": group_test}
    comments = auto_comments(ctx)
    (run_dir / "nhan_xet_tu_dong.md").write_text("\n".join(f"{i}. {t}" for i, t in enumerate(comments, 1)), encoding="utf-8")
    best_table = eu.md_table(pd.DataFrame({"tham_so": list(best.index), "gia_tri": list(best.values)}))
    write_report(run_dir / "report.md", {**ctx, "started": started, "total_seconds": total_s, "data_table": eu.md_table(data_df),
                 "overlap": overlap, "prep_examples": prep_examples, "best_table": best_table,
                 "metrics_table": eu.md_table(eu.metrics_table(table_src).round(4), index=True),
                 "group_table": eu.md_table(group_test.round(4)), "class_report_test": class_report, "comments": comments})
    print("\n" + "=" * 70)
    for k in ("train", "val", "test"):
        r = results[k]
        print(f"{k:5s} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} F1_scam={r['f1_scam']:.4f} "
              f"P={r['precision_scam']:.4f} R={r['recall_scam']:.4f} AUC={r.get('roc_auc', float('nan')):.4f}")
    print(f"Bao cao day du: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
