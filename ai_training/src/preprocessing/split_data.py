"""Chia Data_vi_Scam.xlsx thanh Data_Scam_train/test/vali.xlsx (70/15/15), chong ro ri.

Cac cau cung mau (do may sinh bien the) duoc gom thanh nhom, moi nhom chi nam
trong dung mot tap. Nhom = ket noi giua cac dong co:
  - cung 4 tu dau (sau khi bo dau, ha chu thuong), hoac
  - (tuy chon, --sim > 0) do tuong dong cosine (TF-IDF ky tu) >= --sim.
    Tren du lieu nay noi theo do tuong dong gay hieu ung chuoi (1 nhom ~2600/3545
    dong) nen mac dinh tat; chi dung prefix va bao cao ro ri o cuoi.

Chay: python -m src.preprocessing.split_data   (tu thu muc ai_training)
"""
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower()


def build_groups(texts: pd.Series, sim: float, prefix_words: int = 4) -> list[int]:
    norm = texts.map(lambda t: re.sub(r"\s+", " ", strip_accents(t)).strip())
    n = len(norm)
    rows, cols = [], []

    # nhom theo cum tu dau
    prefix = norm.map(lambda t: " ".join(re.findall(r"\w+", t)[:prefix_words]))
    first_seen: dict[str, int] = {}
    for i, key in enumerate(prefix):
        if len(key.split()) < prefix_words:  # cau qua ngan: khong gom theo prefix
            continue
        if key in first_seen:
            rows.append(first_seen[key])
            cols.append(i)
        else:
            first_seen[key] = i

    # nhom theo do tuong dong noi dung (tat khi sim <= 0)
    if sim <= 0:
        graph = coo_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))
        return connected_components(graph, directed=False)[1].tolist()
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    x = tfidf.fit_transform(norm)
    s = (x @ x.T).tocoo()
    mask = (s.data >= sim) & (s.row < s.col)
    rows += s.row[mask].tolist()
    cols += s.col[mask].tolist()

    graph = coo_matrix(([1] * len(rows), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels.tolist()


def assign_splits(df: pd.DataFrame, seed: int) -> pd.Series:
    """Chia 70/15/15 chinh xac (sai lech <= vai dong) theo tung nhan, khong cat nhom.

    Nhom lon xep truoc, moi nhom vao tap con thieu nhieu nhat so voi muc tieu
    (so muc tieu tinh rieng cho tung nhan); cac nhom 1 dong xep cuoi de lap day.
    """
    import numpy as np

    ratios = {"train": 0.70, "test": 0.15, "val": 0.15}
    labels = sorted(df["label"].unique())
    target = {}
    for lab in labels:
        n = int((df["label"] == lab).sum())
        t = {k: int(round(n * v)) for k, v in ratios.items()}
        t["train"] += n - sum(t.values())  # bu phan lech lam tron vao train
        target[lab] = t

    rng = np.random.RandomState(seed)
    groups = [g for _, g in df.groupby("group")]
    order = rng.permutation(len(groups))
    groups = sorted((groups[i] for i in order), key=lambda g: -len(g))  # on dinh: giu thu tu ngau nhien trong cung kich thuoc

    count = {lab: {k: 0 for k in ratios} for lab in labels}
    split = pd.Series("", index=df.index)
    for g in groups:
        add = g["label"].value_counts().to_dict()
        # chon tap con thieu nhieu nhat (ty le con thieu so voi muc tieu), phat phan vuot muc
        def score(k):
            return sum(
                c * (target[lab][k] - count[lab][k]) / max(target[lab][k], 1)
                - max(0, count[lab][k] + c - target[lab][k]) / max(target[lab][k], 1)
                for lab, c in add.items()
            )

        best = max(ratios, key=score)
        for lab, c in add.items():
            count[lab][best] += c
        split[g.index] = best
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_DIR / "Data_vi_Scam.xlsx"))
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    ap.add_argument("--sim", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_excel(args.input)[["content", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    df["group"] = build_groups(df["content"], args.sim)

    split = assign_splits(df, args.seed)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    file_names = {"train": "Data_Scam_train.xlsx", "test": "Data_Scam_test.xlsx", "val": "Data_Scam_vali.xlsx"}
    for name, fname in file_names.items():
        part = df[split == name][["content", "label"]]
        part.to_excel(out / fname, index=False, sheet_name="Data")
        print(f"{name:5s} {len(part):5d} ({len(part) / len(df):.0%})  label={part['label'].value_counts().sort_index().to_dict()}")

    sizes = df["group"].value_counts()
    print(f"nhom: {len(sizes)} (lon nhat {sizes.iloc[0]} dong)")
    leaked = (split.groupby(df["group"]).nunique() > 1).sum()
    print(f"nhom nam o nhieu hon 1 tap: {leaked}")

    # ro ri con lai: do tuong dong lon nhat cua moi dong test/val voi tap train
    norm = df["content"].map(strip_accents)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(norm)
    xtr = tfidf.transform(norm[split == "train"])
    for name in ("val", "test"):
        xe = tfidf.transform(norm[split == name])
        mx = (xe @ xtr.T).max(axis=1).toarray().ravel()
        print(f"{name}: cosine max toi train  trung vi={pd.Series(mx).median():.2f}  "
              f">=0.9: {(mx >= 0.9).mean():.0%}")


if __name__ == "__main__":
    main()
