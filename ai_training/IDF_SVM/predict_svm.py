"""Du doan tin nhan lua dao bang mo hinh TF-IDF + Linear SVM da train.

    python predict_svm.py --model outputs/<run>/model/tfidf_svm.joblib --text "Tai khoan bi khoa, bam link ..."
    python predict_svm.py --model outputs/<run>/model/tfidf_svm.joblib --file tin_nhan.xlsx --out ketqua.xlsx
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # de joblib tim thay module `preprocess`


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="duong dan file tfidf_svm.joblib")
    ap.add_argument("--text", action="append", default=[], help="1 tin nhan (co the lap lai)")
    ap.add_argument("--file", help="file xlsx/csv co cot 'content'")
    ap.add_argument("--out", help="ghi ket qua ra xlsx (tuy chon)")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    pipe, platt = bundle["pipeline"], bundle["platt"]
    texts = list(args.text)
    if args.file:
        df = pd.read_csv(args.file) if args.file.endswith(".csv") else pd.read_excel(args.file)
        texts += df["content"].astype(str).tolist()
    if not texts:
        raise SystemExit("Can --text hoac --file")

    dec = pipe.decision_function(texts)
    prob = platt.predict_proba(dec.reshape(-1, 1))[:, 1] if platt is not None else 1 / (1 + np.exp(-dec))
    out = pd.DataFrame({"content": texts, "decision_function": dec.round(4), "xac_suat_lua_dao": prob.round(4),
                        "ket_luan": np.where(dec > bundle.get("decision_threshold", 0.0), "LỪA ĐẢO", "Bình thường")})
    print(out.to_string(index=False, max_colwidth=80))
    if args.out:
        out.to_excel(args.out, index=False)


if __name__ == "__main__":
    main()
