"""Du doan tin nhan lua dao bang mo hinh PhoBERT da train (thu muc best_model/).

    python predict_phobert.py --model outputs/<run>/best_model --text "Tai khoan cua ban da bi khoa, bam vao link..."
    python predict_phobert.py --model outputs/<run>/best_model --file tin_nhan.xlsx   # cot 'content'
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="thu muc best_model/ cua 1 lan train")
    ap.add_argument("--text", action="append", default=[], help="1 tin nhan (co the lap lai)")
    ap.add_argument("--file", help="file xlsx/csv co cot 'content'")
    ap.add_argument("--threshold", type=float, default=None, help="mac dinh lay trong inference_config.json (0.5)")
    ap.add_argument("--out", help="ghi ket qua ra file xlsx (tuy chon)")
    args = ap.parse_args()

    model_dir = Path(args.model)
    meta = json.loads((model_dir / "inference_config.json").read_text(encoding="utf-8"))
    thr = args.threshold if args.threshold is not None else meta.get("threshold_default", 0.5)
    texts = list(args.text)
    if args.file:
        df = pd.read_csv(args.file) if args.file.endswith(".csv") else pd.read_excel(args.file)
        texts += df["content"].astype(str).tolist()
    if not texts:
        raise SystemExit("Can --text hoac --file")

    cfg_text = meta["text"]
    clean = [re.sub(r"\s+", " ", t).strip() if cfg_text.get("normalize_whitespace", True) else t for t in texts]
    if cfg_text.get("word_segment", True):
        from pyvi import ViTokenizer
        clean = [ViTokenizer.tokenize(t) for t in clean]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    rows = []
    with torch.no_grad():
        for i in range(0, len(clean), 32):
            enc = tok(clean[i:i + 32], truncation=True, max_length=meta["max_length"], padding=True,
                      return_tensors="pt").to(device)
            prob = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
            rows += prob
    out = pd.DataFrame({"content": texts, "xac_suat_lua_dao": [round(p, 4) for p in rows]})
    out["ket_luan"] = ["LỪA ĐẢO" if p >= thr else "Bình thường" for p in rows]
    print(out.to_string(index=False, max_colwidth=80))
    if args.out:
        out.to_excel(args.out, index=False)


if __name__ == "__main__":
    main()
