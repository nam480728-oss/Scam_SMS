"""Nap va chay 3 mo hinh phat hien lua dao: TF-IDF+SVM, PhoBERT-base-v2, ViSoBERT.

Chi 1 mo hinh duoc nap trong bo nho tai 1 thoi diem (chon toan server). Nap o luong nen
de giao dien hien thi tien trinh; mo hinh cu duoc giai phong sau khi mo hinh moi san sang.
"""
from __future__ import annotations

import gc
import json
import logging
import re
import sys
import threading
import time

from config import AI_ROOT, Config, model_specs

log = logging.getLogger("safechat.ai")


# --------------------------------------------------------------------------- #
# Cac "predictor": moi lop co predict(text) -> (xac_suat_lua_dao, bi_gan_co)
# --------------------------------------------------------------------------- #
class SVMPredictor:
    def __init__(self, path):
        sys.path.append(str(AI_ROOT / "IDF_SVM"))          # de joblib tim thay module `preprocess`
        import joblib
        import numpy as np

        self._np = np
        bundle = joblib.load(path)
        self.pipe, self.platt = bundle["pipeline"], bundle["platt"]
        self.decision_threshold = bundle.get("decision_threshold", 0.0)

    def predict(self, text: str) -> tuple[float, bool]:
        d = float(self.pipe.decision_function([text])[0])
        if self.platt is not None:
            prob = float(self.platt.predict_proba(self._np.array([[d]]))[0, 1])
        else:
            prob = float(1.0 / (1.0 + self._np.exp(-d)))
        return prob, d > self.decision_threshold          # nhan goc cua SVM: decision_function > 0

    def close(self):
        self.pipe = self.platt = None


class TransformerPredictor:
    """PhoBERT (co tach tu pyvi) hoac ViSoBERT (SentencePiece, khong tach tu)."""

    def __init__(self, path, segment: bool, threshold: float):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        meta = json.loads((path / "inference_config.json").read_text(encoding="utf-8"))
        self.max_length = meta["max_length"]
        self.normalize_ws = meta.get("text", {}).get("normalize_whitespace", True)
        self.lowercase = meta.get("text", {}).get("lowercase", False)
        self.threshold = threshold
        self.segmenter = None
        if segment:
            from pyvi import ViTokenizer

            self.segmenter = ViTokenizer.tokenize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        if self.device.type == "cuda":
            self.model.half()                                  # fp16 cho nhanh, tiet kiem VRAM

    def predict(self, text: str) -> tuple[float, bool]:
        if self.normalize_ws:
            text = re.sub(r"\s+", " ", text).strip()
        if self.lowercase:
            text = text.lower()
        if self.segmenter:
            text = self.segmenter(text)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits.float()
        prob = float(self.torch.softmax(logits, dim=-1)[0, 1])
        return prob, prob >= self.threshold

    def close(self):
        self.model = self.tokenizer = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _build(key: str, spec: dict):
    path = spec["path"]
    if path is None or not path.exists():
        raise FileNotFoundError(f"Khong tim thay mo hinh '{spec['name']}' (da train chua?): {path}")
    if key == "svm":
        return SVMPredictor(path)
    if key == "phobert":
        return TransformerPredictor(path, segment=True, threshold=Config.AI_THRESHOLD)
    if key == "viso":
        return TransformerPredictor(path, segment=False, threshold=Config.AI_THRESHOLD)
    raise KeyError(key)


# --------------------------------------------------------------------------- #
# Quan ly mo hinh dang hoat dong (singleton)
# --------------------------------------------------------------------------- #
class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()          # bao ve trang thai
        self._infer_lock = threading.Lock()     # moi lan chi 1 luong chay suy luan
        self._predictor = None
        self.active: str | None = None
        self.loading: str | None = None
        self.error: str | None = None
        self.started_at: float | None = None
        self.on_change = None                  # ham(key) goi sau khi doi sang mo hinh moi

    # ---- thong tin cho giao dien ----
    def catalog(self) -> list[dict]:
        out = []
        for key, spec in model_specs().items():
            metrics, size = {}, None
            run = spec["run"]
            if run and (run / "metrics.json").exists():
                try:
                    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["test"]
                    metrics = {"macro_f1": m["macro_f1"], "recall": m["recall_scam"], "precision": m["precision_scam"]}
                except (OSError, KeyError, ValueError):
                    pass
            if spec["path"] and spec["path"].exists():
                p = spec["path"]
                size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            out.append({"key": key, "name": spec["name"], "tagline": spec["tagline"], "kind": spec["kind"],
                        "icon": spec["icon"], "color": spec["color"], "available": bool(size),
                        "size_mb": None if size is None else round(size / 2**20, 1), "metrics": metrics})
        return out

    def status(self) -> dict:
        with self._lock:
            return {"active": self.active, "loading": self.loading, "error": self.error,
                    "elapsed": None if self.started_at is None else round(time.time() - self.started_at, 1)}

    def ready(self) -> bool:
        return self._predictor is not None

    def active_name(self) -> str | None:
        if not self.active:
            return None
        return model_specs()[self.active]["name"]

    # ---- chon / nap ----
    def select(self, key: str) -> dict:
        specs = model_specs()
        if key not in specs:
            raise KeyError(key)
        with self._lock:
            if self.loading:
                return self.status()                      # dang nap, bo qua yeu cau trung
            if self.active == key and self._predictor is not None:
                return self.status()
            self.loading, self.error, self.started_at = key, None, time.time()
        threading.Thread(target=self._load, args=(key,), daemon=True, name=f"load-{key}").start()
        return self.status()

    def select_blocking(self, key: str):
        """Dung cho test / script: nap xong moi tra ve."""
        self.select(key)
        while self.status()["loading"]:
            time.sleep(0.05)
        if self.error:
            raise RuntimeError(self.error)

    def _load(self, key: str):
        try:
            new = _build(key, model_specs()[key])
            new.predict("khoi dong")                      # chay thu 1 lan de khoi tao/ lam nong
            with self._lock, self._infer_lock:
                old, self._predictor, self.active = self._predictor, new, key
            had_old = old is not None
            if had_old:
                old.close()
                del old
                gc.collect()
            log.info("Da nap mo hinh %s trong %.1fs", key, time.time() - self.started_at)
            if had_old and self.on_change:
                try:
                    self.on_change(key)
                except Exception:  # noqa: BLE001
                    log.exception("on_change that bai")
        except Exception as e:  # noqa: BLE001 - bao loi cho giao dien
            log.exception("Nap mo hinh %s that bai", key)
            with self._lock:
                self.error = f"{type(e).__name__}: {e}"
        finally:
            with self._lock:
                self.loading = None

    def set_predictor(self, key: str, predictor):
        """Gan predictor co san (dung trong test)."""
        with self._lock:
            self._predictor, self.active, self.loading, self.error = predictor, key, None, None

    # ---- suy luan ----
    def predict(self, text: str) -> tuple[float, bool] | None:
        """Tra ve (xac_suat, bi_gan_co) hoac None neu chua co mo hinh / van ban rong."""
        if not text or not text.strip():
            return None
        with self._infer_lock:
            predictor = self._predictor
            if predictor is None:
                return None
            return predictor.predict(text)


manager = ModelManager()
