"""Tien xu ly van ban cho TF-IDF + Linear SVM.

Dat rieng thanh module de file mo hinh .joblib co the nap lai o noi khac (predict_svm.py, app web)
chi can `import preprocess`.
"""
from __future__ import annotations

import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin

URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|vn|net|org|info|xyz|top|me|cc|onl|site|online|club|vip|link|icu|shop|app|io|co)\b(?:/\S*)?",
                    re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)[\s.\-]?\d{2,3}[\s.\-]?\d{3}[\s.\-]?\d{3,4}(?!\d)")
ACCOUNT_RE = re.compile(r"(?<![\d.,])\d{9,19}(?![\d])")
MONEY_RE = re.compile(r"(?<![\w])\d[\d.,]*\s?(?:k|tr|triệu|trieu|tỷ|ty|đ|vnd|vnđ|usd|\$)(?![a-zà-ỹ])",
                      re.IGNORECASE)


def strip_accents(text: str) -> str:
    """Bo dau tieng Viet (ke ca đ -> d)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def preprocess_text(text: str, normalize_whitespace=True, lowercase=True, strip_accents_=True,
                    mask_url=True, mask_phone=True, mask_account=True, mask_money=True) -> str:
    t = str(text)
    if mask_url:
        t = URL_RE.sub(" urltoken ", t)
    if mask_phone:
        t = PHONE_RE.sub(" sdttoken ", t)
    if mask_account:
        t = ACCOUNT_RE.sub(" sotktoken ", t)
    if mask_money:
        t = MONEY_RE.sub(" tientoken ", t)
    if lowercase:
        t = t.lower()
    if strip_accents_:
        t = strip_accents(t)
    if normalize_whitespace:
        t = re.sub(r"\s+", " ", t).strip()
    return t


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Buoc dau cua Pipeline: list[str] -> list[str] da chuan hoa."""

    def __init__(self, normalize_whitespace=True, lowercase=True, strip_accents=True,
                 mask_url=True, mask_phone=True, mask_account=True, mask_money=True):
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.mask_url = mask_url
        self.mask_phone = mask_phone
        self.mask_account = mask_account
        self.mask_money = mask_money

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [preprocess_text(t, self.normalize_whitespace, self.lowercase, self.strip_accents,
                                self.mask_url, self.mask_phone, self.mask_account, self.mask_money) for t in X]

    def get_params_dict(self) -> dict:
        return self.get_params()
