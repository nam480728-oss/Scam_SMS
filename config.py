"""Cau hinh ung dung SafeChat (chat co AI canh bao lua dao)."""
import glob
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

AI_ROOT = BASE_DIR / "ai_training"


def _latest_run(pattern: str) -> Path | None:
    """Thu muc ket qua train day du moi nhat (bo qua cac lan chay --smoke-test)."""
    runs = sorted(p for p in glob.glob(str(AI_ROOT / pattern)) if "smoke" not in Path(p).name)
    return Path(runs[-1]) if runs else None


# Moi mo hinh: key -> thong tin hien thi + duong dan (co the ghi de bang bien moi truong trong .env)
def model_specs() -> dict[str, dict]:
    svm_run = _latest_run("IDF_SVM/outputs/tfidf_linear_svm_2*")
    pho_run = _latest_run("phobert/outputs/phobert_base_v2_2*")
    viso_run = _latest_run("VISO/outputs/visobert_2*")
    return {
        "svm": {
            "key": "svm", "name": "TF-IDF + Linear SVM", "tagline": "Nhẹ, nhanh, giải thích được",
            "kind": "Học máy cổ điển", "run": svm_run,
            "path": Path(os.getenv("MODEL_SVM_PATH", svm_run / "model" / "tfidf_svm.joblib")) if svm_run else None,
            "icon": "⚡", "color": "#1b74e4",
        },
        "phobert": {
            "key": "phobert", "name": "PhoBERT-base-v2", "tagline": "Mô hình ngôn ngữ tiếng Việt (VinAI)",
            "kind": "Transformer · tách từ pyvi", "run": pho_run,
            "path": Path(os.getenv("MODEL_PHOBERT_PATH", pho_run / "best_model")) if pho_run else None,
            "icon": "🧠", "color": "#7b3fe4",
        },
        "viso": {
            "key": "viso", "name": "ViSoBERT", "tagline": "Mô hình cho văn bản mạng xã hội tiếng Việt",
            "kind": "Transformer · SentencePiece", "run": viso_run,
            "path": Path(os.getenv("MODEL_VISO_PATH", viso_run / "best_model")) if viso_run else None,
            "icon": "💬", "color": "#e4572e",
        },
    }


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", "mysql+pymysql://root:@127.0.0.1:3306/data_scam?charset=utf8mb4")
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True, "pool_recycle": 280}
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_DIR = BASE_DIR / "uploads"
    MAX_CONTENT_LENGTH = 12 * 1024 * 1024          # tran 12MB cho 1 request
    MAX_UPLOAD_BYTES = 10 * 1024 * 1024            # 10MB / file
    IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}
    FILE_EXTS = {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "csv", "zip", "rar", "7z"}

    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")   # mat khau trang quan tri (doi mo hinh AI)

    MAX_MESSAGE_LEN = 2000
    AI_THRESHOLD = float(os.getenv("AI_THRESHOLD", "0.5"))   # dung cho PhoBERT/ViSoBERT (xac suat >= nguong)
    PAGE_SIZE = 40

    WTF_CSRF_TIME_LIMIT = None
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_DURATION = 60 * 60 * 24 * 14
    JSON_AS_ASCII = False
