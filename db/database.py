"""Khoi tao cac tien ich CSDL dung chung (SQLAlchemy + Alembic/Flask-Migrate)."""
from datetime import datetime, timezone

from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
migrate = Migrate()


def utcnow() -> datetime:
    """Gio UTC khong mang tzinfo (MySQL DATETIME khong luu mui gio)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(dt: datetime | None) -> str | None:
    """Datetime UTC -> chuoi ISO co 'Z' de trinh duyet doi sang gio dia phuong."""
    return None if dt is None else dt.replace(microsecond=0).isoformat() + "Z"
