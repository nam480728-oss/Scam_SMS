"""Cac blueprint cua ung dung."""
from functools import wraps

from flask import jsonify, redirect, request, url_for
from flask_login import current_user

from services.ai_client import manager


def wants_json() -> bool:
    return request.path.startswith("/api/")


def model_required(view):
    """Chua chon mo hinh AI -> quay ve man hinh chon mo hinh."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not manager.ready():
            if wants_json():
                return jsonify(error="Chưa chọn mô hình AI.", code="no_model"), 503
            return redirect(url_for("models.select_page"))
        return view(*args, **kwargs)
    return wrapper


def login_required_api(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify(error="Bạn cần đăng nhập.", code="auth"), 401
        return view(*args, **kwargs)
    return wrapper
