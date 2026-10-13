"""Man hinh chon mo hinh AI (chu phan mem) + trang quan tri doi mo hinh.

* Khi server vua khoi dong va chua co mo hinh nao: trang chu "/" cho chu phan mem chon mo hinh de demo.
* Sau do nguoi dung chi dang ky / dang nhap, KHONG duoc chon mo hinh.
* Muon doi mo hinh: vao /admin va nhap mat khau quan tri (ADMIN_PASSWORD trong .env).
* Moi mo hinh co tap tai khoan rieng (doi mo hinh = tai khoan cua mo hinh khac khong ton tai).
"""
import hmac
import time

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for
from flask_login import current_user

from services.ai_client import manager

bp = Blueprint("models", __name__)


def is_admin() -> bool:
    return bool(session.get("is_admin"))


@bp.get("/")
def select_page():
    if manager.ready():                                   # da co mo hinh: nguoi dung vao thang dang nhap
        return redirect(url_for("feed.feed_page") if current_user.is_authenticated else url_for("auth.login"))
    return render_template("select_model.html", models=manager.catalog(), status=manager.status(), admin=False,
                           next_url=url_for("auth.login"))


@bp.route("/admin", methods=["GET", "POST"])
def admin_page():
    error = None
    if request.method == "POST":
        ok = hmac.compare_digest((request.form.get("password") or "").encode(), current_app.config["ADMIN_PASSWORD"].encode())
        if ok:
            session["is_admin"] = True
            return redirect(url_for("models.admin_page"))
        time.sleep(0.6)                                   # lam cham do mat khau
        error = "Sai mật khẩu quản trị."
    if not is_admin():
        return render_template("admin_login.html", error=error), (401 if error else 200)
    return render_template("select_model.html", models=manager.catalog(), status=manager.status(), admin=True,
                           next_url=url_for("auth.login"))


@bp.post("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("auth.login") if manager.ready() else url_for("models.select_page"))


@bp.get("/api/models")
def list_models():
    return jsonify(models=manager.catalog(), status=manager.status())


@bp.get("/api/models/status")
def model_status():
    return jsonify(manager.status())


@bp.post("/api/models/select/<key>")
def select_model(key: str):
    # Lan dau (chua co mo hinh) ai mo trang cung chon duoc; da co mo hinh thi chi quan tri vien moi doi duoc.
    if manager.ready() and not is_admin():
        return jsonify(error="Chỉ quản trị viên mới được đổi mô hình AI."), 403
    try:
        status = manager.select(key)
    except KeyError:
        return jsonify(error="Mô hình không tồn tại."), 404
    return jsonify(status)
