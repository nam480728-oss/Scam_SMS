"""Dang ky / dang nhap / dang xuat."""
import re
from urllib.parse import urlparse

from flask import Blueprint, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import func

from db.database import db
from db.models import AVATAR_COLORS, User
from routes import model_required
from services.ai_client import manager

bp = Blueprint("auth", __name__)
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,30}$")


def _safe_next(target: str | None) -> str:
    if target:
        p = urlparse(target)
        if not p.netloc and not p.scheme and target.startswith("/") and not target.startswith("//"):
            return target
    return url_for("feed.feed_page")


@bp.route("/login", methods=["GET", "POST"])
@model_required
def login():
    if current_user.is_authenticated:
        return redirect(url_for("feed.feed_page"))
    error, username = None, ""
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        user = (User.query.filter(User.model_key == manager.active, func.lower(User.username) == username.lower()).first()
                if username else None)
        if user and user.check_password(password):
            login_user(user, remember=True)
            return redirect(_safe_next(request.args.get("next")))
        error = "Tên tài khoản hoặc mật khẩu không đúng."
    return render_template("login.html", error=error, username=username)


@bp.route("/register", methods=["GET", "POST"])
@model_required
def register():
    if current_user.is_authenticated:
        return redirect(url_for("feed.feed_page"))
    error, username = None, ""
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm") or ""
        if not USERNAME_RE.match(username):
            error = "Tên tài khoản gồm 3–30 ký tự: chữ cái không dấu, số, dấu chấm hoặc gạch dưới."
        elif len(password) < 6:
            error = "Mật khẩu phải có ít nhất 6 ký tự."
        elif len(password) > 128:
            error = "Mật khẩu quá dài."
        elif password != confirm:
            error = "Hai lần nhập mật khẩu không khớp."
        elif User.query.filter(User.model_key == manager.active, func.lower(User.username) == username.lower()).first():
            error = "Tên tài khoản đã tồn tại."
        if not error:
            user = User(username=username, model_key=manager.active,
                        avatar_color=AVATAR_COLORS[User.query.filter_by(model_key=manager.active).count() % len(AVATAR_COLORS)])
            user.set_password(password)
            db.session.add(user)
            try:
                db.session.commit()
            except Exception:                                     # dang ky dong thoi cung ten
                db.session.rollback()
                error = "Tên tài khoản đã tồn tại."
            else:
                login_user(user, remember=True)
                return redirect(url_for("feed.feed_page"))
    return render_template("register.html", error=error, username=username)


@bp.post("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
