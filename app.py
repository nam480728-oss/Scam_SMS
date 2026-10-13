"""SafeChat - website chat co AI trong tai phat hien lua dao.

Chay:  python app.py        (hoac: flask --app app run)
CSDL:  flask --app app db upgrade
Du lieu bang tin gia:  flask --app app seed-feed
"""
import logging
import mimetypes
import os

import click
from flask import Flask, jsonify, redirect, request, url_for
from flask_login import LoginManager
from flask_wtf.csrf import CSRFError, CSRFProtect

from config import Config
from db.database import db, migrate
from db.models import User
from services.chat_service import ChatError
from services.realtime import init_socket_events, socketio

# Windows co the khai bao sai MIME (vd .js = text/plain) khien trinh duyet tu choi chay script
for _ext, _type in ((".js", "application/javascript"), (".mjs", "application/javascript"), (".css", "text/css"),
                    (".svg", "image/svg+xml"), (".json", "application/json")):
    mimetypes.add_type(_type, _ext)

login_manager = LoginManager()
csrf = CSRFProtect()


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if config:
        app.config.update(config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    socketio.init_app(app, async_mode="threading", manage_session=False)

    from routes import auth, chat, feed, friends, models
    from services.ai_client import manager
    app.jinja_env.globals["ai_name"] = lambda: manager.active_name() or "Chưa chọn"
    for module in (models, auth, feed, chat, friends):
        app.register_blueprint(module.bp)
    init_socket_events(app)

    @login_manager.user_loader
    def load_user(user_id: str):
        # Tai khoan chi hop le trong dung "the gioi" cua mo hinh dang hoat dong; doi mo hinh -> bi dang xuat.
        user = db.session.get(User, int(user_id))
        return user if user is not None and user.model_key == manager.active else None

    manager.on_change = lambda key: socketio.emit("system:model_changed", {"model": key})

    @login_manager.unauthorized_handler
    def unauthorized():
        if request.path.startswith("/api/"):
            return jsonify(error="Bạn cần đăng nhập.", code="auth"), 401
        return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))

    @app.errorhandler(ChatError)
    def chat_error(e: ChatError):
        return jsonify(error=e.message), e.status

    @app.errorhandler(CSRFError)
    def csrf_error(e):
        if request.path.startswith("/api/"):
            return jsonify(error="Phiên làm việc hết hạn, hãy tải lại trang.", code="csrf"), 400
        return "Phiên làm việc hết hạn, hãy tải lại trang và thử lại.", 400

    @app.errorhandler(413)
    def too_large(e):
        return jsonify(error="Tệp quá lớn."), 413

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify(error="Không tìm thấy."), 404
        return "Không tìm thấy trang.", 404

    @app.errorhandler(500)
    def server_error(e):
        db.session.rollback()
        if request.path.startswith("/api/"):
            return jsonify(error="Lỗi máy chủ, vui lòng thử lại."), 500
        return "Lỗi máy chủ.", 500

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        if request.path.startswith("/api/") and "Cache-Control" not in resp.headers:
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.teardown_request
    def cleanup(exc):
        if exc:
            db.session.rollback()

    @app.cli.command("seed-feed")
    @click.option("--reset", is_flag=True, help="Xoa du lieu bang tin cu roi tao lai")
    def seed_feed_cmd(reset):
        """Tao du lieu bang tin gia."""
        from services.seed_feed import seed_feed
        click.echo(f"Da tao {seed_feed(reset)} bai viet.")

    return app


app = create_app()

if __name__ == "__main__":
    socketio.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False,
                 allow_unsafe_werkzeug=True)
