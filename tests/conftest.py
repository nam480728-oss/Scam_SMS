"""Cau hinh test: dung CSDL rieng data_scam_test (tao moi bang migration that), AI gia lap."""
import os
import sys
from pathlib import Path

os.environ["DATABASE_URL"] = "mysql+pymysql://root:@127.0.0.1:3306/data_scam_test?charset=utf8mb4"
os.environ["SECRET_KEY"] = "test-secret"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymysql  # noqa: E402
import pytest  # noqa: E402


class FakePredictor:
    """Gan co tin co tu khoa lua dao (khong can nap mo hinh that)."""
    KEYWORDS = ("lừa đảo", "otp", "chuyển khoản", "trúng thưởng", "bấm link")

    def predict(self, text):
        hit = any(k in text.lower() for k in self.KEYWORDS)
        return (0.97 if hit else 0.03), hit

    def close(self):
        pass


@pytest.fixture(scope="session")
def app():
    conn = pymysql.connect(host="127.0.0.1", port=3306, user="root", password="")
    with conn.cursor() as cur:
        cur.execute("DROP DATABASE IF EXISTS data_scam_test")
        cur.execute("CREATE DATABASE data_scam_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.close()
    from app import app as flask_app
    flask_app.config.update(TESTING=True, WTF_CSRF_ENABLED=False, UPLOAD_DIR=Path(__file__).parent / "_uploads")
    from flask_migrate import upgrade
    with flask_app.app_context():
        upgrade(directory=str(Path(__file__).resolve().parents[1] / "migrations"))
    yield flask_app
    conn = pymysql.connect(host="127.0.0.1", port=3306, user="root", password="")
    with conn.cursor() as cur:
        cur.execute("DROP DATABASE IF EXISTS data_scam_test")
    conn.close()
    import shutil
    shutil.rmtree(Path(__file__).parent / "_uploads", ignore_errors=True)


@pytest.fixture(autouse=True)
def clean(app):
    from db.database import db
    from services.ai_client import manager
    with app.app_context():
        db.session.remove()
        db.session.execute(db.text("SET FOREIGN_KEY_CHECKS=0"))
        for t in ("scam_counters", "messages", "conversations", "blocks", "friend_requests", "users",
                  "feed_posts", "feed_authors"):
            db.session.execute(db.text(f"TRUNCATE TABLE {t}"))
        db.session.execute(db.text("SET FOREIGN_KEY_CHECKS=1"))
        db.session.commit()
    manager.set_predictor("svm", FakePredictor())
    from services import realtime
    realtime._presence.clear()
    yield
    with app.app_context():
        db.session.remove()


class Actor:
    """Mot nguoi dung gia lap: co trinh duyet (client) rieng."""

    def __init__(self, app, name, password="matkhau1"):
        self.app, self.name, self.password = app, name, password
        self.client = app.test_client()
        r = self.client.post("/register", data={"username": name, "password": password, "confirm": password})
        assert r.status_code == 302, r.data
        with app.app_context():
            from db.models import User
            from services.ai_client import manager
            self.id = User.query.filter_by(username=name, model_key=manager.active).one().id

    def get(self, url, **kw):
        return self.client.get(url, **kw)

    def post(self, url, json=None, **kw):
        return self.client.post(url, json=json, **kw)

    def delete(self, url):
        return self.client.delete(url)

    def befriend(self, other):
        r = self.post("/api/friend-requests", {"user_id": other.id})
        assert r.status_code == 201, r.data
        reqs = other.get("/api/friend-requests").get_json()["incoming"]
        assert other.post(f"/api/friend-requests/{reqs[0]['id']}/accept").status_code == 200

    def conv_with(self, other):
        r = self.post("/api/conversations", {"user_id": other.id})
        assert r.status_code == 200, r.data
        return r.get_json()["conversation"]["id"]

    def say(self, cid, text):
        return self.post(f"/api/conversations/{cid}/messages", {"body": text})


@pytest.fixture
def mk(app):
    def make(name):
        return Actor(app, name)
    return make


@pytest.fixture
def pair(mk):
    """alice va bob da la ban be + cuoc chat."""
    a, b = mk("alice"), mk("bob")
    a.befriend(b)
    return a, b, a.conv_with(b)
