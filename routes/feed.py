"""Bang tin (du lieu gia, chi xem)."""
from datetime import datetime, timezone

from flask import Blueprint, render_template
from flask_login import login_required

from db.models import FeedPost
from routes import model_required

bp = Blueprint("feed", __name__)


def time_ago(dt: datetime) -> str:
    secs = int((datetime.now(timezone.utc).replace(tzinfo=None) - dt).total_seconds())
    if secs < 60:
        return "Vừa xong"
    if secs < 3600:
        return f"{secs // 60} phút"
    if secs < 86400:
        return f"{secs // 3600} giờ"
    if secs < 86400 * 7:
        return f"{secs // 86400} ngày"
    return dt.strftime("%d/%m/%Y")


@bp.get("/feed")
@model_required
@login_required
def feed_page():
    posts = FeedPost.query.order_by(FeedPost.created_at.desc(), FeedPost.id.desc()).limit(50).all()
    return render_template("feed.html", posts=posts, time_ago=time_ago)
