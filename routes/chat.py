"""Trang Tin nhan + API hoi thoai / tin nhan / dinh kem."""
from pathlib import Path

from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file
from flask_login import current_user, login_required

from db.database import db
from db.models import Message
from routes import login_required_api, model_required
from services import chat_service as cs
from services.ai_client import manager

bp = Blueprint("chat", __name__)


@bp.get("/messages")
@bp.get("/messages/<int:cid>")
@model_required
@login_required
def messages_page(cid: int | None = None):
    return render_template("chat.html", initial_cid=cid)


@bp.get("/api/summary")
@login_required_api
def summary():
    convs = cs.list_conversations(current_user.id)
    reqs = cs.list_requests(current_user.id)
    return jsonify(unread_total=sum(c["unread"] for c in convs), requests=len(reqs["incoming"]),
                   model=manager.active_name())


@bp.get("/api/conversations")
@login_required_api
@model_required
def conversations():
    return jsonify(conversations=cs.list_conversations(current_user.id))


@bp.post("/api/conversations")
@login_required_api
@model_required
def open_conversation():
    data = request.get_json(silent=True) or {}
    uid = data.get("user_id")
    if not isinstance(uid, int) or isinstance(uid, bool):
        return jsonify(error="Thiếu user_id."), 400
    conv = cs.get_or_create_conversation(current_user.id, uid)
    return jsonify(conversation=cs.conversation_dto(conv, current_user.id))


@bp.get("/api/conversations/<int:cid>")
@login_required_api
@model_required
def conversation_detail(cid: int):
    conv = cs.get_conversation(cid, current_user.id)
    return jsonify(conversation=cs.conversation_dto(conv, current_user.id))


@bp.get("/api/conversations/<int:cid>/messages")
@login_required_api
@model_required
def conversation_messages(cid: int):
    conv = cs.get_conversation(cid, current_user.id)
    before = request.args.get("before", type=int)
    limit = min(max(request.args.get("limit", current_app.config["PAGE_SIZE"], type=int), 1), 100)
    return jsonify(cs.history(conv, current_user.id, before, limit))


@bp.post("/api/conversations/<int:cid>/messages")
@login_required_api
@model_required
def post_message(cid: int):
    conv = cs.get_conversation(cid, current_user.id)
    if request.is_json:
        body, file = (request.get_json(silent=True) or {}).get("body"), None
    else:
        body, file = request.form.get("body"), request.files.get("file")
    if body is not None and not isinstance(body, str):
        return jsonify(error="Nội dung không hợp lệ."), 400
    dto = cs.send_message(conv, current_user._get_current_object(), body, file)
    return jsonify(message=dto), 201


@bp.post("/api/conversations/<int:cid>/read")
@login_required_api
def read_conversation(cid: int):
    conv = cs.get_conversation(cid, current_user.id)
    return jsonify(cs.mark_read(conv, current_user.id))


@bp.get("/api/attachments/<int:mid>")
@login_required_api
def attachment(mid: int):
    msg = db.session.get(Message, mid)
    if msg is None or not msg.attachment_path:
        abort(404)
    if not msg.conversation.has_user(current_user.id):
        abort(404)
    path = Path(current_app.config["UPLOAD_DIR"]) / msg.attachment_path
    if not path.is_file():
        abort(404)
    is_image = msg.kind == "image"
    resp = send_file(path, mimetype=msg.attachment_mime or "application/octet-stream", as_attachment=not is_image,
                     download_name=msg.attachment_name, conditional=True, max_age=3600)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp
