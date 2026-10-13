"""API tim kiem, ket ban, chan."""
from flask import Blueprint, jsonify, request
from flask_login import current_user

from routes import login_required_api, model_required
from services import chat_service as cs

bp = Blueprint("friends", __name__, url_prefix="/api")


def _uid(data) -> int | None:
    v = (data or {}).get("user_id")
    return v if isinstance(v, int) and not isinstance(v, bool) else None


@bp.get("/users/search")
@login_required_api
@model_required
def search():
    return jsonify(users=cs.search_users(current_user.id, request.args.get("q", "")))


@bp.get("/friends")
@login_required_api
@model_required
def friends():
    return jsonify(friends=cs.list_friends(current_user.id))


@bp.delete("/friends/<int:uid>")
@login_required_api
@model_required
def unfriend(uid: int):
    return jsonify(cs.unfriend(current_user.id, uid))


@bp.get("/friend-requests")
@login_required_api
@model_required
def friend_requests():
    return jsonify(cs.list_requests(current_user.id))


@bp.post("/friend-requests")
@login_required_api
@model_required
def send_request():
    uid = _uid(request.get_json(silent=True))
    if uid is None:
        return jsonify(error="Thiếu user_id."), 400
    return jsonify(cs.send_friend_request(current_user.id, uid)), 201


@bp.post("/friend-requests/<int:rid>/accept")
@login_required_api
@model_required
def accept(rid: int):
    return jsonify(cs.respond_friend_request(current_user.id, rid, True))


@bp.post("/friend-requests/<int:rid>/decline")
@login_required_api
@model_required
def decline(rid: int):
    return jsonify(cs.respond_friend_request(current_user.id, rid, False))


@bp.delete("/friend-requests/<int:rid>")
@login_required_api
@model_required
def cancel(rid: int):
    return jsonify(cs.cancel_friend_request(current_user.id, rid))


@bp.get("/blocks")
@login_required_api
@model_required
def blocks():
    return jsonify(blocked=cs.list_blocked(current_user.id))


@bp.post("/blocks")
@login_required_api
@model_required
def block():
    uid = _uid(request.get_json(silent=True))
    if uid is None:
        return jsonify(error="Thiếu user_id."), 400
    return jsonify(cs.block_user(current_user.id, uid)), 201


@bp.delete("/blocks/<int:uid>")
@login_required_api
@model_required
def unblock(uid: int):
    return jsonify(cs.unblock_user(current_user.id, uid))
