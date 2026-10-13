"""Socket.IO: phong rieng moi nguoi dung, hien dien (online), dang nhap, da nhan."""
from __future__ import annotations

import threading

from flask import request
from flask_login import current_user
from flask_socketio import SocketIO, emit, join_room

socketio = SocketIO()

_presence: dict[int, set[str]] = {}
_lock = threading.Lock()


def room(user_id: int) -> str:
    return f"user:{user_id}"


def is_online(user_id: int) -> bool:
    with _lock:
        return bool(_presence.get(user_id))


def emit_to(user_id: int, event: str, data: dict):
    socketio.emit(event, data, to=room(user_id))


def _register(user_id: int, sid: str) -> bool:
    """Tra ve True neu day la ket noi dau tien (vua chuyen sang online)."""
    with _lock:
        first = not _presence.get(user_id)
        _presence.setdefault(user_id, set()).add(sid)
        return first


def _unregister(user_id: int, sid: str) -> bool:
    """Tra ve True neu vua mat ket noi cuoi cung (chuyen sang offline)."""
    with _lock:
        sids = _presence.get(user_id)
        if not sids:
            return False
        sids.discard(sid)
        if not sids:
            _presence.pop(user_id, None)
            return True
        return False


def init_socket_events(app):
    from services import chat_service as cs

    @socketio.on("connect")
    def on_connect(auth=None):
        if not current_user.is_authenticated:
            return False                                   # tu choi ket noi chua dang nhap
        uid = current_user.id
        join_room(room(uid))
        if _register(uid, request.sid):
            with app.app_context():
                cs.on_user_online(uid)

    @socketio.on("disconnect")
    def on_disconnect(*args):
        uid = current_user.id if current_user.is_authenticated else None
        if uid is not None and _unregister(uid, request.sid):
            with app.app_context():
                cs.on_user_offline(uid)

    @socketio.on("typing")
    def on_typing(data):
        if not current_user.is_authenticated or not isinstance(data, dict):
            return
        cs.relay_typing(current_user.id, data.get("conversation_id"), bool(data.get("is_typing")))
