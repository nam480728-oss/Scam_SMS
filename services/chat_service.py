"""Nghiep vu chat: ban be, chan, hoi thoai, tin nhan, trang thai da nhan/da xem, dinh kem."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import current_app
from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError
from werkzeug.utils import secure_filename

from db.database import db, iso, utcnow
from db.models import Block, Conversation, FriendRequest, Message, User
from services import alert_manager
from services.ai_client import manager
from services.realtime import emit_to, is_online


class ChatError(Exception):
    """Loi nghiep vu co thong diep hien thi cho nguoi dung."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


# --------------------------------------------------------------------------- #
# Quan he giua 2 nguoi dung
# --------------------------------------------------------------------------- #
def _pair_request(a: int, b: int) -> FriendRequest | None:
    return FriendRequest.query.filter(or_(
        and_(FriendRequest.sender_id == a, FriendRequest.receiver_id == b),
        and_(FriendRequest.sender_id == b, FriendRequest.receiver_id == a))).first()


def are_friends(a: int, b: int) -> bool:
    fr = _pair_request(a, b)
    return bool(fr and fr.status == "accepted")


def block_state(me: int, other: int) -> tuple[bool, bool]:
    """(toi da chan ho, ho da chan toi)"""
    rows = Block.query.filter(or_(and_(Block.blocker_id == me, Block.blocked_id == other),
                                  and_(Block.blocker_id == other, Block.blocked_id == me))).all()
    return (any(r.blocker_id == me for r in rows), any(r.blocker_id == other for r in rows))


def friend_ids(uid: int) -> set[int]:
    rows = FriendRequest.query.filter(FriendRequest.status == "accepted",
                                      or_(FriendRequest.sender_id == uid, FriendRequest.receiver_id == uid)).all()
    return {r.receiver_id if r.sender_id == uid else r.sender_id for r in rows}


def relation(me: int, other: int) -> dict:
    i_blocked, blocked_me = block_state(me, other)
    fr = _pair_request(me, other)
    if fr and fr.status == "accepted":
        status = "friend"
    elif fr and fr.status == "pending":
        status = "sent" if fr.sender_id == me else "received"
    else:
        status = "none"
    return {"status": status, "request_id": fr.id if fr and fr.status == "pending" else None,
            "i_blocked": i_blocked, "blocked_me": blocked_me}


def get_peer(me: int, other_id: int) -> User:
    """Nguoi dung khac CUNG mo hinh AI (moi mo hinh co the gioi tai khoan rieng), neu khong thi coi nhu khong ton tai."""
    other = db.session.get(User, other_id)
    mine = db.session.get(User, me)
    if other is None or mine is None or other.model_key != mine.model_key:
        raise ChatError("Người dùng không tồn tại.", 404)
    return other


def user_dto(u: User, viewer_id: int, with_presence: bool = True) -> dict:
    d = {"id": u.id, "username": u.username, "color": u.avatar_color, "online": False, "last_seen": None}
    if with_presence and viewer_id != u.id and are_friends(viewer_id, u.id):
        i_blocked, blocked_me = block_state(viewer_id, u.id)
        if not (i_blocked or blocked_me):                      # nguoi bi chan khong xem duoc trang thai
            d["online"] = is_online(u.id)
            d["last_seen"] = iso(u.last_seen_at)
    return d


# --------------------------------------------------------------------------- #
# Hoi thoai
# --------------------------------------------------------------------------- #
def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def get_conversation(cid: int, me: int) -> Conversation:
    conv = db.session.get(Conversation, cid)
    if conv is None or not conv.has_user(me):
        raise ChatError("Không tìm thấy cuộc trò chuyện.", 404)
    return conv


def get_or_create_conversation(me: int, other_id: int) -> Conversation:
    if other_id == me:
        raise ChatError("Không thể trò chuyện với chính mình.")
    get_peer(me, other_id)
    low, high = _pair(me, other_id)
    conv = Conversation.query.filter_by(user_low_id=low, user_high_id=high).first()
    if conv:
        return conv
    if not are_friends(me, other_id):
        raise ChatError("Hai bạn cần kết bạn trước khi trò chuyện.", 403)
    conv = Conversation(user_low_id=low, user_high_id=high)
    db.session.add(conv)
    try:
        db.session.commit()
    except Exception:  # dua nhau tao dong thoi -> lay ban ghi da co
        db.session.rollback()
        conv = Conversation.query.filter_by(user_low_id=low, user_high_id=high).first()
        if conv is None:
            raise
    return conv


def send_permission(me: int, other: int) -> tuple[bool, str | None]:
    i_blocked, blocked_me = block_state(me, other)
    if i_blocked:
        return False, "Bạn đã chặn người này. Hãy bỏ chặn để nhắn tin."
    if blocked_me:
        return False, "Bạn không thể nhắn tin cho người này."
    if not are_friends(me, other):
        return False, "Hai bạn không còn là bạn bè."
    return True, None


def unread_count(conv_id: int, me: int) -> int:
    return Message.query.filter(Message.conversation_id == conv_id, Message.sender_id != me,
                                Message.read_at.is_(None)).count()


def _preview(m: Message) -> str:
    if m.kind == "image":
        return "📷 Ảnh" + (f": {m.body}" if m.body else "")
    if m.kind == "file":
        return f"📎 {m.attachment_name}"
    return m.body or ""


def message_dto(m: Message, viewer_id: int) -> dict:
    d = {"id": m.id, "conversation_id": m.conversation_id, "sender_id": m.sender_id, "mine": m.sender_id == viewer_id,
         "kind": m.kind, "body": m.body, "created_at": iso(m.created_at), "delivered_at": iso(m.delivered_at),
         "read_at": iso(m.read_at), "attachment": None, "warning_level": 0}
    if m.kind in ("image", "file"):
        d["attachment"] = {"url": f"/api/attachments/{m.id}", "name": m.attachment_name, "mime": m.attachment_mime,
                           "size": m.attachment_size, "is_image": m.kind == "image"}
    # Ket qua AI CHI gui cho NGUOI NHAN. Nguoi gui khong bao gio thay co cua chinh minh qua API.
    if m.sender_id != viewer_id and m.ai_flagged:
        d["warning_level"] = m.warning_level
    return d


def conversation_dto(conv: Conversation, me: int) -> dict:
    other_id = conv.other_user_id(me)
    other = db.session.get(User, other_id)
    last = conv.messages.order_by(Message.id.desc()).first()
    allowed, reason = send_permission(me, other_id)
    state = alert_manager.conversation_state(conv.id, me, other_id)
    i_blocked, blocked_me = block_state(me, other_id)
    return {"id": conv.id, "other": user_dto(other, me), "unread": unread_count(conv.id, me),
            "last_message": None if last is None else {"preview": _preview(last), "mine": last.sender_id == me,
                                                       "created_at": iso(last.created_at)},
            "last_message_at": iso(conv.last_message_at), "can_send": allowed, "send_block_reason": reason,
            "i_blocked": i_blocked, "blocked_me": blocked_me, **state}


def list_conversations(me: int) -> list[dict]:
    convs = (Conversation.query.filter(or_(Conversation.user_low_id == me, Conversation.user_high_id == me),
                                       Conversation.last_message_at.isnot(None))
             .order_by(Conversation.last_message_at.desc()).all())
    return [conversation_dto(c, me) for c in convs]


def history(conv: Conversation, me: int, before: int | None, limit: int) -> dict:
    q = conv.messages
    if before:
        q = q.filter(Message.id < before)
    rows = q.order_by(Message.id.desc()).limit(limit + 1).all()
    has_more = len(rows) > limit
    rows = list(reversed(rows[:limit]))
    return {"messages": [message_dto(m, me) for m in rows], "has_more": has_more}


# --------------------------------------------------------------------------- #
# Gui tin nhan + AI trong tai
# --------------------------------------------------------------------------- #
def _store_attachment(file_storage) -> dict:
    cfg = current_app.config
    original = (file_storage.filename or "").strip()
    ext = original.rsplit(".", 1)[-1].lower() if "." in original else ""
    if ext not in cfg["IMAGE_EXTS"] | cfg["FILE_EXTS"]:
        raise ChatError("Loại tệp này không được hỗ trợ (ảnh png/jpg/gif/webp hoặc pdf, doc, xls, txt, zip...).")
    stream = file_storage.stream
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    if size == 0:
        raise ChatError("Tệp rỗng.")
    if size > cfg["MAX_UPLOAD_BYTES"]:
        raise ChatError(f"Tệp quá lớn (tối đa {cfg['MAX_UPLOAD_BYTES'] // 2**20} MB).")
    stored = f"{uuid.uuid4().hex}.{ext}"
    Path(cfg["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    file_storage.save(Path(cfg["UPLOAD_DIR"]) / stored)
    display = secure_filename(original) or f"tep.{ext}"
    if original:                                         # giu ten co dau de hien thi, bo ky tu nguy hiem
        display = "".join(c for c in original if c not in '\\/:*?"<>|\r\n\x00')[:200] or display
    kind = "image" if ext in cfg["IMAGE_EXTS"] else "file"
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
            "webp": "image/webp"}.get(ext, "application/octet-stream")
    return {"kind": kind, "path": stored, "name": display, "size": size, "mime": mime}


def send_message(conv: Conversation, sender: User, body: str | None, file_storage=None) -> dict:
    cfg = current_app.config
    other_id = conv.other_user_id(sender.id)
    allowed, reason = send_permission(sender.id, other_id)
    if not allowed:
        raise ChatError(reason, 403)
    if not manager.ready():
        raise ChatError("Chưa chọn mô hình AI. Hãy quay lại trang chọn mô hình.", 503)
    body = (body or "").strip()
    if len(body) > cfg["MAX_MESSAGE_LEN"]:
        raise ChatError(f"Tin nhắn quá dài (tối đa {cfg['MAX_MESSAGE_LEN']} ký tự).")
    has_file = file_storage is not None and bool(file_storage.filename)
    if not body and not has_file:
        raise ChatError("Tin nhắn trống.")

    att = _store_attachment(file_storage) if has_file else None
    try:
        msg = Message(conversation_id=conv.id, sender_id=sender.id, kind=att["kind"] if att else "text",
                      body=body or None)
        if att:
            msg.attachment_path, msg.attachment_name = att["path"], att["name"]
            msg.attachment_mime, msg.attachment_size = att["mime"], att["size"]
        # ---- AI dung giua: chi kiem tra phan van ban ----
        verdict = alert_manager.moderate(conv.id, sender.id, body)
        msg.ai_checked, msg.ai_model, msg.ai_score = verdict.checked, verdict.model, verdict.score
        msg.ai_flagged, msg.warning_level = verdict.flagged, verdict.level
        recipient_online = is_online(other_id)
        if recipient_online:
            msg.delivered_at = utcnow()
        conv.last_message_at = msg.created_at = utcnow()
        db.session.add(msg)
        db.session.commit()
    except Exception:
        db.session.rollback()
        if att:
            try:
                (Path(cfg["UPLOAD_DIR"]) / att["path"]).unlink(missing_ok=True)
            except OSError:
                pass
        raise

    for viewer_id in (sender.id, other_id):
        payload = {"message": message_dto(msg, viewer_id), "conversation": conversation_dto(conv, viewer_id)}
        if viewer_id == sender.id and verdict.flagged and verdict.level >= alert_manager.MAX_LEVEL:
            payload["self_warning"] = alert_manager.SELF_WARNING_TEXT      # chi nguoi gui, tu lan 3
            payload["self_warning_count"] = verdict.count
        emit_to(viewer_id, "message:new", payload)
    return message_dto(msg, sender.id)


def mark_read(conv: Conversation, me: int) -> dict:
    now = utcnow()
    rows = Message.query.filter(Message.conversation_id == conv.id, Message.sender_id != me,
                                Message.read_at.is_(None)).all()
    ids = [m.id for m in rows]
    for m in rows:
        m.read_at = now
        if m.delivered_at is None:
            m.delivered_at = now
    db.session.commit()
    other_id = conv.other_user_id(me)
    if ids:
        emit_to(other_id, "message:status", {"conversation_id": conv.id, "message_ids": ids, "status": "seen",
                                             "at": iso(now)})
    emit_to(me, "conversation:read", {"conversation_id": conv.id})        # dong bo cac tab khac cua toi
    return {"marked": len(ids)}


def relay_typing(me: int, cid, is_typing: bool):
    if not isinstance(cid, int):
        return
    conv = db.session.get(Conversation, cid)
    if conv is None or not conv.has_user(me):
        return
    other = conv.other_user_id(me)
    if send_permission(me, other)[0] and not block_state(me, other)[1]:
        emit_to(other, "typing", {"conversation_id": cid, "user_id": me, "is_typing": bool(is_typing)})


# --------------------------------------------------------------------------- #
# Hien dien
# --------------------------------------------------------------------------- #
def _notify_presence(uid: int, online: bool):
    user = db.session.get(User, uid)
    for fid in friend_ids(uid):
        i_blocked, blocked_me = block_state(uid, fid)
        if i_blocked or blocked_me:
            continue
        emit_to(fid, "presence", {"user_id": uid, "online": online, "last_seen": iso(user.last_seen_at) if user else None})


def on_user_online(uid: int):
    """Danh dau cac tin gui cho toi la 'da nhan' va bao cho nguoi gui; thong bao ban be."""
    now = utcnow()
    pending = (Message.query.join(Conversation, Message.conversation_id == Conversation.id)
               .filter(Message.sender_id != uid, Message.delivered_at.is_(None),
                       or_(Conversation.user_low_id == uid, Conversation.user_high_id == uid)).all())
    by_conv: dict[tuple[int, int], list[int]] = {}
    for m in pending:
        m.delivered_at = now
        by_conv.setdefault((m.conversation_id, m.sender_id), []).append(m.id)
    db.session.commit()
    for (cid, sender_id), ids in by_conv.items():
        emit_to(sender_id, "message:status", {"conversation_id": cid, "message_ids": ids, "status": "delivered",
                                              "at": iso(now)})
    _notify_presence(uid, True)


def on_user_offline(uid: int):
    user = db.session.get(User, uid)
    if user:
        user.last_seen_at = utcnow()
        db.session.commit()
    _notify_presence(uid, False)


# --------------------------------------------------------------------------- #
# Ket ban / chan
# --------------------------------------------------------------------------- #
def search_users(me: int, q: str, limit: int = 20) -> list[dict]:
    q = q.strip().lstrip("@")
    if not q:
        return []
    esc = q.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    model_key = db.session.get(User, me).model_key
    users = (User.query.filter(User.id != me, User.model_key == model_key, User.username.like(f"%{esc}%", escape="\\"))
             .order_by(User.username).limit(limit * 2).all())
    out = []
    for u in users:
        rel = relation(me, u.id)
        if rel["blocked_me"]:
            continue                                       # nguoi da chan toi: an khoi tim kiem
        out.append({**user_dto(u, me), "relation": rel["status"], "request_id": rel["request_id"],
                    "i_blocked": rel["i_blocked"]})
        if len(out) >= limit:
            break
    return out


def list_friends(me: int) -> list[dict]:
    users = User.query.filter(User.id.in_(friend_ids(me) or {0})).order_by(User.username).all()
    out = []
    for u in users:
        rel = relation(me, u.id)
        out.append({**user_dto(u, me), "i_blocked": rel["i_blocked"], "blocked_me": rel["blocked_me"]})
    return out


def list_requests(me: int) -> dict:
    inc = FriendRequest.query.filter_by(receiver_id=me, status="pending").order_by(FriendRequest.id.desc()).all()
    out = FriendRequest.query.filter_by(sender_id=me, status="pending").order_by(FriendRequest.id.desc()).all()
    return {"incoming": [{"id": r.id, "user": user_dto(r.sender, me, False), "created_at": iso(r.created_at)}
                         for r in inc if not block_state(me, r.sender_id)[1]],
            "outgoing": [{"id": r.id, "user": user_dto(r.receiver, me, False), "created_at": iso(r.created_at)}
                         for r in out]}


def _notify_friend_change(a: int, b: int, kind: str):
    for uid in (a, b):
        emit_to(uid, "friend:update", {"kind": kind})


def send_friend_request(me: int, other_id: int) -> dict:
    if other_id == me:
        raise ChatError("Không thể tự kết bạn với chính mình.")
    get_peer(me, other_id)
    i_blocked, blocked_me = block_state(me, other_id)
    if blocked_me:
        raise ChatError("Không thể gửi lời mời cho người này.", 403)
    if i_blocked:
        raise ChatError("Bạn đã chặn người này. Hãy bỏ chặn trước.", 403)
    fr = _pair_request(me, other_id)
    if fr:
        if fr.status == "accepted":
            raise ChatError("Hai bạn đã là bạn bè.")
        if fr.status == "pending":
            if fr.sender_id == me:
                raise ChatError("Bạn đã gửi lời mời rồi.")
            fr.status, fr.responded_at = "accepted", utcnow()            # ho da moi minh -> tu dong chap nhan
            db.session.commit()
            _notify_friend_change(me, other_id, "accepted")
            return {"status": "friend"}
        # da bi tu choi truoc do: cho phep moi lai (dao chieu ban ghi cu neu can)
        fr.sender_id, fr.receiver_id, fr.status, fr.responded_at, fr.created_at = me, other_id, "pending", None, utcnow()
    else:
        db.session.add(FriendRequest(sender_id=me, receiver_id=other_id))
    try:
        db.session.commit()
    except IntegrityError:                                   # hai nguoi gui dong thoi -> coi nhu da gui
        db.session.rollback()
        raise ChatError("Lời mời đã tồn tại, hãy tải lại danh sách.", 409)
    _notify_friend_change(me, other_id, "request")
    return {"status": "sent"}


def respond_friend_request(me: int, request_id: int, accept: bool) -> dict:
    fr = db.session.get(FriendRequest, request_id)
    if fr is None or fr.receiver_id != me or fr.status != "pending":
        raise ChatError("Lời mời không tồn tại hoặc đã được xử lý.", 404)
    if block_state(me, fr.sender_id)[1]:
        raise ChatError("Không thể xử lý lời mời này.", 403)
    fr.status, fr.responded_at = ("accepted" if accept else "declined"), utcnow()
    db.session.commit()
    _notify_friend_change(me, fr.sender_id, "accepted" if accept else "declined")
    return {"status": fr.status}


def cancel_friend_request(me: int, request_id: int) -> dict:
    fr = db.session.get(FriendRequest, request_id)
    if fr is None or fr.sender_id != me or fr.status != "pending":
        raise ChatError("Lời mời không tồn tại hoặc đã được xử lý.", 404)
    other = fr.receiver_id
    db.session.delete(fr)
    db.session.commit()
    _notify_friend_change(me, other, "cancelled")
    return {"status": "none"}


def unfriend(me: int, other_id: int) -> dict:
    fr = _pair_request(me, other_id)
    if not fr or fr.status != "accepted":
        raise ChatError("Hai bạn chưa là bạn bè.", 404)
    db.session.delete(fr)
    db.session.commit()
    _notify_friend_change(me, other_id, "unfriended")
    return {"status": "none"}


def block_user(me: int, other_id: int) -> dict:
    if other_id == me:
        raise ChatError("Không thể tự chặn chính mình.")
    get_peer(me, other_id)
    if not Block.query.filter_by(blocker_id=me, blocked_id=other_id).first():
        db.session.add(Block(blocker_id=me, blocked_id=other_id))
        db.session.commit()
    _notify_block_change(me, other_id)
    return {"i_blocked": True}


def unblock_user(me: int, other_id: int) -> dict:
    row = Block.query.filter_by(blocker_id=me, blocked_id=other_id).first()
    if row:
        db.session.delete(row)
        db.session.commit()
    _notify_block_change(me, other_id)
    return {"i_blocked": False}


def _notify_block_change(me: int, other_id: int):
    emit_to(me, "block:update", {"user_id": other_id})
    emit_to(other_id, "block:update", {"user_id": me})
    # cap nhat lai hien dien hai chieu theo trang thai chan moi
    i_blocked, blocked_me = block_state(me, other_id)
    visible = are_friends(me, other_id) and not (i_blocked or blocked_me)
    for a, b in ((me, other_id), (other_id, me)):
        user = db.session.get(User, a)
        emit_to(b, "presence", {"user_id": a, "online": bool(visible and is_online(a)),
                                "last_seen": iso(user.last_seen_at) if user and visible else None})


def list_blocked(me: int) -> list[dict]:
    rows = Block.query.filter_by(blocker_id=me).order_by(Block.id.desc()).all()
    return [user_dto(r.blocked, me, False) for r in rows]
