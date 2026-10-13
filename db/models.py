"""Cac bang cua CSDL data_scam."""
from flask_login import UserMixin
from sqlalchemy import UniqueConstraint, CheckConstraint, Index
from werkzeug.security import check_password_hash, generate_password_hash

from .database import db, utcnow

AVATAR_COLORS = ["#1b74e4", "#e4572e", "#7b3fe4", "#2e9e5b", "#d6336c", "#f08c00", "#0b7285", "#5f3dc4"]


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    # Moi mo hinh AI co "the gioi" tai khoan rieng: cung ten van tao duoc o mo hinh khac.
    model_key = db.Column(db.String(20), nullable=False, default="svm")
    username = db.Column(db.String(30), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    avatar_color = db.Column(db.String(9), nullable=False, default="#1b74e4")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    last_seen_at = db.Column(db.DateTime, nullable=True)

    __table_args__ = (UniqueConstraint("model_key", "username", name="uq_user_model_username"),)

    def set_password(self, raw: str):
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)


class FriendRequest(db.Model):
    """Loi moi ket ban. status: pending -> accepted | declined. Ban be = ban ghi accepted."""
    __tablename__ = "friend_requests"
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    status = db.Column(db.String(10), nullable=False, default="pending")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    responded_at = db.Column(db.DateTime, nullable=True)

    sender = db.relationship("User", foreign_keys=[sender_id])
    receiver = db.relationship("User", foreign_keys=[receiver_id])
    __table_args__ = (
        UniqueConstraint("sender_id", "receiver_id", name="uq_friend_request_pair"),
        CheckConstraint("sender_id <> receiver_id", name="ck_friend_request_not_self"),
        Index("ix_friend_requests_receiver_status", "receiver_id", "status"),
    )


class Block(db.Model):
    __tablename__ = "blocks"
    id = db.Column(db.Integer, primary_key=True)
    blocker_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    blocked_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    blocked = db.relationship("User", foreign_keys=[blocked_id])
    __table_args__ = (UniqueConstraint("blocker_id", "blocked_id", name="uq_block_pair"),)


class Conversation(db.Model):
    """Cuoc chat 1-1. user_low_id < user_high_id de moi cap chi co 1 cuoc chat."""
    __tablename__ = "conversations"
    id = db.Column(db.Integer, primary_key=True)
    user_low_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    user_high_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    last_message_at = db.Column(db.DateTime, nullable=True)

    user_low = db.relationship("User", foreign_keys=[user_low_id])
    user_high = db.relationship("User", foreign_keys=[user_high_id])
    __table_args__ = (
        UniqueConstraint("user_low_id", "user_high_id", name="uq_conversation_pair"),
        CheckConstraint("user_low_id < user_high_id", name="ck_conversation_order"),
    )

    def other_user_id(self, me_id: int) -> int:
        return self.user_high_id if me_id == self.user_low_id else self.user_low_id

    def has_user(self, uid: int) -> bool:
        return uid in (self.user_low_id, self.user_high_id)


class Message(db.Model):
    __tablename__ = "messages"
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kind = db.Column(db.String(10), nullable=False, default="text")          # text | image | file
    body = db.Column(db.Text, nullable=True)
    attachment_path = db.Column(db.String(255), nullable=True)               # ten file da luu trong uploads/
    attachment_name = db.Column(db.String(255), nullable=True)               # ten file goc
    attachment_mime = db.Column(db.String(100), nullable=True)
    attachment_size = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    delivered_at = db.Column(db.DateTime, nullable=True)
    read_at = db.Column(db.DateTime, nullable=True)
    # ---- ket qua kiem duyet cua AI (CHI nguoi nhan duoc thay) ----
    ai_checked = db.Column(db.Boolean, nullable=False, default=False)
    ai_model = db.Column(db.String(20), nullable=True)
    ai_score = db.Column(db.Float, nullable=True)
    ai_flagged = db.Column(db.Boolean, nullable=False, default=False)
    warning_level = db.Column(db.SmallInteger, nullable=False, default=0)    # 0 | 1 | 2 | 3

    conversation = db.relationship("Conversation", backref=db.backref("messages", lazy="dynamic",
                                                                       cascade="all, delete-orphan"))
    sender = db.relationship("User", foreign_keys=[sender_id])
    __table_args__ = (Index("ix_messages_conv_id", "conversation_id", "id"),
                      Index("ix_messages_unread", "conversation_id", "sender_id", "read_at"))


class ScamCounter(db.Model):
    """So tin bi AI gan co cua MOI nguoi gui trong TUNG cuoc chat (cong don, khong reset)."""
    __tablename__ = "scam_counters"
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    flagged_count = db.Column(db.Integer, nullable=False, default=0)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)


class FeedAuthor(db.Model):
    """Tai khoan GIA chi dung cho bang tin (khong dang nhap duoc)."""
    __tablename__ = "feed_authors"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    avatar_color = db.Column(db.String(9), nullable=False)
    bio = db.Column(db.String(120), nullable=True)


class FeedPost(db.Model):
    __tablename__ = "feed_posts"
    id = db.Column(db.Integer, primary_key=True)
    author_id = db.Column(db.Integer, db.ForeignKey("feed_authors.id", ondelete="CASCADE"), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image = db.Column(db.String(200), nullable=True)
    likes = db.Column(db.Integer, nullable=False, default=0)
    comments = db.Column(db.Integer, nullable=False, default=0)
    shares = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    author = db.relationship("FeedAuthor")
    __table_args__ = (Index("ix_feed_posts_created", "created_at"),)
