"""Trong tai AI: kiem duyet tin nhan va quan ly muc canh bao 1 -> 2 -> 3.

Quy tac (da thong nhat voi nguoi dung):
  * Dem so tin bi AI gan co cua MOI NGUOI GUI trong TUNG cuoc chat, cong don, khong reset.
  * Lan 1 = vang, lan 2 = do, lan 3 tro di = do + rung + goi y chan (muc toi da la 3).
  * Chi NGUOI NHAN thay canh bao lan 1, 2, 3. Nguoi gui chi thay canh bao tu lan 3.
  * Khong bao gio tu dong chan - chi goi y.
"""
from __future__ import annotations

from dataclasses import dataclass

from db.database import db
from db.models import ScamCounter
from services.ai_client import manager

MAX_LEVEL = 3
SELF_WARNING_TEXT = ("Hệ thống phát hiện bạn đang là đối tượng đi lừa đảo người khác, "
                     "hãy dừng lại không thì bạn sẽ đi TÒ")


@dataclass
class Verdict:
    checked: bool = False
    model: str | None = None
    score: float | None = None
    flagged: bool = False
    level: int = 0
    count: int = 0                 # tong so tin bi gan co cua nguoi gui (chua gioi han muc 3)


def level_for(count: int) -> int:
    return min(max(count, 0), MAX_LEVEL)


def get_count(conversation_id: int, user_id: int) -> int:
    row = db.session.get((ScamCounter), (conversation_id, user_id))
    return row.flagged_count if row else 0


def moderate(conversation_id: int, sender_id: int, text: str | None) -> Verdict:
    """Cho AI doc tin va cap nhat bo dem. Goi trong cung transaction voi viec luu tin nhan."""
    if not text:
        return Verdict()
    result = manager.predict(text)
    if result is None:
        return Verdict()
    score, flagged = result
    verdict = Verdict(checked=True, model=manager.active, score=round(score, 4), flagged=flagged)
    if flagged:
        row = (ScamCounter.query.filter_by(conversation_id=conversation_id, user_id=sender_id)
               .with_for_update().first())
        if row is None:
            row = ScamCounter(conversation_id=conversation_id, user_id=sender_id, flagged_count=0)
            db.session.add(row)
        row.flagged_count += 1
        db.session.flush()
        verdict.level, verdict.count = level_for(row.flagged_count), row.flagged_count
    return verdict


def conversation_state(conversation_id: int, viewer_id: int, other_id: int) -> dict:
    """Trang thai canh bao cua cuoc chat *tu goc nhin cua viewer*.

    warning:      muc canh bao do DOI PHUONG gay ra (viewer la nguoi nhan) - 0..3
    self_warning: True neu CHINH viewer da bi gan co tu 3 lan tro len (hien canh bao 'di to')
    """
    other = get_count(conversation_id, other_id)
    mine = get_count(conversation_id, viewer_id)
    return {"warning": level_for(other), "self_warning": level_for(mine) >= MAX_LEVEL}
