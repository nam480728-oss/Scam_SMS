"""Du lieu GIA cho bang tin: tac gia hu cau, bai dang, anh SVG sinh san (khong can Internet)."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from config import BASE_DIR
from db.database import db, utcnow
from db.models import FeedAuthor, FeedPost

IMG_DIR = BASE_DIR / "static" / "img" / "posts"

# (ten, mau, tieu su)  -- toan bo la nhan vat/to chuc HU CAU
AUTHORS = [
    ("Minh Anh", "#d6336c", "Thích du lịch và cà phê"),
    ("Hội Cảnh Giác Mạng", "#c92a2a", "Trang cộng đồng chia sẻ cảnh báo lừa đảo (nội dung mô phỏng)"),
    ("Quán Bún Chả Cô Lan", "#f08c00", "Bún chả, nem rán – Hà Nội"),
    ("Tuấn Kiệt", "#1b74e4", "Lập trình viên · Game thủ"),
    ("Thu Hà Review", "#7b3fe4", "Review đồ ăn & mỹ phẩm"),
    ("Mẹo Vặt Cuộc Sống", "#2e9e5b", "Mẹo hay mỗi ngày"),
    ("Hoàng Long", "#0b7285", "Chạy bộ · Đạp xe · Leo núi"),
    ("Mẹ Bống & Chó Mực", "#e4572e", "Nhật ký nuôi thú cưng"),
    ("Góc Công Nghệ", "#5f3dc4", "Tin công nghệ mỗi ngày"),
]

# (chi so tac gia, noi dung, ten anh, tieu de anh, emoji, mau1, mau2, like, comment, share, phut truoc)
POSTS = [
    (1, "⚠️ CẢNH BÁO: Gần đây xuất hiện tin nhắn mạo danh ngân hàng với nội dung 'tài khoản của bạn sắp bị khóa, bấm vào link để xác thực'. "
        "Ngân hàng KHÔNG yêu cầu bạn cung cấp mật khẩu hay mã OTP qua tin nhắn. Hãy xóa tin và không bấm vào đường link lạ!",
     "p_warn1", "Đừng bấm link lạ!", "🚫", "#c92a2a", "#f76707", 1840, 322, 910, 35),
    (0, "Cuối tuần lên Đà Lạt săn mây cùng hội bạn thân ☁️ Sáng nay mây phủ kín đồi, lạnh nhưng cực kỳ đáng!",
     "p_dalat", "Đà Lạt · Săn mây", "⛰️", "#364fc7", "#74c0fc", 956, 48, 12, 90),
    (2, "Quán mới có món bún chả nướng than hoa, thơm lừng cả con phố 🔥 Ai ở gần ghé ủng hộ nhé, hôm nay tặng kèm nem rán cho 20 khách đầu tiên!",
     "p_bunscha", "Bún chả nướng than hoa", "🍜", "#e8590c", "#ffd43b", 421, 67, 20, 180),
    (3, "Vừa build xong con PC mới, chạy mượt 144fps. Ai cần tư vấn cấu hình cứ hỏi nhé 💻🎮",
     "p_pc", "Build PC mới", "🖥️", "#212529", "#845ef7", 312, 54, 6, 260),
    (1, "💡 Cách nhận biết lừa đảo 'người thân nhờ chuyển tiền gấp': kẻ gian thường tạo cảm giác khẩn cấp (tai nạn, nhập viện), "
        "nhắn từ số lạ hoặc tài khoản mới. Hãy GỌI ĐIỆN xác minh trực tiếp trước khi chuyển bất kỳ khoản nào.",
     "p_warn2", "Gọi xác minh trước khi chuyển tiền", "📞", "#e03131", "#ff8787", 2210, 410, 1302, 330),
    (4, "Review nhanh: serum vitamin C giá học sinh, dùng 2 tuần thấy da đều màu hơn hẳn ✨ Link chi tiết mình để ở phần bình luận nha.",
     "p_serum", "Review serum vitamin C", "🧴", "#d6336c", "#ffa8a8", 288, 91, 15, 420),
    (5, "Mẹo nhỏ: bỏ vài lát chanh vào nồi cơm khi nấu, cơm sẽ không bị thiu nhanh vào mùa nóng 🍋",
     "p_meo", "Mẹo vặt nhà bếp", "🍋", "#94d82d", "#fff3bf", 733, 29, 140, 500),
    (6, "5h sáng chạy 10km quanh hồ, không khí trong lành cực kỳ! Ai muốn chạy cùng hội cuối tuần này không? 🏃‍♂️",
     "p_run", "Chạy bộ buổi sáng", "🏃", "#0b7285", "#99e9f2", 505, 33, 4, 640),
    (7, "Bống nhà mình học được trò bắt tay rồi nha 🐶 Chó Mực thì vẫn chỉ thích nằm ngủ thôi 😂",
     "p_dog", "Bống & Mực", "🐕", "#f59f00", "#ffe066", 1290, 140, 30, 720),
    (1, "🎯 Thông báo trúng thưởng giả: 'Chúc mừng bạn nhận được iPhone, nộp phí vận chuyển 500k để nhận quà'. "
        "Đây là chiêu lừa quen thuộc! Không có giải thưởng nào bắt bạn nộp phí trước. Hãy chặn và báo cáo người gửi.",
     "p_warn3", "Trúng thưởng? Coi chừng!", "🎁", "#862e9c", "#e599f7", 1675, 288, 801, 800),
    (8, "Tin công nghệ: các mô hình AI tiếng Việt ngày càng giúp phát hiện tin nhắn rác và lừa đảo chính xác hơn 🤖",
     "p_ai", "AI tiếng Việt", "🤖", "#5f3dc4", "#b197fc", 664, 72, 58, 930),
    (0, "Hoàng hôn Vũng Tàu hôm nay đẹp quá trời 🌅 Cuối tuần nào cũng muốn chạy ra biển!",
     "p_sunset", "Hoàng hôn Vũng Tàu", "🌅", "#d9480f", "#ffa94d", 1102, 61, 9, 1100),
    (2, "Thông báo: quán nghỉ lễ ngày mai, mở lại thứ Ba. Cảm ơn mọi người đã luôn ủng hộ 🙏",
     "p_notice", "Quán nghỉ lễ", "🏮", "#c92a2a", "#ffc9c9", 98, 12, 1, 1300),
    (3, "Mới học xong khóa Python cơ bản, tự viết được con bot nhắc uống nước 😄 Code mình để trên GitHub nha.",
     "p_python", "Học Python", "🐍", "#2b8a3e", "#8ce99a", 377, 45, 11, 1500),
    (1, "🔒 Bật xác thực hai lớp cho tài khoản ngân hàng và mạng xã hội, đặt mật khẩu khác nhau cho từng dịch vụ. "
        "Đó là hai việc đơn giản nhất để tự bảo vệ mình.",
     "p_secure", "Bật xác thực 2 lớp", "🔐", "#1864ab", "#74c0fc", 1520, 190, 620, 1700),
    (5, "Cách gấp quần áo gọn như cửa hàng, vali nhỏ vẫn nhét được cả tuần đồ 🧳",
     "p_fold", "Gấp đồ gọn gàng", "🧳", "#099268", "#96f2d7", 842, 38, 210, 2000),
    (6, "Cung đường đạp xe Ba Vì cuối tuần: dốc vừa đủ mệt, view thì khỏi bàn ⛰️🚴",
     "p_bike", "Đạp xe Ba Vì", "🚴", "#2f9e44", "#d8f5a2", 458, 27, 5, 2400),
    (4, "Chè khúc bạch tự làm: mát lạnh ngày hè, công thức siêu dễ, ai thích mình up công thức nhé 🍮",
     "p_che", "Chè khúc bạch", "🍮", "#e599f7", "#fff0f6", 690, 83, 44, 2800),
]


def _svg(title: str, emoji: str, c1: str, c2: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450" width="800" height="450">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient></defs>
<rect width="800" height="450" fill="url(#g)"/>
<circle cx="680" cy="80" r="150" fill="#fff" opacity="0.10"/><circle cx="90" cy="400" r="190" fill="#fff" opacity="0.10"/>
<text x="400" y="220" font-size="150" text-anchor="middle" dominant-baseline="middle">{emoji}</text>
<text x="400" y="360" font-family="Segoe UI, Arial, sans-serif" font-size="40" font-weight="700" fill="#fff" text-anchor="middle" style="paint-order:stroke" stroke="rgba(0,0,0,.25)" stroke-width="6">{title}</text>
</svg>"""


def seed_feed(reset: bool = False) -> int:
    if reset:
        FeedPost.query.delete()
        FeedAuthor.query.delete()
        db.session.commit()
    elif FeedPost.query.count():
        return 0
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    authors = []
    for name, color, bio in AUTHORS:
        a = FeedAuthor(name=name, avatar_color=color, bio=bio)
        db.session.add(a)
        authors.append(a)
    db.session.flush()
    now = utcnow()
    for ai, content, img, title, emoji, c1, c2, likes, comments, shares, mins in POSTS:
        (IMG_DIR / f"{img}.svg").write_text(_svg(title, emoji, c1, c2), encoding="utf-8")
        db.session.add(FeedPost(author_id=authors[ai].id, content=content, image=f"img/posts/{img}.svg",
                                likes=likes, comments=comments, shares=shares,
                                created_at=now - timedelta(minutes=mins)))
    db.session.commit()
    return len(POSTS)
