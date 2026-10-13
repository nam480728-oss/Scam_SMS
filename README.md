# SafeChat — website chat có AI trọng tài phát hiện lừa đảo

Website chat kiểu Facebook / Messenger. Khi hai người nhắn tin, **AI đứng giữa** đọc từng tin nhắn và cảnh báo người nhận
nếu nghi ngờ lừa đảo. Có thể chọn 1 trong 3 mô hình đã train trong `ai_training/`:
**TF-IDF + Linear SVM**, **PhoBERT-base-v2**, **ViSoBERT**.

## Chạy ứng dụng

```powershell
# 1. Bật MySQL (XAMPP) – CSDL `data_scam` đã được tạo sẵn và đã chạy migration (2 phiên bản)
# 2. Cài thư viện (một lần)
pip install -r requirements.txt

# 3. (Nếu chưa) tạo CSDL + bảng + dữ liệu bảng tin giả
python -m flask --app app db upgrade
python -m flask --app app seed-feed

# 4. Chạy
python app.py            # mở http://127.0.0.1:5000
```

Cấu hình kết nối CSDL nằm trong file `.env` (mặc định `root` không mật khẩu, `127.0.0.1:3306`, CSDL `data_scam`).
Các bảng (xem trong Navicat → refresh): `users`, `friend_requests`, `blocks`, `conversations`, `messages`, `scam_counters`,
`feed_authors`, `feed_posts`, `alembic_version`.

## Công nghệ

- **Giao diện:** HTML + CSS + JavaScript thuần (không dùng React/Bootstrap), Socket.IO client cho thời gian thực.
- **Máy chủ:** Python **Flask** (Flask-Login, Flask-WTF/CSRF, Flask-SocketIO), SQLAlchemy + Alembic (Flask-Migrate).
- **CSDL:** MySQL/MariaDB (`data_scam`). **AI:** scikit-learn, PyTorch + Transformers (từ `ai_training/`).

## Luồng sử dụng

1. **Chủ phần mềm chọn mô hình** (lần đầu mở web, khi server chưa có mô hình nào): bấm 1 trong 3 thẻ; mô hình được nạp
   và áp dụng cho **toàn server**. **Người dùng không được chọn mô hình** — họ chỉ đăng ký / đăng nhập.
2. **Mỗi mô hình có "thế giới" tài khoản riêng**: tài khoản tạo ở SVM không tồn tại ở PhoBERT / ViSoBERT (cùng tên vẫn
   đăng ký lại được), bạn bè / tin nhắn cũng tách biệt. Đổi lại mô hình cũ thì dữ liệu cũ xuất hiện lại.
3. **Đổi mô hình sau này**: vào `/admin` (link "Quản trị" ở trang đăng nhập), nhập mật khẩu quản trị
   (`ADMIN_PASSWORD` trong `.env`, mặc định `admin123` — hãy đổi). Mọi người dùng đang đăng nhập sẽ bị đăng xuất tự động.
4. **Đăng ký / Đăng nhập**: chỉ cần tên tài khoản + mật khẩu nhập 2 lần.
5. **Bảng tin**: bài đăng giả (dữ liệu mô phỏng, chỉ xem).
6. **Tin nhắn**: tìm người dùng → *Kết bạn* → đối phương *Chấp nhận* → nhắn tin thời gian thực
   (đang nhập, online/offline, đã gửi/đã nhận/đã xem, emoji, gửi ảnh/tệp, chặn / bỏ chặn, hủy kết bạn).

## AI cảnh báo 3 mức

Đếm số tin bị AI gán cờ **của mỗi người gửi, trong từng cuộc chat, cộng dồn, không reset**:

| Lần | Người nhận thấy | Người gửi thấy |
|---|---|---|
| 1 | 🟡 nhãn dưới tin + banner vàng | không thấy gì |
| 2 | 🔴 nhãn dưới tin + banner đỏ | không thấy gì |
| 3+ | 🔴 banner đỏ có biểu tượng rung + gợi ý nút **Chặn** (nhãn dưới tin chỉ nhắc, nút chặn nằm ở banner) | hộp thoại (lần 3) + banner luôn hiện, không bao giờ mất trong đoạn chat đó: *“Hệ thống phát hiện bạn đang là đối tượng đi lừa đảo người khác, hãy dừng lại không thì bạn sẽ đi TÒ”* |

Hệ thống **không bao giờ tự chặn**, chỉ gợi ý. AI chỉ kiểm tra phần văn bản (ảnh/tệp không được kiểm duyệt).
Kết quả AI **chỉ gửi cho người nhận**; API không bao giờ trả cờ cho người gửi.

## Cấu trúc

```
app.py  config.py  .env  requirements.txt
db/        database.py  models.py            # SQLAlchemy + các bảng
migrations/                                   # Alembic (Flask-Migrate)
services/  ai_client.py (nạp 3 mô hình)  alert_manager.py (bộ đếm 3 mức)
           chat_service.py (nghiệp vụ)  realtime.py (Socket.IO)  seed_feed.py
routes/    models.py auth.py feed.py chat.py friends.py
templates/ static/  (css, js, vendor/socket.io, img/posts)
uploads/   # ảnh/tệp người dùng gửi (không công khai, tải qua API có kiểm tra quyền)
tests/     # pytest: 24 test (DB thử riêng data_scam_test, AI giả lập)
ai_training/                                  # code train + mô hình
```

Mô hình được lấy tự động từ lần train đầy đủ mới nhất trong `ai_training/*/outputs/` (bỏ qua các lần `--smoke-test`);
có thể chỉ định đường dẫn trong `.env` bằng `MODEL_SVM_PATH`, `MODEL_PHOBERT_PATH`, `MODEL_VISO_PATH`.

## Kiểm thử

```powershell
python -m pytest tests -q        # tạo CSDL tạm data_scam_test (chạy migration thật), xong tự xoá
```

## Lưu ý

- Chạy bằng server phát triển (Werkzeug + Socket.IO chế độ threading) — đủ cho demo; triển khai thật cần WSGI server riêng.
- Mọi người dùng chung 1 mô hình đang hoạt động; chỉ quản trị viên (mật khẩu `ADMIN_PASSWORD`) mới đổi được.
- Trên Windows, `app.py` tự khai báo lại MIME cho `.js/.css/.svg` vì registry máy có thể trả sai (`.js` = `text/plain`) làm trình duyệt chặn script.
