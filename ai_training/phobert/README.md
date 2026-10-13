# PhoBERT-base-v2 — Phân loại tin nhắn lừa đảo

Fine-tune `vinai/phobert-base-v2` để phân loại tin nhắn: **0 = bình thường, 1 = lừa đảo**.
Thư mục này tự chứa mọi thứ: code, tham số, và (sau khi train) toàn bộ kết quả/biểu đồ/báo cáo.

## 1. Cài đặt (làm một lần)

Máy hiện tại có GPU RTX 4060 (8GB) nhưng PyTorch đang là bản **CPU** → phải cài lại bản CUDA.
Mở terminal tại thư mục `ai_training/phobert`:

```powershell
# 1) Gỡ torch CPU và cài torch CUDA (xem lệnh chính xác cho máy bạn tại https://pytorch.org/get-started/locally/)
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) Cài các thư viện còn lại
pip install -r requirements.txt

# 3) Kiểm tra GPU (phải in True và tên card)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Nếu in `False`: driver NVIDIA quá cũ hoặc cài nhầm bản CPU — thử bản `cu126`/`cu124` trên trang PyTorch.
Lần chạy đầu cần Internet để tải mô hình (~540 MB) và tokenizer từ HuggingFace.

## 2. Cách chạy

Luôn chạy từ thư mục `ai_training/phobert`:

```powershell
# (a) Chạy thử nhanh ~1 phút: 64 mẫu, 1 epoch — chỉ để kiểm tra môi trường/GPU/đường dẫn (kết quả vô nghĩa)
python train_phobert.py --smoke-test

# (b) Train đầy đủ bằng config.yaml (ước tính vài phút đến ~15 phút trên RTX 4060, tối đa 10 epoch, có dừng sớm — chưa đo thực tế)
python train_phobert.py

# (c) Đổi tham số không cần sửa file
python train_phobert.py --set train.epochs=5 --set train.learning_rate=3e-5 --set seed=1
```

Kết quả nằm trong `outputs/phobert_base_v2_<ngày_giờ>/`. Mỗi lần chạy tạo một thư mục mới, không ghi đè.
Mở `report.md` trong thư mục đó để xem báo cáo tổng hợp (bảng + biểu đồ + nhận xét).

**Hết bộ nhớ GPU (CUDA out of memory)?** Giảm `train.batch_size` xuống 8 và tăng `train.grad_accum_steps` lên 4
(batch hiệu dụng vẫn là 32), hoặc giảm `model.max_length` xuống 128.

## 3. Quy trình bên trong

```
Tin nhắn → chuẩn hoá khoảng trắng → tách từ (pyvi) → BPE PhoBERT (max 256 token)
        → PhoBERT-base-v2 (12 lớp Transformer, 135M tham số) → Linear 768→2 → softmax → 0/1
```

- Vòng huấn luyện tự viết (không dùng `Trainer`) để ghi được loss/lr/grad-norm từng bước.
- AdamW, warmup + giảm tuyến tính, cắt gradient, mixed precision fp16, trọng số lớp `balanced`.
- Sau mỗi epoch đánh giá trên **validation**; giữ checkpoint có `macro_f1` cao nhất; dừng sớm sau 3 epoch không cải thiện.
- Sau cùng nạp lại checkpoint tốt nhất và đánh giá train / val / **test** (test không dùng để chọn gì cả).
- Ngưỡng quyết định mặc định 0.5; ngưỡng tối ưu F1 chọn trên val được báo cáo riêng.

## 4. Tham số (`config.yaml`)

| Nhóm | Tham số | Giá trị | Ý nghĩa |
|---|---|---|---|
| model | `name` | vinai/phobert-base-v2 | Mô hình tiền huấn luyện |
| model | `max_length` | 256 | Số token tối đa/tin (PhoBERT tối đa 256) |
| model | `hidden_dropout_prob`, `attention_probs_dropout_prob` | 0.1 | Dropout encoder |
| model | `classifier_dropout` | null | Dropout đầu phân loại (null = dùng 0.1) |
| train | `epochs` | 10 | Tối đa; có dừng sớm |
| train | `batch_size` × `grad_accum_steps` | 16 × 2 | Batch hiệu dụng 32 |
| train | `learning_rate` | 2e-5 | Tốc độ học đỉnh |
| train | `weight_decay` | 0.01 | Chính quy hoá AdamW (không áp cho bias/LayerNorm) |
| train | `warmup_ratio` | 0.1 | 10% bước đầu tăng dần lr |
| train | `lr_scheduler` | linear | `linear` hoặc `cosine` |
| train | `max_grad_norm` | 1.0 | Ngưỡng cắt gradient |
| train | `class_weight` | balanced | Bù lệch lớp (lớp 0 chiếm ~58%) |
| train | `label_smoothing` | 0.0 | Làm mềm nhãn (0 = tắt) |
| train | `precision` | fp16 | `fp16` / `bf16` / `fp32` |
| early_stopping | `monitor`, `patience`, `min_delta` | macro_f1, 3, 0.0005 | Tiêu chí chọn mô hình và dừng sớm |
| text | `word_segment` | true | Tách từ bằng pyvi (PhoBERT bắt buộc) |
| seed | `seed` | 42 | Tái lập kết quả |

Mọi giá trị đã dùng được lưu lại ở `outputs/<run>/config_used.yaml`.

## 5. Các file kết quả trong `outputs/<run>/`

| File / thư mục | Dùng để viết tài liệu phần… |
|---|---|
| `report.md` | Báo cáo tổng hợp tự sinh (bảng + biểu đồ + nhận xét) |
| `nhan_xet_tu_dong.md` | Các nhận xét tự động (overfitting, precision/recall, nhóm có dấu/không dấu…) |
| `config_used.yaml`, `environment.json` | Tham số huấn luyện, phiên bản thư viện, GPU |
| `model_params.json`, `model_architecture.txt` | Số tham số theo khối, cấu hình và kiến trúc mô hình |
| `data/data_summary.csv`, `data/segmentation_samples.xlsx` | Thống kê dữ liệu; ví dụ gốc → tách từ → token BPE |
| `history.csv` | Mỗi epoch: train/val loss, accuracy, precision, recall, F1, AUC, thời gian, bộ nhớ GPU |
| `step_log.csv` | Mỗi N bước: loss, learning rate, grad norm |
| `metrics.json`, `metrics_summary.csv` | Chỉ số đầy đủ train/val/test (+ test ở ngưỡng tối ưu) |
| `classification_report_test.txt` | Báo cáo precision/recall/F1 từng lớp trên test |
| `predictions_{train,val,test}.csv` | Dự đoán + xác suất từng tin |
| `errors_val.xlsx`, `errors_test.xlsx` | Các tin dự đoán sai (FP/FN) xếp theo độ tự tin sai |
| `group_metrics_*.csv`, `threshold_val.csv` | Chỉ số theo nhóm (có dấu/không dấu, độ dài) và theo ngưỡng |
| `timing.json` | Thời gian từng giai đoạn |
| `best_model/` | Mô hình + tokenizer tốt nhất (nạp bằng `predict_phobert.py` hoặc `from_pretrained`) |
| `plots/` | Toàn bộ biểu đồ (bên dưới) |

### Biểu đồ trong `plots/`

| File | Nội dung |
|---|---|
| `00_class_distribution`, `00_length_distribution`, `00_token_length` | Phân bố nhãn/tỷ lệ không dấu, độ dài ký tự, độ dài token |
| `00_pipeline` | Sơ đồ khối pipeline |
| `01_loss_curve`, `02_val_metrics_curve` | Loss và các chỉ số val theo epoch |
| `03_lr_loss_gradnorm_steps` | Lịch learning rate, loss, grad norm theo bước |
| `10_roc_pr` | ROC và Precision-Recall (train/val/test) |
| `11/12_confusion_*` | Ma trận nhầm lẫn val/test |
| `13_score_hist_*` | Phân bố xác suất lừa đảo theo nhãn thật |
| `14_threshold_val` | Precision/Recall/F1 theo ngưỡng |
| `15_group_test` | Chỉ số theo nhóm có dấu/không dấu, độ dài |

## 6. Dùng mô hình đã train để dự đoán

```powershell
python predict_phobert.py --model outputs/<run>/best_model --text "Tài khoản của bạn bị khóa, bấm vào link để xác thực"
python predict_phobert.py --model outputs/<run>/best_model --file tin_nhan.xlsx --out ketqua.xlsx   # cột 'content'
```

## 7. Lưu ý

- Khoảng một nửa dữ liệu là tin **không dấu**; PhoBERT được pretrain trên tiếng Việt có dấu nên tách từ/token hoá tin không dấu kém hơn.
  `report.md` có bảng điểm riêng cho nhóm có dấu / không dấu để thấy chênh lệch.
- Dữ liệu có nhiều tin do máy sinh theo khuôn, nên điểm val/test có thể cao hơn thực tế — cần tập tin thật độc lập để kiểm tra cuối.
- `eval_utils.py` giống hệt file cùng tên trong `../IDF_SVM/` (tiện ích đánh giá/biểu đồ dùng chung); sửa thì sửa cả hai.
