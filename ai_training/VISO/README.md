# ViSoBERT — Phân loại tin nhắn lừa đảo

Fine-tune `uitnlp/visobert` (XLM-R tiền huấn luyện trên văn bản mạng xã hội tiếng Việt) để phân loại tin nhắn: **0 = bình thường, 1 = lừa đảo**.
Thư mục này tự chứa mọi thứ: code, tham số, và (sau khi train) toàn bộ kết quả/biểu đồ/báo cáo.

## 1. Cài đặt (làm một lần)

Nếu bạn đã cài torch CUDA cho thư mục `phobert/` thì **bỏ qua bước 1)**; kiểm tra ở bước 3) là đủ.
Mở terminal tại thư mục `ai_training/VISO`:

```powershell
# 1) Gỡ torch CPU và cài torch CUDA (xem lệnh chính xác cho máy bạn tại https://pytorch.org/get-started/locally/)
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) Cài các thư viện còn lại (có thêm `sentencepiece` — BẮT BUỘC để nạp tokenizer của ViSoBERT; không cần pyvi)
pip install -r requirements.txt

# 3) Kiểm tra GPU (phải in True và tên card)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Nếu in `False`: driver NVIDIA quá cũ hoặc cài nhầm bản CPU — thử bản `cu126`/`cu124` trên trang PyTorch.
Lần chạy đầu cần Internet để tải mô hình (~400 MB) và tokenizer từ HuggingFace.

## 2. Cách chạy

Luôn chạy từ thư mục `ai_training/VISO`:

```powershell
# (a) Chạy thử nhanh ~1 phút: 64 mẫu, 1 epoch — chỉ để kiểm tra môi trường/GPU/đường dẫn (kết quả vô nghĩa)
python train_viso.py --smoke-test

# (b) Train đầy đủ bằng config.yaml (ước tính vài phút đến ~15 phút trên RTX 4060, tối đa 10 epoch, có dừng sớm — chưa đo thực tế)
python train_viso.py

# (c) Đổi tham số không cần sửa file
python train_viso.py --set train.epochs=5 --set train.learning_rate=3e-5 --set seed=1
```

Kết quả nằm trong `outputs/visobert_<ngày_giờ>/`. Mỗi lần chạy tạo một thư mục mới, không ghi đè.
Mở `report.md` trong thư mục đó để xem báo cáo tổng hợp (bảng + biểu đồ + nhận xét).

**Hết bộ nhớ GPU (CUDA out of memory)?** Giảm `train.batch_size` xuống 8 và tăng `train.grad_accum_steps` lên 4
(batch hiệu dụng vẫn là 32), hoặc giảm `model.max_length` xuống 128.

## 3. Quy trình bên trong

```
Tin nhắn → chuẩn hoá khoảng trắng → SentencePiece ViSoBERT (max 256 token, KHÔNG tách từ)
        → ViSoBERT (XLM-R, 12 lớp Transformer, ~97M tham số) → Linear 768→2 → softmax → 0/1
```

- Vòng huấn luyện tự viết (không dùng `Trainer`) để ghi được loss/lr/grad-norm từng bước.
- AdamW, warmup + giảm tuyến tính, cắt gradient, mixed precision fp16, trọng số lớp `balanced`.
- Sau mỗi epoch đánh giá trên **validation**; giữ checkpoint có `macro_f1` cao nhất; dừng sớm sau 3 epoch không cải thiện.
- Sau cùng nạp lại checkpoint tốt nhất và đánh giá train / val / **test** (test không dùng để chọn gì cả).
- Ngưỡng quyết định mặc định 0.5; ngưỡng tối ưu F1 chọn trên val được báo cáo riêng.

## 4. Tham số (`config.yaml`)

| Nhóm | Tham số | Giá trị | Ý nghĩa |
|---|---|---|---|
| model | `name` | uitnlp/visobert | Mô hình tiền huấn luyện |
| model | `max_length` | 256 | Số token tối đa/tin (ViSoBERT cho phép tới 510) |
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
| text | `normalize_whitespace`, `lowercase` | true, false | Chuẩn hoá khoảng trắng; giữ nguyên hoa/thường |
| seed | `seed` | 42 | Tái lập kết quả |

Mọi giá trị đã dùng được lưu lại ở `outputs/<run>/config_used.yaml`.

## 5. Các file kết quả trong `outputs/<run>/`

| File / thư mục | Dùng để viết tài liệu phần… |
|---|---|
| `report.md` | Báo cáo tổng hợp tự sinh (bảng + biểu đồ + nhận xét) |
| `nhan_xet_tu_dong.md` | Các nhận xét tự động (overfitting, precision/recall, nhóm có dấu/không dấu…) |
| `config_used.yaml`, `environment.json` | Tham số huấn luyện, phiên bản thư viện, GPU |
| `model_params.json`, `model_architecture.txt` | Số tham số theo khối, cấu hình và kiến trúc mô hình |
| `data/data_summary.csv`, `data/tokenization_samples.xlsx` | Thống kê dữ liệu; ví dụ gốc → token SentencePiece |
| `history.csv` | Mỗi epoch: train/val loss, accuracy, precision, recall, F1, AUC, thời gian, bộ nhớ GPU |
| `step_log.csv` | Mỗi N bước: loss, learning rate, grad norm |
| `metrics.json`, `metrics_summary.csv` | Chỉ số đầy đủ train/val/test (+ test ở ngưỡng tối ưu) |
| `classification_report_test.txt` | Báo cáo precision/recall/F1 từng lớp trên test |
| `predictions_{train,val,test}.csv` | Dự đoán + xác suất từng tin |
| `errors_val.xlsx`, `errors_test.xlsx` | Các tin dự đoán sai (FP/FN) xếp theo độ tự tin sai |
| `group_metrics_*.csv`, `threshold_val.csv` | Chỉ số theo nhóm (có dấu/không dấu, độ dài) và theo ngưỡng |
| `timing.json` | Thời gian từng giai đoạn |
| `best_model/` | Mô hình + tokenizer tốt nhất (nạp bằng `predict_viso.py` hoặc `from_pretrained`) |
| `plots/` | Toàn bộ biểu đồ (bên dưới) |

### Biểu đồ trong `plots/`

| File | Nội dung |
|---|---|
| `00_class_distribution`, `00_length_distribution`, `00_token_length` | Phân bố nhãn/tỷ lệ không dấu, độ dài ký tự, độ dài token (tỉ lệ `<unk>` ghi trong `report.md`) |
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
python predict_viso.py --model outputs/<run>/best_model --text "Tài khoản của bạn bị khóa, bấm vào link để xác thực"
python predict_viso.py --model outputs/<run>/best_model --file tin_nhan.xlsx --out ketqua.xlsx   # cột 'content'
```

## 7. Lưu ý

- ViSoBERT không cần tách từ nên bỏ được `pyvi`, và được pretrain trên văn bản mạng xã hội (teencode, viết tắt, emoji) → kỳ vọng hợp với tin nhắn đời thường hơn PhoBERT.
  Tuy vậy mô hình chủ yếu học tiếng Việt **có dấu**; tin không dấu bị tách thành nhiều mảnh nhỏ hơn. `report.md` có bảng điểm riêng cho nhóm có dấu / không dấu và tỉ lệ token `<unk>`.
- Kho `uitnlp/visobert` chỉ có file `pytorch_model.bin` (không phải safetensors) nên lần nạp đầu có thể cảnh báo; không ảnh hưởng kết quả.
- Dữ liệu có nhiều tin do máy sinh theo khuôn, nên điểm val/test có thể cao hơn thực tế — cần tập tin thật độc lập để kiểm tra cuối.
- So sánh công bằng với PhoBERT và TF-IDF+SVM: cả ba dùng cùng `Data_Scam_train/vali/test.xlsx`, cùng seed 42 và cùng tiêu chí chọn (macro-F1 trên val).
- `eval_utils.py` giống hệt file cùng tên trong `../phobert/` và `../IDF_SVM/`; sửa thì sửa cả ba.
