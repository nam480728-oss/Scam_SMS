# TF-IDF + Linear SVM — Phân loại tin nhắn lừa đảo

Mô hình cơ sở (baseline) nhanh, giải thích được: TF-IDF (từ + ký tự) → Linear SVM → hiệu chỉnh xác suất (Platt).
Nhãn: **0 = bình thường, 1 = lừa đảo**. Thư mục này tự chứa code, tham số và toàn bộ kết quả sau khi train.

## 1. Cài đặt (làm một lần)

Không cần GPU. Mở terminal tại `ai_training/IDF_SVM`:

```powershell
pip install -r requirements.txt
```

## 2. Cách chạy

Luôn chạy từ thư mục `ai_training/IDF_SVM`:

```powershell
# (a) Chạy thử vài giây để kiểm tra môi trường (dữ liệu nhỏ, grid nhỏ — kết quả không có ý nghĩa)
python train_tfidf_svm.py --smoke-test

# (b) Train đầy đủ: thử 96 tổ hợp tham số, mỗi tổ hợp chạy 5-fold CV + validation (ước tính vài phút, chưa đo thực tế)
python train_tfidf_svm.py

# (c) Bỏ tìm siêu tham số, dùng đúng tham số trong config.yaml (chạy rất nhanh)
python train_tfidf_svm.py --set search.enabled=false --set model.C=1

# (d) Đổi tham số tuỳ ý
python train_tfidf_svm.py --set search.selection=cv --set features.char.ngram_range=[3,5]
```

Kết quả nằm trong `outputs/tfidf_linear_svm_<ngày_giờ>/` (mỗi lần chạy một thư mục mới). Mở `report.md` để xem báo cáo tổng hợp.

## 3. Quy trình bên trong

```
Tin nhắn → tiền xử lý → TF-IDF từ (1-2 gram) ∪ TF-IDF ký tự (2-5 gram) → LinearSVC → decision function
        → Platt scaling (hiệu chỉnh trên val) → xác suất lừa đảo
```

1. **Tiền xử lý** (`preprocess.py`): gộp khoảng trắng, hạ chữ thường, **bỏ dấu** (để tin có dấu/không dấu dùng chung đặc trưng),
   thay link / số điện thoại / số tài khoản / số tiền bằng token `urltoken`, `sdttoken`, `sotktoken`, `tientoken`.
2. **Tìm siêu tham số** (`search.grid`): thử mọi tổ hợp `C × class_weight × bỏ dấu × n-gram từ × n-gram ký tự`.
   Mỗi tổ hợp huấn luyện trên **train**, đo macro-F1 trên **validation** và bằng 5-fold CV trên train
   (CV chia theo *nhóm khuôn mẫu* — các tin cùng 4 từ đầu nằm cùng một fold — để không bị thổi phồng).
   Chọn tổ hợp có val macro-F1 cao nhất (hoà thì xét CV, rồi chọn C nhỏ hơn). Đổi sang chọn theo CV bằng `search.selection=cv`.
3. **Huấn luyện mô hình cuối** với bộ tham số được chọn trên toàn bộ train.
4. **Platt scaling** trên validation để có xác suất 0–1 (dùng cho ROC/PR, ngưỡng, và ứng dụng). Nhãn dự đoán vẫn là `decision_function > 0`.
5. **Đánh giá cuối** train / val / **test** (test không dùng để chọn gì).
6. **Phân tích bổ sung**: trọng số đặc trưng, learning curve, so sánh chỉ-từ / chỉ-ký-tự / cả hai, phân nhóm có dấu/không dấu.

> Ghi chú: TF-IDF (từ vựng, IDF) được fit trên toàn bộ train rồi mới chia fold cho CV — đây là rò rỉ rất nhẹ vì không dùng nhãn;
> điểm CV vì thế hơi lạc quan nhưng dùng nhất quán cho mọi tổ hợp. Điểm val/test không bị ảnh hưởng.

## 4. Tham số (`config.yaml`)

| Nhóm | Tham số | Mặc định | Ý nghĩa |
|---|---|---|---|
| preprocess | `lowercase`, `strip_accents` | true, true | Hạ chữ thường, bỏ dấu |
| preprocess | `mask_url/phone/account/money` | true | Thay thành token giữ lại dấu hiệu "có link / có số tiền…" |
| features | `word.ngram_range` | [1, 2] | N-gram từ |
| features | `char.analyzer`, `char.ngram_range` | char_wb, [2, 5] | N-gram ký tự (bắt biến thể "ck", "chuyen khoan", lỗi chính tả) |
| features | `min_df`, `max_df` | 2, 0.95 (từ) | Lọc n-gram quá hiếm / quá phổ biến |
| features | `sublinear_tf`, `norm` | true, l2 | tf → 1+log(tf); chuẩn hoá L2 |
| model | `C` | 1.0 | Độ phạt sai (lớn = ít chính quy hoá). Bị ghi đè nếu `search.enabled` |
| model | `class_weight` | balanced | Bù lệch lớp (lớp 0 ~58%) |
| model | `loss`, `max_iter`, `tol` | squared_hinge, 20000, 1e-4 | Tham số tối ưu của LinearSVC |
| search | `grid.C` | 0.03 … 10 (6 giá trị) | Lưới C |
| search | `grid.class_weight` | [null, balanced] | |
| search | `grid.strip_accents` | [true, false] | Có bỏ dấu hay không |
| search | `grid.word_ngram`, `grid.char_ngram` | 2 × 2 giá trị | Lưới n-gram |
| search | `cv_folds`, `selection` | 5, val | Số fold CV; tiêu chí chọn (`val` hoặc `cv`) |
| calibration | `platt_on_val` | true | Hiệu chỉnh xác suất |
| analysis | `learning_curve_fractions`, `top_features`, `ablation` | … | Phân tích bổ sung |

Mọi giá trị đã dùng được lưu ở `outputs/<run>/config_used.yaml`; tham số mô hình cuối ở `model_params.json`.

## 5. Các file kết quả trong `outputs/<run>/`

| File / thư mục | Dùng để viết tài liệu phần… |
|---|---|
| `report.md`, `nhan_xet_tu_dong.md` | Báo cáo tổng hợp tự sinh và các nhận xét tự động |
| `config_used.yaml`, `environment.json`, `model_params.json` | Tham số, phiên bản thư viện, tham số mô hình cuối (C, số đặc trưng, hệ số Platt…) |
| `search_results.csv` | Kết quả **mọi** tổ hợp tham số (train/CV/val macro-F1, số đặc trưng, đã hội tụ chưa) |
| `top_features.csv` | 500 đặc trưng có |trọng số| lớn nhất (giải thích mô hình) |
| `learning_curve.csv`, `ablation.csv` | Learning curve; so sánh loại đặc trưng |
| `data/data_summary.csv`, `data/preprocess_samples.xlsx` | Thống kê dữ liệu; ví dụ trước/sau tiền xử lý |
| `metrics.json`, `metrics_summary.csv`, `classification_report_test.txt` | Chỉ số train/val/test |
| `predictions_*.csv`, `errors_val.xlsx`, `errors_test.xlsx` | Dự đoán từng tin; các tin sai (FP/FN) |
| `group_metrics_*.csv`, `threshold_val.csv` | Theo nhóm (có dấu/không dấu, độ dài), theo ngưỡng |
| `timing.json` | Thời gian từng giai đoạn |
| `model/tfidf_svm.joblib` | Pipeline + SVM + Platt (dùng với `predict_svm.py`) |
| `plots/` | Biểu đồ (bên dưới) |

### Biểu đồ trong `plots/`

| File | Nội dung |
|---|---|
| `00_*` | Phân bố nhãn, tỷ lệ không dấu, độ dài; sơ đồ pipeline |
| `10_roc_pr`, `11/12_confusion_*`, `13_score_hist_*`, `14_threshold_val`, `15_group_test` | ROC/PR, ma trận nhầm lẫn, phân bố xác suất, ngưỡng, theo nhóm |
| `20_search_top15` | 15 tổ hợp tham số tốt nhất (val và CV) |
| `21_C_curve` | Ảnh hưởng của C lên train / CV / val |
| `22_param_effect` | Ảnh hưởng bỏ dấu và class_weight trên toàn lưới |
| `23_top_features_*` | Đặc trưng đẩy về lừa đảo / bình thường (tất cả, và chỉ từ) |
| `24_learning_curve` | Điểm theo số mẫu train |
| `25_ablation` | Chỉ từ / chỉ ký tự / cả hai |

## 6. Dùng mô hình đã train để dự đoán

```powershell
python predict_svm.py --model outputs/<run>/model/tfidf_svm.joblib --text "Tài khoản bị khóa, bấm link để xác thực"
python predict_svm.py --model outputs/<run>/model/tfidf_svm.joblib --file tin_nhan.xlsx --out ketqua.xlsx
```

## 7. Lưu ý

- File `.joblib` phụ thuộc `preprocess.py` (cùng thư mục) và đúng phiên bản scikit-learn đã dùng để train (ghi ở `environment.json`).
- Dữ liệu có nhiều tin do máy sinh theo khuôn, nên điểm val/test có thể cao hơn thực tế — cần tập tin thật độc lập để kiểm tra cuối.
- `eval_utils.py` giống hệt file cùng tên trong `../phobert/`; sửa thì sửa cả hai.
