# 🔍 Fingerprint Recognition System

> Hệ thống nhận dạng và xác thực vân tay tự động trích xuất đặc trưng thủ công và sử dụng **Siamese Neural Network**.

---

## 📌 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Dataset](#-dataset)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cài đặt & Chạy dự án](#-cài-đặt--chạy-dự-án)
- [Pipeline xử lý](#-pipeline-xử-lý)
- [Kết quả](#-kết-quả)
- [Tác giả](#-tác-giả)

---

## 🎯 Giới thiệu

Dự án **Fingerprint Recognition System** xây dựng hệ thống **xác thực vân tay** (fingerprint verification) dựa trên học sâu. Hệ thống nhận đầu vào là hai ảnh vân tay và xác định xem chúng có thuộc về cùng một người hay không.

Ứng dụng thực tiễn:
- Xác thực danh tính trong hệ thống bảo mật
- Kiểm soát ra vào (access control)
- Xác thực sinh trắc học (biometric authentication)

---

## 🧠 Kiến trúc mô hình

Dự án sử dụng **Siamese Neural Network** — một kiến trúc học sâu đặc biệt phù hợp cho bài toán **so khớp cặp ảnh** (pairwise matching):

```
Ảnh vân tay A ──► [CNN Encoder] ──► Vector đặc trưng A ──┐
                                                           ├──► [Distance Layer] ──► Xác suất khớp
Ảnh vân tay B ──► [CNN Encoder] ──► Vector đặc trưng B ──┘
                  (shared weights)
```

- Hai nhánh CNN **chia sẻ trọng số** để học cùng một không gian embedding
- Hàm khoảng cách (Euclidean/Cosine) đo độ tương đồng giữa hai vector đặc trưng
- Đầu ra: xác suất hai vân tay thuộc cùng một người (`same`) hay khác người (`different`)

---

## 📁 Cấu trúc dự án

```
Fingerprint-Recognition-System/
│
├── Datasets/                         # Ảnh vân tay gốc dùng để huấn luyện & kiểm tra
│
├── Dataset_report.ipynb              # Phân tích và thống kê bộ dữ liệu
├── extract_features_manual.ipynb     # Trích xuất đặc trưng vân tay thủ công
├── Training_model.ipynb              # Huấn luyện Siamese Neural Network
├── Evaluation_model.ipynb            # Đánh giá hiệu năng mô hình
│
├── pairs_train.csv                   # Danh sách cặp ảnh dùng để huấn luyện (label: same/different)
├── pairs_test.csv                    # Danh sách cặp ảnh dùng để kiểm tra
│
├── siamese_model_finally.h5          # Model đã huấn luyện (Keras/TensorFlow)
├── siamese_model_history.csv         # Lịch sử loss/accuracy trong quá trình huấn luyện
│
├── Presentation.pptx                 # Slide thuyết trình dự án
└── Report.pdf                        # Báo cáo chi tiết dự án
```

---

## 📊 Dataset

Bộ dữ liệu gồm các ảnh vân tay được tổ chức trong thư mục `Datasets/`. Dữ liệu được chia thành cặp ảnh (image pairs) theo định dạng:

| File | Mô tả |
|---|---|
| `pairs_train.csv` | Cặp ảnh + nhãn dùng để huấn luyện |
| `pairs_test.csv` | Cặp ảnh + nhãn dùng để kiểm tra |

Mỗi dòng trong file CSV gồm: `image_path_1`, `image_path_2`, `label` (1 = cùng người, 0 = khác người).

---

## 🛠 Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.x |
| Môi trường thực thi | Jupyter Notebook |
| Deep Learning | TensorFlow / Keras |
| Xử lý ảnh | OpenCV, NumPy |
| Trích xuất đặc trưng | Thủ công (Minutiae, Ridge patterns) + CNN |
| Trực quan hóa | Matplotlib, Seaborn |
| Lưu trữ mô hình | HDF5 (`.h5`) |

---

## ⚙️ Cài đặt & Chạy dự án

### 1. Clone repository

```bash
git clone https://github.com/Anhtu12022004/Fingerprint-Recognition-System.git
cd Fingerprint-Recognition-System
```

### 2. Cài đặt thư viện

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib seaborn scikit-learn jupyter
```


### 3. Chạy các notebook theo thứ tự

```bash
jupyter notebook
```

Mở và chạy theo thứ tự sau:

| Bước | Notebook | Mô tả |
|---|---|---|
| 1️⃣ | `Dataset_report.ipynb` | Khám phá và phân tích dữ liệu |
| 2️⃣ | `extract_features_manual.ipynb` | Trích xuất đặc trưng vân tay |
| 3️⃣ | `Training_model.ipynb` | Huấn luyện mô hình Siamese |
| 4️⃣ | `Evaluation_model.ipynb` | Đánh giá và kiểm tra kết quả |

> **Lưu ý:** Nếu muốn dùng thẳng mô hình đã huấn luyện, load file `siamese_model_finally.h5` trong notebook đánh giá.

### 4. Load mô hình đã huấn luyện

```python
from tensorflow.keras.models import load_model

model = load_model('siamese_model_finally.h5')
```

---

## 🔄 Pipeline xử lý

```
Thu thập ảnh vân tay
        ↓
Tiền xử lý ảnh (Grayscale, Normalization, Enhancement)
        ↓
Trích xuất đặc trưng (Minutiae detection / CNN features)
        ↓
Tạo cặp ảnh (Positive pairs & Negative pairs)
        ↓
Huấn luyện Siamese Neural Network
        ↓
Đánh giá mô hình (Accuracy, ROC-AUC, EER)
        ↓
Xác thực vân tay đầu vào
```

---

## 📈 Kết quả

<!-- Cập nhật kết quả thực tế từ Evaluation_model.ipynb -->

| Chỉ số | Giá trị |
|---|---|
| Accuracy | 96% |
| AUC-ROC | 98% |

> Chi tiết phân tích kết quả, đồ thị loss/accuracy và confusion matrix được trình bày trong `Evaluation_model.ipynb` và `Report.pdf`.

---

## 📄 Tài liệu

- 📊 **Slide thuyết trình:** [`Presentation.pptx`](./Presentation.pptx)
- 📝 **Báo cáo đầy đủ:** [`Report.pdf`](./Report.pdf)

---

## 👥 Tác giả

| Tên | GitHub |
|---|---|
| Nguyễn Lê Anh Tú | [@Anhtu12022004](https://github.com/Anhtu12022004) |
| Lê Hồng Tiến | [@LeHongTien](https://github.com/LeHongTien) |

---

## 📝 Giấy phép

Dự án được thực hiện cho mục đích **học thuật và nghiên cứu**.
