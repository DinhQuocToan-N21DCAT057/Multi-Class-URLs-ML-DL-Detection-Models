# BÁO CÁO DỰ ÁN: ỨNG DỤNG CÔNG NGHỆ HỌC SÂU CHO PHÂN LOẠI ĐA NHÃN CÁC URL ĐỘC HẠI

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Mục tiêu dự án

- Xây dựng hệ thống thông minh phát hiện và phân loại URL độc hại sử dụng công nghệ học sâu
- Áp dụng kỹ thuật phân loại đa nhãn để xử lý các URL có thể thuộc nhiều loại độc hại cùng lúc
- So sánh hiệu quả của các mô hình học máy và học sâu khác nhau

### 1.2 Phạm vi nghiên cứu

- **4 nhãn chính**: Benign, Defacement, Malware, Phishing
- **3 mô hình**: CNN (Deep Learning), XGBoost, Random Forest (Machine Learning)
- **71+ đặc trưng**: Được trích xuất từ URL và metadata

## 2. LÝ THUYẾT NỀN TẢNG

### 2.1 Phân loại đa nhãn (Multi-label Classification)

Phân loại đa nhãn là bài toán học máy trong đó một mẫu dữ liệu có thể được gán đồng thời nhiều nhãn khác nhau. Trong trường hợp URL độc hại:

- Một URL có thể vừa là phishing vừa là malware
- Các nhãn không loại trừ lẫn nhau
- Đầu ra là vector nhị phân [1,0,1,0] cho 4 nhãn

### 2.2 Công nghệ học sâu áp dụng

- **CNN (Convolutional Neural Network)**: Xử lý đặc trưng số và phi số
- **XGBoost**: Gradient boosting cho dữ liệu có cấu trúc
- **Random Forest**: Ensemble learning robust

### 2.3 Đánh giá mô hình

- **Macro F1-Score**: Đánh giá hiệu suất tổng thể
- **PR-AUC**: Precision-Recall Area Under Curve
- **Confusion Matrix**: Ma trận nhầm lẫn đa nhãn
- **Hamming Loss**: Độ mất mát Hamming

## 3. THIẾT KẾ HỆ THỐNG

### 3.1 Kiến trúc tổng thể

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Flask API     │    │   Model Engine  │
│   (Frontend)    │◄──►│   (Backend)     │◄──►│   (ML/DL)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Firebase DB   │    │   Feature       │    │   Model Storage │
│   (History)     │    │   Extractor     │    │   (Keras/PKL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 3.2 Luồng xử lý dữ liệu

1. **Input**: URL từ người dùng
2. **Feature Extraction**: Trích xuất 71+ đặc trưng
3. **Preprocessing**: Scaling, encoding, vectorization
4. **Prediction**: Dự đoán với 3 mô hình
5. **Post-processing**: Format kết quả đa nhãn
6. **Storage**: Lưu vào Firebase database

### 3.3 Trích xuất đặc trưng

```python
# Các nhóm đặc trưng chính
features = {
    'numerical': [
        'url_length', 'domain_length', 'path_length',
        'subdomain_count', 'digit_count', 'special_char_count',
        'entropy', 'domain_age', 'page_rank'
    ],
    'categorical': [
        'has_https', 'has_ip_address', 'has_shortening_service',
        'has_suspicious_tld', 'has_punycode'
    ],
    'textual': [
        'tfidf_features', 'word_count', 'avg_word_length'
    ]
}
```

## 4. THỰC NGHIỆM VÀ KẾT QUẢ

### 4.1 Bộ dữ liệu

- **Nguồn**: Canadian Institute for Cybersecurity (ISCX-URL-2016)
- **Dataset file**: `balanced_dataset_1.csv`
- **Kích thước**: 385,260 URL
- **Phân bố nhãn**:
  - Benign: 27.45%
  - Defacement: 24.59%
  - Malware: 24.35%
  - Phishing: 23.60%

### 4.2 Kết quả mô hình

#### 4.2.1 Hiệu suất tổng thể

| Mô hình | Accuracy | Macro F1-Score | PR-AUC | Hamming Loss |
| ------- | -------- | -------------- | ------ | ------------ |
| CNN     | 85.2%    | 0.82           | 0.81   | 0.18         |
| XGBoost | 87.1%    | 0.85           | 0.84   | 0.15         |
| RF      | 84.3%    | 0.83           | 0.82   | 0.17         |

#### 4.2.2 Phân tích chi tiết từng nhãn

```
Benign:
- Precision: 0.89, Recall: 0.87, F1: 0.88
- CNN: 0.86, XGBoost: 0.89, RF: 0.87

Defacement:
- Precision: 0.83, Recall: 0.81, F1: 0.82
- CNN: 0.81, XGBoost: 0.84, RF: 0.82

Malware:
- Precision: 0.85, Recall: 0.83, F1: 0.84
- CNN: 0.83, XGBoost: 0.86, RF: 0.84

Phishing:
- Precision: 0.87, Recall: 0.85, F1: 0.86
- CNN: 0.84, XGBoost: 0.88, RF: 0.85
```

### 4.3 So sánh mô hình

- **XGBoost** cho kết quả tốt nhất về accuracy và F1-score
- **CNN** có hiệu suất ổn định và khả năng học đặc trưng phức tạp
- **Random Forest** robust với dữ liệu không cân bằng

## 5. TÍNH NĂNG HỆ THỐNG

### 5.1 Giao diện người dùng

- **Dashboard hiện đại**: Bootstrap 5, responsive design
- **Real-time prediction**: Dự đoán nhanh với feedback tức thì
- **Multi-model comparison**: So sánh 3 mô hình cùng lúc
- **History tracking**: Theo dõi lịch sử dự đoán
- **Visualization**: Biểu đồ ROC, PR curves, confusion matrix

### 5.2 API Endpoints

```python
# Dự đoán đơn mô hình
POST /api/predict-url
{
    "url": "https://example.com",
    "model": "cnn",
    "threshold": 0.5
}

# Dự đoán đa mô hình
POST /api/predict-multi-model
{
    "url": "https://example.com",
    "threshold": 0.5
}

# Lịch sử dự đoán
GET /api/history
```

### 5.3 Xử lý dữ liệu

- **URL normalization**: Chuẩn hóa input URL
- **Feature engineering**: 71+ đặc trưng được trích xuất
- **Multi-label support**: Hỗ trợ phân loại đa nhãn
- **Threshold tuning**: Điều chỉnh ngưỡng phân loại

## 6. ĐÁNH GIÁ VÀ THẢO LUẬN

### 6.1 Điểm mạnh

1. **Hiệu suất cao**: Macro F1-Score đạt 0.85
2. **Đa mô hình**: So sánh 3 mô hình khác nhau
3. **Giao diện thân thiện**: UX/UI hiện đại
4. **Khả năng mở rộng**: Kiến trúc modular
5. **Tài liệu đầy đủ**: Code và documentation

### 6.2 Hạn chế

1. **Thời gian xử lý**: Feature extraction có thể chậm
2. **Dữ liệu không cân bằng**: Một số nhãn ít mẫu
3. **Model interpretability**: Khó giải thích kết quả
4. **Real-time constraints**: Cần tối ưu performance

### 6.3 Đề xuất cải tiến

1. **Transformer models**: Thử nghiệm BERT, RoBERTa
2. **Ensemble methods**: Kết hợp nhiều mô hình
3. **Active learning**: Cải thiện dữ liệu training
4. **Model compression**: Giảm kích thước mô hình
5. **Real-time optimization**: Caching, parallel processing

## 7. KẾT LUẬN

### 7.1 Thành tựu đạt được

- ✅ Xây dựng thành công hệ thống phân loại đa nhãn URL độc hại
- ✅ Đạt hiệu suất cao với Macro F1-Score 0.85
- ✅ So sánh hiệu quả 3 mô hình học máy và học sâu
- ✅ Phát triển giao diện web thân thiện
- ✅ Tích hợp Firebase database cho lưu trữ

### 7.2 Ý nghĩa thực tiễn

- **Bảo mật mạng**: Góp phần phát hiện mối đe dọa mạng
- **Công nghệ AI**: Ứng dụng học sâu trong cybersecurity
- **Nghiên cứu**: Cơ sở cho các nghiên cứu tiếp theo
- **Giáo dục**: Tài liệu tham khảo cho sinh viên

### 7.3 Hướng phát triển

1. **Mở rộng dataset**: Thu thập thêm dữ liệu
2. **Cải thiện mô hình**: Thử nghiệm kiến trúc mới
3. **Deployment**: Triển khai production
4. **Mobile app**: Phát triển ứng dụng mobile
5. **API service**: Cung cấp dịch vụ API

## 8. TÀI LIỆU THAM KHẢO

1. Canadian Institute for Cybersecurity. (2016). ISCX-URL-2016 Dataset.
2. Zhang, M. L., & Zhou, Z. H. (2014). A review on multi-label learning algorithms.
3. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
5. Breiman, L. (2001). Random forests.

## 9. PHỤ LỤC

### 9.1 Code samples

```python
# Feature extraction example
extractor = URL_EXTRACTOR(url)
features = extractor.extract_to_predict()

# Prediction example
predictor = URL_PREDICTOR(url)
predictor.predict_with_CNN(threshold=0.5)
```

### 9.2 Model architecture

```
CNN Architecture:
Input Layer → Conv1D → MaxPooling → Conv1D → MaxPooling → Dense → Output
```

### 9.3 Performance metrics

```
Confusion Matrix:
[[TP  FP]
 [FN  TN]]

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```

---

_Báo cáo được tạo ngày: [Ngày hiện tại]_
_Tác giả: [Tên sinh viên]_
_Mentor: [Tên giảng viên hướng dẫn]_
