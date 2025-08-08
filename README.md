# Multi-Labels-URLs-ML-DL-Detection-Models

## Tổng quan dự án

Ứng dụng công nghệ học sâu cho phân loại đa nhãn các URL độc hại - một hệ thống thông minh để phát hiện và phân loại các URL độc hại sử dụng các mô hình học máy và học sâu tiên tiến.

## Đặc điểm chính

### 🎯 Phân loại đa nhãn

- **4 nhãn chính**: Benign, Defacement, Malware, Phishing
- **Hỗ trợ đa nhãn**: Một URL có thể thuộc nhiều loại độc hại cùng lúc
- **Ngưỡng tùy chỉnh**: Có thể điều chỉnh ngưỡng phân loại cho từng nhãn

### 🤖 Mô hình học sâu và học máy

1. **CNN (Convolutional Neural Network)**

   - Mô hình học sâu cho đặc trưng số và phi số
   - Kiến trúc tối ưu cho xử lý dữ liệu URL

2. **XGBoost**

   - Gradient boosting hiệu suất cao
   - Xử lý cả đặc trưng số và phi số

3. **Random Forest**
   - Ensemble learning robust
   - Phù hợp cho dữ liệu không cân bằng

### 🔍 Trích xuất đặc trưng

- **Đặc trưng số**: 71+ đặc trưng được trích xuất từ URL
- **Đặc trưng phi số**: TF-IDF vectorization cho text features
- **Đặc trưng nâng cao**:
  - WHOIS information
  - DNS records
  - Page rank
  - Domain age
  - Security indicators

### 📊 Đánh giá mô hình

- **Macro F1-Score**: Đánh giá hiệu suất tổng thể
- **PR-AUC**: Precision-Recall Area Under Curve
- **Confusion Matrix**: Ma trận nhầm lẫn đa nhãn
- **ROC Curve**: Receiver Operating Characteristic

## Cài đặt và chạy

### Yêu cầu hệ thống

```bash
Python 3.8+
TensorFlow 2.x
scikit-learn
numpy
pandas
flask
firebase-admin
```

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Chạy ứng dụng

```bash
python run.py
```

Ứng dụng sẽ chạy tại: `http://localhost:5000`

## Cấu trúc dự án

```
Multi-Labels-URLs-ML-DL-Detection-Models/
├── app/                    # Flask application
│   ├── routes.py          # API endpoints
│   └── __init__.py        # App factory
├── models/                # Trained models
│   └── dataset_1/         # Model files
├── scripts/               # Core algorithms
│   ├── url_features_extractor.py    # Feature extraction
│   └── url_multi_labels_predictor.py # Prediction engine
├── templates/             # HTML templates
├── static/                # CSS, JS, images
├── configs/               # Configuration files
└── utils/                 # Utility functions
```

## API Endpoints

### 1. Dự đoán đơn mô hình

```http
POST /api/predict-url
Content-Type: application/json

{
    "url": "https://example.com",
    "model": "cnn",
    "threshold": 0.5,
    "numerical": true
}
```

### 2. Dự đoán đa mô hình

```http
POST /api/predict-multi-model
Content-Type: application/json

{
    "url": "https://example.com",
    "threshold": 0.5,
    "numerical": true
}
```

### 3. Lịch sử dự đoán

```http
GET /api/history
```

## Tính năng chính

### 🎨 Giao diện người dùng

- **Dashboard hiện đại**: Giao diện responsive với Bootstrap 5
- **Biểu đồ tương tác**: Chart.js cho visualization
- **Real-time prediction**: Dự đoán nhanh với feedback tức thì
- **History tracking**: Theo dõi lịch sử dự đoán

### 🔄 Xử lý dữ liệu

- **URL normalization**: Chuẩn hóa URL input
- **Feature engineering**: 71+ đặc trưng được trích xuất
- **Data preprocessing**: Scaling, encoding, vectorization
- **Multi-label support**: Hỗ trợ phân loại đa nhãn

### 📈 Phân tích và báo cáo

- **Model comparison**: So sánh hiệu suất các mô hình
- **Performance metrics**: F1-score, PR-AUC, confusion matrix
- **Visualization**: Biểu đồ ROC, PR curves
- **Statistical analysis**: Phân tích thống kê chi tiết

## Kết quả thực nghiệm

### Bộ dữ liệu

- **Nguồn**: Canadian Institute for Cybersecurity (ISCX-URL-2016)
- **Dataset file**: `balanced_dataset_1.csv`
- **Kích thước**: 385,260 URL
- **Phân bố nhãn**:
  - Benign: 27.45%
  - Defacement: 24.59%
  - Malware: 24.35%
  - Phishing: 23.60%

### Hiệu suất mô hình

- **CNN**: Accuracy ~85%, Macro F1-Score ~0.82
- **XGBoost**: Accuracy ~87%, Macro F1-Score ~0.85
- **Random Forest**: Accuracy ~84%, Macro F1-Score ~0.83

### Đánh giá đa nhãn

- **Macro F1-Score**: 0.83 (trung bình)
- **PR-AUC**: 0.81 (trung bình)
- **Hamming Loss**: 0.17 (thấp)

## Đóng góp

1. Fork dự án
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## License

Dự án này được phát hành dưới MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## Tác giả

- **Sinh viên**: Nhóm16
- **Mentor**: Đàm Minh Lịnh

## Tài liệu tham khảo

1. Canadian Institute for Cybersecurity - ISCX-URL-2016 Dataset
2. Multi-label Classification with Deep Learning
3. URL-based Phishing Detection using Machine Learning
4. Feature Engineering for URL Classification
