import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from script.url_features_extractor import URL_EXTRACTOR

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "feature.csv")

# Load dataset
# try:
#     df = pd.read_csv(DATASET_PATH)
#     print(f"Shape of raw DataFrame: {df.shape}")
#     print("Read dataset successfully")
# except Exception as e:
#     print(f"Error reading dataset: {e}")
#     print("Current directory contents:", os.listdir(BASE_DIR))
#     exit()

# df = df.drop(columns=['Unnamed: 0'])

def pre_processing(df):
    # Chuyển TRUE/FALSE -> 0/1 
    boolean_columns = ['hasHttp', 'hasHttps', 'urlIsLive']
    for col in boolean_columns:
        df[col] = df[col].apply(lambda x: 1.0 if str(x).strip().lower() in ['true', 't', '1'] else 0.0)

    # MinMax Scaler các cột giá trị số còn lại trừ boolean_columns
    scaler = MinMaxScaler()
    numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if col not in boolean_columns]
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

# Tách đặc trưng và nhãn
# X_test = df.drop(columns=['File'], errors='ignore')  # 21 cột đặc trưng
# y_test = df['File'] if 'File' in df.columns else None

# Thêm chiều cho X_test để phù hợp với mô hình (n_samples, 21, 1)
# X_test = np.expand_dims(X_test, axis=-1)

# Chọn một số mẫu ngẫu nhiên để kiểm thử
# num_samples = 5
# sample_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
# X_samples = X_test[sample_indices]
# print(f"Shape of X_samples: {X_samples.shape}")

# Dự đoán
# predictions = cnn_model1.predict(X_samples)
# predicted_labels = (predictions > 0.5).astype(int)

# Hiển thị kết quả
# label_names = ['benign', 'Defacement', 'malware', 'phishing', 'spam']
# for i, (pred, prob) in enumerate(zip(predicted_labels, predictions)):
#     print(f"\nSample {i + 1} (index: {sample_indices[i]}):")
#     for label, p, pl in zip(label_names, prob, pred):
#         print(f"{label}: Probability = {p:.4f}, Predicted = {bool(pl)}")
#     if y_test is not None:
#         true_label = y_test.iloc[sample_indices[i]]
#         print(f"True label: {true_label}")

if __name__ == '__main__':
    # Đối số dòng lệnh
    parser = argparse.ArgumentParser(description='URL Multi-Labels Detection')
    parser.add_argument('--urls', nargs='+', help='One or more URLs to check')
    args = parser.parse_args()

    # Load model 
    try:
        cnn_model1 = load_model(os.path.join(BASE_DIR, "CNN_MODEL_WITH_MULTI_LABELS_21_FEATURES.keras"))
        print("Load model successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Current directory contents:", os.listdir(BASE_DIR))
        exit()

    temp = []
    # Trích xuất đặc trưng từ URLs
    for url in args.urls:
        data = URL_EXTRACTOR(url).extract()
        temp.append(data)

    df = pd.DataFrame(temp)

    # Tiền xử lý
    df = pre_processing(df)
    
    # Tách đặc trưng và nhãn
    X_test = df.drop(columns=['File'], errors='ignore')  # 21 cột đặc trưng
    y_test = df['File'] if 'File' in df.columns else None

    # Thêm chiều cho X_test để phù hợp với mô hình (n_samples, 21, 1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Dự đoán
    predictions = cnn_model1.predict(X_test)
    predicted_labels = (predictions > 0.5).astype(int)

    # Hiển thị kết quả
    label_names = ['benign', 'Defacement', 'malware', 'phishing', 'spam']
    for i, (pred, prob) in enumerate(zip(predicted_labels, predictions)):
        print(f"\nSample {i + 1} (index: {i}):")
        for label, p, pl in zip(label_names, prob, pred):
            print(f"{label}: Probability = {p:.4f}, Predicted = {bool(pl)}")
        if y_test is not None:
            true_label = y_test.iloc[i]
            print(f"True label: {true_label}")
    