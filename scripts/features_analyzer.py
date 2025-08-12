import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='URL Multi-Labels Dataset Processing')
    parser.add_argument('--dir', type=str, required=True, help='Directory name of dataset')
    parser.add_argument('--file', type=str, required=True, help='Processing files')
    parser.add_argument('--top', type=int, required=True, help='Top number of important features')
    args = parser.parse_args()

    dir_path = os.path.join(BASE_DIR, args.dir)
    if not os.path.isdir(dir_path):
        print(f"Directory not found: '{dir_path}'")
        sys.exit(1)

    if not args.file:
        print(f"No CSV file choosen: '{dir_path}'")
        sys.exit(1)

    # ==== Đọc dữ liệu ====
    file_path = os.path.join(dir_path, args.file)
    df = pd.read_csv(file_path)

    # ==== Xem phân bổ nhãn ====
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="label", order=df['label'].value_counts().index, palette="viridis")
    plt.title("Phân bổ mẫu theo nhãn")
    plt.xlabel("Label")
    plt.ylabel("Số lượng mẫu")
    plt.show()

    # ==== Chuẩn bị dữ liệu cho feature importance ====
    # Loại bỏ cột không phải đặc trưng số (url, label)
    X = df.drop(columns=["url", "label"])
    y = df["label"]

    # Encode label sang số
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # ==== Huấn luyện Random Forest để lấy feature importance ====
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_encoded)

    # Lấy độ quan trọng đặc trưng
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # ==== Hiển thị top 15 đặc trưng ====
    top_num = args.top if args.top else 15
    plt.figure(figsize=(10,6))
    sns.barplot(data=feature_importance.head(top_num), x="Importance", y="Feature", palette="coolwarm")
    plt.title(f"Top {top_num} đặc trưng quan trọng nhất")
    plt.show()

    # ==== In bảng độ quan trọng ====
    print(feature_importance)
