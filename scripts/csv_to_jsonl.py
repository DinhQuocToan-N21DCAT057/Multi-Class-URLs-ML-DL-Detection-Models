import os
import sys
import pandas as pd
import argparse
import json

from scripts.dataset_processing import file

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing from .CSV to .JSONL format')
    parser.add_argument('--dir', type=str, required=True, help='Directory name of dataset')
    parser.add_argument('--file', type=str, required=True, help='Processing files')
    args = parser.parse_args()

    dir_path = os.path.join(BASE_DIR, args.dir)
    if not os.path.isdir(dir_path):
        print(f"Directory not found: '{dir_path}'")
        sys.exit(1)

    if not args.file:
        print(f"No CSV file choosen: '{dir_path}'")
        sys.exit(1)

    file_path = os.path.join(dir_path, args.file)
    
    df = pd.read(file_path)

    with open(f"{args.file}.jsonl", "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # Bỏ cột label ra để làm prompt
            features = {col: row[col] for col in df.columns if col != "label"}

            # Chuyển dict đặc trưng thành chuỗi
            feature_str = ", ".join([f"{k}={v}" for k, v in features.items()])

            # Prompt và completion
            prompt = f"Given the URL features: {feature_str}. Predict the category:"
            completion = f" {row['label']}"  # Có space ở đầu để tránh nối liền chữ

            # Ghi vào JSONL
            json.dump({"prompt": prompt, "completion": completion}, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Complete parsing from .csv to .jsonl! File name: {args.file}.jsonl")
