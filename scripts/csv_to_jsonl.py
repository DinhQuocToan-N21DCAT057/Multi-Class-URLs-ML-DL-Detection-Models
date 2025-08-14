import os
import sys
import pandas as pd
import argparse
import json

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
    if not os.path.isfile(file_path):
        print(f"CSV file not found: '{file_path}'")
        sys.exit(1)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to read CSV '{file_path}': {e}")
        sys.exit(1)

    if 'label' not in df.columns:
        print("Required column 'label' not found in CSV. Available columns:")
        print(list(df.columns))
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(args.file))[0]
    out_path = os.path.join(dir_path, f"{base_name}.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
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

    print(f"✅ Complete parsing from .csv to .jsonl. File: {out_path}")
