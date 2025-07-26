import os
import sys
import pandas as pd
import argparse
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def balance_and_split_dataset(input_file, label_col='type', output_prefix='balanced_dataset'):
    df = pd.read_csv(input_file)
    benign_df = df[df[label_col] == 'benign']
    defacement_df = df[df[label_col] == 'defacement']
    malware_df = df[df[label_col] == 'malware']
    phishing_df = df[df[label_col] == 'phishing']

    benign_splits = np.array_split(benign_df, 4)
    for i, benign_part in enumerate(benign_splits, 1):
        merged = pd.concat([
            benign_part,
            defacement_df,
            malware_df,
            phishing_df
        ], ignore_index=True)
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        out_file = f"{output_prefix}_{i}.csv"
        merged.to_csv(os.path.join(BASE_DIR, out_file), index=False)
        print(f"Saved: {out_file} ({len(merged)} rows)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='URL Multi-Labels Dataset Balancer')
    parser.add_argument('--dir', type=str, required=True, help='Directory name of dataset')
    parser.add_argument('--file', type=str, required=True, help='Processing file')
    parser.add_argument('--o', type=str, required=False, help='Output file name')
    parser.add_argument('--split_benign', action='store_true', help='Split benign and merge with other classes')
    parser.add_argument('--label_col', type=str, default='type', help='Label column name (default: type)')
    args = parser.parse_args()

    dir_path = os.path.join(BASE_DIR, args.dir)
    if not os.path.isdir(dir_path):
        print(f"Directory not found: '{dir_path}'")
        sys.exit(1)

    if not args.file:
        print(f"No CSV file choosen")
        sys.exit(1)

    file_path = os.path.join(dir_path, args.file)
    try:
        df = pd.read_csv(file_path)
    except Exception:
        print(f"File {file_path} not found!")
        sys.exit(1)

    if args.split_benign:
        balance_and_split_dataset(file_path, label_col=args.label_col, output_prefix=args.o or 'balanced_dataset')
        sys.exit(0)
