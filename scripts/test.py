import pandas as pd
from url_features_extractor import URL_EXTRACTOR
import sys
import os
import numpy as np
import time
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "malicious_phish.csv")

df = pd.read_csv(DATASET_PATH)

num_samples = 10
sample_indices = np.random.choice(df.shape[0], num_samples, replace=False)
test_samples = df.iloc[sample_indices]

temp = []

for idx, item in enumerate(test_samples.itertuples(index=False), 1):
    print("="*150)
    print(f"[{idx}/{num_samples}] Extracting features for:")
    print(f"  URL  : {item.url}")
    print(f"  Label: {item.type}")
    extractor = URL_EXTRACTOR(item.url, item.type)
    data = extractor.extract_to_dataset()
    print(f"  URL '{item.url}' took '{round(extractor.exec_time, 2)}' seconds to extract")
    temp.append(data)

print("="*150)
samples_df = pd.DataFrame(temp)
samples_df.to_csv("samples.csv", index=False)
print(samples_df.head(10))

