import pandas as pd
from url_features_extractor import URL_EXTRACTOR
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "malicious_phish.csv")

df = pd.read_csv(DATASET_PATH)
total = len(df)
temp = []
for idx, item in enumerate(df.itertuples(index=False), 1):
    print("="*150)
    print(f"[{idx}/{total}] Extracting features for:")
    print(f"  URL  : {item.url}")
    print(f"  Label: {item.type}")
    extractor = URL_EXTRACTOR(item.url, item.type)
    data = extractor.extract_to_dataset()
    print(f"  URL '{item.url}' took '{round(extractor.exec_time, 2)}' seconds to extract")
    temp.append(data)

print("="*150)
final_dataset = pd.DataFrame(temp)
final_dataset.to_csv("final_dataset.csv", index=False)