import pandas as pd
from url_features_extractor import URL_EXTRACTOR
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "malicious_phish.csv")

df = pd.read_csv(DATASET_PATH)

temp = []
for item in df.itertuples(index=False):
    data = URL_EXTRACTOR(item.url, item.type).extract_to_dataset()
    temp.append(data)

final_dataset = pd.DataFrame(temp)
final_dataset.to_csv("final_dataset.csv", index=False)