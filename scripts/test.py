import pandas as pd
from url_features_extractor import URL_EXTRACTOR
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "malicious_phish.csv")


def is_ascii(s):
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


df = pd.read_csv(DATASET_PATH)

df1 = df.iloc[548000:550000]
# df2 = df.iloc[550000:552000]
# df3 = df.iloc[554000:556000]
# df4 = df.iloc[572000:574000]

df1_clean = df1[df1["url"].apply(is_ascii)]
# df2_clean = df2[df2['url'].apply(is_ascii)]
# df3_clean = df3[df3['url'].apply(is_ascii)]
# df4_clean = df4[df4['url'].apply(is_ascii)]

df1_clean = df1_clean[df1_clean["type"] != "benign"]
# df2_clean = df2_clean[df2_clean['type'] != 'benign']
# df3_clean = df3_clean[df3_clean['type'] != 'benign']
# df4_clean = df4_clean[df4_clean['type'] != 'benign']

df_list = []

df_list.append(df1_clean)
# df_list.append(df2_clean)
# df_list.append(df3_clean)
# df_list.append(df4_clean)

missing_df = pd.concat(df_list, ignore_index=True)
# missing_df.to_csv("missing_urls.csv", index=False)
temp = []
df = missing_df.iloc[0:3]
for row in df.itertuples(index=False):
    data = URL_EXTRACTOR(row.url, row.type).extract_to_dataset()
    temp.append(data)

test_df = pd.DataFrame(temp)
test_df.to_csv("test.csv", index=False)
