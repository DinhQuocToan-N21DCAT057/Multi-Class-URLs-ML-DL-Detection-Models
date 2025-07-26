import pandas as pd
from url_features_extractor import URL_EXTRACTOR
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "url_malware.csv")


def is_ascii(s):
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


df = pd.read_csv(DATASET_PATH)

df = df.iloc[20000:22000]
# df2 = df.iloc[550000:552000]
# df3 = df.iloc[554000:556000]
# df4 = df.iloc[572000:574000]

# df1_clean = df1[df1["url"].apply(is_ascii)]
# df2_clean = df2[df2['url'].apply(is_ascii)]
# df3_clean = df3[df3['url'].apply(is_ascii)]
# df4_clean = df4[df4['url'].apply(is_ascii)]

# df1_clean = df1_clean[df1_clean["type"] != "benign"]
# df2_clean = df2_clean[df2_clean['type'] != 'benign']
# df3_clean = df3_clean[df3_clean['type'] != 'benign']
# df4_clean = df4_clean[df4_clean['type'] != 'benign']

# df_list = []

# df_list.append(df)
# df_list.append(df2_clean)
# df_list.append(df3_clean)
# df_list.append(df4_clean)

# missing_df = pd.concat(df_list, ignore_index=True)
# missing_df.to_csv("missing_urls.csv", index=False)
temp = []
df1 = df.iloc[0:100]
for row in df1.itertuples(index=False):
    r = URL_EXTRACTOR(row.url, row.type)
    data = r.extract_to_dataset()
    print(r.whois_cache)
    print(r.domain)
    temp.append(data)

# data = URL_EXTRACTOR("https://www.microsoft.com/en-us", "benign").extract_to_dataset()
# data = URL_EXTRACTOR("https://www.amazon.com/", "benign").extract_to_dataset()
# temp.append(data)

test_df = pd.DataFrame(temp)
test_df.to_csv("test.csv", index=False)

