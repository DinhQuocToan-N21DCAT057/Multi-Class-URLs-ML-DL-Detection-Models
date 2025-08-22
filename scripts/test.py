from operator import index
from pickle import TRUE
import pandas as pd
import os
import requests
import json

from url_features_extractor_v2 import URL_EXTRACTOR

url = "https://uis.ptithcm.edu.vn/#/home"

temp = []
extractor = URL_EXTRACTOR(url)
data = extractor.extract_to_dataset()
temp.append(data)

test = pd.DataFrame(temp)
test.to_csv("test.csv", index=False)

