from pickle import TRUE
import pandas as pd
from url_multi_labels_predictor import URL_PREDICTOR


url = "https://www.allegrolokalnie.85xc.rest"

predictor = URL_PREDICTOR(url)
predictor.predict_with_RF(threshold=0.9, numerical=True)
#predictor.df.to_csv("test.csv", index=False)
predictor.print_result()
