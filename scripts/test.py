from pickle import TRUE
import pandas as pd
from url_multi_labels_predictor import URL_PREDICTOR
from utils.utils import csv_to_json

rows = csv_to_json("test.csv", "test.json")


# URL_PREDICTOR.preload(['cnn_num', 'cnn_non', 'rf_num'])
# URL_PREDICTOR.preload_scaler()
# URL_PREDICTOR.preload_vectorizers(['cnn'])

# url1 = "https://account.microsoft.com/account"
# url2 = "https://www.amazon.com/"

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_CNN(threshold=0.85, numerical=True)
# predictor.print_result()

# predictor = URL_PREDICTOR(url2)
# predictor.predict_with_CNN(threshold=0.85, numerical=True)
# predictor.print_result()

# predictor = URL_PREDICTOR(url2)
# predictor.predict_with_CNN(threshold=0.85, numerical=False)
# predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_RF(threshold=0.85, numerical=True)
# predictor.print_result()


