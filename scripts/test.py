from operator import index
from pickle import TRUE
import pandas as pd

df = pd.read_csv("balanced_dataset_1.csv")

test = df.iloc[0:10]
test.to_csv("test.csv", index=False)

# from url_multi_labels_predictor import URL_PREDICTOR


# URL_PREDICTOR.preload(['xgb_num', 'xgb_non', 'rf_num', 'rf_non'])
# # URL_PREDICTOR.preload_scaler()
# URL_PREDICTOR.preload_vectorizers(['xgb_rf'])

# url1 = "https://wayground.com/join?source=liveDashboard"
# # url2 = "https://www.amazon.com/"

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_XGB(threshold=0.85, numerical=True)
# #predictor.df.to_csv("test.csv")
# predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_XGB(threshold=0.85, numerical=False)
# predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_RF(threshold=0.85, numerical=True)
# predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_RF(threshold=0.85, numerical=False)
# predictor.print_result()


