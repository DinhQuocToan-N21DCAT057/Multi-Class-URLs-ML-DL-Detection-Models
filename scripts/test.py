from pickle import TRUE
import pandas as pd
from url_multi_labels_predictor import URL_PREDICTOR


URL_PREDICTOR.preload(['bert_non', 'cnn_num'])
# URL_PREDICTOR.preload_scaler()
# URL_PREDICTOR.preload_vectorizers(['cnn'])

url1 = "https://www.coursera.org/"
# url2 = "https://www.amazon.com/"

predictor = URL_PREDICTOR(url1)
predictor.predict_with_CNN(threshold=0.85, numerical=True)
predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_CNN(threshold=0.85, numerical=False)
# predictor.print_result()

# predictor = URL_PREDICTOR(url2)
# predictor.predict_with_CNN(threshold=0.85, numerical=False)
# predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_RF(threshold=0.85, numerical=True)
# predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_TF_BERT()
# predictor.print_result()


