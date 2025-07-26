import os
import numpy as np
import pandas as pd
import nltk
import re
import joblib

from tensorflow.keras.models import load_model
from url_features_extractor import URL_EXTRACTOR
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class URL_PREDICTOR(object):
    def __init__(self, url):
        self.CNN_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "CNN_MODEL_ON_NUMERICAL_FEATURES.keras")
        self.CNN_NON_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "CNN_MODEL_ON_NON_NUMERICAL_FEATURES.keras")
        self.SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
        self.VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
        self.df = self.extract_urls_features_to_df(url)
        self.y = self.df['label'] if 'label' in self.df.columns else None
        self.ps = PorterStemmer()
        self.corpus = []
        self.label_names = ['benign', 'defacement', 'malware', 'phishing']

    def predict_with_CNN(self, threshold = 0.5, numerical=True):
        if numerical:
            self.X1_pre_processing()
            self.CNN_NUMERICAL_MODEL = self.model_loader(self.CNN_NUMERICAL_MODEL_PATH)
            self.predictions = self.CNN_NUMERICAL_MODEL.predict(self.X1)
            self.predicted_labels = (self.predictions > threshold).astype(int)
        else:
            self.X2_pre_processing()
            self.CNN_NON_NUMERICAL_MODEL = self.model_loader(self.CNN_NON_NUMERICAL_MODEL_PATH)
            self.predictions = self.CNN_NON_NUMERICAL_MODEL.predict(self.X2)
            self.predicted_labels = (self.predictions > threshold).astype(int)

    def print_result(self):
        if self.predictions is not None and self.predicted_labels is not None:
            for i, (pred, prob) in enumerate(zip(self.predicted_labels, self.predictions)):
                print(f"\nSample {i + 1} (index: {i}):")
                for label, p, pl in zip(self.label_names, prob, pred):
                    print(f"{label}: Probability = {p:.4f}, Predicted = {bool(pl)}")
                if self.y is not None:
                    true_label = self.y.iloc[i]
                    print(f"True label: {true_label}")
        
    def X1_pre_processing(self):
        self.X1 = self.df.drop(columns = ['label', 'url'])
        columns_to_scale = [col for col in self.X1.columns if col not in ['domain_reg_len', 'domain_age', 'page_rank', 'google_index']]
        self.load_scaler()
        self.X1[columns_to_scale] = self.scaler.transform(self.X1[columns_to_scale])
        #self.X1.to_csv("test.csv", index=False)
        self.X1 = np.expand_dims(self.X1, axis=-1)

    def load_scaler(self):
        try:
            if os.path.exists(self.SCALER_PATH):
                self.scaler = joblib.load(self.SCALER_PATH)
                print(f"{self.SCALER_PATH} loaded successfully!")
        except Exception as e:
            print(f"Error loading {self.SCALER_PATH}: {e}")
            raise

    def X2_pre_processing(self):
        self.X2 = self.df['url']
        self.albumentations(self.X2)
        self.load_vectorizer()
        self.X2 = self.cv.transform(self.corpus).toarray()
        self.X2 = np.expand_dims(self.X2, axis=-1)

    def load_vectorizer(self):
        try:
            if os.path.exists(self.VECTORIZER_PATH):
                self.cv = joblib.load(self.VECTORIZER_PATH)
                print(f"{self.VECTORIZER_PATH} loaded successfully!")
        except Exception as e:
            print(f"Error loading {self.VECTORIZER_PATH}: {e}")
            raise

    @staticmethod
    def download_stopsword():
        try:
            print("Check nltk's stopwords")
            stopwords.words("english")
        except LookupError:
            print("ntlk's stopwords not found! Downloading NLTK stopwords...")
            nltk.download('stopwords')

    def albumentations(self, X):
        self.download_stopsword()
        for i in range(len(X)):
            print(i, "/", len(X))
            review = X[i].decode('utf-8') if isinstance(X[i], bytes) else str(X[i])
            review = re.sub(r'\?.*', '', review)
            review = re.sub(r'[^a-zA-Z0-9\-\/.]', ' ', review)
            review = review.lower()
            review = review.split()
            review = [self.ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
            review = " ".join(review)
            self.corpus.append(review)

    def extract_urls_features_to_df(self, url):
        df = []
        try:
            extractor = URL_EXTRACTOR(url)
            data = extractor.extract_to_predict()
            print(f"  URL '{url}' took {round(extractor.exec_time, 2)} seconds to extract")
            df.append(data)
        except Exception as e:
            print(f"Error extracting features: {e}")
            raise
        return pd.DataFrame(df)
    
    def model_loader(self, PATH):
        model = None
        try:
            if os.path.exists(PATH):
                model = load_model(PATH)
                print(f"{PATH} loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        return model

