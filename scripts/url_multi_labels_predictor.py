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
    def __init__(self, url, dataset="dataset_1"):
        self.url = url
        self.dataset = dataset
        self.CNN_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "CNN_MODEL_ON_NUMERICAL_FEATURES.keras")
        self.CNN_NON_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "CNN_MODEL_ON_NON_NUMERICAL_FEATURES.keras")
        self.XGB_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "XGB_MODEL_ON_NUMERICAL_FEATURES.pkl")
        self.XGB_NON_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "XGB_MODEL_ON_NON_NUMERICAL_FEATURES.pkl")
        self.RF_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "RF_MODEL_ON_NUMERICAL_FEATURES.pkl")
        self.RF_NON_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "RF_MODEL_ON_NON_NUMERICAL_FEATURES.pkl")
        self.SCALER_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "scaler.pkl")
        self.CNN_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "tfidf_vectorizer_CNN.pkl")
        self.XGB_RF_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "tfidf_vectorizer_XGB_RF.pkl")
        self.df = self.extract_url_features_to_df()
        self.df.to_csv(f"test.csv", index=False)
        self.y = self.df['label'] if 'label' in self.df.columns else None
        self.ps = PorterStemmer()
        self.corpus = []
        self.label_names = ['benign', 'defacement', 'malware', 'phishing']


    def predict_with_RF(self, threshold=0.5, numerical=True):
        if numerical:
            self.X3_pre_processing()
            self.RF_NUMERICAL_MODEL = self.model_loader(self.RF_NUMERICAL_MODEL_PATH)
            if hasattr(self.RF_NUMERICAL_MODEL, 'predict_proba'):
                self.predictions = self.RF_NUMERICAL_MODEL.predict_proba(self.X3)
                # Nếu là multi-label, predict_proba trả về list các mảng, cần stack lại
                if isinstance(self.predictions, list):
                    self.predictions = np.stack([p[:,1] if p.shape[1]==2 else p for p in self.predictions], axis=1)
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                self.predictions = self.RF_NUMERICAL_MODEL.predict(self.X3)
                self.predicted_labels = self.predictions
        else:
            self.X4_pre_processing()
            self.RF_NON_NUMERICAL_MODEL = self.model_loader(self.RF_NON_NUMERICAL_MODEL_PATH)
            if hasattr(self.RF_NON_NUMERICAL_MODEL, 'predict_proba'):
                self.predictions = self.RF_NON_NUMERICAL_MODEL.predict_proba(self.X4)
                if isinstance(self.predictions, list):
                    self.predictions = np.stack([p[:,1] if p.shape[1]==2 else p for p in self.predictions], axis=1)
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                self.predictions = self.RF_NON_NUMERICAL_MODEL.predict(self.X4)
                self.predicted_labels = self.predictions

    def predict_with_CNN(self, threshold=0.5, numerical=True):
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

    def predict_with_XGB(self, threshold=0.5, numerical=True):
        if numerical:
            self.X3_pre_processing()
            self.XGB_NUMERICAL_MODEL = self.model_loader(self.XGB_NUMERICAL_MODEL_PATH)
            # Lấy xác suất dự đoán
            if hasattr(self.XGB_NUMERICAL_MODEL, 'predict_proba'):
                self.predictions = self.XGB_NUMERICAL_MODEL.predict_proba(self.X3)
                # Nếu là multi-label, predict_proba trả về list các mảng, cần stack lại
                if isinstance(self.predictions, list):
                    self.predictions = np.stack([p[:,1] if p.shape[1]==2 else p for p in self.predictions], axis=1)
                # Nếu là binary/multi-label dạng (n_samples, n_classes)
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                # Nếu model không có predict_proba (rất hiếm), fallback predict trả về nhãn
                self.predictions = self.XGB_NUMERICAL_MODEL.predict(self.X3)
                self.predicted_labels = self.predictions
        else:
            self.X4_pre_processing()
            self.XGB_NON_NUMERICAL_MODEL = self.model_loader(self.XGB_NON_NUMERICAL_MODEL_PATH)
            if hasattr(self.XGB_NON_NUMERICAL_MODEL, 'predict_proba'):
                self.predictions = self.XGB_NON_NUMERICAL_MODEL.predict_proba(self.X4)
                if isinstance(self.predictions, list):
                    self.predictions = np.stack([p[:,1] if p.shape[1]==2 else p for p in self.predictions], axis=1)
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                self.predictions = self.XGB_NON_NUMERICAL_MODEL.predict(self.X4)
                self.predicted_labels = self.predictions

    def print_result(self):
        if self.predictions is not None and self.predicted_labels is not None:
            for i, (pred, prob) in enumerate(zip(self.predicted_labels, self.predictions)):
                print(f"\nSample {i + 1} - Url: {self.url}")
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
        self.load_vectorizer(self.CNN_VECTORIZER_PATH)
        self.X2 = self.cv.transform(self.corpus).toarray()
        self.X2 = np.expand_dims(self.X2, axis=-1)

    def load_vectorizer(self, PATH):
        try:
            if os.path.exists(PATH):
                self.cv = joblib.load(PATH)
                print(f"{PATH} loaded successfully!")
        except Exception as e:
            print(f"Error loading {PATH}: {e}")
            raise
    
    def X3_pre_processing(self):
        self.X3 = self.df.drop(columns = ['label', 'url'])
    
    def X4_pre_processing(self):
        self.X4 = self.df['url']
        self.albumentations2(self.X4)
        self.load_vectorizer(self.XGB_RF_VECTORIZER_PATH)
        self.X4 = self.cv.transform(self.corpus).toarray()

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

    def albumentations2(self, X):
        self.download_stopsword()
        for i in range(len(X)):
            print(i, "/", len(X))
            review = X[i].decode('utf-8') if isinstance(X[i], bytes) else str(X[i])
            review = re.sub(r'\?.*', '', review)
            review = re.sub(r'[^a-zA-Z0-9\-\/.]', ' ', review)
            review = review.lower()
            review = review.split()
            review = [self.ps.stem(word) for word in review if word not in set(stopwords.words("english")) and len(word) > 2]
            review = " ".join(review)
            self.corpus.append(review)

    def extract_url_features_to_df(self):
        df = []
        try:
            extractor = URL_EXTRACTOR(self.url)
            data = extractor.extract_to_predict()
            print(f"  URL '{self.url}' took {round(extractor.exec_time, 2)} seconds to extract")
            df.append(data)
        except Exception as e:
            print(f"Error extracting features: {e}")
            raise
        return pd.DataFrame(df)
    
    def model_loader(self, PATH):
        model = None
        try:
            if os.path.exists(PATH):
                if PATH.endswith('.keras'):
                    model = load_model(PATH)
                    print(f"{PATH} (Keras model) loaded successfully!")
                elif PATH.endswith('.pkl'):
                    model = joblib.load(PATH)
                    print(f"{PATH} (Pickle model) loaded successfully!")
                else:
                    raise ValueError(f"Unsupported model file extension for {PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        return model

