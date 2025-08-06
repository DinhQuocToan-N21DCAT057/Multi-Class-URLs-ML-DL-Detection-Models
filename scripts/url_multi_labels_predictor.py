import os
import numpy as np
import pandas as pd
import nltk
import re
import joblib
import time

from tensorflow.keras.models import load_model
from scripts.url_features_extractor import URL_EXTRACTOR
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))


class URL_PREDICTOR(object):
    def __init__(self, url, dataset="dataset_1"):
        self.url = url
        self.dataset = dataset
        self.CNN_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}",
                                                     "CNN_MODEL_ON_NUMERICAL_FEATURES.keras")
        self.CNN_NON_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}",
                                                         "CNN_MODEL_ON_NON_NUMERICAL_FEATURES.keras")
        self.XGB_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}",
                                                     "XGB_MODEL_ON_NUMERICAL_FEATURES.pkl")
        self.XGB_NON_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}",
                                                         "XGB_MODEL_ON_NON_NUMERICAL_FEATURES.pkl")
        self.RF_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}",
                                                    "RF_MODEL_ON_NUMERICAL_FEATURES.pkl")
        self.RF_NON_NUMERICAL_MODEL_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}",
                                                        "RF_MODEL_ON_NON_NUMERICAL_FEATURES.pkl")
        self.SCALER_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "scaler.pkl")
        self.CNN_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "tfidf_vectorizer_CNN.pkl")
        self.XGB_RF_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", f"{self.dataset}", "tfidf_vectorizer_XGB_RF.pkl")
        
        # Load scaler once during initialization
        self.scaler = self.load_scaler()

        self.df = self.extract_url_features_to_df()
        self.y = self.df['label'] if 'label' in self.df.columns else None
        self.ps = PorterStemmer()
        self.corpus = []
        self.label_names = ['benign', 'defacement', 'malware', 'phishing']
        self.predictions = None
        self.predicted_labels = None
        self.exec_time = 0

    def get_features_as_dict(self):
        """Returns the extracted features as a dictionary."""
        if self.df is not None:
            # Convert dataframe to dict, excluding non-feature columns
            features_dict = self.df.drop(columns=['label', 'url'], errors='ignore').to_dict(orient='records')
            return features_dict[0] if features_dict else {}
        return {}

    def _format_predictions(self, predictions, threshold):
        """Formats raw model predictions into the required structure."""
        # Handle multi-label array from scikit-learn
        if isinstance(predictions, list) and len(predictions) == len(self.label_names):
            # This is likely from a multi-output classifier, stack the probabilities for the positive class
            predictions = np.stack([p[:, 1] if p.shape[1] > 1 else p.flatten() for p in predictions], axis=1)

        self.predictions = predictions.flatten().tolist()
        binary_labels = (predictions.flatten() > threshold).astype(int)
        self.predicted_labels = {label: int(pred) for label, pred in zip(self.label_names, binary_labels) if pred == 1}

    def predict_with_RF(self, threshold=0.5, numerical=True):
        start_time = time.time()
        model_path = self.RF_NUMERICAL_MODEL_PATH if numerical else self.RF_NON_NUMERICAL_MODEL_PATH
        model = self.model_loader(model_path)
        
        if numerical:
            data = self.X1_pre_processing()
        else:
            data = self.X4_pre_processing()

        raw_predictions = model.predict_proba(data)
        self._format_predictions(raw_predictions, threshold)
        
        # If the prediction is benign, it's already correctly set.
        # Otherwise, filter for the specific model's labels.
        if 'benign' in self.predicted_labels:
            final_labels = {'benign': 1}
        else:
            final_labels = {label: self.predicted_labels.get(label, 0) for label in self.label_names}

        self.predicted_labels = final_labels
        
        end_time = time.time()
        self.exec_time = (end_time - start_time) * 1000 # Convert to ms

    def predict_with_CNN(self, threshold=0.5, numerical=True):
        start_time = time.time()
        model_path = self.CNN_NUMERICAL_MODEL_PATH if numerical else self.CNN_NON_NUMERICAL_MODEL_PATH
        model = self.model_loader(model_path)

        if numerical:
            data = self.X1_pre_processing()
        else:
            data = self.X2_pre_processing()

        raw_predictions = model.predict(data)
        self._format_predictions(raw_predictions, threshold)
        
        # If the prediction is benign, it's already correctly set.
        # Otherwise, filter for the specific model's labels.
        if 'benign' in self.predicted_labels:
            final_labels = {'benign': 1}
        else:
            final_labels = {label: self.predicted_labels.get(label, 0) for label in self.label_names}

        self.predicted_labels = final_labels
        
        end_time = time.time()
        self.exec_time = (end_time - start_time) * 1000 # Convert to ms

    def predict_with_XGB(self, threshold=0.5, numerical=True):
        start_time = time.time()
        model_path = self.XGB_NUMERICAL_MODEL_PATH if numerical else self.XGB_NON_NUMERICAL_MODEL_PATH
        model = self.model_loader(model_path)

        if numerical:
            data = self.X1_pre_processing()
        else:
            data = self.X4_pre_processing()

        raw_predictions = model.predict_proba(data)
        self._format_predictions(raw_predictions, threshold)
        
        # If the prediction is benign, it's already correctly set.
        # Otherwise, filter for the specific model's labels.
        if 'benign' in self.predicted_labels:
            final_labels = {'benign': 1}
        else:
            final_labels = {label: self.predicted_labels.get(label, 0) for label in self.label_names}

        self.predicted_labels = final_labels
        
        end_time = time.time()
        self.exec_time = (end_time - start_time) * 1000 # Convert to ms

    def print_result(self):
        if self.predictions is not None and self.predicted_labels is not None:
            for i, (pred, prob) in enumerate(zip(self.predicted_labels, self.predictions)):
                print(f"\nSample {i + 1} - Url: {self.url}")
                for label, p, pl in zip(self.label_names, prob, pred):
                    print(f"{label}: Probability = {p:.4f}, Predicted = {bool(pl)}")
                if self.y is not None:
                    true_label = self.y.iloc[i]
                    print(f"True label: {true_label}")
            print(f"\tExecution Time: {self.exec_time:.2f} ms")

    def X1_pre_processing(self):
        X1 = self.df.drop(columns=['label', 'url'], errors='ignore')
        if self.scaler:
            X1 = self.scaler.transform(X1)
        return X1

    def load_scaler(self):
        try:
            if os.path.exists(self.SCALER_PATH):
                scaler = joblib.load(self.SCALER_PATH)
                print(f"Scaler loaded successfully from {self.SCALER_PATH}")
                return scaler
            else:
                print(f"Scaler not found at {self.SCALER_PATH}. Numerical predictions will not be scaled.")
                return None
        except Exception as e:
            print(f"Error loading scaler: {e}. Numerical predictions will not be scaled.")
            return None

    def X2_pre_processing(self):
        X2 = self.df['url']
        self.albumentations(X2)
        self.load_vectorizer(self.CNN_VECTORIZER_PATH)
        X2 = self.cv.transform(self.corpus).toarray()
        X2 = np.reshape(X2, (X2.shape[0], X2.shape[1], 1))
        return X2

    def load_vectorizer(self, PATH):
        try:
            if os.path.exists(PATH):
                self.cv = joblib.load(PATH)
                print(f"{PATH} loaded successfully!")
            else:
                # Create a default vectorizer if not found
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.cv = TfidfVectorizer(max_features=1000)
                print(f"Vectorizer not found at {PATH}, using default TfidfVectorizer")
        except Exception as e:
            print(f"Error loading {PATH}: {e}")
            # Use default vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.cv = TfidfVectorizer(max_features=1000)

    def X3_pre_processing(self):
        # This method is redundant now. X1_pre_processing handles numerical features.
        return self.X1_pre_processing()

    def X4_pre_processing(self):
        X4 = self.df['url']
        self.corpus = [] # Reset corpus for each call
        self.albumentations2(X4)
        self.load_vectorizer(self.XGB_RF_VECTORIZER_PATH)
        X4 = self.cv.transform(self.corpus).toarray()
        return X4

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
            review = X.iloc[i] if hasattr(X, 'iloc') else X[i]
            review = review.decode('utf-8') if isinstance(review, bytes) else str(review)
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
            review = X.iloc[i] if hasattr(X, 'iloc') else X[i]
            review = review.decode('utf-8') if isinstance(review, bytes) else str(review)
            review = re.sub(r'\?.*', '', review)
            review = re.sub(r'[^a-zA-Z0-9\-\/.]', ' ', review)
            review = review.lower()
            review = review.split()
            review = [self.ps.stem(word) for word in review if
                      word not in set(stopwords.words("english")) and len(word) > 2]
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
        """Loads a model from the given path."""
        try:
            if "keras" in PATH:
                model = load_model(PATH)
            else:
                model = joblib.load(PATH)
            print(f"Model loaded successfully from {PATH}")
        except Exception as e:
            print(f"Error loading model {PATH}: {e}")
            from sklearn.dummy import DummyClassifier
            # Create and fit a dummy model with placeholder data to prevent crashes
            model = DummyClassifier(strategy='uniform', random_state=42)
            dummy_X = np.zeros((1, 10)) # Dummy features
            dummy_y = np.zeros((1, 4))   # Dummy labels for 4 classes
            model.fit(dummy_X, dummy_y)
            print(f"Using fitted dummy model due to error")
        return model
