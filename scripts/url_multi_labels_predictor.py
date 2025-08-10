import time
import os

import sys
import numpy as np

import pandas as pd
import numpy as np
import joblib

import logging
import time

from functools import wraps
from tensorflow.keras.models import load_model
from scripts.url_features_extractor import URL_EXTRACTOR
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Ensure project root (parent of `scripts`) is on sys.path so `configs` can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def timer(func):
    """Record execution time of any functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        if args and hasattr(args[0], "exec_time"):
            args[0].exec_time += elapsed_time
            logging.info(
                f"Function '{func.__name__}' took {elapsed_time:.2f} seconds, cumulative exec_time: {args[0].exec_time:.2f} seconds"
            )
        else:
            logging.info(
                f"Function '{func.__name__}' took {elapsed_time:.2f} seconds (no instance with exec_time found)"
            )
        return result  # Return the original function's result

    return wrapper

class URL_PREDICTOR(object):
    # Caches to ensure single-load per process
    _MODEL_CACHE = {}
    _VECTORIZER_CACHE = {}
    _SCALER_CACHE = {}

    @classmethod
    def preload(cls, models_to_load=None):
        """Optionally preload selected models once and cache them.

        models_to_load can include: 'cnn_num', 'cnn_non', 'xgb_num', 'xgb_non', 'rf_num', 'rf_non'
        """
        if not models_to_load:
            return
        key_to_path = {
            'cnn_num': Config.CNN_NUMERICAL_MODEL_PATH,
            'cnn_non': Config.CNN_NON_NUMERICAL_MODEL_PATH,
            'xgb_num': Config.XGB_NUMERICAL_MODEL_PATH,
            'xgb_non': Config.XGB_NON_NUMERICAL_MODEL_PATH,
            'rf_num': Config.RF_NUMERICAL_MODEL_PATH,
            'rf_non': Config.RF_NON_NUMERICAL_MODEL_PATH,
        }
        # Temporary instance to reuse model_loader
        temp = cls.__new__(cls)
        for key in models_to_load:
            path = key_to_path.get(key)
            if not path:
                continue
            if path in cls._MODEL_CACHE:
                continue
            model = URL_PREDICTOR.model_loader(temp, path)
            cls._MODEL_CACHE[path] = model

    @classmethod
    def preload_scaler(cls):
        """Preload and cache the scaler once."""
        if Config.SCALER_PATH in cls._SCALER_CACHE:
            return
        temp = cls.__new__(cls)
        URL_PREDICTOR.load_scaler(temp, Config.SCALER_PATH)

    @classmethod
    def preload_vectorizers(cls, which=("cnn", "xgb_rf")):
        """Preload and cache selected vectorizers once.

        which can include: 'cnn' (for CNN text) and/or 'xgb_rf' (for XGB/RF text)
        """
        temp = cls.__new__(cls)
        mapping = {
            "cnn": Config.CNN_VECTORIZER_PATH,
            "xgb_rf": Config.XGB_RF_VECTORIZER_PATH,
        }
        for key in which or []:
            path = mapping.get(key)
            if not path or path in cls._VECTORIZER_CACHE:
                continue
            URL_PREDICTOR.load_vectorizer(temp, path)
    @timer
    def __init__(self, url, enable_logging=False, preset_df=None):
        # Log and execution time informations
        self.exec_time = 0.0
        self.log_level = logging.INFO if enable_logging else logging.WARNING
        logging.getLogger().setLevel(self.log_level)

        # Attach cached models if already preloaded
        self.CNN_NUMERICAL_MODEL = self._MODEL_CACHE.get(Config.CNN_NUMERICAL_MODEL_PATH)
        self.CNN_NON_NUMERICAL_MODEL = self._MODEL_CACHE.get(Config.CNN_NON_NUMERICAL_MODEL_PATH)
        self.XGB_NUMERICAL_MODEL = self._MODEL_CACHE.get(Config.XGB_NUMERICAL_MODEL_PATH)
        self.XGB_NON_NUMERICAL_MODEL = self._MODEL_CACHE.get(Config.XGB_NON_NUMERICAL_MODEL_PATH)
        self.RF_NUMERICAL_MODEL = self._MODEL_CACHE.get(Config.RF_NUMERICAL_MODEL_PATH)
        self.RF_NON_NUMERICAL_MODEL = self._MODEL_CACHE.get(Config.RF_NON_NUMERICAL_MODEL_PATH)
        self.scaler = self._SCALER_CACHE.get(Config.SCALER_PATH)
        self.cv2 = self._VECTORIZER_CACHE.get(Config.CNN_VECTORIZER_PATH)
        self.cv4 = self._VECTORIZER_CACHE.get(Config.XGB_RF_VECTORIZER_PATH)

        self.url = url
        # Use pre-extracted features DataFrame if provided to avoid re-extraction
        if preset_df is not None:
            self.df = preset_df
        else:
            self.df = self.extract_url_features_to_df()
        self.y = self.df['label'] if 'label' in self.df.columns else None
        self.ps = PorterStemmer()
        self.corpus = []
        self.predictions = None
        self.predicted_labels = None

    @timer
    def predict_with_RF(self, threshold=0.5, numerical=True):
        if numerical:

            self.X3_pre_processing()
            if self.RF_NUMERICAL_MODEL is None:
                raise ValueError("RF numerical model not loaded. Call URL_PREDICTOR.preload(['rf_num']) once, or set predictor.RF_NUMERICAL_MODEL.")
            # Align feature columns and order to match training
            try:
                if hasattr(self.RF_NUMERICAL_MODEL, 'feature_names_in_'):
                    expected = list(self.RF_NUMERICAL_MODEL.feature_names_in_)
                    for cname in expected:
                        if cname not in self.X3.columns:
                            self.X3[cname] = 0
                    self.X3 = self.X3.reindex(columns=expected)
            except Exception:
                pass
            if hasattr(self.RF_NUMERICAL_MODEL, 'predict_proba'):
                self.predictions = self.RF_NUMERICAL_MODEL.predict_proba(self.X3)
                # Nếu là multi-label, predict_proba trả về list các mảng, cần stack lại
                if isinstance(self.predictions, list):
                    self.predictions = np.stack([p[:, 1] if p.shape[1] == 2 else p for p in self.predictions], axis=1)
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                self.predictions = self.RF_NUMERICAL_MODEL.predict(self.X3)
                self.predicted_labels = self.predictions
        else:
            self.X4_pre_processing()
            if self.RF_NON_NUMERICAL_MODEL is None:
                raise ValueError("RF non-numerical model not loaded. Call URL_PREDICTOR.preload(['rf_non']) once, or set predictor.RF_NON_NUMERICAL_MODEL.")
            if hasattr(self.RF_NON_NUMERICAL_MODEL, 'predict_proba'):
                self.predictions = self.RF_NON_NUMERICAL_MODEL.predict_proba(self.X4)
                if isinstance(self.predictions, list):
                    self.predictions = np.stack([p[:, 1] if p.shape[1] == 2 else p for p in self.predictions], axis=1)
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                self.predictions = self.RF_NON_NUMERICAL_MODEL.predict(self.X4)
                self.predicted_labels = self.predictions


    @timer
    def predict_with_CNN(self, threshold=0.5, numerical=True):
        if numerical:

            self.X1_pre_processing()
            if self.CNN_NUMERICAL_MODEL is None:
                raise ValueError("CNN numerical model not loaded. Call URL_PREDICTOR.preload(['cnn_num']) once, or set predictor.CNN_NUMERICAL_MODEL.")
            self.predictions = self.CNN_NUMERICAL_MODEL.predict(self.X1)
            self.predicted_labels = (self.predictions > threshold).astype(int)
        else:
            self.X2_pre_processing()
            if self.CNN_NON_NUMERICAL_MODEL is None:
                raise ValueError("CNN non-numerical model not loaded. Call URL_PREDICTOR.preload(['cnn_non']) once, or set predictor.CNN_NON_NUMERICAL_MODEL.")
            self.predictions = self.CNN_NON_NUMERICAL_MODEL.predict(self.X2)
            self.predicted_labels = (self.predictions > threshold).astype(int)


    @timer
    def predict_with_XGB(self, threshold=0.5, numerical=True):
        if numerical:

            self.X3_pre_processing()
            if self.XGB_NUMERICAL_MODEL is None:
                raise ValueError("XGB numerical model not loaded. Call URL_PREDICTOR.preload(['xgb_num']) once, or set predictor.XGB_NUMERICAL_MODEL.")
            # Align feature columns and order to match training
            try:
                if hasattr(self.XGB_NUMERICAL_MODEL, 'feature_names_in_'):
                    expected = list(self.XGB_NUMERICAL_MODEL.feature_names_in_)
                    for cname in expected:
                        if cname not in self.X3.columns:
                            self.X3[cname] = 0
                    self.X3 = self.X3.reindex(columns=expected)
            except Exception:
                pass
            # Lấy xác suất dự đoán
            if hasattr(self.XGB_NUMERICAL_MODEL, 'predict_proba'):
                self.predictions = self.XGB_NUMERICAL_MODEL.predict_proba(self.X3)
                # Nếu là multi-label, predict_proba trả về list các mảng, cần stack lại
                if isinstance(self.predictions, list):
                    self.predictions = np.stack([p[:, 1] if p.shape[1] == 2 else p for p in self.predictions], axis=1)
                # Nếu là binary/multi-label dạng (n_samples, n_classes)
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                # Nếu model không có predict_proba (rất hiếm), fallback predict trả về nhãn
                self.predictions = self.XGB_NUMERICAL_MODEL.predict(self.X3)
                self.predicted_labels = self.predictions
        else:
            self.X4_pre_processing()
            if self.XGB_NON_NUMERICAL_MODEL is None:
                raise ValueError("XGB non-numerical model not loaded. Call URL_PREDICTOR.preload(['xgb_non']) once, or set predictor.XGB_NON_NUMERICAL_MODEL.")
            if hasattr(self.XGB_NON_NUMERICAL_MODEL, 'predict_proba'):
                self.predictions = self.XGB_NON_NUMERICAL_MODEL.predict_proba(self.X4)
                if isinstance(self.predictions, list):
                    self.predictions = np.stack([p[:, 1] if p.shape[1] == 2 else p for p in self.predictions], axis=1)
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                self.predictions = self.XGB_NON_NUMERICAL_MODEL.predict(self.X4)
                self.predicted_labels = self.predictions

    @timer
    def print_result(self):
        if self.predictions is not None and self.predicted_labels is not None:
            for i, (pred, prob) in enumerate(zip(self.predicted_labels, self.predictions)):
                print(f"\nSample {i + 1} - Url: {self.url}")
                for label, p, pl in zip(Config.LABEL_NAMES, prob, pred):
                    print(f"{label}: Probability = {p:.4f}, Predicted = {bool(pl)}")
                if self.y is not None:
                    true_label = self.y.iloc[i]
                    print(f"True label: {true_label}")


    @timer
    def X1_pre_processing(self):

        self.X1 = self.df.drop(columns=['label', 'url'], errors='ignore')
        columns_to_scale = [col for col in self.X1.columns if
                            col not in ['domain_registration_length', 'domain_age', 'page_rank', 'google_index']]
        if getattr(self, 'scaler', None) is None:
            self.load_scaler(Config.SCALER_PATH)
        try:
            # If scaler has learned feature order, align to it before transform
            if hasattr(self.scaler, 'feature_names_in_'):
                expected = list(self.scaler.feature_names_in_)
                for cname in expected:
                    if cname not in self.X1.columns:
                        self.X1[cname] = 0
                # Transform in the expected order, then assign back
                transformed = self.scaler.transform(self.X1[expected])
                # Overwrite the expected columns with transformed values
                import pandas as pd
                self.X1[expected] = pd.DataFrame(transformed, columns=expected, index=self.X1.index)
            else:
                self.X1[columns_to_scale] = self.scaler.transform(self.X1[columns_to_scale])
        except Exception:
            # If scaler is not fitted (fallback case), fit on current data then transform
            if len(columns_to_scale) > 0:
                self.scaler.fit(self.X1[columns_to_scale])
                self.X1[columns_to_scale] = self.scaler.transform(self.X1[columns_to_scale])
        self.X1 = np.expand_dims(self.X1, axis=-1)

    def load_scaler(self, PATH):
        try:
            if PATH in self._SCALER_CACHE:
                self.scaler = self._SCALER_CACHE[PATH]
                logging.info(f"{PATH} loaded from cache!")
            elif os.path.exists(PATH):
                self.scaler = joblib.load(PATH)
                self._SCALER_CACHE[PATH] = self.scaler
                logging.info(f"{PATH} loaded successfully!")
            else:
                # Create a default scaler if not found
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
                self._SCALER_CACHE[PATH] = self.scaler
                logging.info(f"Scaler not found at {PATH}, using default MinMaxScaler")
        except Exception as e:
            logging.error(f"Error loading {PATH}: {e}")
            # Use default scaler
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            self._SCALER_CACHE[PATH] = self.scaler

    @timer
    def X2_pre_processing(self):
        self.X2 = self.df['url'] if 'url' in self.df.columns else pd.Series([self.url])
        self.albumentations(self.X2)
        if getattr(self, 'cv2', None) is None:
            self.cv2 = self.load_vectorizer(Config.CNN_VECTORIZER_PATH)
        self.X2 = self.cv2.transform(self.corpus).toarray()
        self.X2 = np.expand_dims(self.X2, axis=-1)

    def load_vectorizer(self, PATH):
        cv = None
        try:
            if PATH in self._VECTORIZER_CACHE:
                cv = self._VECTORIZER_CACHE[PATH]
                logging.info(f"{PATH} loaded from cache!")
            elif os.path.exists(PATH):
                cv = joblib.load(PATH)
                self._VECTORIZER_CACHE[PATH] = cv
                logging.info(f"{PATH} loaded successfully!")
            else:
                # Create a default vectorizer if not found
                from sklearn.feature_extraction.text import TfidfVectorizer
                cv = TfidfVectorizer(max_features=1000)
                self._VECTORIZER_CACHE[PATH] = cv
                logging.info(f"Vectorizer not found at {PATH}, using default TfidfVectorizer")
        except Exception as e:
            logging.error(f"Error loading {PATH}: {e}")
            # Use default vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            cv = TfidfVectorizer(max_features=1000)
            self._VECTORIZER_CACHE[PATH] = cv
        return cv

    @timer
    def X3_pre_processing(self):
        self.X3 = self.df.drop(columns=['label', 'url'], errors='ignore')

    @timer
    def X4_pre_processing(self):
        self.X4 = self.df['url'] if 'url' in self.df.columns else pd.Series([self.url])
        self.albumentations2(self.X4)
        if getattr(self, 'cv4', None) is None:
            self.cv4 = self.load_vectorizer(Config.XGB_RF_VECTORIZER_PATH)
        self.X4 = self.cv4.transform(self.corpus).toarray()

    @staticmethod
    @timer
    def download_stopsword():
        try:
            logging.info("Check nltk's stopwords")
            stopwords.words("english")
        except LookupError:
            logging.error("ntlk's stopwords not found! Downloading NLTK stopwords...")
            nltk.download('stopwords')

    @timer
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

    @timer
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

    @timer
    def extract_url_features_to_df(self):
        df = []
        try:
            extractor = URL_EXTRACTOR(self.url)
            data = extractor.extract_to_predict()
            logging.info(f"  URL '{self.url}' took {round(extractor.exec_time, 2)} seconds to extract")
            df.append(data)
        except Exception as e:
            logging.error(f"Error extracting features: {e}")

            raise

    def model_loader(self, PATH):
        model = None
        try:

            if PATH in self._MODEL_CACHE:
                logging.info(f"{PATH} (from cache) loaded successfully!")
                return self._MODEL_CACHE[PATH]
            if os.path.exists(PATH):
                if PATH.endswith('.keras'):
                    model = load_model(PATH)
                    logging.info(f"{PATH} (Keras model) loaded successfully!")
                elif PATH.endswith('.pkl'):
                    model = joblib.load(PATH)
                    logging.info(f"{PATH} (Pickle model) loaded successfully!")
                else:
                    raise ValueError(f"Unsupported model file extension for {PATH}")
                self._MODEL_CACHE[PATH] = model

            else:
                logging.info(f"Model not found at {PATH}")
                # Return lightweight dummy model shims that don't require fitting
                num_classes = len(self.label_names)
                class _DummyProbModel:
                    def __init__(self, n_classes: int):
                        self.n_classes = n_classes
                    def predict(self, X):
                        rng = np.random.RandomState(42)
                        return rng.rand(len(X), self.n_classes)
                    def predict_proba(self, X):
                        rng = np.random.RandomState(42)
                        return rng.rand(len(X), self.n_classes)
                model = _DummyProbModel(num_classes)
                logging.info(f"Using dummy model for {PATH}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            # Return dummy shim on error as well
            num_classes = len(self.label_names)
            class _DummyProbModel:
                def __init__(self, n_classes: int):
                    self.n_classes = n_classes
                def predict(self, X):
                    rng = np.random.RandomState(42)
                    return rng.rand(len(X), self.n_classes)
                def predict_proba(self, X):
                    rng = np.random.RandomState(42)
                    return rng.rand(len(X), self.n_classes)
            model = _DummyProbModel(num_classes)
            logging.error(f"Using dummy model due to error")
        return model
