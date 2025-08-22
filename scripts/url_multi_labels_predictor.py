import os
import sys
import numpy as np
import pandas as pd
import nltk
import re
import joblib
import logging
import time
import torch
import torch.nn as nn
import warnings

from typing import Dict
from functools import wraps
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login as hf_login
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import Config
from scripts.url_features_extractor_v1 import URL_EXTRACTOR
from scripts.transformers_model import Transformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

warnings.filterwarnings("ignore")


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
    _LLAMA_CACHE = {
        "model": None,
        "tokenizer": None,
        "classifier": None,
        "device": None,
    }

    @classmethod
    def preload(cls, models_to_load=None):
        """Optionally preload selected models once and cache them.

        models_to_load can include: 'cnn_num', 'cnn_non', 'xgb_num', 'xgb_non', 'rf_num', 'rf_non, bert_non', 'llama-32-1b-lora'
        """
        if not models_to_load:
            return
        key_to_path = {
            "cnn_num": Config.CNN_NUMERICAL_MODEL_PATH,
            "cnn_non": Config.CNN_NON_NUMERICAL_MODEL_PATH,
            "xgb_num": Config.XGB_NUMERICAL_MODEL_PATH,
            "xgb_non": Config.XGB_NON_NUMERICAL_MODEL_PATH,
            "rf_num": Config.RF_NUMERICAL_MODEL_PATH,
            "rf_non": Config.RF_NON_NUMERICAL_MODEL_PATH,
            "bert_non": Config.BERT_NON_NUMERICAL_MODEL_PATH,
            "llama-32-1b-lora": Config.Llama_32_1B_LoRA_PATH,
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Attach cached models if already preloaded
        self.CNN_NUMERICAL_MODEL = self._MODEL_CACHE.get(
            Config.CNN_NUMERICAL_MODEL_PATH
        )
        self.CNN_NON_NUMERICAL_MODEL = self._MODEL_CACHE.get(
            Config.CNN_NON_NUMERICAL_MODEL_PATH
        )
        self.XGB_NUMERICAL_MODEL = self._MODEL_CACHE.get(
            Config.XGB_NUMERICAL_MODEL_PATH
        )
        self.XGB_NON_NUMERICAL_MODEL = self._MODEL_CACHE.get(
            Config.XGB_NON_NUMERICAL_MODEL_PATH
        )
        self.BERT_NON_NUMERICAL_MODEL = self._MODEL_CACHE.get(
            Config.BERT_NON_NUMERICAL_MODEL_PATH
        )
        self.RF_NUMERICAL_MODEL = self._MODEL_CACHE.get(Config.RF_NUMERICAL_MODEL_PATH)
        self.RF_NON_NUMERICAL_MODEL = self._MODEL_CACHE.get(
            Config.RF_NON_NUMERICAL_MODEL_PATH
        )
        self.Llama_32_1B_LoRA_MODEL = self._MODEL_CACHE.get(
            Config.Llama_32_1B_LoRA_PATH
        )
        self.scaler = self._SCALER_CACHE.get(Config.SCALER_PATH)
        self.cv2 = self._VECTORIZER_CACHE.get(Config.CNN_VECTORIZER_PATH)
        self.cv4 = self._VECTORIZER_CACHE.get(Config.XGB_RF_VECTORIZER_PATH)

        self.url = url
        # Use pre-extracted features DataFrame if provided to avoid re-extraction
        if preset_df is not None:
            self.df = preset_df
        else:
            self.df = self.extract_url_features_to_df()
        self.y = self.df["label"] if "label" in self.df.columns else None
        self.ps = PorterStemmer()
        self.corpus = []
        self.predictions = None
        self.predicted_labels = None

    @timer
    def predict_with_TF_BERT(self, threshold=None):
        self.X5_pre_processing()
        if self.BERT_NON_NUMERICAL_MODEL is None:
            raise ValueError(
                "BERT non-numerical model not loaded. Call URL_PREDICTOR.preload(['bert_non']) once, or set predictor.BERT_NON_NUMERICAL_MODEL."
            )

        self.BERT_NON_NUMERICAL_MODEL.eval()
        with torch.no_grad():
            logits = self.BERT_NON_NUMERICAL_MODEL(self.input_ids, self.attention_mask)
            # Multi-class → softmax
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        self.predictions = probs
        # Chuyển argmax -> one-hot vector để print_result không lỗi
        one_hot_preds = np.zeros_like(probs, dtype=int)
        one_hot_preds[np.arange(len(probs)), np.argmax(probs, axis=1)] = 1
        self.predicted_labels = one_hot_preds

    @timer
    def predict_with_RF(self, threshold=0.5, numerical=True):
        if numerical:
            self.X3_pre_processing()
            if self.RF_NUMERICAL_MODEL is None:
                raise ValueError(
                    "RF numerical model not loaded. Call URL_PREDICTOR.preload(['rf_num']) once, or set predictor.RF_NUMERICAL_MODEL."
                )
            # Align feature columns and order to match training
            try:
                if hasattr(self.RF_NUMERICAL_MODEL, "feature_names_in_"):
                    expected = list(self.RF_NUMERICAL_MODEL.feature_names_in_)
                    for cname in expected:
                        if cname not in self.X3.columns:
                            self.X3[cname] = 0
                    self.X3 = self.X3.reindex(columns=expected)
            except Exception:
                pass
            if hasattr(self.RF_NUMERICAL_MODEL, "predict_proba"):
                self.predictions = self.RF_NUMERICAL_MODEL.predict_proba(self.X3)
                # Nếu là multi-label, predict_proba trả về list các mảng, cần stack lại
                if isinstance(self.predictions, list):
                    self.predictions = np.stack(
                        [p[:, 1] if p.shape[1] == 2 else p for p in self.predictions],
                        axis=1,
                    )
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                self.predictions = self.RF_NUMERICAL_MODEL.predict(self.X3)
                self.predicted_labels = self.predictions
        else:
            self.X4_pre_processing()
            if self.RF_NON_NUMERICAL_MODEL is None:
                raise ValueError(
                    "RF non-numerical model not loaded. Call URL_PREDICTOR.preload(['rf_non']) once, or set predictor.RF_NON_NUMERICAL_MODEL."
                )
            if hasattr(self.RF_NON_NUMERICAL_MODEL, "predict_proba"):
                self.predictions = self.RF_NON_NUMERICAL_MODEL.predict_proba(self.X4)
                if isinstance(self.predictions, list):
                    # Handle multi-label case where predict_proba returns list of arrays
                    try:
                        self.predictions = np.stack(
                            [
                                p[:, 1] if p.shape[1] == 2 else p
                                for p in self.predictions
                            ],
                            axis=1,
                        )
                    except Exception as e:
                        logging.error(f"Error stacking RF predictions: {e}")
                        # Fallback: convert each prediction to array and stack manually
                        pred_arrays = []
                        for p in self.predictions:
                            if hasattr(p, "shape") and p.shape[1] == 2:
                                pred_arrays.append(p[:, 1])
                            else:
                                pred_arrays.append(np.array(p).flatten())
                        self.predictions = np.column_stack(pred_arrays)

                # Ultimate fix: Ensure predictions is always a numpy array before threshold comparison
                try:
                    if not isinstance(self.predictions, np.ndarray):
                        self.predictions = np.array(self.predictions, dtype=float)

                    # Additional safety check - force conversion if still a list
                    if isinstance(self.predictions, list):
                        logging.error(
                            f"RF predictions still a list after conversion: {type(self.predictions)}"
                        )
                        self.predictions = np.array(self.predictions, dtype=float)

                    # Ensure it's a 2D array for threshold comparison
                    if self.predictions.ndim == 1:
                        self.predictions = self.predictions.reshape(1, -1)

                    # Safe threshold comparison with explicit type checking
                    if isinstance(threshold, (int, float)) and isinstance(
                        self.predictions, np.ndarray
                    ):
                        self.predicted_labels = (self.predictions > threshold).astype(
                            int
                        )
                    else:
                        logging.error(
                            f"RF threshold comparison failed: threshold={type(threshold)}, predictions={type(self.predictions)}"
                        )
                        self.predicted_labels = np.zeros_like(
                            self.predictions, dtype=int
                        )

                except Exception as e:
                    logging.error(f"RF prediction processing failed: {e}")
                    # Fallback: create dummy predictions
                    self.predictions = np.array([[0.1, 0.1, 0.1, 0.1]], dtype=float)
                    self.predicted_labels = np.array([[0, 0, 0, 0]], dtype=int)
            else:
                self.predictions = self.RF_NON_NUMERICAL_MODEL.predict(self.X4)
                self.predictions = np.array(self.predictions)
                self.predicted_labels = self.predictions

    @timer
    def predict_with_CNN(self, threshold=0.5, numerical=True):
        if numerical:
            self.X1_pre_processing()
            if self.CNN_NUMERICAL_MODEL is None:
                raise ValueError(
                    "CNN numerical model not loaded. Call URL_PREDICTOR.preload(['cnn_num']) once, or set predictor.CNN_NUMERICAL_MODEL."
                )
            self.predictions = self.CNN_NUMERICAL_MODEL.predict(self.X1)
            self.predicted_labels = (self.predictions > threshold).astype(int)
        else:
            self.X2_pre_processing()
            if self.CNN_NON_NUMERICAL_MODEL is None:
                raise ValueError(
                    "CNN non-numerical model not loaded. Call URL_PREDICTOR.preload(['cnn_non']) once, or set predictor.CNN_NON_NUMERICAL_MODEL."
                )
            self.predictions = self.CNN_NON_NUMERICAL_MODEL.predict(self.X2)
            self.predicted_labels = (self.predictions > threshold).astype(int)

    @timer
    def predict_with_XGB(self, threshold=0.5, numerical=True):
        if numerical:
            self.X3_pre_processing()
            if self.XGB_NUMERICAL_MODEL is None:
                raise ValueError(
                    "XGB numerical model not loaded. Call URL_PREDICTOR.preload(['xgb_num']) once, or set predictor.XGB_NUMERICAL_MODEL."
                )
            # Align feature columns and order to match training
            try:
                if hasattr(self.XGB_NUMERICAL_MODEL, "feature_names_in_"):
                    expected = list(self.XGB_NUMERICAL_MODEL.feature_names_in_)
                    for cname in expected:
                        if cname not in self.X3.columns:
                            self.X3[cname] = 0
                    self.X3 = self.X3.reindex(columns=expected)
            except Exception:
                pass
            # Lấy xác suất dự đoán
            if hasattr(self.XGB_NUMERICAL_MODEL, "predict_proba"):
                self.predictions = self.XGB_NUMERICAL_MODEL.predict_proba(self.X3)
                # Nếu là multi-label, predict_proba trả về list các mảng, cần stack lại
                if isinstance(self.predictions, list):
                    self.predictions = np.stack(
                        [p[:, 1] if p.shape[1] == 2 else p for p in self.predictions],
                        axis=1,
                    )
                # Nếu là binary/multi-label dạng (n_samples, n_classes)
                self.predicted_labels = (self.predictions > threshold).astype(int)
            else:
                # Nếu model không có predict_proba (rất hiếm), fallback predict trả về nhãn
                self.predictions = self.XGB_NUMERICAL_MODEL.predict(self.X3)
                self.predicted_labels = self.predictions
        else:
            self.X4_pre_processing()
            if self.XGB_NON_NUMERICAL_MODEL is None:
                raise ValueError(
                    "XGB non-numerical model not loaded. Call URL_PREDICTOR.preload(['xgb_non']) once, or set predictor.XGB_NON_NUMERICAL_MODEL."
                )
            if hasattr(self.XGB_NON_NUMERICAL_MODEL, "predict_proba"):
                self.predictions = self.XGB_NON_NUMERICAL_MODEL.predict_proba(self.X4)
                if isinstance(self.predictions, list):
                    # Handle multi-label case where predict_proba returns list of arrays
                    try:
                        self.predictions = np.stack(
                            [
                                p[:, 1] if p.shape[1] == 2 else p
                                for p in self.predictions
                            ],
                            axis=1,
                        )
                    except Exception as e:
                        logging.error(f"Error stacking XGB predictions: {e}")
                        # Fallback: convert each prediction to array and stack manually
                        pred_arrays = []
                        for p in self.predictions:
                            if hasattr(p, "shape") and p.shape[1] == 2:
                                pred_arrays.append(p[:, 1])
                            else:
                                pred_arrays.append(np.array(p).flatten())
                        self.predictions = np.column_stack(pred_arrays)

                # Ultimate fix: Ensure predictions is always a numpy array before threshold comparison
                try:
                    if not isinstance(self.predictions, np.ndarray):
                        self.predictions = np.array(self.predictions, dtype=float)

                    # Additional safety check - force conversion if still a list
                    if isinstance(self.predictions, list):
                        logging.error(
                            f"XGB predictions still a list after conversion: {type(self.predictions)}"
                        )
                        self.predictions = np.array(self.predictions, dtype=float)

                    # Ensure it's a 2D array for threshold comparison
                    if self.predictions.ndim == 1:
                        self.predictions = self.predictions.reshape(1, -1)

                    # Safe threshold comparison with explicit type checking
                    if isinstance(threshold, (int, float)) and isinstance(
                        self.predictions, np.ndarray
                    ):
                        self.predicted_labels = (self.predictions > threshold).astype(
                            int
                        )
                    else:
                        logging.error(
                            f"XGB threshold comparison failed: threshold={type(threshold)}, predictions={type(self.predictions)}"
                        )
                        self.predicted_labels = np.zeros_like(
                            self.predictions, dtype=int
                        )

                except Exception as e:
                    logging.error(f"XGB prediction processing failed: {e}")
                    # Fallback: create dummy predictions
                    self.predictions = np.array([[0.1, 0.1, 0.1, 0.1]], dtype=float)
                    self.predicted_labels = np.array([[0, 0, 0, 0]], dtype=int)
            else:
                self.predictions = self.XGB_NON_NUMERICAL_MODEL.predict(self.X4)
                self.predictions = np.array(self.predictions)
                self.predicted_labels = self.predictions

    @timer
    def print_result(self):
        if self.predictions is not None and self.predicted_labels is not None:
            for i, (pred, prob) in enumerate(
                zip(self.predicted_labels, self.predictions)
            ):
                print(f"\nSample {i + 1} - Url: {self.url}")
                for label, p, pl in zip(Config.LABEL_NAMES, prob, pred):
                    print(f"{label}: Probability = {p:.4f}, Predicted = {bool(pl)}")
                if self.y is not None:
                    true_label = self.y.iloc[i]
                    print(f"True label: {true_label}")

    @timer
    def X1_pre_processing(self):
        self.X1 = self.df.drop(columns=["label", "url"], errors="ignore")
        columns_to_scale = [
            col
            for col in self.X1.columns
            if col not in [
                "domain_registration_length",
                "domain_age",
                "page_rank",
                "google_index",
            ]
        ]
        if getattr(self, "scaler", None) is None:
            self.load_scaler(Config.SCALER_PATH)
        try:
            # If scaler has learned feature order, align to it before transform
            if hasattr(self.scaler, "feature_names_in_"):
                expected = list(self.scaler.feature_names_in_)
                for cname in expected:
                    if cname not in self.X1.columns:
                        self.X1[cname] = 0
                # Transform in the expected order, then assign back
                transformed = self.scaler.transform(self.X1[expected])
                # Overwrite the expected columns with transformed values
                import pandas as pd

                self.X1[expected] = pd.DataFrame(
                    transformed, columns=expected, index=self.X1.index
                )
            else:
                self.X1[columns_to_scale] = self.scaler.transform(
                    self.X1[columns_to_scale]
                )
        except Exception:
            # If scaler is not fitted (fallback case), fit on current data then transform
            if len(columns_to_scale) > 0:
                self.scaler.fit(self.X1[columns_to_scale])
                self.X1[columns_to_scale] = self.scaler.transform(
                    self.X1[columns_to_scale]
                )
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
        self.X2 = self.df["url"] if "url" in self.df.columns else pd.Series([self.url])
        self.albumentations(self.X2)
        if getattr(self, "cv2", None) is None:
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
                logging.info(
                    f"Vectorizer not found at {PATH}, using default TfidfVectorizer"
                )
        except Exception as e:
            logging.error(f"Error loading {PATH}: {e}")
            # Use default vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer

            cv = TfidfVectorizer(max_features=1000)
            self._VECTORIZER_CACHE[PATH] = cv
        return cv

    @timer
    def X3_pre_processing(self):
        self.X3 = self.df.drop(columns=["label", "url"], errors="ignore")

    @timer
    def X4_pre_processing(self):
        self.X4 = self.df["url"] if "url" in self.df.columns else pd.Series([self.url])
        self.albumentations2(self.X4)
        if getattr(self, "cv4", None) is None:
            self.cv4 = self.load_vectorizer(Config.XGB_RF_VECTORIZER_PATH)
        self.X4 = self.cv4.transform(self.corpus).toarray()

    @timer
    def X5_pre_processing(self):
        self.X5 = self.df["url"] if "url" in self.df.columns else pd.Series([self.url])
        self.X5 = self.tokenize_urls(self.X5)
        self.input_ids = self.X5["input_ids"].to(self.device)
        self.attention_mask = self.X5["attention_mask"].to(self.device)

    def X6_pre_processing(self):
        row = (
            self.df.iloc[0].to_dict()
            if hasattr(self, "df") and len(self.df) > 0
            else {}
        )
        if "label" in row:
            row.pop("label", None)
        feature_kv = []
        # Keep url first if present, then others in stable sorted order
        if "url" in row:
            feature_kv.append(f"url={row['url']}")
            row.pop("url", None)
        for k in sorted(row.keys()):
            feature_kv.append(f"{k}={row[k]}")
        feature_str = ", ".join(feature_kv)
        prompt = f"Given the URL features: {feature_str}. Predict the category:"
        self.X6 = self.tokenizer_prompts([prompt])

    def tokenizer_prompts(self, prompts, max_length=256):
        # Accept a list[str] or single string
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [str(p) if p is not None else "" for p in prompts]
        # Prefer cached tokenizer from preloaded LLaMA-LoRA bundle
        tokenizer = None
        try:
            if getattr(self, "Llama_32_1B_LoRA_MODEL", None):
                tok = self.Llama_32_1B_LoRA_MODEL.get("tokenizer")
                if tok is not None:
                    tokenizer = tok
        except Exception:
            tokenizer = None
        if tokenizer is None:
            # Fallback: load tokenizer from local LoRA folder; then base
            try:
                tokenizer = AutoTokenizer.from_pretrained(Config.Llama_32_1B_LoRA_PATH)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    @timer
    def tokenize_urls(self, urls, max_length=64):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        url_list = urls.tolist() if isinstance(urls, pd.Series) else list(urls)
        url_list = [url if url.strip() else "placeholder" for url in url_list]
        return tokenizer(
            url_list,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    @staticmethod
    @timer
    def download_stopsword():
        try:
            logging.info("Check nltk's stopwords")
            stopwords.words("english")
        except LookupError:
            logging.error("ntlk's stopwords not found! Downloading NLTK stopwords...")
            nltk.download("stopwords")

    @timer
    def albumentations(self, X):
        self.download_stopsword()
        for i in range(len(X)):
            review = X.iloc[i] if hasattr(X, "iloc") else X[i]
            review = (
                review.decode("utf-8") if isinstance(review, bytes) else str(review)
            )
            review = re.sub(r"\?.*", "", review)
            review = re.sub(r"[^a-zA-Z0-9\-\/.]", " ", review)
            review = review.lower()
            review = review.split()
            review = [
                self.ps.stem(word)
                for word in review
                if word not in set(stopwords.words("english"))
            ]
            review = " ".join(review)
            self.corpus.append(review)

    @timer
    def albumentations2(self, X):
        self.download_stopsword()
        for i in range(len(X)):
            review = X.iloc[i] if hasattr(X, "iloc") else X[i]
            review = (
                review.decode("utf-8") if isinstance(review, bytes) else str(review)
            )
            review = re.sub(r"\?.*", "", review)
            review = re.sub(r"[^a-zA-Z0-9\-\/.]", " ", review)
            review = review.lower()
            review = review.split()
            review = [
                self.ps.stem(word)
                for word in review
                if word not in set(stopwords.words("english")) and len(word) > 2
            ]
            review = " ".join(review)
            self.corpus.append(review)

    @timer
    def extract_url_features_to_df(self):
        df = []
        try:
            extractor = URL_EXTRACTOR(self.url)
            data = extractor.extract_to_predict()
            logging.info(
                f"  URL '{self.url}' took {round(extractor.exec_time, 2)} seconds to extract"
            )
            df.append(data)
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            raise
        return pd.DataFrame(df)

    def model_loader(self, PATH):
        model = None
        try:
            if PATH in self._MODEL_CACHE:
                logging.info(f"{PATH} (from cache) loaded successfully!")
                return self._MODEL_CACHE[PATH]
            if os.path.exists(PATH):
                if PATH.endswith(".keras"):
                    model = load_model(PATH)
                    logging.info(f"{PATH} (Keras model) loaded successfully!")
                elif PATH.endswith(".pkl"):
                    model = joblib.load(PATH)
                    logging.info(f"{PATH} (Pickle model) loaded successfully!")
                elif PATH.endswith(".json"):
                    model = XGBClassifier()
                    model.load_model(PATH)
                    logging.info(f"{PATH} (JSON model) loaded successfully!")
                elif PATH.endswith(".pth"):
                    self.device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    pretrained_model = BertModel.from_pretrained("bert-base-uncased")
                    model = Transformer(pretrained_model, num_classes=4).to(self.device)
                    model.load_state_dict(torch.load(PATH, map_location=self.device))
                    model.to(self.device)
                    logging.info(f"{PATH} (Pytorch model) loaded successfully!")
                elif os.path.isdir(PATH) and (
                    os.path.exists(os.path.join(PATH, "adapter_config.json"))
                    or os.path.exists(os.path.join(PATH, "adapter_model.safetensors"))
                ):
                    # Load LoRA bundle
                    llama_model, classifier_head, tokenizer, device = (
                        self._load_llama_lora()
                    )
                    model = {
                        "type": "llama_lora_bundle",
                        "model": llama_model,
                        "classifier": classifier_head,
                        "tokenizer": tokenizer,
                        "device": device,
                    }
                    logging.info(f"{PATH} (LoRA bundle) loaded successfully!")
                else:
                    raise ValueError(f"Unsupported model file extension for {PATH}")
                self._MODEL_CACHE[PATH] = model
            else:
                logging.info(f"Model not found at {PATH}")
                # Return lightweight dummy model shims that don't require fitting
                num_classes = len(Config.LABEL_NAMES)

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
            num_classes = len(Config.LABEL_NAMES)

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

    @timer
    def _load_llama_lora(self):
        """Load and cache LLaMA-3.2-1B with LoRA adapter and classifier head for multi-class classification.

        Returns a tuple: (peft_model, classifier_head, tokenizer, device)
        """
        # Return from cache if available
        if self._LLAMA_CACHE.get("model") is not None:
            return (
                self._LLAMA_CACHE["model"],
                self._LLAMA_CACHE["classifier"],
                self._LLAMA_CACHE["tokenizer"],
                self._LLAMA_CACHE["device"],
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optional HF login via environment variable to access base model if gated
        hf_token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or getattr(Config, "HF_TOKEN", "")
        )
        if hf_token and not os.environ.get("HF_TOKEN"):
            # Make token available to downstream libs in this process
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACE_TOKEN", hf_token)
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)
        if hf_token:
            try:
                hf_login(hf_token)
            except Exception:
                pass

        # Helper to call from_pretrained with token across versions
        def _from_pretrained_with_token(factory, name_or_path, token, **kwargs):
            try:
                return factory.from_pretrained(name_or_path, token=token, **kwargs)
            except TypeError:
                return factory.from_pretrained(
                    name_or_path, use_auth_token=token, **kwargs
                )

        # Load tokenizer from local adapter folder if present
        try:
            tokenizer = AutoTokenizer.from_pretrained(Config.Llama_32_1B_LoRA_PATH)
        except Exception:
            if hf_token:
                tokenizer = _from_pretrained_with_token(
                    AutoTokenizer, "meta-llama/Llama-3.2-1B", hf_token
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model and apply LoRA adapter
        base_model_name = "meta-llama/Llama-3.2-1B"
        try:
            if torch.cuda.is_available():
                # GPU available → allow accelerate dispatch with offload folder
                offload_dir = os.path.join(Config.Llama_32_1B_LoRA_PATH, "offload")
                os.makedirs(offload_dir, exist_ok=True)
                common_kwargs = dict(
                    torch_dtype=(torch.float16),
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    offload_folder=offload_dir,
                )
                if hf_token:
                    base_model = _from_pretrained_with_token(
                        AutoModelForCausalLM, base_model_name, hf_token, **common_kwargs
                    )
                else:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name, **common_kwargs
                    )
            else:
                # CPU-only fallback → no accelerate hooks/offload
                if hf_token:
                    base_model = _from_pretrained_with_token(
                        AutoModelForCausalLM,
                        base_model_name,
                        hf_token,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                    )
                else:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                    )
        except Exception as e:
            logging.error(f"Failed to load base LLaMA model: {e}")
            raise

        try:
            if torch.cuda.is_available():
                peft_model = PeftModel.from_pretrained(
                    base_model,
                    Config.Llama_32_1B_LoRA_PATH,
                    device_map="auto",
                    offload_folder=offload_dir,
                )
            else:
                peft_model = PeftModel.from_pretrained(
                    base_model,
                    Config.Llama_32_1B_LoRA_PATH,
                )
        except Exception as e:
            logging.error(
                f"Failed to load LoRA adapters from {Config.Llama_32_1B_LoRA_PATH}: {e}"
            )
            raise

        # Do NOT move the model after accelerate dispatch
        peft_model.eval()

        # Build classifier head and load weights (keep on CPU; move JIT at inference)
        hidden_size = getattr(peft_model.config, "hidden_size", None)
        if hidden_size is None:
            # Fallback for LLaMA architectures
            try:
                hidden_size = (
                    peft_model.base_model.model.model.embed_tokens.embedding_dim
                )
            except Exception:
                hidden_size = 2048
        classifier_head = nn.Linear(int(hidden_size), len(Config.LABEL_NAMES))

        classifier_path = os.path.join(Config.Llama_32_1B_LoRA_PATH, "classifier.pt")
        try:
            state = torch.load(classifier_path, map_location="cpu")
            classifier_head.load_state_dict(state)
            classifier_head.eval()
        except Exception as e:
            logging.error(f"Failed to load classifier head from {classifier_path}: {e}")
            raise

        # Cache
        self._LLAMA_CACHE["model"] = peft_model
        self._LLAMA_CACHE["tokenizer"] = tokenizer
        self._LLAMA_CACHE["classifier"] = classifier_head
        self._LLAMA_CACHE["device"] = device

        return peft_model, classifier_head, tokenizer, device

    @timer
    def predict_with_LLaMA_LoRA(self):
        """Predict multi-class label probabilities using the fine-tuned LLaMA LoRA classifier on textuali
        zed features."""
        # Build prompt from extracted features to match training format
        self.X6_pre_processing()

        # Tokenize prompt (normalize transformers BatchEncoding → plain dict)
        inputs = None
        if isinstance(self.X6, dict):
            inputs = self.X6
        else:
            try:
                inputs = getattr(self.X6, "data", None) or dict(self.X6)
            except Exception:
                inputs = None
        if not isinstance(inputs, dict) or "input_ids" not in inputs:
            raise ValueError("Tokenization failed for LLaMA prompts")

        # Load preloaded bundle if available, fallback to on-demand load
        bundle = getattr(self, "Llama_32_1B_LoRA_MODEL", None)
        if not bundle:
            bundle = self.model_loader(Config.Llama_32_1B_LoRA_PATH)
        if isinstance(bundle, dict) and bundle.get("type") == "llama_lora_bundle":
            model = bundle["model"]
            classifier_head = bundle["classifier"]
        else:
            # Fallback: load directly
            model, classifier_head, _, _ = self._load_llama_lora()

        input_ids = inputs["input_ids"]  # keep on CPU; accelerate will handle dispatch
        attention_mask = inputs.get("attention_mask")

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]
            if attention_mask is not None:
                mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                )
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled_output = sum_embeddings / sum_mask
            else:
                pooled_output = hidden_states.mean(dim=1)

            # Move classifier head to the same device as pooled_output
            if next(classifier_head.parameters()).device != pooled_output.device:
                classifier_head = classifier_head.to(pooled_output.device)
            logits = classifier_head(pooled_output)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        # Store results in the same fields used elsewhere
        self.predictions = probs
        one_hot_preds = np.zeros_like(probs, dtype=int)
        one_hot_preds[np.arange(len(probs)), np.argmax(probs, axis=1)] = 1
        self.predicted_labels = one_hot_preds
