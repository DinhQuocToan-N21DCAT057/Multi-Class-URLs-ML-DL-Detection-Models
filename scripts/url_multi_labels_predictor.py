import time
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from scripts.url_features_extractor import URL_EXTRACTOR
import logging

# Get the absolute path of the script's directory to ensure robust file loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class URL_PREDICTOR:
    def __init__(self, url, enable_logging=False):
        self.url = url
        self.exec_time = 0.0
        self.log_level = logging.INFO if enable_logging else logging.WARNING
        logging.getLogger().setLevel(self.log_level)

        # Dataset and model paths
        self.dataset = 'dataset_1' 
        # Correctly go up one directory from 'scripts' to the project root
        self.base_model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", self.dataset
        )

        # Check for model files and set up paths
        self.SCALER_PATH = os.path.join(self.base_model_path, "scaler.pkl")
        self.CNN_NUMERICAL_MODEL_PATH = os.path.join(self.base_model_path, "CNN_MODEL_ON_NUMERICAL_FEATURES.keras")
        self.CNN_NON_NUMERICAL_MODEL_PATH = os.path.join(self.base_model_path, "CNN_MODEL_ON_NON_NUMERICAL_FEATURES.keras")
        self.XGB_NUMERICAL_MODEL_PATH = os.path.join(self.base_model_path, "XGB_MODEL_ON_NUMERICAL_FEATURES.pkl")
        self.XGB_NON_NUMERICAL_MODEL_PATH = os.path.join(self.base_model_path, "XGB_MODEL_ON_NON_NUMERICAL_FEATURES.pkl")
        self.RF_NUMERICAL_MODEL_PATH = os.path.join(self.base_model_path, "RF_MODEL_ON_NUMERICAL_FEATURES.pkl")
        self.RF_NON_NUMERICAL_MODEL_PATH = os.path.join(self.base_model_path, "RF_MODEL_ON_NON_NUMERICAL_FEATURES.pkl")
        self.CNN_VECTORIZER_PATH = os.path.join(self.base_model_path, "tfidf_vectorizer_CNN.pkl")
        self.XGB_RF_VECTORIZER_PATH = os.path.join(self.base_model_path, "tfidf_vectorizer_XGB_RF.pkl")
        
        # Load scaler once during initialization
        self.scaler = self.load_scaler()

        # Initialize attributes
        self.label_names = ['phishing', 'defacement', 'malware', 'benign']
        self.predictions = []
        self.predicted_labels = {}
        self.exec_time = 0

    def load_scaler(self):
        """Loads the scaler from the specified path."""
        try:
            return joblib.load(self.SCALER_PATH)
        except FileNotFoundError:
            print(f"Scaler not found at {self.SCALER_PATH}. Numerical predictions will not be scaled.")
            return None
        except Exception as e:
            print(f"Error loading scaler: {e}")
            return None

    def _format_predictions(self, predictions, threshold=0.5):
        # Ensure predictions are a flat list or array of probabilities
        if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], list):
            predictions = predictions[0]

        predicted_labels = {label: 0 for label in self.label_names}
        has_malicious_label = False

        # Assumes the order is [phishing, defacement, malware]
        # The 'benign' label is handled by exclusion.
        malicious_labels = ['phishing', 'defacement', 'malware']
        for i, label in enumerate(malicious_labels):
            if predictions[i] >= threshold:
                predicted_labels[label] = 1
                has_malicious_label = True

        # If no malicious labels were assigned, mark it as benign
        if not has_malicious_label:
            predicted_labels['benign'] = 1

        self.predicted_labels = predicted_labels
        # Ensure self.predictions is a simple list of floats for JSON serialization
        self.predictions = [float(p) for p in predictions]

    def predict_with_RF(self, threshold=0.5, numerical=True):
        start_time = time.time()
        model_path = self.RF_NUMERICAL_MODEL_PATH if numerical else self.RF_NON_NUMERICAL_MODEL_PATH
        model = self.model_loader(model_path)
        
        if numerical:
            data = self.X1_pre_processing()
            if self.scaler:
                data = self.scaler.transform(data)
        else:
            data = self.X2_pre_processing(self.XGB_RF_VECTORIZER_PATH)

        predictions = model.predict_proba(data)[0] # Get probabilities for the first (only) sample
        self.exec_time = round((time.time() - start_time) * 1000, 4)
        self._format_predictions(predictions, threshold)

    def predict_with_CNN(self, threshold=0.5, numerical=True):
        start_time = time.time()
        model_path = self.CNN_NUMERICAL_MODEL_PATH if numerical else self.CNN_NON_NUMERICAL_MODEL_PATH
        model = self.model_loader(model_path)

        if numerical:
            data = self.X1_pre_processing()
            if self.scaler:
                data = self.scaler.transform(data)
            # Reshape data for Conv1D layer: (batch_size, timesteps, features)
            data = np.expand_dims(data, axis=2)
        else:
            data = self.X2_pre_processing(self.CNN_VECTORIZER_PATH)
            data = data.toarray()
            data = np.expand_dims(data, axis=2)

        predictions = model.predict(data)[0]
        self.exec_time = round((time.time() - start_time) * 1000, 4)
        self._format_predictions(predictions, threshold)

    def predict_with_XGB(self, threshold=0.5, numerical=True):
        start_time = time.time()
        model_path = self.XGB_NUMERICAL_MODEL_PATH if numerical else self.XGB_NON_NUMERICAL_MODEL_PATH
        model = self.model_loader(model_path)

        if numerical:
            data = self.X1_pre_processing()
            if self.scaler:
                data = self.scaler.transform(data)
        else:
            data = self.X2_pre_processing(self.XGB_RF_VECTORIZER_PATH)

        predictions = model.predict_proba(data)[0]
        self.exec_time = round((time.time() - start_time) * 1000, 4)
        self._format_predictions(predictions, threshold)

    def X1_pre_processing(self):
        """Pre-processes the URL for numerical feature extraction."""
        try:
            feature_extractor = URL_EXTRACTOR(self.url)
            features_dict = feature_extractor.extract_to_predict()

            # Remove features not used in the trained model to prevent mismatch
            unused_features = [
                'url', 'label', 'domain_reg_len', 'domain_age',
                'google_index', 'page_rank'
            ]
            for feature in unused_features:
                if feature in features_dict:
                    del features_dict[feature]

            return pd.DataFrame([features_dict])
        except Exception as e:
            print(f"Error during X1 pre-processing: {e}")
            raise

    def X2_pre_processing(self, vectorizer_path):
        """Pre-processes the URL for lexical feature extraction."""
        try:
            feature_extractor = URL_EXTRACTOR(self.url)
            url_string = feature_extractor.url
            vectorizer = joblib.load(vectorizer_path)
            return vectorizer.transform([url_string])
        except Exception as e:
            print(f"Error during X2 pre-processing: {e}")
            raise

    def model_loader(self, PATH):
        """Loads a model from the given path."""
        try:
            print(f"Attempting to load model from: {PATH}")
            if PATH.endswith('.keras'):
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
