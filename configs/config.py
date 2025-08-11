import os


class Config:
    """Application configuration"""

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"

    # Firebase configuration
    FIREBASE_CRED_PATH = os.environ.get(
        "FIREBASE_CRED_PATH",
        os.path.join(
            ROOT_DIR,
            "secrets",
            "multi-labels-urls-firebase-db-firebase-adminsdk-fbsvc-87b7743762.json",
        ),
    )
    FIREBASE_DATABASE_URL = os.environ.get(
        "FIREBASE_DATABASE_URL",
        "https://multi-labels-urls-firebase-db-default-rtdb.asia-southeast1.firebasedatabase.app/",
    )

    # Model configuration
    DEFAULT_DATASET = os.environ.get("DEFAULT_DATASET", "dataset_1")
    DEFAULT_MODEL_TYPE = os.environ.get("DEFAULT_MODEL_TYPE", "cnn")
    DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.5"))

    # Model path
    CNN_NUMERICAL_MODEL_PATH = os.path.join(
        ROOT_DIR,
        "models",
        f"{DEFAULT_DATASET}",
        "CNN_MODEL_ON_NUMERICAL_FEATURES.keras",
    )
    CNN_NON_NUMERICAL_MODEL_PATH = os.path.join(
        ROOT_DIR,
        "models",
        f"{DEFAULT_DATASET}",
        "CNN_MODEL_ON_NON_NUMERICAL_FEATURES.keras",
    )
    XGB_NUMERICAL_MODEL_PATH = os.path.join(
            ROOT_DIR, "models", f"{DEFAULT_DATASET}", "XGB_MODEL_ON_NUMERICAL_FEATURES.pkl"
    )
    XGB_NON_NUMERICAL_MODEL_PATH = os.path.join(
        ROOT_DIR,
        "models",
        f"{DEFAULT_DATASET}",
        "XGB_MODEL_ON_NON_NUMERICAL_FEATURES.pkl",
    )
    RF_NUMERICAL_MODEL_PATH = os.path.join(
        ROOT_DIR, "models", f"{DEFAULT_DATASET}", "RF_MODEL_ON_NUMERICAL_FEATURES.pkl"
    )
    RF_NON_NUMERICAL_MODEL_PATH = os.path.join(
        ROOT_DIR,
        "models",
        f"{DEFAULT_DATASET}",
        "RF_MODEL_ON_NON_NUMERICAL_FEATURES.pkl",
    )
    BERT_NON_NUMERICAL_MODEL_PATH = os.path.join(
        ROOT_DIR,
        "models",
        f"{DEFAULT_DATASET}",
        "BERT_MODEL_ON_NON_NUMERICAL_FEATURES.pth"
    )
    SCALER_PATH = os.path.join(ROOT_DIR, "models", f"{DEFAULT_DATASET}", "scaler.pkl")
    CNN_VECTORIZER_PATH = os.path.join(
        ROOT_DIR, "models", f"{DEFAULT_DATASET}", "tfidf_vectorizer_CNN.pkl"
    )
    XGB_RF_VECTORIZER_PATH = os.path.join(
        ROOT_DIR, "models", f"{DEFAULT_DATASET}", "tfidf_vectorizer_XGB_RF.pkl"
    )

    # Application settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
    JSONIFY_PRETTYPRINT_REGULAR = True

    # Available datasets
    AVAILABLE_DATASETS = ["dataset_1", "dataset_2", "dataset_3"]

    # Available model types
    AVAILABLE_MODELS = ["cnn", "xgb", "rf", "bert"]

    # Label mappings
    LABEL_NAMES = ["benign", "defacement", "malware", "phishing"]
    LABEL_COLORS = {
        "benign": "#28a745",
        "defacement": "#ffc107",
        "malware": "#dc3545",
        "phishing": "#fd7e14",
    }
