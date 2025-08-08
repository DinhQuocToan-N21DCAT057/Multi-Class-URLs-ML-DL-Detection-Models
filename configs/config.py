import os


class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

    # Firebase configuration
    FIREBASE_CONFIG_PATH = os.environ.get('FIREBASE_CONFIG_PATH', 'firebase-config.json')
    FIREBASE_DATABASE_URL = os.environ.get('FIREBASE_DATABASE_URL', 'https://your-project.firebaseio.com')

    # Model configuration
    DEFAULT_DATASET = os.environ.get('DEFAULT_DATASET', 'dataset_1')
    DEFAULT_MODEL_NAME = os.environ.get('DEFAULT_MODEL_NAME', 'cnn')
    DEFAULT_THRESHOLD = float(os.environ.get('DEFAULT_THRESHOLD', '0.5'))

    # Application settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
    JSONIFY_PRETTYPRINT_REGULAR = True

    # Model paths (relative to project root)
    MODELS_BASE_PATH = os.environ.get('MODELS_BASE_PATH', '../models')

    # Available datasets
    AVAILABLE_DATASETS = ['dataset_1', 'dataset_2', 'dataset_3']

    # Available model types
    AVAILABLE_MODELS = ['cnn', 'xgb', 'rf']

    # Label mappings
    LABEL_NAMES = ['benign', 'defacement', 'malware', 'phishing']
    LABEL_COLORS = {
        'benign': '#28a745',
        'defacement': '#ffc107',
        'malware': '#dc3545',
        'phishing': '#fd7e14'
    }
