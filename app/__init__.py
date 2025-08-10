import os
import firebase_admin

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from firebase_admin import credentials
from configs.config import Config
from scripts.url_multi_labels_predictor import URL_PREDICTOR

def create_app():
    # Create and configure the app
    app = Flask(__name__, 
               template_folder='../templates',
               static_folder='../static')
    
    # Load configuration
    app.config.from_object(Config)
    app.secret_key = os.environ.get("SECRET_KEY", Config.SECRET_KEY)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    # Load models
    URL_PREDICTOR.preload(['cnn_num', 'cnn_non', 'xgb_num', 'xgb_non', 'rf_num', 'rf_non', 'bert_non'])
    URL_PREDICTOR.preload_scaler()
    URL_PREDICTOR.preload_vectorizers(['cnn', 'xgb_rf'])

    # Initialize Firebase
    firebase_cred_path = os.environ.get('FIREBASE_CRED_PATH', Config.FIREBASE_CRED_PATH)
    if os.path.exists(firebase_cred_path):
        try:
            cred = credentials.Certificate(firebase_cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.environ.get('FIREBASE_DATABASE_URL', Config.FIREBASE_DATABASE_URL)
            })
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
    
    # Register blueprints or routes
    from . import routes
    app.register_blueprint(routes.bp)
    
    return app
