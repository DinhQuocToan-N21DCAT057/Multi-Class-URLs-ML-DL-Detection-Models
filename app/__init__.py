import os
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
import firebase_admin
from firebase_admin import credentials
from configs.config import Config

def create_app():
    # Create and configure the app
    app = Flask(__name__, 
               template_folder='../templates',
               static_folder='../static')
    
    # Load configuration
    app.config.from_object(Config)
    app.secret_key = os.environ.get("SESSION_SECRET", "your-secret-key-change-in-production")
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    # Initialize Firebase
    firebase_config_path = os.environ.get('FIREBASE_CONFIG_PATH', 'firebase-config.json')
    if os.path.exists(firebase_config_path):
        try:
            cred = credentials.Certificate(firebase_config_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.environ.get('FIREBASE_DATABASE_URL', 'https://your-project.firebaseio.com')
            })
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
    
    # Register blueprints or routes
    from . import routes
    app.register_blueprint(routes.bp)
    
    return app
