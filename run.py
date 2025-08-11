import os
import firebase_admin
from firebase_admin import credentials
from app import create_app
from flask_cors import CORS

# --- Main Application ---
if __name__ == "__main__":
    print("Hello")
    app = create_app()
    CORS(app)
    app.run(debug=True, host='0.0.0.0', port=5000)
