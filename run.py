import os
import firebase_admin
from firebase_admin import credentials
from app import create_app

# --- Main Application ---
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
