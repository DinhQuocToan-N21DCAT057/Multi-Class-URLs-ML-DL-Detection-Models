import os
import firebase_admin
from firebase_admin import credentials
from app import create_app

# --- Firebase Initialization ---
def initialize_firebase():
    """Initializes the Firebase Admin SDK."""
    try:
        # Path to your service account key file
        cred_path = os.path.join(os.path.dirname(__file__), 'configs', 'firebase-credentials.json')
        
        # Your Firebase Realtime Database URL
        db_url = "https://multi-labels-urls-firebase-db-default-rtdb.asia-southeast1.firebasedatabase.app/" # e.g., "https://my-project-12345-default-rtdb.firebaseio.com"

        if not db_url:
            raise ValueError("FIREBASE_DATABASE_URL environment variable not set.")

        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': db_url
            })
            print("Firebase Admin SDK initialized successfully.")
        else:
            print("Firebase credentials file not found. Running without Firebase connection.")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")

# --- Main Application ---
if __name__ == "__main__":
    initialize_firebase()
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
