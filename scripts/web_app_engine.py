import hashlib
import firebase_admin
import os
import json
from urllib.parse import urlparse

from firebase_admin import credentials, db
from flask import Flask, jsonify, render_template, url_for, request, redirect, flash

# Get the current directory (scripts folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the project root
project_root = os.path.dirname(current_dir)
# Set template folder path
template_dir = os.path.join(project_root, 'templates')

# Initialize Firebase Admin SDK with your service account credentials
cred = credentials.Certificate(os.path.join(project_root, 'multi-labels-urls-firebase-db-firebase-adminsdk-fbsvc-87b7743762.json'))
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://multi-labels-urls-firebase-db-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Initialize Flask app with correct template folder
app = Flask(__name__, template_folder=template_dir)

# Set a secret key for session management
app.secret_key = 'your-super-secret-key-12345'

def hash_url(url: str) -> str:
    """Hash url using SHA256"""
    try:
        hash_obj = hashlib.sha256()
        hash_obj.update(url.encode('utf-8'))
        return hash_obj.hexdigest()
    except Exception as e:
        raise Exception(f"Error hashing URL: {str(e)}")

def analyze_url_security(url: str) -> dict:
    """Analyze URL security and return results"""
    try:
        parsed_url = urlparse(url)
        
        # Basic security checks
        checks = []
        
        # Check if URL has valid scheme
        if parsed_url.scheme in ['http', 'https']:
            checks.append({
                'status': 'success',
                'message': f'Protocol {parsed_url.scheme.upper()} hợp lệ'
            })
        else:
            checks.append({
                'status': 'danger',
                'message': 'Protocol không hợp lệ'
            })
        
        # Check if domain exists
        if parsed_url.netloc:
            checks.append({
                'status': 'success',
                'message': 'Domain hợp lệ'
            })
        else:
            checks.append({
                'status': 'danger',
                'message': 'Domain không hợp lệ'
            })
        
        # Check for suspicious patterns
        suspicious_keywords = ['login', 'bank', 'secure', 'verify', 'update', 'account']
        url_lower = url.lower()
        suspicious_found = any(keyword in url_lower for keyword in suspicious_keywords)
        
        if suspicious_found:
            checks.append({
                'status': 'warning',
                'message': 'Có từ khóa đáng ngờ trong URL'
            })
        else:
            checks.append({
                'status': 'success',
                'message': 'Không có từ khóa đáng ngờ'
            })
        
        # Determine overall safety
        danger_count = sum(1 for check in checks if check['status'] == 'danger')
        warning_count = sum(1 for check in checks if check['status'] == 'warning')
        
        is_safe = danger_count == 0 and warning_count <= 1
        
        return {
            'url': url,
            'is_safe': is_safe,
            'details': checks,
            'hash': hash_url(url)
        }
        
    except Exception as e:
        return {
            'url': url,
            'is_safe': False,
            'details': [{
                'status': 'danger',
                'message': f'Lỗi phân tích: {str(e)}'
            }],
            'hash': hash_url(url)
        }

@app.route('/')
def dashboard():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    """Render the analysis page"""
    return render_template('analysis.html')

@app.route('/history')
def history():
    """Render the history page"""
    return render_template('history.html')

@app.route('/settings')
def settings():
    """Render the settings page"""
    return render_template('settings.html')

@app.route('/predict', methods=['POST'])
def predict_url():
    """API endpoint for URL prediction"""
    try:
        # Get URL from request
        if request.is_json:
            data = request.get_json()
            url = data.get('url', '')
        else:
            url = request.form.get('url', '')
        
        if not url:
            return jsonify({
                'error': 'URL không được cung cấp'
            }), 400
        
        # Analyze URL
        result = analyze_url_security(url)
        
        # Store result in Firebase (optional)
        try:
            ref = db.reference('url_analysis')
            ref.push({
                'url': url,
                'hash': result['hash'],
                'is_safe': result['is_safe'],
                'timestamp': firebase_admin.db.ServerValue.TIMESTAMP
            })
        except Exception as e:
            print(f"Firebase storage error: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': f'Lỗi xử lý: {str(e)}'
        }), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
