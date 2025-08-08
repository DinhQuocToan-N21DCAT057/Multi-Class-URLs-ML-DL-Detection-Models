import os
import json
import logging
import hashlib
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, current_app
import firebase_admin
from firebase_admin import db
import numpy as np
from scripts.url_multi_labels_predictor import URL_PREDICTOR
import time
from urllib.parse import urlparse

# Create a Blueprint for routes
bp = Blueprint('main', __name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def hash_url(url: str) -> str:
    """Generate SHA256 hash of URL."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def get_domain(url: str) -> str:
    """Extract the domain from a URL."""
    return urlparse(url).netloc

# Page Routes
@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/quick-prediction')
def quick_prediction():
    return render_template('quick_prediction.html')

@bp.route('/multi-model-prediction')
def multi_model_prediction():
    return render_template('multi_model_prediction.html')

@bp.route('/analysis-dashboard')
def analysis_dashboard():
    return render_template('analysis_dashboard.html')

@bp.route('/history')
def history():
    predictions = []
    return render_template('history.html', predictions=predictions)

@bp.route('/settings')
def settings():
    return render_template('settings.html')

@bp.route('/model-info')
def model_info():
    return render_template('model_info.html')

# API Routes
@bp.route('/api/predict-url', methods=['POST'])
def predict_url():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON received'}), 400

        url = data.get('url')
        model_type = data.get('model', 'cnn').lower()
        dataset = data.get('dataset', 'dataset_1')
        numerical = data.get('numerical', False)

        # Gracefully handle all incompatible numerical models
        if numerical and model_type in ['cnn', 'rf', 'xgb']:
            error_message = f'The {model_type.upper()} model for numerical features is currently unavailable due to an architectural or feature mismatch. Please try another model or feature type.'
            current_app.logger.error(error_message)
            return jsonify({'error': error_message}), 503  # Service Unavailable

        if not url: return jsonify({'error': 'URL not provided'}), 400

        predictor = URL_PREDICTOR(url)
        model_function_map = {
            'rf': predictor.predict_with_RF,
            'cnn': predictor.predict_with_CNN,
            'xgb': predictor.predict_with_XGB
        }

        if model_type in model_function_map:
            model_function_map[model_type](numerical=numerical)
        else:
            return jsonify({'error': 'Invalid model type specified'}), 400

        prediction_data = {
            'url': url,
            'domain': get_domain(url),
            'predicted_at': int(time.time()),
            'prediction_details': {
                'model_used': model_type,
                'dataset_used': dataset,
                'features_type': 'numerical' if numerical else 'lexical',
                'execution_time_ms': predictor.exec_time,
                'predicted_labels': predictor.predicted_labels,
                'prediction_probabilities': predictor.predictions
            }
        }

        db.reference(f'url_info/{hash_url(url)}').set(prediction_data)

        response_data = {
            'model_name': model_type.upper(),
            'execution_time_ms': predictor.exec_time,
            'predicted_labels': predictor.predicted_labels,
            'probabilities': predictor.predictions
        }

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Prediction error for URL '{url if 'url' in locals() else 'N/A'}': {e}", exc_info=True)
        return jsonify({'model_name': model_type if 'model_type' in locals() else 'Unknown', 'error': str(e)}), 500

@bp.route('/api/predict-multi-model', methods=['POST'])
def predict_multi_model():
    """Multi-model prediction endpoint, updated for the new schema."""
    try:
        data = request.get_json()
        url = data.get('url')
        numerical = data.get('numerical', False)

        if not url:
            return jsonify({'error': 'URL not provided'}), 400

        predictor = URL_PREDICTOR(url)
        results = {}
        
        # Define models to use
        models_to_run = {
            'cnn': predictor.predict_with_CNN,
            'xgb': predictor.predict_with_XGB,
            'rf': predictor.predict_with_RF
        }

        for model_name, model_function in models_to_run.items():
            try:
                model_function(numerical=numerical)
                results[model_name] = {
                    'predicted_labels': predictor.predicted_labels,
                    'probabilities': predictor.predictions,
                    'execution_time_ms': predictor.exec_time
                }
            except Exception as e:
                logging.error(f"Error with {model_name} model: {e}")
                results[model_name] = {'error': str(e)}

        return jsonify({
            'url': url,
            'comparison_results': results
        })

    except Exception as e:
        logging.error(f"Multi-model prediction error: {e}")
        return jsonify({'error': f'Lỗi dự đoán đa mô hình: {str(e)}'}), 500

@bp.route('/api/history')
def api_history():
    predictions = []
    return jsonify(predictions)

@bp.route('/api/stats')
def api_stats():
    """API endpoint for detailed statistics from the new database structure."""
    try:
        history_ref = db.reference('prediction_history')
        predictions = history_ref.get()

        total_predictions = len(predictions) if predictions else 0
        safe_count = 0
        phishing_count = 0
        defacement_count = 0
        malware_count = 0

        for pred_id, pred_data in predictions.items():
            labels = {}
            if pred_data.get('comparison_results'):
                first_result = pred_data['comparison_results'][0]
                labels = first_result.get('predicted_labels', {})
            elif pred_data.get('predicted_labels'):
                labels = pred_data.get('predicted_labels', {})

            if labels.get('benign') == 1:
                safe_count += 1
            if labels.get('phishing') == 1:
                phishing_count += 1
            if labels.get('defacement') == 1:
                defacement_count += 1
            if labels.get('malware') == 1:
                malware_count += 1

        malicious_count = phishing_count + defacement_count + malware_count

        safe_change = "+5%"
        malicious_change = "+10%"
        predictions_change = "+20%"
        time_change = "+30%"

        return jsonify({
            'safe_count': safe_count,
            'malicious_count': malicious_count,
            'total_predictions': total_predictions,
            'avg_response_time': 0,
            'safe_change': safe_change,
            'malicious_change': malicious_change,
            'predictions_change': predictions_change,
            'time_change': time_change,
            'phishing_count': phishing_count,
            'defacement_count': defacement_count,
            'malware_count': malware_count
        })

    except Exception as e:
        print(f"Error fetching stats: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@bp.app_errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@bp.app_errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500
