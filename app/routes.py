import os
import json
import logging
import hashlib
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import firebase_admin
from firebase_admin import db
import numpy as np
from scripts.url_multi_labels_predictor import URL_PREDICTOR

# Create a Blueprint for routes
bp = Blueprint('main', __name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def hash_url(url: str) -> str:
    """Generate SHA256 hash of URL."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def save_prediction_to_firebase(url, features, comparison_results):
    """Save prediction results to the new Firebase structure."""
    try:
        if not firebase_admin._apps:
            return None

        url_hash = hash_url(url)
        
        # 1. Save static URL info (only if it doesn't exist)
        url_info_ref = db.reference(f'url_info/{url_hash}')
        if url_info_ref.get() is None:
            url_info_ref.set({
                'url': url,
                'first_seen_timestamp': datetime.now().isoformat(),
                'extracted_features': features
            })

        # 2. Save the new prediction event to history
        history_ref = db.reference('prediction_history')
        new_prediction = history_ref.push({
            'url_hash': url_hash,
            'prediction_timestamp': datetime.now().isoformat(),
            'server_timestamp': db.ServerValue.TIMESTAMP,
            'comparison_results': comparison_results
        })
        
        return new_prediction.key
    except Exception as e:
        logging.error(f"Firebase save error: {e}")
        return None

def get_predictions_from_firebase(limit=50):
    """Retrieve prediction history from the new Firebase structure."""
    try:
        if not firebase_admin._apps:
            return []

        ref = db.reference('prediction_history')
        predictions = ref.order_by_child('server_timestamp').limit_to_last(limit).get()

        if predictions:
            prediction_list = []
            for key, value in predictions.items():
                value['id'] = key
                url_hash = value.get('url_hash')
                if url_hash:
                    url_data = db.reference(f'url_info/{url_hash}/url').get()
                    value['url'] = url_data or 'URL not found'
                prediction_list.append(value)
            return list(reversed(prediction_list))
        return []
    except Exception as e:
        logging.error(f"Firebase retrieve error: {e}")
        return []

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
    predictions = get_predictions_from_firebase()
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
    """Single model prediction endpoint, updated for the new schema."""
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'Invalid JSON'}), 400

        url = data.get('url', '').strip()
        model_type = data.get('model', 'cnn').lower()
        dataset = data.get('dataset', 'dataset_1')
        threshold = float(data.get('threshold', 0.5))
        numerical = data.get('numerical', 'true').lower() == 'true'

        if not url: return jsonify({'error': 'URL không được cung cấp'}), 400

        predictor = URL_PREDICTOR(url, dataset=dataset)
        features = predictor.get_features_as_dict()

        if model_type == 'cnn':
            predictor.predict_with_CNN(threshold=threshold, numerical=numerical)
        elif model_type == 'xgb':
            predictor.predict_with_XGB(threshold=threshold, numerical=numerical)
        elif model_type == 'rf':
            predictor.predict_with_RF(threshold=threshold, numerical=numerical)
        else:
            return jsonify({'error': 'Mô hình không hợp lệ'}), 400

        # Structure the result to be consistent with multi-model endpoint
        result = {
            'model_name': model_type.upper(),
            'predicted_labels': predictor.predicted_labels,
            'probabilities': predictor.predictions, # Already a list
            'execution_time_ms': predictor.exec_time
        }

        # Save the single prediction as a list containing one result
        save_prediction_to_firebase(url, features, [result])

        return jsonify(result)

    except Exception as e:
        logging.error(f"Single model prediction error: {e}")
        return jsonify({'error': f'Lỗi dự đoán: {str(e)}'}), 500

@bp.route('/api/predict-multi-model', methods=['POST'])
def predict_multi_model():
    """Multi-model prediction endpoint, updated for the new schema."""
    try:
        data = request.get_json()
        if not data: return jsonify({'error': 'Invalid JSON'}), 400

        url = data.get('url', '').strip()
        dataset = data.get('dataset', 'dataset_1')
        threshold = float(data.get('threshold', 0.5))
        numerical = data.get('numerical', 'true').lower() == 'true'

        if not url: return jsonify({'error': 'URL không được cung cấp'}), 400

        # Instantiate predictor once for efficiency
        predictor = URL_PREDICTOR(url, dataset=dataset)
        features = predictor.get_features_as_dict()
        
        comparison_results = []
        model_types = ['cnn', 'xgb', 'rf']

        for model_type in model_types:
            try:
                if model_type == 'cnn': predictor.predict_with_CNN(threshold=threshold, numerical=numerical)
                elif model_type == 'xgb': predictor.predict_with_XGB(threshold=threshold, numerical=numerical)
                elif model_type == 'rf': predictor.predict_with_RF(threshold=threshold, numerical=numerical)

                comparison_results.append({
                    'model_name': model_type.upper(),
                    'predicted_labels': predictor.predicted_labels,
                    'probabilities': predictor.predictions, # Already a list
                    'execution_time_ms': predictor.exec_time
                })

            except Exception as e:
                logging.error(f"Error with {model_type} model: {e}")
                comparison_results.append({'model_name': model_type.upper(), 'error': str(e)})

        if features and comparison_results:
            save_prediction_to_firebase(url, features, comparison_results)

        return jsonify({
            'url': url,
            'comparison_results': comparison_results
        })

    except Exception as e:
        logging.error(f"Multi-model prediction error: {e}")
        return jsonify({'error': f'Lỗi dự đoán đa mô hình: {str(e)}'}), 500

@bp.route('/api/history')
def api_history():
    predictions = get_predictions_from_firebase()
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
            # Determine which set of labels to use (multi-model vs. single)
            labels = {}
            if pred_data.get('comparison_results'):
                # Use the first model's result for simplicity in stats
                first_result = pred_data['comparison_results'][0]
                labels = first_result.get('predicted_labels', {})
            elif pred_data.get('predicted_labels'):
                labels = pred_data.get('predicted_labels', {})

            # Correctly count each category based on its specific label
            if labels.get('benign') == 1:
                safe_count += 1
            if labels.get('phishing') == 1:
                phishing_count += 1
            if labels.get('defacement') == 1:
                defacement_count += 1
            if labels.get('malware') == 1:
                malware_count += 1

        malicious_count = phishing_count + defacement_count + malware_count

        # Simple change calculation (dummy)
        safe_change = "+5%"
        malicious_change = "+10%"
        predictions_change = "+20%"
        time_change = "+30%"

        return jsonify({
            'total_safe': safe_count,
            'total_malicious': malicious_count,
            'total_predictions': total_predictions,
            'avg_response_time': 0, # Placeholder
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
