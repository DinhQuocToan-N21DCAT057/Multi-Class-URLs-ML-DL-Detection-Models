# import os
# import json
# import logging
# import hashlib
# from datetime import datetime
# from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
# import firebase_admin
# from firebase_admin import credentials, db
# from werkzeug.middleware.proxy_fix import ProxyFix
# from url_multi_labels_predictor import URL_PREDICTOR
# from config import Config
# import numpy as np

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

# # Create Flask app
# app = Flask(__name__)
# app.secret_key = os.environ.get("SESSION_SECRET", "your-secret-key-change-in-production")
# app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# # Load configuration
# app.config.from_object(Config)

# # Initialize Firebase
# firebase_config_path = os.environ.get('FIREBASE_CONFIG_PATH', 'firebase-config.json')
# if os.path.exists(firebase_config_path):
#     try:
#         cred = credentials.Certificate(firebase_config_path)
#         firebase_admin.initialize_app(cred, {
#             'databaseURL': os.environ.get('FIREBASE_DATABASE_URL', 'https://your-project.firebaseio.com')
#         })
#         logging.info("Firebase initialized successfully")
#     except Exception as e:
#         logging.error(f"Firebase initialization failed: {e}")
# else:
#     logging.warning("Firebase config not found, running without Firebase")


# def hash_url(url: str) -> str:
#     """Generate SHA256 hash of URL"""
#     return hashlib.sha256(url.encode('utf-8')).hexdigest()


# def save_prediction_to_firebase(url, predictions, model_type, dataset):
#     """Save prediction results to Firebase"""
#     try:
#         if not firebase_admin._apps:
#             return False

#         ref = db.reference('predictions')
#         prediction_data = {
#             'url': url,
#             'url_hash': hash_url(url),
#             'model_type': model_type,
#             'dataset': dataset,
#             'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
#             'timestamp': datetime.now().isoformat(),
#             'server_timestamp': db.ServerValue.TIMESTAMP
#         }
#         ref.push(prediction_data)
#         return True
#     except Exception as e:
#         logging.error(f"Firebase save error: {e}")
#         return False


# def get_predictions_from_firebase(limit=50):
#     """Retrieve prediction history from Firebase"""
#     try:
#         if not firebase_admin._apps:
#             return []

#         ref = db.reference('predictions')
#         predictions = ref.order_by_child('server_timestamp').limit_to_last(limit).get()

#         if predictions:
#             # Convert to list and reverse to show newest first
#             prediction_list = []
#             for key, value in predictions.items():
#                 value['id'] = key
#                 prediction_list.append(value)
#             return list(reversed(prediction_list))
#         return []
#     except Exception as e:
#         logging.error(f"Firebase retrieve error: {e}")
#         return []


# @app.route('/')
# def index():
#     """Main dashboard page"""
#     return render_template('index.html')


# @app.route('/quick-prediction')
# def quick_prediction():
#     """Quick prediction page using best model"""
#     return render_template('quick_prediction.html')


# @app.route('/multi-model-prediction')
# def multi_model_prediction():
#     """Multi-model comparison page"""
#     return render_template('multi_model_prediction.html')


# @app.route('/analysis-dashboard')
# def analysis_dashboard():
#     """Analysis dashboard with metrics"""
#     return render_template('analysis_dashboard.html')


# @app.route('/history')
# def history():
#     """Prediction history page"""
#     predictions = get_predictions_from_firebase()
#     return render_template('history.html', predictions=predictions)


# @app.route('/settings')
# def settings():
#     """Settings page"""
#     return render_template('settings.html')


# @app.route('/model-info')
# def model_info():
#     """Model information and metrics page"""
#     return render_template('model_info.html')


# @app.route('/predict', methods=['POST'])
# def predict_url():
#     """Single model prediction endpoint"""
#     try:
#         data = request.get_json() if request.is_json else request.form
#         url = data.get('url', '').strip()
#         model_type = data.get('model_type', 'cnn')
#         dataset = data.get('dataset', 'dataset_1')
#         threshold = float(data.get('threshold', 0.5))
#         numerical = data.get('numerical', 'true').lower() == 'true'

#         if not url:
#             return jsonify({'error': 'URL không được cung cấp'}), 400

#         # Create predictor instance
#         predictor = URL_PREDICTOR(url, dataset=dataset)

#         # Run prediction based on model type
#         if model_type == 'cnn':
#             predictor.predict_with_CNN(threshold=threshold, numerical=numerical)
#         elif model_type == 'xgb':
#             predictor.predict_with_XGB(threshold=threshold, numerical=numerical)
#         elif model_type == 'rf':
#             predictor.predict_with_RF(threshold=threshold, numerical=numerical)
#         else:
#             return jsonify({'error': 'Loại mô hình không hợp lệ'}), 400

#         # Format results
#         results = {
#             'url': url,
#             'model_type': model_type,
#             'dataset': dataset,
#             'threshold': threshold,
#             'numerical': numerical,
#             'predictions': predictor.predictions.tolist() if hasattr(predictor.predictions,
#                                                                      'tolist') else predictor.predictions,
#             'predicted_labels': predictor.predicted_labels.tolist() if hasattr(predictor.predicted_labels,
#                                                                                'tolist') else predictor.predicted_labels,
#             'label_names': predictor.label_names,
#             'execution_time': getattr(predictor, 'exec_time', 0)
#         }

#         # Save to Firebase
#         save_prediction_to_firebase(url, predictor.predictions, model_type, dataset)

#         return jsonify(results)

#     except Exception as e:
#         logging.error(f"Prediction error: {e}")
#         return jsonify({'error': f'Lỗi dự đoán: {str(e)}'}), 500


# @app.route('/predict-multi', methods=['POST'])
# def predict_multi_model():
#     """Multi-model prediction endpoint"""
#     try:
#         data = request.get_json() if request.is_json else request.form
#         url = data.get('url', '').strip()
#         dataset = data.get('dataset', 'dataset_1')
#         threshold = float(data.get('threshold', 0.5))
#         numerical = data.get('numerical', 'true').lower() == 'true'

#         if not url:
#             return jsonify({'error': 'URL không được cung cấp'}), 400

#         results = {}
#         model_types = ['cnn', 'xgb', 'rf']

#         for model_type in model_types:
#             try:
#                 predictor = URL_PREDICTOR(url, dataset=dataset)

#                 if model_type == 'cnn':
#                     predictor.predict_with_CNN(threshold=threshold, numerical=numerical)
#                 elif model_type == 'xgb':
#                     predictor.predict_with_XGB(threshold=threshold, numerical=numerical)
#                 elif model_type == 'rf':
#                     predictor.predict_with_RF(threshold=threshold, numerical=numerical)

#                 results[model_type] = {
#                     'predictions': predictor.predictions.tolist() if hasattr(predictor.predictions,
#                                                                              'tolist') else predictor.predictions,
#                     'predicted_labels': predictor.predicted_labels.tolist() if hasattr(predictor.predicted_labels,
#                                                                                        'tolist') else predictor.predicted_labels,
#                     'execution_time': getattr(predictor, 'exec_time', 0)
#                 }

#                 # Save each model result to Firebase
#                 save_prediction_to_firebase(url, predictor.predictions, model_type, dataset)

#             except Exception as e:
#                 logging.error(f"Error with {model_type} model: {e}")
#                 results[model_type] = {'error': str(e)}

#         return jsonify({
#             'url': url,
#             'dataset': dataset,
#             'threshold': threshold,
#             'numerical': numerical,
#             'results': results,
#             'label_names': ['benign', 'defacement', 'malware', 'phishing']
#         })

#     except Exception as e:
#         logging.error(f"Multi-model prediction error: {e}")
#         return jsonify({'error': f'Lỗi dự đoán đa mô hình: {str(e)}'}), 500


# @app.route('/api/history')
# def api_history():
#     """API endpoint for prediction history"""
#     predictions = get_predictions_from_firebase()
#     return jsonify(predictions)


# @app.route('/api/stats')
# def api_stats():
#     """API endpoint for statistics"""
#     try:
#         predictions = get_predictions_from_firebase(limit=1000)

#         total_predictions = len(predictions)
#         safe_count = 0
#         malicious_count = 0

#         for pred in predictions:
#             if 'predicted_labels' in pred:
#                 labels = pred['predicted_labels']
#                 # If benign label (index 0) is 1 and others are 0, it's safe
#                 if isinstance(labels, list) and len(labels) >= 4:
#                     if labels[0] == 1 and sum(labels[1:]) == 0:
#                         safe_count += 1
#                     else:
#                         malicious_count += 1

#         return jsonify({
#             'total_predictions': total_predictions,
#             'safe_count': safe_count,
#             'malicious_count': malicious_count,
#             'models_used': {
#                 'cnn': len([p for p in predictions if p.get('model_type') == 'cnn']),
#                 'xgb': len([p for p in predictions if p.get('model_type') == 'xgb']),
#                 'rf': len([p for p in predictions if p.get('model_type') == 'rf'])
#             }
#         })
#     except Exception as e:
#         logging.error(f"Stats error: {e}")
#         return jsonify({'error': str(e)}), 500


# @app.errorhandler(404)
# def not_found_error(error):
#     return render_template('404.html'), 404


# @app.errorhandler(500)
# def internal_error(error):
#     return render_template('500.html'), 500


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
