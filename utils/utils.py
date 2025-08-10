import os
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


def format_execution_time(seconds: float) -> str:
    """Format execution time in human readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def format_prediction_confidence(predictions: List[float]) -> List[Dict[str, Any]]:
    """Format prediction results with confidence levels"""
    label_names = ['benign', 'defacement', 'malware', 'phishing']
    label_colors = {
        'benign': '#28a745',
        'defacement': '#ffc107',
        'malware': '#dc3545',
        'phishing': '#fd7e14'
    }

    formatted_results = []
    for i, (label, prob) in enumerate(zip(label_names, predictions)):
        confidence_level = "Cao" if prob > 0.7 else "Trung bình" if prob > 0.4 else "Thấp"
        formatted_results.append({
            'label': label,
            'probability': float(prob),
            'percentage': f"{prob * 100:.1f}%",
            'confidence_level': confidence_level,
            'color': label_colors[label],
            'is_predicted': prob > 0.5
        })

    return formatted_results


def get_overall_safety_assessment(predictions: List[float], threshold: float = 0.5) -> Dict[str, Any]:
    """Determine overall safety assessment from predictions"""
    benign_prob = predictions[0] if len(predictions) > 0 else 0
    malicious_probs = predictions[1:] if len(predictions) > 1 else []

    max_malicious_prob = max(malicious_probs) if malicious_probs else 0
    is_safe = benign_prob > threshold and max_malicious_prob <= threshold

    # Determine risk level
    if is_safe:
        risk_level = "An toàn"
        risk_color = "#28a745"
        risk_icon = "fa-shield-alt"
    elif max_malicious_prob > 0.8:
        risk_level = "Rất nguy hiểm"
        risk_color = "#dc3545"
        risk_icon = "fa-exclamation-triangle"
    elif max_malicious_prob > 0.6:
        risk_level = "Nguy hiểm"
        risk_color = "#fd7e14"
        risk_icon = "fa-exclamation-circle"
    else:
        risk_level = "Đáng ngờ"
        risk_color = "#ffc107"
        risk_icon = "fa-question-circle"

    return {
        'is_safe': is_safe,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'risk_icon': risk_icon,
        'benign_confidence': benign_prob,
        'max_malicious_confidence': max_malicious_prob
    }


def validate_url(url: str) -> tuple[bool, str]:
    """Validate URL format and accessibility"""
    import re
    from urllib.parse import urlparse

    if not url:
        return False, "URL không được để trống"

    # Basic URL format validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if not url_pattern.match(url):
        # Try adding protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            if not url_pattern.match(url):
                return False, "Định dạng URL không hợp lệ"
        else:
            return False, "Định dạng URL không hợp lệ"

    # Parse URL components
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False, "Domain không hợp lệ"
    except Exception:
        return False, "Không thể phân tích URL"

    return True, url


def generate_model_comparison_chart_data(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate data for model comparison charts"""
    chart_data = {
        'labels': ['Benign', 'Defacement', 'Malware', 'Phishing'],
        'datasets': []
    }

    colors = {
        'cnn': '#667eea',
        'xgb': '#764ba2',
        'rf': '#f093fb'
    }

    for model_name, model_results in results.items():
        if 'error' not in model_results and 'predictions' in model_results:
            predictions = model_results['predictions']
            chart_data['datasets'].append({
                'label': model_name.upper(),
                'data': predictions,
                'backgroundColor': colors.get(model_name, '#666666'),
                'borderColor': colors.get(model_name, '#666666'),
                'borderWidth': 2,
                'fill': False
            })

    return chart_data


def calculate_model_performance_metrics(predictions_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate performance metrics from prediction history"""
    model_stats = {
        'cnn': {'total': 0, 'avg_time': 0, 'safe_predictions': 0},
        'xgb': {'total': 0, 'avg_time': 0, 'safe_predictions': 0},
        'rf': {'total': 0, 'avg_time': 0, 'safe_predictions': 0}
    }

    for pred in predictions_history:
        model_name = pred.get('model_name')
        if model_name in model_stats:
            model_stats[model_name]['total'] += 1
            model_stats[model_name]['avg_time'] += pred.get('execution_time', 0)

            # Check if prediction is safe (benign with high confidence)
            predictions = pred.get('predictions', [])
            if predictions and len(predictions) >= 4:
                if predictions[0] > 0.5 and max(predictions[1:]) < 0.5:
                    model_stats[model_name]['safe_predictions'] += 1

    # Calculate averages
    for model_name in model_stats:
        if model_stats[model_name]['total'] > 0:
            model_stats[model_name]['avg_time'] /= model_stats[model_name]['total']
            model_stats[model_name]['safety_rate'] = (
                    model_stats[model_name]['safe_predictions'] /
                    model_stats[model_name]['total'] * 100
            )
        else:
            model_stats[model_name]['safety_rate'] = 0

    return model_stats


def export_predictions_to_csv(predictions: List[Dict[str, Any]]) -> str:
    """Export predictions to CSV format"""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        'URL', 'Model Type', 'Dataset', 'Benign', 'Defacement',
        'Malware', 'Phishing', 'Execution Time', 'Timestamp'
    ])

    # Write data
    for pred in predictions:
        predictions_list = pred.get('predictions', [])
        if len(predictions_list) >= 4:
            writer.writerow([
                pred.get('url', ''),
                pred.get('model_name', ''),
                pred.get('dataset', ''),
                predictions_list[0],
                predictions_list[1],
                predictions_list[2],
                predictions_list[3],
                pred.get('execution_time', 0),
                pred.get('timestamp', '')
            ])

    return output.getvalue()


def clean_old_predictions(predictions: List[Dict[str, Any]], max_age_days: int = 30) -> List[Dict[str, Any]]:
    """Remove predictions older than specified days"""
    from datetime import datetime, timedelta

    cutoff_date = datetime.now() - timedelta(days=max_age_days)

    filtered_predictions = []
    for pred in predictions:
        try:
            pred_date = datetime.fromisoformat(pred.get('timestamp', ''))
            if pred_date >= cutoff_date:
                filtered_predictions.append(pred)
        except ValueError:
            # Keep predictions with invalid timestamps
            filtered_predictions.append(pred)

    return filtered_predictions


def csv_to_json(csv_path: str, json_output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Convert a CSV file to a list of JSON objects.

    Args:
        csv_path: Path to the input CSV file.
        json_output_path: Optional path to write the JSON output. When provided,
            the parsed data will be written to this file using UTF-8 encoding.

    Returns:
        A list of dictionaries where each dict represents a row from the CSV.
    """
    import csv

    with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        data: List[Dict[str, Any]] = [dict(row) for row in reader]

    if json_output_path:
        with open(json_output_path, mode="w", encoding="utf-8") as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)

    return data


def json_to_csv(json_path: str, csv_output_path: Optional[str] = None) -> str:
    """Convert a JSON file (array of objects) to CSV.

    Args:
        json_path: Path to the input JSON file. Expected format is a list of
            objects (dictionaries) with string keys.
        csv_output_path: Optional path to write the CSV output. When provided,
            the CSV will be written using UTF-8 with BOM for Excel compatibility.

    Returns:
        The CSV content as a string. If csv_output_path is provided, the same
        content is also written to that file.
    """
    import csv
    import io

    with open(json_path, mode="r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, list):
        raise ValueError("Input JSON must be a list of objects (dictionaries)")

    # Determine field order: keys from the first item, then union of remaining keys
    fieldnames: List[str] = []
    seen_keys: set[str] = set()
    if loaded:
        for key in loaded[0].keys():
            fieldnames.append(key)
            seen_keys.add(key)
        # Include any additional keys from other items (stable, sorted for determinism)
        extra_keys: set[str] = set()
        for item in loaded[1:]:
            if isinstance(item, dict):
                extra_keys.update(set(item.keys()) - seen_keys)
        for key in sorted(extra_keys):
            fieldnames.append(key)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for item in loaded:
        if not isinstance(item, dict):
            raise ValueError("All elements in the JSON array must be objects (dictionaries)")
        writer.writerow({k: item.get(k, "") for k in fieldnames})

    csv_content = output.getvalue()

    if csv_output_path:
        # Write with BOM for better compatibility with Excel on Windows
        with open(csv_output_path, mode="w", encoding="utf-8-sig", newline="") as out_f:
            out_f.write(csv_content)

    return csv_content