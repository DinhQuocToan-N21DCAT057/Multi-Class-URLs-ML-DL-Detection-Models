import os
import json
import logging
import hashlib
from datetime import date
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import firebase_admin
from firebase_admin import db
import numpy as np
from scripts.url_multi_labels_predictor import URL_PREDICTOR
from utils.utils import json_to_csv
from configs.config import Config

# Create a Blueprint for routes
bp = Blueprint("main", __name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)


def hash_url(url: str) -> str:
    """Generate SHA256 hash of URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def save_prediction_to_firebase(
    url: str,
    features: dict,
    comparison_results: list,
    *,
    numerical: bool = True,
    threshold: float = 0.5,
    model_type: str | None = None,
):
    """Save prediction(s) into Firebase using the expected JSON structure.

    Writes to:
      - prediction_results/<YYYY-MM-DD>/<url_hash>/url
      - prediction_results/<YYYY-MM-DD>/<url_hash>/<MODEL_KEY> {threshold, pred_labels, pred_probs}
      - urls_features/<YYYY-MM-DD>/<url_hash> { url, extracted_features, label }
    MODEL_KEY is determined per result and numerical flag:
      CNN → CNN_NUM/CNN_NON, XGB → XGB_NUM/XGB_NON, RF → RF_NUM/RF_NON.
    """
    try:
        if not firebase_admin._apps:
            return None

        today_key = date.today().isoformat()
        url_hash = hash_url(url)

        # Ensure URL stored under the hash node without clobbering other keys
        pr_base_ref = db.reference(f"prediction_results/{today_key}/{url_hash}")
        pr_base_ref.update({"url": url})

        # Helper to compute MODEL_KEY per result
        def model_key_for(result_model_name: str) -> str:
            name = (result_model_name or "").upper()
            if name.startswith("CNN"):
                return "CNN_NUM" if numerical else "CNN_NON"
            if name.startswith("XGB"):
                return "XGB_NUM" if numerical else "XGB_NON"
            if name.startswith("RF"):
                return "RF_NUM" if numerical else "RF_NON"
            return "TRANS"

        # Normalize predictions and save each model result
        for res in comparison_results or []:
            if not isinstance(res, dict) or res.get("error"):
                continue

            model_name = res.get("model_name") or (model_type or "").upper()
            key = model_key_for(model_name)

            # Normalize labels → list[int]
            labels_val = res.get("predicted_labels")
            labels_list: list[int]
            if isinstance(labels_val, dict):
                labels_list = [int(labels_val.get(k, 0)) for k in Config.LABEL_NAMES]
            else:
                try:
                    arr = np.squeeze(np.array(labels_val)).tolist()
                except Exception:
                    arr = labels_val or [0, 0, 0, 0]
                if isinstance(arr, (list, tuple)):
                    labels_list = [int(arr[i]) if i < len(arr) else 0 for i in range(4)]
                else:
                    labels_list = [0, 0, 0, 0]

            # Normalize probs → list[float]
            probs_val = res.get("probabilities")
            try:
                probs_list = np.squeeze(np.array(probs_val)).tolist()
            except Exception:
                probs_list = probs_val or [0.0, 0.0, 0.0, 0.0]
            if isinstance(probs_list, (int, float)):
                probs_list = [float(probs_list)]
            if isinstance(probs_list, (list, tuple)):
                probs_list = [
                    float(probs_list[i]) if i < len(probs_list) else 0.0
                    for i in range(4)
                ]
            else:
                probs_list = [0.0, 0.0, 0.0, 0.0]

            db.reference(f"prediction_results/{today_key}/{url_hash}/{key}").set(
                {
                    "threshold": float(threshold),
                    "pred_labels": labels_list,
                    "pred_probs": probs_list,
                }
            )

        # Save features snapshot under urls_features
        db.reference(f"urls_features/{today_key}/{url_hash}").set(
            {
                "url": url,
                "extracted_features": {k: str(v) for k, v in (features or {}).items()},
                "label": "Unknown",
            }
        )

        return url_hash
    except Exception as e:
        logging.error(f"Firebase save error: {e}")
        return None


def get_predictions_from_firebase(limit: int = 50):
    """Retrieve recent prediction history from prediction_results with minimal payload.

    Output shape per item:
      {
        url: str,
        comparison_results: [ { model_name, predicted_labels, probabilities } ]
      }
    """
    try:
        if not firebase_admin._apps:
            return []

        results_root = db.reference("prediction_results").get() or {}
        if not isinstance(results_root, dict):
            return []

        items = []
        reached_limit = False
        # Sort dates descending (YYYY-MM-DD lexical order works)
        for date_key in sorted(results_root.keys(), reverse=True):
            by_hash = results_root.get(date_key) or {}
            if not isinstance(by_hash, dict):
                continue
            for url_hash, url_node in by_hash.items():
                if not isinstance(url_node, dict):
                    continue
                url_value = url_node.get("url")
                if not url_value:
                    continue

                comparison_results = []
                for model_key, payload in url_node.items():
                    if model_key == "url" or not isinstance(payload, dict):
                        continue
                    pred_labels_list = payload.get("pred_labels")
                    pred_probs_list = payload.get("pred_probs")
                    if pred_labels_list is None or pred_probs_list is None:
                        continue

                    labels_dict = {
                        k: int(pred_labels_list[i]) if i < len(pred_labels_list) else 0
                        for i, k in enumerate(Config.LABEL_NAMES)
                    }

                    comparison_results.append(
                        {
                            "model_name": model_key,
                            "predicted_labels": labels_dict,
                            "probabilities": pred_probs_list,
                        }
                    )

                if comparison_results:
                    items.append(
                        {
                            "url": url_value,
                            "comparison_results": comparison_results,
                            "prediction_timestamp": f"{date_key}",
                        }
                    )
                    if limit and limit > 0 and len(items) >= limit:
                        reached_limit = True
                        break
            if reached_limit:
                break
        return items
    except Exception as e:
        logging.error(f"Firebase retrieve error: {e}")
        return []


# Page Routes
@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/quick-prediction")
def quick_prediction():
    return render_template("quick_prediction.html")


@bp.route("/multi-model-prediction")
def multi_model_prediction():
    return render_template("multi_model_prediction.html")


@bp.route("/analysis-dashboard")
def analysis_dashboard():
    return render_template("analysis_dashboard.html")


@bp.route("/history")
def history():
    predictions = get_predictions_from_firebase()
    return render_template("history.html", predictions=predictions)


@bp.route("/settings")
def settings():
    return render_template("settings.html")


@bp.route("/model-info")
def model_info():
    return render_template("model_info.html")


# API Routes
@bp.route("/api/predict-url", methods=["POST"])
def predict_url():
    """Single model prediction endpoint, updated for the new schema."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        url = data.get("url", "").strip()
        model_type = data.get("model", "cnn").lower()
        dataset = data.get("dataset", "dataset_1")
        threshold = float(data.get("threshold", 0.5))
        # Handle boolean or string inputs for numerical
        numerical_val = data.get("numerical", True)
        if isinstance(numerical_val, str):
            numerical = numerical_val.lower() == "true"
        else:
            numerical = bool(numerical_val)

        if not url:
            return jsonify({"error": "URL không được cung cấp"}), 400

        # Before any heavy work, check same-day cache in Firebase
        url_hash = hash_url(url)
        today_key = date.today().isoformat()

        model_key_map = {
            "cnn": "CNN_NUM" if numerical else "CNN_NON",
            "xgb": "XGB_NUM" if numerical else "XGB_NON",
            "rf": "RF_NUM" if numerical else "RF_NON",
        }
        model_firebase_key = model_key_map.get(model_type)
        if model_firebase_key is None:
            return jsonify({"error": "Mô hình không hợp lệ"}), 400

        cached = None
        try:
            if firebase_admin._apps:
                cached_ref = db.reference(
                    f"prediction_results/{today_key}/{url_hash}/{model_firebase_key}"
                )
                cached = cached_ref.get()
        except Exception as fe:
            logging.error(f"Firebase read error: {fe}")
            cached = None

        if isinstance(cached, dict) and cached.get("pred_probs") is not None:
            pred_probs_list = cached.get("pred_probs") or [0, 0, 0, 0]
            pred_labels_list = cached.get("pred_labels") or [0, 0, 0, 0]
            labels_dict = {
                k: int(pred_labels_list[i]) if i < len(pred_labels_list) else 0
                for i, k in enumerate(Config.LABEL_NAMES)
            }

            result_payload = {
                "model_name": model_firebase_key,
                "predicted_labels": labels_dict,
                "probabilities": [float(x) for x in pred_probs_list],
            }
            return jsonify({"url": url, "comparison_results": [result_payload]})

        # Not cached → try to reuse features if already extracted today
        preset_df = None
        features = {}
        try:
            if firebase_admin._apps:
                feat_ref = db.reference(f"urls_features/{today_key}/{url_hash}")
                feat_node = feat_ref.get()
                if isinstance(feat_node, dict) and isinstance(
                    feat_node.get("extracted_features"), dict
                ):
                    # Build a one-row DataFrame from stored features
                    import pandas as pd

                    # These are stored as strings; convert to numerics where possible
                    raw = feat_node["extracted_features"].copy()
                    for k, v in list(raw.items()):
                        try:
                            if v is None:
                                continue
                            # Try float, then int fallback
                            fv = float(v)
                            if fv.is_integer():
                                raw[k] = int(fv)
                            else:
                                raw[k] = fv
                        except Exception:
                            pass
                    preset_df = pd.DataFrame([raw])
                    features = raw
        except Exception as fe:
            logging.error(f"Reuse features read error: {fe}")

        # If no preset_df, extract as usual
        if preset_df is None:
            predictor = URL_PREDICTOR(url)
            try:
                df_no_meta = predictor.df.drop(
                    columns=["label", "url"], errors="ignore"
                )
                features = df_no_meta.iloc[0].to_dict() if not df_no_meta.empty else {}
            except Exception:
                features = {}
        else:
            predictor = URL_PREDICTOR(url, preset_df=preset_df)

        if model_type == "cnn":
            predictor.predict_with_CNN(threshold=threshold, numerical=numerical)
        elif model_type == "xgb":
            predictor.predict_with_XGB(threshold=threshold, numerical=numerical)
        elif model_type == "rf":
            predictor.predict_with_RF(threshold=threshold, numerical=numerical)
        else:
            return jsonify({"error": "Mô hình không hợp lệ"}), 400

        # Convert predictions/labels to JSON-safe and UI-expected format
        try:
            probs_list = np.squeeze(np.array(predictor.predictions)).tolist()
        except Exception:
            probs_list = predictor.predictions or [0, 0, 0, 0]
        if isinstance(probs_list, (int, float)):
            probs_list = [float(probs_list)]
        probs_list = [
            float(probs_list[i]) if i < len(probs_list) else 0.0 for i in range(4)
        ]

        try:
            labels_arr = np.squeeze(np.array(predictor.predicted_labels)).tolist()
        except Exception:
            labels_arr = predictor.predicted_labels or [0, 0, 0, 0]
        if not isinstance(labels_arr, (list, tuple)):
            labels_arr = [0, 0, 0, 0]
        labels_dict = {
            k: int(labels_arr[i]) if i < len(labels_arr) else 0
            for i, k in enumerate(Config.LABEL_NAMES)
        }

        # Structure the result to be consistent with multi-model endpoint
        result = {
            "model_name": model_firebase_key,
            "predicted_labels": labels_dict,
            "probabilities": probs_list,
        }

        # Save the single prediction as a list containing one result
        save_prediction_to_firebase(
            url,
            features,
            [result],
            numerical=numerical,
            threshold=threshold,
            model_type=model_type,
        )

        return jsonify({"url": url, "comparison_results": [result]})

    except Exception as e:
        logging.error(f"Single model prediction error: {e}")
        return jsonify({"error": f"Lỗi dự đoán: {str(e)}"}), 500


@bp.route("/api/predict-multi-model", methods=["POST"])
def predict_multi_model():
    """Multi-model prediction endpoint with caching and feature reuse optimization, predicting both numerical and non-numerical."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        url = data.get("url", "").strip()
        threshold = float(data.get("threshold", 0.5))
        dataset = data.get("dataset", "dataset_1")

        if not url:
            return jsonify({"error": "URL không được cung cấp"}), 400

        # Check cache for all models for both numerical and non-numerical
        url_hash = hash_url(url)
        today_key = date.today().isoformat()

        model_key_map = {
            "cnn": ["CNN_NUM", "CNN_NON"],
            "xgb": ["XGB_NUM", "XGB_NON"],
            "rf": ["RF_NUM", "RF_NON"],
        }

        cached_results = []
        missing_models = []

        # Check cache for each model and both numerical/non-numerical
        for model_type in Config.AVAILABLE_MODELS:
            for numerical in [True, False]:
                model_firebase_key = model_key_map.get(model_type)[
                    0 if numerical else 1
                ]
                cached = None
                try:
                    if firebase_admin._apps:
                        cached_ref = db.reference(
                            f"prediction_results/{today_key}/{url_hash}/{model_firebase_key}"
                        )
                        cached = cached_ref.get()
                except Exception as fe:
                    logging.error(f"Firebase read error for {model_firebase_key}: {fe}")
                    cached = None

                if isinstance(cached, dict) and cached.get("pred_probs") is not None:
                    # Found cached result
                    pred_probs_list = cached.get("pred_probs") or [0, 0, 0, 0]
                    pred_labels_list = cached.get("pred_labels") or [0, 0, 0, 0]
                    labels_dict = {
                        k: int(pred_labels_list[i]) if i < len(pred_labels_list) else 0
                        for i, k in enumerate(Config.LABEL_NAMES)
                    }

                    cached_results.append(
                        {
                            "model_name": model_firebase_key,
                            "predicted_labels": labels_dict,
                            "probabilities": [float(x) for x in pred_probs_list],
                        }
                    )
                else:
                    # Need to compute this model for this numerical setting
                    missing_models.append((model_type, numerical))

        # If all models are cached, return immediately
        if not missing_models:
            return jsonify({"url": url, "comparison_results": cached_results})

        # Need to compute some models - check if we can reuse features
        preset_df = None
        features = {}
        try:
            if firebase_admin._apps:
                feat_ref = db.reference(f"urls_features/{today_key}/{url_hash}")
                feat_node = feat_ref.get()
                if isinstance(feat_node, dict) and isinstance(
                    feat_node.get("extracted_features"), dict
                ):
                    # Build a one-row DataFrame from stored features
                    import pandas as pd

                    # These are stored as strings; convert to numerics where possible
                    raw = feat_node["extracted_features"].copy()
                    for k, v in list(raw.items()):
                        try:
                            if v is None:
                                continue
                            # Try float, then int fallback
                            fv = float(v)
                            if fv.is_integer():
                                raw[k] = int(fv)
                            else:
                                raw[k] = fv
                        except Exception:
                            pass
                    preset_df = pd.DataFrame([raw])
                    features = raw
        except Exception as fe:
            logging.error(f"Reuse features read error: {fe}")

        # Initialize predictor (with preset features if available)
        if preset_df is None:
            predictor = URL_PREDICTOR(url)
            try:
                df_no_meta = predictor.df.drop(
                    columns=["label", "url"], errors="ignore"
                )
                features = df_no_meta.iloc[0].to_dict() if not df_no_meta.empty else {}
            except Exception:
                features = {}
        else:
            predictor = URL_PREDICTOR(url, preset_df=preset_df)

        # Compute missing models for both numerical and non-numerical
        new_results = []
        for model_type, numerical in missing_models:
            try:
                if model_type == "cnn":
                    predictor.predict_with_CNN(threshold=threshold, numerical=numerical)
                elif model_type == "xgb":
                    predictor.predict_with_XGB(threshold=threshold, numerical=numerical)
                elif model_type == "rf":
                    predictor.predict_with_RF(threshold=threshold, numerical=numerical)

                # Convert predictions/labels to JSON-safe format
                try:
                    probs_list = np.squeeze(np.array(predictor.predictions)).tolist()
                except Exception:
                    probs_list = predictor.predictions or [0, 0, 0, 0]
                if isinstance(probs_list, (int, float)):
                    probs_list = [float(probs_list)]
                probs_list = [
                    float(probs_list[i]) if i < len(probs_list) else 0.0
                    for i in range(4)
                ]

                try:
                    labels_arr = np.squeeze(
                        np.array(predictor.predicted_labels)
                    ).tolist()
                except Exception:
                    labels_arr = predictor.predicted_labels or [0, 0, 0, 0]
                if not isinstance(labels_arr, (list, tuple)):
                    labels_arr = [0, 0, 0, 0]
                labels_dict = {
                    k: int(labels_arr[i]) if i < len(labels_arr) else 0
                    for i, k in enumerate(Config.LABEL_NAMES)
                }

                model_firebase_key = model_key_map.get(model_type)[
                    0 if numerical else 1
                ]
                result = {
                    "model_name": model_firebase_key,
                    "predicted_labels": labels_dict,
                    "probabilities": probs_list,
                }

                new_results.append(result)

            except Exception as e:
                logging.error(f"Error with {model_type} (numerical={numerical}): {e}")
                new_results.append({"model_name": model_firebase_key, "error": str(e)})

        # Combine cached and new results
        all_results = cached_results + new_results

        # Save new results to Firebase (only the ones we computed)
        if features and new_results:
            for numerical in [True, False]:
                numerical_results = [
                    res
                    for res in new_results
                    if res.get("model_name").endswith("_NUM" if numerical else "_NON")
                ]
                if numerical_results:
                    save_prediction_to_firebase(
                        url,
                        features,
                        numerical_results,
                        numerical=numerical,
                        threshold=threshold,
                    )

        return jsonify({"url": url, "comparison_results": all_results})

    except Exception as e:
        logging.error(f"Multi-model prediction error: {e}")
        return jsonify({"error": f"Lỗi dự đoán đa mô hình: {str(e)}"}), 500


# Error handlers
@bp.app_errorhandler(404)
def not_found_error(error):
    return render_template("404.html"), 404


@bp.app_errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template("500.html"), 500
