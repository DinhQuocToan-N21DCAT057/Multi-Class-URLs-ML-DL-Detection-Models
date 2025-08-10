#!/usr/bin/env python3
"""
Model Evaluator for Multi-Label URL Classification
Đánh giá và so sánh hiệu suất các mô hình học sâu và học máy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve, auc,
    f1_score, accuracy_score, hamming_loss
)
from sklearn.preprocessing import label_binarize
import joblib
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, dataset_path="dataset_1"):
        self.dataset_path = dataset_path
        self.models = {}
        self.results = {}
        self.label_names = ['benign', 'defacement', 'malware', 'phishing']
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load all trained models"""
        base_path = os.path.join("models", self.dataset_path)
        
        # CNN Models
        try:
            self.models['CNN_numerical'] = load_model(
                os.path.join(base_path, "CNN_MODEL_ON_NUMERICAL_FEATURES.keras")
            )
            self.models['CNN_non_numerical'] = load_model(
                os.path.join(base_path, "CNN_MODEL_ON_NON_NUMERICAL_FEATURES.keras")
            )
        except Exception as e:
            print(f"Error loading CNN models: {e}")
            
        # XGBoost Models
        try:
            self.models['XGB_numerical'] = joblib.load(
                os.path.join(base_path, "XGB_MODEL_ON_NUMERICAL_FEATURES.pkl")
            )
            self.models['XGB_non_numerical'] = joblib.load(
                os.path.join(base_path, "XGB_MODEL_ON_NON_NUMERICAL_FEATURES.pkl")
            )
        except Exception as e:
            print(f"Error loading XGBoost models: {e}")
            
        # Random Forest Models
        try:
            self.models['RF_numerical'] = joblib.load(
                os.path.join(base_path, "RF_MODEL_ON_NUMERICAL_FEATURES.pkl")
            )
            self.models['RF_non_numerical'] = joblib.load(
                os.path.join(base_path, "RF_MODEL_ON_NON_NUMERICAL_FEATURES.pkl")
            )
        except Exception as e:
            print(f"Error loading Random Forest models: {e}")
            
        # Load scaler and vectorizers
        try:
            self.scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
            self.cnn_vectorizer = joblib.load(os.path.join(base_path, "tfidf_vectorizer_CNN.pkl"))
            self.xgb_rf_vectorizer = joblib.load(os.path.join(base_path, "tfidf_vectorizer_XGB_RF.pkl"))
        except Exception as e:
            print(f"Error loading preprocessing models: {e}")
    
    def evaluate_model(self, model_name, X_test, y_test, features_type='numerical'):
        """Evaluate a single model"""
        try:
            model = self.models.get(f"{model_name}_{features_type}")
            if model is None:
                return None
                
            # Make predictions
            if 'CNN' in model_name:
                predictions = model.predict(X_test)
                if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                    predictions = np.concatenate([1 - predictions, predictions], axis=1)
            else:
                predictions = model.predict_proba(X_test)
                
            # Calculate metrics
            y_pred = (predictions > 0.5).astype(int)
            
            # Multi-label metrics
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_micro = f1_score(y_test, y_pred, average='micro')
            hamming = hamming_loss(y_test, y_pred)
            
            # Per-label metrics
            precision = []
            recall = []
            f1_scores = []
            
            for i in range(len(self.label_names)):
                p, r, f1, _ = precision_recall_fscore_support(
                    y_test[:, i], y_pred[:, i], average='binary'
                )
                precision.append(p)
                recall.append(r)
                f1_scores.append(f1)
            
            return {
                'model_name': f"{model_name}_{features_type}",
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'hamming_loss': hamming,
                'precision': precision,
                'recall': recall,
                'f1_scores': f1_scores,
                'predictions': predictions,
                'y_pred': y_pred
            }
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None
    
    def plot_confusion_matrices(self, results):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for idx, (model_name, result) in enumerate(results.items()):
            if result is None:
                continue
                
            cm = confusion_matrix(
                result['y_test'].flatten(), 
                result['y_pred'].flatten()
            )
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'Confusion Matrix - {model_name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, results):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(15, 10))
        
        for model_name, result in results.items():
            if result is None:
                continue
                
            y_test = result['y_test']
            predictions = result['predictions']
            
            # Plot ROC for each label
            for i, label in enumerate(self.label_names):
                fpr, tpr, _ = roc_curve(y_test[:, i], predictions[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{model_name} - {label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-Label Classification')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pr_curves(self, results):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(15, 10))
        
        for model_name, result in results.items():
            if result is None:
                continue
                
            y_test = result['y_test']
            predictions = result['predictions']
            
            # Plot PR for each label
            for i, label in enumerate(self.label_names):
                precision, recall, _ = precision_recall_curve(y_test[:, i], predictions[:, i])
                pr_auc = auc(recall, precision)
                
                plt.plot(recall, precision, label=f'{model_name} - {label} (PR-AUC = {pr_auc:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Multi-Label Classification')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('pr_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results):
        """Generate comprehensive evaluation report"""
        report = []
        report.append("# Model Evaluation Report")
        report.append("## Multi-Label URL Classification")
        report.append("")
        
        # Summary table
        report.append("## Performance Summary")
        report.append("| Model | F1-Macro | F1-Micro | Hamming Loss |")
        report.append("|-------|----------|----------|--------------|")
        
        for model_name, result in results.items():
            if result is None:
                continue
            report.append(f"| {model_name} | {result['f1_macro']:.3f} | {result['f1_micro']:.3f} | {result['hamming_loss']:.3f} |")
        
        report.append("")
        
        # Per-label analysis
        report.append("## Per-Label Analysis")
        for model_name, result in results.items():
            if result is None:
                continue
                
            report.append(f"### {model_name}")
            report.append("| Label | Precision | Recall | F1-Score |")
            report.append("|-------|-----------|--------|----------|")
            
            for i, label in enumerate(self.label_names):
                report.append(f"| {label} | {result['precision'][i]:.3f} | {result['recall'][i]:.3f} | {result['f1_scores'][i]:.3f} |")
            report.append("")
        
        # Save report
        with open('model_evaluation_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("Evaluation report saved as 'model_evaluation_report.md'")
    
    def compare_models(self, X_test, y_test):
        """Compare all models and generate comprehensive evaluation"""
        print("Starting model evaluation...")
        
        results = {}
        
        # Evaluate all models
        for model_name in ['CNN', 'XGB', 'RF']:
            for features_type in ['numerical', 'non_numerical']:
                result = self.evaluate_model(model_name, X_test, y_test, features_type)
                if result:
                    results[f"{model_name}_{features_type}"] = result
                    print(f"✓ Evaluated {model_name}_{features_type}")
        
        # Generate visualizations
        print("Generating visualizations...")
        self.plot_confusion_matrices(results)
        self.plot_roc_curves(results)
        self.plot_pr_curves(results)
        
        # Generate report
        print("Generating evaluation report...")
        self.generate_report(results)
        
        return results

def main():
    """Main function to run evaluation"""
    evaluator = ModelEvaluator()
    
    # Load test data from balanced_dataset_1.csv
    # Note: This requires pre-processed features and labels
    # The raw dataset needs to be processed through the same pipeline as training
    
    print("Loading test data from balanced_dataset_1.csv...")
    print("Note: This requires pre-processed features and labels.")
    print("The raw dataset needs to be processed through the same pipeline as training.")
    
    # Example implementation (commented out as it requires pre-processed data):
    """
    # Load pre-processed test data
    # These files should be generated from balanced_dataset_1.csv using the same preprocessing pipeline
    
    # Load pre-processed numerical features
    X_test_numerical = pd.read_csv('processed_data/X_test_numerical.csv')
    
    # Load pre-processed non-numerical features (TF-IDF vectors)
    X_test_non_numerical = pd.read_csv('processed_data/X_test_non_numerical.csv')
    
    # Load one-hot encoded labels
    y_test = pd.read_csv('processed_data/y_test_one_hot.csv')
    
    # Scale numerical features using the same scaler used during training
    X_test_numerical_scaled = evaluator.scaler.transform(X_test_numerical)
    
    # Run evaluation
    results = evaluator.compare_models(X_test_numerical_scaled, y_test.values)
    """
    
    print("Please implement test data loading based on your dataset structure")
    print("Example usage:")
    print("evaluator = ModelEvaluator()")
    print("results = evaluator.compare_models(X_test, y_test)")

if __name__ == "__main__":
    main()
