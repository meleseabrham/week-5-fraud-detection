import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve, 
    auc, f1_score, roc_curve, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def prepare_data(df, target_col, features_to_drop=None):
    """
    Separate features from target and drop unnecessary columns.
    """
    if features_to_drop is None:
        features_to_drop = []
    
    X = df.drop(columns=[target_col] + features_to_drop)
    y = df[target_col]
    
    return X, y

def get_preprocessor(X):
    """
    Create a preprocessor for numeric and categorical features.
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    return preprocessor

def get_feature_names(column_transformer):
    """
    Extract feature names from a ColumnTransformer.
    """
    feature_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if name == 'remainder' and transformer == 'drop':
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(columns))
        else:
            feature_names.extend(columns)
    return feature_names

def plot_evaluation(y_test, y_probs, y_pred, model_name="Model"):
    """
    Plot Precision-Recall curve and Confusion Matrix heatmap.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auc_pr = auc(recall, precision)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Precision-Recall Curve
    ax1.plot(recall, precision, label=f'AUC-PR = {auc_pr:.4f}')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(f'Precision-Recall Curve: {model_name}')
    ax1.legend(loc='best')
    ax1.grid(True)
    
    # Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'Confusion Matrix: {model_name}')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importances for a fitted model.
    """
    if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
        clf = model.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            
            # If we have a preprocessor, we need to handle feature names transformation
            # For now, if we don't have exact transformed names, we match by index if possible
            # or just use what we have.
            
            # Sort importances
            indices = np.argsort(importances)[-top_n:]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.show()

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name="Model", plot=True):
    """
    Train a model and evaluate using AUC-PR, F1-Score, and Confusion Matrix.
    """
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    f1 = f1_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auc_pr = auc(recall, precision)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if plot:
        plot_evaluation(y_test, y_probs, y_pred, model_name)
    
    return {
        "model": model,
        "f1": f1,
        "auc_pr": auc_pr,
        "cm": cm
    }

def cross_validate_model(model, X, y, cv=5):
    """
    Perform Stratified K-Fold cross-validation.
    """
    logger.info(f"Performing {cv}-fold Cross-Validation...")
    scoring = ['f1', 'precision', 'recall', 'average_precision']
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    results = cross_validate(model, X, y, cv=skf, scoring=scoring, return_train_score=False)
    
    print(f"\n--- {cv}-Fold Cross-Validation Results ---")
    print(f"Mean F1: {results['test_f1'].mean():.4f} (+/- {results['test_f1'].std():.4f})")
    print(f"Mean AUC-PR: {results['test_average_precision'].mean():.4f} (+/- {results['test_average_precision'].std():.4f})")
    
    return results

def save_model(model, file_path):
    """
    Save a model to a file using joblib.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def load_model(file_path):
    """
    Load a model from a file using joblib.
    """
    try:
        model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
