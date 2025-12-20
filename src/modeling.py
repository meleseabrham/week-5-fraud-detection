import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name="Model"):
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
