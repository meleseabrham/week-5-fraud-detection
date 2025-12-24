import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib

# Add src to path
sys.path.append(os.path.abspath('src'))
from modeling import load_model, get_feature_names

def main():
    # Load model
    model_path = 'models/best_rf_fraud.joblib'
    if not os.path.exists(model_path):
        print("Model not found. Please run scripts/train_models.py first.")
        return
    
    best_rf = joblib.load(model_path)
    
    # Load data
    fraud_path = 'data/processed/fraud_data_engineered.csv'
    fraud_data = pd.read_csv(fraud_path)
    
    # Prepare data
    fraud_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
    X = fraud_data.drop(columns=['class'] + fraud_drop)
    y = fraud_data['class']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Extract components
    preprocessor = best_rf.named_steps['prep']
    clf = best_rf.named_steps['clf']
    feature_names = get_feature_names(preprocessor)
    
    # 1. Feature Importance Baseline
    print("\n--- extracting Feature Importance ---")
    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Top 10 Important Features (Built-in):")
    print(feat_imp_df.head(10))
    
    # Save importance plot
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df.head(10), x='Importance', y='Feature')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    plt.close()
    print("Feature importance plot saved to reports/feature_importance.png")

    # 2. SHAP Analysis
    print("\n--- Generating SHAP Analysis ---")
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    X_test_transformed = np.nan_to_num(X_test_transformed)

    explainer = shap.TreeExplainer(clf)
    
    # Use small subset for faster summary plot
    X_subset = X_test_transformed[:500]
    try:
        # Use simpler approach for summary plot
        shap_values = explainer.shap_values(X_subset)
        
        if isinstance(shap_values, list):
            sv_summary = shap_values[1]
        elif len(shap_values.shape) == 3:
            # (samples, features, classes) or (classes, samples, features)
            if shap_values.shape[0] == 2: # classes first
                sv_summary = shap_values[1]
            else:
                sv_summary = shap_values[:, :, 1]
        else:
            sv_summary = shap_values

        plt.figure()
        shap.summary_plot(sv_summary, X_subset, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('reports/shap_summary_plot.png')
        plt.close()
        print("SHAP Summary Plot saved.")
    except Exception as e:
        print(f"Error generating summary plot: {e}")

    # 3. Individual Predictions
    y_probs = best_rf.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.5).astype(int)
    
    tp_indices = np.where((y_test.values == 1) & (y_pred == 1))[0]
    fp_indices = np.where((y_test.values == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_test.values == 1) & (y_pred == 0))[0]
    
    cases = [('tp', tp_indices), ('fp', fp_indices), ('fn', fn_indices)]
    
    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value

    for name, indices in cases:
        if len(indices) > 0:
            idx = indices[0]
            try:
                # Use explainer(X) for simpler force plots if possible
                # But TreeExplainer might not support it for all RF types
                # Fallback to shap_values
                sv = explainer.shap_values(X_test_transformed[idx:idx+1])
                if isinstance(sv, list):
                    sv_inst = sv[1][0]
                elif len(sv.shape) == 3:
                    if sv.shape[0] == 2: sv_inst = sv[1][0]
                    else: sv_inst = sv[0, :, 1]
                else:
                    sv_inst = sv[0]
                
                shap.save_html(f'reports/shap_{name}_force.html', 
                               shap.force_plot(base_val, sv_inst, X_test_transformed[idx:idx+1], feature_names=feature_names))
                print(f"Force plot for {name} saved.")
            except Exception as e:
                print(f"Error for {name}: {e}")

    # 4. Final Drivers
    print("\n--- Top 5 Drivers of Fraud ---")
    print(feat_imp_df['Feature'].head(5).tolist())

if __name__ == "__main__":
    main()
