import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, 
                             accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

def load_feature_data(feature_csv):
    df = pd.read_csv(feature_csv)
    return df

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_rf_classifier(X_train, y_train):
    # 'balanced_subsample' is often better for RF as it balances weights per tree/bootstrap
    rf = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2]
    }
    # Using 'average_precision' (PR-AUC) as a more robust metric for imbalanced classes
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='average_precision', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best RF params: {grid.best_params_}")
    return grid.best_estimator_

def find_optimal_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold

def evaluate_classifier(clf, X_test, y_test, threshold=0.5):
    y_probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    if y_probs is not None:
        y_pred = (y_probs >= threshold).astype(int)
    else:
        y_pred = clf.predict(X_test)
    
    print(f"Evaluation Results (Threshold: {threshold:.4f}):")
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    roc_auc = roc_auc_score(y_test, y_probs) if y_probs is not None else None
    pr_auc = average_precision_score(y_test, y_probs) if y_probs is not None else None
    
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR AUC (Average Precision): {pr_auc:.4f}")
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Plot Curves
    if y_probs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PR Curve
        prec, rec, _ = precision_recall_curve(y_test, y_probs)
        ax1.plot(rec, prec, label=f'PR curve (area = {pr_auc:.2f})')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.legend(loc='lower left')
        
        # Distribution of probabilities
        ax2.hist(y_probs[y_test == 0], bins=50, alpha=0.5, label='Normal', color='blue', density=True)
        ax2.hist(y_probs[y_test == 1], bins=50, alpha=0.5, label='Abnormal', color='red', density=True)
        ax2.axvline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold:.2f})')
        ax2.set_xlabel('Predicted Probability of Abnormal')
        ax2.set_ylabel('Density')
        ax2.set_title('Probability Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def run_pipeline(feature_csv, label_column='annotation'):
    print(f"\n--- Running Pipeline for {feature_csv} ---")
    df = load_feature_data(feature_csv)
    
    # Map annotation to 1 (Abnormal) and 0 (Normal)
    y = df[label_column].map({'Abnormal': 1, 'Normal': 0}).values
    
    # Identify feature columns (excluding metadata and raw signal)
    feature_columns = [col for col in df.columns if col not in ['patient_id', 'interval_id', label_column] and not col.startswith('signal_')]
    
    # Ensure numeric
    for col in feature_columns:
        if df[col].dtype == object:
           df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=feature_columns)
    X = df[feature_columns].values
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training with {X.shape[1]} features. Total samples: {len(X)}")
    
    # Optional: We can use SMOTE to balance the training set
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    X_train_scaled, X_test_scaled = scale_features(X_train_res, X_test)
    
    print("Training Random Forest...")
    clf = train_rf_classifier(X_train_scaled, y_train_res)
    
    # Find optimal threshold on the train/val set probabilities to maximize F1
    # Note: Strictly speaking, we should use a separate validation set, 
    # but with cross-validation in GridSearchCV, we use the final model's train probs as a proxy.
    y_train_probs = clf.predict_proba(X_train_scaled)[:, 1]
    optimal_threshold = find_optimal_threshold(y_train_res, y_train_probs)
    
    evaluate_classifier(clf, X_test_scaled, y_test, threshold=optimal_threshold)

def main():
    # Run pipeline for FFT features 
    run_pipeline('data/qrs_rr_seg_fft_30s.csv')

    # Run pipeline for QFT features
    run_pipeline('data/qrs_rr_seg_qft_30s.csv')

if __name__ == "__main__":
    main()
