import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, 
                             accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import argparse
from scipy.stats import ttest_rel, wilcoxon

def load_feature_data(feature_csv):
    df = pd.read_csv(feature_csv)
    return df

def apply_smote(X_train, y_train):
    # Only apply SMOTE if there are at least 2 classes and enough samples
    if len(np.unique(y_train)) < 2:
        return X_train, y_train
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_classifier(X_train, y_train, n_epochs=10):
    # n_epochs is currently a placeholder for RF but could be used for n_estimators 
    clf = RandomForestClassifier(n_estimators=n_epochs, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_fold(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
    }
    return metrics

def run_pipeline(feature_csv, n_epochs=100, label_column='annotation'):
    print(f"\n--- Running LOOCV Pipeline for {feature_csv} (Epochs: {n_epochs}) ---")
    df = load_feature_data(feature_csv)
    
    feature_columns = [col for col in df.columns if col not in ['patient_id', 'interval_id', label_column] and not col.startswith('signal_')]
    for col in feature_columns:
        if df[col].dtype == object:
           df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=feature_columns)
    
    X = df[feature_columns].values
    y = df[label_column].map({'Abnormal': 1, 'Normal': 0}).values
    groups = df['patient_id'].values
    unique_patients = np.unique(groups)
    
    logo = LeaveOneGroupOut()
    fold_metrics = []
    
    print(f"Starting LOOCV over {len(unique_patients)} patients...")
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if len(np.unique(y_train)) < 2:
            # For LOOCV, we might skip if training set is homogeneous
            continue
            
        X_train_res, y_train_res = apply_smote(X_train, y_train)
        X_train_scaled, X_test_scaled = scale_features(X_train_res, X_test)
        
        clf = train_classifier(X_train_scaled, y_train_res, n_epochs=n_epochs)
        metrics = evaluate_fold(clf, X_test_scaled, y_test)
        fold_metrics.append(metrics)
        
        if (fold + 1) % 5 == 0:
            print(f"Completed {fold+1}/{len(unique_patients)} folds...")

    summary = {}
    if fold_metrics:
        metric_names = fold_metrics[0].keys()
        for name in metric_names:
            vals = [m[name] for m in fold_metrics]
            summary[name] = (np.mean(vals), np.std(vals))
            
    print("\n" + "="*40)
    print(f"AGGREGATED RESULTS ({len(fold_metrics)} folds)")
    print("="*40)
    for name, (m, s) in summary.items():
        print(f"{name.capitalize():<12}: {m:.4f} \u00B1 {s:.4f}")
    print("="*40)
    
    return fold_metrics

def perform_statistical_tests(fft_metrics, qft_metrics):
    if not fft_metrics or not qft_metrics:
        print("Insufficient metrics for statistical testing.")
        return

    metric_names = fft_metrics[0].keys()
    
    print("\n" + "="*85)
    print(f"{'METRIC':<15} | {'FFT MEAN':<10} | {'QFT MEAN':<10} | {'T-TEST P':<10} | {'WILCOXON P':<10}")
    print("-" * 85)
    
    for name in metric_names:
        fft_vals = np.array([m[name] for m in fft_metrics])
        qft_vals = np.array([m[name] for m in qft_metrics])
        
        # Paired t-test (parametric)
        _, p_ttest = ttest_rel(fft_vals, qft_vals)
        
        # Wilcoxon signed-rank test (non-parametric)
        if np.all(fft_vals == qft_vals):
            p_wilcoxon = 1.0
        else:
            _, p_wilcoxon = wilcoxon(fft_vals, qft_vals)
        
        fft_mean = np.mean(fft_vals)
        qft_mean = np.mean(qft_vals)
        
        sig_t = "*" if p_ttest < 0.05 else " "
        sig_w = "*" if p_wilcoxon < 0.05 else " "
        
        print(f"{name.capitalize():<15} | {fft_mean:<10.4f} | {qft_mean:<10.4f} | {p_ttest:<10.4f}{sig_t} | {p_wilcoxon:<10.4f}{sig_w}")
    
    print("="*85)
    print("* indicates p < 0.05 (statistically significant)")

def main():
    parser = argparse.ArgumentParser(description='LOOCV ECG Classifier with Statistical Tests')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (estimators)')
    parser.add_argument('--fft_csv', type=str, default='data/qrs_rr_seg_fft_128_30s.csv')
    parser.add_argument('--qft_csv', type=str, default='data/qrs_rr_seg_qft_128_30s.csv')
    args = parser.parse_args()
    
    fft_results = run_pipeline(args.fft_csv, n_epochs=args.epochs)
    qft_results = run_pipeline(args.qft_csv, n_epochs=args.epochs)
    
    perform_statistical_tests(fft_results, qft_results)

if __name__ == "__main__":
    main()
