import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, 
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

def load_feature_data(feature_csv):
    df = pd.read_csv(feature_csv)
    label_column = 'annotation'
    feature_columns = [col for col in df.columns if col not in ['patient_id', 'interval_id', label_column] and not col.startswith('signal_')]
    for col in feature_columns:
        if df[col].dtype == object:
           df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=feature_columns)
    return df, feature_columns

def apply_smote(X_train, y_train):
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

def train_mlp(X_train, y_train):
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, 
                        early_stopping=True, validation_fraction=0.1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_probs) if y_probs is not None else 0.5,
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'specificity': specificity,
    }
    return results

def run_iteration(df_fft, df_qft, features_fft, features_qft, iteration_seed):
    unique_patients = df_fft['patient_id'].unique()
    train_p, test_p = train_test_split(unique_patients, test_size=0.2, random_state=iteration_seed)
    
    results = {}
    label_col = 'annotation'
    
    for name, df, feats in [('FFT', df_fft, features_fft), ('QFT', df_qft, features_qft)]:
        train_df = df[df['patient_id'].isin(train_p)]
        test_df = df[df['patient_id'].isin(test_p)]
        
        X_train = train_df[feats].values
        X_test = test_df[feats].values
        y_train = train_df[label_col].map({'Abnormal': 1, 'Normal': 0}).values
        y_test = test_df[label_col].map({'Abnormal': 1, 'Normal': 0}).values
        
        X_train_res, y_train_res = apply_smote(X_train, y_train)
        X_train_scaled, X_test_scaled = scale_features(X_train_res, X_test)
        
        clf = train_mlp(X_train_scaled, y_train_res)
        results[name] = evaluate_model(clf, X_test_scaled, y_test)
        
    return results

def main():
    iterations = 10
    dim = 64
    output_dir = 'results/mlp_64_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running MLP 64 Features Analysis for {iterations} iterations...")
    
    fft_csv = f'data/qrs_rr_seg_fft_{dim}_30s.csv'
    qft_csv = f'data/qrs_rr_seg_qft_{dim}_30s.csv'
    
    df_fft, feats_fft = load_feature_data(fft_csv)
    df_qft, feats_qft = load_feature_data(qft_csv)
    
    fft_history = []
    qft_history = []
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        res = run_iteration(df_fft, df_qft, feats_fft, feats_qft, iteration_seed=42+i)
        fft_history.append(res['FFT'])
        qft_history.append(res['QFT'])
        
    metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall', 'specificity']
    iters = np.arange(1, iterations + 1)
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        fft_vals = [h[metric] for h in fft_history]
        qft_vals = [h[metric] for h in qft_history]
        
        plt.plot(iters, fft_vals, marker='o', label='FFT', linestyle='--', color='blue')
        plt.plot(iters, qft_vals, marker='s', label='QFT', linestyle='-', color='red')
        
        plt.title(f'MLP 64 Features: {metric.capitalize()} Comparison', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        plt.xticks(iters)
        
        plot_path = os.path.join(output_dir, f'mlp_64_{metric}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

    print("\nProcessing complete. Graphs available in results/mlp_64_plots/")

if __name__ == "__main__":
    main()
