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
    if not os.path.exists(feature_csv):
        raise FileNotFoundError(f"CSV not found: {feature_csv}")
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

def run_iteration(df_qft, features_qft, iteration_seed):
    unique_patients = df_qft['patient_id'].unique()
    train_p, test_p = train_test_split(unique_patients, test_size=0.2, random_state=iteration_seed)
    
    label_col = 'annotation'
    train_df = df_qft[df_qft['patient_id'].isin(train_p)]
    test_df = df_qft[df_qft['patient_id'].isin(test_p)]
    
    X_train = train_df[features_qft].values
    X_test = test_df[features_qft].values
    y_train = train_df[label_col].map({'Abnormal': 1, 'Normal': 0}).values
    y_test = test_df[label_col].map({'Abnormal': 1, 'Normal': 0}).values
    
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    X_train_scaled, X_test_scaled = scale_features(X_train_res, X_test)
    
    clf = train_mlp(X_train_scaled, y_train_res)
    return evaluate_model(clf, X_test_scaled, y_test)

def main():
    qubits = [4, 5, 6, 7]
    dims = [16, 32, 64, 128]
    iterations = 10
    output_dir = 'results/qft_qubit_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    summary_results = [] # List of dicts: {'qubits': 5, 'accuracy': ..., ...}

    for q, d in zip(qubits, dims):
        print(f"\nAnalyzing QFT with {q} Qubits ({d} features)...")
        csv_path = f'data/qrs_rr_seg_qft_{d}_30s.csv'
        
        try:
            df, feats = load_feature_data(csv_path)
            iter_results = []
            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}...")
                res = run_iteration(df, feats, iteration_seed=42+i)
                iter_results.append(res)
            
            # Mean results for this qubit count
            mean_res = {'qubits': q}
            for metric in iter_results[0].keys():
                mean_res[metric] = np.mean([r[metric] for r in iter_results])
            summary_results.append(mean_res)
            
        except Exception as e:
            print(f"Error processing {d} features: {e}")

    if not summary_results:
        print("No results to plot.")
        return

    df_summary = pd.DataFrame(summary_results)
    metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall', 'specificity']

    # Print text table
    print("\n" + "="*80)
    print(f"{'QUBITS':<8} | {'DIM':<5} | {'ACC':<6} | {'AUC':<6} | {'F1':<6} | {'PREC':<6} | {'RECALL':<6} | {'SPEC':<6}")
    print("-" * 80)
    for i, row in df_summary.iterrows():
        d = dims[i]
        print(f"{int(row['qubits']):<8} | {d:<5} | {row['accuracy']:.3f} | {row['roc_auc']:.3f} | {row['f1']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['specificity']:.3f}")
    print("="*80 + "\n")

    plt.figure(figsize=(12, 8))
    for metric in metrics:
        plt.plot(df_summary['qubits'], df_summary[metric], marker='o', label=metric.capitalize())

    plt.title('MLP Performance vs QFT Qubit Count', fontsize=16)
    plt.xlabel('Number of Qubits (log2 of features)', fontsize=14)
    plt.ylabel('Mean Score (10 Iterations)', fontsize=14)
    plt.xticks(qubits)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'qft_qubit_metric_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved comparison plot: {plot_path}")
    
    # Save a CSV of the means
    csv_out = os.path.join(output_dir, 'qft_qubit_means.csv')
    df_summary.to_csv(csv_out, index=False)
    print(f"Saved summary data: {csv_out}")

if __name__ == "__main__":
    main()
