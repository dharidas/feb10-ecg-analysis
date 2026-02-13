import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import argparse
from scipy.stats import ttest_rel, wilcoxon
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

def train_classifier(X_train, y_train):
    # MLP hyperparams for small but non-linear dataset
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, 
                        early_stopping=True, validation_fraction=0.1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
    }

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
        
        clf = train_classifier(X_train_scaled, y_train_res)
        results[name] = evaluate_model(clf, X_test_scaled, y_test)
        
    return results

def perform_statistical_tests(fft_results, qft_results):
    metric_names = fft_results[0].keys()
    summary = {}
    
    for name in metric_names:
        fft_vals = np.array([r[name] for r in fft_results])
        qft_vals = np.array([r[name] for r in qft_results])
        
        _, p_ttest = ttest_rel(fft_vals, qft_vals)
        if np.all(fft_vals == qft_vals):
            p_wilcoxon = 1.0
        else:
            _, p_wilcoxon = wilcoxon(fft_vals, qft_vals)
            
        summary[name] = {
            'fft_mean': np.mean(fft_vals),
            'fft_std': np.std(fft_vals),
            'qft_mean': np.mean(qft_vals),
            'qft_std': np.std(qft_vals),
            'p_ttest': p_ttest,
            'p_wilcoxon': p_wilcoxon
        }
    return summary

def main():
    parser = argparse.ArgumentParser(description='MLP Iterative 80:20 Patient Split Statistical Analysis')
    parser.add_argument('--iterations', type=int, default=10, help='Number of random splits')
    args = parser.parse_args()
    
    dimensions = [64, 128]
    
    for dim in dimensions:
        print(f"\n" + "="*85)
        print(f"ANALYZING {dim} FEATURES WITH MLP (NN) OVER {args.iterations} ITERATIONS")
        print("="*85)
        
        fft_csv = f'data/qrs_rr_seg_fft_{dim}_30s.csv'
        qft_csv = f'data/qrs_rr_seg_qft_{dim}_30s.csv'
        
        try:
            df_fft, feats_fft = load_feature_data(fft_csv)
            df_qft, feats_qft = load_feature_data(qft_csv)
        except Exception as e:
            print(f"Error loading files for dim {dim}: {e}")
            continue
        
        fft_all = []
        qft_all = []
        
        for i in range(args.iterations):
            if (i+1) % 5 == 0:
                print(f"Iteration {i+1}/{args.iterations}...")
            res = run_iteration(df_fft, df_qft, feats_fft, feats_qft, iteration_seed=42+i)
            fft_all.append(res['FFT'])
            qft_all.append(res['QFT'])
            
        stats = perform_statistical_tests(fft_all, qft_all)
        
        print("\n" + f"{'METRIC':<12} | {'FFT MEAN \u00B1 STD':<20} | {'QFT MEAN \u00B1 STD':<20} | {'TT P-VAL':<8} | {'WIL P-VAL':<8}")
        print("-" * 85)
        for metric, s in stats.items():
            fft_str = f"{s['fft_mean']:.3f} \u00B1 {s['fft_std']:.3f}"
            qft_str = f"{s['qft_mean']:.3f} \u00B1 {s['qft_std']:.3f}"
            sig_t = "*" if s['p_ttest'] < 0.05 else " "
            sig_w = "*" if s['p_wilcoxon'] < 0.05 else " "
            print(f"{metric.capitalize():<12} | {fft_str:<20} | {qft_str:<20} | {s['p_ttest']:.4f}{sig_t} | {s['p_wilcoxon']:.4f}{sig_w}")
        print("-" * 85)

if __name__ == "__main__":
    main()
