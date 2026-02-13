import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             f1_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import argparse
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from scipy.stats import ttest_rel, wilcoxon
import warnings

warnings.filterwarnings('ignore')

def load_feature_data(feature_csv):
    df = pd.read_csv(feature_csv)
    # Ensure numeric
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

def train_classifier(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    elif model_type == 'svm':
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    elif model_type == 'xgb':
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        spw = neg_count / pos_count if pos_count > 0 else 1.0
        clf = XGBClassifier(n_estimators=100, scale_pos_weight=spw, random_state=42, n_jobs=-1, eval_metric='logloss')
    elif model_type == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, 
                            early_stopping=True, validation_fraction=0.1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
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
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'specificity': specificity,
    }
    
    if y_probs is not None:
        try:
            results['roc_auc'] = roc_auc_score(y_test, y_probs)
        except ValueError:
            # Handle case where test set contains only one class
            results['roc_auc'] = 0.5
            
    return results

def run_iteration(df_fft, df_qft, features_fft, features_qft, iteration_seed, model_type='rf'):
    # ... (rest stays same)
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
        
        clf = train_classifier(X_train_scaled, y_train_res, model_type=model_type)
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
    parser = argparse.ArgumentParser(description='Iterative 80:20 Patient Split Statistical Analysis')
    parser.add_argument('--iterations', type=int, default=20, help='Number of random splits')
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'svm', 'xgb', 'mlp'], help='Classifier to use')
    args = parser.parse_args()
    
    # dimensions = [32, 64, 128]
    dimensions = [64]
    
    for dim in dimensions:
        print(f"\n" + "="*85)
        print(f"ANALYZING {dim} FEATURES OVER {args.iterations} ITERATIONS (Model: {args.model.upper()})")
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
            res = run_iteration(df_fft, df_qft, feats_fft, feats_qft, iteration_seed=42+i, model_type=args.model)
            fft_all.append(res['FFT'])
            qft_all.append(res['QFT'])
            
        stats = perform_statistical_tests(fft_all, qft_all)
        
        print("\n" + "{:<12} | {:<20} | {:<20} | {:<8} | {:<8}".format('METRIC', 'FFT MEAN \u00B1 STD', 'QFT MEAN \u00B1 STD', 'TT P-VAL', 'WIL P-VAL'))
        print("-" * 88)
        # Ensure consistent order for predictable printing
        ordered_metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall', 'specificity']
        for metric in ordered_metrics:
            if metric not in stats: continue
            s = stats[metric]
            fft_str = f"{s['fft_mean']:.3f} \u00B1 {s['fft_std']:.3f}"
            qft_str = f"{s['qft_mean']:.3f} \u00B1 {s['qft_std']:.3f}"
            sig_t = "*" if s['p_ttest'] < 0.05 else " "
            sig_w = "*" if s['p_wilcoxon'] < 0.05 else " "
            print(f"{metric.capitalize():<12} | {fft_str:<20} | {qft_str:<20} | {s['p_ttest']:.4f}{sig_t} | {s['p_wilcoxon']:.4f}{sig_w}")
        print("-" * 88)

if __name__ == "__main__":
    main()
