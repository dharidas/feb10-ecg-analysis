import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import SMOTE
import argparse
from scipy.stats import ttest_rel, wilcoxon
import warnings
import time

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

def scale_and_normalize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Amplitude embedding requires normalized state vectors (sum of squares = 1)
    X_train_norm = normalize(X_train_scaled, norm='l2')
    X_test_norm = normalize(X_test_scaled, norm='l2')
    
    return X_train_norm, X_test_norm

def create_qml_model(n_qubits, n_layers):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(weights, features):
        qml.AmplitudeEmbedding(features, wires=range(n_qubits), pad_with=0.0, normalize=True)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    def variational_classifier(weights, bias, x):
        return circuit(weights, x) + bias

    return variational_classifier

def loss_func(params, model, x, y):
    weights, bias = params
    # Map expectation value [-1, 1] to [0, 1] roughly or use directly in square loss
    predictions = [model(weights, bias, x_) for x_ in x]
    # Simple square loss against labels shifted to [-1, 1]
    y_shifted = [2 * y_ - 1 for y_ in y]
    return np.mean((np.array(predictions) - np.array(y_shifted)) ** 2)

def train_qml_classifier(X_train, y_train, n_qubits, n_layers=2, max_steps=50):
    # Initialize parameters
    weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    weights = pnp.random.random(size=weights_shape, requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)
    params = (weights, bias)
    
    model = create_qml_model(n_qubits, n_layers)
    opt = qml.AdamOptimizer(stepsize=0.1)
    
    batch_size = 32
    for step in range(max_steps):
        # Select batch
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = X_train[batch_index]
        y_batch = y_train[batch_index]
        
        params, cost = opt.step_and_cost(lambda p: loss_func(p, model, X_batch, y_batch), params)
        
        if (step + 1) % 10 == 0:
            # print(f"Step {step+1}/{max_steps} | Cost: {cost.item():.4f}")
            pass
            
    return params, model

def predict_qml(params, model, X):
    weights, bias = params
    raw_preds = [model(weights, bias, x) for x in X]
    # Map expectation [-1, 1] to probability [0, 1]
    probs = (np.array(raw_preds) + 1) / 2
    probs = np.clip(probs, 0, 1)
    preds = (probs >= 0.5).astype(int)
    return preds, probs

def evaluate_qml(params, model, X_test, y_test):
    y_pred, y_probs = predict_qml(params, model, X_test)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
    }
    
    try:
        results['roc_auc'] = roc_auc_score(y_test, y_probs)
    except ValueError:
        results['roc_auc'] = 0.5
        
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
        X_train_norm, X_test_norm = scale_and_normalize_features(X_train_res, X_test)
        
        # Calculate qubits needed for AmplitudeEmbedding
        n_qubits = int(np.ceil(np.log2(X_train.shape[1])))
        
        # Train QML model
        params, model = train_qml_classifier(X_train_norm, y_train_res, n_qubits)
        results[name] = evaluate_qml(params, model, X_test_norm, y_test)
        
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
    parser = argparse.ArgumentParser(description='PennyLane QML Iterative 80:20 Patient Split Statistical Analysis')
    parser.add_argument('--iterations', type=int, default=10, help='Number of random splits')
    args = parser.parse_args()
    
    # dimensions = [32, 64, 128]
    dimensions = [64]
    
    for dim in dimensions:
        print(f"\n" + "="*85)
        print(f"ANALYZING {dim} FEATURES WITH PENNYLANE QML OVER {args.iterations} ITERATIONS")
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
            start_time = time.time()
            res = run_iteration(df_fft, df_qft, feats_fft, feats_qft, iteration_seed=42+i)
            fft_all.append(res['FFT'])
            qft_all.append(res['QFT'])
            elapsed = time.time() - start_time
            print(f"Iteration {i+1}/{args.iterations} completed in {elapsed:.1f}s")
            
        stats = perform_statistical_tests(fft_all, qft_all)
        
        print("\n" + f"{'METRIC':<12} | {'FFT MEAN \u00B1 STD':<20} | {'QFT MEAN \u00B1 STD':<20} | {'TT P-VAL':<8} | {'WIL P-VAL':<8}")
        print("-" * 88)
        ordered_metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
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
