import wfdb
import numpy as np
import os
import pandas as pd
import zipfile
import argparse
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, resample
# Qiskit for QFT
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate, Initialize

# Static defaults or constants
ZIP_PATH = 'data/mit-bih-malignant-ventricular-ectopy-database-1.0.0.zip'
NORMAL_LABELS = {'N', 'Nor', 'NOISE'}

def detect_qrs_peaks(segment, fs):
    # Simple QRS detection using find_peaks on absolute value (for demo; use Pan-Tompkins or biosppy for production)
    # Typical QRS width: 0.06-0.12s, so min distance ~0.25s
    distance = int(0.25 * fs)
    peaks, _ = find_peaks(np.abs(segment), distance=distance, height=np.percentile(np.abs(segment), 90))
    return peaks

def extract_qrs_rr_segment_features(segment, fs):
    r_peaks = detect_qrs_peaks(segment, fs)
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs
    else:
        rr_intervals = np.array([0])
    
    # rr stats
    heart_rate = len(r_peaks) * 60 / (len(segment) / fs)
    rr_mean = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
    rr_std = np.std(rr_intervals) if len(rr_intervals) > 0 else 0
    rr_min = np.min(rr_intervals) if len(rr_intervals) > 0 else 0
    rr_max = np.max(rr_intervals) if len(rr_intervals) > 0 else 0
    rr_median = np.median(rr_intervals) if len(rr_intervals) > 0 else 0
    rr_diff_std = np.std(np.diff(rr_intervals)) if len(rr_intervals) > 1 else 0
    
    qrs_amplitudes = segment[r_peaks] if len(r_peaks) > 0 else np.array([0])
    # qrs stats
    qrs_mean = np.mean(qrs_amplitudes)
    qrs_std = np.std(qrs_amplitudes)
    qrs_min = np.min(qrs_amplitudes)
    qrs_max = np.max(qrs_amplitudes)
    
    # segment stats
    seg_mean = np.mean(segment)
    seg_std = np.std(segment)
    seg_min = np.min(segment)
    seg_max = np.max(segment)
    seg_skew = skew(segment)
    seg_kurt = kurtosis(segment)
    
    return [heart_rate, rr_mean, rr_std, rr_min, rr_max, rr_median, rr_diff_std, 
            qrs_mean, qrs_std, qrs_min, qrs_max,
            seg_mean, seg_std, seg_min, seg_max, seg_skew, seg_kurt]

def extract_fft_features(segment, n_fft_features):
    fft_features = np.abs(np.fft.rfft(segment))
    if len(fft_features) > n_fft_features:
        fft_features = fft_features[1:n_fft_features+1]
    else:
        fft_features = np.pad(fft_features[1:], (0, n_fft_features-len(fft_features[1:])), 'constant')
    return list(fft_features)

def extract_qft_features(segment, n_qft_features):
    n_qubits = int(np.log2(n_qft_features))
    n_points = 2 ** n_qubits
    
    if len(segment) < n_points:
        padded = np.pad(segment, (0, n_points - len(segment)), 'constant')
    else:
        padded = resample(segment, n_points)
    
    # Normalize for amplitude encoding
    shifted = padded - np.min(padded) if np.min(padded) < 0 else padded
    l2normed = np.linalg.norm(shifted)
    normed = shifted / l2normed if l2normed > 0 else np.zeros_like(shifted)
    
    qc = QuantumCircuit(n_qubits)
    qc.append(Initialize(normed.astype(complex)), range(n_qubits))
    qc.compose(QFTGate(n_qubits), inplace=True)
    
    qft_state = Statevector(qc)
    # Combine real and imag into a single positive value: abs(real) + abs(imag)
    # This captures both magnitude and phase info as a single positive float.
    # combined_features = [float(np.abs(np.real(v)) + np.abs(np.imag(v))) for v in qft_state.data]
    # Stacked Polar (Magnitude + Normalized Phase)
    # combined_features = [float(np.abs(v) + (np.angle(v) + np.pi) / (2 * np.pi)) for v in qft_state.data]
    # Phase-Modulated Magnitude
    combined_features = [float(np.abs(v) * (1 + np.sin(np.angle(v)))) for v in qft_state.data]

    return combined_features

def get_interval_label(ann_syms):
    unique = set(ann_syms)
    if unique.issubset(NORMAL_LABELS):
        return 'Normal'
    return 'Abnormal'

def process_data(db_path, interval_sec, n_features, sample_features, feature_type='fft'):
    # Ensure data is extracted if it's a path that doesn't exist yet but ZIP exists
    if not os.path.exists(db_path) and os.path.exists(ZIP_PATH):
        print(f"Extracting {ZIP_PATH} to {os.path.dirname(db_path)}...")
        extract_to = os.path.dirname(db_path.rstrip('/')) or '.'
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # If the ZIP extracted into a different name, handle nesting
        if not os.path.exists(db_path):
             contents = [c for c in os.listdir(extract_to) if os.path.isdir(os.path.join(extract_to, c))]
             if contents:
                 potential_path = os.path.join(extract_to, contents[0])
                 if os.path.exists(os.path.join(potential_path, os.listdir(potential_path)[0])):
                     db_path = potential_path

    if not os.path.exists(db_path):
        print(f"Error: Database path {db_path} does not exist.")
        return

    all_rows = []
    record_files = [f for f in os.listdir(db_path) if f.endswith('.atr')]
    record_names = [os.path.splitext(f)[0] for f in record_files]

    print(f"Starting {feature_type.upper()} extraction for {len(record_names)} records...")

    for RECORD in record_names:
        try:
            record_full_path = os.path.join(db_path, RECORD)
            record = wfdb.rdrecord(record_full_path)
            annotation = wfdb.rdann(record_full_path, 'atr')
            fs = record.fs
            signal = record.p_signal[:, 0]
            interval_samples = int(interval_sec * fs)
            total_samples = len(signal)
            ann_samples = annotation.sample
            ann_symbols = annotation.symbol
            
            for start in range(0, total_samples, interval_samples):
                end = min(start + interval_samples, total_samples)
                segment = signal[start:end]
                syms_in_interval = [sym for s, sym in zip(ann_samples, ann_symbols) if start <= s < end]
                label = get_interval_label(syms_in_interval)
                
                # QRS/RR and segment features
                qrs_rr_seg_feats = extract_qrs_rr_segment_features(segment, fs)
                
                # Transformation features
                if feature_type == 'fft':
                    transform_feats = extract_fft_features(segment, n_features)
                else:
                    transform_feats = extract_qft_features(segment, n_features)
                
                # Downsample for raw signal
                step = int(fs * interval_sec / sample_features) if sample_features > 0 else 1
                segment_downsampled = segment[::step][:sample_features] if sample_features > 0 else []
                if len(segment_downsampled) < sample_features:
                    segment_downsampled = np.pad(segment_downsampled, (0, sample_features - len(segment_downsampled)), 'constant')
                
                # REORDERED: metadata, then transform, then qrs_rr_seg, then raw signal
                row = [RECORD, start // interval_samples, label] + transform_feats + qrs_rr_seg_feats + list(segment_downsampled)
                all_rows.append(row)
            print(f"Processed record {RECORD} ({feature_type.upper()})")
        except Exception as e:
            print(f"Error processing record {RECORD}: {e}")

    feat_prefix = 'fft' if feature_type == 'fft' else 'qft'
    columns = ['patient_id', 'interval_id', 'annotation'] + \
            [f'{feat_prefix}_{i+1}' for i in range(n_features)] + \
            ['heart_rate', 'rr_mean', 'rr_std', 'rr_min', 'rr_max', 'rr_median', 'rr_diff_std',
            'qrs_mean', 'qrs_std', 'qrs_min', 'qrs_max',
            'seg_mean', 'seg_std', 'seg_min', 'seg_max', 'seg_skew', 'seg_kurt'] + \
            [f'signal_{i}' for i in range(sample_features)]
    
    df = pd.DataFrame(all_rows, columns=columns)
    output_file = f'data/qrs_rr_seg_{feature_type}_{n_features}_30s.csv'
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} intervals to {output_file}")

def main():
    default_db_path = 'data/mit-bih-malignant-ventricular-ectopy-database-1.0.0/'
    default_interval_sec = 30
    default_n_fft = 128
    default_n_qft = 128
    default_sample_features = 180
    
    parser = argparse.ArgumentParser(description='ECG FFT & QFT Feature Extractor')
    parser.add_argument('--n_fft_features', type=int, default=default_n_fft,
                        help=f'Number of FFT features (default: {default_n_fft})')
    parser.add_argument('--n_qft_features', type=int, default=default_n_qft,
                        help=f'Number of QFT features (default: {default_n_qft})')
    
    args = parser.parse_args()
    
    process_data(
        db_path=default_db_path,
        interval_sec=default_interval_sec,
        n_features=args.n_fft_features,
        sample_features=default_sample_features,
        feature_type='fft'
    )

    process_data(
        db_path=default_db_path,
        interval_sec=default_interval_sec,
        n_features=args.n_qft_features,
        sample_features=default_sample_features,
        feature_type='qft'
    )

if __name__ == "__main__":
    main()
