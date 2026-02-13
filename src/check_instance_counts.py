import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)

def print_distribution(df, split_name):
    counts = df['annotation'].value_counts()
    normal = counts.get('Normal', 0)
    abnormal = counts.get('Abnormal', 0)
    total = normal + abnormal
    print(f"{split_name:<10} | Normal: {normal:<6} | Abnormal: {abnormal:<6} | Total: {total}")

def main():
    # Use one of the existing feature files
    csv_path = 'data/qrs_rr_seg_fft_64_30s.csv'
    df = load_data(csv_path)
    if df is None:
        return

    print(f"Data Source: {csv_path}\n")
    print(f"{'Split':<10} | {'Normal':<13} | {'Abnormal':<14} | {'Total'}")
    print("-" * 55)

    # Patient-based 80:20 split (matching standard_classifier_stat_tests.py)
    unique_patients = df['patient_id'].unique()
    train_p, test_p = train_test_split(unique_patients, test_size=0.2, random_state=42)

    train_df = df[df['patient_id'].isin(train_p)]
    test_df = df[df['patient_id'].isin(test_p)]

    print_distribution(train_df, "Train")
    print_distribution(test_df, "Test")
    print_distribution(df, "Overall")
    print("-" * 55)

if __name__ == "__main__":
    main()
