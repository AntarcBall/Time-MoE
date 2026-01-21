import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SRC_DIR = 'Renomeado'
OUT_DIR = 'dataset_npy'
TRAIN_DIR = os.path.join(OUT_DIR, 'train')
TEST_DIR = os.path.join(OUT_DIR, 'test')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Find Class 0 files
class0_files = glob.glob(os.path.join(SRC_DIR, '0_*.csv'))
# Find Anomaly files
anomaly_files = []
for i in range(1, 7):
    anomaly_files.extend(glob.glob(os.path.join(SRC_DIR, f'{i}_*.csv')))

print(f"Found {len(class0_files)} Class 0 files")
print(f"Found {len(anomaly_files)} Anomaly files")

if not class0_files:
    print("No Class 0 files found!")
    exit(1)

# Split Class 0
train_files, test_normal_files = train_test_split(class0_files, test_size=0.2, random_state=42)

def process_file(filepath, dest_dir):
    try:
        # Line 21 is header (1-based), so skip 20 lines
        df = pd.read_csv(filepath, skiprows=20)
        
        # Verify columns
        expected_cols = ['CH1', 'CH2', 'CH3', 'CH4']
        if not all(col in df.columns for col in expected_cols):
            # Sometimes header might be different line?
            # Try to find header line
            # But let's assume consistent format for now based on inspection
            pass
            
        filename = os.path.basename(filepath).replace('.csv', '')
        
        for ch in expected_cols:
            if ch in df.columns:
                data = df[ch].values.astype(np.float32)
                
                # Flux Masking: Assuming flux (CH4) didn't exist at the beginning
                if ch == 'CH4':
                    # print(f"Masking flux channel {ch} for {filepath}")
                    data = np.zeros_like(data)
                
                # Check for NaNs
                if np.isnan(data).any():
                    print(f"Warning: NaNs found in {filepath} {ch}. Interpolating...")
                    data = pd.Series(data).interpolate().fillna(0).values.astype(np.float32)
                
                # Save as npy
                out_name = f"{filename}_{ch}.npy"
                np.save(os.path.join(dest_dir, out_name), data)
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

print(f"Processing {len(train_files)} Train Normal files...")
for i, f in enumerate(train_files):
    if i % 100 == 0: print(f"Processing {i}...")
    process_file(f, TRAIN_DIR)

print(f"Processing {len(test_normal_files)} Test Normal files...")
for f in test_normal_files:
    process_file(f, TEST_DIR)

print(f"Processing {len(anomaly_files)} Test Anomaly files...")
for i, f in enumerate(anomaly_files):
    if i % 100 == 0: print(f"Processing {i}...")
    process_file(f, TEST_DIR)

print("Done.")
