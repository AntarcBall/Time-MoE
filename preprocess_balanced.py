import glob
import os
import shutil
import numpy as np
import pandas as pd
import random

# ==========================================
# 1. Configuration
# ==========================================
SOURCE_DIR = "Renomeado/dataset"  # Where .csv files are located
NPY_SAVE_DIR = "dataset_npy"      # Intermediate .npy files
BIN_SAVE_DIR = "processed_bin"    # Final binary files for training

# Data Sampling Strategy
NORMAL_SEGMENTS = 6000  # Target number of normal segments for validation
ANOMALY_SEGMENTS = 1000 # Target number of anomaly segments for validation (total across all anomaly types)
SEGMENT_LENGTH = 2048

# ==========================================
# 2. Helpers
# ==========================================
def load_csv_to_npy(csv_path):
    """
    Loads a single-column CSV (no header) and converts to float32 numpy array.
    """
    try:
        # Assuming the CSV has no header and 1 column of data
        df = pd.read_csv(csv_path, header=None)
        data = df.values.flatten().astype(np.float32)
        return data
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def chop_and_save_segments(data, file_prefix, out_dir, count, is_random_sampling=False, num_samples=0):
    """
    Chops data into segments and saves them as .npy files.
    """
    n_points = len(data)
    saved_paths = []
    
    if n_points < SEGMENT_LENGTH:
        return []

    if is_random_sampling:
        # Random Sampling (for Anomaly Val & Normal Val Augmentation)
        for _ in range(num_samples):
            max_start = n_points - SEGMENT_LENGTH
            if max_start <= 0: start = 0
            else: start = random.randint(0, max_start)
            
            segment = data[start : start + SEGMENT_LENGTH]
            fname = f"{file_prefix}_seg{count}.npy"
            path = os.path.join(out_dir, fname)
            np.save(path, segment)
            saved_paths.append(path)
            count += 1
    else:
        # Sequential Chopping (for Train Normal)
        # Use full data, non-overlapping
        for start in range(0, n_points, SEGMENT_LENGTH):
            end = start + SEGMENT_LENGTH
            if end > n_points: break # Drop tail
            
            segment = data[start:end]
            fname = f"{file_prefix}_seg{count}.npy"
            path = os.path.join(out_dir, fname)
            np.save(path, segment)
            saved_paths.append(path)
            count += 1
            
    return saved_paths, count

# ==========================================
# 3. Main Logic
# ==========================================
def main():
    # A. Clean up old directories
    for d in [NPY_SAVE_DIR, BIN_SAVE_DIR]:
        if os.path.exists(d):
            print(f"Cleaning {d}...")
            shutil.rmtree(d)
    
    os.makedirs(os.path.join(NPY_SAVE_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(NPY_SAVE_DIR, 'val'), exist_ok=True)

    # B. Gather File Lists
    print("Scanning CSV files...")
    all_csvs = glob.glob(os.path.join(SOURCE_DIR, "*.csv"))
    normal_files = [f for f in all_csvs if os.path.basename(f).startswith('0_')]
    anomaly_files = [f for f in all_csvs if not os.path.basename(f).startswith('0_')]
    
    print(f"Found {len(normal_files)} Normal CSVs, {len(anomaly_files)} Anomaly CSVs.")
    
    # Random Seed
    random.seed(42)
    random.shuffle(normal_files)
    random.shuffle(anomaly_files)
    
    # C. Split Normal for Train/Val (90/10 File Split)
    split_idx = int(len(normal_files) * 0.9)
    train_normal_files = normal_files[:split_idx]
    val_normal_files = normal_files[split_idx:]
    
    print(f"Split Normals: {len(train_normal_files)} Train files, {len(val_normal_files)} Val files.")

    # ---------------------------------------------------------
    # D. Process TRAIN Set (Pure Normal, Sequential Chopping)
    # ---------------------------------------------------------
    print("\n[Processing Train Set]")
    train_npy_dir = os.path.join(NPY_SAVE_DIR, 'train')
    train_seg_count = 0
    
    for f in train_normal_files:
        data = load_csv_to_npy(f)
        if data is None: continue
        
        _, train_seg_count = chop_and_save_segments(
            data, 
            os.path.basename(f).replace('.csv', ''), 
            train_npy_dir, 
            train_seg_count,
            is_random_sampling=False
        )
    print(f"Generated {train_seg_count} training segments.")

    # ---------------------------------------------------------
    # E. Process VAL Set (Balanced Sampling)
    # ---------------------------------------------------------
    print("\n[Processing Val Set]")
    val_npy_dir = os.path.join(NPY_SAVE_DIR, 'val')
    val_seg_count = 0
    
    # 1. Normal Sampling (Target: 6000 segments)
    # Distribute 6000 samples across available val_normal_files
    if len(val_normal_files) > 0:
        samples_per_file = max(1, NORMAL_SEGMENTS // len(val_normal_files))
        print(f"Sampling {samples_per_file} segments per Normal Val file...")
        
        for f in val_normal_files:
            data = load_csv_to_npy(f)
            if data is None: continue
            
            _, val_seg_count = chop_and_save_segments(
                data, 
                os.path.basename(f).replace('.csv', ''), 
                val_npy_dir, 
                val_seg_count,
                is_random_sampling=True,
                num_samples=samples_per_file
            )
            
    # 2. Anomaly Sampling (Target: 1000 segments)
    # Distribute 1000 samples across anomaly_files
    if len(anomaly_files) > 0:
        samples_per_file = max(1, ANOMALY_SEGMENTS // len(anomaly_files))
        # If too many files, sample at least 1, but cap total? 
        # Actually anomaly files are usually fewer than normals in many datasets, but here 1:6 ratio?
        # User said "1:고장 6". Let's just try to be balanced.
        # If anomaly files are many, samples_per_file might be 0.
        if samples_per_file == 0: samples_per_file = 1
        
        print(f"Sampling {samples_per_file} segments per Anomaly file...")
        
        for f in anomaly_files:
            data = load_csv_to_npy(f)
            if data is None: continue
            
            _, val_seg_count = chop_and_save_segments(
                data, 
                os.path.basename(f).replace('.csv', ''), 
                val_npy_dir, 
                val_seg_count,
                is_random_sampling=True,
                num_samples=samples_per_file
            )

    print(f"Generated {val_seg_count} validation segments in total.")

    # ---------------------------------------------------------
    # F. Convert .npy to .bin (Pre-shuffled)
    # ---------------------------------------------------------
    print("\n[Converting to Shuffled Binaries]")
    from convert_all import save_shuffled_segments
    
    # Train Bin
    train_npy_files = glob.glob(os.path.join(train_npy_dir, '*.npy'))
    save_shuffled_segments(train_npy_files, os.path.join(BIN_SAVE_DIR, 'train'))
    
    # Val Bin
    val_npy_files = glob.glob(os.path.join(val_npy_dir, '*.npy'))
    save_shuffled_segments(val_npy_files, os.path.join(BIN_SAVE_DIR, 'val'))

if __name__ == "__main__":
    main()
