import glob
import os
import shutil
import numpy as np
import pandas as pd
import random

# ==========================================
# 1. Configuration
# ==========================================
SOURCE_DIR = "Renomeado/dataset"  
NPY_SAVE_DIR = "dataset_npy"      
BIN_SAVE_DIR = "processed_bin"    

NORMAL_SEGMENTS = 6000  
ANOMALY_SEGMENTS = 1000 
SEGMENT_LENGTH = 2048

# ==========================================
# 2. Helpers
# ==========================================
def load_csv_to_npy(csv_path):
    """
    Robust CSV Loader.
    Strategy 1: Find 'TIME' header and select CH2, CH3, CH4 by name.
    Strategy 2: If no header found, assume 5 columns and take indices [2, 3, 4].
    """
    try:
        # 1. Detect Header Line
        header_row = None
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if 'TIME' in line.upper() and ',' in line:
                    header_row = i
                    break
        
        df = None
        data = None

        if header_row is not None:
            # Case A: Header found
            df = pd.read_csv(csv_path, header=header_row)
            df.columns = [str(c).strip().upper() for c in df.columns]
            
            target_cols = ['CH2', 'CH3', 'CH4']
            available_cols = [c for c in target_cols if c in df.columns]
            
            if len(available_cols) == 3:
                data = df[available_cols].values
            else:
                # Fallback to column index if names don't match perfectly but header exists
                if df.shape[1] >= 5:
                    data = df.iloc[:, 2:5].values
        
        if data is None:
            # Case B: No header found or parsing failed -> Try reading raw
            # Skip metadata lines (usually first 15-20 lines). 
            # Heuristic: Find first line starting with a number or minus sign
            try:
                with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                    start_row = 0
                    for i, line in enumerate(lines):
                        parts = line.split(',')
                        # Check if first token is a number (time)
                        try:
                            float(parts[0])
                            if len(parts) >= 5: # Make sure it has enough columns
                                start_row = i
                                break
                        except:
                            continue
                
                # Read from start_row, no header
                df = pd.read_csv(csv_path, header=None, skiprows=start_row)
                if df.shape[1] >= 5:
                    # Take columns 2, 3, 4 (0-based index) -> CH2, CH3, CH4
                    data = df.iloc[:, 2:5].values
            except Exception as e:
                print(f"  Fallback failed for {csv_path}: {e}")
                return None

        if data is not None:
            return data.flatten().astype(np.float32)
        else:
            print(f"Warning: Could not extract data from {csv_path}")
            return None

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def chop_and_save_segments(data, file_prefix, out_dir, count, is_random_sampling=False, num_samples=0):
    n_points = len(data)
    saved_paths = []
    
    if n_points < SEGMENT_LENGTH:
        return [], count

    if is_random_sampling:
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
        for start in range(0, n_points, SEGMENT_LENGTH):
            end = start + SEGMENT_LENGTH
            if end > n_points: break 
            
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
    for d in [NPY_SAVE_DIR, BIN_SAVE_DIR]:
        if os.path.exists(d):
            print(f"Cleaning {d}...")
            shutil.rmtree(d)
    
    os.makedirs(os.path.join(NPY_SAVE_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(NPY_SAVE_DIR, 'val'), exist_ok=True)

    print("Scanning CSV files...")
    all_csvs = glob.glob(os.path.join(SOURCE_DIR, "*.csv"))
    if not all_csvs:
        print(f"Error: No CSV files found in {SOURCE_DIR}")
        return

    normal_files = [f for f in all_csvs if os.path.basename(f).startswith('0_')]
    anomaly_files = [f for f in all_csvs if not os.path.basename(f).startswith('0_')]
    
    print(f"Found {len(normal_files)} Normal CSVs, {len(anomaly_files)} Anomaly CSVs.")
    
    random.seed(42)
    random.shuffle(normal_files)
    random.shuffle(anomaly_files)
    
    split_idx = int(len(normal_files) * 0.9)
    train_normal_files = normal_files[:split_idx]
    val_normal_files = normal_files[split_idx:]
    
    print(f"Split Normals: {len(train_normal_files)} Train files, {len(val_normal_files)} Val files.")

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

    print("\n[Processing Val Set]")
    val_npy_dir = os.path.join(NPY_SAVE_DIR, 'val')
    val_seg_count = 0
    
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
            
    if len(anomaly_files) > 0:
        samples_per_file = max(1, ANOMALY_SEGMENTS // len(anomaly_files))
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

    print("\n[Converting to Shuffled Binaries]")
    try:
        from convert_all import save_shuffled_segments
    except ImportError:
        print("Error: Could not import save_shuffled_segments from convert_all.py")
        return
    
    train_npy_files = glob.glob(os.path.join(train_npy_dir, '*.npy'))
    save_shuffled_segments(train_npy_files, os.path.join(BIN_SAVE_DIR, 'train'))
    
    val_npy_files = glob.glob(os.path.join(val_npy_dir, '*.npy'))
    save_shuffled_segments(val_npy_files, os.path.join(BIN_SAVE_DIR, 'val'))

if __name__ == "__main__":
    main()
