import os
import glob
import numpy as np
import math
import json
import random
import shutil

# Re-engineered Data Pipeline: Strict Isolation for Single-Class AD
# Author: Sisyphus
# Strategy: 
# 1. TRAIN: Only Group 0 (Normal). Purity 100%.
# 2. TEST: Mixed Group 0 + Group 1-6. For realistic Anomaly Detection evaluation.

def save_array_to_bin(arr, fn):
    with open(fn, mode='wb') as file:
        arr.tofile(file)

def process_to_bin(npy_files, out_folder, dtype='float32'):
    try:
        max_chunk_size = (1 << 30) * 1  # 1GB chunks
        
        sequence = []
        meta = {}
        meta['dtype'] = dtype
        meta['files'] = {}
        meta['scales'] = [] 
        
        idx = 0
        file_name_format = 'data-{}-of-{}.bin'
        
        print(f"Processing {len(npy_files)} files into {out_folder}...")
        
        for f in npy_files:
            # Load and ensure float32
            seq = np.load(f).astype(np.float32)
            
            # Metadata tracking for random access
            meta['scales'].append({
                'offset': idx,
                'length': len(seq),
                'file': os.path.basename(f)
            })
            
            idx += len(seq)
            sequence.append(seq)
            
        if not sequence:
            print(f"Warning: No data for {out_folder}")
            return 0
            
        sequence = np.concatenate(sequence, axis=0)
        meta['num_sequences'] = len(npy_files)
        # Add total_points to meta for better debugging
        meta['total_points'] = len(sequence)
        
        # Save sequence in chunks
        memory_size = sequence.nbytes
        num_chunks = math.ceil(memory_size / max_chunk_size)
        if num_chunks == 0: num_chunks = 1
        chunk_length = math.ceil(len(sequence) / num_chunks)
        
        os.makedirs(out_folder, exist_ok=True)
        
        for i in range(num_chunks):
            start_idx = i * chunk_length
            end_idx = min(start_idx + chunk_length, len(sequence))
            sub_seq = sequence[start_idx: end_idx]
            sub_fn = file_name_format.format(i + 1, num_chunks)
            out_fn = os.path.join(out_folder, sub_fn)
            save_array_to_bin(sub_seq, out_fn)
            meta['files'][sub_fn] = len(sub_seq)
            
        # Save meta
        with open(os.path.join(out_folder, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
            
        return len(sequence)
        
    except Exception as e:
        print(f"Error processing {out_folder}: {e}")
        return 0

def strict_isolation_split():
    # 1. Gather ALL npy files from both directories
    # We ignore the current folder structure and classify by FILENAME ONLY.
    src_dirs = ['dataset_npy/train', 'dataset_npy/test']
    all_files = []
    for d in src_dirs:
        files = glob.glob(os.path.join(d, '*.npy'))
        all_files.extend(files)
    
    print(f"Total files found: {len(all_files)}")
    
    # 2. Strict Separation
    # Normal: Starts with '0_'
    # Anomaly: Starts with '1_' to '6_' (or just not '0_')
    
    normals = []
    anomalies = []
    
    for f in all_files:
        fname = os.path.basename(f)
        if fname.startswith('0_'):
            normals.append(f)
        else:
            anomalies.append(f)
            
    print(f"  - Normal Pool (Group 0): {len(normals)}")
    print(f"  - Anomaly Pool (Group 1-6): {len(anomalies)}")
    
    # 3. Construct Sets
    # Shuffle Normals first to ensure random distribution
    random.seed(42)
    random.shuffle(normals)
    
    # Train Set: 90% of Normals ONLY
    # This guarantees the model ONLY sees normal data during training.
    split_idx = int(len(normals) * 0.90)
    train_set = normals[:split_idx]
    
    # Validation/Test Set: Remaining 10% Normals + 100% Anomalies
    # This creates a realistic evaluation scenario where we test if the model 
    # can distinguish the held-out normals from the anomalies.
    test_set = normals[split_idx:] + anomalies
    
    # CRITICAL: Shuffle Test Set
    # This prevents "all normals then all anomalies" which breaks partial evaluation in DataLoader
    random.shuffle(test_set)
    
    print(f"  -> Final Train Set: {len(train_set)} files (Pure Normal)")
    print(f"  -> Final Test Set : {len(test_set)} files (Mixed: {len(normals)-split_idx} Normal + {len(anomalies)} Anomaly)")
    
    # 4. Execute Conversion
    # Clear existing bins to prevent contamination from old chunks
    if os.path.exists('dataset_bin'):
        print("Cleaning old dataset_bin...")
        shutil.rmtree('dataset_bin')
    
    print("\n[Processing Train Set]")
    process_to_bin(train_set, 'dataset_bin/train')
    
    print("\n[Processing Test Set]")
    process_to_bin(test_set, 'dataset_bin/test')
    
    print("\nStrict Isolation Split Completed Successfully.")

if __name__ == "__main__":
    strict_isolation_split()
