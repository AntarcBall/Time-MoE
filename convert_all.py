import os
import glob
import numpy as np
import math
import json
import random
import shutil

DATA_ROOTS = ['dataset_npy/train', 'dataset_npy/test']
SAVE_ROOT = 'processed_bin'

def save_to_bin_chunked(npy_files, out_folder, dtype='float32'):
    """
    Reads files sequentially and writes in fixed-size chunks to minimize memory usage.
    """
    os.makedirs(out_folder, exist_ok=True)
    
    meta = {
        'dtype': dtype,
        'files': {},
        'scales': [],
        'num_sequences': 0,
        'total_points': 0
    }
    
    MAX_CHUNK_BYTES = 1 << 30
    MAX_CHUNK_ELEMENTS = MAX_CHUNK_BYTES // 4
    
    buffer = []
    buffer_elements = 0
    current_chunk_idx = 1
    
    processed_count = 0
    global_offset = 0
    
    print(f"Processing {len(npy_files)} files into {out_folder}...")
    
    for f in npy_files:
        try:
            seq = np.load(f).astype(np.float32)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
            
        seq_len = len(seq)
        
        meta['scales'].append({
            'offset': global_offset,
            'length': seq_len,
            'file': os.path.basename(f)
        })
        
        global_offset += seq_len
        processed_count += 1
        
        buffer.append(seq)
        buffer_elements += seq_len
        
        if buffer_elements >= MAX_CHUNK_ELEMENTS:
            full_data = np.concatenate(buffer, axis=0)
            buffer = []
            buffer_elements = 0
            
            start = 0
            while start < len(full_data):
                end = start + MAX_CHUNK_ELEMENTS
                chunk = full_data[start:end]
                
                if len(chunk) == MAX_CHUNK_ELEMENTS:
                    bin_fn = f'data-{current_chunk_idx}-of-placeholder.bin'
                    out_path = os.path.join(out_folder, bin_fn)
                    with open(out_path, 'wb') as f_out:
                        chunk.tofile(f_out)
                    
                    meta['files'][bin_fn] = len(chunk)
                    current_chunk_idx += 1
                    start = end
                else:
                    buffer.append(chunk)
                    buffer_elements += len(chunk)
                    break

    if buffer_elements > 0:
        full_data = np.concatenate(buffer, axis=0)
        bin_fn = f'data-{current_chunk_idx}-of-placeholder.bin'
        out_path = os.path.join(out_folder, bin_fn)
        with open(out_path, 'wb') as f_out:
            full_data.tofile(f_out)
        meta['files'][bin_fn] = len(full_data)
        current_chunk_idx += 1
        
    num_chunks = current_chunk_idx - 1
    
    final_files_map = {}
    for old_name, length in meta['files'].items():
        idx = int(old_name.split('-')[1])
        new_name = f'data-{idx}-of-{num_chunks}.bin'
        os.rename(os.path.join(out_folder, old_name), os.path.join(out_folder, new_name))
        final_files_map[new_name] = length
        
    meta['files'] = final_files_map
    meta['num_sequences'] = processed_count
    meta['total_points'] = global_offset
    
    with open(os.path.join(out_folder, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        
    print(f"Saved {processed_count} sequences to {out_folder} in {num_chunks} chunks.")

def prepare_split_dataset_and_convert():
    normal_files = []
    anomaly_files = []
    
    print("Scanning for .npy files...")
    for d in DATA_ROOTS:
        if not os.path.isdir(d):
            print(f"Warning: Directory {d} not found.")
            continue
            
        all_files = glob.glob(os.path.join(d, '*.npy'))
        for f in all_files:
            fname = os.path.basename(f)
            if fname.startswith('0_'):
                normal_files.append(f)
            else:
                anomaly_files.append(f)
    
    normal_files = sorted(list(set(normal_files)))
    anomaly_files = sorted(list(set(anomaly_files)))

    print(f"Found {len(normal_files)} Normal files, {len(anomaly_files)} Anomaly files.")

    random.seed(42)

    random.shuffle(normal_files)
    
    split_ratio = 0.9
    split_idx = int(len(normal_files) * split_ratio)
    
    train_files = normal_files[:split_idx]
    val_normal_files = normal_files[split_idx:]
    
    val_files = val_normal_files + anomaly_files
    random.shuffle(val_files)

    print(f"Train Set: {len(train_files)} (All Normal)")
    print(f"Val Set: {len(val_files)} ({len(val_normal_files)} Normal + {len(anomaly_files)} Anomaly)")

    if os.path.exists(SAVE_ROOT):
        print(f"Cleaning {SAVE_ROOT}...")
        shutil.rmtree(SAVE_ROOT)
        
    print("\n[Processing Train Set]")
    save_to_bin_chunked(train_files, os.path.join(SAVE_ROOT, 'train'))
    
    print("\n[Processing Val Set]")
    save_to_bin_chunked(val_files, os.path.join(SAVE_ROOT, 'val'))

if __name__ == "__main__":
    prepare_split_dataset_and_convert()
