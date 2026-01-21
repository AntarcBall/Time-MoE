import os
import glob
import numpy as np
import math
import json
import shutil

# Reuse logic from Time-MoE/scripts/convert_dataset_to_bin.py
# But simplified for our NPY files

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
        meta['scales'] = [] # We'll store file info here
        
        idx = 0
        file_name_format = 'data-{}-of-{}.bin'
        
        print(f"Concatenating {len(npy_files)} files...")
        
        for f in npy_files:
            seq = np.load(f).astype(np.float32)
            
            # Simple metadata
            meta['scales'].append({
                'offset': idx,
                'length': len(seq),
                'file': os.path.basename(f)
            })
            
            idx += len(seq)
            sequence.append(seq)
            
        if not sequence:
            return 0
            
        sequence = np.concatenate(sequence, axis=0)
        # Fix: num_sequences should be the count of individual time series, not total points
        meta['num_sequences'] = len(npy_files)
        
        # Save sequence
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
        print(f"Error: {e}")
        return 0

def convert_folder(src_dir, dest_dir):
    print(f"Converting {src_dir} to {dest_dir}")
    files = sorted(glob.glob(os.path.join(src_dir, '*.npy')))
    if not files:
        print(f"No files in {src_dir}")
        return
        
    process_to_bin(files, dest_dir)

if __name__ == "__main__":
    convert_folder('dataset_npy/train', 'dataset_bin/train')
    convert_folder('dataset_npy/test', 'dataset_bin/test')
