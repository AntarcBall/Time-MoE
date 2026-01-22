import os
import glob
import numpy as np
import math
import json
import random
import shutil

DATA_ROOTS = ['dataset_npy/train', 'dataset_npy/test']
SAVE_ROOT = 'processed_bin'
SEGMENT_LENGTH = 2048  # Chop data into small segments for physical shuffling

def save_shuffled_segments(file_list, out_folder, dtype='float32'):
    """
    Reads files, chops them into segments, shuffles the segments in memory buffers,
    and writes them to disk. This ensures physical mixing of data for fast sequential I/O.
    """
    os.makedirs(out_folder, exist_ok=True)
    
    meta = {
        'dtype': dtype,
        'files': {},
        'scales': [],
        'num_sequences': 0,
        'total_points': 0
    }
    
    # 512MB Buffer for shuffling
    # Float32 = 4 bytes. 512MB = 128M elements.
    MAX_BUFFER_ELEMENTS = 128 * 1024 * 1024 
    
    # Store tuples of (data_array, original_filename)
    segment_buffer = []
    buffer_current_elements = 0
    current_chunk_idx = 1
    global_offset = 0
    processed_count = 0
    
    # Randomize file processing order first
    random.shuffle(file_list)
    
    print(f"Processing {len(file_list)} files into {out_folder} with shuffling...")
    
    def flush_buffer():
        nonlocal current_chunk_idx, global_offset, processed_count, segment_buffer, buffer_current_elements
        
        if not segment_buffer:
            return

        # SHUFFLE THE BUFFER! This is the key for Stratified/Mixed Sequential Read
        random.shuffle(segment_buffer)
        
        # Prepare batch data
        data_to_write = []
        for seq, fname in segment_buffer:
            data_to_write.append(seq)
            
            # Record metadata
            meta['scales'].append({
                'offset': global_offset,
                'length': len(seq),
                'file': os.path.basename(fname)
            })
            global_offset += len(seq)
            processed_count += 1
            
        # Concatenate and write
        full_data = np.concatenate(data_to_write, axis=0)
        
        bin_fn = f'data-{current_chunk_idx}-of-placeholder.bin'
        out_path = os.path.join(out_folder, bin_fn)
        with open(out_path, 'wb') as f_out:
            full_data.tofile(f_out)
            
        meta['files'][bin_fn] = len(full_data)
        current_chunk_idx += 1
        
        # Clear buffer
        segment_buffer = []
        buffer_current_elements = 0
        print(f"  - Flushed chunk {current_chunk_idx-1} (Total seqs: {processed_count})")

    for f in file_list:
        try:
            full_seq = np.load(f).astype(np.float32)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
            
        # Chop into segments
        n_points = len(full_seq)
        if n_points < 2: continue
        
        # Calculate how many full segments
        # We want to use as much data as possible.
        # If we just chop non-overlapping, we might lose the tail.
        # But for 'Physical Shuffling', fixed size is good.
        # Let's chop into non-overlapping SEGMENT_LENGTH chunks.
        
        for start in range(0, n_points, SEGMENT_LENGTH):
            end = min(start + SEGMENT_LENGTH, n_points)
            length = end - start
            
            # Skip very short tails (optional, but <16 points might be useless)
            if length < 16: continue
            
            sub_seq = full_seq[start:end]
            
            segment_buffer.append((sub_seq, f))
            buffer_current_elements += length
            
            if buffer_current_elements >= MAX_BUFFER_ELEMENTS:
                flush_buffer()

    # Flush remaining
    flush_buffer()
        
    num_chunks = current_chunk_idx - 1
    
    # Rename files
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
        
    print(f"Saved {processed_count} shuffled segments to {out_folder}.")

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
    # val_files will be shuffled internally by save_shuffled_segments

    print(f"Train Set: {len(train_files)} files (Pure Normal)")
    print(f"Val Set: {len(val_files)} files ({len(val_normal_files)} Normal + {len(anomaly_files)} Anomaly)")

    if os.path.exists(SAVE_ROOT):
        print(f"Cleaning {SAVE_ROOT}...")
        shutil.rmtree(SAVE_ROOT)
        
    print("\n[Processing Train Set - Shuffled Segments]")
    save_shuffled_segments(train_files, os.path.join(SAVE_ROOT, 'train'))
    
    print("\n[Processing Val Set - Shuffled Segments]")
    save_shuffled_segments(val_files, os.path.join(SAVE_ROOT, 'val'))

if __name__ == "__main__":
    prepare_split_dataset_and_convert()
