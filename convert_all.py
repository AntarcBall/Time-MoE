import os
import glob
import numpy as np
import math
import json
import random
import shutil

# Re-use the robust binary saver from previous convert_all.py
DATA_ROOTS = ['dataset_npy/train', 'dataset_npy/test'] # Dummy, not used by this script directly when imported
SAVE_ROOT = 'processed_bin'
SEGMENT_LENGTH = 2048

def save_shuffled_segments(file_list, out_folder, dtype='float32'):
    """
    Reads files (already chopped .npy segments), shuffles them in memory buffers,
    and writes them to disk.
    """
    os.makedirs(out_folder, exist_ok=True)
    
    meta = {
        'dtype': dtype,
        'files': {},
        'scales': [],
        'num_sequences': 0,
        'total_points': 0
    }
    
    # 512MB Buffer
    MAX_BUFFER_ELEMENTS = 128 * 1024 * 1024 
    
    segment_buffer = []
    buffer_current_elements = 0
    current_chunk_idx = 1
    global_offset = 0
    processed_count = 0
    
    random.shuffle(file_list)
    
    print(f"Processing {len(file_list)} segments into {out_folder}...")
    
    def flush_buffer():
        nonlocal current_chunk_idx, global_offset, processed_count, segment_buffer, buffer_current_elements
        
        if not segment_buffer: return

        random.shuffle(segment_buffer)
        
        data_to_write = []
        for seq, fname in segment_buffer:
            data_to_write.append(seq)
            meta['scales'].append({
                'offset': global_offset,
                'length': len(seq),
                'file': os.path.basename(fname)
            })
            global_offset += len(seq)
            processed_count += 1
            
        full_data = np.concatenate(data_to_write, axis=0)
        
        bin_fn = f'data-{current_chunk_idx}-of-placeholder.bin'
        out_path = os.path.join(out_folder, bin_fn)
        with open(out_path, 'wb') as f_out:
            full_data.tofile(f_out)
            
        meta['files'][bin_fn] = len(full_data)
        current_chunk_idx += 1
        
        segment_buffer = []
        buffer_current_elements = 0
        print(f"  - Flushed chunk {current_chunk_idx-1}")

    for f in file_list:
        try:
            # These are already small segments
            seq = np.load(f).astype(np.float32)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
            
        segment_buffer.append((seq, f))
        buffer_current_elements += len(seq)
        
        if buffer_current_elements >= MAX_BUFFER_ELEMENTS:
            flush_buffer()

    flush_buffer()
        
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
        
    print(f"Saved {processed_count} segments to {out_folder}.")

if __name__ == "__main__":
    print("Please run preprocess_balanced.py instead.")
