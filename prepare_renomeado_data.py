import os
import pandas as pd
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json

# Configuration
SOURCE_DIR = 'Time-MoE/Renomeado'
OUTPUT_BASE = 'Time-MoE/dataset_renomeado'
TRAIN_DIR = os.path.join(OUTPUT_BASE, 'train')
TEST_DIR = os.path.join(OUTPUT_BASE, 'test')
NPY_DIR = os.path.join(OUTPUT_BASE, 'npy')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)

def process_file(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return None
            
        # Parse filename
        filename = os.path.basename(file_path)
        # Assuming format: class_...csv (e.g., 0_c2264c30c100.csv)
        class_id = int(filename.split('_')[0])
        
        # Read CSV - skip header lines based on inspection (first 4 lines seem to be meta)
        # Line 5 seems to be data start based on previous `tail` output showing data
        # Let's try reading with header=None and skipping rows until data
        # Actually, `tail` output showed standard CSV format. `head` showed metadata.
        # We need to find where data starts. 
        # Inspecting `head` again:
        # Model,MSO4034B
        # Firmware Version,2.90
        # 
        # Waveform Type,ANALOG,,,
        # Point Format,Y,,,
        # Horizontal Units,s,,,
        # ...
        # usually 18-20 lines of header in Tektronix scopes?
        # Let's try to detect header. Or read all and drop non-numeric.
        
        # Robust reading: read all lines, find first line with 3+ commas that parses to floats?
        # Faster: Just use skiprows=20? 
        # Let's re-read the head to be sure.
        # The user provided `head` output was short.
        # Let's safer assume pandas can handle it if we find the header "TIME" or similar?
        # Actually, let's just read as text and filter.
        
        # Simpler approach: Read with pandas, coerce to numeric, drop NaNs
        # But we need specific columns.
        # Channel Independence: 3 currents + 1 flux?
        # The user said "3상 전류 및 1상 자속".
        # The file content sample: "4.99960e+00,1.608,0.76,0.7,1.72"
        # 5 columns. 1st is Time? Next 4 are CH1-CH4?
        # If so, we typically ignore Time for DL models or use it for delta.
        # Time-MoE takes univariate. So we split into 4 independent series?
        
        # FIX: The header "TIME,CH1,CH2,CH3,CH4" is at line 21 (0-indexed line 20).
        # Data starts at line 22 (0-indexed line 21).
        # We can detect the line starting with "TIME" or just skiprows=21 (to skip header).
        # Let's try reading with header=None and finding the data.
        
        try:
            # Efficiently find data start
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            start_idx = 0
            for idx, line in enumerate(lines):
                if line.startswith("TIME"):
                    start_idx = idx + 1
                    break
            
            if start_idx == 0:
                # Fallback: look for 5 columns of floats
                for idx, line in enumerate(lines):
                    parts = line.split(',')
                    if len(parts) >= 5:
                        try:
                            float(parts[0])
                            start_idx = idx
                            break
                        except:
                            continue
                            
            if start_idx == 0:
                print(f"Skipping {filename}: Could not find data start")
                return None

            # Read data from start_idx
            # Use engine='c' for speed, pass header=None
            df = pd.read_csv(file_path, skiprows=start_idx, header=None, on_bad_lines='skip')
        except Exception as e:
            print(f"Read error {filename}: {e}")
            return None
        
        # Filter for numeric rows only
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        if len(df) == 0:
            return None
            
        # Columns: 0=Time, 1=CH1, 2=CH2, 3=CH3, 4=CH4
        # We treat each channel as a separate sample for "Channel Independence" strategy
        # Data shape: (T, 4)
        data = df.iloc[:, 1:].values # Shape (T, 4)
        
        # Save each channel as separate NPY for flexibility
        saved_files = []
        for ch in range(4):
            ch_data = data[:, ch].astype(np.float32)
            # Normalize? Time-MoE pipeline does normalization during loading usually (TimeMoEDataset).
            # But here we just convert to NPY.
            
            # Naming: {class}_{original_name}_CH{ch+1}.npy
            npy_name = f"{filename.replace('.csv', '')}_CH{ch+1}.npy"
            save_path = os.path.join(NPY_DIR, npy_name)
            np.save(save_path, ch_data)
            saved_files.append((class_id, save_path))
            
        return saved_files
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_bin_dataset(file_list, output_dir, split_name):
    # Concatenate all NPYs into one binary file and create meta.json
    bin_path = os.path.join(output_dir, f'data-1-of-1.bin')
    meta_path = os.path.join(output_dir, 'meta.json')
    
    meta_infos = []
    current_offset = 0
    
    with open(bin_path, 'wb') as f_bin:
        for class_id, npy_path in tqdm(file_list, desc=f"Writing {split_name} binary"):
            data = np.load(npy_path)
            # Flatten just in case
            data = data.flatten()
            length = len(data)
            
            # Write to binary
            # Ensure float32
            data = data.astype(np.float32)
            f_bin.write(data.tobytes())
            
            # Meta info
            meta_infos.append({
                "file": os.path.basename(npy_path),
                "offset": current_offset,
                "length": length,
                "class": class_id
            })
            current_offset += length
            
    with open(meta_path, 'w') as f_meta:
        json.dump(meta_infos, f_meta, indent=2)

def main():
    files = glob.glob(os.path.join(SOURCE_DIR, '*.csv'))
    if not files:
        print(f"No CSV files found in {SOURCE_DIR}")
        return

    print(f"Found {len(files)} files. Processing...")
    
    # Process files in parallel
    all_npy_records = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_file, files), total=len(files)))
        
    for res in results:
        if res:
            all_npy_records.extend(res)
            
    # Split into Train (Class 0 only) and Test (Class 0-6)
    # Train: ONLY Class 0
    # Test: ALL Classes (0-6) to evaluate detection
    
    train_records = [rec for rec in all_npy_records if rec[0] == 0]
    # For test, we use all records. 
    # Ideally, we should not test on training data.
    # Split Class 0 into Train/Test?
    # User said "just only use pretrained data" -> maybe implies using Class 0 for training.
    # Standard AD: Train on normal (0), Test on Normal(0) + Anomalous(1-6).
    # We should split Class 0. Let's say 80% Train, 20% Test.
    # And all Class 1-6 are Test.
    
    class_0_records = [rec for rec in all_npy_records if rec[0] == 0]
    other_records = [rec for rec in all_npy_records if rec[0] != 0]
    
    # Shuffle Class 0
    np.random.shuffle(class_0_records)
    split_idx = int(len(class_0_records) * 0.8)
    
    train_set = class_0_records[:split_idx]
    test_set_normal = class_0_records[split_idx:]
    test_set = test_set_normal + other_records
    
    # Sort for deterministic behavior
    train_set.sort(key=lambda x: x[1])
    test_set.sort(key=lambda x: x[1])
    
    print(f"Train Set (Class 0): {len(train_set)} sequences")
    print(f"Test Set (Total): {len(test_set)} sequences")
    print(f"  - Class 0 (Normal): {len(test_set_normal)}")
    print(f"  - Class 1-6 (Anomaly): {len(other_records)}")
    
    create_bin_dataset(train_set, TRAIN_DIR, "Train")
    create_bin_dataset(test_set, TEST_DIR, "Test")
    
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
