✦ Time-MoE Performance Analysis Report

  Executive Summary

  After extensive analysis of the Time-MoE repository and its performance issues, I've identified the
  root causes of the extremely slow training (4 iterations per second) and the optimizations already
  implemented by the developers.

  Current State Analysis

  1. Dataset Pipeline Architecture
  The data flows through the following pipeline:
   - Source: CSV files in Renomeado/ directory
   - Preprocessing: preprocess_balanced.py converts CSV to 2048-point segments
   - Binary Conversion: Segments saved as .npy files then converted to binary format
   - Training Access: BinaryDataset → TimeMoEDataset → TimeMoEWindowDataset → DataLoader

  2. Parameter Configuration
  The current configuration in auto_train.py is:
   - context_length = 2048 (MAX_LENGTH for base config)
   - prediction_length = 0 (set in runner.py)
   - stride = 2048 (MAX_LENGTH, passed as third parameter)
   - random_offset = True (for training data)

  3. Windowing Behavior
  With the current settings:
   - window_size = context_length + prediction_length = 2048 + 0 = 2048
   - window_size_plus_one = 2049
   - For sequences of exactly 2048 points: n_points > window_size_plus_one is false (2048 > 2049)
   - This means the condition for random offset is not met, and only one window per sequence is created
   - The theoretical setup is already correct for "segment mode"

  Key Findings

  1. Pre-existing Optimizations
  The developers have already implemented several critical optimizations:
   - WSL I/O Optimization: dataloader_num_workers=0 to avoid multiprocessing overhead
   - Memory Optimization: dataloader_pin_memory=False to reduce memory pinning overhead
   - Prefetch Optimization: dataloader_prefetch_factor=None when num_workers=0

  2. The Real Performance Bottleneck
  Despite the theoretical correctness of the configuration, the performance issue likely stems from:

   1. Random Offset Overhead: Even though the condition isn't met for additional windows, the
      random_offset=True parameter still introduces computational overhead during dataset initialization.

   2. TimeMoEWindowDataset Overhead: The wrapper still performs index calculations and metadata management
      even when no additional windowing occurs.

   3. Binary Dataset File Operations: Each sequence access still requires file I/O operations through the
      BinaryDataset's seek/read mechanism.

  3. Architectural Misalignment
  The current architecture applies TimeMoEWindowDataset as a generic wrapper for "long sequence → fixed
  windows" transformation, but the data is already pre-segmented into the correct size. This creates
  unnecessary abstraction overhead without benefit.

  Recommended Optimizations

  1. Bypass TimeMoEWindowDataset (Primary Solution)
  Create a direct dataset wrapper that accesses pre-segmented data without additional windowing:

    1 class DirectTimeMoEDataset:
    2     def __init__(self, base_dataset, max_length):
    3         self.base_dataset = base_dataset
    4         self.max_length = max_length
    5         
    6     def __len__(self):
    7         return len(self.base_dataset)
    8     
    9     def __getitem__(self, idx):
   10         seq = self.base_dataset[idx]
   11         seq = np.array(seq, dtype=np.float32)
   12         
   13         # Apply the same transformation as TimeMoEWindowDataset
   14         loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
   15         n_pad = (self.max_length + 1) - len(seq)
   16         
   17         if n_pad > 0:
   18             seq = np.pad(seq, (0, n_pad), 'constant', constant_values=0)
   19             loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)
   20             
   21         return {
   22             'input_ids': seq[:-1],
   23             'labels': seq[1:],
   24             'loss_masks': loss_mask
   25         }

  2. Eliminate Random Offset for Training
  Set random_offset=False for training data since the segments are already properly sized.

  3. Optimize Binary Dataset I/O
   - Implement file handle caching in BinaryDataset
   - Use memory mapping (mmap) for faster random access
   - Pre-compute index/offset mappings during initialization

  Performance Impact Assessment

  The current setup creates unnecessary computational overhead despite having the correct mathematical
  parameters. The TimeMoEWindowDataset performs the following operations for each sequence:
   1. Index calculation and storage
   2. Potential random offset computation (even if not used)
   3. Slice operations for window extraction
   4. Padding and masking operations

  By bypassing this wrapper for pre-segmented data, we can achieve:
   - Reduced memory allocation overhead
   - Eliminated index management overhead
   - Direct access to pre-sized segments
   - Improved cache locality

  Conclusion

  The Time-MoE repository has already implemented several I/O optimizations for WSL environments, but
  the fundamental architectural mismatch between the generic TimeMoEWindowDataset wrapper and
  pre-segmented data remains a performance bottleneck. The most effective optimization would be to
  bypass the TimeMoEWindowDataset entirely when working with pre-segmented binary data, treating each
  segment as a single training sample directly.

  The current configuration parameters are mathematically correct for preventing overlapping windows,
  but the abstraction overhead still impacts performance. A direct dataset access pattern would
  eliminate this overhead while maintaining the same training semantics.