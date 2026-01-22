import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

# Add Time-MoE to path
sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))

from time_moe.utils.log_util import log_in_local_rank_0
from time_moe.datasets.time_moe_dataset import TimeMoEDataset

class DirectTimeMoEDataset:
    """
    Optimization: Direct wrapper for pre-segmented binary datasets.
    Bypasses the heavy TimeMoEWindowDataset logic since data is already chopped to correct size.
    """
    def __init__(self, base_dataset, max_length, shuffle=False):
        self.dataset = base_dataset
        self.max_length = max_length
        self.indices = list(range(len(base_dataset)))
        if shuffle:
            random.shuffle(self.indices)
            
        # Mocking sub_seq_indexes for compatibility with compute_detailed_scores
        # format: (seq_idx, offset)
        self.sub_seq_indexes = [(i, 0) for i in self.indices]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        seq = self.dataset[real_idx]
        if not isinstance(seq, np.ndarray):
            seq = np.array(seq, dtype=np.float32)
            
        target_len = self.max_length + 1
        loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
        
        if len(seq) < target_len:
            n_pad = target_len - len(seq)
            seq = np.pad(seq, (0, n_pad), 'constant', constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)
        elif len(seq) > target_len:
            seq = seq[:target_len]
            loss_mask = loss_mask[:target_len-1]
            
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:],
            'loss_masks': loss_mask
        }

def prepare_datasets(runner, config):
    train_ds = runner.get_train_dataset(config['train_data'], config['max_length'], config['max_length'], "zero", random_offset=True)
    
    if "processed_bin" in config['train_data']:
        log_in_local_rank_0('Optimization: Using DirectTimeMoEDataset for TRAIN...')
        train_ds = DirectTimeMoEDataset(train_ds.dataset, config['max_length'], shuffle=True)

    log_in_local_rank_0('Loading TEST dataset...')
    test_raw_ds = TimeMoEDataset(config['test_data'], normalization_method="zero")
    
    log_in_local_rank_0('Optimization: Using DirectTimeMoEDataset for TEST...')
    test_ds = DirectTimeMoEDataset(test_raw_ds, config['max_length'], shuffle=True)
    
    return train_ds, test_ds
