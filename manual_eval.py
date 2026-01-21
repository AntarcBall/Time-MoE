import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add Time-MoE to path
sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))

from time_moe.runner import TimeMoeRunner
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from auto_train import compute_normal_stats, compute_detailed_scores, search_best_f1

def manual_eval(ckpt_path):
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Config and Model
    config = TimeMoeConfig.from_pretrained(ckpt_path)
    model = TimeMoeForPrediction.from_pretrained(ckpt_path)
    model.to('cuda')
    model.eval()
    
    # Data
    TRAIN_DATA = 'dataset_bin/train'
    TEST_DATA = 'dataset_bin/test'
    MAX_LENGTH = 2048
    BATCH_SIZE = 2
    
    runner = TimeMoeRunner(output_path='eval_results', seed=42)
    train_ds = runner.get_train_dataset(TRAIN_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=True)
    test_ds = runner.get_train_dataset(TEST_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=False)
    
    # 1. Compute Normal Stats (Mean and Balance)
    print("Computing Normal statistics from training data...")
    # TimeMoeRunner provides data_collator in TimeMoeTrainer, but we can just use default torch collator
    # since TimeMoEDataset returns dicts with tensors
    def custom_collator(features):
        batch = {}
        for k in features[0].keys():
            if k in ['input_ids', 'labels', 'loss_masks']:
                batch[k] = torch.stack([torch.as_tensor(f[k]) for f in features])
        return batch

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collator)
    mean_vector, gating_balance = compute_normal_stats(model, train_loader, num_batches=30)
    
    # 2. Detailed Inference on Test Set
    print("Running detailed inference on test set (Class 0-6)...")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collator)
    # Limit for speed, but enough for meaningful F1
    scores_mse, scores_latent, labels = compute_detailed_scores(model, test_loader, mean_vector, test_ds, BATCH_SIZE, limit=200)
    
    # 3. Search for best F1 per score type
    f1_l1, t_l1 = search_best_f1(scores_mse, labels)
    f1_l2, t_l2 = search_best_f1(scores_latent, labels)
    f1_total, t_total = search_best_f1(scores_mse + 0.1 * scores_latent, labels)
    
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS FOR {os.path.basename(ckpt_path)}")
    print("="*50)
    print(f"F1-L1 (Prediction MSE):   {f1_l1:.4f}")
    print(f"F1-L2 (Latent Distance): {f1_l2:.4f}")
    print(f"F1-Total (Hybrid):       {f1_total:.4f}")
    print("-" * 50)
    print(f"Gating Balance (STD/Mean): {gating_balance.item() if torch.is_tensor(gating_balance) else gating_balance:.4f}")
    print("="*50)

if __name__ == "__main__":
    latest_ckpt = "checkpoints/checkpoint-step-10"
    manual_eval(latest_ckpt)
