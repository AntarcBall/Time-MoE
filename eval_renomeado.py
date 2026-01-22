import sys
import os
import torch
import numpy as np
import json
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))

from time_moe.runner import TimeMoeRunner
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.datasets.time_moe_dataset import binary_search

# Evaluation Logic
def search_best_f1(scores, labels):
    valid_mask = ~np.isnan(scores)
    if not np.any(valid_mask):
        return 0.0, 0.5
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    beta = 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        f_beta_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
    
    best_idx = np.argmax(f_beta_scores)
    return f_beta_scores[best_idx], thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def compute_normal_stats(model, dataloader, num_batches=20):
    hidden_sums = None
    count = 0
    device = model.device
    
    print(f"Computing normal stats over {num_batches} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            input_ids = batch['input_ids'].to(device)
            # Disable cache for inference
            outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True, use_cache=False)
            
            h = outputs.hidden_states[-1] 
            if torch.isnan(h).any(): continue
                
            curr_sum = h.sum(dim=[0, 1])
            curr_count = h.shape[0] * h.shape[1]
            
            if hidden_sums is None:
                hidden_sums = curr_sum
            else:
                hidden_sums += curr_sum
            count += curr_count

    if count == 0:
        return torch.zeros(model.config.hidden_size, device=device)
        
    mean_vector = hidden_sums / count
    return mean_vector

def compute_detailed_scores(model, dataloader, mean_vector, dataset, batch_size, limit=None):
    device = model.device
    mean_vector = mean_vector.to(device)
    all_mse, all_latent, all_labels = [], [], []
    tm_dataset = dataset.dataset
    
    def get_label_for_seq(seq_idx):
        try:
            ds_idx = binary_search(tm_dataset.cumsum_lengths, seq_idx)
            sub_ds = tm_dataset.datasets[ds_idx]
            offset_in_ds = seq_idx - tm_dataset.cumsum_lengths[ds_idx]
            
            label = 1
            if hasattr(sub_ds, 'meta_infos'):
                meta = sub_ds.meta_infos[offset_in_ds]
                if 'class' in meta:
                    label = 0 if meta['class'] == 0 else 1
                elif 'file' in meta:
                    label = 0 if os.path.basename(meta['file']).startswith('0_') else 1
            elif hasattr(sub_ds, 'seq_infos'):
                file_path = sub_ds.seq_infos[offset_in_ds]['file']
                label = 0 if os.path.basename(file_path).startswith('0_') else 1
            
            return label
        except:
            return 1

    print(f"Scoring test set (Limit: {limit})...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if limit and batch_idx >= limit: break
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            if len(labels.shape) == 2: labels = labels.unsqueeze(-1)
            
            outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True, use_cache=False)
            h = outputs.hidden_states[-1]
            
            # L1: MSE
            total_mse = 0
            for head_idx in range(len(model.lm_heads)):
                preds = model.lm_heads[head_idx](h)[:, :, 0:1]
                total_mse += ((preds - labels) ** 2).mean(dim=[1, 2])
            mse_score = total_mse / len(model.lm_heads)
            
            # L2: Latent
            dist = (h - mean_vector).pow(2).sum(dim=-1).mean(dim=-1)
            
            all_mse.append(mse_score.cpu().numpy())
            all_latent.append(dist.cpu().numpy())
            
            start_idx = batch_idx * batch_size
            for i in range(input_ids.shape[0]):
                seq_idx, _ = dataset.sub_seq_indexes[start_idx + i]
                all_labels.append(get_label_for_seq(seq_idx))
            
    return np.concatenate(all_mse), np.concatenate(all_latent), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints_renomeado/checkpoint-step-50')
    args = parser.parse_args()
    
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}. Looking for others...")
        # Fallback to output dir if checkpoint not explicitly found
        if os.path.exists('checkpoints_renomeado'):
            subs = [os.path.join('checkpoints_renomeado', d) for d in os.listdir('checkpoints_renomeado') if 'checkpoint' in d]
            if subs:
                ckpt_path = sorted(subs, key=lambda x: int(x.split('-')[-1]))[-1]
                print(f"Using latest checkpoint: {ckpt_path}")
            else:
                print("No checkpoints found. Exiting.")
                return
    
    print(f"Loading model from {ckpt_path}")
    config = TimeMoeConfig.from_pretrained(ckpt_path)
    model = TimeMoeForPrediction.from_pretrained(ckpt_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Custom Collator
    def custom_collator(features):
        batch = {}
        for k in features[0].keys():
            if k in ['input_ids', 'labels', 'loss_masks']:
                batch[k] = torch.stack([torch.as_tensor(f[k]) for f in features])
        return batch

    runner = TimeMoeRunner(output_path='eval_results', seed=42)
    MAX_LENGTH = 2048
    BATCH_SIZE = 8
    
    TRAIN_DATA = 'Time-MoE/dataset_renomeado/train'
    TEST_DATA = 'Time-MoE/dataset_renomeado/test'
    
    train_ds = runner.get_train_dataset(TRAIN_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=True)
    test_ds = runner.get_train_dataset(TEST_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collator)
    mean_vector = compute_normal_stats(model, train_loader, num_batches=20)
    
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collator)
    # Full eval on test set (or limited if large)
    scores_mse, scores_latent, labels = compute_detailed_scores(model, test_loader, mean_vector, test_ds, BATCH_SIZE, limit=200)
    
    # Analysis
    f1_l1, t_l1 = search_best_f1(scores_mse, labels)
    f1_l2, t_l2 = search_best_f1(scores_latent, labels)
    f1_hybrid, t_hybrid = search_best_f1(scores_mse + 0.1 * scores_latent, labels)
    
    try:
        auc_l1 = roc_auc_score(labels, scores_mse)
        auc_l2 = roc_auc_score(labels, scores_latent)
    except:
        auc_l1, auc_l2 = 0.0, 0.0
    
    print("\n" + "="*60)
    print(f"ANOMALY DETECTION RESULTS (Renomeado Class 0 vs 1-6)")
    print("="*60)
    print(f"Test Samples: {len(labels)}")
    print(f"Normal: {np.sum(labels==0)}, Anomaly: {np.sum(labels==1)}")
    print("-" * 60)
    print(f"MSE Score (L1):")
    print(f"  F1-Score: {f1_l1:.4f} (Threshold: {t_l1:.4f})")
    print(f"  AUC-ROC:  {auc_l1:.4f}")
    print("-" * 60)
    print(f"Latent Score (L2):")
    print(f"  F1-Score: {f1_l2:.4f} (Threshold: {t_l2:.4f})")
    print(f"  AUC-ROC:  {auc_l2:.4f}")
    print("-" * 60)
    print(f"Hybrid Score (L1 + 0.1*L2):")
    print(f"  F1-Score: {f1_hybrid:.4f} (Threshold: {t_hybrid:.4f})")
    print("="*60)

if __name__ == "__main__":
    main()
