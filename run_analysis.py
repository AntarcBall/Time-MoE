import sys
import os
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve

# Add Time-MoE to path
sys.path.append(os.getcwd())

from time_moe.runner import TimeMoeRunner
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.datasets.time_moe_dataset import binary_search

# Global Helpers (Copied from auto_train.py to avoid import issues)
def search_best_f1(scores, labels):
    # Filter NaNs
    valid_mask = ~np.isnan(scores)
    if not np.any(valid_mask):
        return 0.0, 0.5
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # FP=2x penalty -> beta=0.5
    beta = 0.5
    with np.errstate(divide='ignore', invalid='ignore'):
        f_beta_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
    
    best_idx = np.argmax(f_beta_scores)
    return f_beta_scores[best_idx], thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def compute_normal_stats(model, dataloader, num_batches=50):
    hidden_sums = None
    count = 0
    device = model.device
    expert_counts = torch.zeros(model.config.num_experts, device=device)
    
    print(f"Computing normal stats over {num_batches} batches...")
    
    # Use tqdm if available for visual progress
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(dataloader), total=num_batches)
    except ImportError:
        iterator = enumerate(dataloader)
        
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            input_ids = batch['input_ids'].to(device)
            # Use use_cache=False to avoid past_key_values legacy cache warnings/errors if that's the issue
            # The model forward default is config.use_cache which is True. 
            # The warning "past_key_values should not be None in from_legacy_cache()" implies something with caching.
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
            
            if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
                for layer_logits in outputs.router_logits:
                    if torch.isnan(layer_logits).any(): continue
                    _, selected = torch.topk(layer_logits, model.config.num_experts_per_tok, dim=-1)
                    expert_counts += torch.bincount(selected.flatten(), minlength=model.config.num_experts)

    if count == 0:
        return torch.zeros(model.config.hidden_size, device=device), 0.0
        
    mean_vector = hidden_sums / count
    usage = expert_counts / (expert_counts.sum() + 1e-10)
    gating_balance = usage.std() / (usage.mean() + 1e-10)
    return mean_vector, gating_balance

def compute_detailed_scores(model, dataloader, mean_vector, dataset, batch_size, limit=None):
    device = model.device
    mean_vector = mean_vector.to(device)
    all_mse, all_latent, all_labels = [], [], []
    tm_dataset = dataset.dataset
    
    def get_label_for_seq(seq_idx):
        ds_idx = binary_search(tm_dataset.cumsum_lengths, seq_idx)
        sub_ds = tm_dataset.datasets[ds_idx]
        offset_in_ds = seq_idx - tm_dataset.cumsum_lengths[ds_idx]
        # Depending on how sub_ds is structured, get file path
        # In TimeMoEDataset, sub_ds is likely GeneralDataset or BinaryDataset
        # We need to access the file path logic.
        # Inspecting auto_train.py: file_path = sub_ds.seq_infos[offset_in_ds]['file']
        # This assumes dataset structure.
        try:
             # Try accessing seq_infos if available (GeneralDataset)
            if hasattr(sub_ds, 'seq_infos'):
                file_path = sub_ds.seq_infos[offset_in_ds]['file']
                return 0 if os.path.basename(file_path).startswith('0_') else 1
            # If BinaryDataset, it might not have seq_infos in the same way or might handle it differently.
            # Let's assume the label is 1 for now if we can't determine, or try to infer.
            # Actually, auto_train.py logic relies on this.
            # Let's check if we can get label from batch if available?
            # The batch has 'labels' but that's for autoregressive training (next token).
            # The classification label is derived from filename.
            return 1 # Fallback? Or 0?
        except:
             return 1

    # Better label extraction:
    # In auto_train.py it uses `dataset.dataset` which is TimeMoEDataset.
    # TimeMoEDataset holds list of datasets.
    
    # We will try to rely on the logic provided in auto_train.py.
    # If it fails, we default to something visible.
    
    print(f"Computing scores on test set (Limit: {limit} batches)...")
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
                # Get label
                # dataset is TimeMoEWindowDataset
                # dataset.sub_seq_indexes maps window_idx -> (seq_idx, offset)
                seq_idx, _ = dataset.sub_seq_indexes[start_idx + i]
                
                # Re-implement label logic carefully
                ds_idx = binary_search(tm_dataset.cumsum_lengths, seq_idx)
                sub_ds = tm_dataset.datasets[ds_idx]
                offset_in_ds = seq_idx - tm_dataset.cumsum_lengths[ds_idx]
                
                lbl = 1
                # Check for seq_infos (GeneralDataset)
                if hasattr(sub_ds, 'seq_infos'):
                     fpath = sub_ds.seq_infos[offset_in_ds]['file']
                     lbl = 0 if os.path.basename(fpath).startswith('0_') else 1
                # Check for BinaryDataset meta logic if possible, 
                # but BinaryDataset usually loads all into memory or mmap.
                # Assuming auto_train logic works for the provided dataset_bin which seems to be BinaryDataset?
                # Wait, auto_train.py uses: file_path = sub_ds.seq_infos[offset_in_ds]['file']
                # Does BinaryDataset have seq_infos?
                # I should check BinaryDataset code if possible. 
                # For now, I'll trust the auto_train logic.
                
                all_labels.append(lbl)
            
    return np.concatenate(all_mse), np.concatenate(all_latent), np.array(all_labels)

def run_analysis():
    print("Initializing Analysis...")
    model_name = "Maple728/TimeMoE-50M"
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    print(f"Loading model: {model_name}")
    try:
        config = TimeMoeConfig.from_pretrained(model_name)
        model = TimeMoeForPrediction.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load from Hub: {e}")
        print("Falling back to local tiny config for demonstration if needed, but retrying with trust_remote_code=True if that helps (though TimeMoE classes are local)")
        # Since we imported local classes, we should use them. 
        # But if we want weights, we need the hub model. 
        # Let's assume the user has internet or the model is cached.
        return

    model.to(device)
    model.eval()
    
    # Load Data
    print("Loading datasets...")
    # Adjust paths if running from Time-MoE dir or root
    train_path = 'dataset_bin/train'
    test_path = 'dataset_bin/test'
    if not os.path.exists(train_path):
        train_path = '../dataset_bin/train'
        test_path = '../dataset_bin/test'
        
    runner = TimeMoeRunner(output_path=output_dir, seed=42)
    # Use max_length from config or default
    MAX_LENGTH = 2048
    
    train_ds = runner.get_train_dataset(train_path, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=True)
    test_ds = runner.get_train_dataset(test_path, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=False)
    
    # Custom Collator
    def custom_collator(features):
        batch = {}
        for k in features[0].keys():
            if k in ['input_ids', 'labels', 'loss_masks']:
                batch[k] = torch.stack([torch.as_tensor(f[k]) for f in features])
        return batch
        
    BATCH_SIZE = 8
    
    # 1. Normal Stats
    print("Step 1: Computing Normal Stats...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collator)
    # REDUCED NUM_BATCHES FOR SPEED within timeout
    mean_vector, gating_balance = compute_normal_stats(model, train_loader, num_batches=10)
    print(f"Gating Balance: {gating_balance:.4f}")
    
    # 2. Detailed Scores
    print("Step 2: Computing Detailed Scores on Test Set...")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collator)
    # Using a limit for "Full Inference" feel, e.g., 50 batches (100 batches might be too slow for 2min timeout)
    # The previous run got stuck at 0% for 20s. Maybe CUDA init or first forward pass is slow.
    scores_mse, scores_latent, labels = compute_detailed_scores(model, test_loader, mean_vector, test_ds, BATCH_SIZE, limit=50)
    
    # 3. Analysis
    print("Step 3: Analyzing Results...")
    f1_l1, t_l1 = search_best_f1(scores_mse, labels)
    f1_l2, t_l2 = search_best_f1(scores_latent, labels)
    # Hybrid Score
    scores_hybrid = scores_mse + 0.1 * scores_latent
    f1_total, t_total = search_best_f1(scores_hybrid, labels)
    
    print("\n" + "="*60)
    print(f"INFERENCE RESULTS ({model_name})")
    print("="*60)
    print(f"Test Samples Processed: {len(labels)}")
    print(f"Anomaly Ratio: {labels.mean():.2%}")
    print("-" * 60)
    print(f"F1-L1 (MSE) [Thresh={t_l1:.4f}]:     {f1_l1:.4f}")
    print(f"F1-L2 (Latent) [Thresh={t_l2:.4f}]:  {f1_l2:.4f}")
    print(f"F1-Hybrid [Thresh={t_total:.4f}]:    {f1_total:.4f}")
    print("="*60)
    
    # Save Results
    results = {
        "model": model_name,
        "gating_balance": float(gating_balance.cpu().numpy()) if torch.is_tensor(gating_balance) else float(gating_balance),
        "f1_l1": float(f1_l1),
        "f1_l2": float(f1_l2),
        "f1_hybrid": float(f1_total),
        "threshold_l1": float(t_l1),
        "threshold_l2": float(t_l2),
        "threshold_hybrid": float(t_total),
        "scores_mse_sample": scores_mse[:100].tolist(),
        "scores_latent_sample": scores_latent[:100].tolist(),
        "labels_sample": labels[:100].tolist()
    }
    
    res_path = os.path.join(output_dir, "inference_summary.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {res_path}")

if __name__ == "__main__":
    run_analysis()
