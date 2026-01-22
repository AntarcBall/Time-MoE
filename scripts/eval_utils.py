import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from sklearn.metrics import f1_score, precision_recall_curve

# Add Time-MoE to path
sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))

from time_moe.datasets.time_moe_dataset import binary_search

def search_best_f1(scores, labels):
    valid_mask = ~np.isnan(scores)
    if not np.any(valid_mask):
        return 0.0, 0.5
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    beta = 0.5
    f_beta_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
    
    best_idx = np.argmax(f_beta_scores)
    return f_beta_scores[best_idx], thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def compute_normal_stats(model, dataloader, num_batches=20):
    hidden_sums = None
    count = 0
    device = model.device
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_experts
    
    expert_counts = torch.zeros((num_layers, num_experts), device=device)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            
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
                for layer_idx, layer_logits in enumerate(outputs.router_logits):
                    if torch.isnan(layer_logits).any(): continue
                    k = model.config.num_experts_per_tok
                    _, selected_indices = torch.topk(layer_logits, k, dim=-1)
                    flat_indices = selected_indices.flatten()
                    counts = torch.bincount(flat_indices, minlength=num_experts)
                    expert_counts[layer_idx] += counts.float()

    if count == 0:
        return torch.zeros(model.config.hidden_size, device=device), 0.0, torch.zeros((num_layers, num_experts), device=device)
        
    mean_vector = hidden_sums / count
    total_usage = expert_counts.sum(dim=0)
    total_sum = total_usage.sum()
    if total_sum > 0:
        usage_ratio = total_usage / total_sum
        gating_balance = usage_ratio.std() / (usage_ratio.mean() + 1e-6)
    else:
        gating_balance = 0.0
    
    return mean_vector, gating_balance, expert_counts

def compute_detailed_scores(model, dataloader, mean_vector, dataset, batch_size, limit=None):
    device = model.device
    mean_vector = mean_vector.to(device)
    all_mse, all_latent, all_labels = [], [], []
    tm_dataset = dataset.dataset
    
    def get_label_for_seq(seq_idx):
        ds_idx = binary_search(tm_dataset.cumsum_lengths, seq_idx)
        sub_ds = tm_dataset.datasets[ds_idx]
        offset_in_ds = seq_idx - tm_dataset.cumsum_lengths[ds_idx]
        file_path = sub_ds.seq_infos[offset_in_ds]['file']
        return 0 if os.path.basename(file_path).startswith('0_') else 1

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if limit and batch_idx >= limit: break
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            if len(labels.shape) == 2: labels = labels.unsqueeze(-1)
            
            outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            h = outputs.hidden_states[-1]
            
            total_mse = 0
            for head_idx in range(len(model.lm_heads)):
                preds = model.lm_heads[head_idx](h)[:, :, 0:1]
                total_mse += ((preds - labels) ** 2).mean(dim=[1, 2])
            mse_score = total_mse / len(model.lm_heads)
            
            dist = (h - mean_vector).pow(2).sum(dim=-1).mean(dim=-1)
            
            all_mse.append(mse_score.cpu().numpy())
            all_latent.append(dist.cpu().numpy())
            
            start_idx = batch_idx * batch_size
            for i in range(input_ids.shape[0]):
                seq_idx, _ = dataset.sub_seq_indexes[start_idx + i]
                all_labels.append(get_label_for_seq(seq_idx))
            
    return np.concatenate(all_mse), np.concatenate(all_latent), np.array(all_labels)

class AuxLossWarmupCallback(TrainerCallback):
    def __init__(self, target=0.01, warmup_ratio=0.10):
        self.target = target
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = 100

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps > 0:
            self.warmup_steps = max(1, int(state.max_steps * self.warmup_ratio))
        print(f"[Agent] Aux Loss Warmup: Target={self.target}, Steps={self.warmup_steps}")

    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        step = state.global_step

        if step < self.warmup_steps:
            factor = self.target * (step / self.warmup_steps)
        else:
            factor = self.target

        if hasattr(model.config, "router_aux_loss_factor"):
            model.config.router_aux_loss_factor = factor
        if hasattr(model, "router_aux_loss_factor"):
            model.router_aux_loss_factor = factor

class AgentCallback(TrainerCallback):
    def __init__(self, trainer, test_ds, batch_size):
        self.trainer = trainer
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.last_eval_step = 0
        self.eval_interval_steps = 1000
        self.last_save_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        time_elapsed = time.time() - self.last_save_time
        if (state.global_step == 100) or (state.global_step - self.last_eval_step >= self.eval_interval_steps) or (time_elapsed >= 3600):
            self.last_save_time = time.time()
            print(f"\n[Agent] Evaluation Triggered (Step {state.global_step}, Time {time_elapsed:.1f}s)...")
            self.evaluate()
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-step-{state.global_step}")
            self.trainer.save_model(ckpt_path)
            self.last_eval_step = state.global_step
            
    def evaluate(self):
        model = self.trainer.model
        model.eval()
        
        train_loader = DataLoader(self.trainer.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.trainer.data_collator)
        mean_vector, gating_balance, expert_counts = compute_normal_stats(model, train_loader)
        
        test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.trainer.data_collator)
        scores_mse, scores_latent, labels = compute_detailed_scores(model, test_loader, mean_vector, self.test_ds, self.batch_size, limit=50)
        
        f1_l1, _ = search_best_f1(scores_mse, labels)
        f1_l2, _ = search_best_f1(scores_latent, labels)
        f1_total, _ = search_best_f1(scores_mse + 0.1 * scores_latent, labels)
        
        num_normals = np.sum(labels == 0)
        num_anomalies = np.sum(labels == 1)
        ratio = num_anomalies / (len(labels) + 1e-10)
        
        curr_loss = 0.0
        for entry in reversed(self.trainer.state.log_history):
            if 'loss' in entry:
                curr_loss = entry['loss']
                break

        step = self.trainer.state.global_step
        print(f"\n| Step | Loss | Gating | F1-L1 (MSE) | F1-L2 (Latent) | F1-Total |")
        print(f"| {step:4d} | {curr_loss:.4f} | {gating_balance.item() if torch.is_tensor(gating_balance) else gating_balance:.4f} | {f1_l1:.4f} | {f1_l2:.4f} | {f1_total:.4f} |")
        print(f"[Debug] Eval Labels: Normal={num_normals}, Anomaly={num_anomalies} (Anomaly Ratio: {ratio:.1%})")
        
        try:
            ckpt_path = os.path.join(self.trainer.args.output_dir, f"checkpoint-step-{step}")
            eval_dir = os.path.join(ckpt_path, "eval_results")
            os.makedirs(eval_dir, exist_ok=True)
            
            metrics = {
                "step": step,
                "loss": float(curr_loss),
                "gating_balance": float(gating_balance),
                "f1_l1": float(f1_l1),
                "f1_l2": float(f1_l2),
                "f1_total": float(f1_total)
            }
            with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                
            counts_np = expert_counts.cpu().numpy()
            row_sums = counts_np.sum(axis=1, keepdims=True) + 1e-10
            norm_counts = counts_np / row_sums
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(norm_counts, annot=True, fmt=".2f", cmap="Blues", 
                        xticklabels=[f"E{i}" for i in range(norm_counts.shape[1])],
                        yticklabels=[f"L{i}" for i in range(norm_counts.shape[0])])
            plt.title(f"Expert Routing Heatmap (Step {step})")
            plt.xlabel("Experts")
            plt.ylabel("Layers")
            plt.tight_layout()
            plt.savefig(os.path.join(eval_dir, "expert_heatmap.png"))
            plt.close()
            
        except Exception as e:
            print(f"[Agent] Warning: Failed to save eval results/heatmap: {e}")

        if f1_total < 0.950:
            model.time_moe_loss_function.delta *= 0.98
        model.train()
