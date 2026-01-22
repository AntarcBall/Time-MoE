import sys
import os
import torch
import numpy as np
import json
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainingArguments
from sklearn.metrics import f1_score, precision_recall_curve

# Add Time-MoE to path
sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))

from time_moe.runner import TimeMoeRunner
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.utils.log_util import log_in_local_rank_0
from time_moe.datasets.time_moe_dataset import TimeMoEDataset, binary_search
from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset

# Global Helpers
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
    f_beta_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
    
    best_idx = np.argmax(f_beta_scores)
    return f_beta_scores[best_idx], thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def compute_normal_stats(model, dataloader, num_batches=20):
    hidden_sums = None
    count = 0
    device = model.device
    # [Layer, Expert] matrix
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
                    # layer_logits: [batch, seq, num_experts]
                    _, selected = torch.topk(layer_logits, model.config.num_experts_per_tok, dim=-1)
                    # selected: [batch, seq, k]
                    
                    # Flatten and count
                    counts = torch.bincount(selected.flatten(), minlength=num_experts)
                    expert_counts[layer_idx] += counts

    if count == 0:
        return torch.zeros(model.config.hidden_size, device=device), 0.0, torch.zeros((num_layers, num_experts), device=device)
        
    mean_vector = hidden_sums / count
    
    # Calculate global balance score for logging (aggregated)
    total_usage = expert_counts.sum(dim=0)
    usage_ratio = total_usage / (total_usage.sum() + 1e-10)
    gating_balance = usage_ratio.std() / (usage_ratio.mean() + 1e-10)
    
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

class AuxLossWarmupCallback(TrainerCallback):
    def __init__(self, target=0.01, warmup_ratio=0.10):
        self.target = target
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = 100 # Default fallback

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

        # Update both config and model attribute if present
        if hasattr(model.config, "router_aux_loss_factor"):
            model.config.router_aux_loss_factor = factor
        
        # Some implementations might copy it to the model instance
        if hasattr(model, "router_aux_loss_factor"):
            model.router_aux_loss_factor = factor

def init_router_weights(model):
    print("[Agent] Re-initializing Router weights for stability...")
    for name, module in model.named_modules():
        # Identify the gating layer (usually 'gate' or 'router')
        if "gate" in name and isinstance(module, torch.nn.Linear):
            # Init with small std (0.01) and 0 bias
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

class AgentCallback(TrainerCallback):
    def __init__(self, trainer, test_ds, batch_size):
        self.trainer = trainer
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.last_eval_step = 0
        self.eval_interval_steps = 1000  # Evaluate every 1000 steps (approx 15-20 mins)
        self.last_save_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # Immediate verification at step 100, then every 1000 steps, or every 60 minutes (3600s)
        time_elapsed = time.time() - self.last_save_time
        if (state.global_step == 100) or (state.global_step - self.last_eval_step >= self.eval_interval_steps) or (time_elapsed >= 3600):
            self.last_save_time = time.time()
            print(f"\n[Agent] Evaluation Triggered (Step {state.global_step}, Time {time_elapsed:.1f}s)...")
            self.evaluate()
            # Save checkpoint
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-step-{state.global_step}")
            self.trainer.save_model(ckpt_path)
            self.last_eval_step = state.global_step
            
    def evaluate(self):
        model = self.trainer.model
        model.eval()
        
        # 1. Compute Normal Stats (Train Set)
        train_loader = DataLoader(self.trainer.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.trainer.data_collator)
        mean_vector, gating_balance, expert_counts = compute_normal_stats(model, train_loader)
        
        # 2. Compute Anomaly Scores (Test Set)
        test_loader = DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=self.trainer.data_collator)
        scores_mse, scores_latent, labels = compute_detailed_scores(model, test_loader, mean_vector, self.test_ds, self.batch_size, limit=50)
        
        f1_l1, _ = search_best_f1(scores_mse, labels)
        f1_l2, _ = search_best_f1(scores_latent, labels)
        f1_total, _ = search_best_f1(scores_mse + 0.1 * scores_latent, labels)
        
        curr_loss = 0.0
        for entry in reversed(self.trainer.state.log_history):
            if 'loss' in entry:
                curr_loss = entry['loss']
                break

        step = self.trainer.state.global_step
        print(f"\n| Step | Loss | Gating | F1-L1 (MSE) | F1-L2 (Latent) | F1-Total |")
        print(f"| {step:4d} | {curr_loss:.4f} | {gating_balance.item() if torch.is_tensor(gating_balance) else gating_balance:.4f} | {f1_l1:.4f} | {f1_l2:.4f} | {f1_total:.4f} |")
        
        # 3. Save Detailed Results
        try:
            # Create directory: checkpoints/step-XXX/eval_results
            ckpt_path = os.path.join(self.trainer.args.output_dir, f"checkpoint-step-{step}")
            eval_dir = os.path.join(ckpt_path, "eval_results")
            os.makedirs(eval_dir, exist_ok=True)
            
            # Save Metrics
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
                
            # Plot Expert Heatmap
            # expert_counts: [Layers, Experts]
            counts_np = expert_counts.cpu().numpy()
            # Normalize per layer for better visualization
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='tiny', choices=['full', 'base', 'tiny'])
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    args = parser.parse_args()

    if args.config == 'full':
        print("Using FULL config (RTX 3090 mode - 2.4B Corrected)")
        CONFIG_PATH, BATCH_SIZE, GRAD_ACCUM, BF16, GRAD_CHK, MAX_LENGTH, OPTIM = 'model_config/config.json', 2, 64, True, True, 2048, "adamw_bnb_8bit"
    elif args.config == 'base':
        print("Using BASE config (Time-MoE 50M - Optimized for 3090 SAFETY)")
        CONFIG_PATH, BATCH_SIZE, GRAD_ACCUM, BF16, GRAD_CHK, MAX_LENGTH, OPTIM = 'model_config/base_50m.json', 8, 16, True, False, 2048, "adamw_torch"
    else:
        print("Using TINY config (Debug mode)")
        CONFIG_PATH, BATCH_SIZE, GRAD_ACCUM, BF16, GRAD_CHK, MAX_LENGTH, OPTIM = 'model_config/tiny_config.json', 4, 8, False, False, 1024, "adamw_torch"

    OUTPUT_DIR, TRAIN_DATA, TEST_DATA = 'checkpoints', 'processed_bin/train', 'processed_bin/val'
    runner = TimeMoeRunner(output_path=OUTPUT_DIR, seed=42)
    config = TimeMoeConfig.from_pretrained(CONFIG_PATH)
    config.output_hidden_states = True 
    model = TimeMoeForPrediction(config)
    
    train_ds = runner.get_train_dataset(TRAIN_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=True)
    # CRITICAL FIX: Ensure test dataset is shuffled to mix Normal/Anomaly in the first few batches
    log_in_local_rank_0('Loading TEST dataset...')
    test_raw_ds = TimeMoEDataset(TEST_DATA, normalization_method="zero")
    log_in_local_rank_0('Processing TEST dataset with shuffle=True...')
    # Using shuffle=True here ensures the sub_seq_indexes are randomized
    test_ds = TimeMoEWindowDataset(test_raw_ds, context_length=MAX_LENGTH, prediction_length=0, stride=MAX_LENGTH, shuffle=True, random_offset=True)
    
    # 2. Re-init router for stability
    init_router_weights(model)

    try:
        import bitsandbytes
    except ImportError:
        if OPTIM == "adamw_bnb_8bit": OPTIM = "adamw_torch"

    from time_moe.trainer.hf_trainer import TimeMoETrainingArguments, TimeMoeTrainer
    training_args = TimeMoETrainingArguments(
        output_dir=OUTPUT_DIR, max_steps=args.steps, per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=1e-4, min_learning_rate=1e-5,
        max_grad_norm=1.0, save_steps=args.eval_steps, 
        bf16=BF16, gradient_checkpointing=GRAD_CHK, dataloader_num_workers=8, dataloader_pin_memory=True,
        dataloader_prefetch_factor=2, remove_unused_columns=False
    )
    
    trainer = TimeMoeT = TimeMoeTrainer(model=model, args=training_args, train_dataset=train_ds)
    
    agent_cb = AgentCallback(trainer, test_ds, BATCH_SIZE)
    agent_cb.eval_interval_steps = args.eval_steps
    trainer.add_callback(agent_cb)
    trainer.add_callback(AuxLossWarmupCallback(target=0.1, warmup_ratio=0.1))
    
    print("Starting Training with Agent...")
    trainer.train()

if __name__ == "__main__":
    main()
