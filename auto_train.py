import sys
import os
import torch
import numpy as np
import json
import time
import argparse
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainingArguments
from sklearn.metrics import f1_score, precision_recall_curve

# Add Time-MoE to path
sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))

from time_moe.runner import TimeMoeRunner
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.datasets.time_moe_dataset import binary_search

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
    expert_counts = torch.zeros(model.config.num_experts, device=device)
    
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

def init_router_weights(model):
    print("[Agent] Re-initializing Router weights for stability...")
    for name, module in model.named_modules():
        if "gate" in name and isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

class AgentCallback(TrainerCallback):
    def __init__(self, trainer, test_ds, batch_size):
        self.trainer = trainer
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.last_eval_time = time.time()
        self.eval_interval_sec = 3 * 3600 
        self.first_run = True

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        elapsed = current_time - self.last_eval_time
        
        # Immediate verification at step 10, then every 3 hours
        if (self.first_run and state.global_step >= 10) or elapsed >= self.eval_interval_sec:
            print(f"\n[Agent] Evaluation Triggered ({'Initial' if self.first_run else '3h Cycle'})...")
            self.evaluate()
            # Save checkpoint
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-step-{state.global_step}")
            self.trainer.save_model(ckpt_path)
            self.last_eval_time = current_time
            self.first_run = False
            
    def evaluate(self):
        model = self.trainer.model
        model.eval()
        
        train_loader = DataLoader(self.trainer.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.trainer.data_collator)
        mean_vector, gating_balance = compute_normal_stats(model, train_loader)
        
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

        print(f"\n| Step | Loss | Gating | F1-L1 (MSE) | F1-L2 (Latent) | F1-Total |")
        print(f"| {self.trainer.state.global_step:4d} | {curr_loss:.4f} | {gating_balance.item() if torch.is_tensor(gating_balance) else gating_balance:.4f} | {f1_l1:.4f} | {f1_l2:.4f} | {f1_total:.4f} |")
        
        if f1_total < 0.950:
            model.time_moe_loss_function.delta *= 0.98
        model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='full', choices=['full', 'base', 'tiny'])
    args = parser.parse_args()

    if args.config == 'full':
        print("Using FULL config (RTX 3090 mode - 2.4B Corrected)")
        CONFIG_PATH, BATCH_SIZE, GRAD_ACCUM, BF16, GRAD_CHK, MAX_LENGTH, OPTIM = 'model_config/config.json', 2, 64, True, True, 2048, "adamw_bnb_8bit"
    elif args.config == 'base':
        print("Using BASE config (Time-MoE 50M - Optimized for 3090 SAFETY)")
        # Option 1.5: Safe Speed
        # Batch 8 (Safe), Accum 16 = 128 effective
        # Optimizer: adamw_torch (Pure GPU, faster than 8bit cpu offload)
        CONFIG_PATH, BATCH_SIZE, GRAD_ACCUM, BF16, GRAD_CHK, MAX_LENGTH, OPTIM = 'model_config/base_50m.json', 8, 16, True, False, 2048, "adamw_torch"
    else:
        print("Using TINY config (Debug mode)")
        CONFIG_PATH, BATCH_SIZE, GRAD_ACCUM, BF16, GRAD_CHK, MAX_LENGTH, OPTIM = 'model_config/tiny_config.json', 4, 8, False, False, 1024, "adamw_torch"

    OUTPUT_DIR, TRAIN_DATA, TEST_DATA = 'checkpoints', 'dataset_bin/train', 'dataset_bin/test'
    runner = TimeMoeRunner(output_path=OUTPUT_DIR, seed=42)
    config = TimeMoeConfig.from_pretrained(CONFIG_PATH)
    config.output_hidden_states = True 
    model = TimeMoeForPrediction(config)
    
    train_ds = runner.get_train_dataset(TRAIN_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=True)
    test_ds = runner.get_train_dataset(TEST_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=False) 
    
    # 2. Re-init router for stability
    init_router_weights(model)

    try:
        import bitsandbytes
    except ImportError:
        if OPTIM == "adamw_bnb_8bit": OPTIM = "adamw_torch"

    from time_moe.trainer.hf_trainer import TimeMoETrainingArguments, TimeMoeTrainer
    training_args = TimeMoETrainingArguments(
        output_dir=OUTPUT_DIR, max_steps=100000, per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=1e-4, min_learning_rate=1e-5,
        warmup_steps=100, max_grad_norm=0.5, optim=OPTIM, logging_steps=1, save_steps=10000, 
        bf16=BF16, gradient_checkpointing=GRAD_CHK, dataloader_num_workers=4, dataloader_pin_memory=True, remove_unused_columns=False
    )
    
    trainer = TimeMoeTrainer(model=model, args=training_args, train_dataset=train_ds)
    trainer.add_callback(AgentCallback(trainer, test_ds, BATCH_SIZE))
    trainer.add_callback(AuxLossWarmupCallback(target=0.1, warmup_ratio=0.1))
    
    print("Starting Training with Agent...")
    trainer.train()

if __name__ == "__main__":
    main()
