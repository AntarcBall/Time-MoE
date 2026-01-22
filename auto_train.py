import sys
import os
import torch
import numpy as np
import json
import time
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import TrainerCallback, TrainingArguments, AutoModelForCausalLM, AutoConfig
from sklearn.metrics import f1_score, precision_recall_curve

# Add Time-MoE to path
sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))

from time_moe.runner import TimeMoeRunner
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.utils.log_util import log_in_local_rank_0
from time_moe.datasets.time_moe_dataset import TimeMoEDataset, binary_search
from time_moe.datasets.time_moe_window_dataset import TimeMoEWindowDataset

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
        # Direct access via shuffled index
        real_idx = self.indices[idx]
        seq = self.dataset[real_idx]
        
        # Ensure numpy float32
        if not isinstance(seq, np.ndarray):
            seq = np.array(seq, dtype=np.float32)
            
        # Safety padding if segment is shorter than max_length + 1
        # Target length is max_length + 1 (input + label)
        target_len = self.max_length + 1
        loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
        
        if len(seq) < target_len:
            n_pad = target_len - len(seq)
            seq = np.pad(seq, (0, n_pad), 'constant', constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)
        elif len(seq) > target_len:
            # Crop if too long (shouldn't happen with current preprocess)
            seq = seq[:target_len]
            loss_mask = loss_mask[:target_len-1]
            
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:],
            'loss_masks': loss_mask
        }

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
                # router_logits is a tuple of tensors (one per layer)
                for layer_idx, layer_logits in enumerate(outputs.router_logits):
                    if torch.isnan(layer_logits).any(): continue
                    # layer_logits: [batch, seq, num_experts]
                    
                    # Top-K selection logic matching the model
                    # Usually Top-2
                    k = model.config.num_experts_per_tok
                    
                    # Apply softmax to get probabilities (optional for counting, but good for debug)
                    # probs = torch.softmax(layer_logits, dim=-1)
                    
                    # Get indices
                    _, selected_indices = torch.topk(layer_logits, k, dim=-1)
                    # selected_indices: [batch, seq, k]
                    
                    # Flatten to [batch * seq * k]
                    flat_indices = selected_indices.flatten()
                    
                    # Count occurrences of each expert (0 to num_experts-1)
                    counts = torch.bincount(flat_indices, minlength=num_experts)
                    
                    # Accumulate
                    expert_counts[layer_idx] += counts.float()

    if count == 0:
        return torch.zeros(model.config.hidden_size, device=device), 0.0, torch.zeros((num_layers, num_experts), device=device)
        
    mean_vector = hidden_sums / count
    
    # Calculate global balance score for logging (aggregated)
    # expert_counts: [Layer, Expert]
    total_usage = expert_counts.sum(dim=0)  # [Experts]
    
    # Avoid division by zero
    total_sum = total_usage.sum()
    if total_sum > 0:
        usage_ratio = total_usage / total_sum
        # Coefficient of Variation: std / mean
        # If mean is 0, it's 0. If perfectly balanced, std is 0 -> balance is 0.
        # Wait, usually high balance score means BAD balance (high variance).
        # Let's use Normalized Entropy instead? Or stick to CV but fix logging.
        # Ideally, we want CV close to 0 (Perfect Balance).
        # But if it's returning 0.00 and heatmap is 0.00, it means counts are ZERO.
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
        
        # DEBUG: Check Label Distribution
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
        MODEL_ID = "Maple728/TimeMoE-2.4B" # Assuming naming convention, need to verify if exists, fallback to 50M
        CONFIG_PATH = 'model_config/config.json'
    elif args.config == 'base':
        print("Using BASE config (Pre-trained TimeMoE-50M)")
        MODEL_ID = "Maple728/TimeMoE-50M"
        CONFIG_PATH = 'model_config/base_50m.json'
    else:
        print("Using TINY config (Fast-Track Debug Mode with 50M weights)")
        # Tiny config will also use 50M weights for transfer learning demo, just with fewer steps
        MODEL_ID = "Maple728/TimeMoE-50M"
        CONFIG_PATH = 'model_config/tiny_config.json'

    # Load Configuration from Local JSON to get training params
    # But we will use the PRE-TRAINED CONFIG for the model structure
    local_config = TimeMoeConfig.from_pretrained(CONFIG_PATH)
    
    # Extract training parameters from local config
    train_conf = getattr(local_config, 'training_config', {})
    
    BATCH_SIZE = train_conf.get('batch_size', 4)
    GRAD_ACCUM = train_conf.get('gradient_accumulation_steps', 8)
    MAX_STEPS = train_conf.get('max_steps', args.steps)
    EVAL_STEPS = train_conf.get('eval_steps', args.eval_steps)
    BF16 = train_conf.get('bf16', False)
    GRAD_CHK = train_conf.get('gradient_checkpointing', False)
    MAX_LENGTH = train_conf.get('max_length', 2048)
    OPTIM = train_conf.get('optim', "adamw_torch")
    
    # Override steps from CLI
    if args.steps != 100000: MAX_STEPS = args.steps
    if args.eval_steps != 1000: EVAL_STEPS = args.eval_steps

    print(f"Configuration Loaded: Batch={BATCH_SIZE}, Accum={GRAD_ACCUM}, Steps={MAX_STEPS}, BF16={BF16}")
    print(f"[Transfer Learning] Loading pre-trained weights from: {MODEL_ID}")

    OUTPUT_DIR, TRAIN_DATA, TEST_DATA = 'checkpoints_transfer' if args.config == 'tiny' else 'checkpoints_transfer_base', 'processed_bin/train', 'processed_bin/val'
    runner = TimeMoeRunner(output_path=OUTPUT_DIR, seed=42)
    
    # LOAD PRE-TRAINED MODEL
    try:
        model = TimeMoeForPrediction.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if BF16 else torch.float32,
            device_map="auto" # Use accelerate to handle placement
        )
        # Ensure config matches for training
        model.config.use_cache = False # Disable cache for training
        model.config.output_hidden_states = True # Needed for our Anomaly Detection logic
        
        # Verify if input_size matches
        if model.config.input_size != 1:
            print(f"Warning: Pre-trained model input_size is {model.config.input_size}, but dataset is univariate (1).")
            # Time-MoE is usually univariate, so this should be fine.
            
    except Exception as e:
        print(f"Error loading pre-trained model {MODEL_ID}: {e}")
        print("Fallback to training from scratch with local config...")
        config = local_config
        config.output_hidden_states = True
        model = TimeMoeForPrediction(config)
    
    train_ds = runner.get_train_dataset(TRAIN_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=True)
    # Optimization: Wrap with DirectDataset if using pre-segmented data
    if "processed_bin" in TRAIN_DATA:
        log_in_local_rank_0('Optimization: Using DirectTimeMoEDataset for TRAIN...')
        train_ds = DirectTimeMoEDataset(train_ds.dataset, MAX_LENGTH, shuffle=True)

    # CRITICAL FIX: Ensure test dataset is shuffled to mix Normal/Anomaly in the first few batches
    log_in_local_rank_0('Loading TEST dataset...')
    test_raw_ds = TimeMoEDataset(TEST_DATA, normalization_method="zero")
    
    # Optimization: Use DirectDataset for Test as well
    log_in_local_rank_0('Optimization: Using DirectTimeMoEDataset for TEST...')
    test_ds = DirectTimeMoEDataset(test_raw_ds, MAX_LENGTH, shuffle=True)
    
    # 2. Re-init router? NO! We want transfer learning.
    # But maybe re-init ONLY the router if we want to learn routing from scratch on new data?
    # Spec says: "Router Weight Re-initialization... to prevent collapse".
    # Since we are fine-tuning on specific single-class data, re-init router is safer 
    # to avoid carrying over bias from general pre-training.
    init_router_weights(model)

    print("[Agent] Applying Partial Freezing Strategy...")
    frozen_params = []
    unfrozen_params = []
    
    for name, param in model.named_parameters():
        # Default unfreeze
        param.requires_grad = True
        
        # Freeze experts AND Attention (MLP layers inside MoE + Self-Attention)
        # Check if parameter is part of the 'experts' ModuleList OR 'self_attn'
        if ("experts." in name and "shared_expert" not in name) or ("self_attn." in name):
             param.requires_grad = False
             frozen_params.append(name)
        else:
             unfrozen_params.append(name)
             
    print(f"[Agent] Frozen {len(frozen_params)} parameters (Sparse Experts + Attention).")
    print(f"[Agent] Unfrozen {len(unfrozen_params)} parameters (Router, Shared, Head, Embed).")

    try:
        import bitsandbytes
    except ImportError:
        if OPTIM == "adamw_bnb_8bit": OPTIM = "adamw_torch"

    from time_moe.trainer.hf_trainer import TimeMoETrainingArguments, TimeMoeTrainer
    training_args = TimeMoETrainingArguments(
        output_dir=OUTPUT_DIR, max_steps=MAX_STEPS, per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM, learning_rate=1e-4, min_learning_rate=1e-5,
        max_grad_norm=1.0, save_steps=EVAL_STEPS, 
        bf16=BF16, gradient_checkpointing=GRAD_CHK, 
        dataloader_num_workers=0, # Revert to 0 for WSL stability
        dataloader_pin_memory=False, # Revert to False for WSL stability
        dataloader_prefetch_factor=None, # Disabled when num_workers=0
        remove_unused_columns=False
    )
    
    trainer = TimeMoeT = TimeMoeTrainer(model=model, args=training_args, train_dataset=train_ds)
    
    agent_cb = AgentCallback(trainer, test_ds, BATCH_SIZE)
    agent_cb.eval_interval_steps = EVAL_STEPS
    trainer.add_callback(agent_cb)
    
    # Scale warmup steps proportionally (10% of total steps)
    warmup_steps = int(MAX_STEPS * 0.1)
    trainer.add_callback(AuxLossWarmupCallback(target=0.1, warmup_ratio=0.1))
    
    print("Starting Training with Agent...")
    trainer.train()

if __name__ == "__main__":
    main()
