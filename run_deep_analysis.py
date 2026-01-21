import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc

# Add path
sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))
from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig
from time_moe.runner import TimeMoeRunner
from auto_train import compute_detailed_scores, compute_normal_stats

def run_analysis(ckpt_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading model from {ckpt_path}...")
    
    try:
        model = TimeMoeForPrediction.from_pretrained(ckpt_path)
        model.to('cuda')
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Data Loading (Sample)
    TRAIN_DATA = 'dataset_bin/train'
    TEST_DATA = 'dataset_bin/test'
    MAX_LENGTH = 2048
    BATCH_SIZE = 4
    
    runner = TimeMoeRunner(output_path='temp', seed=42)
    # Load small subset for analysis speed
    test_ds = runner.get_train_dataset(TEST_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=False)
    
    # 1. Normal Stats
    print("Computing Normal Stats...")
    
    # Custom Collator Definition
    def custom_collator(features):
        batch = {}
        for k in features[0].keys():
            if k in ['input_ids', 'labels', 'loss_masks']:
                batch[k] = torch.stack([torch.as_tensor(f[k]) for f in features])
        return batch

    # Use test_ds normal subset ideally, but here we use loaded mean from training logic if saved, 
    # or recompute quickly from a few batches of train data.
    train_ds = runner.get_train_dataset(TRAIN_DATA, MAX_LENGTH, MAX_LENGTH, "zero", random_offset=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collator)
    mean_vector, _ = compute_normal_stats(model, train_loader, num_batches=20)
    
    # 2. Run Inference
    print("Running Inference on Test Set...")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collator)
    # Analyze first 200 batches (~800 samples) to save time
    mse_scores, latent_scores, labels = compute_detailed_scores(model, test_loader, mean_vector, test_ds, BATCH_SIZE, limit=200)
    
    # --- Analysis 1: Score Distribution ---
    print("Generating Score Distribution Plot...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(x=mse_scores, hue=labels, bins=50, kde=True, palette={0: 'blue', 1: 'red'}, alpha=0.5)
    plt.title("MSE Score Distribution (Normal vs Anomaly)")
    plt.xlabel("MSE Score")
    
    plt.subplot(1, 2, 2)
    sns.histplot(x=latent_scores, hue=labels, bins=50, kde=True, palette={0: 'blue', 1: 'red'}, alpha=0.5)
    plt.title("Latent Distance Distribution")
    plt.xlabel("Latent Distance")
    plt.savefig(os.path.join(output_dir, "1_score_distribution.png"))
    plt.close()
    
    # --- Analysis 2: Precision-Recall Curve (F-0.5 Focus) ---
    print("Generating PR Curve...")
    total_scores = mse_scores + 0.1 * latent_scores
    precision, recall, thresholds = precision_recall_curve(labels, total_scores)
    beta = 0.5
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
    best_idx = np.argmax(f_beta)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='PR Curve')
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label=f'Best F-0.5 (Thresh={thresholds[best_idx]:.4f})')
    plt.title(f"Precision-Recall Curve (Max F-0.5 = {f_beta[best_idx]:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "5_pr_curve.png"))
    plt.close()
    
    # --- Analysis 3: FFT Spectrum Analysis (Sample) ---
    print("Generating FFT Spectrum...")
    # Get one batch
    batch = next(iter(test_loader))
    input_ids = batch['input_ids'].to('cuda')
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        # Prediction from first head (Short term)
        pred = model.lm_heads[0](outputs.hidden_states[-1])
    
    # Take first sample in batch, first channel
    real_sig = input_ids[0, :, 0].cpu().numpy()
    pred_sig = pred[0, :, 0].cpu().numpy()
    
    fft_real = np.abs(np.fft.rfft(real_sig))
    fft_pred = np.abs(np.fft.rfft(pred_sig))
    freqs = np.fft.rfftfreq(len(real_sig))
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_real, label='Real Signal', alpha=0.7)
    plt.plot(freqs, fft_pred, label='Predicted Signal', alpha=0.7, linestyle='--')
    plt.yscale('log')
    plt.title("Frequency Domain Analysis (FFT Spectrum)")
    plt.xlabel("Frequency")
    plt.ylabel("Energy (Log Scale)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "3_fft_spectrum.png"))
    plt.close()
    
    # --- Analysis 4: Expert Heatmap (Gating via Hook) ---
    print("Generating Expert Heatmap with Hook...")
    
    # Hook to capture router logits directly from the module
    captured_logits = []
    def get_router_logits_hook(module, input, output):
        # output is likely logits or (hidden, logits)
        # TimeMoeSparseExpertsLayer.forward returns (final_hidden_states, router_logits)
        if isinstance(output, tuple) and len(output) > 1:
            captured_logits.append(output[1].detach().cpu())
        elif torch.is_tensor(output):
            captured_logits.append(output.detach().cpu())

    # Find the last MoE layer
    # Structure: model -> model (TimeMoeModel) -> layers -> [TimeMoeDecoderLayer] -> ffn_layer (TimeMoeSparseExpertsLayer)
    target_layer = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        last_layer = model.model.layers[-1]
        if hasattr(last_layer, 'ffn_layer'):
            target_layer = last_layer.ffn_layer
    
    if target_layer:
        handle = target_layer.register_forward_hook(get_router_logits_hook)
        
        # Run forward
        with torch.no_grad():
            _ = model(input_ids=input_ids)
        
        handle.remove()
        
        if captured_logits:
            last_layer_logits = captured_logits[-1] # [Batch*Seq, Experts]
            
            # Reshape logic
            seq_len = input_ids.shape[1]
            batch_size = input_ids.shape[0]
            
            if last_layer_logits.dim() == 2 and last_layer_logits.shape[0] == batch_size * seq_len:
                sample_logits = last_layer_logits.view(batch_size, seq_len, -1)[0].float().numpy()
            else:
                # Just take first N elements that match seq_len
                sample_logits = last_layer_logits[:seq_len].float().numpy() if last_layer_logits.shape[0] >= seq_len else None

            if sample_logits is not None:
                # Softmax
                exp_logits = np.exp(sample_logits - np.max(sample_logits, axis=1, keepdims=True))
                weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                
                plt.figure(figsize=(12, 6))
                sns.heatmap(weights.T, cmap="viridis", cbar_kws={'label': 'Gate Probability'})
                plt.title("Expert Activation Map (Time vs Expert)")
                plt.xlabel("Time Step")
                plt.ylabel("Expert Index")
                plt.savefig(os.path.join(output_dir, "2_expert_heatmap.png"))
                plt.close()
                print("Heatmap saved via Hook.")
            else:
                print("Failed to reshape captured logits.")
        else:
            print("Hook captured nothing.")
    else:
        print("Could not find MoE layer to hook.")

    print(f"Analysis completed. Results saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_deep_analysis.py <ckpt_path> <output_dir>")
    else:
        run_analysis(sys.argv[1], sys.argv[2])
