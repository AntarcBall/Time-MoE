# Time-MoE Anomaly Detection: Usage Guide

This guide explains how to run the training, inference, and evaluation pipeline for the **2.4B Parameter Time-MoE** model on a single RTX 3090.

## 📋 Prerequisites

Ensure you are in the project root directory:
```bash
cd /home/car/moe
```

Dependencies (already installed):
*   Python 3.10+
*   PyTorch (CUDA 12.x)
*   Hugging Face Transformers
*   Flash-Attention 2
*   BitsAndBytes (for 8-bit optimization)

---

## 🚀 1. Data Preprocessing

Before training, raw CSV data must be processed into binary format. This step handles:
*   Train/Test splitting (Normal vs. Anomaly).
*   Channel independence (3 Current + 1 Flux).
*   **Flux Masking**: Automatically zeros out `CH4` (Flux) to simulate missing sensors.

```bash
# Step 1: Read CSVs, mask flux, split data, and save as NPY
python3 preprocess_renomeado.py

# Step 2: Convert NPY to optimized Binary format for training
python3 convert_all.py
```
*Output*: Data will be ready in `dataset_bin/train` and `dataset_bin/test`.

---

## 🏋️ 2. Training (The Agent)

This command starts the automated training agent. It handles:
*   **Model Loading**: 2.4B MoE with BF16 precision.
*   **Optimization**: 8-bit AdamW + Flash-Attention 2 to fit in 24GB VRAM.
*   **Agent Logic**: Evaluates F1 score every **3 hours** and saves checkpoints.

### **Start Production Training (Background Mode)**
Run this to start training and log output to a file, so it doesn't stop if your terminal closes.

```bash
# Run in background with unbuffered logging
PYTHONUNBUFFERED=1 python3 auto_train.py --config full > run_full.log 2>&1 &
```

### **Monitor Progress**
To see the training logs, F1 scores, and Agent reports in real-time:

```bash
tail -f run_full.log
```
*Press `Ctrl+C` to exit the log view (training continues).*

### **Stop Training**
To stop the background training process:

```bash
pkill -f auto_train.py
```

---

## 📊 3. Inference & Evaluation

The Agent automatically evaluates the model every 3 hours during training. To view these results, look at the `run_full.log`.

**Result Format:**
```text
| Step | Loss | Gating | F1-L1 (MSE) | F1-L2 (Latent) | F1-Total |
|   10 | 0.54 | 0.0210 | 0.4200      | 0.3800         | 0.4500   |
```

### **Manual Evaluation (Optional)**
If you want to manually evaluate a specific checkpoint (e.g., `checkpoint-step-10`) **after training is stopped**:

1.  **Stop Training First** (Crucial to free up VRAM):
    ```bash
    pkill -f auto_train.py
    ```

2.  **Run Evaluation Script**:
    *(Note: You may need to edit `manual_eval.py` to point to the specific checkpoint folder)*
    ```bash
    python3 manual_eval.py
    ```

---

## ⚙️ Configuration

*   **Model Config**: `model_config/config.json`
*   **Hyperparameters**: Defined in `auto_train.py` (Learning Rate, Batch Size, etc.).

### **Hardware Notes (RTX 3090)**
*   **VRAM**: The configuration is tuned for **24GB**. Do not increase `BATCH_SIZE` above 2.
*   **Speed**: Expect ~1.3 hours per global step due to the massive model size and gradient accumulation.
