import argparse
import sys
import os
import torch

# Add Time-MoE to path
sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))

from time_moe.runner import TimeMoeRunner
from time_moe.trainer.hf_trainer import TimeMoETrainingArguments, TimeMoeTrainer

# Import modules
from scripts.data_utils import prepare_datasets
from scripts.train_utils import load_config, load_model, init_router_weights, freeze_parameters
from scripts.eval_utils import AgentCallback, AuxLossWarmupCallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='tiny', choices=['full', 'base', 'tiny'])
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args)
    runner = TimeMoeRunner(output_path=config['output_dir'], seed=42)

    # 2. Load Model
    model = load_model(config)
    
    # 3. Initialize Router & Apply Freezing
    init_router_weights(model)
    freeze_parameters(model)

    try:
        import bitsandbytes
    except ImportError:
        if config['optim'] == "adamw_bnb_8bit": 
            config['optim'] = "adamw_torch"

    # 4. Prepare Datasets
    train_ds, test_ds = prepare_datasets(runner, config)

    # 5. Setup Trainer
    training_args = TimeMoETrainingArguments(
        output_dir=config['output_dir'], 
        max_steps=config['max_steps'], 
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'], 
        learning_rate=1e-4, 
        min_learning_rate=1e-5,
        max_grad_norm=1.0, 
        save_steps=config['eval_steps'], 
        bf16=config['bf16'], 
        fp16=config['fp16'],
        gradient_checkpointing=config['gradient_checkpointing'], 
        dataloader_num_workers=0, 
        dataloader_pin_memory=False, 
        dataloader_prefetch_factor=None, 
        remove_unused_columns=False
    )
    
    trainer = TimeMoeTrainer(model=model, args=training_args, train_dataset=train_ds)
    
    agent_cb = AgentCallback(trainer, test_ds, config['batch_size'])
    agent_cb.eval_interval_steps = config['eval_steps']
    trainer.add_callback(agent_cb)
    
    warmup_steps = int(config['max_steps'] * 0.1)
    trainer.add_callback(AuxLossWarmupCallback(target=0.1, warmup_ratio=0.1))
    
    print("Starting Training with Agent...")
    trainer.train()
    trainer.save_model() # Final save

if __name__ == "__main__":
    main()
