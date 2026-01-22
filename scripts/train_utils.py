import os
import sys
import torch
from transformers import TimeMoeForPrediction, TimeMoeConfig

# Add Time-MoE to path
sys.path.append(os.path.join(os.getcwd(), 'Time-MoE'))

from time_moe.models.modeling_time_moe import TimeMoeForPrediction, TimeMoeConfig

def load_config(args):
    if args.config == 'full':
        print("Using FULL config (RTX 3090 mode - 2.4B Corrected)")
        model_id = "Maple728/TimeMoE-2.4B"
        config_path = 'model_config/config.json'
    elif args.config == 'base':
        print("Using BASE config (Pre-trained TimeMoE-50M)")
        model_id = "Maple728/TimeMoE-50M"
        config_path = 'model_config/base_50m.json'
    else:
        print("Using TINY config (Fast-Track Debug Mode with 50M weights)")
        model_id = "Maple728/TimeMoE-50M"
        config_path = 'model_config/tiny_config.json'
        
    local_config = TimeMoeConfig.from_pretrained(config_path)
    train_conf = getattr(local_config, 'training_config', {})
    
    # Merge JSON config with CLI overrides
    config = {
        'model_id': model_id,
        'local_config': local_config,
        'config_path': config_path,
        'batch_size': train_conf.get('batch_size', 4),
        'gradient_accumulation_steps': train_conf.get('gradient_accumulation_steps', 8),
        'max_steps': args.steps if args.steps != 100000 else train_conf.get('max_steps', args.steps),
        'eval_steps': args.eval_steps if args.eval_steps != 1000 else train_conf.get('eval_steps', args.eval_steps),
        'bf16': train_conf.get('bf16', False),
        'fp16': train_conf.get('fp16', False),
        'gradient_checkpointing': train_conf.get('gradient_checkpointing', False),
        'max_length': train_conf.get('max_length', 2048),
        'optim': train_conf.get('optim', "adamw_torch"),
        'output_dir': 'checkpoints_transfer' if args.config == 'tiny' else 'checkpoints_transfer_base',
        'train_data': 'processed_bin/train',
        'test_data': 'processed_bin/val'
    }
    
    print(f"Configuration Loaded: Batch={config['batch_size']}, Accum={config['gradient_accumulation_steps']}, "
          f"Steps={config['max_steps']}, BF16={config['bf16']}, FP16={config['fp16']}")
    
    return config

def load_model(config):
    print(f"[Transfer Learning] Loading pre-trained weights from: {config['model_id']}")
    try:
        model = TimeMoeForPrediction.from_pretrained(
            config['model_id'],
            torch_dtype=torch.float16 if config['fp16'] else (torch.bfloat16 if config['bf16'] else torch.float32),
            device_map="auto"
        )
        model.config.use_cache = False
        model.config.output_hidden_states = True
        
        if model.config.input_size != 1:
            print(f"Warning: Pre-trained model input_size is {model.config.input_size}, but dataset is univariate (1).")
            
    except Exception as e:
        print(f"Error loading pre-trained model {config['model_id']}: {e}")
        print("Fallback to training from scratch with local config...")
        model_config = config['local_config']
        model_config.output_hidden_states = True
        model = TimeMoeForPrediction(model_config)
        
    return model

def init_router_weights(model):
    print("[Agent] Re-initializing Router weights for stability...")
    for name, module in model.named_modules():
        if "gate" in name and isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

def freeze_parameters(model):
    print("[Agent] Applying Partial Freezing Strategy...")
    frozen_params = []
    unfrozen_params = []
    
    for name, param in model.named_parameters():
        param.requires_grad = True # Default
        
        # Freeze experts AND Attention (MLP layers inside MoE + Self-Attention)
        if ("experts." in name and "shared_expert" not in name) or ("self_attn." in name):
             param.requires_grad = False
             frozen_params.append(name)
        else:
             unfrozen_params.append(name)
             
    print(f"[Agent] Frozen {len(frozen_params)} parameters (Sparse Experts + Attention).")
    print(f"[Agent] Unfrozen {len(unfrozen_params)} parameters (Router, Shared, Head, Embed).")
