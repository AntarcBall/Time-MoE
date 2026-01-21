import os
import json

# Tiny config for verification in low-resource environment
config = {
    "model_type": "time_moe",
    "input_size": 1,
    "hidden_size": 64,        # Reduced from 2560
    "intermediate_size": 256, # Reduced from 10240
    "horizon_lengths": [1, 8, 32, 64],
    "num_hidden_layers": 2,   # Reduced from 36
    "num_attention_heads": 4, # Reduced from 32
    "num_key_value_heads": 4, # Reduced from 32
    "hidden_act": "silu",
    "num_experts_per_tok": 2,
    "num_experts": 4,         # Reduced from 8
    "max_position_embeddings": 2048, # Reduced context
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "use_cache": True,
    "use_dense": False,
    "rope_theta": 10000,
    "attention_dropout": 0.0,
    "apply_aux_loss": True,
    "router_aux_loss_factor": 0.02,
    "tie_word_embeddings": False,
    "torch_dtype": "float32", # Use float32 for compatibility
    "transformers_version": "4.40.1"
}

os.makedirs('model_config', exist_ok=True)
with open('model_config/tiny_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Tiny config saved to model_config/tiny_config.json")
