import os
import json
from transformers import PretrainedConfig

# Define TimeMoeConfig since I can't import it easily without path setup or just defining dict
# I will just create the dict and save as json.

config = {
    "model_type": "time_moe",
    "input_size": 1,
    "hidden_size": 1024,      # Adjusted for 2.4B total params
    "intermediate_size": 4096, # Adjusted for 2.4B total params
    "horizon_lengths": [1, 8, 32, 64],
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "hidden_act": "silu",
    "num_experts_per_tok": 2,
    "num_experts": 8,
    "max_position_embeddings": 2048,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "use_dense": False,
    "rope_theta": 10000,
    "attention_dropout": 0.0,
    "apply_aux_loss": True,
    "router_aux_loss_factor": 0.02,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.1"
}

os.makedirs('model_config', exist_ok=True)
with open('model_config/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Config saved to model_config/config.json")
