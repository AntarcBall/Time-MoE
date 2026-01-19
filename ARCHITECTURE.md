# Time-MoE Model Architecture

## Overview
Time-MoE (Time Series Mixture of Experts) is a specialized transformer architecture designed for time series forecasting. It combines traditional transformer components with Mixture of Experts (MoE) mechanisms to efficiently handle temporal data with varying patterns.

## Key Components

### 1. Input Embedding Layer
- Uses an MLP-based approach to embed time series data
- Employs gated linear units for better feature representation
- Maps input time series to hidden space

### 2. Transformer Blocks
- Multiple stacked transformer layers
- Each contains self-attention mechanism with RoPE (Rotary Position Embedding)
- RMS normalization for stable training
- Residual connections for gradient flow

### 3. Mixture of Experts (MoE) Feed-Forward Network
- **Router/Gating Network**: Determines which experts to activate
- **Expert Networks**: Multiple feed-forward networks specialized for different patterns
- **Top-K Selection**: Selects top-k experts based on routing weights
- **Shared Expert**: A universal expert available to all tokens
- **Load Balancing**: Auxiliary loss to ensure even expert utilization

### 4. Output Layer
- Multiple prediction heads for different forecast horizons
- Projects hidden representations to target dimensions
- Supports multi-horizon forecasting

## Architecture Variants Created

Three visualizations were created to represent the Time-MoE architecture:

1. **Basic Architecture** (`time_moe_architecture.png`): High-level overview of the model components
2. **Detailed Architecture** (`detailed_time_moe_architecture.png`): In-depth view of internal mechanisms
3. **Simplified Architecture** (`simple_time_moe_architecture.png`): Conceptual view focusing on core ideas

## Key Innovations

- **Temporal Specialization**: Experts specialize in different temporal patterns
- **Efficient Computation**: Only activates relevant experts per token
- **Multi-Horizon Forecasting**: Simultaneous prediction for multiple time horizons
- **Load Balancing**: Ensures optimal expert utilization through auxiliary loss

## Configuration Parameters

Based on the model configuration:
- Hidden size: 4096
- Number of attention heads: 32
- Number of hidden layers: 32
- Number of experts: Configurable
- Experts per token: 2 (by default)
- Maximum position embeddings: 32768
- RMS normalization epsilon: 1e-6