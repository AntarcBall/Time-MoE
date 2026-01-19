#!/usr/bin/env python3
"""
Script to generate Time-MoE model architecture diagrams using Graphviz
"""

import os
from graphviz import Digraph

def create_basic_architecture():
    """Create basic Time-MoE architecture diagram"""
    dot = Digraph('TimeMoEArchitecture', comment='Time-MoE Model Architecture')
    
    # Graph properties
    dot.attr(rankdir='TB', label='Time-MoE Model Architecture', labelloc='top', fontsize='16', fontname='Arial Bold')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Arial')
    
    # Define nodes
    dot.node('input_layer', 'Input Embedding\\n(MLP-based)', fillcolor='lightgreen')
    
    # Subgraph for model
    with dot.subgraph(name='cluster_model') as c:
        c.attr(label='Time-MoE Model', color='gray', style='dashed')
        
        # Decoder layers
        with c.subgraph(name='cluster_layers') as cl:
            cl.attr(label='Transformer Decoder Layers (N layers)', color='lightgray')
            cl.node('layer_1', 'Decoder Layer 1', fillcolor='lightyellow')
            cl.node('layer_2', 'Decoder Layer 2', fillcolor='lightyellow')
            cl.node('layer_n', '...', shape='plaintext', fillcolor='white')
            cl.node('layer_last', 'Decoder Layer N', fillcolor='lightyellow')
        
        # Inner layer structure
        with c.subgraph(name='cluster_inner_layer') as cil:
            cil.attr(label='Decoder Layer Structure', color='lightcoral', style='dotted')
            cil.node('layer_norm_1', 'RMSNorm', fillcolor='lightcyan', shape='ellipse')
            cil.node('attention', 'Multi-Head\\nAttention', fillcolor='orange')
            cil.node('layer_norm_2', 'RMSNorm', fillcolor='lightcyan', shape='ellipse')
            
            # MoE FFN
            with cil.subgraph(name='cluster_ffn') as ffn:
                ffn.attr(label='Mixture of Experts (MoE)\\nor Dense FFN', color='plum')
                ffn.node('gate', 'Router/Gating\\nNetwork', fillcolor='pink')
                ffn.node('expert_1', 'Expert 1', fillcolor='gold')
                ffn.node('expert_2', 'Expert 2', fillcolor='gold')
                ffn.node('expert_k', 'Expert K', fillcolor='gold')
                ffn.node('shared_expert', 'Shared Expert', fillcolor='khaki')
                ffn.node('combine', 'Combine\\nOutputs', fillcolor='plum')
    
    # Output
    dot.node('output_norm', 'Final RMSNorm', fillcolor='lightcyan', shape='ellipse')
    dot.node('output_layer', 'Output Layer\\n(Prediction Head)', fillcolor='lightgreen')
    
    # Edges
    dot.edge('input_layer', 'layer_1')
    dot.edge('layer_1', 'layer_2')
    dot.edge('layer_2', 'layer_n')
    dot.edge('layer_n', 'layer_last')
    
    # Within-layer edges
    dot.edge('layer_norm_1', 'attention')
    dot.edge('attention', 'layer_norm_2')
    dot.edge('layer_norm_2', 'gate')
    dot.edge('gate', 'expert_1')
    dot.edge('gate', 'expert_2')
    dot.edge('gate', 'expert_k')
    dot.edge('gate', 'shared_expert')
    dot.edge('expert_1', 'combine')
    dot.edge('expert_2', 'combine')
    dot.edge('expert_k', 'combine')
    dot.edge('shared_expert', 'combine')
    dot.edge('combine', 'output_norm')
    
    dot.edge('layer_last', 'output_norm')
    dot.edge('output_norm', 'output_layer')
    
    return dot

def create_detailed_architecture():
    """Create detailed Time-MoE architecture diagram"""
    dot = Digraph('TimeMoEArchitectureDetailed', comment='Time-MoE Model Architecture - Detailed View')
    
    # Graph properties
    dot.attr(rankdir='TB', label='Time-MoE Model Architecture - Detailed View', labelloc='top', 
             fontsize='18', fontname='Arial Bold')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Arial')
    
    # Input
    dot.node('input_ts', 'Time Series Input\\n[Batch, Seq Len, Features]', fillcolor='lightgreen', shape='box3d')
    dot.node('input_embed', 'Input Embedding Layer\\nMLP with Gate Activation\\n(Linear + SiLU activation)', fillcolor='lightseagreen')
    
    # Transformer blocks
    with dot.subgraph(name='cluster_transformer_blocks') as ct:
        ct.attr(label='Transformer Blocks (Stacked)', color='gray', style='dashed', 
                fontname='Arial Bold', fontsize='14')
        
        with ct.subgraph(name='cluster_block') as cb:
            cb.attr(label='Transformer Block (Repeat N times)', color='darkolivegreen', style='solid')
            
            # Attention section
            with cb.subgraph(name='cluster_attention') as ca:
                ca.attr(label='Self-Attention Module', color='orange', style='filled', fillcolor='navajowhite')
                ca.node('ln1', 'RMSNorm', fillcolor='lightcyan', shape='ellipse')
                ca.node('att_module', 'Multi-Head Attention\\nwith RoPE\\n(Q, K, V projections)', fillcolor='orange')
                ca.node('res1', 'Residual Connection', fillcolor='lightsalmon', shape='parallelogram')
            
            # Feed-forward section with MoE
            with cb.subgraph(name='cluster_ffn') as ff:
                ff.attr(label='Feed-Forward Module (MoE)', color='purple', style='filled', fillcolor='thistle')
                ff.node('ln2', 'RMSNorm', fillcolor='lightcyan', shape='ellipse')
                
                # Router and experts
                ff.node('router', 'Router/Gating Network\\nLinear Layer (H -> Num Experts)', fillcolor='pink')
                
                with ff.subgraph(name='cluster_experts') as exp:
                    exp.attr(label='Expert Networks', color='gold', style='filled', fillcolor='lemonchiffon')
                    exp.node('expert1', 'Expert 1\\nTemporal Block\\n(Gate, Up, Down Projections)', fillcolor='gold')
                    exp.node('expert2', 'Expert 2\\nTemporal Block\\n(Gate, Up, Down Projections)', fillcolor='gold')
                    exp.node('expertn', 'Expert N\\nTemporal Block\\n(Gate, Up, Down Projections)', fillcolor='gold')
                    exp.node('shared_expert', 'Shared Expert\\nTemporal Block\\n(Full Intermediate Size)', fillcolor='khaki')
                
                ff.node('combine', 'Top-K Selection\\nand Weighted Combination\\n+ Shared Expert Gate', fillcolor='orchid')
                ff.node('res2', 'Residual Connection', fillcolor='lightsalmon', shape='parallelogram')
    
    # Output processing
    dot.node('final_norm', 'Final RMSNorm', fillcolor='lightcyan', shape='ellipse')
    dot.node('output_proj', 'Output Projection\\nMultiple Horizon Heads\\n(H -> Horizon Length * Features)', fillcolor='lightgreen', shape='box3d')
    dot.node('output_pred', 'Predictions\\nMultiple Horizons', fillcolor='lightsteelblue', shape='box3d')
    
    # Connections
    dot.edge('input_ts', 'input_embed')
    dot.edge('input_embed', 'ln1')
    dot.edge('ln1', 'att_module')
    dot.edge('att_module', 'res1')
    dot.edge('res1', 'ln2')
    dot.edge('ln2', 'router')
    dot.edge('router', 'expert1')
    dot.edge('router', 'expert2')
    dot.edge('router', 'expertn')
    dot.edge('router', 'shared_expert')
    dot.edge('expert1', 'combine')
    dot.edge('expert2', 'combine')
    dot.edge('expertn', 'combine')
    dot.edge('shared_expert', 'combine')
    dot.edge('combine', 'res2')
    dot.edge('res2', 'final_norm')
    dot.edge('final_norm', 'output_proj')
    dot.edge('output_proj', 'output_pred')
    
    return dot

def create_simplified_architecture():
    """Create simplified Time-MoE architecture diagram"""
    dot = Digraph('TimeMoESimplified', comment='Time-MoE: Time Series Mixture of Experts Architecture')
    
    # Graph properties
    dot.attr(rankdir='TB', label='Time-MoE: Time Series Mixture of Experts Architecture', 
             labelloc='top', fontsize='16', fontname='Arial Bold')
    dot.attr('node', style='filled', fontname='Arial', fontsize='12')
    
    # Input
    dot.node('input', 'Time Series Data\\n(Univariate/Multivariate)', fillcolor='lightblue', shape='box')
    dot.node('embed', 'Input Embedding\\n(MLP-based)', fillcolor='lightgreen', shape='box')
    dot.node('transformer', 'N Transformer Layers\\nEach with Attention + MoE-FFN', fillcolor='lightyellow', shape='box')
    
    # MoE Component
    with dot.subgraph(name='cluster_moe') as cm:
        cm.attr(label='Mixture of Experts (MoE) Component', color='red', style='dashed')
        cm.node('router', 'Router\\n(Gating Network)', fillcolor='pink', shape='ellipse')
        cm.node('topk', 'Top-K Selection', fillcolor='plum', shape='ellipse')
        
        with cm.subgraph(name='cluster_experts') as ce:
            ce.attr(label='Expert Networks', color='orange')
            ce.node('expert1', 'Expert 1', fillcolor='gold', shape='box')
            ce.node('expert2', 'Expert 2', fillcolor='gold', shape='box')
            ce.node('expertn', 'Expert N', fillcolor='gold', shape='box')
        
        cm.node('shared', 'Shared Expert', fillcolor='khaki', shape='box')
        cm.node('combine', 'Weighted Sum', fillcolor='orchid', shape='ellipse')
    
    # Output
    dot.node('norm', 'RMS Normalization', fillcolor='lightcyan', shape='box')
    dot.node('output', 'Output Layer\\n(Multiple Prediction Horizons)', fillcolor='lightblue', shape='box')
    
    # Connections
    dot.edge('input', 'embed')
    dot.edge('embed', 'transformer')
    dot.edge('transformer', 'router')
    dot.edge('router', 'topk')
    dot.edge('topk', 'expert1')
    dot.edge('topk', 'expert2')
    dot.edge('topk', 'expertn')
    dot.edge('router', 'shared')
    dot.edge('expert1', 'combine')
    dot.edge('expert2', 'combine')
    dot.edge('expertn', 'combine')
    dot.edge('shared', 'combine')
    dot.edge('combine', 'norm')
    dot.edge('norm', 'output')
    
    return dot

def main():
    """Generate all architecture diagrams"""
    print("Generating Time-MoE architecture diagrams...")
    
    # Create and save basic architecture
    basic_dot = create_basic_architecture()
    basic_dot.render('/home/car/moe/Time-MoE/time_moe_architecture', format='png', cleanup=True)
    print("✓ Basic architecture saved as time_moe_architecture.png")
    
    # Create and save detailed architecture
    detailed_dot = create_detailed_architecture()
    detailed_dot.render('/home/car/moe/Time-MoE/detailed_time_moe_architecture', format='png', cleanup=True)
    print("✓ Detailed architecture saved as detailed_time_moe_architecture.png")
    
    # Create and save simplified architecture
    simple_dot = create_simplified_architecture()
    simple_dot.render('/home/car/moe/Time-MoE/simple_time_moe_architecture', format='png', cleanup=True)
    print("✓ Simplified architecture saved as simple_time_moe_architecture.png")
    
    print("\\nAll diagrams generated successfully!")

if __name__ == "__main__":
    main()