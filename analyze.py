import os
import json
from pathlib import Path
import torch
from transformers import AutoConfig
from typing import Dict, Any
import humanize

def get_directory_size(directory: str) -> int:
    """Calculate total size of a directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size

def analyze_model_files(model_path: str) -> Dict[str, Any]:
    """Analyze individual model files and their sizes"""
    file_sizes = {}
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            file_sizes[item] = {
                'size_bytes': size,
                'size_human': humanize.naturalsize(size)
            }
    return file_sizes

def get_model_config(model_path: str) -> Dict[str, Any]:
    """Get model configuration details"""
    config = AutoConfig.from_pretrained(model_path)
    return config.to_dict()

def analyze_model(model_path: str = "./llama-3-sqlcoder-8b-w4a16") -> None:
    """Analyze model size and configuration"""
    print(f"\n{'='*50}")
    print("Model Analysis Report")
    print(f"{'='*50}\n")
    
    # 1. Total Size Analysis
    total_size = get_directory_size(model_path)
    print(f"Total Model Size:")
    print(f"- Bytes: {total_size:,}")
    print(f"- Human readable: {humanize.naturalsize(total_size)}")
    
    # 2. Individual File Analysis
    print(f"\nIndividual File Sizes:")
    file_sizes = analyze_model_files(model_path)
    for filename, size_info in file_sizes.items():
        print(f"- {filename}: {size_info['size_human']}")
    
    # 3. Model Configuration
    try:
        config = get_model_config(model_path)
        print(f"\nModel Configuration:")
        important_configs = {
            'model_type': config.get('model_type', 'N/A'),
            'vocab_size': config.get('vocab_size', 'N/A'),
            'hidden_size': config.get('hidden_size', 'N/A'),
            'num_attention_heads': config.get('num_attention_heads', 'N/A'),
            'num_hidden_layers': config.get('num_hidden_layers', 'N/A'),
            'max_position_embeddings': config.get('max_position_embeddings', 'N/A'),
        }
        
        for key, value in important_configs.items():
            print(f"- {key}: {value}")
            
        # 4. Quantization Info
        print(f"\nQuantization Details:")
        print("- Weight precision: 4-bit (W4)")
        print("- Activation precision: 16-bit (A16)")
        print("- Quantization method: QInt4 (optimum.quanto)")
        
        # 5. Memory Requirements (estimated)
        approx_memory = total_size * 1.2  # 20% overhead estimate
        print(f"\nEstimated Runtime Memory Requirements:")
        print(f"- Minimum: {humanize.naturalsize(approx_memory)}")
        print(f"- Recommended: {humanize.naturalsize(approx_memory * 1.5)}")
        
    except Exception as e:
        print(f"\nError reading configuration: {str(e)}")

if __name__ == "__main__":
    analyze_model()