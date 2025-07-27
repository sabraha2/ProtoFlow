#!/usr/bin/env python3
"""
Script to find and analyze ProtoFlow checkpoints.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
import glob

def find_checkpoints(search_dir=".", extensions=None):
    """Find checkpoint files in the given directory."""
    if extensions is None:
        extensions = ['*.pth', '*.pt', '*.ckpt', '*.pkl']
    
    checkpoints = []
    for ext in extensions:
        pattern = os.path.join(search_dir, '**', ext)
        checkpoints.extend(glob.glob(pattern, recursive=True))
    
    return sorted(checkpoints)

def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint file."""
    print(f"\nAnalyzing: {checkpoint_path}")
    print("-" * 50)
    
    try:
        # Try to load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"✓ Successfully loaded checkpoint")
        print(f"File size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
        
        # Print keys
        print(f"Keys: {list(checkpoint.keys())}")
        
        # Check for important components
        if 'model_state_dict' in checkpoint:
            print("✓ Contains model state dict")
            state_dict = checkpoint['model_state_dict']
            print(f"  State dict keys: {len(state_dict)}")
            print(f"  Sample keys: {list(state_dict.keys())[:5]}")
        else:
            print("⚠️  No model state dict found")
        
        if 'prototypes' in checkpoint:
            print("✓ Contains prototypes")
            prototypes = checkpoint['prototypes']
            print(f"  Number of prototype classes: {len(prototypes)}")
            if prototypes:
                sample_proto = next(iter(prototypes.values()))
                if isinstance(sample_proto, dict):
                    print(f"  Prototype structure: {list(sample_proto.keys())}")
                else:
                    print(f"  Prototype type: {type(sample_proto)}")
        else:
            print("⚠️  No prototypes found")
        
        if 'num_classes' in checkpoint:
            print(f"✓ Number of classes: {checkpoint['num_classes']}")
        
        if 'features_shape' in checkpoint:
            print(f"✓ Features shape: {checkpoint['features_shape']}")
        
        # Check for other useful keys
        useful_keys = ['epoch', 'step', 'optimizer_state_dict', 'scheduler_state_dict']
        for key in useful_keys:
            if key in checkpoint:
                print(f"✓ Contains {key}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Find and analyze ProtoFlow checkpoints')
    parser.add_argument('--search_dir', default='.', help='Directory to search for checkpoints')
    parser.add_argument('--checkpoint', help='Specific checkpoint to analyze')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Analyze specific checkpoint
        if os.path.exists(args.checkpoint):
            analyze_checkpoint(args.checkpoint)
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
    else:
        # Find and analyze all checkpoints
        print(f"Searching for checkpoints in: {args.search_dir}")
        checkpoints = find_checkpoints(args.search_dir)
        
        if not checkpoints:
            print("No checkpoint files found!")
            print("Searched for: *.pth, *.pt, *.ckpt, *.pkl")
            return
        
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for i, ckpt in enumerate(checkpoints):
            print(f"{i+1}. {ckpt}")
        
        print(f"\nAnalyzing all checkpoints...")
        successful = 0
        for ckpt in checkpoints:
            if analyze_checkpoint(ckpt):
                successful += 1
        
        print(f"\nSummary: {successful}/{len(checkpoints)} checkpoints loaded successfully")

if __name__ == '__main__':
    main() 