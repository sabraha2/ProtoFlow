#!/usr/bin/env python3
"""
Simple script to find ProtoFlow checkpoints without requiring PyTorch.
"""

import os
import sys
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

def analyze_checkpoint_simple(checkpoint_path):
    """Analyze a checkpoint file without loading it."""
    print(f"\nAnalyzing: {checkpoint_path}")
    print("-" * 50)
    
    try:
        # Get file info
        file_size = os.path.getsize(checkpoint_path)
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Check if it's a valid file
        if file_size == 0:
            print("✗ Empty file")
            return False
        
        if file_size < 1024:  # Less than 1KB
            print("⚠️  Very small file - may not be a valid checkpoint")
        
        print("✓ File exists and has content")
        
        # Try to peek at the file structure (very basic)
        try:
            with open(checkpoint_path, 'rb') as f:
                # Read first few bytes to check if it looks like a PyTorch file
                header = f.read(100)
                if b'PK' in header[:10]:  # ZIP file (PyTorch uses ZIP)
                    print("✓ Appears to be a PyTorch file (ZIP format)")
                elif b'\x80\x02' in header[:10]:  # Pickle file
                    print("✓ Appears to be a pickle file")
                else:
                    print("⚠️  Unknown file format")
        except Exception as e:
            print(f"⚠️  Could not read file header: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error analyzing file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Find ProtoFlow checkpoints (simple version)')
    parser.add_argument('--search_dir', default='.', help='Directory to search for checkpoints')
    parser.add_argument('--checkpoint', help='Specific checkpoint to analyze')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Analyze specific checkpoint
        if os.path.exists(args.checkpoint):
            analyze_checkpoint_simple(args.checkpoint)
        else:
            print(f"Checkpoint not found: {args.checkpoint}")
    else:
        # Find and analyze all checkpoints
        print(f"Searching for checkpoints in: {args.search_dir}")
        checkpoints = find_checkpoints(args.search_dir)
        
        if not checkpoints:
            print("No checkpoint files found!")
            print("Searched for: *.pth, *.pt, *.ckpt, *.pkl")
            print("\nCommon locations to check:")
            print("- logs/")
            print("- checkpoints/")
            print("- models/")
            print("- runs/")
            print("- experiments/")
            return
        
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for i, ckpt in enumerate(checkpoints):
            print(f"{i+1}. {ckpt}")
        
        print(f"\nAnalyzing all checkpoints...")
        successful = 0
        for ckpt in checkpoints:
            if analyze_checkpoint_simple(ckpt):
                successful += 1
        
        print(f"\nSummary: {successful}/{len(checkpoints)} checkpoints found")
        
        if successful > 0:
            print("\nTo analyze checkpoints in detail (requires PyTorch):")
            print("1. Set up the environment: ./setup_env.sh")
            print("2. Activate the environment")
            print("3. Run: python3 find_checkpoints.py")

if __name__ == '__main__':
    main() 