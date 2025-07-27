#!/usr/bin/env python3
"""
Simple test script to check ProtoFlow model loading and sampling.
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_loading(checkpoint_path, device='cuda'):
    """Test if the model can be loaded and used for basic operations."""
    print(f"Testing model loading from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("✓ Checkpoint loaded successfully")
        
        # Print checkpoint keys
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Check for model state dict
        if 'model_state_dict' in checkpoint:
            print("✓ Model state dict found")
        else:
            print("⚠️  No model state dict found")
        
        # Check for prototypes
        if 'prototypes' in checkpoint:
            print("✓ Prototypes found")
            print(f"Number of prototype classes: {len(checkpoint['prototypes'])}")
        else:
            print("⚠️  No prototypes found")
        
        # Try to import ProtoFlow
        try:
            from protoflow.proto import ProtoFlowGMM
            from experiments.image.model.dense_flow import DenseFlow
            print("✓ ProtoFlow modules imported successfully")
            
            # Create a simple model
            data_shape = (3, 32, 32)
            flow_model = DenseFlow(
                data_shape=data_shape,
                block_config=[2, 4, 3],
                layers_config=[2, 2, 2],
                layer_mid_chnls=[6, 12, 20],
                growth_rate=12,
                num_bits=8,
                checkpointing=False,
                base_dist=True
            )
            
            num_classes = checkpoint.get('num_classes', 10)
            features_shape = checkpoint.get('features_shape', [3072])
            
            protoflow_model = ProtoFlowGMM(
                model=flow_model,
                n_classes=num_classes,
                features_shape=features_shape,
                protos_per_class=10,
                likelihood_approach='total',
                gaussian_approach='GaussianMixture'
            )
            
            print("✓ ProtoFlow model created successfully")
            
            # Load state dict if available
            if 'model_state_dict' in checkpoint:
                try:
                    protoflow_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("✓ State dict loaded successfully")
                except Exception as e:
                    print(f"⚠️  Could not load state dict: {e}")
            
            protoflow_model.to(device)
            protoflow_model.eval()
            
            # Test forward pass
            test_input = torch.randn(1, 3, 32, 32).to(device)
            with torch.no_grad():
                try:
                    output = protoflow_model(test_input)
                    print(f"✓ Forward pass successful, output shape: {output.shape}")
                except Exception as e:
                    print(f"✗ Forward pass failed: {e}")
                    return False
            
            # Test encoding
            try:
                z, log_prob = protoflow_model.model.log_prob(test_input, return_z=True)
                print(f"✓ Encoding successful, z shape: {z.shape}, log_prob shape: {log_prob.shape}")
            except Exception as e:
                print(f"✗ Encoding failed: {e}")
                return False
            
            # Test sampling
            try:
                # Try different sampling methods
                print("Testing sampling methods...")
                
                # Method 1: Direct sampling
                try:
                    samples = protoflow_model.model.sample(1)
                    print(f"✓ Direct sampling successful, shape: {samples.shape}")
                except Exception as e:
                    print(f"✗ Direct sampling failed: {e}")
                
                # Method 2: Sampling with z
                try:
                    samples = protoflow_model.model.sample(1, z=z)
                    print(f"✓ Sampling with z successful, shape: {samples.shape}")
                except Exception as e:
                    print(f"✗ Sampling with z failed: {e}")
                
                # Method 3: Inverse transform
                try:
                    inverse_samples = protoflow_model.model.inverse(z)
                    print(f"✓ Inverse transform successful, shape: {inverse_samples.shape}")
                except Exception as e:
                    print(f"✗ Inverse transform failed: {e}")
                
            except Exception as e:
                print(f"✗ Sampling tests failed: {e}")
            
            return True
            
        except ImportError as e:
            print(f"✗ Could not import ProtoFlow modules: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False

def test_simple_sampling(device='cuda'):
    """Test simple sampling without loading a checkpoint."""
    print("\nTesting simple sampling...")
    
    try:
        from experiments.image.model.dense_flow import DenseFlow
        
        # Create a simple DenseFlow model
        data_shape = (3, 32, 32)
        flow_model = DenseFlow(
            data_shape=data_shape,
            block_config=[2, 4, 3],
            layers_config=[2, 2, 2],
            layer_mid_chnls=[6, 12, 20],
            growth_rate=12,
            num_bits=8,
            checkpointing=False,
            base_dist=True
        )
        
        flow_model.to(device)
        flow_model.eval()
        
        print("✓ DenseFlow model created")
        
        # Test basic operations
        test_input = torch.randn(1, 3, 32, 32).to(device)
        
        with torch.no_grad():
            # Test encoding
            try:
                z, log_prob = flow_model.log_prob(test_input, return_z=True)
                print(f"✓ Encoding successful, z shape: {z.shape}")
            except Exception as e:
                print(f"✗ Encoding failed: {e}")
                return False
            
            # Test sampling
            try:
                samples = flow_model.sample(1)
                print(f"✓ Sampling successful, shape: {samples.shape}")
                
                # Save a sample image
                sample_img = samples[0].cpu()
                if sample_img.min() < 0:
                    sample_img = (sample_img + 1) / 2
                sample_img = torch.clamp(sample_img, 0, 1)
                
                plt.figure(figsize=(4, 4))
                plt.imshow(sample_img.permute(1, 2, 0).numpy())
                plt.title("Generated Sample")
                plt.axis('off')
                plt.savefig('test_sample.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✓ Sample image saved as 'test_sample.png'")
                
            except Exception as e:
                print(f"✗ Sampling failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in simple sampling test: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test ProtoFlow model loading and sampling')
    parser.add_argument('--checkpoint', help='Checkpoint path to test')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    success = True
    
    # Test simple sampling first
    print("\n" + "="*50)
    print("TESTING SIMPLE SAMPLING")
    print("="*50)
    success &= test_simple_sampling(device)
    
    # Test model loading if checkpoint provided
    if args.checkpoint:
        print("\n" + "="*50)
        print("TESTING MODEL LOADING")
        print("="*50)
        success &= test_model_loading(args.checkpoint, device)
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return 0 if success else 1

if __name__ == '__main__':
    import sys
    sys.exit(main()) 