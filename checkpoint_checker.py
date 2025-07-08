#!/usr/bin/env python3
"""
Inspect the ClassConditionalPrototypes object structure to understand how to access the data.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_checkpoint(checkpoint_path):
    """Inspect the checkpoint structure in detail."""
    print(f"Inspecting checkpoint: {checkpoint_path}")
    
    # Load checkpoint with safe globals
    try:
        import torch.serialization
        from protoflow.counterfactual import ClassConditionalPrototypes
        
        with torch.serialization.safe_globals([ClassConditionalPrototypes]):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        print("✓ Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print("✓ Checkpoint loaded with weights_only=False")
        except Exception as e2:
            print(f"Failed to load checkpoint: {e2}")
            return
    
    print("\n" + "="*60)
    print("CHECKPOINT STRUCTURE")
    print("="*60)
    
    for key, value in checkpoint.items():
        print(f"\nKey: {key}")
        print(f"Type: {type(value)}")
        if isinstance(value, torch.Tensor):
            print(f"Shape: {value.shape}")
        elif isinstance(value, (int, float, str)):
            print(f"Value: {value}")
        elif hasattr(value, '__len__'):
            try:
                print(f"Length: {len(value)}")
            except:
                pass
    
    # Focus on the prototypes object
    if 'prototypes' in checkpoint:
        print("\n" + "="*60)
        print("PROTOTYPES OBJECT INSPECTION")
        print("="*60)
        
        protos = checkpoint['prototypes']
        print(f"Type: {type(protos)}")
        print(f"Class: {protos.__class__}")
        print(f"Module: {protos.__class__.__module__}")
        
        # Get all attributes
        attrs = [attr for attr in dir(protos) if not attr.startswith('_')]
        print(f"\nPublic attributes: {attrs}")
        
        # Try to access each attribute
        print("\nAttribute inspection:")
        for attr in attrs:
            try:
                value = getattr(protos, attr)
                if callable(value):
                    print(f"  {attr}: <method/function>")
                else:
                    print(f"  {attr}: {type(value)} - {value}")
                    if isinstance(value, torch.Tensor):
                        print(f"    Shape: {value.shape}")
                    elif isinstance(value, dict):
                        print(f"    Keys: {list(value.keys())}")
                        # Inspect first item
                        if value:
                            first_key = list(value.keys())[0]
                            first_val = value[first_key]
                            print(f"    First item ({first_key}): {type(first_val)}")
                            if isinstance(first_val, torch.Tensor):
                                print(f"      Shape: {first_val.shape}")
                            elif isinstance(first_val, dict):
                                print(f"      Keys: {list(first_val.keys())}")
                    elif hasattr(value, '__len__'):
                        try:
                            print(f"    Length: {len(value)}")
                        except:
                            pass
            except Exception as e:
                print(f"  {attr}: <error accessing: {e}>")
        
        # Check if it has __dict__
        if hasattr(protos, '__dict__'):
            print(f"\n__dict__ keys: {list(protos.__dict__.keys())}")
            for key, value in protos.__dict__.items():
                print(f"  {key}: {type(value)}")
                if isinstance(value, torch.Tensor):
                    print(f"    Shape: {value.shape}")
                elif isinstance(value, dict):
                    print(f"    Keys: {list(value.keys())}")
        
        # Try to see if it's iterable
        try:
            if hasattr(protos, '__iter__'):
                print("\nObject is iterable, trying to iterate...")
                for i, item in enumerate(protos):
                    print(f"  Item {i}: {type(item)}")
                    if i >= 3:  # Just show first few
                        print(f"  ... (stopping after first few items)")
                        break
        except Exception as e:
            print(f"Error iterating: {e}")
        
        # Try indexing
        try:
            print("\nTrying indexing...")
            for i in range(min(3, checkpoint.get('num_classes', 10))):
                try:
                    item = protos[i]
                    print(f"  protos[{i}]: {type(item)}")
                    if isinstance(item, torch.Tensor):
                        print(f"    Shape: {item.shape}")
                    elif isinstance(item, dict):
                        print(f"    Keys: {list(item.keys())}")
                except Exception as e:
                    print(f"  protos[{i}]: <error: {e}>")
        except Exception as e:
            print(f"Error with indexing: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = 'enhanced_checkpoint_ultra_fast.pt'
    
    inspect_checkpoint(checkpoint_path)