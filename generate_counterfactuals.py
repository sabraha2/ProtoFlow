#!/usr/bin/env python3
"""
Generate counterfactual explanations using enhanced ProtoFlow.

Usage:
    python generate_counterfactuals.py --checkpoint enhanced_checkpoint.pt --dataset cifar10 --num_samples 10
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add parent directory to path to import protoflow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ProtoFlow modules
from protoflow.counterfactual import ProtoFlowCounterfactual, get_dataset_config
from protoflow.proto import ProtoFlowGMM
from protoflow.datasets import get_dataset
from protoflow.training import get_transform

def load_protoflow_model(checkpoint_path: str, device: str = 'cuda'):
    """Load ProtoFlow model from checkpoint."""
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration
    features_shape = checkpoint['features_shape']
    num_classes = checkpoint['num_classes'] 
    latent_dim = checkpoint['latent_dim']
    
    print(f"Model config: {num_classes} classes, features shape {features_shape}, latent dim {latent_dim}")
    
    class DummyFlow:
        def __init__(self):
            pass
            
        def log_prob(self, x, return_z=True):
            # This is just a placeholder 
            batch_size = x.shape[0]
            z_shape = [batch_size] + features_shape
            z = torch.randn(z_shape, device=x.device)
            log_prob = torch.randn(batch_size, device=x.device)
            return z, log_prob
            
        def sample(self, z):
            # Placeholder implementation
            return torch.randn_like(z)
            
        def inverse(self, x):
            # Placeholder implementation  
            return torch.randn(x.shape[0], *features_shape, device=x.device)
    
    dummy_flow = DummyFlow()
    
    # Create ProtoFlowGMM with dummy flow (will be replaced by state dict)
    model = ProtoFlowGMM(
        model=dummy_flow,
        n_classes=num_classes,
        features_shape=features_shape,
        protos_per_class=10,  # Default - will be overwritten
        likelihood_approach='total',
        gaussian_approach='GaussianMixture'
    )
    
    # Load the actual trained state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model state dict loaded successfully")
    
    return model, checkpoint

def create_enhanced_model(model: ProtoFlowGMM, checkpoint: dict) -> ProtoFlowCounterfactual:
    """Create enhanced model with counterfactual capabilities."""
    
    features_shape = checkpoint['features_shape']
    
    # Create enhanced model
    enhanced_model = ProtoFlowCounterfactual(
        protoflow_model=model,
        features_shape=features_shape
    )
    
    # Restore prototype distributions
    enhanced_model.prototypes = checkpoint['prototypes']
    
    print("✓ Prototype distributions restored")
    return enhanced_model

def get_test_dataloader(dataset_name: str, batch_size: int = 1):
    """Get test dataloader for the specified dataset."""
    
    # Get appropriate image size based on dataset
    img_size_map = {
        'cifar10': 32,
        'cifar100': 32,
        'mnist': 28,
        'stl10': 96,
        'imagenet': 224,
        'pets': 224,
        'flowers': 224,
        'aircraft': 224,
        'food': 224,
        'caltech101': 224,
        'cub200': 224,
    }
    
    img_size = img_size_map.get(dataset_name, 32)
    
    # Get transform (same as used in training)
    transform = get_transform(
        interpolation='bicubic',
        size=img_size,
        train=False,
        augmentation='v1'
    )
    
    # Get dataset
    test_dataset = get_dataset(
        name=dataset_name,
        train=False,
        transform=transform
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader

def safe_generate_explanation(enhanced_model: ProtoFlowCounterfactual, 
                             image: torch.Tensor, 
                             source_class: int,
                             max_retries: int = 3):
    """Safely generate explanation with error handling."""
    
    for attempt in range(max_retries):
        try:
            explanation = enhanced_model.generate_explanation(
                image=image,
                source_class=source_class
            )
            return explanation, None
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1} failed: {e}, retrying...")
                continue
            else:
                return None, str(e)

def generate_and_save_counterfactuals(enhanced_model: ProtoFlowCounterfactual, 
                                    test_loader: DataLoader,
                                    args: argparse.Namespace):
    """Generate and save counterfactual explanations."""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get dataset config for class names
    dataset_config = get_dataset_config(args.dataset)
    class_names = dataset_config.get('class_names', [f'class_{i}' for i in range(10)])
    
    all_metrics = []
    successful_samples = 0
    
    print(f"Generating counterfactuals for {args.num_samples} samples...")
    
    enhanced_model.protoflow.eval()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= args.num_samples:
                break
                
            # Process one image at a time
            image, label = images[0:1], labels[0:1]
            
            if torch.cuda.is_available():
                image, label = image.cuda(), label.cuda()
                
            source_class = label.item()
            
            print(f"\nSample {batch_idx+1}/{args.num_samples}: {class_names[source_class]} (class {source_class})")
            
            # Generate explanation with error handling
            explanation, error = safe_generate_explanation(
                enhanced_model, image, source_class
            )
            
            if explanation is None:
                print(f"✗ Failed to generate counterfactual: {error}")
                continue
                
            try:
                # Save gallery
                gallery_path = output_dir / f'gallery_{batch_idx:03d}_{class_names[source_class]}.png'
                explanation['gallery'].savefig(gallery_path, dpi=200, bbox_inches='tight')
                plt.close(explanation['gallery'])
                
                # Print evaluation metrics
                print(f"Counterfactual quality metrics:")
                for target_class, metrics in explanation['evaluations'].items():
                    success = "✓" if metrics['prediction_success'] else "✗"
                    print(f"  → Class {target_class} ({class_names[target_class]}): {success} "
                          f"conf={metrics['target_confidence']:.3f} "
                          f"L2={metrics['l2_distance']:.3f} "
                          f"α={explanation['alphas'][target_class]:.3f}")
                
                # Collect metrics
                sample_metrics = {
                    'sample_idx': batch_idx,
                    'source_class': source_class,
                    'evaluations': explanation['evaluations'],
                    'alphas': explanation['alphas']
                }
                all_metrics.append(sample_metrics)
                successful_samples += 1
                
                print(f"✓ Gallery saved to {gallery_path}")
                
            except Exception as e:
                print(f"✗ Error saving results for sample {batch_idx}: {e}")
                continue
    
    # Save metrics
    if all_metrics:
        metrics_path = output_dir / 'evaluation_metrics.pt'
        torch.save(all_metrics, metrics_path)
        print(f"\n✓ Metrics saved to {metrics_path}")
        
        # Print summary statistics
        print_summary_statistics(all_metrics, class_names)
    
    print(f"\n✓ Successfully processed {successful_samples}/{args.num_samples} samples")

def print_summary_statistics(metrics, class_names):
    """Print summary statistics across all samples."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    total_counterfactuals = 0
    successful_counterfactuals = 0
    total_confidence = 0
    total_distance = 0
    
    for sample in metrics:
        for target_class, eval_metrics in sample['evaluations'].items():
            total_counterfactuals += 1
            if eval_metrics['prediction_success']:
                successful_counterfactuals += 1
            total_confidence += eval_metrics['target_confidence']
            total_distance += eval_metrics['l2_distance']
    
    if total_counterfactuals > 0:
        success_rate = successful_counterfactuals / total_counterfactuals
        avg_confidence = total_confidence / total_counterfactuals
        avg_distance = total_distance / total_counterfactuals
        
        print(f"Success Rate: {success_rate:.1%} ({successful_counterfactuals}/{total_counterfactuals})")
        print(f"Average Target Confidence: {avg_confidence:.3f}")
        print(f"Average L2 Distance: {avg_distance:.3f}")
        
        # Per-class success rates
        class_success = {}
        for sample in metrics:
            source_class = sample['source_class']
            if source_class not in class_success:
                class_success[source_class] = {'total': 0, 'successful': 0}
            
            for target_class, eval_metrics in sample['evaluations'].items():
                class_success[source_class]['total'] += 1
                if eval_metrics['prediction_success']:
                    class_success[source_class]['successful'] += 1
        
        print("\nPer-class success rates:")
        for class_idx, stats in class_success.items():
            if stats['total'] > 0:
                rate = stats['successful'] / stats['total']
                print(f"  {class_names[class_idx]}: {rate:.1%} ({stats['successful']}/{stats['total']})")

def main():
    parser = argparse.ArgumentParser(description='Generate ProtoFlow counterfactuals')
    
    # Model and data arguments
    parser.add_argument('--checkpoint', required=True, help='Enhanced checkpoint path')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--output_dir', default='./counterfactual_results', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load model
        model, checkpoint = load_protoflow_model(args.checkpoint, device)
        print("✓ Base model loaded successfully")
        
        # Create enhanced model
        enhanced_model = create_enhanced_model(model, checkpoint)
        print("✓ Enhanced model created successfully")
        
        # Get test dataloader
        test_loader = get_test_dataloader(args.dataset, batch_size=1)
        print(f"✓ Test dataloader created for {args.dataset}")
        
        # Generate counterfactuals
        generate_and_save_counterfactuals(enhanced_model, test_loader, args)
        
        print(f"\n🎉 All counterfactuals generated and saved to {args.output_dir}")
        print(f"📁 Check the gallery images: {args.output_dir}/gallery_*.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == '__main__':
    main()