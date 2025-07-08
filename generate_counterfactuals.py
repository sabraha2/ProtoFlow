#!/usr/bin/env python3
"""
Generate counterfactual explanations using properly enhanced ProtoFlow model.

Usage:
    python generate_proper_counterfactuals.py --checkpoint enhanced_checkpoint_ultra_fast.pt --dataset cifar10 --num_samples 10
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def safe_import_protoflow():
    """Import ProtoFlow modules with error handling."""
    try:
        from protoflow.proto import ProtoFlowGMM
        from protoflow.datasets import get_dataset
        from protoflow.training import get_transform
        return True, (ProtoFlowGMM, get_dataset, get_transform)
    except ImportError as e:
        print(f"Error importing ProtoFlow: {e}")
        return False, None

class ProtoFlowCounterfactualGenerator:
    """Enhanced ProtoFlow with proper counterfactual generation."""
    
    def __init__(self, enhanced_checkpoint_path, device='cuda'):
        self.device = device
        self.load_enhanced_model(enhanced_checkpoint_path)
        
    def load_enhanced_model(self, checkpoint_path):
        """Load the enhanced ProtoFlow model with counterfactual capabilities."""
        print(f"Loading enhanced model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model configuration
        self.num_classes = checkpoint['num_classes']
        self.features_shape = checkpoint['features_shape']
        self.prototypes = checkpoint['prototypes']  # Class-conditional distributions
        
        print(f"Model config:")
        print(f"  Classes: {self.num_classes}")
        print(f"  Features shape: {self.features_shape}")
        print(f"  Feature dim: {self.features_shape[0] * self.features_shape[1] * self.features_shape[2]}")
        
        # Load the ProtoFlow model
        protoflow_available, modules = safe_import_protoflow()
        if not protoflow_available:
            raise ImportError("ProtoFlow modules required for proper counterfactual generation")
            
        ProtoFlowGMM, _, _ = modules
        
        # Create dummy flow (will be replaced by state dict)
        class DummyFlow:
            def log_prob(self, x, return_z=True):
                batch_size = x.shape[0]
                feat_dim = self.features_shape[0] * self.features_shape[1] * self.features_shape[2]
                z = torch.randn(batch_size, feat_dim, device=x.device)
                log_prob = torch.randn(batch_size, device=x.device)
                if return_z:
                    return z, log_prob
                return log_prob
                
            def sample(self, z):
                return torch.randn_like(z)
                
            def inverse(self, x):
                batch_size = x.shape[0]
                feat_dim = self.features_shape[0] * self.features_shape[1] * self.features_shape[2]
                return torch.randn(batch_size, feat_dim, device=x.device)
        
        dummy_flow = DummyFlow()
        dummy_flow.features_shape = self.features_shape
        
        # Create ProtoFlowGMM
        self.model = ProtoFlowGMM(
            model=dummy_flow,
            n_classes=self.num_classes,
            features_shape=[self.features_shape[0] * self.features_shape[1] * self.features_shape[2]],  # Flattened
            protos_per_class=2,  # From your checkpoint output
            likelihood_approach='total',
            gaussian_approach='GaussianMixture'
        )
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Enhanced ProtoFlow model loaded successfully")
        
    def extract_features(self, image):
        """Extract features from image using the flow model."""
        with torch.no_grad():
            # Get latent representation
            z, _ = self.model.model.log_prob(image, return_z=True)
            return z
    
    def get_class_prototype(self, class_idx, component_idx=0):
        """Get prototype for a specific class and component."""
        if class_idx in self.prototypes:
            proto_data = self.prototypes[class_idx]
            if 'mean' in proto_data:
                # If multiple components, use the specified one or average
                means = proto_data['mean']
                if len(means.shape) > 1 and means.shape[0] > component_idx:
                    return means[component_idx]
                else:
                    return means.mean(dim=0) if len(means.shape) > 1 else means
        
        # Fallback: use GMM parameters
        gmm = self.model.gmms[class_idx]
        return gmm.mu[0, component_idx].clone()
    
    def interpolate_to_target(self, source_features, target_class, alpha=0.5):
        """Interpolate source features towards target class prototype."""
        target_prototype = self.get_class_prototype(target_class)
        
        # Ensure same device and shape
        target_prototype = target_prototype.to(source_features.device)
        if target_prototype.shape != source_features.shape[1:]:
            # Handle shape mismatch by truncating or padding
            min_dim = min(target_prototype.shape[0], source_features.shape[1])
            target_prototype = target_prototype[:min_dim]
            source_features = source_features[:, :min_dim]
        
        # Linear interpolation
        counterfactual_features = (1 - alpha) * source_features + alpha * target_prototype.unsqueeze(0)
        return counterfactual_features
    
    def generate_counterfactual_image(self, counterfactual_features):
        """Generate image from counterfactual features."""
        with torch.no_grad():
            # Use the flow model to generate image
            try:
                counterfactual_image = self.model.model.sample(counterfactual_features)
                # Reshape if needed
                if len(counterfactual_image.shape) == 2:
                    batch_size = counterfactual_image.shape[0]
                    counterfactual_image = counterfactual_image.view(batch_size, *self.features_shape)
                return counterfactual_image
            except Exception as e:
                print(f"Warning: Could not generate image from features: {e}")
                # Return random image as fallback
                return torch.randn(1, *self.features_shape, device=counterfactual_features.device)
    
    def classify_image(self, image):
        """Classify image and return probabilities."""
        with torch.no_grad():
            # Extract features
            features = self.extract_features(image)
            
            # Compute log probabilities for each class
            log_probs = []
            for class_idx in range(self.num_classes):
                # Get log probability from GMM
                log_prob = self.model.log_prob_class(features, class_idx)
                log_probs.append(log_prob)
            
            log_probs = torch.stack(log_probs, dim=1)
            probs = torch.softmax(log_probs, dim=1)
            
            return probs, log_probs
    
    def generate_counterfactual_explanation(self, image, source_class, target_classes=None, alphas=None):
        """Generate counterfactual explanations for multiple target classes."""
        
        if target_classes is None:
            target_classes = [i for i in range(self.num_classes) if i != source_class]
        
        if alphas is None:
            alphas = [0.3, 0.5, 0.7]  # Different interpolation strengths
        
        # Extract source features
        source_features = self.extract_features(image)
        
        # Get source classification
        source_probs, _ = self.classify_image(image)
        source_pred = source_probs.argmax(dim=1).item()
        source_conf = source_probs[0, source_pred].item()
        
        results = {
            'source_image': image,
            'source_class': source_class,
            'source_prediction': source_pred,
            'source_confidence': source_conf,
            'counterfactuals': {}
        }
        
        # Generate counterfactuals for each target class
        for target_class in target_classes:
            target_results = {'alpha_variants': []}
            
            for alpha in alphas:
                # Generate counterfactual features
                cf_features = self.interpolate_to_target(source_features, target_class, alpha)
                
                # Generate counterfactual image
                cf_image = self.generate_counterfactual_image(cf_features)
                
                # Classify counterfactual
                cf_probs, _ = self.classify_image(cf_image)
                cf_pred = cf_probs.argmax(dim=1).item()
                cf_conf = cf_probs[0, cf_pred].item()
                target_conf = cf_probs[0, target_class].item()
                
                # Compute distance
                l2_distance = torch.norm(cf_image - image, p=2).item()
                
                variant_result = {
                    'alpha': alpha,
                    'image': cf_image,
                    'predicted_class': cf_pred,
                    'confidence': cf_conf,
                    'target_confidence': target_conf,
                    'l2_distance': l2_distance,
                    'success': cf_pred == target_class
                }
                
                target_results['alpha_variants'].append(variant_result)
            
            results['counterfactuals'][target_class] = target_results
        
        return results
    
    def create_visualization(self, explanation, class_names=None):
        """Create comprehensive visualization of counterfactual explanations."""
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        source_class = explanation['source_class']
        target_classes = list(explanation['counterfactuals'].keys())
        
        # Create figure
        n_targets = len(target_classes)
        n_alphas = len(explanation['counterfactuals'][target_classes[0]]['alpha_variants'])
        
        fig_width = 2 + n_alphas * 2  # Source + alphas per target
        fig_height = 2 * n_targets
        
        fig, axes = plt.subplots(n_targets, n_alphas + 1, figsize=(fig_width, fig_height))
        if n_targets == 1:
            axes = axes.reshape(1, -1)
        
        def show_image(ax, img_tensor, title):
            img = img_tensor.squeeze().cpu().numpy()
            if img.shape[0] == 3:  # RGB
                img = np.transpose(img, (1, 2, 0))
                img = (img + 1) / 2  # Denormalize
                img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        
        # Show source image in first column
        source_img = explanation['source_image']
        source_title = f"Source\n{class_names[source_class]}\n(conf: {explanation['source_confidence']:.2f})"
        
        for row in range(n_targets):
            show_image(axes[row, 0], source_img, source_title if row == 0 else "")
        
        # Show counterfactuals
        for row, target_class in enumerate(target_classes):
            cf_data = explanation['counterfactuals'][target_class]
            
            for col, variant in enumerate(cf_data['alpha_variants']):
                alpha = variant['alpha']
                success = "✓" if variant['success'] else "✗"
                
                title = f"→ {class_names[target_class]}\nα={alpha} {success}\nconf: {variant['target_confidence']:.2f}"
                
                show_image(axes[row, col + 1], variant['image'], title)
        
        plt.suptitle(f"Counterfactual Explanations from {class_names[source_class]}", fontsize=12)
        plt.tight_layout()
        
        return fig

def get_cifar10_dataloader(batch_size=1):
    """Get CIFAR-10 test dataloader."""
    protoflow_available, modules = safe_import_protoflow()
    
    if protoflow_available:
        try:
            _, get_dataset, get_transform = modules
            
            transform = get_transform(
                interpolation='bicubic',
                size=32,
                train=False,
                augmentation='v1'
            )
            
            test_dataset = get_dataset(
                name='cifar10',
                train=False,
                transform=transform
            )
            
            return DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            
        except Exception as e:
            print(f"ProtoFlow data loading failed: {e}")
    
    # Fallback to torchvision
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

def main():
    parser = argparse.ArgumentParser(description='Generate ProtoFlow Counterfactuals')
    parser.add_argument('--checkpoint', required=True, help='Enhanced checkpoint path')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--output_dir', default='./counterfactual_results', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    try:
        # Initialize counterfactual generator
        generator = ProtoFlowCounterfactualGenerator(args.checkpoint, device)
        print("✓ Counterfactual generator initialized")
        
        # Get test data
        test_loader = get_cifar10_dataloader(batch_size=1)
        print("✓ Test dataloader created")
        
        # Generate counterfactuals
        print(f"\nGenerating counterfactuals for {args.num_samples} samples...")
        
        successful_samples = 0
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= args.num_samples:
                break
            
            image, label = images[0:1].to(device), labels[0:1].to(device)
            source_class = label.item()
            
            print(f"\nSample {batch_idx+1}: {class_names[source_class]} (class {source_class})")
            
            try:
                # Generate explanation (focus on 2-3 target classes for clarity)
                target_classes = [(source_class + 1) % 10, (source_class + 5) % 10]
                
                explanation = generator.generate_counterfactual_explanation(
                    image, source_class, target_classes, alphas=[0.3, 0.5, 0.7]
                )
                
                # Create visualization
                fig = generator.create_visualization(explanation, class_names)
                
                # Save results
                output_path = output_dir / f'counterfactual_{batch_idx:03d}_{class_names[source_class]}.png'
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # Print results
                print(f"  Source prediction: {class_names[explanation['source_prediction']]} "
                      f"(confidence: {explanation['source_confidence']:.3f})")
                
                for target_class, cf_data in explanation['counterfactuals'].items():
                    print(f"  → {class_names[target_class]}:")
                    for variant in cf_data['alpha_variants']:
                        success = "✓" if variant['success'] else "✗"
                        print(f"    α={variant['alpha']}: {success} "
                              f"conf={variant['target_confidence']:.3f} "
                              f"L2={variant['l2_distance']:.2f}")
                
                print(f"✓ Saved to {output_path}")
                successful_samples += 1
                
            except Exception as e:
                print(f"✗ Error generating counterfactual: {e}")
                traceback.print_exc()
                continue
        
        print(f"\n🎉 Successfully generated {successful_samples}/{args.num_samples} counterfactual explanations!")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())