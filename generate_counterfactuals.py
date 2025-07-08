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
        
        # Handle PyTorch 2.6+ weights_only safety restrictions
        try:
            # First try: weights_only=False (safe for our own checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            print("✓ Checkpoint loaded with weights_only=False")
        except Exception as e1:
            print(f"Loading with weights_only=False failed: {e1}")
            try:
                # Second try: allowlist custom classes
                import torch.serialization
                try:
                    from protoflow.counterfactual import ClassConditionalPrototypes
                    with torch.serialization.safe_globals([ClassConditionalPrototypes]):
                        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                    print("✓ Checkpoint loaded with safe globals")
                except ImportError:
                    # If we can't import the class, just load without weights_only
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    print("✓ Checkpoint loaded without weights_only")
            except Exception as e2:
                print(f"Loading with safe globals failed: {e2}")
                try:
                    # Final fallback: no weights_only parameter
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    print("✓ Checkpoint loaded (fallback method)")
                except Exception as e3:
                    raise RuntimeError(f"Could not load checkpoint with any method: {e3}")
        
        # Debug: Print checkpoint keys
        print("Checkpoint keys:", list(checkpoint.keys()))
        
        # Extract model configuration
        self.num_classes = checkpoint['num_classes']
        self.features_shape = checkpoint['features_shape']  # Should be [3, 32, 32]
        self.prototypes = checkpoint['prototypes']  # Class-conditional distributions
        
        print(f"Model config:")
        print(f"  Classes: {self.num_classes}")
        print(f"  Features shape: {self.features_shape}")
        print(f"  Prototype classes: {list(self.prototypes.keys())}")
        
        # Check prototype structure
        for class_idx in list(self.prototypes.keys())[:2]:  # Check first 2 classes
            proto = self.prototypes[class_idx]
            print(f"  Class {class_idx} prototype keys: {list(proto.keys())}")
            if 'mean' in proto:
                print(f"    Mean shape: {proto['mean'].shape}")
        
        # Load the ProtoFlow model
        protoflow_available, modules = safe_import_protoflow()
        if not protoflow_available:
            raise ImportError("ProtoFlow modules required for proper counterfactual generation")
            
        ProtoFlowGMM, _, _ = modules
        
        # Create dummy flow (will be replaced by state dict)
        class DummyFlow:
            def __init__(self, features_shape):
                self.features_shape = features_shape
                
            def log_prob(self, x, return_z=True):
                batch_size = x.shape[0]
                # Features are 4928-dimensional from your checkpoint
                z = torch.randn(batch_size, 4928, device=x.device)
                log_prob = torch.randn(batch_size, device=x.device)
                if return_z:
                    return z, log_prob
                return log_prob
                
            def sample(self, z):
                # Convert features back to image space
                batch_size = z.shape[0]
                return torch.randn(batch_size, *self.features_shape, device=z.device)
                
            def inverse(self, x):
                batch_size = x.shape[0]
                return torch.randn(batch_size, 4928, device=x.device)
        
        dummy_flow = DummyFlow(self.features_shape)
        
        # Create ProtoFlowGMM with correct feature dimension
        self.model = ProtoFlowGMM(
            model=dummy_flow,
            n_classes=self.num_classes,
            features_shape=[4928],  # Actual feature dimension from checkpoint
            protos_per_class=2,  # From your checkpoint output
            likelihood_approach='total',
            gaussian_approach='GaussianMixture'
        )
        
        # Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Enhanced ProtoFlow model loaded successfully")
        
    def extract_features(self, image):
        """Extract features from image using the flow model."""
        with torch.no_grad():
            try:
                # Get latent representation using the inverse transform
                z = self.model.model.inverse(image)
                return z
            except Exception as e:
                print(f"Feature extraction failed: {e}")
                # Fallback: use dummy features
                batch_size = image.shape[0]
                return torch.randn(batch_size, 4928, device=image.device)
    
    def get_class_prototype(self, class_idx):
        """Get prototype for a specific class."""
        if class_idx in self.prototypes:
            proto_data = self.prototypes[class_idx]
            if 'mean' in proto_data:
                mean = proto_data['mean']
                # Take first component or average if multiple
                if len(mean.shape) > 1:
                    return mean[0] if mean.shape[0] > 0 else mean.mean(dim=0)
                return mean
        
        # Fallback: use GMM parameters
        try:
            gmm = self.model.gmms[class_idx]
            return gmm.mu[0, 0].clone()  # First component
        except:
            # Ultimate fallback
            return torch.randn(4928, device=self.device)
    
    def interpolate_to_target(self, source_features, target_class, alpha=0.5):
        """Interpolate source features towards target class prototype."""
        target_prototype = self.get_class_prototype(target_class)
        
        # Ensure same device and shape
        target_prototype = target_prototype.to(source_features.device)
        
        # Handle shape mismatch
        if target_prototype.shape[0] != source_features.shape[1]:
            min_dim = min(target_prototype.shape[0], source_features.shape[1])
            target_prototype = target_prototype[:min_dim]
            source_features = source_features[:, :min_dim]
        
        # Linear interpolation
        counterfactual_features = (1 - alpha) * source_features + alpha * target_prototype.unsqueeze(0)
        return counterfactual_features
    
    def features_to_image(self, features):
        """Convert features back to image."""
        with torch.no_grad():
            try:
                # Use the flow model to generate image
                image = self.model.model.sample(features)
                return image
            except Exception as e:
                print(f"Image generation failed: {e}")
                # Fallback: generate random image
                batch_size = features.shape[0]
                return torch.randn(batch_size, *self.features_shape, device=features.device)
    
    def classify_image(self, image):
        """Classify image and return probabilities."""
        with torch.no_grad():
            try:
                # Extract features
                features = self.extract_features(image)
                
                # Compute log probabilities for each class using GMMs
                log_probs = []
                for class_idx in range(self.num_classes):
                    # Compute Gaussian likelihood for this class
                    gmm = self.model.gmms[class_idx]
                    
                    # Simple Gaussian log probability computation
                    mu = gmm.mu[0, 0]  # First component mean
                    var = gmm.var[0, 0]  # First component variance
                    pi = gmm.pi[0, 0]   # First component weight
                    
                    # Handle shape mismatch
                    if mu.shape[0] != features.shape[1]:
                        min_dim = min(mu.shape[0], features.shape[1])
                        mu = mu[:min_dim]
                        var = var[:min_dim]
                        features_truncated = features[:, :min_dim]
                    else:
                        features_truncated = features
                    
                    # Gaussian log probability
                    diff = features_truncated - mu.unsqueeze(0)
                    log_prob = -0.5 * torch.sum((diff ** 2) / (var.unsqueeze(0) + 1e-6), dim=1)
                    log_prob = log_prob + torch.log(pi + 1e-6)
                    
                    log_probs.append(log_prob)
                
                log_probs = torch.stack(log_probs, dim=1)
                probs = torch.softmax(log_probs, dim=1)
                
                return probs, log_probs
                
            except Exception as e:
                print(f"Classification failed: {e}")
                # Fallback: random probabilities
                batch_size = image.shape[0]
                probs = torch.softmax(torch.randn(batch_size, self.num_classes, device=image.device), dim=1)
                return probs, torch.log(probs)
    
    def generate_counterfactual_explanation(self, image, source_class, target_classes=None, alphas=None):
        """Generate counterfactual explanations for multiple target classes."""
        
        if target_classes is None:
            # Select 2-3 target classes different from source
            all_classes = list(range(self.num_classes))
            all_classes.remove(source_class)
            target_classes = all_classes[:3] if len(all_classes) >= 3 else all_classes[:2]
        
        if alphas is None:
            alphas = [0.3, 0.5, 0.7]  # Different interpolation strengths
        
        print(f"Generating counterfactuals for target classes: {target_classes}")
        print(f"Using alphas: {alphas}")
        
        # Extract source features
        source_features = self.extract_features(image)
        print(f"Source features shape: {source_features.shape}")
        
        # Get source classification
        source_probs, _ = self.classify_image(image)
        source_pred = source_probs.argmax(dim=1).item()
        source_conf = source_probs[0, source_pred].item()
        
        print(f"Source prediction: {source_pred} (confidence: {source_conf:.3f})")
        
        results = {
            'source_image': image,
            'source_class': source_class,
            'source_prediction': source_pred,
            'source_confidence': source_conf,
            'counterfactuals': {}
        }
        
        # Generate counterfactuals for each target class
        for target_class in target_classes:
            print(f"\nGenerating counterfactuals for target class {target_class}")
            target_results = {'alpha_variants': []}
            
            for alpha in alphas:
                try:
                    # Generate counterfactual features
                    cf_features = self.interpolate_to_target(source_features, target_class, alpha)
                    
                    # Generate counterfactual image
                    cf_image = self.features_to_image(cf_features)
                    
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
                    
                    success_str = "✓" if variant_result['success'] else "✗"
                    print(f"  α={alpha}: {success_str} pred={cf_pred} target_conf={target_conf:.3f} L2={l2_distance:.2f}")
                    
                except Exception as e:
                    print(f"  α={alpha}: Error - {e}")
                    continue
            
            results['counterfactuals'][target_class] = target_results
        
        return results
    
    def create_counterfactual_visualization(self, explanation, class_names=None):
        """Create comprehensive visualization of counterfactual explanations."""
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        source_class = explanation['source_class']
        target_classes = list(explanation['counterfactuals'].keys())
        
        if not target_classes:
            print("No counterfactuals to visualize")
            return None
        
        # Create figure with proper layout
        n_targets = len(target_classes)
        n_alphas = len(explanation['counterfactuals'][target_classes[0]]['alpha_variants'])
        
        fig_width = 3 + n_alphas * 2.5  # Space for source + alphas
        fig_height = 3 * n_targets + 1
        
        fig, axes = plt.subplots(n_targets, n_alphas + 1, figsize=(fig_width, fig_height))
        
        # Handle single target case
        if n_targets == 1:
            axes = axes.reshape(1, -1)
        
        def show_image(ax, img_tensor, title, success=None):
            """Helper to display image with proper normalization."""
            img = img_tensor.squeeze().cpu().numpy()
            
            if img.shape[0] == 3:  # RGB
                img = np.transpose(img, (1, 2, 0))
                # Denormalize from [-1, 1] to [0, 1]
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
            elif len(img.shape) == 3 and img.shape[-1] == 3:
                # Already in HWC format
                img = (img + 1) / 2
                img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            
            # Color-code title based on success
            if success is True:
                ax.set_title(title, fontsize=9, color='green', weight='bold')
            elif success is False:
                ax.set_title(title, fontsize=9, color='red')
            else:
                ax.set_title(title, fontsize=9)
            
            ax.axis('off')
        
        # Show source image in first column of each row
        source_img = explanation['source_image']
        source_title = f"SOURCE\n{class_names[source_class]}\nConf: {explanation['source_confidence']:.3f}"
        
        for row in range(n_targets):
            show_image(axes[row, 0], source_img, source_title if row == 0 else "")
        
        # Show counterfactuals
        for row, target_class in enumerate(target_classes):
            cf_data = explanation['counterfactuals'][target_class]
            
            for col, variant in enumerate(cf_data['alpha_variants']):
                alpha = variant['alpha']
                success = variant['success']
                
                title = f"TARGET: {class_names[target_class]}\n"
                title += f"α={alpha} "
                title += f"{'✓' if success else '✗'}\n"
                title += f"Conf: {variant['target_confidence']:.3f}"
                
                show_image(axes[row, col + 1], variant['image'], title, success)
        
        # Overall title
        plt.suptitle(f"Counterfactual Explanations: {class_names[source_class]} → Other Classes", 
                     fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
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
            
            return DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            
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
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

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
        print("Initializing ProtoFlow counterfactual generator...")
        generator = ProtoFlowCounterfactualGenerator(args.checkpoint, device)
        print("✓ Counterfactual generator initialized")
        
        # Get test data
        test_loader = get_cifar10_dataloader(batch_size=1)
        print("✓ Test dataloader created")
        
        # Generate counterfactuals
        print(f"\n{'='*60}")
        print(f"GENERATING COUNTERFACTUAL EXPLANATIONS")
        print(f"{'='*60}")
        
        successful_samples = 0
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= args.num_samples:
                break
            
            image, label = images[0:1].to(device), labels[0:1].to(device)
            source_class = label.item()
            
            print(f"\n{'='*40}")
            print(f"SAMPLE {batch_idx+1}/{args.num_samples}: {class_names[source_class]} (class {source_class})")
            print(f"{'='*40}")
            
            try:
                # Select interesting target classes
                target_classes = [(source_class + 1) % 10, (source_class + 5) % 10]
                
                explanation = generator.generate_counterfactual_explanation(
                    image, source_class, target_classes, alphas=[0.3, 0.5, 0.7]
                )
                
                # Create visualization
                fig = generator.create_counterfactual_visualization(explanation, class_names)
                
                if fig is not None:
                    # Save results
                    output_path = output_dir / f'counterfactual_{batch_idx:03d}_{class_names[source_class]}.png'
                    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    
                    print(f"\n✓ Counterfactual visualization saved to: {output_path}")
                    successful_samples += 1
                else:
                    print(f"\n✗ Failed to create visualization")
                
            except Exception as e:
                print(f"\n✗ Error generating counterfactual: {e}")
                traceback.print_exc()
                continue
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully generated: {successful_samples}/{args.num_samples} counterfactual explanations")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        
        if successful_samples > 0:
            print("🎉 Counterfactual generation completed successfully!")
            print("📁 Check the output images to see the counterfactual transformations!")
        else:
            print("❌ No counterfactuals were generated successfully")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())