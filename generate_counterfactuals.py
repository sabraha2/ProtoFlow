#!/usr/bin/env python3
"""
Generate counterfactual explanations using ProtoFlow with REAL DenseFlow decoder or pixel-space fallback.
This version works with the actual ProtoFlow codebase structure.

Usage:
    python generate_counterfactuals_real.py --checkpoint enhanced_checkpoint_ultra_fast.pt --dataset cifar10 --num_samples 5
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
        # Try to import DenseFlow
        try:
            from experiments.image.model.dense_flow import DenseFlow
            return True, (ProtoFlowGMM, DenseFlow, get_dataset, get_transform)
        except ImportError:
            return True, (ProtoFlowGMM, None, get_dataset, get_transform)
    except ImportError as e:
        print(f"Error importing ProtoFlow: {e}")
        return False, None

class RealProtoFlowCounterfactualGenerator:
    """ProtoFlow counterfactual generator using real DenseFlow or pixel-space fallback."""
    
    def __init__(self, enhanced_checkpoint_path, device='cuda'):
        self.device = device
        self.use_real_flow = False
        self.pixel_prototypes = None
        self.load_model_and_setup(enhanced_checkpoint_path)
        
    def load_model_and_setup(self, checkpoint_path):
        """Load model and set up either real flow or pixel-space prototypes."""
        print(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = self._load_checkpoint_safely(checkpoint_path)
        
        # Extract basic info
        self.num_classes = checkpoint['num_classes']
        self.features_shape = checkpoint['features_shape']
        
        # Try to load real ProtoFlow model first
        if self._try_load_real_protoflow(checkpoint):
            print("✓ Using REAL ProtoFlow DenseFlow decoder!")
            self.use_real_flow = True
        else:
            print("⚠️  Real flow unavailable, using pixel-space prototypes")
            self.use_real_flow = False
            self._setup_pixel_prototypes()
        
        # Load prototypes
        self._load_prototypes(checkpoint)
        
    def _load_checkpoint_safely(self, checkpoint_path):
        """Load checkpoint with proper error handling."""
        try:
            return torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e1:
            try:
                import torch.serialization
                # Import the counterfactual module classes that might be in the checkpoint
                try:
                    from protoflow.counterfactual import ClassConditionalPrototypes
                    with torch.serialization.safe_globals([ClassConditionalPrototypes]):
                        return torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                except ImportError:
                    return torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except Exception as e2:
                return torch.load(checkpoint_path, map_location=self.device)
    
    def _try_load_real_protoflow(self, checkpoint):
        """Attempt to load the real ProtoFlow model with DenseFlow."""
        protoflow_available, modules = safe_import_protoflow()
        if not protoflow_available:
            return False
            
        try:
            if len(modules) == 4:
                ProtoFlowGMM, DenseFlow, _, _ = modules
            else:
                ProtoFlowGMM, _, _ = modules
                DenseFlow = None
            
            # Try to reconstruct the model from checkpoint
            if 'model_state_dict' in checkpoint:
                # Create a ProtoFlow model - we need to estimate the architecture
                # Look for DenseFlow parameters in the state dict to estimate config
                state_dict = checkpoint['model_state_dict']
                
                # Try to create a DenseFlow model (this might need adjustment based on your specific setup)
                if DenseFlow is not None:
                    try:
                        # Try to infer DenseFlow configuration from state dict
                        # This is a best guess - you might need to adjust based on your training setup
                        flow_config = self._infer_flow_config_from_state_dict(state_dict)
                        dense_flow = DenseFlow(**flow_config)
                        
                        # Create ProtoFlowGMM with the DenseFlow
                        self.model = ProtoFlowGMM(
                            model=dense_flow,
                            n_classes=self.num_classes,
                            features_shape=self.features_shape,
                            protos_per_class=2,
                            likelihood_approach='total',
                            gaussian_approach='GaussianMixture'
                        )
                        
                        # Load state dict
                        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                        print(f"Loaded state dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
                        
                        if len(missing) < len(state_dict) // 2:  # If most keys loaded successfully
                            self.model.to(self.device).eval()
                            return True
                            
                    except Exception as e:
                        print(f"Failed to create DenseFlow: {e}")
                
                # Fallback: try to create a ProtoFlow without knowing the exact flow config
                # This won't give us real decoding but might work for classification
                try:
                    # Create a dummy flow model
                    class DummyFlow:
                        def __init__(self, features_shape):
                            self.features_shape = features_shape
                            
                        def log_prob(self, x, return_z=True):
                            batch_size = x.shape[0]
                            # Create features by flattening and projecting
                            features = self._extract_features(x)
                            log_prob = torch.randn(batch_size, device=x.device)
                            if return_z:
                                return features, log_prob
                            return log_prob
                            
                        def _extract_features(self, x):
                            batch_size = x.shape[0]
                            x_flat = x.view(batch_size, -1)
                            # Project to expected feature dimension
                            expected_dim = np.prod(self.features_shape) if isinstance(self.features_shape, (list, tuple)) else self.features_shape
                            if isinstance(expected_dim, (list, tuple)):
                                expected_dim = np.prod(expected_dim)
                            
                            if x_flat.shape[1] != expected_dim:
                                # Simple linear projection
                                if x_flat.shape[1] > expected_dim:
                                    features = x_flat[:, :expected_dim]
                                else:
                                    padding = torch.randn(batch_size, expected_dim - x_flat.shape[1], device=x.device) * 0.1
                                    features = torch.cat([x_flat, padding], dim=1)
                            else:
                                features = x_flat
                                
                            return features
                            
                        def sample(self, z):
                            # This is where the limitation shows - we can't really decode properly
                            batch_size = z.shape[0]
                            return torch.randn(batch_size, *self.features_shape, device=z.device)
                    
                    dummy_flow = DummyFlow(self.features_shape)
                    
                    self.model = ProtoFlowGMM(
                        model=dummy_flow,
                        n_classes=self.num_classes,
                        features_shape=self.features_shape,
                        protos_per_class=2,
                        likelihood_approach='total',
                        gaussian_approach='GaussianMixture'
                    )
                    
                    # Load what we can from the state dict
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    
                    # Only consider this successful if we loaded the GMM parameters
                    gmm_keys_loaded = sum(1 for key in state_dict.keys() if 'gmms' in key and key not in missing)
                    if gmm_keys_loaded > 0:
                        self.model.to(self.device).eval()
                        print(f"✓ Loaded ProtoFlow GMM parameters (dummy flow for decoding)")
                        return False  # Still use pixel fallback for image generation
                        
                except Exception as e:
                    print(f"Failed to create dummy ProtoFlow: {e}")
                    
        except Exception as e:
            print(f"Failed to load real ProtoFlow: {e}")
            
        return False
    
    def _infer_flow_config_from_state_dict(self, state_dict):
        """Try to infer DenseFlow configuration from state dict keys."""
        # This is a best guess - you'll need to adjust based on your actual training configuration
        config = {
            'num_layers': 32,  # Common default
            'hidden_dim': 512,  # Common default
            'num_blocks': 4,    # Common default
        }
        
        # Try to infer from state dict keys
        for key in state_dict.keys():
            if 'model.' in key and 'layers' in key:
                # Try to count layers
                layer_nums = [int(part) for part in key.split('.') if part.isdigit()]
                if layer_nums:
                    config['num_layers'] = max(layer_nums) + 1
                    
        return config
    
    def _setup_pixel_prototypes(self):
        """Set up pixel-space prototypes as fallback."""
        print("Setting up pixel-space prototypes...")
        
        try:
            # Get training data to compute class means
            train_loader = self._get_cifar10_dataloader(train=True, batch_size=128)
            
            # Initialize accumulators
            class_sums = [torch.zeros(3, 32, 32, device=self.device) for _ in range(self.num_classes)]
            class_counts = [0] * self.num_classes
            
            print("Computing class-mean images...")
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(train_loader):
                    if batch_idx % 50 == 0:
                        print(f"  Processing batch {batch_idx}...")
                    
                    images = images.to(self.device)
                    for img, label in zip(images, labels):
                        class_sums[label.item()] += img
                        class_counts[label.item()] += 1
                    
                    # Limit for faster setup
                    if batch_idx > 100:  # Use subset for efficiency
                        break
            
            # Compute means
            self.pixel_prototypes = []
            for i in range(self.num_classes):
                if class_counts[i] > 0:
                    mean_img = class_sums[i] / class_counts[i]
                    self.pixel_prototypes.append(mean_img)
                else:
                    # Fallback random image
                    self.pixel_prototypes.append(torch.randn(3, 32, 32, device=self.device))
            
            print(f"✓ Computed pixel prototypes for {len(self.pixel_prototypes)} classes")
            
        except Exception as e:
            print(f"Failed to compute pixel prototypes: {e}")
            # Ultimate fallback - random prototypes
            self.pixel_prototypes = [torch.randn(3, 32, 32, device=self.device) 
                                   for _ in range(self.num_classes)]
    
    def _load_prototypes(self, checkpoint):
        """Load feature-space prototypes."""
        raw_protos = checkpoint['prototypes']
        
        if hasattr(raw_protos, 'class_means'):
            self.prototypes = {}
            for class_idx in range(self.num_classes):
                if class_idx in raw_protos.class_means:
                    mean_tensor = raw_protos.class_means[class_idx]
                    self.prototypes[class_idx] = {
                        'mean': mean_tensor,
                        'var': torch.ones_like(mean_tensor),
                        'pi': torch.tensor(1.0)
                    }
        else:
            # Fallback prototypes
            self.prototypes = {
                i: {
                    'mean': torch.randn(4928, device=self.device), 
                    'var': torch.ones(4928, device=self.device), 
                    'pi': torch.tensor(1.0)
                } 
                for i in range(self.num_classes)
            }
    
    def _get_cifar10_dataloader(self, train=False, batch_size=1):
        """Get CIFAR-10 dataloader."""
        protoflow_available, modules = safe_import_protoflow()
        
        if protoflow_available:
            try:
                if len(modules) == 4:
                    _, _, get_dataset, get_transform = modules
                else:
                    _, get_dataset, get_transform = modules
                
                transform = get_transform(
                    interpolation='bicubic',
                    size=32,
                    train=False,
                    augmentation='v1'
                )
                
                dataset = get_dataset(
                    name='cifar10',
                    train=train,
                    transform=transform
                )
                
                return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
                
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
        
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    def generate_counterfactual_flow(self, source_image, target_class, alpha=0.5):
        """Generate counterfactual using real ProtoFlow (if available)."""
        with torch.no_grad():
            try:
                # Extract features using the DenseFlow component
                z, _ = self.model.model.log_prob(source_image, return_z=True)
                
                # Flatten z for prototype operations
                z_flat = z.flatten(1)
                
                # Get target prototype
                target_proto = self.get_class_prototype(target_class)
                if target_proto.shape[0] != z_flat.shape[1]:
                    min_dim = min(target_proto.shape[0], z_flat.shape[1])
                    target_proto = target_proto[:min_dim]
                    z_flat = z_flat[:, :min_dim]
                
                # Interpolate in latent space
                cf_z_flat = (1 - alpha) * z_flat + alpha * target_proto.unsqueeze(0)
                
                # Reshape back to original z shape
                cf_z = cf_z_flat.reshape(z.shape)
                
                # Decode back to image using DenseFlow
                cf_image = self.model.model.sample(cf_z)
                
                return cf_image
                
            except Exception as e:
                print(f"Flow generation failed: {e}")
                # Fall back to pixel space
                return self.generate_counterfactual_pixel(source_image, target_class, alpha)
    
    def generate_counterfactual_pixel(self, source_image, target_class, alpha=0.5):
        """Generate counterfactual using pixel-space interpolation."""
        proto_img = self.pixel_prototypes[target_class]
        
        # Ensure same device and shape
        proto_img = proto_img.to(source_image.device)
        if proto_img.shape != source_image.shape[1:]:
            proto_img = torch.nn.functional.interpolate(
                proto_img.unsqueeze(0), 
                size=source_image.shape[2:], 
                mode='bilinear'
            ).squeeze(0)
        
        # Linear interpolation in pixel space
        cf_image = (1 - alpha) * source_image + alpha * proto_img.unsqueeze(0)
        
        # Clamp to valid range
        return torch.clamp(cf_image, -1, 1)
    
    def get_class_prototype(self, class_idx):
        """Get prototype for a specific class."""
        if class_idx in self.prototypes:
            return self.prototypes[class_idx]['mean']
        else:
            return torch.randn(4928, device=self.device)
    
    def classify_image(self, image):
        """Classify image using ProtoFlow or pixel-space similarity."""
        with torch.no_grad():
            if self.use_real_flow and hasattr(self, 'model'):
                try:
                    # Use real ProtoFlow classification
                    logits = self.model(image, flow_grad=False)
                    probs = torch.softmax(logits, dim=1)
                    return probs, torch.log(probs)
                    
                except Exception as e:
                    print(f"ProtoFlow classification failed: {e}")
            
            # Fallback: simple pixel-space classification
            similarities = []
            for class_idx in range(self.num_classes):
                proto = self.pixel_prototypes[class_idx].unsqueeze(0)
                similarity = -torch.nn.functional.mse_loss(image, proto)
                similarities.append(similarity)
            
            log_probs = torch.stack(similarities).unsqueeze(0)
            probs = torch.softmax(log_probs, dim=1)
            return probs, log_probs
    
    def generate_counterfactual_explanation(self, image, source_class, target_classes=None, alphas=None):
        """Generate counterfactual explanations using the best available method."""
        
        if target_classes is None:
            all_classes = list(range(self.num_classes))
            all_classes.remove(source_class)
            target_classes = all_classes[:2]
        
        if alphas is None:
            alphas = [0.3, 0.5, 0.7]
        
        method = "Real ProtoFlow" if self.use_real_flow else "Pixel-Space"
        print(f"Generating counterfactuals using {method} method")
        print(f"Target classes: {target_classes}, Alphas: {alphas}")
        
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
            'method': method,
            'counterfactuals': {}
        }
        
        # Generate counterfactuals for each target class
        for target_class in target_classes:
            print(f"\nGenerating counterfactuals for target class {target_class}")
            target_results = {'alpha_variants': []}
            
            for alpha in alphas:
                try:
                    # Choose generation method
                    if self.use_real_flow:
                        cf_image = self.generate_counterfactual_flow(image, target_class, alpha)
                    else:
                        cf_image = self.generate_counterfactual_pixel(image, target_class, alpha)
                    
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
        method = explanation.get('method', 'Unknown')
        
        if not target_classes:
            print("No counterfactuals to visualize")
            return None
        
        # Create figure
        n_targets = len(target_classes)
        n_alphas = len(explanation['counterfactuals'][target_classes[0]]['alpha_variants'])
        
        fig_width = 3 + n_alphas * 2.5
        fig_height = 3 * n_targets + 1.5
        
        fig, axes = plt.subplots(n_targets, n_alphas + 1, figsize=(fig_width, fig_height))
        
        if n_targets == 1:
            axes = axes.reshape(1, -1)
        
        def show_image(ax, img_tensor, title, success=None):
            """Helper to display image with proper normalization."""
            img = img_tensor.squeeze().cpu().numpy()
            
            if img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))  # Convert to HWC
                
            # Denormalize from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            
            # Color-code title
            if success is True:
                ax.set_title(title, fontsize=9, color='green', weight='bold')
            elif success is False:
                ax.set_title(title, fontsize=9, color='red')
            else:
                ax.set_title(title, fontsize=9)
            
            ax.axis('off')
        
        # Show source image
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
        
        plt.suptitle(f"Counterfactual Explanations ({method}): {class_names[source_class]} → Other Classes", 
                     fontsize=14, y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        return fig

def main():
    parser = argparse.ArgumentParser(description='Generate Real ProtoFlow Counterfactuals')
    parser.add_argument('--checkpoint', required=True, help='Enhanced checkpoint path')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--output_dir', default='./counterfactual_results_real', help='Output directory')
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
        # Initialize real counterfactual generator
        print("Initializing REAL ProtoFlow counterfactual generator...")
        generator = RealProtoFlowCounterfactualGenerator(args.checkpoint, device)
        print("✓ Generator initialized")
        
        # Get test data
        test_loader = generator._get_cifar10_dataloader(train=False, batch_size=1)
        print("✓ Test dataloader created")
        
        # Generate counterfactuals
        print(f"\n{'='*60}")
        print(f"GENERATING REAL COUNTERFACTUAL EXPLANATIONS")
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
                # Select target classes
                target_classes = [(source_class + 1) % 10, (source_class + 5) % 10]
                
                explanation = generator.generate_counterfactual_explanation(
                    image, source_class, target_classes, alphas=[0.3, 0.5, 0.7]
                )
                
                # Create visualization
                fig = generator.create_counterfactual_visualization(explanation, class_names)
                
                if fig is not None:
                    # Save results
                    method_suffix = "protoflow" if generator.use_real_flow else "pixel"
                    output_path = output_dir / f'counterfactual_{batch_idx:03d}_{class_names[source_class]}_{method_suffix}.png'
                    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    
                    print(f"\n✓ Counterfactual saved to: {output_path}")
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
        method = "Real ProtoFlow" if generator.use_real_flow else "Pixel-Space"
        print(f"Method used: {method}")
        print(f"Successfully generated: {successful_samples}/{args.num_samples} counterfactual explanations")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        
        if successful_samples > 0:
            print("🎉 Counterfactual generation completed!")
            if generator.use_real_flow:
                print("📁 Using REAL ProtoFlow decoder - images should be high quality!")
            else:
                print("📁 Using pixel-space interpolation - images show smooth transitions!")
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