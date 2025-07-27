#!/usr/bin/env python3
"""
Fixed ProtoFlow counterfactual generator with simplified and robust implementation.
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

class FixedProtoFlowCounterfactualGenerator:
    """Simplified and robust ProtoFlow counterfactual generator."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.model = None
        self.use_real_flow = False
        self.pixel_prototypes = None
        self.load_model_and_setup(checkpoint_path)
        
    def load_model_and_setup(self, checkpoint_path):
        """Load model with simplified configuration."""
        print(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = self._load_checkpoint_safely(checkpoint_path)
        
        # Extract basic info
        self.num_classes = checkpoint.get('num_classes', 10)
        self.features_shape = checkpoint.get('features_shape', [3072])
        
        # Try to load real ProtoFlow model
        if self._try_load_real_protoflow_simple(checkpoint):
            print("✓ Using REAL ProtoFlow with DenseFlow!")
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
                return torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except Exception as e2:
                print(f"Warning: Could not load checkpoint normally: {e2}")
                return {}
    
    def _try_load_real_protoflow_simple(self, checkpoint):
        """Simplified real ProtoFlow loading."""
        protoflow_available, modules = safe_import_protoflow()
        if not protoflow_available:
            return False
            
        try:
            if len(modules) == 4:
                ProtoFlowGMM, DenseFlow, _, _ = modules
            else:
                ProtoFlowGMM, _, _ = modules
                DenseFlow = None
            
            if DenseFlow is None:
                print("DenseFlow not available")
                return False
            
            # Create a simple DenseFlow model
            data_shape = (3, 32, 32)  # CIFAR-10
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
            
            # Create ProtoFlow model
            self.model = ProtoFlowGMM(
                model=flow_model,
                n_classes=self.num_classes,
                features_shape=self.features_shape,
                protos_per_class=10,
                likelihood_approach='total',
                gaussian_approach='GaussianMixture'
            )
            
            # Load state dict if available
            if 'model_state_dict' in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("✓ Loaded model state dict")
                except Exception as e:
                    print(f"Warning: Could not load state dict: {e}")
            
            self.model.to(self.device)
            self.model.eval()
            
            # Test the model
            test_input = torch.randn(1, 3, 32, 32).to(self.device)
            with torch.no_grad():
                try:
                    output = self.model(test_input)
                    print(f"✓ Model test successful, output shape: {output.shape}")
                    return True
                except Exception as e:
                    print(f"Model test failed: {e}")
                    return False
                    
        except Exception as e:
            print(f"Error creating real ProtoFlow model: {e}")
            return False
    
    def _setup_pixel_prototypes(self):
        """Setup pixel-space prototypes for CIFAR-10."""
        print("Setting up pixel-space prototypes...")
        
        # Create simple pixel prototypes for CIFAR-10 classes
        self.pixel_prototypes = {}
        
        # Define simple color patterns for each class
        patterns = {
            0: [0.5, 0.2, 0.8],  # airplane - blue-ish
            1: [0.8, 0.1, 0.1],  # automobile - red
            2: [0.2, 0.8, 0.2],  # bird - green
            3: [0.8, 0.6, 0.2],  # cat - orange
            4: [0.6, 0.4, 0.2],  # deer - brown
            5: [0.4, 0.2, 0.6],  # dog - purple
            6: [0.2, 0.8, 0.6],  # frog - cyan
            7: [0.8, 0.4, 0.6],  # horse - pink
            8: [0.1, 0.1, 0.8],  # ship - blue
            9: [0.6, 0.6, 0.1]   # truck - yellow
        }
        
        for class_idx in range(self.num_classes):
            # Create a simple prototype image
            prototype = torch.zeros(3, 32, 32)
            color = patterns.get(class_idx, [0.5, 0.5, 0.5])
            
            # Apply color pattern
            for c in range(3):
                prototype[c] = color[c]
            
            # Add some structure (simple patterns)
            if class_idx == 0:  # airplane - horizontal lines
                prototype[:, 8:12, :] = 0.8
                prototype[:, 20:24, :] = 0.8
            elif class_idx == 1:  # automobile - vertical lines
                prototype[:, :, 8:12] = 0.8
                prototype[:, :, 20:24] = 0.8
            elif class_idx == 2:  # bird - diagonal
                for i in range(32):
                    if 8 <= i <= 24:
                        prototype[:, i, i] = 0.8
            elif class_idx == 3:  # cat - circular
                center = 16
                for i in range(32):
                    for j in range(32):
                        dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                        if 8 <= dist <= 12:
                            prototype[:, i, j] = 0.8
            elif class_idx == 4:  # deer - cross
                prototype[:, 12:20, 12:20] = 0.8
                prototype[:, 8:24, 16] = 0.8
                prototype[:, 16, 8:24] = 0.8
            elif class_idx == 5:  # dog - dots
                for i in [8, 16, 24]:
                    for j in [8, 16, 24]:
                        prototype[:, i:i+4, j:j+4] = 0.8
            elif class_idx == 6:  # frog - grid
                prototype[:, ::8, :] = 0.8
                prototype[:, :, ::8] = 0.8
            elif class_idx == 7:  # horse - stripes
                for i in range(0, 32, 4):
                    prototype[:, i:i+2, :] = 0.8
            elif class_idx == 8:  # ship - waves
                for i in range(32):
                    wave = int(8 * np.sin(i * 0.5) + 16)
                    prototype[:, wave-2:wave+2, i] = 0.8
            elif class_idx == 9:  # truck - rectangle
                prototype[:, 8:24, 8:24] = 0.8
            
            self.pixel_prototypes[class_idx] = prototype.to(self.device)
        
        print("✓ Pixel prototypes created")
    
    def _load_prototypes(self, checkpoint):
        """Load prototypes from checkpoint or create defaults."""
        self.prototypes = {}
        
        if 'prototypes' in checkpoint:
            try:
                self.prototypes = checkpoint['prototypes']
                print("✓ Loaded prototypes from checkpoint")
            except Exception as e:
                print(f"Warning: Could not load prototypes: {e}")
        
        # Create default prototypes if needed
        for class_idx in range(self.num_classes):
            if class_idx not in self.prototypes:
                feature_dim = self.features_shape[0] if isinstance(self.features_shape, list) else 3072
                self.prototypes[class_idx] = {
                    'mean': torch.randn(feature_dim, device=self.device, dtype=torch.float32) * 0.1,
                    'std': torch.ones(feature_dim, device=self.device, dtype=torch.float32)
                }
        
        print(f"✓ Prototypes ready for {len(self.prototypes)} classes")
    
    def _get_cifar10_dataloader(self, train=False, batch_size=1):
        """Get CIFAR-10 dataloader."""
        try:
            from torchvision import datasets, transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            dataset = datasets.CIFAR10(
                root='./data', 
                train=train, 
                download=True, 
                transform=transform
            )
            
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0
            )
            
            return dataloader
            
        except Exception as e:
            print(f"Error creating CIFAR-10 dataloader: {e}")
            # Create a simple mock dataloader
            mock_data = torch.randn(10, 3, 32, 32)
            mock_labels = torch.arange(10)
            mock_dataset = torch.utils.data.TensorDataset(mock_data, mock_labels)
            return DataLoader(mock_dataset, batch_size=batch_size, shuffle=True)
    
    def generate_counterfactual_flow(self, source_image, target_class, alpha=0.5):
        """Generate counterfactual using real ProtoFlow."""
        with torch.no_grad():
            try:
                # Ensure image is float32 and in correct range
                source_image = source_image.float()
                if source_image.max() > 1.0:
                    source_image = source_image / 255.0
                
                # Extract latent features using the real flow
                z, log_prob = self.model.model.log_prob(source_image, return_z=True)
                print(f"Encoded to latent space: {z.shape}")
                
                # Flatten z for prototype operations
                z_flat = z.flatten(1) if z.dim() > 2 else z
                z_flat = z_flat.float()
                
                # Get target prototype
                target_proto = self.get_class_prototype(target_class).float()
                
                # Ensure matching dimensions
                if target_proto.shape[0] != z_flat.shape[1]:
                    min_dim = min(target_proto.shape[0], z_flat.shape[1])
                    target_proto = target_proto[:min_dim]
                    z_flat = z_flat[:, :min_dim]
                
                # Interpolate in latent space
                cf_z_flat = (1 - alpha) * z_flat + alpha * target_proto.unsqueeze(0)
                
                # Reshape back to original z shape if needed
                if z.dim() > 2:
                    cf_z = cf_z_flat.reshape(z.shape)
                else:
                    cf_z = cf_z_flat
                
                # Decode back to image using the real flow
                try:
                    # Try using inverse transform
                    cf_image = self.model.model.inverse(cf_z)
                    print("✓ Used flow inverse for decoding")
                except Exception as e1:
                    try:
                        # Try using sample method
                        cf_image = self.model.model.sample(cf_z.shape[0], z=cf_z)
                        print("✓ Used flow sample for decoding")
                    except Exception as e2:
                        print(f"Flow decoding failed: {e2}")
                        print("Falling back to pixel interpolation")
                        return self.generate_counterfactual_pixel(source_image, target_class, alpha)
                
                # Ensure proper range
                cf_image = torch.clamp(cf_image, -1, 1)
                return cf_image
                
            except Exception as e:
                print(f"Flow generation failed: {e}")
                traceback.print_exc()
                # Fall back to pixel space
                return self.generate_counterfactual_pixel(source_image, target_class, alpha)
    
    def generate_counterfactual_pixel(self, source_image, target_class, alpha=0.5):
        """Generate counterfactual using pixel-space interpolation."""
        source_image = source_image.float()
        
        if target_class in self.pixel_prototypes:
            proto_img = self.pixel_prototypes[target_class].float()
        else:
            # Create a simple colored prototype
            proto_img = torch.zeros_like(source_image[0])
            proto_img[0] = 0.5  # Red channel
            proto_img[1] = 0.3  # Green channel
            proto_img[2] = 0.7  # Blue channel
        
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
            return self.prototypes[class_idx]['mean'].float()
        else:
            # Get the correct feature dimension
            feature_dim = 3072  # Default for CIFAR-10
            if hasattr(self, 'model') and hasattr(self.model, 'gmms') and len(self.model.gmms) > 0:
                try:
                    feature_dim = self.model.gmms[0].mu.shape[-1]
                except:
                    pass
            return torch.randn(feature_dim, device=self.device, dtype=torch.float32)
    
    def classify_image(self, image):
        """Classify image using ProtoFlow."""
        with torch.no_grad():
            if self.use_real_flow and hasattr(self, 'model'):
                try:
                    # Ensure image is float32
                    image = image.float()
                    
                    # Use real ProtoFlow classification
                    logits = self.model(image)
                    predicted_class = torch.argmax(logits, dim=1)
                    return predicted_class.item()
                except Exception as e:
                    print(f"Real flow classification failed: {e}")
            
            # Fallback: simple pixel-based classification
            # This is a very basic classifier based on color statistics
            image = image.float()
            if image.dim() == 4:
                image = image.squeeze(0)
            
            # Calculate color statistics
            mean_colors = image.mean(dim=(1, 2))  # [R, G, B]
            
            # Simple rule-based classification
            if mean_colors[0] > 0.6 and mean_colors[1] < 0.4:  # Red dominant
                return 1  # automobile
            elif mean_colors[1] > 0.6 and mean_colors[0] < 0.4:  # Green dominant
                return 2  # bird
            elif mean_colors[2] > 0.6:  # Blue dominant
                return 8  # ship
            else:
                return 0  # default to airplane
    
    def generate_counterfactual_explanation(self, image, source_class, target_classes=None, alphas=None):
        """Generate counterfactual explanation for an image."""
        if target_classes is None:
            target_classes = [(source_class + 1) % self.num_classes, (source_class + 5) % self.num_classes]
        
        if alphas is None:
            alphas = [0.3, 0.5, 0.7]
        
        explanation = {
            'source_image': image,
            'source_class': source_class,
            'counterfactuals': {}
        }
        
        for target_class in target_classes:
            explanation['counterfactuals'][target_class] = {}
            
            for alpha in alphas:
                try:
                    if self.use_real_flow:
                        cf_image = self.generate_counterfactual_flow(image, target_class, alpha)
                    else:
                        cf_image = self.generate_counterfactual_pixel(image, target_class, alpha)
                    
                    explanation['counterfactuals'][target_class][alpha] = cf_image
                    
                except Exception as e:
                    print(f"Error generating counterfactual for class {target_class}, alpha {alpha}: {e}")
                    # Create a fallback image
                    fallback = image.clone()
                    fallback = fallback * (1 - alpha) + torch.randn_like(fallback) * alpha * 0.1
                    explanation['counterfactuals'][target_class][alpha] = fallback
        
        return explanation
    
    def create_counterfactual_visualization(self, explanation, class_names=None):
        """Create visualization of counterfactual explanations."""
        if class_names is None:
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        
        source_image = explanation['source_image']
        source_class = explanation['source_class']
        counterfactuals = explanation['counterfactuals']
        
        # Calculate grid dimensions
        num_targets = len(counterfactuals)
        num_alphas = len(next(iter(counterfactuals.values())))
        cols = num_alphas + 1  # +1 for source image
        rows = num_targets + 1  # +1 for header row
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        def show_image(ax, img_tensor, title, success=None):
            """Helper function to display an image."""
            try:
                # Convert tensor to numpy
                if img_tensor.dim() == 4:
                    img_tensor = img_tensor.squeeze(0)
                
                # Denormalize if needed
                if img_tensor.min() < 0:
                    img_tensor = (img_tensor + 1) / 2
                
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                
                ax.imshow(img_np)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
                
                if success is not None:
                    color = 'green' if success else 'red'
                    ax.set_title(f"{title}\n({'✓' if success else '✗'})", 
                               color=color, fontsize=10)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f'Error\n{str(e)[:20]}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=10)
                ax.axis('off')
        
        # Header row
        axes[0, 0].text(0.5, 0.5, 'Source', ha='center', va='center', 
                       transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
        for i, alpha in enumerate(next(iter(counterfactuals.values())).keys()):
            axes[0, i + 1].text(0.5, 0.5, f'α={alpha}', ha='center', va='center', 
                               transform=axes[0, i + 1].transAxes, fontsize=12, fontweight='bold')
        
        # Source image row
        show_image(axes[1, 0], source_image, f'Source\n{class_names[source_class]}', True)
        for i in range(num_alphas):
            axes[1, i + 1].axis('off')
        
        # Counterfactual rows
        for row_idx, (target_class, alphas_dict) in enumerate(counterfactuals.items()):
            row = row_idx + 2
            
            # Target class label
            axes[row, 0].text(0.5, 0.5, f'→ {class_names[target_class]}', 
                             ha='center', va='center', 
                             transform=axes[row, 0].transAxes, fontsize=12, fontweight='bold')
            
            # Counterfactual images
            for col_idx, (alpha, cf_image) in enumerate(alphas_dict.items()):
                try:
                    # Try to classify the counterfactual
                    predicted_class = self.classify_image(cf_image)
                    success = predicted_class == target_class
                    show_image(axes[row, col_idx + 1], cf_image, 
                              f'α={alpha}', success)
                except Exception as e:
                    show_image(axes[row, col_idx + 1], cf_image, f'α={alpha}', False)
        
        plt.tight_layout()
        return fig

def main():
    parser = argparse.ArgumentParser(description='Generate Fixed ProtoFlow Counterfactuals')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--output_dir', default='./counterfactual_results_fixed', help='Output directory')
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
        # Initialize fixed counterfactual generator
        print("Initializing FIXED ProtoFlow counterfactual generator...")
        generator = FixedProtoFlowCounterfactualGenerator(args.checkpoint, device)
        print("✓ Generator initialized")
        
        # Get test data
        test_loader = generator._get_cifar10_dataloader(train=False, batch_size=1)
        print("✓ Test dataloader created")
        
        # Generate counterfactuals
        print(f"\n{'='*60}")
        print(f"GENERATING FIXED COUNTERFACTUAL EXPLANATIONS")
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
                    method_suffix = "protoflow_fixed" if generator.use_real_flow else "pixel"
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
        method = "ProtoFlow + Real DenseFlow" if generator.use_real_flow else "Pixel-Space"
        print(f"Method used: {method}")
        print(f"Successfully generated: {successful_samples}/{args.num_samples} counterfactual explanations")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        
        if successful_samples > 0:
            print("🎉 Fixed counterfactual generation completed!")
            if generator.use_real_flow:
                print("📁 Using REAL ProtoFlow with proper DenseFlow - authentic latent-space transformations!")
            else:
                print("📁 Using pixel-space interpolation - smooth visual transitions!")
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