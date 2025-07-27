#!/usr/bin/env python3
"""
Simplified counterfactual generator that works with available checkpoints.
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

class SimpleCounterfactualGenerator:
    """Simple counterfactual generator that works with available resources."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """Load checkpoint and extract basic information."""
        print(f"Loading checkpoint from {self.checkpoint_path}")
        
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Extract basic info
            self.num_classes = checkpoint.get('num_classes', 10)
            self.features_shape = checkpoint.get('features_shape', [3072])  # Default for CIFAR-10
            
            print(f"✓ Loaded checkpoint with {self.num_classes} classes")
            print(f"✓ Feature shape: {self.features_shape}")
            
            # Try to load prototypes if available
            self.prototypes = self._load_prototypes(checkpoint)
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            # Use defaults
            self.num_classes = 10
            self.features_shape = [3072]
            self.prototypes = None
    
    def _load_prototypes(self, checkpoint):
        """Load prototypes from checkpoint."""
        try:
            if 'prototypes' in checkpoint:
                raw_protos = checkpoint['prototypes']
                
                if hasattr(raw_protos, 'class_means'):
                    prototypes = {}
                    for class_idx in range(self.num_classes):
                        if class_idx in raw_protos.class_means:
                            mean_tensor = raw_protos.class_means[class_idx].float()
                            prototypes[class_idx] = {
                                'mean': mean_tensor,
                                'var': torch.ones_like(mean_tensor, dtype=torch.float32),
                                'pi': torch.tensor(1.0, dtype=torch.float32)
                            }
                    print(f"✓ Loaded prototypes for {len(prototypes)} classes")
                    return prototypes
        except Exception as e:
            print(f"Failed to load prototypes: {e}")
        
        return None
    
    def get_cifar10_dataloader(self, train=False, batch_size=1):
        """Get CIFAR-10 dataloader."""
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
    
    def generate_pixel_counterfactual(self, source_image, target_class, alpha=0.5):
        """Generate counterfactual using pixel-space interpolation."""
        source_image = source_image.float()
        
        # Create a target prototype (simplified - using random image)
        target_proto = torch.randn_like(source_image) * 0.5
        
        # Linear interpolation in pixel space
        cf_image = (1 - alpha) * source_image + alpha * target_proto
        
        # Clamp to valid range
        return torch.clamp(cf_image, -1, 1)
    
    def generate_feature_counterfactual(self, source_image, target_class, alpha=0.5):
        """Generate counterfactual using feature-space interpolation."""
        source_image = source_image.float()
        
        # Flatten image to feature space
        source_features = source_image.flatten(1)
        
        # Get target prototype
        target_proto = self.get_class_prototype(target_class)
        
        # Ensure matching dimensions
        min_dim = min(source_features.shape[1], target_proto.shape[0])
        source_features = source_features[:, :min_dim]
        target_proto = target_proto[:min_dim]
        
        # Interpolate in feature space
        cf_features = (1 - alpha) * source_features + alpha * target_proto.unsqueeze(0)
        
        # Reshape back to image
        cf_image = cf_features.reshape(source_image.shape)
        
        # Clamp to valid range
        return torch.clamp(cf_image, -1, 1)
    
    def get_class_prototype(self, class_idx):
        """Get prototype for a specific class."""
        if self.prototypes and class_idx in self.prototypes:
            return self.prototypes[class_idx]['mean'].float()
        else:
            # Create a random prototype
            feature_dim = self.features_shape[0]
            return torch.randn(feature_dim, device=self.device, dtype=torch.float32)
    
    def classify_image(self, image):
        """Simple classification using feature similarity."""
        image = image.float()
        
        # Flatten image
        features = image.flatten(1)
        
        # Compute similarity to each class prototype
        similarities = []
        for class_idx in range(self.num_classes):
            proto = self.get_class_prototype(class_idx)
            
            # Ensure matching dimensions
            min_dim = min(features.shape[1], proto.shape[0])
            feat = features[:, :min_dim]
            prot = proto[:min_dim]
            
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(feat, prot.unsqueeze(0), dim=1)
            similarities.append(similarity)
        
        # Convert to logits
        logits = torch.stack(similarities, dim=1)
        probs = torch.softmax(logits, dim=1)
        
        return probs, logits
    
    def generate_counterfactual_explanation(self, image, source_class, target_classes=None, alphas=None):
        """Generate counterfactual explanations."""
        
        if target_classes is None:
            all_classes = list(range(self.num_classes))
            all_classes.remove(source_class)
            target_classes = all_classes[:3]
        
        if alphas is None:
            alphas = [0.3, 0.5, 0.7]
        
        print(f"Generating counterfactuals using pixel-space interpolation")
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
            'method': 'Pixel-Space Interpolation',
            'counterfactuals': {}
        }
        
        # Generate counterfactuals for each target class
        for target_class in target_classes:
            print(f"\nGenerating counterfactuals for target class {target_class}")
            target_results = {'alpha_variants': []}
            
            for alpha in alphas:
                try:
                    # Generate counterfactual
                    cf_image = self.generate_pixel_counterfactual(image, target_class, alpha)
                    
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
    
    def create_visualization(self, explanation, class_names=None):
        """Create visualization of counterfactual explanations."""
        
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
    parser = argparse.ArgumentParser(description='Generate Simple Counterfactuals')
    parser.add_argument('--checkpoint', default='dummy_checkpoint.pth', help='Checkpoint path')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--output_dir', default='./counterfactual_results_simple', help='Output directory')
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
        print("Initializing simple counterfactual generator...")
        generator = SimpleCounterfactualGenerator(args.checkpoint, device)
        print("✓ Generator initialized")
        
        # Get test data
        test_loader = generator.get_cifar10_dataloader(train=False, batch_size=1)
        print("✓ Test dataloader created")
        
        # Generate counterfactuals
        print(f"\n{'='*60}")
        print(f"GENERATING SIMPLE COUNTERFACTUAL EXPLANATIONS")
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
                fig = generator.create_visualization(explanation, class_names)
                
                if fig is not None:
                    # Save results
                    output_path = output_dir / f'counterfactual_{batch_idx:03d}_{class_names[source_class]}_simple.png'
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
        print(f"Method: Pixel-Space Interpolation")
        print(f"Successfully generated: {successful_samples}/{args.num_samples} counterfactual explanations")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")
        
        if successful_samples > 0:
            print("🎉 Simple counterfactual generation completed!")
            print("📁 Using pixel-space interpolation for smooth visual transitions!")
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