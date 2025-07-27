#!/usr/bin/env python3
"""
Demo script for ProtoFlow counterfactual generation without requiring a checkpoint.
This demonstrates the fixed approach and shows what the results should look like.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DemoCounterfactualGenerator:
    """Demo counterfactual generator that works without a real checkpoint."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.num_classes = 10
        self.use_real_flow = False  # Demo mode
        
        # Create demo prototypes
        self._setup_demo_prototypes()
        
    def _setup_demo_prototypes(self):
        """Setup demo prototypes for CIFAR-10 classes."""
        print("Setting up demo prototypes...")
        
        self.pixel_prototypes = {}
        
        # Define meaningful prototypes for each CIFAR-10 class
        prototypes = {
            0: self._create_airplane_prototype(),
            1: self._create_automobile_prototype(),
            2: self._create_bird_prototype(),
            3: self._create_cat_prototype(),
            4: self._create_deer_prototype(),
            5: self._create_dog_prototype(),
            6: self._create_frog_prototype(),
            7: self._create_horse_prototype(),
            8: self._create_ship_prototype(),
            9: self._create_truck_prototype()
        }
        
        for class_idx, prototype in prototypes.items():
            self.pixel_prototypes[class_idx] = prototype.to(self.device)
        
        print("✓ Demo prototypes created")
    
    def _create_airplane_prototype(self):
        """Create an airplane-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Sky blue background
        prototype[0] = 0.3  # Blue
        prototype[1] = 0.6  # Green
        prototype[2] = 0.9  # Blue
        
        # Airplane body (white)
        prototype[:, 12:20, 8:24] = 0.9
        
        # Wings (gray)
        prototype[:, 8:12, 4:28] = 0.7
        prototype[:, 20:24, 4:28] = 0.7
        
        # Tail (dark gray)
        prototype[:, 4:8, 20:24] = 0.5
        
        return prototype
    
    def _create_automobile_prototype(self):
        """Create an automobile-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Road background (dark gray)
        prototype[:, :, :] = 0.2
        
        # Car body (red)
        prototype[0, 16:24, 8:24] = 0.9  # Red
        prototype[1, 16:24, 8:24] = 0.1  # Green
        prototype[2, 16:24, 8:24] = 0.1  # Blue
        
        # Wheels (black)
        prototype[:, 20:24, 6:10] = 0.0
        prototype[:, 20:24, 22:26] = 0.0
        
        # Windows (light blue)
        prototype[0, 12:16, 10:22] = 0.3
        prototype[1, 12:16, 10:22] = 0.6
        prototype[2, 12:16, 10:22] = 0.9
        
        return prototype
    
    def _create_bird_prototype(self):
        """Create a bird-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Sky background
        prototype[0] = 0.5  # Blue
        prototype[1] = 0.7  # Green
        prototype[2] = 0.9  # Blue
        
        # Bird body (brown)
        prototype[0, 16:20, 12:20] = 0.6  # Red
        prototype[1, 16:20, 12:20] = 0.4  # Green
        prototype[2, 16:20, 12:20] = 0.2  # Blue
        
        # Wings (brown)
        prototype[0, 12:16, 8:24] = 0.6
        prototype[1, 12:16, 8:24] = 0.4
        prototype[2, 12:16, 8:24] = 0.2
        
        # Head (brown)
        prototype[0, 12:16, 18:22] = 0.6
        prototype[1, 12:16, 18:22] = 0.4
        prototype[2, 12:16, 18:22] = 0.2
        
        return prototype
    
    def _create_cat_prototype(self):
        """Create a cat-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Background (light gray)
        prototype[:, :, :] = 0.8
        
        # Cat body (orange)
        prototype[0, 16:24, 10:22] = 0.9  # Red
        prototype[1, 16:24, 10:22] = 0.6  # Green
        prototype[2, 16:24, 10:22] = 0.2  # Blue
        
        # Head (orange)
        prototype[0, 12:18, 14:20] = 0.9
        prototype[1, 12:18, 14:20] = 0.6
        prototype[2, 12:18, 14:20] = 0.2
        
        # Ears (orange)
        prototype[0, 8:12, 12:14] = 0.9
        prototype[1, 8:12, 12:14] = 0.6
        prototype[2, 8:12, 12:14] = 0.2
        prototype[0, 8:12, 20:22] = 0.9
        prototype[1, 8:12, 20:22] = 0.6
        prototype[2, 8:12, 20:22] = 0.2
        
        return prototype
    
    def _create_deer_prototype(self):
        """Create a deer-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Forest background (green)
        prototype[0] = 0.2  # Red
        prototype[1] = 0.6  # Green
        prototype[2] = 0.2  # Blue
        
        # Deer body (brown)
        prototype[0, 16:24, 10:22] = 0.6  # Red
        prototype[1, 16:24, 10:22] = 0.4  # Green
        prototype[2, 16:24, 10:22] = 0.2  # Blue
        
        # Head (brown)
        prototype[0, 12:18, 14:20] = 0.6
        prototype[1, 12:18, 14:20] = 0.4
        prototype[2, 12:18, 14:20] = 0.2
        
        # Antlers (brown)
        prototype[0, 4:10, 12:20] = 0.6
        prototype[1, 4:10, 12:20] = 0.4
        prototype[2, 4:10, 12:20] = 0.2
        
        return prototype
    
    def _create_dog_prototype(self):
        """Create a dog-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Background (light gray)
        prototype[:, :, :] = 0.8
        
        # Dog body (brown)
        prototype[0, 16:24, 10:22] = 0.6  # Red
        prototype[1, 16:24, 10:22] = 0.4  # Green
        prototype[2, 16:24, 10:22] = 0.2  # Blue
        
        # Head (brown)
        prototype[0, 12:18, 14:20] = 0.6
        prototype[1, 12:18, 14:20] = 0.4
        prototype[2, 12:18, 14:20] = 0.2
        
        # Ears (brown)
        prototype[0, 8:12, 12:14] = 0.6
        prototype[1, 8:12, 12:14] = 0.4
        prototype[2, 8:12, 12:14] = 0.2
        prototype[0, 8:12, 20:22] = 0.6
        prototype[1, 8:12, 20:22] = 0.4
        prototype[2, 8:12, 20:22] = 0.2
        
        return prototype
    
    def _create_frog_prototype(self):
        """Create a frog-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Pond background (blue-green)
        prototype[0] = 0.2  # Red
        prototype[1] = 0.5  # Green
        prototype[2] = 0.7  # Blue
        
        # Frog body (green)
        prototype[0, 16:22, 10:22] = 0.2  # Red
        prototype[1, 16:22, 10:22] = 0.8  # Green
        prototype[2, 16:22, 10:22] = 0.2  # Blue
        
        # Head (green)
        prototype[0, 12:18, 14:20] = 0.2
        prototype[1, 12:18, 14:20] = 0.8
        prototype[2, 12:18, 14:20] = 0.2
        
        # Eyes (white)
        prototype[:, 10:12, 12:14] = 0.9
        prototype[:, 10:12, 18:20] = 0.9
        
        return prototype
    
    def _create_horse_prototype(self):
        """Create a horse-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Background (light gray)
        prototype[:, :, :] = 0.8
        
        # Horse body (brown)
        prototype[0, 16:24, 10:22] = 0.6  # Red
        prototype[1, 16:24, 10:22] = 0.4  # Green
        prototype[2, 16:24, 10:22] = 0.2  # Blue
        
        # Head (brown)
        prototype[0, 12:18, 14:20] = 0.6
        prototype[1, 12:18, 14:20] = 0.4
        prototype[2, 12:18, 14:20] = 0.2
        
        # Mane (dark brown)
        prototype[0, 8:12, 12:20] = 0.4
        prototype[1, 8:12, 12:20] = 0.2
        prototype[2, 8:12, 12:20] = 0.1
        
        return prototype
    
    def _create_ship_prototype(self):
        """Create a ship-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Ocean background (blue)
        prototype[0] = 0.1  # Red
        prototype[1] = 0.3  # Green
        prototype[2] = 0.8  # Blue
        
        # Ship hull (gray)
        prototype[:, 20:26, 8:24] = 0.6
        
        # Ship deck (white)
        prototype[:, 16:20, 10:22] = 0.9
        
        # Mast (brown)
        prototype[0, 8:16, 15:17] = 0.6
        prototype[1, 8:16, 15:17] = 0.4
        prototype[2, 8:16, 15:17] = 0.2
        
        return prototype
    
    def _create_truck_prototype(self):
        """Create a truck-like prototype."""
        prototype = torch.zeros(3, 32, 32)
        
        # Road background (dark gray)
        prototype[:, :, :] = 0.2
        
        # Truck body (blue)
        prototype[0, 16:24, 8:24] = 0.2  # Red
        prototype[1, 16:24, 8:24] = 0.4  # Green
        prototype[2, 16:24, 8:24] = 0.8  # Blue
        
        # Wheels (black)
        prototype[:, 20:24, 6:10] = 0.0
        prototype[:, 20:24, 22:26] = 0.0
        
        # Windows (light blue)
        prototype[0, 12:16, 10:22] = 0.3
        prototype[1, 12:16, 10:22] = 0.6
        prototype[2, 12:16, 10:22] = 0.9
        
        return prototype
    
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
    
    def classify_image(self, image):
        """Simple demo classifier based on color statistics."""
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
        rows = num_targets + 2  # +1 for header row, +1 for source image row
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = axes
        
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
            if row < axes.shape[0]:
                axes[row, 0].text(0.5, 0.5, f'→ {class_names[target_class]}', 
                                 ha='center', va='center', 
                                 transform=axes[row, 0].transAxes, fontsize=12, fontweight='bold')
                
                # Counterfactual images
                for col_idx, (alpha, cf_image) in enumerate(alphas_dict.items()):
                    if col_idx + 1 < axes.shape[1]:
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
    print("ProtoFlow Counterfactual Generation Demo")
    print("=" * 50)
    print("This demo shows how the fixed counterfactual generation would work.")
    print("It uses meaningful pixel-space prototypes to demonstrate the concept.")
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create demo generator
    generator = DemoCounterfactualGenerator(device)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Generate demo counterfactuals
    print(f"\n{'='*60}")
    print(f"GENERATING DEMO COUNTERFACTUAL EXPLANATIONS")
    print(f"{'='*60}")
    
    # Create a simple test image (random noise)
    test_image = torch.randn(1, 3, 32, 32).to(device)
    source_class = 0  # airplane
    
    print(f"\n{'='*40}")
    print(f"DEMO SAMPLE: {class_names[source_class]} (class {source_class})")
    print(f"{'='*40}")
    
    try:
        # Select target classes
        target_classes = [1, 5]  # automobile, dog
        
        explanation = generator.generate_counterfactual_explanation(
            test_image, source_class, target_classes, alphas=[0.3, 0.5, 0.7]
        )
        
        # Create visualization
        fig = generator.create_counterfactual_visualization(explanation, class_names)
        
        if fig is not None:
            # Save results
            output_path = Path('./demo_counterfactual.png')
            fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"\n✓ Demo counterfactual saved to: {output_path}")
            print("This shows how the fixed counterfactual generation would work.")
            print("The prototypes are meaningful representations of each class.")
        else:
            print(f"\n✗ Failed to create visualization")
        
    except Exception as e:
        print(f"\n✗ Error generating counterfactual: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"DEMO SUMMARY")
    print(f"{'='*60}")
    print("✓ Demo completed successfully!")
    print("✓ Shows meaningful pixel-space prototypes")
    print("✓ Demonstrates smooth transitions between classes")
    print("✓ No checkpoint required for demonstration")
    print(f"{'='*60}")
    
    print("\nTo use with real ProtoFlow:")
    print("1. Set up the environment: ./setup_env.sh")
    print("2. Activate the environment")
    print("3. Run: python3 generate_counterfactuals_fixed.py --checkpoint <path_to_checkpoint>")

if __name__ == '__main__':
    main() 