"""
GNU GPL v2.0
Copyright (c) 2024 Zachariah Carmichael, Timothy Redgrave, Daniel Gonzalez Cedre
ProtoFlow Project

Counterfactual extensions for ProtoFlow.
Adds bidirectional explanation capabilities to ProtoFlowGMM.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Import existing ProtoFlow components
from .proto import ProtoFlowGMM
from .gmm import GaussianMixture


class ClassConditionalPrototypes:
    """Enhanced prototype management for counterfactual generation."""
    
    def __init__(self, num_classes: int, latent_dim: int, n_prototypes_per_class: int = 5):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.n_prototypes_per_class = n_prototypes_per_class
        
        # Class-conditional prototype statistics
        self.class_means = {}  # μ^(c)
        self.class_covariances = {}  # Σ^(c) 
        self.class_latent_codes = {}  # {z_i^(c)}
        
    def collect_latent_codes(self, latent_codes: torch.Tensor, labels: torch.Tensor):
        """Collect latent codes for each class during training/inference."""
        latent_codes = latent_codes.detach().cpu()
        labels = labels.detach().cpu()
        
        for class_idx in range(self.num_classes):
            class_mask = (labels == class_idx)
            if class_mask.sum() > 0:
                class_latents = latent_codes[class_mask]
                if class_idx not in self.class_latent_codes:
                    self.class_latent_codes[class_idx] = []
                self.class_latent_codes[class_idx].append(class_latents)
    
    def fit_class_distributions(self, fit_covariance: bool = True):
        """Fit Gaussian distributions for each class prototype."""
        for class_idx in range(self.num_classes):
            if class_idx in self.class_latent_codes and self.class_latent_codes[class_idx]:
                # Concatenate all latent codes for this class
                class_latents = torch.cat(self.class_latent_codes[class_idx], dim=0)
                
                # Flatten if needed (ProtoFlow uses flattened latents)
                if class_latents.dim() > 2:
                    class_latents = class_latents.flatten(1)
                
                # Compute mean
                self.class_means[class_idx] = class_latents.mean(dim=0)
                
                if fit_covariance and len(class_latents) > 1:
                    # Compute covariance matrix
                    centered = class_latents - self.class_means[class_idx]
                    cov = torch.mm(centered.t(), centered) / (len(class_latents) - 1)
                    self.class_covariances[class_idx] = cov
                else:
                    # Use spherical covariance
                    var = class_latents.var(dim=0).mean()
                    self.class_covariances[class_idx] = torch.eye(self.latent_dim) * var
                    
                print(f"✓ Class {class_idx}: {len(class_latents)} samples, "
                      f"mean shape {self.class_means[class_idx].shape}")


class ProtoFlowInversion:
    """High-quality inversion for ProtoFlow using its DenseFlow component."""
    
    def __init__(self, protoflow_model: ProtoFlowGMM):
        # Unwrap from DistributedDataParallel if needed
        if hasattr(protoflow_model, 'module'):
            self.protoflow = protoflow_model.module
        else:
            self.protoflow = protoflow_model
        self.flow_model = self.protoflow.model  # DenseFlow component
        
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space using ProtoFlow's DenseFlow.
        Returns the latent codes z.
        """
        with torch.no_grad():
            # Use DenseFlow's log_prob with return_z=True to get latent codes
            z, log_prob = self.flow_model.log_prob(image, return_z=True)
            return z
    
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes back to images using ProtoFlow's DenseFlow.
        This uses the inverse transformation of the flow.
        """
        with torch.no_grad():
            # Use DenseFlow's sample method to generate images from latent codes
            # Note: You might need to adjust this based on exact DenseFlow API
            try:
                # Try direct sampling first
                images = self.flow_model.sample(latent)
            except:
                # Alternative: use inverse if sample doesn't work
                images = self.flow_model.inverse(latent)
            return images


class CounterfactualGenerator:
    """Main counterfactual generation module for ProtoFlow."""
    
    def __init__(self, protoflow_model: ProtoFlowGMM, prototypes: ClassConditionalPrototypes):
        # Unwrap from DistributedDataParallel if needed
        if hasattr(protoflow_model, 'module'):
            self.protoflow = protoflow_model.module
        else:
            self.protoflow = protoflow_model
        self.prototypes = prototypes
        self.inversion = ProtoFlowInversion(protoflow_model)
        
    def generate_counterfactual_analytic(self, 
                                       source_latent: torch.Tensor,
                                       source_class: int,
                                       target_class: int) -> Tuple[torch.Tensor, float]:
        """Generate counterfactual using analytic class direction."""
        device = source_latent.device
        
        # Ensure latent is flattened for consistency with ProtoFlow
        if source_latent.dim() > 2:
            source_latent_flat = source_latent.flatten(1)
        else:
            source_latent_flat = source_latent
        
        # Get class means
        source_mean = self.prototypes.class_means[source_class].to(device)
        target_mean = self.prototypes.class_means[target_class].to(device)
        
        # Compute direction vector
        direction = target_mean - source_mean
        direction_norm = torch.norm(direction)
        
        if direction_norm > 1e-6:
            direction_unit = direction / direction_norm
            
            # Binary search for minimal α that crosses decision boundary
            alpha = self._find_minimal_alpha(source_latent, direction_unit, target_class)
        else:
            # If no direction (same class means), use small random perturbation
            direction_unit = torch.randn_like(direction) * 0.1
            alpha = 1.0
        
        # Generate counterfactual in flattened space
        counterfactual_latent_flat = source_latent_flat + alpha * direction_unit
        
        # Reshape back to original latent shape
        counterfactual_latent = counterfactual_latent_flat.reshape(source_latent.shape)
        
        return counterfactual_latent, alpha
    
    def _find_minimal_alpha(self, source_latent: torch.Tensor, 
                           direction: torch.Tensor, target_class: int, 
                           max_alpha: float = 5.0) -> float:
        """Binary search for minimal α to cross decision boundary."""
        
        def classify_at_alpha(alpha):
            """Helper function to classify at given alpha."""
            # Work with flattened latent for direction computation
            if source_latent.dim() > 2:
                source_latent_flat = source_latent.flatten(1)
            else:
                source_latent_flat = source_latent
                
            modified_latent_flat = source_latent_flat + alpha * direction
            modified_latent = modified_latent_flat.reshape(source_latent.shape)
            
            try:
                with torch.no_grad():
                    # Generate image from modified latent
                    generated_image = self.inversion.decode_latent(modified_latent)
                    
                    # Classify using ProtoFlow
                    logits = self.protoflow(generated_image, flow_grad=False)
                    pred_class = logits.argmax(dim=1).item()
                return pred_class
            except Exception as e:
                print(f"Warning: Classification failed at alpha {alpha}: {e}")
                return -1  # Invalid classification
        
        # Start with binary search
        alpha_low, alpha_high = 0.0, max_alpha
        
        # Check if target is reachable
        if classify_at_alpha(alpha_high) != target_class:
            # Expand search range
            for mult in [2, 4, 8]:
                alpha_high = max_alpha * mult
                if classify_at_alpha(alpha_high) == target_class:
                    break
            else:
                # If still not reachable, return max alpha
                print(f"Warning: Target class {target_class} not reachable, using max alpha")
                return alpha_high
        
        # Binary search for minimal alpha
        for iteration in range(15):  # 15 iterations should be sufficient
            alpha_mid = (alpha_low + alpha_high) / 2
            if classify_at_alpha(alpha_mid) == target_class:
                alpha_high = alpha_mid
            else:
                alpha_low = alpha_mid
                
            # Early stopping if precision is good enough
            if abs(alpha_high - alpha_low) < 0.01:
                break
                
        return alpha_high


class ProtoFlowCounterfactual:
    """Main integration class for ProtoFlow counterfactuals."""
    
    def __init__(self, protoflow_model: ProtoFlowGMM, features_shape: List[int]):
        # Unwrap from DistributedDataParallel if needed
        if hasattr(protoflow_model, 'module'):
            self.protoflow = protoflow_model.module
        else:
            self.protoflow = protoflow_model
            
        self.features_shape = features_shape
        self.latent_dim = math.prod(features_shape)  # Flattened latent dimension
        self.num_classes = len(self.protoflow.gmms)
        
        # Initialize components
        self.prototypes = ClassConditionalPrototypes(self.num_classes, self.latent_dim)
        self.generator = CounterfactualGenerator(protoflow_model, self.prototypes)
        self.inversion = ProtoFlowInversion(protoflow_model)
        
    def fit_prototype_distributions(self, dataloader, max_batches: int = None):
        """Fit class-conditional prototype distributions from data."""
        print(f"\n{'='*60}")
        print("FITTING CLASS-CONDITIONAL PROTOTYPE DISTRIBUTIONS")
        print(f"{'='*60}")
        
        self.protoflow.eval()
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                    
                images = images.cuda() if torch.cuda.is_available() else images
                labels = labels.cuda() if torch.cuda.is_available() else labels
                
                try:
                    # Get latent codes using ProtoFlow's forward pass with ret_z=True
                    _, z = self.protoflow(images, flow_grad=False, ret_z=True)
                    
                    # Collect for prototype fitting
                    self.prototypes.collect_latent_codes(z, labels)
                    
                    if batch_idx % 50 == 0:
                        print(f"Processed {batch_idx} batches...")
                        
                except Exception as e:
                    print(f"Warning: Error processing batch {batch_idx}: {e}")
                    continue
        
        # Fit distributions
        print("Fitting class distributions...")
        self.prototypes.fit_class_distributions()
        print("✓ Prototype distributions fitted!")
        
    def generate_explanation(self, 
                           image: torch.Tensor, 
                           source_class: int,
                           target_classes: List[int] = None,
                           save_path: str = None) -> Dict:
        """Generate complete counterfactual explanation."""
        
        if target_classes is None:
            target_classes = [i for i in range(self.num_classes) if i != source_class]
        
        # 1. Get source latent representation
        source_latent = self.inversion.encode_image(image)
        
        # 2. Generate counterfactuals for each target class
        counterfactuals = {}
        counterfactual_images = {}
        alphas = {}
        evaluations = {}
        
        for target_class in target_classes:
            try:
                # Generate counterfactual latent
                cf_latent, alpha = self.generator.generate_counterfactual_analytic(
                    source_latent, source_class, target_class
                )
                
                # Generate counterfactual image
                cf_image = self.inversion.decode_latent(cf_latent)
                
                # Evaluate the counterfactual
                evaluation = self._evaluate_counterfactual(
                    image, cf_image, target_class
                )
                
                counterfactuals[target_class] = cf_latent
                counterfactual_images[target_class] = cf_image
                alphas[target_class] = alpha
                evaluations[target_class] = evaluation
                
            except Exception as e:
                print(f"Warning: Failed to generate counterfactual for class {target_class}: {e}")
                continue
        
        # 3. Create visualization
        gallery = self._create_visualization_gallery(
            image, source_class, counterfactual_images, target_classes
        )
        
        # 4. Save if requested
        if save_path:
            gallery.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Gallery saved to {save_path}")
        
        return {
            'source_latent': source_latent,
            'counterfactuals': counterfactuals,
            'counterfactual_images': counterfactual_images,
            'alphas': alphas,
            'evaluations': evaluations,
            'gallery': gallery
        }
    
    def _evaluate_counterfactual(self, 
                               original_image: torch.Tensor,
                               counterfactual_image: torch.Tensor,
                               target_class: int) -> Dict[str, float]:
        """Evaluate counterfactual quality."""
        metrics = {}
        
        # 1. Classification confidence
        with torch.no_grad():
            cf_logits = self.protoflow(counterfactual_image, flow_grad=False)
            cf_probs = torch.softmax(cf_logits, dim=1)
            target_confidence = cf_probs[0, target_class].item()
            
        metrics['target_confidence'] = target_confidence
        metrics['prediction_success'] = float(cf_logits.argmax().item() == target_class)
        
        # 2. L2 distance
        l2_distance = torch.norm(original_image - counterfactual_image).item()
        metrics['l2_distance'] = l2_distance
        
        # 3. Pixel change ratio
        metrics['pixel_change_ratio'] = torch.mean((original_image != counterfactual_image).float()).item()
        
        return metrics
    
    def _create_visualization_gallery(self, 
                                    original_image: torch.Tensor,
                                    source_class: int,
                                    counterfactual_images: Dict[int, torch.Tensor],
                                    target_classes: List[int]) -> plt.Figure:
        """Create visualization gallery showing original and counterfactuals."""
        
        n_images = 1 + len(counterfactual_images)
        fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
        
        if n_images == 1:
            axes = [axes]
        elif n_images == 2:
            axes = [axes[0], axes[1]]
        
        # Helper function to convert tensor to displayable image
        def tensor_to_image(tensor):
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            img = tensor.detach().cpu().numpy()
            if img.shape[0] <= 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            if img.shape[2] == 1:  # Grayscale
                img = img.squeeze(2)
            return img
        
        # Show original image
        axes[0].imshow(tensor_to_image(original_image))
        axes[0].set_title(f'Original\nClass {source_class}')
        axes[0].axis('off')
        
        # Show counterfactuals
        cf_idx = 1
        for target_class in target_classes:
            if target_class in counterfactual_images and cf_idx < len(axes):
                cf_image = counterfactual_images[target_class]
                axes[cf_idx].imshow(tensor_to_image(cf_image))
                axes[cf_idx].set_title(f'→ Class {target_class}')
                axes[cf_idx].axis('off')
                cf_idx += 1
        
        plt.tight_layout()
        return fig


# Helper function to integrate with existing training
def add_counterfactual_capability(model: ProtoFlowGMM, 
                                dataloader,
                                features_shape: List[int],
                                save_path: str = None) -> ProtoFlowCounterfactual:
    """Add counterfactual capability to existing ProtoFlow model."""
    
    # Unwrap from DistributedDataParallel if needed
    actual_model = model.module if hasattr(model, 'module') else model
    
    enhanced_model = ProtoFlowCounterfactual(
        protoflow_model=actual_model,  # Pass the unwrapped model
        features_shape=features_shape
    )
    
    # Fit prototype distributions
    enhanced_model.fit_prototype_distributions(dataloader, max_batches=200)
    
    # Save if path provided
    if save_path:
        torch.save({
            'model_state_dict': actual_model.state_dict(),  # Use unwrapped model
            'prototypes': enhanced_model.prototypes,
            'features_shape': features_shape,
            'num_classes': enhanced_model.num_classes,
            'latent_dim': enhanced_model.latent_dim,
        }, save_path)
        print(f"Enhanced model saved to {save_path}")
    
    return enhanced_model


# Dataset-specific configurations
def get_dataset_config(dataset_name: str) -> Dict:
    """Get dataset-specific configuration for counterfactuals."""
    configs = {
        'cifar10': {
            'num_classes': 10,
            'features_shape': [3, 32, 32],  # Adjust based on your flow architecture
            'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        },
        'cifar100': {
            'num_classes': 100,
            'features_shape': [3, 32, 32],
            'class_names': [f'class_{i}' for i in range(100)]
        },
        'mnist': {
            'num_classes': 10,
            'features_shape': [1, 28, 28],
            'class_names': [str(i) for i in range(10)]
        },
        'imagenet': {
            'num_classes': 1000,
            'features_shape': [3, 224, 224],
            'class_names': [f'class_{i}' for i in range(1000)]
        }
    }
    return configs.get(dataset_name, {
        'num_classes': 10,
        'features_shape': [3, 32, 32],
        'class_names': [f'class_{i}' for i in range(10)]
    })