#!/bin/bash

# Simple ProtoFlow Environment Setup Script (No Conda)

echo "Setting up ProtoFlow environment (simple version)..."
echo "This avoids conda conflicts by using pip directly."

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv protoflow_simple_env

# Activate virtual environment
echo "Activating virtual environment..."
source protoflow_simple_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version to avoid CUDA issues)
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install basic requirements
echo "Installing basic requirements..."
pip install numpy matplotlib scikit-learn scipy pillow tqdm

# Install additional requirements from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements from requirements.txt..."
    # Filter out torch-related packages since we already installed them
    grep -v "torch" requirements.txt | pip install -r /dev/stdin || true
fi

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "source protoflow_simple_env/bin/activate"
echo ""
echo "To test the setup:"
echo "python3 test_model_simple.py"
echo ""
echo "To run the demo:"
echo "python3 demo_counterfactuals.py" 