#!/bin/bash

# ProtoFlow Environment Setup Script

echo "Setting up ProtoFlow environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Creating conda environment..."
    
    # Create conda environment from environment.yml
    if [ -f "environment.yml" ]; then
        conda env create -f environment.yml
        echo "✓ Conda environment created from environment.yml"
    else
        echo "environment.yml not found, creating basic environment..."
        conda create -n protoflow python=3.9 -y
        conda activate protoflow
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        conda install numpy matplotlib scikit-learn scipy -y
        pip install -r requirements.txt
    fi
    
    echo "To activate the environment, run: conda activate protoflow"
    
elif command -v python3 &> /dev/null; then
    echo "Conda not found. Creating virtual environment..."
    
    # Create virtual environment
    python3 -m venv protoflow_env
    source protoflow_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install other requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        pip install numpy matplotlib scikit-learn scipy pillow tqdm
    fi
    
    echo "✓ Virtual environment created"
    echo "To activate the environment, run: source protoflow_env/bin/activate"
    
else
    echo "Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

echo ""
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   - For conda: conda activate protoflow"
echo "   - For venv: source protoflow_env/bin/activate"
echo ""
echo "2. Test the setup:"
echo "   python3 test_model_simple.py"
echo ""
echo "3. Run counterfactual generation:"
echo "   python3 generate_counterfactuals_fixed.py --checkpoint <path_to_checkpoint>" 