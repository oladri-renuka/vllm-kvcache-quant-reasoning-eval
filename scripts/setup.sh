#!/bin/bash
set -e

echo "=== vLLM KV-Cache Quantization Experiment Setup ==="
echo ""

# Check Python
python3 --version
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install PyTorch (adjust based on your CUDA version if needed)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo ""

# Install vLLM from source (latest main branch)
echo "Installing vLLM from source..."
if [ ! -d "vllm_repo" ]; then
    git clone https://github.com/vllm-project/vllm.git vllm_repo
    cd vllm_repo
    pip install -e .
    cd ..
else
    echo "vLLM repo already cloned, skipping clone"
fi
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo ""

# Verify installations
echo "=== Verification ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
echo ""

echo "=== Setup Complete ==="
echo "Activate the environment with: source venv/bin/activate"
echo ""
