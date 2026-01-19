#!/bin/bash
set -e  # Exit on error

ENV_NAME="causvid"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "CausVid Installation Script"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Create conda environment if it doesn't exist
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Activate conda environment
echo "Activating conda environment '${ENV_NAME}'..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Get the conda environment path
CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')
PIP_CMD="${CONDA_ENV_PATH}/bin/pip"
PYTHON_CMD="${CONDA_ENV_PATH}/bin/python"

# Install PyTorch with CUDA support
echo "Installing PyTorch and torchvision with CUDA 12.1 support..."
${PIP_CMD} install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install ninja for faster builds
echo "Installing ninja..."
${PIP_CMD} install ninja

# Setup CUDA environment for flash-attn
# Check if CUDA 12.2 is available (common on clusters)
CUDA_HOME="/usr/local/cuda-12.2"
if [ ! -d "${CUDA_HOME}" ]; then
    # Try alternative locations
    if [ -d "/usr/local/cuda" ]; then
        CUDA_HOME="/usr/local/cuda"
    else
        echo "Warning: CUDA toolkit not found at /usr/local/cuda-12.2 or /usr/local/cuda"
        echo "Attempting to install flash-attn without explicit CUDA_HOME..."
        CUDA_HOME=""
    fi
fi

# Create temp directory on home filesystem to avoid cross-device link issues
mkdir -p ~/.tmp
export TMPDIR=~/.tmp

# Install flash-attn with proper CUDA environment
echo "Installing flash-attn (this may take a while)..."
if [ -n "${CUDA_HOME}" ] && [ -d "${CUDA_HOME}" ]; then
    echo "Using CUDA toolkit at: ${CUDA_HOME}"
    CUDA_HOME=${CUDA_HOME} \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-} \
    TORCH_CUDA_ARCH_LIST=8.0 \
    MAX_JOBS=8 \
    TMPDIR=~/.tmp \
    ${PIP_CMD} install flash-attn --no-build-isolation
else
    echo "Installing flash-attn without explicit CUDA_HOME..."
    TMPDIR=~/.tmp \
    ${PIP_CMD} install flash-attn --no-build-isolation
fi

# Install other requirements
echo "Installing other requirements from requirements.txt..."
${PIP_CMD} install -r "${SCRIPT_DIR}/requirements.txt"

# Install the package in development mode
echo "Installing CausVid package in development mode..."
cd "${SCRIPT_DIR}"
${PYTHON_CMD} setup.py develop

echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Note: Make sure to download the Wan base models from:"
echo "  https://github.com/Wan-Video/Wan2.1"
echo "  and save them to wan_models/Wan2.1-T2V-1.3B/"
