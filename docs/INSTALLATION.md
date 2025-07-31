# Installation Guide

This guide provides detailed installation instructions for the Deepfake Detection project.

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Minimum: 6GB VRAM (GTX 1060, RTX 2060)
  - Recommended: 8GB+ VRAM (RTX 3070, RTX 4080, A100)
- **RAM**: 16GB+ system RAM
- **Storage**: 50GB+ free space for datasets and models

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)
- **Operating System**: Linux (Ubuntu 18.04+), Windows 10/11, or macOS 10.15+

## Installation Methods

### Method 1: Standard Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Kbc981/deepfake-detection-using-vit-and-gru.git
cd deepfake-detection-using-vit-and-gru
```

2. **Create a virtual environment (recommended):**
```bash
# Using conda
conda create -n deepfake-detection python=3.9
conda activate deepfake-detection

# Using venv
python -m venv deepfake-detection
source deepfake-detection/bin/activate  # Linux/macOS
# OR
deepfake-detection\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Method 2: GPU-Optimized Installation

For CUDA-enabled systems:

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install timm numpy matplotlib seaborn scikit-learn pandas jupyter
```

### Method 3: Google Colab Setup

For running in Google Colab:

1. Upload the notebook to Google Colab
2. Run the installation cell in the notebook:
```python
!pip install torch torchvision torchaudio timm numpy matplotlib seaborn scikit-learn pandas
```

3. Mount Google Drive for dataset storage:
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Verification

### Test Installation
```python
import torch
import torchvision
import timm
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Test ViT Model Loading
```python
from timm import create_model

# Test loading the ViT model
model = create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
print(f"ViT model loaded successfully. Feature dimension: {model.embed_dim}")
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce batch size (try 16, 8, or 4)
- Reduce number of frames per video
- Use gradient checkpointing
- Close other GPU applications

#### Missing CUDA
```
torch.cuda.is_available() returns False
```
**Solutions:**
- Install CUDA toolkit from NVIDIA
- Reinstall PyTorch with CUDA support
- Check GPU drivers are up to date

#### Import Errors
```
ModuleNotFoundError: No module named 'timm'
```
**Solutions:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version compatibility

#### Dataset Loading Issues
```
FileNotFoundError: Dataset directory not found
```
**Solutions:**
- Verify dataset path in configuration
- Check directory structure matches requirements
- Ensure proper file permissions

### Performance Optimization

#### For Limited GPU Memory:
```python
# Reduce batch size
BATCH_SIZE = 8

# Use mixed precision training
use_amp = True

# Reduce number of frames
NUM_FRAMES = 10
```

#### For Faster Training:
```python
# Increase batch size if memory allows
BATCH_SIZE = 64

# Use multiple workers for data loading
num_workers = 4

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
```

## Next Steps

After successful installation:
1. Review the [Usage Guide](USAGE.md)
2. Prepare your dataset according to the requirements
3. Configure hyperparameters in the notebook
4. Start training your model

## Getting Help

If you encounter issues not covered here:
- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review [GitHub Issues](https://github.com/Kbc981/deepfake-detection-using-vit-and-gru/issues)
- Create a new issue with detailed error information