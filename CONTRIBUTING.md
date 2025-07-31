# Contributing Guidelines

Thank you for your interest in contributing to the Deepfake Detection project! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Issue Reporting](#issue-reporting)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## 🤝 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. By participating, you agree to uphold this code.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.8 or higher
- Git installed and configured
- Basic understanding of PyTorch and deep learning
- Familiarity with Vision Transformers and RNNs

### Areas for Contribution

We welcome contributions in the following areas:

1. **Model Improvements**
   - New architectures or model variants
   - Performance optimizations
   - Memory efficiency improvements

2. **Data Handling**
   - New data augmentation techniques
   - Dataset format support
   - Data preprocessing improvements

3. **Evaluation Metrics**
   - Additional evaluation metrics
   - Visualization improvements
   - Benchmarking tools

4. **Documentation**
   - Tutorial improvements
   - Code documentation
   - Example notebooks

5. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks

6. **Bug Fixes**
   - Bug reports and fixes
   - Performance issues
   - Compatibility improvements

## 💻 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/your-username/deepfake-detection-using-vit-and-gru.git
cd deepfake-detection-using-vit-and-gru

# Add upstream remote
git remote add upstream https://github.com/Kbc981/deepfake-detection-using-vit-and-gru.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv deepfake-dev
source deepfake-dev/bin/activate  # Linux/macOS
# OR
deepfake-dev\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 jupyter pre-commit
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

### 4. Verify Setup

```bash
# Run basic tests
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import timm; print('timm installed successfully')"

# Test model loading
python -c "
from timm import create_model
model = create_model('vit_base_patch16_224_in21k', pretrained=False)
print('Model loaded successfully')
"
```

## 🔄 Contributing Process

### 1. Create Feature Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# OR
git checkout -b bugfix/issue-description
```

### 2. Make Changes

- Write clear, readable code
- Follow existing code style
- Add appropriate comments and docstrings
- Include tests for new functionality
- Update documentation as needed

### 3. Test Changes

```bash
# Run existing tests
python -m pytest tests/

# Test your specific changes
python test_your_feature.py

# Run code quality checks
black --check .
flake8 .
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new data augmentation technique

- Implement mixup augmentation for video frames
- Add configuration options for mixup parameters
- Include tests for new augmentation methods
- Update documentation with usage examples"
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## 🎨 Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

```python
# Use Black formatter
black --line-length 88 your_file.py

# Use meaningful variable names
num_frames = 15  # Good
n = 15          # Bad

# Add type hints
def process_frames(frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Process video frames for model input."""
    pass

# Use docstrings for functions and classes
class CustomDataset(Dataset):
    """
    Custom dataset for loading video frame sequences.
    
    Args:
        root_dir (str): Root directory containing videos
        transform (transforms.Compose): Image transformations
        num_frames (int): Number of frames to sample per video
    """
    
    def __init__(self, root_dir: str, transform=None, num_frames: int = 15):
        pass
```

### PyTorch Best Practices

```python
# Use device-agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Clear memory when needed
torch.cuda.empty_cache()

# Use context managers
with torch.no_grad():
    # Evaluation code here
    pass

# Proper model modes
model.train()  # For training
model.eval()   # For evaluation
```

### File Organization

```python
# Import order: standard library, third-party, local imports
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from .dataset import CustomDataset
from .model import FaceClassifierViTGRU
```

## 🧪 Testing Guidelines

### Unit Tests

Write unit tests for new functionality:

```python
import pytest
import torch
from src.model import FaceClassifierViTGRU

class TestFaceClassifierViTGRU:
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = FaceClassifierViTGRU(num_classes=2)
        assert model.num_classes == 2
        assert model.gru_hidden_dim == 512
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = FaceClassifierViTGRU(num_classes=2)
        batch_size, num_frames = 4, 15
        input_tensor = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        output = model(input_tensor)
        assert output.shape == (batch_size, 2)
    
    def test_different_input_sizes(self):
        """Test model handles different input sizes."""
        model = FaceClassifierViTGRU(num_classes=2)
        
        # Test different batch sizes
        for batch_size in [1, 8, 16]:
            input_tensor = torch.randn(batch_size, 15, 3, 224, 224)
            output = model(input_tensor)
            assert output.shape == (batch_size, 2)
```

### Integration Tests

Test complete workflows:

```python
def test_training_pipeline():
    """Test complete training pipeline."""
    # Create dummy dataset
    dataset = create_dummy_dataset()
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Initialize model
    model = FaceClassifierViTGRU(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Test one training step
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        break  # Test only one batch
    
    assert loss.item() > 0  # Loss should be positive
```

### Performance Tests

```python
def test_model_performance():
    """Test model performance benchmarks."""
    import time
    
    model = FaceClassifierViTGRU(num_classes=2)
    model.eval()
    
    batch_size = 8
    input_tensor = torch.randn(batch_size, 15, 3, 224, 224)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            output = model(input_tensor)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    assert avg_time < 0.1  # Should process batch in <100ms
```

## 📖 Documentation Guidelines

### Code Documentation

```python
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer for training
        criterion (nn.Module): Loss function
        device (torch.device): Device to run training on
        scaler (GradScaler, optional): AMP scaler for mixed precision
    
    Returns:
        Tuple[float, float]: Average loss and accuracy for the epoch
    
    Example:
        >>> model = FaceClassifierViTGRU()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> criterion = nn.CrossEntropyLoss()
        >>> loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    """
    pass
```

### Markdown Documentation

- Use clear headings and structure
- Include code examples
- Add links to relevant sections
- Use tables for parameter descriptions
- Include visual diagrams when helpful

## 🐛 Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. Use configuration '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9]
- PyTorch version: [e.g. 2.0.0]
- GPU: [e.g. RTX 3080]

**Additional Context**
Add any other context about the problem.
```

### Feature Requests

Use the feature request template:

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How do you envision this feature working?

**Alternative Solutions**
Any alternative approaches you've considered.

**Additional Context**
Any other context or screenshots about the feature request.
```

## 📤 Pull Request Process

### PR Checklist

Before submitting a PR, ensure:

- [ ] Code follows the style guidelines
- [ ] Tests pass locally
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] Branch is up-to-date with main

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

### Review Process

1. **Automated Checks**: CI runs tests and style checks
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Reviewers test functionality
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to maintain clean history

## 🌟 Recognition

Contributors will be recognized:

- In the project README
- In release notes for significant contributions
- Through GitHub's contributor graphs
- Special recognition for major features or fixes

## 📞 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check existing docs first

### Communication Guidelines

- Be respectful and constructive
- Provide clear, detailed information
- Search existing issues before creating new ones
- Use appropriate labels and templates

Thank you for contributing to the Deepfake Detection project! 🎉