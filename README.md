# Deepfake Detection using Vision Transformer (ViT) and Gated Recurrent Unit (GRU)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A robust deep learning solution for detecting deepfake videos using a hybrid architecture that combines Vision Transformers (ViT) for spatial feature extraction and Gated Recurrent Units (GRU) for temporal sequence modeling.

## 🎯 Overview

This project implements a state-of-the-art deepfake detection system that addresses the growing challenge of synthetic media detection. The hybrid model leverages:

- **Vision Transformer (ViT)**: Pre-trained ViT model for extracting spatial features from individual video frames
- **Gated Recurrent Unit (GRU)**: Sequential modeling to capture temporal dependencies across frame sequences
- **Class Imbalance Handling**: Advanced techniques including weighted loss, weighted sampling, and oversampling strategies
- **Comprehensive Evaluation**: Multiple metrics including ROC-AUC, Precision-Recall AUC, F1-score, and confusion matrices

## 🏗️ Architecture

```
Video Frames → ViT Feature Extraction → Temporal Features → GRU → Classification
    (224x224)         (768-dim)            (sequence)      (512-dim)    (Real/Fake)
```

### Model Components:
1. **Feature Extraction**: Pre-trained ViT (`vit_base_patch16_224_in21k`) extracts 768-dimensional features from each frame
2. **Temporal Modeling**: Bidirectional GRU processes sequence of frame features to capture temporal patterns
3. **Classification**: Final dense layers with dropout for binary classification (Real/Fake)

## 📊 Dataset Requirements

The model expects a structured dataset with the following hierarchy:

```
dataset_root/
├── train/
│   ├── real/
│   │   ├── video_001/
│   │   │   ├── frame_001.jpg
│   │   │   ├── frame_002.jpg
│   │   │   └── ...
│   │   └── video_002/
│   └── fake/
│       ├── video_001/
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Dataset Characteristics:
- **Frame Sampling**: 15 frames per video (configurable)
- **Image Size**: 224×224 pixels (ViT standard input)
- **Class Distribution**: ~4000 fake videos, ~1361 real videos
- **Frame Range**: Videos typically contain 27-32 frames

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Kbc981/deepfake-detection-using-vit-and-gru.git
cd deepfake-detection-using-vit-and-gru
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **For GPU support (recommended):**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Running the Model

1. **Prepare your dataset** according to the structure shown above

2. **Open the Jupyter notebook:**
```bash
jupyter notebook maincode.ipynb
```

3. **Configure parameters** in the notebook:
   - Update `DATASET_ROOT` to point to your dataset
   - Adjust hyperparameters as needed
   - Set appropriate batch size based on your GPU memory

4. **Run the cells sequentially** to train and evaluate the model

## ⚙️ Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_FRAMES` | 15 | Number of frames to sample per video |
| `IMG_DIM` | 224 | Input image dimension for ViT |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 1e-5 | Adam optimizer learning rate |
| `NUM_EPOCHS` | 25 | Maximum training epochs |
| `GRU_HIDDEN_DIM` | 512 | GRU hidden state dimension |
| `GRU_LAYERS` | 2 | Number of GRU layers |
| `DROPOUT_RATE` | 0.5 | Dropout rate for regularization |

### Class Imbalance Handling Options

- **Weighted Loss**: Automatically computed class weights
- **Weighted Random Sampler**: Balanced sampling during training
- **Manual Oversampling**: Data augmentation for minority class

## 📈 Performance Metrics

The model is evaluated using comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve
- **Confusion Matrix**: Detailed breakdown of predictions

## 🔧 Advanced Features

### Data Augmentation
- Random resized crop
- Horizontal flipping
- Color jittering
- Random rotation
- Random erasing

### Training Optimizations
- Automatic Mixed Precision (AMP) for faster training
- Early stopping with patience
- Learning rate scheduling (CosineAnnealingLR)
- Model checkpointing

### Visualization
- Training/validation loss and accuracy curves
- ROC curves and PR curves
- Confusion matrices with detailed statistics
- Feature importance analysis

## 📁 Project Structure

```
├── maincode.ipynb          # Main implementation notebook
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── LICENSE               # License information
└── docs/                 # Additional documentation
    ├── INSTALLATION.md   # Detailed setup guide
    ├── USAGE.md         # Usage examples
    └── API.md           # API documentation
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the [FAQ](docs/FAQ.md)
- Review the [troubleshooting guide](docs/TROUBLESHOOTING.md)

## 🏆 Acknowledgments

- Thanks to the PyTorch and timm communities for excellent pre-trained models
- Inspired by recent advances in transformer architectures for computer vision
- Built with modern deep learning best practices for reproducibility and performance