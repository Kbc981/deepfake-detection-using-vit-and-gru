# Usage Guide

This guide explains how to use the Deepfake Detection model for training, evaluation, and inference.

## Quick Start

### 1. Dataset Preparation

#### Dataset Structure
Organize your dataset in the following structure:
```
your_dataset/
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

#### Dataset Guidelines
- **Frame Format**: JPEG or PNG images
- **Frame Size**: Any size (will be resized to 224×224)
- **Frames per Video**: 15-32 frames recommended
- **File Naming**: Sequential naming preferred (frame_001.jpg, frame_002.jpg, etc.)

### 2. Configuration

Open `maincode.ipynb` and modify the configuration section:

```python
# Update dataset path
DATASET_ROOT = "/path/to/your/dataset"

# Adjust hyperparameters based on your hardware
BATCH_SIZE = 32  # Reduce if GPU memory is limited
NUM_EPOCHS = 25
LEARNING_RATE = 1e-5

# Model configuration
NUM_FRAMES = 15  # Number of frames to sample per video
GRU_HIDDEN_DIM = 512
GRU_LAYERS = 2
```

### 3. Training

#### Basic Training
Run all cells in the notebook sequentially:

1. **Environment Setup**: Install dependencies and import libraries
2. **Data Loading**: Load and preprocess the dataset
3. **Model Definition**: Initialize the ViT-GRU hybrid model
4. **Training Loop**: Train the model with early stopping
5. **Evaluation**: Evaluate on test set and generate metrics

#### Advanced Training Options

**Class Imbalance Handling:**
```python
# Option 1: Weighted Loss (recommended)
APPLY_WEIGHTED_LOSS = True

# Option 2: Weighted Random Sampler
APPLY_WEIGHTED_RANDOM_SAMPLER_TRAIN = True

# Option 3: Manual Oversampling
APPLY_OVERSAMPLING_TRAIN_DATASET = True
OVERSAMPLE_MINORITY_RATIO_TRAIN = 1.0
```

**Training Optimization:**
```python
# Mixed precision training for faster training
use_amp = True

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Early stopping
PATIENCE = 7  # Stop if no improvement for 7 epochs
```

### 4. Monitoring Training

The notebook provides real-time monitoring:

#### Training Metrics
- **Loss**: Training and validation loss curves
- **Accuracy**: Training and validation accuracy
- **Learning Rate**: Learning rate schedule visualization

#### Live Plotting
```python
# Training progress is automatically plotted
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.legend()
```

### 5. Model Evaluation

#### Comprehensive Metrics
The model generates detailed evaluation reports:

```python
# Classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# ROC and PR curves
roc_auc = roc_auc_score(y_true, y_scores)
pr_auc = average_precision_score(y_true, y_scores)
```

#### Confusion Matrix
```python
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
```

### 6. Model Inference

#### Single Video Prediction
```python
def predict_video(model, video_frames, device):
    """
    Predict if a video is real or fake
    
    Args:
        model: Trained ViT-GRU model
        video_frames: List of PIL Images or tensor
        device: torch.device
    
    Returns:
        prediction: 0 (real) or 1 (fake)
        confidence: probability score
    """
    model.eval()
    with torch.no_grad():
        # Preprocess frames
        frames_tensor = preprocess_frames(video_frames)
        frames_tensor = frames_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        outputs = model(frames_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
    return prediction, confidence
```

#### Batch Prediction
```python
def predict_batch(model, dataloader, device):
    """Predict on a batch of videos"""
    model.eval()
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for frames_batch, _ in dataloader:
            frames_batch = frames_batch.to(device)
            outputs = model(frames_batch)
            probabilities = torch.softmax(outputs, dim=1)
            
            batch_predictions = torch.argmax(probabilities, dim=1)
            batch_confidences = torch.max(probabilities, dim=1)[0]
            
            predictions.extend(batch_predictions.cpu().numpy())
            confidences.extend(batch_confidences.cpu().numpy())
    
    return predictions, confidences
```

### 7. Model Saving and Loading

#### Saving the Model
```python
# Save model checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_val_accuracy': best_val_accuracy,
    'hyperparameters': {
        'num_frames': NUM_FRAMES,
        'gru_hidden_dim': GRU_HIDDEN_DIM,
        'gru_layers': GRU_LAYERS,
        'dropout_rate': DROPOUT_RATE
    }
}, 'vit_gru_deepfake_best.pt')
```

#### Loading the Model
```python
# Load model checkpoint
checkpoint = torch.load('vit_gru_deepfake_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 8. Hyperparameter Tuning

#### Grid Search Example
```python
hyperparameter_grid = {
    'learning_rate': [1e-5, 5e-5, 1e-4],
    'batch_size': [16, 32, 64],
    'gru_hidden_dim': [256, 512, 768],
    'dropout_rate': [0.3, 0.5, 0.7]
}

# Implement grid search logic
best_params = {}
best_score = 0

for params in parameter_combinations:
    # Train model with current parameters
    score = train_and_evaluate(params)
    if score > best_score:
        best_score = score
        best_params = params
```

#### Random Search
```python
import random

def random_search(n_trials=20):
    best_params = None
    best_score = 0
    
    for trial in range(n_trials):
        # Randomly sample hyperparameters
        params = {
            'learning_rate': random.uniform(1e-6, 1e-3),
            'batch_size': random.choice([16, 32, 64]),
            'gru_hidden_dim': random.choice([256, 512, 768, 1024]),
            'dropout_rate': random.uniform(0.2, 0.8)
        }
        
        # Train and evaluate
        score = train_and_evaluate(params)
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score
```

### 9. Performance Optimization

#### Memory Optimization
```python
# For limited GPU memory
BATCH_SIZE = 8
NUM_FRAMES = 10

# Enable gradient checkpointing
model.gradient_checkpointing = True

# Clear cache periodically
if batch_idx % 10 == 0:
    torch.cuda.empty_cache()
```

#### Speed Optimization
```python
# Use mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Optimize data loading
DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

# Compile model (PyTorch 2.0+)
model = torch.compile(model)
```

### 10. Troubleshooting

#### Common Issues and Solutions

**CUDA Out of Memory:**
- Reduce batch size
- Reduce number of frames per video
- Use gradient accumulation

**Poor Performance:**
- Check data quality and labeling
- Adjust learning rate
- Try different augmentation strategies
- Increase training epochs

**Overfitting:**
- Increase dropout rate
- Add more data augmentation
- Reduce model complexity
- Use early stopping

## Next Steps

- Explore advanced architectures
- Implement ensemble methods
- Add multi-scale feature extraction
- Experiment with different pre-trained models

## Getting Help

For additional support:
- Check the [API Documentation](API.md)
- Review [Troubleshooting Guide](TROUBLESHOOTING.md)
- Open an issue on GitHub