# Troubleshooting Guide

This guide helps you resolve common issues when working with the Deepfake Detection project.

## 🚨 Common Issues and Solutions

### Installation Issues

#### Issue: PyTorch CUDA version mismatch
```
RuntimeError: The NVIDIA driver on your system is too old
```

**Solution:**
1. Check your CUDA version:
```bash
nvidia-smi
nvcc --version
```

2. Install compatible PyTorch version:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Issue: timm model loading fails
```
KeyError: 'vit_base_patch16_224_in21k'
```

**Solution:**
1. Update timm to latest version:
```bash
pip install --upgrade timm
```

2. Check available models:
```python
import timm
available_models = timm.list_models('vit*')
print(available_models)
```

3. Use alternative model name if needed:
```python
# Try these alternatives
VIT_MODEL_NAME = 'vit_base_patch16_224.augreg_in21k'
# OR
VIT_MODEL_NAME = 'vit_base_patch16_224'
```

### Memory Issues

#### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. **Reduce batch size:**
```python
BATCH_SIZE = 8  # Instead of 32
# Or even smaller
BATCH_SIZE = 4
```

2. **Reduce number of frames:**
```python
NUM_FRAMES = 10  # Instead of 15
```

3. **Use gradient checkpointing:**
```python
# Add to model initialization
model.gradient_checkpointing = True
```

4. **Clear cache periodically:**
```python
# Add in training loop
if batch_idx % 10 == 0:
    torch.cuda.empty_cache()
```

5. **Use mixed precision training:**
```python
# Enable AMP
use_amp = True
scaler = torch.cuda.amp.GradScaler()

# In training loop
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

#### Issue: CPU Memory Issues
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce data loader workers:**
```python
DataLoader(dataset, batch_size=32, num_workers=2)  # Instead of 4+
```

2. **Use memory mapping for large datasets:**
```python
# Implement lazy loading in dataset class
def __getitem__(self, idx):
    # Load only when needed
    image = Image.open(self.image_paths[idx])
    return transform(image)
```

3. **Process data in smaller chunks:**
```python
# Split dataset into smaller batches
chunk_size = 1000
for i in range(0, len(dataset), chunk_size):
    chunk = dataset[i:i+chunk_size]
    # Process chunk
```

### Dataset Issues

#### Issue: Dataset directory not found
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/dataset'
```

**Solutions:**

1. **Check absolute vs relative paths:**
```python
import os
DATASET_ROOT = os.path.abspath("/path/to/your/dataset")
print(f"Dataset path exists: {os.path.exists(DATASET_ROOT)}")
```

2. **Verify directory structure:**
```python
# Check expected structure
expected_dirs = ['train/real', 'train/fake', 'val/real', 'val/fake', 'test/real', 'test/fake']
for dir_path in expected_dirs:
    full_path = os.path.join(DATASET_ROOT, dir_path)
    print(f"{dir_path}: {os.path.exists(full_path)}")
```

3. **List directory contents:**
```python
import os
for root, dirs, files in os.walk(DATASET_ROOT):
    level = root.replace(DATASET_ROOT, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files) - 5} more files")
```

#### Issue: No frames found in video directories
```
IndexError: list index out of range
```

**Solutions:**

1. **Check supported image formats:**
```python
# Modify CustomImageFolder to support more formats
supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
frame_files = [f for f in os.listdir(video_dir) 
               if any(f.lower().endswith(ext) for ext in supported_formats)]
```

2. **Debug frame loading:**
```python
def debug_dataset_loading(dataset_path):
    """Debug dataset loading issues."""
    total_videos = 0
    empty_videos = 0
    
    for split in ['train', 'val', 'test']:
        for class_name in ['real', 'fake']:
            class_dir = os.path.join(dataset_path, split, class_name)
            if os.path.exists(class_dir):
                video_dirs = [d for d in os.listdir(class_dir) 
                             if os.path.isdir(os.path.join(class_dir, d))]
                
                for video_dir in video_dirs:
                    video_path = os.path.join(class_dir, video_dir)
                    frame_files = [f for f in os.listdir(video_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    total_videos += 1
                    if len(frame_files) == 0:
                        empty_videos += 1
                        print(f"Empty video directory: {video_path}")
    
    print(f"Total videos: {total_videos}")
    print(f"Empty videos: {empty_videos}")

debug_dataset_loading(DATASET_ROOT)
```

### Training Issues

#### Issue: Loss not decreasing
```
Epoch 1: Loss=0.693, Acc=0.500
Epoch 2: Loss=0.693, Acc=0.500
...
```

**Solutions:**

1. **Check learning rate:**
```python
# Try different learning rates
LEARNING_RATE = 1e-4  # Increase if too small
# OR
LEARNING_RATE = 1e-6  # Decrease if too large
```

2. **Verify data preprocessing:**
```python
# Check if labels are correct
def verify_labels(dataloader):
    label_counts = {}
    for _, labels in dataloader:
        for label in labels:
            label_counts[label.item()] = label_counts.get(label.item(), 0) + 1
    print(f"Label distribution: {label_counts}")

verify_labels(train_loader)
```

3. **Check data augmentation:**
```python
# Temporarily disable augmentation to test
transform_simple = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

4. **Monitor gradients:**
```python
# Add gradient monitoring
def monitor_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

# In training loop
grad_norm = monitor_gradients(model)
print(f"Gradient norm: {grad_norm}")
```

#### Issue: Model overfitting quickly
```
Train Acc: 0.95, Val Acc: 0.60
```

**Solutions:**

1. **Increase dropout:**
```python
DROPOUT_RATE = 0.7  # Increase from 0.5
```

2. **Add more data augmentation:**
```python
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.RandomErasing(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

3. **Reduce model complexity:**
```python
GRU_HIDDEN_DIM = 256  # Reduce from 512
GRU_LAYERS = 1        # Reduce from 2
```

4. **Use early stopping:**
```python
# Reduce patience
PATIENCE = 5  # Stop earlier
```

### Performance Issues

#### Issue: Slow training
```
Training taking much longer than expected
```

**Solutions:**

1. **Profile data loading:**
```python
import time

# Time data loading
start_time = time.time()
for i, (data, target) in enumerate(train_loader):
    if i == 10:  # Test first 10 batches
        break
end_time = time.time()
print(f"Average batch loading time: {(end_time - start_time) / 10:.3f}s")
```

2. **Optimize data loading:**
```python
# Increase number of workers
DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True)

# Use persistent workers (PyTorch 1.7+)
DataLoader(dataset, batch_size=32, num_workers=4, persistent_workers=True)
```

3. **Enable compilation (PyTorch 2.0+):**
```python
# Compile model for faster execution
model = torch.compile(model)
```

4. **Use appropriate device:**
```python
# Ensure using GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check GPU utilization
nvidia-smi  # Run in terminal
```

### Model Loading Issues

#### Issue: Checkpoint loading fails
```
KeyError: 'model_state_dict'
```

**Solutions:**

1. **Check checkpoint structure:**
```python
checkpoint = torch.load('model_checkpoint.pt')
print(f"Checkpoint keys: {checkpoint.keys()}")
```

2. **Handle different checkpoint formats:**
```python
def load_model_flexible(model, checkpoint_path):
    """Load model with flexible checkpoint format handling."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume entire checkpoint is state_dict
        model.load_state_dict(checkpoint)
    
    return model
```

3. **Handle model parameter mismatch:**
```python
def load_model_partial(model, checkpoint_path):
    """Load model with partial parameter matching."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    
    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint.items() 
                      if k in model_dict and v.size() == model_dict[k].size()}
    
    # Update model dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
    return model
```

## 🔧 Debugging Tools

### Debug Dataset
```python
def debug_dataset(dataset, num_samples=5):
    """Debug dataset by examining samples."""
    for i in range(min(num_samples, len(dataset))):
        try:
            sample, label = dataset[i]
            print(f"Sample {i}: shape={sample.shape}, label={label}")
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
```

### Debug Model
```python
def debug_model(model, input_shape=(2, 15, 3, 224, 224)):
    """Debug model architecture and forward pass."""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    try:
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Model forward pass error: {e}")
        
    # Print model architecture
    print("\nModel architecture:")
    print(model)
```

### Monitor Training
```python
def monitor_training(model, dataloader, criterion, device):
    """Monitor training metrics in detail."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
```

## 📞 Getting Help

If you're still experiencing issues:

1. **Check GitHub Issues**: Search for similar problems
2. **Create New Issue**: Use the bug report template
3. **Provide Details**: Include error messages, environment info, and reproduction steps
4. **Share Code**: Include minimal code that reproduces the issue

### Issue Template
```markdown
**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- PyTorch: [e.g., 2.0.0]
- CUDA: [e.g., 11.8]
- GPU: [e.g., RTX 3080]

**Error Message:**
```
[Paste full error message here]
```

**Code to Reproduce:**
```python
# Minimal code that reproduces the issue
```

**Expected Behavior:**
[What you expected to happen]

**Actual Behavior:**
[What actually happened]
```

Remember: The more details you provide, the easier it is to help you solve the problem!