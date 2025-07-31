# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is this project about?
**A:** This project implements a deepfake detection system using a hybrid architecture that combines Vision Transformers (ViT) for spatial feature extraction and Gated Recurrent Units (GRU) for temporal sequence modeling. It's designed to identify whether a video contains real or artificially generated (deepfake) content.

### Q: What makes this approach unique?
**A:** The hybrid ViT-GRU architecture leverages the best of both worlds:
- **ViT**: Excellent at extracting spatial features from individual frames using self-attention mechanisms
- **GRU**: Captures temporal dependencies and patterns across frame sequences
- **Class Imbalance Handling**: Sophisticated techniques to handle imbalanced datasets
- **Modern Best Practices**: Includes mixed precision training, early stopping, and comprehensive evaluation

### Q: What datasets can I use with this model?
**A:** The model expects video frame datasets with the following structure:
```
dataset/
├── train/real/video_id/frames/
├── train/fake/video_id/frames/
├── val/real/video_id/frames/
├── val/fake/video_id/frames/
├── test/real/video_id/frames/
└── test/fake/video_id/frames/
```

Popular deepfake datasets include:
- FaceForensics++
- Celeb-DF
- DFDC (Deepfake Detection Challenge)
- DeeperForensics-1.0

## Technical Questions

### Q: What are the system requirements?
**A:** 
**Minimum Requirements:**
- Python 3.8+
- 8GB RAM
- 4GB GPU VRAM (GTX 1060 or equivalent)
- 20GB storage

**Recommended:**
- Python 3.9+
- 16GB+ RAM
- 8GB+ GPU VRAM (RTX 3070 or better)
- 50GB+ storage

### Q: How long does training take?
**A:** Training time depends on your hardware and dataset size:
- **RTX 3070**: ~4-6 hours for 5000 videos
- **RTX 4080**: ~3-5 hours for 5000 videos  
- **A100**: ~2-3 hours for 5000 videos

For faster training, use mixed precision (AMP) and larger batch sizes.

### Q: What accuracy can I expect?
**A:** Typical performance on balanced test sets:
- **Accuracy**: 85-95%
- **ROC-AUC**: 90-98%
- **F1-Score**: 80-90%

Performance varies based on dataset quality, preprocessing, and hyperparameter tuning.

### Q: Can I use this for real-time detection?
**A:** The current implementation is optimized for batch processing rather than real-time inference. For real-time applications:
- Reduce model size (smaller ViT, fewer GRU layers)
- Use model compression techniques
- Implement frame sampling strategies
- Consider using TensorRT or ONNX for optimization

## Usage Questions

### Q: How do I prepare my video dataset?
**A:** 
1. **Extract frames** from videos (15-30 frames per video recommended)
2. **Organize** into the required directory structure
3. **Resize** frames to 224×224 (handled automatically by transforms)
4. **Split** into train/validation/test sets (70/15/15 typical)

Example script:
```python
import cv2
import os

def extract_frames(video_path, output_dir, max_frames=15):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{output_dir}/frame_{i:03d}.jpg", frame)
    cap.release()
```

### Q: How do I handle class imbalance in my dataset?
**A:** The project provides several strategies:

1. **Weighted Loss** (recommended):
```python
APPLY_WEIGHTED_LOSS = True
```

2. **Weighted Random Sampler**:
```python
APPLY_WEIGHTED_RANDOM_SAMPLER_TRAIN = True
```

3. **Manual Oversampling**:
```python
APPLY_OVERSAMPLING_TRAIN_DATASET = True
OVERSAMPLE_MINORITY_RATIO_TRAIN = 1.0
```

Choose one primary method and tune parameters based on your specific imbalance ratio.

### Q: What hyperparameters should I tune first?
**A:** Priority order for hyperparameter tuning:

1. **Learning Rate** (1e-6 to 1e-3)
2. **Batch Size** (8, 16, 32, 64)
3. **Number of Frames** (10, 15, 20)
4. **Dropout Rate** (0.3, 0.5, 0.7)
5. **GRU Hidden Dimension** (256, 512, 768)

Start with default values and adjust based on validation performance.

### Q: My model is overfitting. What should I do?
**A:** Common solutions for overfitting:

1. **Increase dropout rate**:
```python
DROPOUT_RATE = 0.7  # Increase from 0.5
```

2. **Add more data augmentation**:
```python
transforms.ColorJitter(brightness=0.3, contrast=0.3),
transforms.RandomRotation(degrees=15),
transforms.RandomErasing(p=0.2)
```

3. **Reduce model complexity**:
```python
GRU_HIDDEN_DIM = 256  # Reduce from 512
GRU_LAYERS = 1        # Reduce from 2
```

4. **Use early stopping**:
```python
PATIENCE = 5  # Stop earlier
```

5. **Get more training data** if possible

### Q: How do I save and load my trained model?
**A:** 

**Saving:**
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_val_accuracy': best_val_accuracy,
    'hyperparameters': {
        'num_frames': NUM_FRAMES,
        'gru_hidden_dim': GRU_HIDDEN_DIM,
        # ... other hyperparameters
    }
}, 'best_model.pt')
```

**Loading:**
```python
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Error Questions

### Q: I get "CUDA out of memory" errors. What should I do?
**A:** Several solutions:

1. **Reduce batch size**:
```python
BATCH_SIZE = 8  # Or even 4
```

2. **Reduce number of frames**:
```python
NUM_FRAMES = 10  # Instead of 15
```

3. **Use gradient checkpointing**:
```python
model.gradient_checkpointing = True
```

4. **Enable mixed precision**:
```python
use_amp = True
```

5. **Clear cache periodically**:
```python
torch.cuda.empty_cache()
```

### Q: The ViT model fails to load. What's wrong?
**A:** Common issues:

1. **Update timm**:
```bash
pip install --upgrade timm
```

2. **Check model name**:
```python
import timm
print(timm.list_models('vit*'))
```

3. **Try alternative model names**:
```python
VIT_MODEL_NAME = 'vit_base_patch16_224.augreg_in21k'
# OR
VIT_MODEL_NAME = 'vit_base_patch16_224'
```

### Q: Training loss isn't decreasing. What could be wrong?
**A:** Debugging steps:

1. **Check learning rate** (try 1e-4 or 1e-6)
2. **Verify data loading** (check labels and preprocessing)
3. **Monitor gradients**:
```python
total_norm = sum(p.grad.data.norm(2) ** 2 for p in model.parameters()) ** 0.5
print(f"Gradient norm: {total_norm}")
```

4. **Simplify the problem** (reduce augmentation, check with smaller dataset)

## Performance Questions

### Q: How can I improve model performance?
**A:** Optimization strategies:

1. **Data Quality**:
   - Use high-quality, diverse datasets
   - Ensure proper frame extraction
   - Balance real/fake samples

2. **Model Architecture**:
   - Try different ViT models (larger/smaller)
   - Experiment with LSTM instead of GRU
   - Add attention mechanisms

3. **Training Strategy**:
   - Use learning rate scheduling
   - Try different optimizers (AdamW, SGD)
   - Implement curriculum learning

4. **Ensemble Methods**:
   - Combine multiple models
   - Use different architectures
   - Apply test-time augmentation

### Q: Can I use transfer learning with pre-trained models?
**A:** Yes! The project already uses transfer learning:

- **ViT**: Pre-trained on ImageNet-21k
- **Feature Extraction**: ViT backbone frozen or fine-tuned
- **Classification Head**: Trained from scratch

You can also:
1. Use different pre-trained ViT variants
2. Fine-tune with lower learning rates
3. Implement progressive unfreezing

### Q: How do I evaluate my model properly?
**A:** Comprehensive evaluation approach:

1. **Multiple Metrics**:
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC, Precision-Recall AUC
   - Confusion matrix analysis

2. **Cross-validation**:
   - K-fold validation
   - Stratified sampling

3. **Test Set Evaluation**:
   - Hold-out test set
   - Never used during training/validation

4. **Error Analysis**:
   - Analyze false positives/negatives
   - Identify failure modes
   - Check bias across different demographics

## Advanced Questions

### Q: Can I modify the architecture?
**A:** Absolutely! The modular design allows easy modifications:

1. **Different Backbones**:
   - ResNet, EfficientNet, ConvNext
   - Different ViT variants

2. **Temporal Models**:
   - LSTM, Transformer encoders
   - 3D CNNs, temporal attention

3. **Fusion Strategies**:
   - Early/late fusion
   - Attention-based fusion
   - Multi-scale features

### Q: How do I handle videos of different lengths?
**A:** Current implementation uses fixed frame sampling:

```python
# Modify sampling strategy in dataset class
def sample_frames(self, frame_files, num_frames):
    if len(frame_files) >= num_frames:
        # Uniform sampling
        indices = np.linspace(0, len(frame_files)-1, num_frames, dtype=int)
    else:
        # Repeat frames if video is shorter
        indices = np.random.choice(len(frame_files), num_frames, replace=True)
    
    return [frame_files[i] for i in indices]
```

### Q: Can I use this for other video classification tasks?
**A:** Yes! The architecture is general and can be adapted for:

- Action recognition
- Video quality assessment
- Content classification
- Temporal event detection

Simply modify:
- Number of classes
- Dataset structure
- Loss function (if needed)
- Evaluation metrics

## Support Questions

### Q: Where can I get help if I'm stuck?
**A:** Multiple support channels:

1. **Documentation**:
   - Check [INSTALLATION.md](INSTALLATION.md)
   - Review [USAGE.md](USAGE.md)
   - Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

2. **GitHub**:
   - Search existing issues
   - Create new issue with details
   - Use appropriate issue templates

3. **Community**:
   - GitHub Discussions
   - Stack Overflow (tag: deepfake-detection)

### Q: How can I contribute to the project?
**A:** We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for:

- Code contributions
- Bug reports
- Feature requests
- Documentation improvements
- Testing and benchmarking

### Q: Is there a paper or more technical details?
**A:** The implementation is based on:

- Vision Transformer: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- GRU: [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)
- Deepfake Detection: Various recent papers on video-based detection

For more technical details, check the code comments and architecture documentation in the notebook.

---

**Don't see your question here?** Feel free to:
- Open an issue on GitHub
- Check the other documentation files
- Start a discussion in the repository