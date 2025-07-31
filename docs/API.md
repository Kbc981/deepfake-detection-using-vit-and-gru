# API Documentation

This document provides detailed API documentation for the Deepfake Detection model components.

## Model Architecture

### FaceClassifierViTGRU

The main model class that combines Vision Transformer and GRU for deepfake detection.

```python
class FaceClassifierViTGRU(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        vit_model_name: str = 'vit_base_patch16_224_in21k',
        gru_hidden_dim: int = 512,
        gru_layers: int = 2,
        bidirectional_gru: bool = True,
        dropout_rate: float = 0.5
    ):
```

#### Parameters
- **num_classes** (int): Number of output classes (default: 2 for real/fake)
- **vit_model_name** (str): Name of the pre-trained ViT model from timm
- **gru_hidden_dim** (int): Hidden dimension of GRU layers
- **gru_layers** (int): Number of GRU layers
- **bidirectional_gru** (bool): Whether to use bidirectional GRU
- **dropout_rate** (float): Dropout rate for regularization

#### Methods

##### forward(x)
Forward pass through the model.

```python
def forward(self, x):
    """
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_frames, 3, 224, 224)
    
    Returns:
        torch.Tensor: Output logits of shape (batch_size, num_classes)
    """
```

**Parameters:**
- **x** (torch.Tensor): Video frames tensor with shape `(batch_size, num_frames, channels, height, width)`

**Returns:**
- **torch.Tensor**: Classification logits with shape `(batch_size, num_classes)`

#### Example Usage
```python
# Initialize model
model = FaceClassifierViTGRU(
    num_classes=2,
    vit_model_name='vit_base_patch16_224_in21k',
    gru_hidden_dim=512,
    gru_layers=2,
    bidirectional_gru=True,
    dropout_rate=0.5
)

# Forward pass
batch_size, num_frames = 8, 15
input_tensor = torch.randn(batch_size, num_frames, 3, 224, 224)
output = model(input_tensor)  # Shape: (8, 2)
```

## Dataset Classes

### CustomImageFolder

Custom dataset class for loading video frame sequences.

```python
class CustomImageFolder(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        num_frames: int = 15,
        apply_oversampling: bool = False,
        oversample_minority_ratio: float = 1.0
    ):
```

#### Parameters
- **root_dir** (str): Root directory containing class subdirectories
- **transform** (transforms.Compose, optional): Image transformations to apply
- **num_frames** (int): Number of frames to sample per video
- **apply_oversampling** (bool): Whether to apply oversampling to minority class
- **oversample_minority_ratio** (float): Ratio for minority class oversampling

#### Methods

##### __len__()
Returns the total number of samples in the dataset.

```python
def __len__(self) -> int:
    """Returns total number of video samples"""
```

##### __getitem__(idx)
Retrieves a single sample from the dataset.

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
    """
    Args:
        idx (int): Sample index
    
    Returns:
        Tuple[torch.Tensor, int]: (frames_tensor, label)
            - frames_tensor: Shape (num_frames, 3, 224, 224)
            - label: 0 for real, 1 for fake
    """
```

#### Example Usage
```python
# Create dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = CustomImageFolder(
    root_dir="/path/to/train",
    transform=transform,
    num_frames=15,
    apply_oversampling=True,
    oversample_minority_ratio=1.0
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Training Functions

### train_one_epoch()

Trains the model for one epoch.

```python
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, float]:
```

#### Parameters
- **model** (nn.Module): The model to train
- **dataloader** (DataLoader): Training data loader
- **optimizer** (torch.optim.Optimizer): Optimizer for training
- **criterion** (nn.Module): Loss function
- **device** (torch.device): Device to run training on
- **scaler** (GradScaler, optional): AMP scaler for mixed precision

#### Returns
- **Tuple[float, float]**: (average_loss, accuracy)

### evaluate()

Evaluates the model on a dataset.

```python
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
```

#### Parameters
- **model** (nn.Module): The model to evaluate
- **dataloader** (DataLoader): Evaluation data loader
- **criterion** (nn.Module): Loss function
- **device** (torch.device): Device to run evaluation on

#### Returns
- **Tuple[float, float, np.ndarray, np.ndarray]**: (loss, accuracy, all_labels, all_preds)

## Utility Functions

### compute_class_weights()

Computes class weights for imbalanced datasets.

```python
def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """
    Computes class weights based on class frequencies.
    
    Args:
        dataset (Dataset): Dataset to compute weights for
    
    Returns:
        torch.Tensor: Class weights tensor
    """
```

### create_weighted_sampler()

Creates a weighted random sampler for balanced training.

```python
def create_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Creates a weighted random sampler for the dataset.
    
    Args:
        dataset (Dataset): Dataset to create sampler for
    
    Returns:
        WeightedRandomSampler: Weighted sampler instance
    """
```

### plot_training_history()

Plots training and validation metrics.

```python
def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    save_path: Optional[str] = None
) -> None:
```

#### Parameters
- **train_losses** (List[float]): Training loss values
- **val_losses** (List[float]): Validation loss values
- **train_accuracies** (List[float]): Training accuracy values
- **val_accuracies** (List[float]): Validation accuracy values
- **save_path** (str, optional): Path to save the plot

### plot_confusion_matrix()

Plots a confusion matrix with statistics.

```python
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Real', 'Fake'],
    save_path: Optional[str] = None
) -> None:
```

#### Parameters
- **y_true** (np.ndarray): True labels
- **y_pred** (np.ndarray): Predicted labels
- **class_names** (List[str]): Names of the classes
- **save_path** (str, optional): Path to save the plot

### plot_roc_curve()

Plots ROC curve and calculates AUC.

```python
def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
) -> float:
```

#### Parameters
- **y_true** (np.ndarray): True binary labels
- **y_scores** (np.ndarray): Target scores (probabilities)
- **save_path** (str, optional): Path to save the plot

#### Returns
- **float**: AUC score

### plot_precision_recall_curve()

Plots Precision-Recall curve and calculates AUC.

```python
def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
) -> float:
```

#### Parameters
- **y_true** (np.ndarray): True binary labels
- **y_scores** (np.ndarray): Target scores (probabilities)
- **save_path** (str, optional): Path to save the plot

#### Returns
- **float**: Average precision score

## Configuration Constants

### Default Hyperparameters

```python
# Data parameters
NUM_FRAMES = 15          # Number of frames to sample per video
IMG_DIM = 224           # Image dimension for ViT input
NUM_CLASSES = 2         # Real and fake classes

# Model parameters
VIT_MODEL_NAME = 'vit_base_patch16_224_in21k'
GRU_HIDDEN_DIM = 512
GRU_LAYERS = 2
BIDIRECTIONAL_GRU = True
DROPOUT_RATE = 0.5

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 7            # For early stopping

# Class imbalance handling
APPLY_WEIGHTED_LOSS = True
APPLY_OVERSAMPLING_TRAIN_DATASET = False
OVERSAMPLE_MINORITY_RATIO_TRAIN = 1.0
APPLY_WEIGHTED_RANDOM_SAMPLER_TRAIN = True
```

## Error Handling

### Common Exceptions

#### DatasetError
Raised when dataset structure is invalid.

```python
class DatasetError(Exception):
    """Raised when dataset structure is invalid"""
    pass
```

#### ModelError
Raised when model configuration is invalid.

```python
class ModelError(Exception):
    """Raised when model configuration is invalid"""
    pass
```

#### TrainingError
Raised when training encounters an error.

```python
class TrainingError(Exception):
    """Raised when training encounters an error"""
    pass
```

## Example Workflow

### Complete Training Pipeline

```python
# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# 2. Data preparation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

train_dataset = CustomImageFolder(
    root_dir="data/train",
    transform=transform_train,
    num_frames=15
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4
)

# 3. Model initialization
model = FaceClassifierViTGRU(
    num_classes=2,
    gru_hidden_dim=512,
    gru_layers=2,
    dropout_rate=0.5
).to(device)

# 4. Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scaler = torch.cuda.amp.GradScaler()

# 5. Training loop
for epoch in range(25):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device, scaler
    )
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

# 6. Evaluation
test_loss, test_acc, y_true, y_pred = evaluate(
    model, test_loader, criterion, device
)

# 7. Visualization
plot_confusion_matrix(y_true, y_pred)
roc_auc = plot_roc_curve(y_true, y_pred)
print(f"Test Accuracy: {test_acc:.4f}, ROC-AUC: {roc_auc:.4f}")
```

## Performance Benchmarks

### Expected Performance Metrics

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| Accuracy | 85-95% | On balanced test set |
| Precision | 80-90% | Fake class detection |
| Recall | 85-95% | Fake class detection |
| F1-Score | 80-90% | Harmonic mean |
| ROC-AUC | 90-98% | Area under ROC curve |
| PR-AUC | 85-95% | Precision-Recall AUC |

### Training Time Estimates

| GPU | Batch Size | Time per Epoch | Total Training Time |
|-----|------------|----------------|-------------------|
| RTX 3070 | 16 | 10-15 min | 4-6 hours |
| RTX 4080 | 32 | 8-12 min | 3-5 hours |
| A100 | 64 | 5-8 min | 2-3 hours |

*Estimates based on ~5000 video dataset with 15 frames per video*