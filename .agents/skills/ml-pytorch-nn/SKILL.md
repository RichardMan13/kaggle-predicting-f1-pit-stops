---
name: ml-pytorch-nn
description: Train, evaluate, and structure deep learning models using PyTorch. Use when building neural networks, configuring PyTorch DataLoaders, optimizing PyTorch custom training loops, or ensuring GPU acceleration compatibility.
---

# PyTorch Deep Learning Models

Use this skill to build reproducible, bug-free, and high-performance deep learning workflows using PyTorch.

## Quick start

Build a clean neural network class and train it safely using a GPU if available:
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Setup device (optimized for Windows GPU support, like GTX 1650)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

# Move model to device
model = SimpleNet(input_dim=10).to(device)
```

## Workflows

### 1. Robust Training Loop Checklist
- [ ] **Gradient Zeroing:** Always call `optimizer.zero_grad()` at the start of the batch loop to prevent gradient accumulation.
- [ ] **Mode Setting:** Always call `model.train()` before starting training steps, and `model.eval()` along with `with torch.no_grad():` block during validation and prediction steps.
- [ ] **Loss Backpropagation:** Verify `loss.backward()` and `optimizer.step()` are executed inside the training block.

### 2. Device Management & Seeds
- [ ] **Device Transfer:** Ensure both input features `inputs.to(device)` and labels `labels.to(device)` are placed on the exact same device as the model.
- [ ] **Reproducibility Seeds:** Set all seeds cleanly:
  ```python
  import random
  import numpy as np
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(42)
  ```
