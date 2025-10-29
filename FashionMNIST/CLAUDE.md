# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **PyTorch-based Fashion MNIST image classification project** with support for distributed training across CUDA, CPU, and Ascend NPU hardware. The codebase implements a convolutional neural network (CNN) for clothing/fashion item classification using the Fashion MNIST dataset (70,000 28x28 grayscale images across 10 classes).

## Repository Structure

```
FashionMNIST/
├── mnist/src/
│   ├── mnist.py              # Main training script (CPU/CUDA variants)
│   ├── mnist-gpu.py          # NPU/Ascend optimized variant
│   └── show_train_data.py    # Visualization tool for training data
├── mnist/data/               # Dataset directory (82 MB, includes FashionMNIST images)
├── .gitattributes            # Git LFS configuration for large files
└── README.md                 # Chinese language project documentation
```

## Technology Stack

- **Framework**: PyTorch (deep learning)
- **Hardware Support**: CUDA (NVIDIA GPUs), CPU, Ascend NPU (Huawei)
- **Distributed Training**: PyTorch Distributed Data Parallel (DDP) with GLOO/NCCL/MPI/HCCL backends
- **Visualization**: TensorBoard (tensorboardX)
- **Version Control**: Git with Git LFS for binary files
- **Python**: 3.13.5 (Anaconda)

## Running the Project

### Basic Training

```bash
# CPU/GPU training with default settings
python mnist/src/mnist.py

# Train for multiple epochs
python mnist/src/mnist.py --epochs 10

# Custom hyperparameters
python mnist/src/mnist.py --epochs 20 --batch-size 128 --lr 0.001 --momentum 0.9

# Save trained model checkpoint
python mnist/src/mnist.py --epochs 10 --save-model

# Train on CPU only (disable CUDA)
python mnist/src/mnist.py --no-cuda --epochs 10

# NPU/Ascend variant
python mnist/src/mnist-gpu.py --epochs 10
```

### Key Command-Line Arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `--epochs` | 1 | Number of training epochs |
| `--batch-size` | 64 | Training batch size |
| `--test-batch-size` | 1000 | Testing batch size |
| `--lr` | 0.01 | Learning rate (SGD optimizer) |
| `--momentum` | 0.5 | SGD momentum |
| `--no-cuda` | False | Disable CUDA, force CPU training |
| `--use-npu` | False | Enable NPU training (mnist-gpu.py only) |
| `--seed` | 1 | Random seed for reproducibility |
| `--log-interval` | 10 | Log training metrics every N batches |
| `--save-model` | False | Save model checkpoint after training |
| `--dataset` | ../data | Path to dataset directory |
| `--save-model-dir` | /data/mnt | Directory to save model checkpoints |
| `--dir` | logs | TensorBoard logs directory |
| `--backend` | gloo | DDP backend (GLOO/NCCL/MPI/HCCL) |

## Architecture Overview

The CNN model (`Net` class) processes 28×28 grayscale images through:

1. **Conv1**: 1 input → 20 filters (5×5 kernel) + ReLU + MaxPool(2×2)
2. **Conv2**: 20 input → 50 filters (5×5 kernel) + ReLU + MaxPool(2×2)
3. **FC1**: Flattened (4×4×50=800) → 500 neurons + ReLU
4. **FC2**: 500 → 10 output classes (Log Softmax)

**Training Details**:
- Loss: Negative Log Likelihood (NLL)
- Optimizer: SGD with configurable momentum
- Data Normalization: FashionMNIST statistics (mean=0.1307, std=0.3081)

## Key Differences: mnist.py vs mnist-gpu.py

| Feature | mnist.py | mnist-gpu.py |
|---------|----------|-------------|
| NPU Support | ❌ | ✅ (torch_npu) |
| Target Hardware | CPU/CUDA | Ascend accelerators |
| Debug Output | ✅ (verbose) | ❌ (clean) |
| Default Backends | GLOO, NCCL, MPI | GLOO, NCCL, MPI |

**When to use**:
- **mnist.py**: Standard deployments, development, NVIDIA GPUs
- **mnist-gpu.py**: Production on Ascend hardware, optimized performance

## Development Workflow

### Single Run with Custom Hyperparameters

```bash
# Quick test with 1 epoch
python mnist/src/mnist.py --epochs 1 --batch-size 32

# Full training pipeline
python mnist/src/mnist.py --epochs 50 --batch-size 128 --lr 0.01 --save-model
```

### Monitoring Training

TensorBoard logs are written to the directory specified by `--dir` (default: `logs/`). Monitor in real-time:

```bash
tensorboard --logdir logs/
```

Metrics tracked:
- Training loss per batch
- Test accuracy per epoch
- Test loss per epoch

### Visualizing Training Data

Use `show_train_data.py` to explore and visualize the Fashion MNIST dataset before training:

```bash
# Display first 16 samples in a 4x4 grid
python mnist/src/show_train_data.py

# Show 25 samples in a 5x5 grid
python mnist/src/show_train_data.py --num-samples 25 --grid-cols 5

# Show 2 random samples from each class (organized by class)
python mnist/src/show_train_data.py --by-class --num-per-class 2

# Show 20 random samples
python mnist/src/show_train_data.py --random --num-samples 20

# Display dataset statistics only (no visualization)
python mnist/src/show_train_data.py --stats-only

# Save visualization to file
python mnist/src/show_train_data.py --save-path training_samples.png
```

**Features**:
- Displays images with class labels and sample indices
- Three visualization modes:
  - **Sequential**: Show first N samples in order (default)
  - **By-Class**: Organize samples by clothing category
  - **Random**: Display random samples
- Statistical summary of training set (sample counts per class)
- Option to save visualizations to file (PNG, JPG, etc.)
- Requires: `matplotlib`

### Distributed Training (Multi-GPU)

```bash
# Use DDP with NCCL backend for multi-GPU
python -m torch.distributed.launch --nproc_per_node=2 mnist/src/mnist.py \
  --epochs 10 --backend nccl
```

## Data

**Dataset**: Fashion MNIST (70,000 total images)
- **Training**: 60,000 images + labels
- **Testing**: 10,000 images + labels
- **Classes**: 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Format**: IDX binary format (standard ML dataset format)
- **Location**: `mnist/data/FashionMNIST/raw/`
- **Size**: 82 MB (includes both compressed .gz and raw formats)

Data is automatically downloaded on first run if not present. Git LFS tracks large binary files.

## Important Implementation Details

### Model Saving

When using `--save-model`, the trained model is saved to `--save-model-dir`. The filename includes a timestamp for easy identification.

```python
# Example checkpoint path: /data/mnt/mnist_model_<timestamp>.pt
```

### Device Selection Priority

1. If `--use-npu` flag set → NPU (Ascend)
2. If `--no-cuda` flag set → CPU
3. Otherwise → CUDA if available, else CPU

### Distributed Training Setup

The code initializes DDP when `RANK` environment variable is detected (set by torch.distributed.launch). Manual distributed setup requires:

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0
python mnist/src/mnist.py --backend gloo
```

## Notes for Contributors

- **Default epochs = 1**: Change when doing actual training (quick test setting)
- **Console output in mnist.py**: Uses print statements for debugging (verbose)
- **No test framework**: Validation is embedded in training loop via `test()` function
- **No requirements.txt**: Install dependencies separately based on your hardware:
  - Core ML: CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` or CPU: `pip install torch torchvision torchaudio`
  - Visualization: `pip install matplotlib`
  - NPU: Requires torch_npu from Ascend toolkit
