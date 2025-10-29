"""
Visualize training samples from Fashion MNIST dataset.
Display images in a grid with their corresponding labels.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


# Fashion MNIST class labels
CLASS_NAMES = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


def load_fashion_mnist(data_dir):
    """Load Fashion MNIST training dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    return dataset


def visualize_batch(dataset, num_samples=16, grid_cols=4, save_path=None):
    """
    Visualize a batch of training samples.

    Args:
        dataset: FashionMNIST dataset
        num_samples: Number of samples to display
        grid_cols: Number of columns in the grid
        save_path: Optional path to save the figure
    """
    # Calculate grid dimensions
    grid_rows = (num_samples + grid_cols - 1) // grid_cols

    # Create figure
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 2 * grid_rows))

    # Flatten axes for easier iteration
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Display samples
    for idx in range(num_samples):
        image, label = dataset[idx]

        # Convert tensor to numpy array
        img_array = image.numpy().squeeze()

        # Plot image
        axes[idx].imshow(img_array, cmap='gray')
        axes[idx].set_title(f'{CLASS_NAMES[label]}\n(ID: {label})', fontsize=10)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_by_class(dataset, num_per_class=2, save_path=None):
    """
    Visualize samples from each class.

    Args:
        dataset: FashionMNIST dataset
        num_per_class: Number of samples to show per class
        save_path: Optional path to save the figure
    """
    num_classes = len(CLASS_NAMES)
    total_samples = num_classes * num_per_class

    # Create figure
    fig, axes = plt.subplots(num_classes, num_per_class,
                             figsize=(2 * num_per_class, 2 * num_classes))

    if num_per_class == 1:
        axes = axes.reshape(-1, 1)

    # Collect samples for each class
    class_samples = {i: [] for i in range(num_classes)}

    for image, label in dataset:
        if len(class_samples[label]) < num_per_class:
            class_samples[label].append(image)

        # Check if we have enough samples for each class
        if all(len(samples) == num_per_class for samples in class_samples.values()):
            break

    # Display samples organized by class
    for class_id in range(num_classes):
        for sample_idx in range(num_per_class):
            if sample_idx < len(class_samples[class_id]):
                image = class_samples[class_id][sample_idx]
                img_array = image.numpy().squeeze()

                ax = axes[class_id, sample_idx]
                ax.imshow(img_array, cmap='gray')

                # Only show label on first column
                if sample_idx == 0:
                    ax.set_ylabel(CLASS_NAMES[class_id], fontsize=10, fontweight='bold')
                else:
                    ax.set_ylabel('')

                ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_random_samples(dataset, num_samples=16, grid_cols=4, save_path=None):
    """
    Visualize random samples from the dataset.

    Args:
        dataset: FashionMNIST dataset
        num_samples: Number of samples to display
        grid_cols: Number of columns in the grid
        save_path: Optional path to save the figure
    """
    import random

    # Get random indices
    random_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Calculate grid dimensions
    grid_rows = (num_samples + grid_cols - 1) // grid_cols

    # Create figure
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 2 * grid_rows))
    axes = axes.flatten()

    # Display samples
    for plot_idx, data_idx in enumerate(random_indices):
        image, label = dataset[data_idx]
        img_array = image.numpy().squeeze()

        axes[plot_idx].imshow(img_array, cmap='gray')
        axes[plot_idx].set_title(f'{CLASS_NAMES[label]}\n(Sample {data_idx})', fontsize=10)
        axes[plot_idx].axis('off')

    # Hide unused subplots
    for idx in range(len(random_indices), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_statistics(dataset):
    """
    Display statistics about the training dataset.

    Args:
        dataset: FashionMNIST dataset
    """
    # Count samples per class
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}

    for _, label in dataset:
        class_counts[label] += 1

    # Print statistics
    print("=" * 60)
    print("Fashion MNIST Training Dataset Statistics")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print(f"Image shape: 28x28 pixels (grayscale)")
    print("\nSamples per class:")
    print("-" * 60)

    for class_id, count in class_counts.items():
        print(f"  {class_id:2d}. {CLASS_NAMES[class_id]:20s}: {count:5d} samples")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Fashion MNIST training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Show first 16 samples in 4x4 grid
  python show_train_data.py

  # Show first 25 samples in 5x5 grid
  python show_train_data.py --num-samples 25 --grid-cols 5

  # Show 2 random samples from each class (organized by class)
  python show_train_data.py --by-class --num-per-class 2

  # Show 20 random samples
  python show_train_data.py --random --num-samples 20

  # Display dataset statistics only
  python show_train_data.py --stats-only

  # Save visualization to file
  python show_train_data.py --save-path output.png
        '''
    )

    parser.add_argument('--dataset', type=str, default='../data',
                        help='Path to dataset directory (default: ../data)')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of samples to display (default: 16)')
    parser.add_argument('--grid-cols', type=int, default=4,
                        help='Number of columns in grid (default: 4)')
    parser.add_argument('--by-class', action='store_true',
                        help='Organize samples by class instead of sequential')
    parser.add_argument('--num-per-class', type=int, default=2,
                        help='Number of samples per class when using --by-class (default: 2)')
    parser.add_argument('--random', action='store_true',
                        help='Show random samples instead of first N samples')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only display dataset statistics without visualization')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the visualization (e.g., output.png)')

    args = parser.parse_args()

    # Load dataset
    print("Loading Fashion MNIST training dataset...")
    dataset = load_fashion_mnist(args.dataset)
    print(f"✓ Loaded {len(dataset)} training samples")

    # Display statistics
    visualize_statistics(dataset)

    if args.stats_only:
        return

    # Visualize samples
    print("\nGenerating visualization...")

    if args.by_class:
        visualize_by_class(dataset, num_per_class=args.num_per_class,
                          save_path=args.save_path)
    elif args.random:
        visualize_random_samples(dataset, num_samples=args.num_samples,
                                grid_cols=args.grid_cols, save_path=args.save_path)
    else:
        visualize_batch(dataset, num_samples=args.num_samples,
                       grid_cols=args.grid_cols, save_path=args.save_path)


if __name__ == '__main__':
    main()
