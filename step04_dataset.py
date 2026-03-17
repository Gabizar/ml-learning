"""
Step 4: Datasets and DataLoaders — Loading CIFAR-10
====================================================
CIFAR-10 is a standard image classification dataset.
60,000 color images (32x32 pixels) across 10 categories:
airplane, car, bird, cat, deer, dog, frog, horse, ship, truck.

Usage:
    python step04_dataset.py
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms

os.makedirs("output/step04", exist_ok=True)

# ============================================================================
# PART 1: TRANSFORMS — preprocessing images before feeding them to a network
# ============================================================================

print("=" * 60)
print("PART 1: TRANSFORMS")
print("=" * 60)

# TRANSFORM: a preprocessing step applied to each image automatically.
# Raw images are pixels 0-255 (integers). Networks work better with small
# floats centered around 0.
#
# ToTensor()       — converts PIL image to tensor, scales pixels 0-255 → 0.0-1.0
# Normalize(m, s)  — shifts and scales: output = (input - mean) / std
#                    these mean/std values are the channel-wise averages across
#                    the entire CIFAR-10 training set (precomputed, standard values)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),   # mean per channel (R, G, B)
        std=(0.2470, 0.2435, 0.2616)     # std per channel (R, G, B)
    )
])

print(f"\ntransform pipeline:")
print(f"  1. ToTensor()  — pixel values 0-255 → 0.0-1.0, shape (H,W,3) → (3,H,W)")
print(f"  2. Normalize() — center values around 0 using CIFAR-10 channel statistics")
print(f"     mean per channel (R,G,B): (0.4914, 0.4822, 0.4465)")
print(f"     std  per channel (R,G,B): (0.2470, 0.2435, 0.2616)")

# ============================================================================
# PART 2: LOADING THE DATASET
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: LOADING THE DATASET")
print("=" * 60)

# torchvision.datasets.CIFAR10 downloads the dataset if not already present,
# then wraps it so you can index it like a list: dataset[0] = (image, label)
#
# train=True  → 50,000 training images
# train=False → 10,000 test images (used to evaluate after training)

print("\nDownloading/loading CIFAR-10 (will download ~170MB on first run)...")

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",       # where to save/look for the dataset
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print(f"\ntrain_dataset: {train_dataset}")
print(f"len(train_dataset): {len(train_dataset)}  (50,000 training images)")
print(f"len(test_dataset):  {len(test_dataset)}  (10,000 test images)")

# The 10 class names
class_names = train_dataset.classes
print(f"\nclass_names: {class_names}")
print(f"number of classes: {len(class_names)}")

# ============================================================================
# PART 3: INSPECTING A SINGLE SAMPLE
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: SINGLE SAMPLE")
print("=" * 60)

# Indexing the dataset returns a tuple: (image_tensor, label_int)
image, label = train_dataset[0]
print(f"\ntrain_dataset[0] returns: (image, label)")
print(f"type(image): {type(image)}")
print(f"image.shape: {image.shape}   (channels, height, width) = (3, 32, 32)")
print(f"image.dtype: {image.dtype}")
print(f"image.min():  {image.min():.4f}  (after normalize, values go below 0)")
print(f"image.max():  {image.max():.4f}  (after normalize, values go above 1)")
print(f"\nlabel: {label}  → class name: '{class_names[label]}'")

# Look at a few more samples to see different labels
print(f"\nFirst 10 samples and their labels:")
for i in range(10):
    _, lbl = train_dataset[i]
    print(f"  sample {i}: label={lbl} ({class_names[lbl]})")

# Save a grid of sample images to disk so you can view them.
# We need a separate dataset WITHOUT normalize — normalized values can be
# negative or >1 which don't map to valid pixel colors for display.
import matplotlib
matplotlib.use("Agg")   # no display window needed — saves to file
import matplotlib.pyplot as plt

viz_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False,
    transform=transforms.ToTensor()   # only ToTensor, no normalize
)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
fig.suptitle("CIFAR-10 sample images", fontsize=14)
for i, ax in enumerate(axes.flat):
    img, lbl = viz_dataset[i]
    ax.imshow(img.permute(1, 2, 0))  # (3,32,32) → (32,32,3) for matplotlib
    ax.set_title(class_names[lbl], fontsize=7)
    ax.axis("off")
plt.tight_layout()
plt.savefig("output/step04/cifar10_samples.png", dpi=150)
plt.close()
print(f"\nSaved 32 sample images to: output/step04/cifar10_samples.png")

# ============================================================================
# PART 4: DATALOADER — batching and shuffling automatically
# ============================================================================

print("\n" + "=" * 60)
print("PART 4: DATALOADER")
print("=" * 60)

# DATALOADER wraps a dataset and delivers batches one at a time.
# You can't feed all 50,000 images at once — won't fit in GPU memory.
# Instead, the DataLoader slices the dataset into batches of batch_size.
#
# BATCH: a small chunk of the dataset processed together. Standard sizes: 32, 64, 128.
# EPOCH: one full pass through the entire dataset.
#   50,000 images / batch_size 64 = 782 batches per epoch.
#
# shuffle=True: randomizes the order each epoch so the network doesn't
#               memorize the sequence. Always True for training, False for test.
# num_workers=2: number of CPU processes loading data in parallel while GPU trains.

batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,    # no need to shuffle test data
    num_workers=2
)

print(f"\nbatch_size: {batch_size}")
print(f"len(train_loader): {len(train_loader)}  (number of batches per epoch)")
print(f"  = ceil({len(train_dataset)} images / {batch_size} batch size)")
print(f"len(test_loader):  {len(test_loader)}")

# ============================================================================
# PART 5: INSPECTING A BATCH
# ============================================================================

print("\n" + "=" * 60)
print("PART 5: INSPECTING A BATCH")
print("=" * 60)

# Iterating over the DataLoader gives you one batch at a time.
# Each batch is a tuple: (images_tensor, labels_tensor)
# images shape: (batch_size, channels, height, width) = (64, 3, 32, 32)
# labels shape: (batch_size,) = (64,)

images, labels = next(iter(train_loader))   # grab the first batch
print(f"\nFirst batch:")
print(f"images.shape: {images.shape}   (batch_size, channels, height, width)")
print(f"labels.shape: {labels.shape}   (batch_size,)")
print(f"images.dtype: {images.dtype}")
print(f"labels.dtype: {labels.dtype}")
print(f"images.min(): {images.min():.4f}")
print(f"images.max(): {images.max():.4f}")

print(f"\nLabels in this batch:")
print(f"  raw:   {labels[:16].tolist()}")
print(f"  names: {[class_names[l] for l in labels[:16].tolist()]}")

# Label distribution in this batch
from collections import Counter
label_counts = Counter(labels.tolist())
print(f"\nLabel distribution in batch (label: count):")
for lbl in sorted(label_counts):
    print(f"  {lbl} ({class_names[lbl]:>10}): {label_counts[lbl]}")

# ============================================================================
# PART 6: LOOPING OVER THE DATALOADER
# ============================================================================

print("\n" + "=" * 60)
print("PART 6: LOOPING OVER THE DATALOADER")
print("=" * 60)

# This is what a real training loop looks like at the outer level.
# For now we just count batches — we'll add the actual model in Step 5.

print(f"\nSimulating one epoch (looping over all batches)...")
total_images_seen = 0
for batch_idx, (images, labels) in enumerate(train_loader):
    total_images_seen += len(images)
    # training step will go here in Step 5

print(f"Total batches in one epoch: {batch_idx + 1}")
print(f"Total images seen in one epoch: {total_images_seen}")
print(f"(matches len(train_dataset) = {len(train_dataset)})")

print("\n" + "=" * 60)
print("DONE! Key takeaways:")
print("=" * 60)
print("""
1. TRANSFORM preprocesses images — ToTensor scales to 0-1, Normalize centers around 0
2. Dataset wraps raw data — index it like a list, each item is (image, label)
3. BATCH: small chunk of data processed together (64 images at a time)
4. EPOCH: one full pass through the entire dataset
5. DataLoader handles batching, shuffling, and parallel loading automatically
6. Each batch has shape (batch_size, channels, height, width) = (64, 3, 32, 32)
7. shuffle=True for training (randomizes order), shuffle=False for test

Next up (Step 5): Build a CNN to classify CIFAR-10 images.
""")
