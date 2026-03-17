"""
Step 7 (Overfitting Fix): Same pipeline as step07_evaluate.py with three targeted fixes:
  1. Data augmentation  — training images look slightly different each epoch
  2. Dropout            — randomly disables neurons during training
  3. Weight decay       — penalizes large weights via the optimizer

Compare train_acc vs test_acc here against step07_evaluate.py to see the gap shrink.

Usage:
    python step07_fix_overfitting.py
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================================
# PART 1: DATA
# ============================================================================

print("\n" + "=" * 60)
print("PART 1: DATA")
print("=" * 60)

# FIX 1: DATA AUGMENTATION
# The original script fed the model the exact same 50k images every epoch.
# The model eventually memorized them. Augmentation randomly transforms each
# image before it's shown, so the model never sees the exact same thing twice.
#
# RandomHorizontalFlip — flip the image left-right 50% of the time.
#   A cat facing left is still a cat. This doubles effective image variety.
#
# RandomCrop(32, padding=4) — pad the image by 4 pixels on each side, then
#   randomly crop back to 32x32. Forces the model to recognize objects that
#   are not perfectly centered. One of the most effective augmentations for CIFAR.
#
# These are only applied to the TRAINING set.
# The test set uses plain ToTensor + Normalize — we evaluate on clean images.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4)

class_names = train_dataset.classes
print(f"\ntrain: {len(train_dataset)} samples, test: {len(test_dataset)} samples")
print("train transform: RandomHorizontalFlip + RandomCrop + Normalize")
print("test transform:  Normalize only")

# ============================================================================
# PART 2: MODEL
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: MODEL")
print("=" * 60)

# FIX 2: DROPOUT
# Dropout randomly zeroes out a fraction of neuron outputs during training.
# p=0.5 means each neuron has a 50% chance of being silenced each forward pass.
#
# Why it works: the network can't rely on any single neuron always being present,
# so it's forced to learn redundant, distributed representations — which generalize
# better to unseen data.
#
# Dropout is only active during training (model.train()).
# During evaluation (model.eval()) it is automatically disabled — all neurons fire.
# That's why switching between model.train() and model.eval() in the loop matters.
#
# Placement: after the first fully-connected layer, before the final classifier.
# Putting it after conv layers is possible but less common for small CNNs.
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(p=0.5)   # <-- added: 50% dropout before classifier
        self.fc2     = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = x.flatten(start_dim=1)
        x = self.fc1(x).relu()
        x = self.dropout(x)                # <-- added: applied after fc1, before fc2
        return self.fc2(x)

model = SimpleCNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nmodel: SimpleCNN with Dropout(0.5) — {total_params:,} parameters")

# ============================================================================
# PART 3: LOSS, OPTIMIZER, SCHEDULER
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: LOSS, OPTIMIZER, SCHEDULER")
print("=" * 60)

loss_fn = nn.CrossEntropyLoss()

# FIX 3: WEIGHT DECAY
# Weight decay adds a penalty to the loss for having large weights.
# After each update, weights are nudged slightly toward zero.
#
# Why it works: large weights mean the model is making very confident, rigid
# decisions based on specific training patterns. Keeping weights small forces
# smoother, more general decision boundaries that work on unseen data.
#
# weight_decay=1e-4 is a standard starting value for CIFAR-10.
# It's small enough not to hurt training, but enough to regularize.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # <-- added weight_decay

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

print(f"\nloss_fn:      CrossEntropyLoss")
print(f"optimizer:    Adam, lr=0.001, weight_decay=1e-4")
print(f"scheduler:    ReduceLROnPlateau — halve lr after 3 epochs of no improvement")

# ============================================================================
# PART 4: TRAINING LOOP WITH CHECKPOINTING
# ============================================================================

print("\n" + "=" * 60)
print("PART 4: TRAINING WITH CHECKPOINTING")
print("=" * 60)

best_acc = 0.0
checkpoint_path = "cifar10_best_fix_overfitting.pth"

num_epochs = 20
history = {"train_loss": [], "train_acc": [], "test_acc": [], "lr": []}

print(f"\nTraining for {num_epochs} epochs...")
print(f"{'epoch':>6}  {'lr':>8}  {'train_loss':>10}  {'train_acc':>10}  {'test_acc':>10}  {'saved':>6}")
print("-" * 62)

for epoch in range(num_epochs):

    # --- TRAINING ---
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total

    # --- EVALUATION ---
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)

    test_acc = correct / total

    # --- SCHEDULER STEP ---
    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(test_acc)
    new_lr = optimizer.param_groups[0]["lr"]
    if new_lr != prev_lr:
        print(f"  [scheduler] lr reduced: {prev_lr:.6f} → {new_lr:.6f}")

    # --- CHECKPOINTING ---
    saved = ""
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), checkpoint_path)
        saved = "✓"

    # --- LOGGING ---
    current_lr = optimizer.param_groups[0]["lr"]
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)
    history["lr"].append(current_lr)

    print(f"{epoch+1:>6}  {current_lr:>8.6f}  {train_loss:>10.4f}  {train_acc:>10.2%}  {test_acc:>10.2%}  {saved:>6}")

print(f"\nBest test accuracy: {best_acc:.2%}  (saved to {checkpoint_path})")

# ============================================================================
# PART 5: VISUALIZE TRAINING CURVES
# ============================================================================

print("\n" + "=" * 60)
print("PART 5: TRAINING CURVES")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs = range(1, num_epochs + 1)

ax1.plot(epochs, history["train_loss"], label="train loss")
ax1.set_title("Training Loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()

ax2.plot(epochs, history["train_acc"], label="train acc")
ax2.plot(epochs, history["test_acc"],  label="test acc")
ax2.set_title("Accuracy")
ax2.set_xlabel("epoch")
ax2.set_ylabel("accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves_fix_overfitting.png", dpi=150)
plt.close()
print(f"\nSaved training curves to: training_curves_fix_overfitting.png")

with open("training_curves_fix_overfitting.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "test_acc", "lr"])
    for i in range(num_epochs):
        writer.writerow([i + 1, history["train_loss"][i], history["train_acc"][i], history["test_acc"][i], history["lr"][i]])
print(f"Saved training curves data to: training_curves_fix_overfitting.csv")

# ============================================================================
# PART 6: PER-CLASS ACCURACY
# ============================================================================

print("\n" + "=" * 60)
print("PART 6: PER-CLASS ACCURACY")
print("=" * 60)

model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
model.eval()
print(f"\nLoaded best model from: {checkpoint_path}")

class_correct = [0] * 10
class_total   = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).argmax(dim=1)
        for label, pred in zip(labels, predicted):
            class_total[label]   += 1
            class_correct[label] += (pred == label).item()

print(f"\n{'class':>12}  {'correct':>8}  {'total':>8}  {'accuracy':>10}")
print("-" * 44)
for i, name in enumerate(class_names):
    acc = class_correct[i] / class_total[i]
    print(f"{name:>12}  {class_correct[i]:>8}  {class_total[i]:>8}  {acc:>10.2%}")

overall = sum(class_correct) / sum(class_total)
print(f"\nOverall test accuracy: {overall:.2%}")

# ============================================================================
# PART 7: CONFUSION MATRIX
# ============================================================================

print("\n" + "=" * 60)
print("PART 7: CONFUSION MATRIX")
print("=" * 60)

confusion = torch.zeros(10, 10, dtype=torch.int)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).argmax(dim=1)
        for true, pred in zip(labels, predicted):
            confusion[true][pred] += 1

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(confusion.numpy(), cmap="Blues")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")

for i in range(10):
    for j in range(10):
        ax.text(j, i, confusion[i][j].item(), ha="center", va="center", fontsize=8,
                color="white" if confusion[i][j] > confusion.max() * 0.5 else "black")

plt.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix_fix_overfitting.png", dpi=150)
plt.close()
print(f"\nSaved confusion matrix to: confusion_matrix_fix_overfitting.png")

with open("confusion_matrix_fix_overfitting.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["true \\ pred"] + class_names)
    for i, name in enumerate(class_names):
        writer.writerow([name] + confusion[i].tolist())
print(f"Saved confusion matrix data to: confusion_matrix_fix_overfitting.csv")

# ============================================================================
# PART 8: SAMPLE PREDICTIONS
# ============================================================================

print("\n" + "=" * 60)
print("PART 8: SAMPLE PREDICTIONS")
print("=" * 60)

viz_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False,
    transform=transforms.ToTensor()
)

model.eval()
fig, axes = plt.subplots(4, 8, figsize=(14, 8))
fig.suptitle("Sample predictions (green=correct, red=wrong)", fontsize=12)

sample_rows = []
for i, ax in enumerate(axes.flat):
    img, true_label = viz_dataset[i]
    norm_img, _ = test_dataset[i]
    with torch.no_grad():
        logits = model(norm_img.unsqueeze(0).to(device))
        pred_label = logits.argmax(dim=1).item()

    ax.imshow(img.permute(1, 2, 0))
    correct = pred_label == true_label
    color = "green" if correct else "red"
    ax.set_title(f"{class_names[pred_label]}", fontsize=7, color=color)
    ax.axis("off")
    sample_rows.append([i, class_names[true_label], class_names[pred_label], "correct" if correct else "wrong"])

plt.tight_layout()
plt.savefig("sample_predictions_fix_overfitting.png", dpi=150)
plt.close()
print(f"\nSaved sample predictions to: sample_predictions_fix_overfitting.png")

with open("sample_predictions_fix_overfitting.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "true_label", "predicted_label", "result"])
    writer.writerows(sample_rows)
print(f"Saved sample predictions data to: sample_predictions_fix_overfitting.csv")

print("\n" + "=" * 60)
print("DONE! What changed vs step07_evaluate.py:")
print("=" * 60)
print("""
1. Data augmentation (RandomHorizontalFlip + RandomCrop)
   — model sees different versions of each image every epoch
   — most impactful fix for small datasets

2. Dropout(0.5) after fc1
   — 50% of neurons randomly silenced each forward pass during training
   — forces redundant learning, reduces reliance on memorized patterns

3. Weight decay=1e-4 in Adam
   — penalizes large weights, keeps decision boundaries smooth

Expected result: train_acc comes down slightly, test_acc goes up,
and the gap between them shrinks — that's less overfitting.
""")
