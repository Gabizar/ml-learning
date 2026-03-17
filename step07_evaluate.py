"""
Step 7: Train, Evaluate, and Visualize
=======================================
Builds on Step 5 with:
- Learning rate scheduling (reduce LR when progress stalls)
- Model checkpointing (save best model during training, not just at the end)
- Per-class accuracy
- Confusion matrix
- Sample prediction visualization

Usage:
    python step07_evaluate.py
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
import os

os.makedirs("output/step07", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================================
# PART 1: DATA
# ============================================================================

print("\n" + "=" * 60)
print("PART 1: DATA")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4)

class_names = train_dataset.classes
print(f"\ntrain: {len(train_dataset)} samples, test: {len(test_dataset)} samples")

# ============================================================================
# PART 2: MODEL — same CNN as Step 5
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: MODEL")
print("=" * 60)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = x.flatten(start_dim=1)
        x = self.fc1(x).relu()
        return self.fc2(x)

model = SimpleCNN().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nmodel: SimpleCNN — {total_params:,} parameters")

# ============================================================================
# PART 3: LOSS, OPTIMIZER, SCHEDULER
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: LOSS, OPTIMIZER, SCHEDULER")
print("=" * 60)

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ReduceLROnPlateau — watches a metric (here: test accuracy) and reduces
# the learning rate when it stops improving.
#
# mode="max"    → we're maximizing accuracy (use "min" for loss)
# factor=0.5    → multiply lr by 0.5 when triggered (0.001 → 0.0005)
# patience=3    → wait 3 epochs of no improvement before reducing
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

print(f"\nloss_fn:   CrossEntropyLoss")
print(f"optimizer: Adam, initial lr=0.001")
print(f"scheduler: ReduceLROnPlateau — halve lr after 3 epochs of no improvement")

# ============================================================================
# PART 4: TRAINING LOOP WITH CHECKPOINTING
# ============================================================================

print("\n" + "=" * 60)
print("PART 4: TRAINING WITH CHECKPOINTING")
print("=" * 60)

# CHECKPOINTING: save the model whenever test accuracy improves.
# This way we always keep the best version, not just the final one.
# If accuracy peaks at epoch 8 then drops (overfitting), we still have epoch 8.

best_acc = 0.0
checkpoint_path = "output/step07/cifar10_best.pth"

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
    # Pass test_acc so the scheduler knows whether we improved
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

# Plot loss and accuracy over epochs to see how training progressed.
# GAP between train_acc and test_acc = overfitting indicator.
# If train_acc is much higher than test_acc, the model memorized training data.

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
plt.savefig("output/step07/training_curves.png", dpi=150)
plt.close()
print(f"\nSaved training curves to: output/step07/training_curves.png")

with open("output/step07/training_curves.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "test_acc", "lr"])
    for i in range(num_epochs):
        writer.writerow([i + 1, history["train_loss"][i], history["train_acc"][i], history["test_acc"][i], history["lr"][i]])
print(f"Saved training curves data to: output/step07/training_curves.csv")

# ============================================================================
# PART 6: PER-CLASS ACCURACY
# ============================================================================

print("\n" + "=" * 60)
print("PART 6: PER-CLASS ACCURACY")
print("=" * 60)

# Load the best checkpoint for evaluation
model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
model.eval()
print(f"\nLoaded best model from: {checkpoint_path}")

# Track correct and total per class separately
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

# CONFUSION MATRIX: a 10×10 grid where:
#   row    = true class
#   column = predicted class
#   cell   = how many times true=row was predicted as column
#
# Perfect model: only the diagonal has values (true=predicted every time).
# Off-diagonal values reveal which classes the model confuses with each other.

confusion = torch.zeros(10, 10, dtype=torch.int)

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).argmax(dim=1)
        for true, pred in zip(labels, predicted):
            confusion[true][pred] += 1

# Plot as a heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(confusion.numpy(), cmap="Blues")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")

# Annotate each cell with its count
for i in range(10):
    for j in range(10):
        ax.text(j, i, confusion[i][j].item(), ha="center", va="center", fontsize=8,
                color="white" if confusion[i][j] > confusion.max() * 0.5 else "black")

plt.colorbar(im)
plt.tight_layout()
plt.savefig("output/step07/confusion_matrix.png", dpi=150)
plt.close()
print(f"\nSaved confusion matrix to: output/step07/confusion_matrix.png")
print("Diagonal = correct predictions. Off-diagonal = what it confused with what.")

with open("output/step07/confusion_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["true \\ pred"] + class_names)
    for i, name in enumerate(class_names):
        writer.writerow([name] + confusion[i].tolist())
print(f"Saved confusion matrix data to: output/step07/confusion_matrix.csv")

# ============================================================================
# PART 8: SAMPLE PREDICTIONS
# ============================================================================

print("\n" + "=" * 60)
print("PART 8: SAMPLE PREDICTIONS")
print("=" * 60)

# Visualize actual test images with true and predicted labels.
# Use the unnormalized dataset for display (same reason as Step 4).

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

    # Run through model — need normalized version for the model
    norm_img, _ = test_dataset[i]
    with torch.no_grad():
        logits = model(norm_img.unsqueeze(0).to(device))  # add batch dim
        pred_label = logits.argmax(dim=1).item()

    ax.imshow(img.permute(1, 2, 0))
    correct = pred_label == true_label
    color = "green" if correct else "red"
    ax.set_title(f"{class_names[pred_label]}", fontsize=7, color=color)
    ax.axis("off")
    sample_rows.append([i, class_names[true_label], class_names[pred_label], "correct" if correct else "wrong"])

plt.tight_layout()
plt.savefig("output/step07/sample_predictions.png", dpi=150)
plt.close()
print(f"\nSaved sample predictions to: output/step07/sample_predictions.png")
print("Title color: green = correct, red = wrong")

with open("output/step07/sample_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "true_label", "predicted_label", "result"])
    writer.writerows(sample_rows)
print(f"Saved sample predictions data to: output/step07/sample_predictions.csv")

print("\n" + "=" * 60)
print("DONE! Key takeaways:")
print("=" * 60)
print("""
1. LR scheduling reduces the learning rate when progress stalls — helps fine-tune
2. Checkpointing saves the best model during training, not just the final one
3. Per-class accuracy reveals which classes the model struggles with
4. Confusion matrix shows which classes get confused with each other
5. Gap between train_acc and test_acc = overfitting
6. scheduler.step() is called after each epoch (not each batch) with the metric

Next up (Step 8): Fine-tune a pretrained model (ResNet) on a custom dataset.
""")
