"""
Step 5: CNN — Convolutional Neural Network for Image Classification
====================================================================
We use the CIFAR-10 dataset from Step 4 and build a CNN to classify
images into 10 categories. This is where everything comes together:
dataset → dataloader → model → training loop → evaluation.

Usage:
    python step05_cnn.py
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

os.makedirs("output/step05", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================================
# PART 1: DATA — same as Step 4, condensed
# ============================================================================

print("\n" + "=" * 60)
print("PART 1: DATA")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4)

class_names = train_dataset.classes
print(f"\ntrain samples: {len(train_dataset)}, test samples: {len(test_dataset)}")
print(f"batches per epoch: {len(train_loader)}")
print(f"classes: {class_names}")

# ============================================================================
# PART 2: THE MODEL — a CNN with two convolutional layers
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: MODEL")
print("=" * 60)

# ARCHITECTURE OVERVIEW:
#
# input image: (batch, 3, 32, 32)
#
# [Conv layer 1]  3 channels in → 32 feature maps, kernel 3×3, padding 1
#                 shape stays (batch, 32, 32, 32)  ← padding keeps size
# [ReLU]
# [MaxPool 2×2]   halves spatial size → (batch, 32, 16, 16)
#
# [Conv layer 2]  32 channels in → 64 feature maps, kernel 3×3, padding 1
#                 shape stays (batch, 64, 16, 16)
# [ReLU]
# [MaxPool 2×2]   halves again → (batch, 64, 8, 8)
#
# [Flatten]       (batch, 64, 8, 8) → (batch, 64*8*8) = (batch, 4096)
#
# [Linear 4096→256]  fully connected layer
# [ReLU]
# [Linear 256→10]    one score per class
#
# The final 10 numbers are called LOGITS — raw scores, one per class.
# The class with the highest score is the prediction.
# (No softmax here — CrossEntropyLoss applies it internally)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # CONVOLUTIONAL LAYERS — learn to detect visual features
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # MAXPOOL — reduces spatial size by half, keeps the strongest activations
        # A 2×2 maxpool looks at each 2×2 region and keeps only the largest value.
        # This makes the network less sensitive to exact position of features.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FULLY CONNECTED LAYERS — combine features to make final classification
        # 64 channels × 8 × 8 spatial = 4096 values after two maxpools on 32×32 input
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)   # 10 output scores, one per class

    def forward(self, x):
        # x shape: (batch, 3, 32, 32)

        x = self.conv1(x)           # (batch, 32, 32, 32)  — padding=1 keeps size
        x = x.relu()
        x = self.pool(x)            # (batch, 32, 16, 16)  — halved by maxpool

        x = self.conv2(x)           # (batch, 64, 16, 16)
        x = x.relu()
        x = self.pool(x)            # (batch, 64, 8, 8)    — halved again

        x = x.flatten(start_dim=1) # (batch, 4096)  — flatten everything except batch
        x = self.fc1(x).relu()      # (batch, 256)
        x = self.fc2(x)             # (batch, 10)   — raw scores (logits)

        return x

model = SimpleCNN().to(device)
print(f"\nmodel:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\ntotal parameters: {total_params:,}")

# Verify shapes by doing one forward pass with a dummy batch
dummy = torch.zeros(1, 3, 32, 32, device=device)
print(f"\nShape trace (batch size=1):")
print(f"  input:          {dummy.shape}")
with torch.no_grad():
    x = model.conv1(dummy).relu()
    print(f"  after conv1+relu: {x.shape}")
    x = model.pool(x)
    print(f"  after pool:       {x.shape}")
    x = model.conv2(x).relu()
    print(f"  after conv2+relu: {x.shape}")
    x = model.pool(x)
    print(f"  after pool:       {x.shape}")
    x = x.flatten(start_dim=1)
    print(f"  after flatten:    {x.shape}")
    x = model.fc1(x).relu()
    print(f"  after fc1+relu:   {x.shape}")
    x = model.fc2(x)
    print(f"  after fc2:        {x.shape}  ← 10 scores (logits)")

# ============================================================================
# PART 3: LOSS AND OPTIMIZER
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: LOSS AND OPTIMIZER")
print("=" * 60)

# CrossEntropyLoss — the standard loss for classification problems.
# Takes raw logits (10 scores) + correct label index, returns one loss number.
# Internally applies softmax to turn scores into probabilities, then penalizes
# the probability assigned to the wrong class.
loss_fn = nn.CrossEntropyLoss()
print(f"\nloss_fn: {loss_fn}")

# Adam optimizer — adaptive learning rate, works better than plain SGD here
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"optimizer: Adam, lr=0.001")

# ============================================================================
# PART 4: TRAINING LOOP
# ============================================================================

print("\n" + "=" * 60)
print("PART 4: TRAINING")
print("=" * 60)

num_epochs = 10
print(f"\nTraining for {num_epochs} epochs...")
print(f"{'epoch':>6}  {'train_loss':>10}  {'train_acc':>10}  {'test_acc':>10}")
print("-" * 45)

for epoch in range(num_epochs):

    # --- TRAINING PHASE ---
    model.train()   # set to training mode (opposite of model.eval())
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)             # (batch, 10) raw scores
        loss = loss_fn(logits, labels)     # compare scores to correct labels

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        predicted = logits.argmax(dim=1)   # class with highest score = prediction
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total

    # --- EVALUATION PHASE ---
    # Run on test set to see how well the model generalises to unseen images
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"{epoch+1:>6}  {train_loss:>10.4f}  {train_acc:>10.2%}  {test_acc:>10.2%}")

# ============================================================================
# PART 5: INSPECTING PREDICTIONS
# ============================================================================

print("\n" + "=" * 60)
print("PART 5: PREDICTIONS ON SAMPLE IMAGES")
print("=" * 60)

model.eval()
images, labels = next(iter(test_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    logits = model(images)                 # (64, 10)
    probs = logits.softmax(dim=1)          # convert scores to probabilities
    predicted = logits.argmax(dim=1)       # predicted class index

print(f"\n{'sample':>8}  {'true':>10}  {'predicted':>12}  {'confidence':>12}  {'correct':>8}")
print("-" * 58)
for i in range(16):
    true_name = class_names[labels[i].item()]
    pred_name = class_names[predicted[i].item()]
    confidence = probs[i, predicted[i]].item()
    correct_str = "✓" if predicted[i] == labels[i] else "✗"
    print(f"{i:>8}  {true_name:>10}  {pred_name:>12}  {confidence:>11.1%}  {correct_str:>8}")

# ============================================================================
# PART 6: SAVE THE MODEL
# ============================================================================

print("\n" + "=" * 60)
print("PART 6: SAVE")
print("=" * 60)

torch.save(model.state_dict(), "output/step05/cifar10_cnn.pth")
print(f"\nModel saved to: output/step05/cifar10_cnn.pth")

print("\n" + "=" * 60)
print("DONE! Key takeaways:")
print("=" * 60)
print("""
1. Conv2d detects local patterns — filters slide across the image
2. MaxPool reduces spatial size, keeping the strongest activations
3. After conv layers, flatten to a 1D vector before fully connected layers
4. CrossEntropyLoss is the standard loss for classification (not MSE)
5. logits.argmax(dim=1) gives the predicted class index
6. model.train() during training, model.eval() during evaluation
7. Track both train accuracy AND test accuracy — gap between them
   indicates overfitting (memorizing training data, failing on new data)

Next up (Step 6): Write a proper training loop with more controls,
learning rate scheduling, and better evaluation metrics.
""")
