"""
Step 3: nn.Module — Building Networks Properly
===============================================
In Step 2 Part 4 we managed weights as raw tensors by hand.
This works but doesn't scale — real networks have millions of parameters.
nn.Module is PyTorch's way to package a network cleanly.

Usage:
    python step03_nn_module.py
"""

import torch
import torch.nn as nn

# ============================================================================
# PART 1: YOUR FIRST nn.Module — rewriting Part 4 from Step 2
# ============================================================================

print("=" * 60)
print("PART 1: DEFINING A NETWORK WITH nn.Module")
print("=" * 60)

# To define a network, you create a class that inherits from nn.Module.
# INHERIT means your class gets all of nn.Module's built-in functionality
# (parameter tracking, .to(device), etc.) for free.
#
# You must define two methods:
#   __init__  — declare your layers here
#   forward   — describe how data flows through the network

class SineNet(nn.Module):
    def __init__(self):
        super().__init__()  # required — initializes nn.Module's internals
        # nn.Linear(in, out) is a fully connected layer.
        # It creates a weight matrix of shape (in, out) and a bias of shape (out,)
        # automatically — no need to declare w1, b1 etc. by hand.
        # self.layer1 = nn.Linear(1, 32)   # 1 input → 32 hidden neurons
        # self.layer2 = nn.Linear(32, 1)   # 32 hidden → 1 output
        self.layer1 = nn.Linear(1, 128)
        self.layer2 = nn.Linear(128, 1)

    def forward(self, x):
        # This is where you describe the flow of data through the network.
        # PyTorch calls this automatically when you do model(x_data).
        x = self.layer1(x)        # linear: (100,1) → (100,32)
        x = x.relu()              # relu: zeroes out negatives
        x = self.layer2(x)        # linear: (100,32) → (100,1)
        # No relu on output — sin(x) has negative values
        return x

# Create the network
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\ndevice: {device}")

model = SineNet()
print(f"\nmodel:\n{model}")
# nn.Module gives you a nice summary of the network structure for free

# Move entire network to GPU in one call — no need to move each tensor individually
model = model.to(device)
print(f"\nAfter model.to(device) — all parameters are now on: {device}")

# ============================================================================
# PART 2: INSPECTING PARAMETERS
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: INSPECTING PARAMETERS")
print("=" * 60)

# model.parameters() returns all weights and biases automatically
print("\nAll parameters in the network:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, device={param.device}")

# Count total number of trainable values
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable parameters: {total_params}")
# 1×32 (w1) + 32 (b1) + 32×1 (w2) + 1 (b2) = 97

# ============================================================================
# PART 3: THE OPTIMIZER — replaces the manual update loop
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: OPTIMIZER")
print("=" * 60)

# In Step 2, we updated weights manually:
#   for param in [w1, b1, w2, b2]:
#       param -= lr * param.grad
#       param.grad.zero_()
#
# An OPTIMIZER does this for you. You hand it the model's parameters
# and a learning rate, and it handles the update step.
#
# SGD = Stochastic Gradient Descent — the same -= lr * grad logic we did by hand.
# (There are fancier optimizers like Adam, but SGD is the simplest.)

learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(f"\noptimizer: {optimizer}")
print(f"learning_rate: {learning_rate}")

# ============================================================================
# PART 4: LOSS FUNCTION — nn also provides pre-built loss functions
# ============================================================================

print("\n" + "=" * 60)
print("PART 4: LOSS FUNCTION")
print("=" * 60)

# In Step 2, we computed MSE manually:
#   loss = ((y_pred - y_true) ** 2).mean()
#
# nn.MSELoss() does exactly the same thing, but as a reusable object.
# Using the built-in version is standard practice.

loss_fn = nn.MSELoss()
print(f"\nloss_fn: {loss_fn}")
print("(MSELoss = mean squared error — same as ((y_pred - y_true)**2).mean())")

# ============================================================================
# PART 5: TRAINING LOOP — putting it all together
# ============================================================================

print("\n" + "=" * 60)
print("PART 5: TRAINING LOOP")
print("=" * 60)

# Same task as Step 2 Part 4: learn to approximate sin(x)
x_data = torch.linspace(-3, 3, 100, device=device).unsqueeze(1)  # shape (100, 1)
y_true = torch.sin(x_data)                                        # shape (100, 1)
print(f"\nx_data.shape: {x_data.shape}")
print(f"y_true.shape: {y_true.shape}")

print(f"\nTraining for 2000 steps...\n")

for step in range(5000):
    # 1. Forward pass — call model like a function, it runs forward() internally
    y_pred = model(x_data)

    # 2. Loss
    loss = loss_fn(y_pred, y_true)

    # 3. Zero gradients BEFORE backward (optimizer owns this now)
    #    Must be done before backward, not after — otherwise gradients accumulate
    optimizer.zero_grad()

    # 4. Backward pass — same as before
    loss.backward()

    # 5. Update weights — optimizer replaces the manual param -= lr * grad loop
    optimizer.step()

    if step % 400 == 0:
        print(f"  Step {step:4d}: loss={loss.item():.6f}")

print(f"\nFinal loss: {loss.item():.6f}")
print("(Close to 0 = the network learned to approximate sin(x)!)")

# ============================================================================
# PART 6: INFERENCE — using the trained model to make predictions
# ============================================================================

print("\n" + "=" * 60)
print("PART 6: INFERENCE")
print("=" * 60)

# INFERENCE means using a trained model to make predictions (not training).
# Two important things to do during inference:
#   1. model.eval() — tells the network we're not training
#      (some layer types behave differently during training vs inference)
#   2. torch.no_grad() — disables gradient tracking, saves memory and compute

model.eval()
print(f"\nmodel.eval() called — network is in inference mode")

test_inputs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device).unsqueeze(1)
print(f"\ntest_inputs: {test_inputs.squeeze()}")

with torch.no_grad():
    predictions = model(test_inputs)

print(f"\n{'input':>8}  {'predicted':>12}  {'actual sin(x)':>14}")
print("-" * 38)
for x_val, pred in zip(test_inputs.squeeze(), predictions.squeeze()):
    actual = torch.sin(x_val).item()
    print(f"{x_val.item():>8.1f}  {pred.item():>12.4f}  {actual:>14.4f}")

# ============================================================================
# PART 7: SAVING AND LOADING A MODEL
# ============================================================================

print("\n" + "=" * 60)
print("PART 7: SAVING AND LOADING")
print("=" * 60)

# state_dict = all the learned weights and biases as a dictionary
# This is the standard way to save a trained model.

import os
os.makedirs("output/step03", exist_ok=True)

save_path = "output/step03/sine_model.pth"
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to: {save_path}")
print(f"state_dict keys: {list(model.state_dict().keys())}")

# Also save as JSON so you can inspect the weights as human-readable numbers.
# .tolist() converts a tensor to a nested Python list, which JSON can serialize.
import json
json_path = "output/step03/sine_model.json"
json_state = {key: val.tolist() for key, val in model.state_dict().items()}
with open(json_path, "w") as f:
    json.dump(json_state, f, indent=2)
print(f"Model weights also saved as JSON to: {json_path}")

# Loading: create a fresh model with the same architecture, then load weights
model2 = SineNet().to(device)
model2.load_state_dict(torch.load(save_path, weights_only=True))
model2.eval()
print(f"\nLoaded model2 from: {save_path}")

# Verify it gives the same predictions
with torch.no_grad():
    predictions2 = model2(test_inputs)

match = torch.allclose(predictions, predictions2)
print(f"predictions match original model: {match}")

print("\n" + "=" * 60)
print("DONE! Key takeaways:")
print("=" * 60)
print("""
1. nn.Module lets you define a network as a class — cleaner than raw tensors
2. nn.Linear(in, out) creates a layer with weights + bias automatically
3. forward() defines how data flows — PyTorch calls it when you do model(x)
4. model.parameters() finds all weights and biases automatically
5. Optimizer handles the weight update loop — replaces manual param -= lr * grad
6. optimizer.zero_grad() before backward(), optimizer.step() after
7. model.eval() + torch.no_grad() for inference (not training)
8. torch.save / load_state_dict for saving and loading trained models

Next up (Step 4): Load a real dataset (CIFAR-10) and build a CNN to classify images.
""")
