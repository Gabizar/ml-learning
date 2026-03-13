"""
Step 2: PyTorch Basics — Tensors, GPU, and Autograd
====================================================
Run this on your GPU laptop. Work through each section, read the comments,
and check the output. Modify things, break things, re-run — that's how you learn.

Usage:
    python step02_pytorch_basics.py
"""

import torch

# ============================================================================
# PART 1: TENSORS — PyTorch's version of arrays/matrices
# ============================================================================

print("=" * 60)
print("PART 1: TENSORS")
print("=" * 60)

# Creating tensors (like numpy arrays, but can run on GPU)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
print(f"\na = {a}")
print(f"b = {b}")

# Basic math — works element-wise, just like numpy
print(f"\na + b = {a + b}")
print(f"a * b = {a * b}")       # element-wise multiply
print(f"a @ b = {a @ b}")       # dot product (1*4 + 2*5 + 3*6 = 32)

# Making bigger tensors
zeros = torch.zeros(3, 4)       # 3 rows, 4 columns of zeros
ones = torch.ones(2, 3)         # 2x3 of ones
rand = torch.randn(2, 3)        # 2x3 of random numbers (normal distribution)
print(f"\nRandom 2x3 tensor:\n{rand}")

# Shape matters — this is something you'll deal with constantly
print(f"\nrand shape: {rand.shape}")
print(f"rand reshaped to 3x2:\n{rand.reshape(3, 2)}")
print(f"rand reshaped to 6:\n{rand.reshape(6)}")

# Indexing — same as numpy/Python lists
matrix = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
print(f"\nmatrix:\n{matrix}")
print(f"Row 0: {matrix[0]}")
print(f"Element [1,2]: {matrix[1, 2]}")   # row 1, col 2 = 6
print(f"Column 1: {matrix[:, 1]}")         # all rows, col 1

# ============================================================================
# PART 2: GPU — Moving tensors to your RTX 4080
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: GPU")
print("=" * 60)

# Check what's available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Moving tensors to GPU — this is all you need to do
x_cpu = torch.randn(3, 3)
x_gpu = x_cpu.to(device)
print(f"\nx_cpu device: {x_cpu.device}")
print(f"x_gpu device: {x_gpu.device}")

# Shorthand: create directly on GPU
y_gpu = torch.randn(3, 3, device=device)

# Math on GPU — same syntax, just faster for big tensors
result = x_gpu @ y_gpu
print(f"GPU matrix multiply result:\n{result}")

# IMPORTANT: can't mix CPU and GPU tensors
# This would crash: x_cpu + x_gpu
# You need: x_cpu.to(device) + x_gpu, or x_gpu.cpu() + x_cpu

# Why GPU matters: let's see the speed difference
import time

size = 4000  # big matrices

# CPU timing
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start

# GPU timing
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)
torch.cuda.synchronize()  # make sure transfer is done
start = time.time()
c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()  # make sure computation is done
gpu_time = time.time() - start

print(f"\n{size}x{size} matrix multiply:")
print(f"  CPU: {cpu_time:.4f}s")
print(f"  GPU: {gpu_time:.4f}s")
print(f"  Speedup: {cpu_time / gpu_time:.1f}x")

# ============================================================================
# PART 3: AUTOGRAD — Automatic differentiation (the magic behind training)
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: AUTOGRAD")
print("=" * 60)

# The big idea: PyTorch can automatically compute gradients (derivatives).
# This is how neural nets learn — they need gradients to know which direction
# to adjust their weights.

# Simple example: y = x^2, derivative is 2x
x = torch.tensor(3.0, requires_grad=True)  # tell PyTorch to track this
y = x ** 2
print(f"\nx = {x}")
print(f"y = x^2 = {y}")

y.backward()  # compute the gradient (derivative)
print(f"dy/dx = 2x = {x.grad}")  # should be 6.0 (2 * 3)

# More realistic example: a tiny "model" with weights
# Imagine we want to learn: y = w*x + b (a line)

# Our "training data" — we know the answer is y = 3x + 1
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([4.0, 7.0, 10.0, 13.0])  # 3*x + 1

# Our model's parameters — start with random guesses
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

print(f"\nLearning y = w*x + b (true answer: w=3, b=1)")
print(f"Starting with w={w.item():.2f}, b={b.item():.2f}")

# Training loop — this is the core of how all neural nets learn
learning_rate = 0.01

for step in range(200):
    # Forward pass: make a prediction
    y_pred = w * x_data + b

    # Loss: how wrong are we? (mean squared error)
    loss = ((y_pred - y_true) ** 2).mean()

    # Backward pass: compute gradients
    loss.backward()

    # Update weights (gradient descent) — move in the direction that reduces loss
    with torch.no_grad():  # don't track these operations
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero the gradients (they accumulate by default)
    w.grad.zero_()
    b.grad.zero_()

    if step % 40 == 0:
        print(f"  Step {step:3d}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\nFinal: w={w.item():.4f}, b={b.item():.4f}")
print(f"Target: w=3.0000, b=1.0000")

# ============================================================================
# PART 4: PUTTING IT TOGETHER — A tiny neural network from scratch
# ============================================================================

print("\n" + "=" * 60)
print("PART 4: MINI NEURAL NETWORK")
print("=" * 60)

# Let's learn a non-linear function: y = sin(x)
# A straight line can't fit this — we need a neural network!

x_data = torch.linspace(-3, 3, 100, device=device).unsqueeze(1)  # 100 points, shape [100,1]
y_true = torch.sin(x_data)

# A tiny network: input -> 32 hidden neurons -> output
# Random starting weights
w1 = torch.randn(1, 32, device=device, requires_grad=True)    # input to hidden
b1 = torch.zeros(32, device=device, requires_grad=True)
w2 = torch.randn(32, 1, device=device, requires_grad=True)    # hidden to output
b2 = torch.zeros(1, device=device, requires_grad=True)

learning_rate = 0.01

for step in range(2000):
    # Forward pass
    hidden = (x_data @ w1 + b1).relu()   # relu = max(0, x) — the "activation"
    y_pred = hidden @ w2 + b2

    # Loss
    loss = ((y_pred - y_true) ** 2).mean()

    # Backward pass
    loss.backward()

    # Update weights
    with torch.no_grad():
        for param in [w1, b1, w2, b2]:
            param -= learning_rate * param.grad
            param.grad.zero_()

    if step % 400 == 0:
        print(f"  Step {step:4d}: loss={loss.item():.6f}")

print(f"\nFinal loss: {loss.item():.6f}")
print("(Close to 0 = the network learned to approximate sin(x)!)")

# Quick check — how close are we?
with torch.no_grad():
    test_x = torch.tensor([[1.0]], device=device)
    test_hidden = (test_x @ w1 + b1).relu()
    test_pred = test_hidden @ w2 + b2
    print(f"\nPrediction for sin(1.0): {test_pred.item():.4f}")
    print(f"Actual sin(1.0):         {torch.sin(test_x).item():.4f}")

print("\n" + "=" * 60)
print("DONE! Key takeaways:")
print("=" * 60)
print("""
1. TENSORS are like numpy arrays — they hold your data and model weights
2. GPU makes big matrix math fast — just use .to("cuda")
3. AUTOGRAD tracks operations and computes gradients automatically
4. TRAINING = forward pass → loss → backward pass → update weights → repeat
5. A neural net is just: input × weights + bias, with activations in between

Next up (Step 3): We'll use PyTorch's nn.Module to build networks
properly instead of managing raw tensors by hand.
""")
