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
print(f"\ntorch.zeros(3, 4):\n{zeros}")

ones = torch.ones(2, 3)         # 2x3 of ones
print(f"\ntorch.ones(2, 3):\n{ones}")

rand = torch.randn(2, 3)        # 2x3 of random numbers (normal distribution)
print(f"\ntorch.randn(2, 3):\n{rand}")

# Shape matters — this is something you'll deal with constantly
print(f"\nrand.shape: {rand.shape}")
reshaped_3x2 = rand.reshape(3, 2)
print(f"rand.reshape(3, 2):\n{reshaped_3x2}")
reshaped_6 = rand.reshape(6)
print(f"rand.reshape(6): {reshaped_6}")

# Indexing — same as numpy/Python lists
matrix = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
print(f"\nmatrix:\n{matrix}")
print(f"matrix[0] (row 0): {matrix[0]}")
print(f"matrix[1, 2] (row 1, col 2): {matrix[1, 2]}")   # = 6
print(f"matrix[:, 1] (all rows, col 1): {matrix[:, 1]}")

# dtype — tensors have a data type (float32 is the default for ML)
print(f"\na.dtype: {a.dtype}")        # float32
print(f"matrix.dtype: {matrix.dtype}")  # int64 (created from ints)
a_int = torch.tensor([1, 2, 3])
print(f"torch.tensor([1, 2, 3]).dtype: {a_int.dtype}")   # int64
a_float = torch.tensor([1.0, 2.0, 3.0])
print(f"torch.tensor([1.0, 2.0, 3.0]).dtype: {a_float.dtype}")  # float32

# Converting dtype
matrix_float = matrix.float()
print(f"\nmatrix.float():\n{matrix_float}")
print(f"matrix.float().dtype: {matrix_float.dtype}")

# ============================================================================
# PART 2: GPU — Moving tensors to your RTX 4080
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: GPU")
print("=" * 60)

# Check what's available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"Using device: {device}")
if device == "cuda":
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

# Moving tensors to GPU — this is all you need to do
x_cpu = torch.randn(3, 3)
print(f"\nx_cpu:\n{x_cpu}")
print(f"x_cpu.device: {x_cpu.device}")

x_gpu = x_cpu.to(device)
print(f"\nx_gpu (after .to(device)):\n{x_gpu}")
print(f"x_gpu.device: {x_gpu.device}")

# Shorthand: create directly on GPU
y_gpu = torch.randn(3, 3, device=device)
print(f"\ny_gpu (created directly on GPU with device=device):\n{y_gpu}")
print(f"y_gpu.device: {y_gpu.device}")

# Math on GPU — same syntax, just faster for big tensors
result = x_gpu @ y_gpu
print(f"\nx_gpu @ y_gpu (matrix multiply on GPU):\n{result}")
print(f"result.device: {result.device}")

# Moving back to CPU (needed to convert to numpy, for example)
result_cpu = result.cpu()
print(f"\nresult.cpu().device: {result_cpu.device}")

# IMPORTANT: can't mix CPU and GPU tensors
# This would crash: x_cpu + x_gpu
# You need: x_cpu.to(device) + x_gpu, or x_gpu.cpu() + x_cpu

# Why GPU matters: let's see the speed difference
import time

size = 4000  # big matrices

# CPU timing
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
print(f"\nBenchmark tensors: a_cpu.shape={a_cpu.shape}, b_cpu.shape={b_cpu.shape}")

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"c_cpu = a_cpu @ b_cpu (CPU): shape={c_cpu.shape}")

# GPU timing
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)
print(f"a_gpu.device: {a_gpu.device}, b_gpu.device: {b_gpu.device}")
torch.cuda.synchronize()  # make sure transfer is done
start = time.time()
c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()  # make sure computation is done
gpu_time = time.time() - start
print(f"c_gpu = a_gpu @ b_gpu (GPU): shape={c_gpu.shape}, device={c_gpu.device}")

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
print(f"\nx = torch.tensor(3.0, requires_grad=True)")
print(f"x = {x}")
print(f"x.requires_grad: {x.requires_grad}")

y = x ** 2
print(f"\ny = x ** 2 = {y}")
print(f"y.grad_fn: {y.grad_fn}")   # shows PyTorch is tracking how y was made

y.backward()  # compute the gradient (derivative)
print(f"\nAfter y.backward():")
print(f"x.grad (dy/dx = 2*3 = 6): {x.grad}")

# More realistic example: a tiny "model" with weights
# Imagine we want to learn: y = w*x + b (a line)

# Our "training data" — we know the answer is y = 3x + 1
x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([4.0, 7.0, 10.0, 13.0])  # 3*x + 1
print(f"\nx_data = {x_data}")
print(f"y_true = {y_true}   (target: y = 3x + 1)")

# Our model's parameters — start with random guesses
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
print(f"\nInitial w = {w.item():.2f} (target: 3.0)")
print(f"Initial b = {b.item():.2f} (target: 1.0)")

# Training loop — this is the core of how all neural nets learn
learning_rate = 0.01
print(f"\nlearning_rate = {learning_rate}")
print(f"Training for 200 steps...\n")

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
print(f"\ntorch.linspace(-3, 3, 100): 100 evenly spaced points from -3 to 3")
print(f"x_data.shape after .unsqueeze(1): {x_data.shape}  (100 rows, 1 column)")
print(f"x_data[:5]: {x_data[:5].squeeze()}")   # first 5 values

y_true = torch.sin(x_data)
print(f"\ny_true = torch.sin(x_data)")
print(f"y_true.shape: {y_true.shape}")
print(f"y_true[:5]: {y_true[:5].squeeze()}")   # first 5 sin values

# A tiny network: input -> 32 hidden neurons -> output
# Random starting weights
w1 = torch.randn(1, 32, device=device, requires_grad=True)    # input to hidden
b1 = torch.zeros(32, device=device, requires_grad=True)
w2 = torch.randn(32, 1, device=device, requires_grad=True)    # hidden to output
b2 = torch.zeros(1, device=device, requires_grad=True)

print(f"\nNetwork weights:")
print(f"  w1.shape: {w1.shape}  (1 input → 32 hidden)")
print(f"  b1.shape: {b1.shape}  (bias for each hidden neuron)")
print(f"  w2.shape: {w2.shape}  (32 hidden → 1 output)")
print(f"  b2.shape: {b2.shape}  (bias for the output)")

learning_rate = 0.01
print(f"\nlearning_rate = {learning_rate}")
print(f"Training for 2000 steps...\n")

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

# Show what the forward pass looks like at inference time
print(f"\n--- Forward pass breakdown for input x=1.0 ---")
with torch.no_grad():
    test_x = torch.tensor([[1.0]], device=device)
    print(f"test_x: {test_x}  shape: {test_x.shape}")

    pre_activation = test_x @ w1 + b1
    print(f"(test_x @ w1 + b1).shape: {pre_activation.shape}  (32 values, one per hidden neuron)")
    print(f"(test_x @ w1 + b1)[:5]: {pre_activation[0, :5]}")  # first 5

    test_hidden = pre_activation.relu()
    print(f"After relu — negatives zeroed out:")
    print(f"  test_hidden[:5]: {test_hidden[0, :5]}")

    test_pred = test_hidden @ w2 + b2
    print(f"\ntest_pred (network output): {test_pred.item():.4f}")
    print(f"Actual sin(1.0):            {torch.sin(test_x).item():.4f}")

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
