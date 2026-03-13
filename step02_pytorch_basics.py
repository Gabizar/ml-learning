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
# NOTE: @ on two 1D vectors = dot product = multiply pairs then sum = one number.
# @ on two 2D matrices = matrix multiply = many dot products = a matrix.
# Same operator, different behavior based on the number of dimensions.
#
# DOT PRODUCT: take two vectors of the same length, multiply pairs, sum everything.
#   a · b = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
#
# MATRIX MULTIPLY rule: (A×B) @ (B×C) → (A×C)
#   Inner dimensions must match (B), output shape is outer dimensions (A×C).
#   result[i][j] = dot(row i of left, col j of right)

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

# unsqueeze/squeeze — add or remove dimensions of size 1
# unsqueeze(1): (3,) → (3, 1)   adds a dimension at position 1
# squeeze():    (1, 1, 3) → (3,) removes all size-1 dimensions
# Why useful: matrix multiply rules are strict about shapes.
# e.g. (100,) can't multiply (1, 32), but (100, 1) can → gives (100, 32)

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
# x_gpu and y_gpu are both (3,3) — this is full matrix multiply, not dot product.
# result[i][j] = dot(row i of x_gpu, col j of y_gpu)
# result shape = (3, 3)
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
# NOTE: GPU operations are asynchronous — PyTorch sends the job to the GPU
# and immediately returns to the CPU without waiting for it to finish.
# torch.cuda.synchronize() forces the CPU to wait until the GPU is done.
# Without it, time.time() would measure almost nothing (GPU still running!).
# You only need synchronize() for benchmarking — normal training doesn't need it.
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)
print(f"a_gpu.device: {a_gpu.device}, b_gpu.device: {b_gpu.device}")
torch.cuda.synchronize()  # wait for transfer to finish before starting timer
start = time.time()
c_gpu = a_gpu @ b_gpu
torch.cuda.synchronize()  # wait for computation to finish before stopping timer
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
# NOTE: x.grad = 6 means "if x increases by a tiny amount, y increases 6x as fast".
# We check x.grad (not y.grad) because x is what we control (the weight).
# y is the output we want to minimize — gradients flow backward FROM y TO x.
#
# The gradient is only accurate for infinitely small nudges, not whole numbers:
#   x=3 → y=9, x=4 → y=16 (actual +7), but grad predicts +6 (off by 1 for nudge=1)
#   nudge=0.01: actual=0.0601, predicted=0.06 (off by only 0.0001)
# This is fine for training — learning rates are tiny (0.01), so the gradient
# just needs to point in the right direction.
#
# IMPORTANT: gradients accumulate by default. Always zero them after each step
# or they'll add up across iterations and corrupt your updates.

# More realistic example: a tiny "model" with weights
# Imagine we want to learn: y = w*x + b (a line)
# A NEURON is just: output = input × weight + bias
# - weight controls the slope (how much the input matters)
# - bias shifts the output up/down regardless of input (like the +1 in y=3x+1)
# This is y = mx + b from school. Both weight AND bias are needed because
# weight alone can't shift the line — weight×0 is always 0.

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
    # w and b are scalars, x_data is shape (4,) — broadcasting applies the
    # scalar to every element automatically.
    y_pred = w * x_data + b

    # Loss: how wrong are we? (mean squared error = MSE)
    # Step 1: y_pred - y_true  →  errors (how far off each prediction is)
    # Step 2: ** 2             →  square them (removes negatives, punishes big errors harder)
    # Step 3: .mean()          →  average into one number
    # Result: one number measuring total wrongness. Goal: drive this to 0.
    loss = ((y_pred - y_true) ** 2).mean()

    # Backward pass: compute gradients
    # We call backward() on LOSS (not y_pred) because loss is what we're minimizing.
    # y_pred alone has no sense of "good" or "bad" — loss converts prediction
    # into "how wrong am I", so that's where the backward pass starts.
    loss.backward()

    # Update weights (gradient descent) — move in the direction that reduces loss
    # We use -= (not +=) because the gradient points UPHILL (direction of increase).
    # To reduce loss, we go the opposite direction: -= lr * grad goes downhill.
    with torch.no_grad():  # don't track these operations
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero the gradients (they accumulate by default — must reset each step)
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
#
# NETWORK STRUCTURE:
#   input (1 value) → [layer 1: 32 neurons] → relu → [layer 2: 1 neuron] → output
#
# LAYER: a group of neurons that all receive the same input and each produce one number.
# NEURON: computes output = (input × weight) + bias, then optionally an activation.
# HIDDEN: the 32 values between input and output are called "hidden" because they're
#         internal — you don't directly control them, training figures out what they store.
#
# RELU (activation function): max(0, x) — zeroes out negatives, keeps positives.
# Why needed: without it, stacking layers collapses into one straight line (y=mx+b)
# because two linear equations combined are still linear. ReLU breaks linearity,
# turning each neuron into a "ramp" (flat on left, sloped on right). With 32 ramps
# each at different positions/slopes, the network can piece together curves like sin(x).
# ReLU is applied to HIDDEN neurons only — not the output neuron, because we need
# the output to predict negative values (sin(x) goes negative).

x_data = torch.linspace(-3, 3, 100, device=device).unsqueeze(1)  # shape: (100,) → (100, 1)
print(f"\ntorch.linspace(-3, 3, 100): 100 evenly spaced points from -3 to 3")
print(f"x_data.shape after .unsqueeze(1): {x_data.shape}  (100 rows, 1 column)")
# unsqueeze(1) is needed so matrix multiply works: (100,1) @ (1,32) → (100,32)
# Without it, shape (100,) can't multiply against (1,32) — dimensions don't align.
print(f"x_data[:5]: {x_data[:5].squeeze()}")   # first 5 values

y_true = torch.sin(x_data)
print(f"\ny_true = torch.sin(x_data)")
print(f"y_true.shape: {y_true.shape}")
print(f"y_true[:5]: {y_true[:5].squeeze()}")   # first 5 sin values

# Network weights — start random/zeros, training will adjust them.
# Why random (not zeros)? If all neurons start identical, they all compute
# the same thing and learn the same thing — they never differentiate.
# Random starting points break that symmetry.
w1 = torch.randn(1, 32, device=device, requires_grad=True)    # input to hidden
b1 = torch.zeros(32, device=device, requires_grad=True)
w2 = torch.randn(32, 1, device=device, requires_grad=True)    # hidden to output
b2 = torch.zeros(1, device=device, requires_grad=True)

print(f"\nNetwork weights:")
print(f"  w1.shape: {w1.shape}  (1 input → 32 hidden neurons, one weight per neuron)")
print(f"  b1.shape: {b1.shape}  (one bias per hidden neuron)")
print(f"  w2.shape: {w2.shape}  (32 hidden → 1 output, one weight per incoming neuron)")
print(f"  b2.shape: {b2.shape}  (one bias for the output neuron)")

learning_rate = 0.01
print(f"\nlearning_rate = {learning_rate}")
print(f"Training for 2000 steps...\n")

for step in range(2000):
    # Forward pass
    # Layer 1: (100,1) @ (1,32) + (32,) = (100,32) — 32 values per input point
    # relu: zeroes out negatives → turns straight lines into ramps → enables curves
    hidden = (x_data @ w1 + b1).relu()
    # Layer 2: (100,32) @ (32,1) + (1,) = (100,1) — 1 prediction per input point
    # No relu here — output needs to go negative (sin(x) ranges from -1 to 1)
    y_pred = hidden @ w2 + b2

    # Loss: MSE — same as part 3
    loss = ((y_pred - y_true) ** 2).mean()

    # Backward pass from loss (not y_pred) — loss is the single number we minimize
    loss.backward()
    # Note: .grad matches the shape of each param — one gradient value per weight/bias

    # Update weights — -= because gradient points uphill, we want to go downhill
    with torch.no_grad():
        for param in [w1, b1, w2, b2]:
            param -= learning_rate * param.grad
            param.grad.zero_()  # reset so gradients don't accumulate into next step

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
6. NEURON = (input × weight) + bias  — needs both: weight=slope, bias=offset
7. RELU turns lines into ramps, enabling the network to learn curves
8. GRADIENT points uphill — use -= to go downhill (reduce loss)

Next up (Step 3): We'll use PyTorch's nn.Module to build networks
properly instead of managing raw tensors by hand.
""")
