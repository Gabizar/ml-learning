# Session Log

## Session 1 — 2026-03-12
- **Step:** Planning
- **What we did:** Defined learning goals, chose stack (Python + PyTorch), planned two-machine workflow, created curriculum
- **Decisions made:**
  - Hybrid learning approach (build from scratch, concepts as they come up)
  - Code on macOS laptop, train on GPU laptop, sync via GitHub
  - Start with image classification, progress to transfer learning, then generative models
- **Next:** Step 0 — set up the project (repo structure, requirements.txt, .gitignore), then init GitHub repo

## Session 2 — 2026-03-13
- **Step:** 0 ✅, 1 ✅, 2 ✅
- **What we did:** Confirmed GPU laptop setup works (verify_setup.py passed), created and walked through Step 2 PyTorch basics script
- **Script:** `step02_pytorch_basics.py` — covers tensors, GPU usage, autograd, and a mini neural net from scratch
- **Concepts covered:** tensors, dot product, matrix multiply, broadcasting, unsqueeze/squeeze, GPU async, autograd, gradients, MSE loss, gradient descent, neurons, layers, bias, ReLU, activation functions, hidden layers
- **Next:** Step 3 — nn.Module, building networks properly instead of managing raw tensors by hand

## Session 3 — 2026-03-13
- **Step:** 3 ✅
- **What we did:** Built step03_nn_module.py — nn.Module, nn.Linear, SGD/Adam optimizers, MSELoss, training loop, inference mode, save/load (.pth and .json)
- **Concepts covered:** nn.Module, nn.Linear, forward(), optimizer, loss functions, model.eval(), torch.no_grad(), state_dict, layer types, activation choices, optimizer choices
- **Next:** Step 4 — load CIFAR-10 dataset, understand dataloaders

## Session 4 — 2026-03-15
- **Step:** 4 ✅
- **What we did:** Built step04_dataset.py — CIFAR-10 loading, transforms, DataLoader, batch inspection, image visualization
- **Concepts covered:** transforms, ToTensor, Normalize, mean/std, PIL, batches, epochs, num_workers, DataLoader, dataset filesystem structure
- **Next:** Step 5 — build a CNN to classify CIFAR-10 images

## Session 5 — 2026-03-16
- **Step:** 5 ✅
- **What we did:** Built step05_cnn.py — full CNN for CIFAR-10 classification
- **Concepts covered:** Conv2d, MaxPool2d, flatten, logits, CrossEntropyLoss, argmax, train/test accuracy, depth, spatial size, padding, filter output shape
- **Next:** Step 6 — proper training loop with learning rate scheduling and better evaluation
