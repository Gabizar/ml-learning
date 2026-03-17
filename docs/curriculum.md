# ML/Deep Learning - Learning Curriculum

## Goal
Learn ML/deep learning hands-on using Python + PyTorch. Hybrid approach: build from scratch to understand, but stay practical and fun.

## Setup
- **Coding machine:** macOS laptop (coding, committing to GitHub)
- **Training machine:** separate laptop with GPU (pull from GitHub, run training)
- **Stack:** Python, PyTorch

---

## Current Status
- **Current Step:** 7 (in progress)
- **Last Session:** 2026-03-16 — Steps 5+6 combined and complete, starting evaluation

---

## Phase 1: Setup & Foundations

- [x] **Step 0** — Project setup: GitHub repo, project structure, requirements.txt
- [x] **Step 1** — GPU laptop setup: identify GPU, install CUDA, PyTorch, verify everything works
- [x] **Step 2** — PyTorch basics: tensors, operations, autograd (small script exercises)
- [x] **Step 3** — Core concepts: what is a neural net, layers, activation functions, loss, backprop (explained through code, not math)

## Phase 2: First Model (Image Classification)

- [x] **Step 4** — Load & explore a dataset (CIFAR-10 or similar), understand dataloaders
- [x] **Step 5** — Build a simple CNN from scratch in PyTorch, including the full manual training loop (forward pass, loss, backward, optimizer) — steps 5+6 combined in step05_cnn.py
- [ ] **Step 7** — Train, evaluate, visualize results (accuracy, confusion matrix, sample predictions, LR scheduling, checkpointing)

## Phase 3: Level Up (Transfer Learning)

- [ ] **Step 8** — Fine-tune a pretrained model (ResNet or similar) on a custom dataset
- [ ] **Step 9** — Collect your own small dataset (photos from phone/web), train on it
- [ ] **Step 10** — Model saving, loading, and inference

## Phase 4: Fun Stretch Goal (Generative)

- [ ] **Step 11** — Choose a generative project (style transfer, text generation, image generation)
- [ ] **Step 12** — Build and train it
- [ ] **Step 13** — Polish and share
