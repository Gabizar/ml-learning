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
- **Step:** 0 ✅, 1 ✅, starting Step 2
- **What we did:** Confirmed GPU laptop setup works (verify_setup.py passed), created Step 2 PyTorch basics script
- **Script:** `step02_pytorch_basics.py` — covers tensors, GPU usage, autograd, and a mini neural net from scratch
- **Next:** Run step02 on GPU laptop, then move to Step 3 (neural net concepts with nn.Module)
