# GPU Laptop Setup Guide

## OS: Ubuntu 24.04 LTS Desktop
## Hardware: ASUS ROG 16, Intel i9-14900HX, NVIDIA RTX 4080, 32 GB RAM, 1 TB NVMe

---

## Step 1: Prepare BIOS

1. Boot into BIOS/UEFI (usually hold **F2**, **F12**, or **Del** during boot — varies by laptop brand)
2. Disable **Secure Boot** — avoids issues with NVIDIA drivers
3. Save and exit BIOS

---

## Step 2: Install Ubuntu 24.04 (primary OS, wipe entire drive)

### 2a. Create bootable USB
1. Download Ubuntu 24.04 LTS Desktop ISO: https://releases.ubuntu.com/24.04/
2. Create a bootable USB from another computer using [Balena Etcher](https://etcher.balena.io/) or [Rufus](https://rufus.ie/) (Windows)

### 2b. Install Ubuntu
1. Boot from USB (usually **F12** or **F2** for boot menu)
2. Choose **"Install Ubuntu"** (or "Try or Install Ubuntu")
3. When you reach the **"Installation type"** screen:
   - Choose **"Something else"** (manual partitioning)
   - Delete all existing partitions on the drive (`nvme0n1` only — don't touch USB)
   - Create the following partitions (use **"+"** button on free space):
     - **EFI System Partition** — 512 MB (needed for UEFI boot and later for Windows dual boot)
     - **swap** — 32768 MB (32 GB, matches your RAM)
     - **ext4** with mount point **/** — 460800 MB (~450 GB)
     - Leave ~500 GB as free space for Windows later
   - Set **device for bootloader installation** to `/dev/nvme0n1`
4. Other recommended settings:
   - Normal installation (includes Firefox, utilities)
   - Check **"Install third-party software for graphics and Wi-Fi hardware"** — this may install NVIDIA drivers automatically
5. Complete the install, reboot, remove USB

---

## Step 2.5 (Optional): Install Windows as Secondary OS

Do this whenever you want — now or later. Ubuntu will remain the primary OS.

### 2.5a. Intel IRST/VMD driver
The Windows installer won't see the NVMe drive without this driver. You need to:
1. Download the IRST/VMD driver from the [ASUS ROG support page](https://rog.asus.com/) for your model (under Chipset drivers)
2. The download is an exe — extract it on a Windows machine (it can't be extracted on Linux)
3. Copy the extracted folder (containing `.inf` files) onto the Windows install USB

### 2.5b. Create Windows bootable USB
1. Download Windows 11 ISO from https://www.microsoft.com/software-download/windows11
2. Create a bootable USB using [Rufus](https://rufus.ie/) (from another Windows machine) or [WoeUSB](https://github.com/WoeUSB/WoeUSB) (from Ubuntu: `sudo apt install woeusb && sudo woeusb --device /path/to/Windows.iso /dev/sdX`)
3. Copy the extracted IRST/VMD driver folder onto the USB

### 2.5c. Install Windows into the free space
1. Boot from the Windows USB
2. Click **"I don't have a product key"** — the license is embedded in your laptop's hardware and will activate automatically
3. On the "Where to install" screen, click **"Load driver"** → **Browse** → navigate to the IRST/VMD driver folder on the USB
4. After the driver loads, the NVMe drive partitions will appear
5. Select the **unallocated free space** — do NOT touch the Ubuntu or EFI partitions
6. Complete the Windows install
7. If the installer requires internet and Wi-Fi doesn't work, press **Shift+F10** and type `OOBE\BYPASSNRO` to skip the requirement

### 2.5d. Restore GRUB
After Windows installs, it takes over the bootloader. Fix it from Ubuntu:

1. Reboot into BIOS, change **boot order** to put Ubuntu first
2. If Ubuntu boots, open a terminal and run:
   ```bash
   sudo update-grub
   ```
   It should say "Found Windows Boot Manager".

3. If Ubuntu doesn't appear in BIOS boot options, boot from an Ubuntu live USB and run:
   ```bash
   sudo fdisk -l
   # Find your ext4 partition (e.g., /dev/nvme0n1p3) and EFI partition (e.g., /dev/nvme0n1p1)

   sudo mount /dev/<your-ubuntu-partition> /mnt
   sudo mount /dev/<your-efi-partition> /mnt/boot/efi
   sudo mount --bind /dev /mnt/dev
   sudo mount --bind /proc /mnt/proc
   sudo mount --bind /sys /mnt/sys

   sudo chroot /mnt
   grub-install /dev/nvme0n1
   update-grub
   exit

   sudo umount -R /mnt
   sudo reboot
   ```

4. If GRUB doesn't show the menu, edit `/etc/default/grub`:
   ```bash
   sudo nano /etc/default/grub
   ```
   - Set `GRUB_TIMEOUT=5`
   - Add `GRUB_DISABLE_OS_PROBER=false`
   - Save, then `sudo update-grub`

### 2.5e. Disable Fast Startup in Windows
Boot into Windows and disable Fast Startup to prevent filesystem corruption:
1. Open **Control Panel → Power Options → Choose what the power buttons do**
2. Click **"Change settings that are currently unavailable"**
3. Uncheck **"Turn on fast startup"**
4. Save changes

### 2.5f. Install Windows drivers
Connect an ethernet cable, then:
1. **Settings → Windows Update → Check for updates**
2. Click **"Advanced options" → "Optional updates"** — install driver updates
3. Restart and check for updates again (may take multiple rounds)
4. This should install Wi-Fi, trackpad, GPU, and other drivers automatically

---

## Step 3: Update System & Install Essential Tools

```bash
sudo apt update && sudo apt upgrade -y
```

### Git
```bash
sudo apt install -y git
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### SSH key (for GitHub)
```bash
ssh-keygen -t ed25519 -C "your@email.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```
Copy the output and add it to GitHub: **Settings → SSH and GPG keys → New SSH key**

Test it:
```bash
ssh -T git@github.com
```

### Build essentials (needed for compiling Python and various packages)
```bash
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev \
  xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### Useful tools
```bash
sudo apt install -y htop tree vim tmux unzip
```

### VS Code
```bash
sudo apt install -y software-properties-common apt-transport-https
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
sudo apt update
sudo apt install -y code
rm packages.microsoft.gpg
```

Recommended extensions (install from VS Code terminal):
```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

### Docker
```bash
# Add Docker's official GPG key and repo
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Run Docker without sudo
sudo usermod -aG docker $USER
```
> **Note:** You need to log out and back in for the Docker group to take effect.

### Reboot
```bash
sudo reboot
```

---

## Step 4: ASUS ROG Laptop Support

### ASUS ROG Linux tools (asusctl + supergfxctl)

Provides keyboard backlight, Aura RGB, fan profiles, GPU switching, function keys, etc.

**Option A — Install via apt (try this first):**
```bash
sudo apt install -y asusctl supergfxctl
sudo systemctl enable --now supergfxd
```

**Option B — If apt doesn't have it, use the community setup script (requires Docker):**
```bash
git clone https://github.com/dariomncs/asus-ubuntu.git /tmp/asus-ubuntu
cd /tmp/asus-ubuntu
chmod +x asus-ubuntu-builder.sh
sudo ./asus-ubuntu-builder.sh
```

#### What this gives you:
- **Keyboard backlight & Aura RGB:**
  ```bash
  asusctl led-mode static -c ff0000   # set color (hex)
  asusctl led-mode breathe            # breathing effect
  asusctl led-mode off                # turn off
  ```
- **Fan profiles:**
  ```bash
  asusctl profile -p balanced         # balanced / performance / quiet
  ```
- **GPU mode switching (integrated / hybrid / dedicated):**
  ```bash
  supergfxctl -m hybrid               # use both iGPU and dGPU
  supergfxctl -m dedicated            # force NVIDIA only (best for ML training)
  ```
- **Function keys** (volume, brightness, fan profile toggle) should work automatically after install

### ROG GUI control panel (optional)
If the community script installed it, run `rog-control-center` from the terminal or find it in your apps menu. It provides a graphical interface for all the above settings. Requires Wayland (Ubuntu 24.04 default).

### Audio fix (if no sound output)
Try these in order:

1. Install SOF firmware:
   ```bash
   sudo apt install -y firmware-sof-signed
   sudo reboot
   ```

2. If still no audio, add kernel parameter:
   ```bash
   sudo nano /etc/default/grub
   ```
   Change `GRUB_CMDLINE_LINUX_DEFAULT` to:
   ```
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash snd_hda_intel.dmic_detect=0"
   ```
   Then:
   ```bash
   sudo update-grub
   sudo reboot
   ```

3. Check what PipeWire/PulseAudio sees:
   ```bash
   pactl list sinks short
   ```

---

## Step 5: NVIDIA Container Toolkit (for running GPU workloads in Docker)
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Test GPU in Docker (run after NVIDIA drivers are installed):
```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 6: Install NVIDIA Drivers

Check if a driver was already installed:
```bash
nvidia-smi
```

If that works and shows your GPU — skip to Step 7.

If not, install the driver:
```bash
# List available drivers
ubuntu-drivers devices

# Install the recommended driver (usually the latest tested)
sudo ubuntu-drivers autoinstall

sudo reboot
```

After reboot, verify:
```bash
nvidia-smi
```

You should see your GPU model, driver version, and CUDA version. **Write down the CUDA version shown** — we'll need it.

---

## Step 7: Install CUDA Toolkit

The `nvidia-smi` CUDA version shows what your driver *supports*. We still need the toolkit.

Go to: https://developer.nvidia.com/cuda-toolkit-archive

Pick the version matching what `nvidia-smi` showed (e.g., 12.x). Select:
- Linux → x86_64 → Ubuntu → 24.04 → deb (local)

Follow the commands NVIDIA gives you. It will look something like:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda
sudo reboot
```

After reboot, add to your shell config (`~/.bashrc`):
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
nvcc --version
```

---

## Step 8: Install uv and Python

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Verify uv is installed
uv --version
```

uv will automatically manage Python versions — no need to install Python separately.

---

## Step 9: Clone the Project & Install Dependencies

```bash
git clone <YOUR_REPO_URL> ~/ml-learning
cd ~/ml-learning

# Create a virtual environment with Python 3.11 and install dependencies
# uv will download Python 3.11 automatically if not present
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

> **Note on cuDNN:** PyTorch's pip package bundles its own cuDNN, so no separate cuDNN install is needed. cuDNN is NVIDIA's library of optimized operations for deep learning (convolution, pooling, etc.) — PyTorch uses it under the hood to make GPU training fast.

---

## Step 10: Verify Everything Works

```bash
cd ~/ml-learning
source .venv/bin/activate
python verify_setup.py
```

This script checks:
- Python version
- PyTorch installed and importable
- CUDA available to PyTorch
- GPU name and memory
- A small tensor operation on GPU

---

## Troubleshooting

### nvidia-smi says "command not found"
- Driver not installed. Redo Step 6.

### PyTorch says CUDA not available
- Version mismatch between CUDA toolkit and PyTorch. Check:
  ```bash
  python -c "import torch; print(torch.version.cuda)"
  nvcc --version
  ```
  These should be compatible (don't need to match exactly, but same major version).
- Reinstall PyTorch with the correct CUDA version from https://pytorch.org/get-started/locally/

### "Out of memory" errors during training
- Reduce batch size
- Use `torch.cuda.empty_cache()`
- Run `nvidia-smi` to see what's using GPU memory

---

## Daily Workflow

```bash
cd ~/ml-learning
git pull
source .venv/bin/activate
# run whatever script we're working on
```
