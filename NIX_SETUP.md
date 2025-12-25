# NixOS/Nix Development Environment Setup

This project uses Nix for reproducible development environments across different machines.

## Prerequisites

### Install Nix (if not already installed)

**On NixOS**: Already installed!

**On other Linux/macOS**:
```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```

**On Windows (WSL2)**:
```bash
sh <(curl -L https://nixos.org/nix/install) --no-daemon
```

### Enable Flakes (recommended)

Add to `~/.config/nix/nix.conf` or `/etc/nix/nix.conf`:
```
experimental-features = nix-command flakes
```

## Quick Start

### Option 1: Using Flakes (Modern, Recommended)

```bash
# Enter development environment
nix develop

# Or run a command directly
nix develop -c python train.py
```

### Option 2: Using shell.nix (Traditional)

```bash
# Enter development environment
nix-shell

# Or run a command directly
nix-shell --run "python train.py"
```

### Option 3: Using direnv (Automatic)

Install direnv for automatic environment activation:

```bash
# Install direnv (if not already)
nix-env -iA nixpkgs.direnv

# Hook into your shell (add to ~/.bashrc or ~/.zshrc)
eval "$(direnv hook bash)"  # for bash
eval "$(direnv hook zsh)"   # for zsh

# Allow direnv in this directory
direnv allow
```

Now the environment activates automatically when you `cd` into the project directory!

## What's Included

The Nix environment provides:
- Python 3.11
- PyTorch with CPU support (configurable for GPU)
- TorchVision
- NumPy, Matplotlib, Pillow
- Training utilities (tqdm, tensorboard)
- scikit-learn for metrics

## GPU Support

To enable CUDA support, uncomment the CUDA line in `flake.nix` or `shell.nix`:

```nix
# Uncomment this line:
pkgs.cudaPackages.cudatoolkit
```

Note: GPU support in Nix requires NixOS or proper CUDA setup on your system.

## Customizing Dependencies

### Adding Python Packages

Edit the `pythonEnv` section in `flake.nix` or `shell.nix`:

```nix
pythonEnv = pkgs.python311.withPackages (ps: with ps; [
  torch
  torchvision
  # Add your package here:
  pandas
  seaborn
]);
```

### Adding System Packages

Add to the `buildInputs` list:

```nix
buildInputs = [
  pythonEnv
  pkgs.git
  pkgs.tmux  # Add system tools here
];
```

## Reproducibility

### Lock Dependencies

Flakes automatically lock dependencies in `flake.lock`. Commit this file:

```bash
git add flake.lock
git commit -m "Lock Nix dependencies"
```

### Update Dependencies

```bash
# Update all dependencies
nix flake update

# Update specific input
nix flake lock --update-input nixpkgs
```

### Share Environment

Others can replicate your exact environment:

```bash
git clone <your-repo>
cd alexnet-nixos
nix develop  # Exact same environment!
```

## Troubleshooting

### "experimental feature 'flakes' is disabled"
Enable flakes in your Nix configuration (see Prerequisites above).

### "error: getting status of '/nix/store/...': No such file or directory"
Run: `nix-collect-garbage -d` to clean up broken store paths.

### Import errors when running Python
The `PYTHONPATH` is set automatically in the shell. If you have issues:
```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Useful Commands

```bash
# Enter development shell
nix develop

# Run a single command
nix develop -c python train.py

# Build the package
nix build

# Check flake
nix flake check

# Show environment info
nix develop -c env | grep -E "(PYTHON|PATH)"
```

## Benefits

✅ **Reproducible**: Same environment on every machine  
✅ **Isolated**: Won't interfere with system Python  
✅ **Declarative**: Dependencies defined in code  
✅ **Cacheable**: Binary caches speed up setup  
✅ **Version Control**: Lock files track exact versions  

## Additional Resources

- [NixOS Manual](https://nixos.org/manual/nix/stable/)
- [Nix Flakes Guide](https://nixos.wiki/wiki/Flakes)
- [Python on Nix](https://nixos.wiki/wiki/Python)
