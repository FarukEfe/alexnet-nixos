# Traditional Nix shell for compatibility
# Use this if you don't want to enable flakes
{ pkgs ? import <nixpkgs> {} }:

let
  # Python environment - PyTorch via pip, not Nix
  pythonEnv = pkgs.python311.withPackages (ps: with ps; [
    pip
    virtualenv
    # Don't install torch/torchvision from Nix - use pip instead
    numpy
    matplotlib
    pillow
    tqdm
  ]);
in
pkgs.mkShell {
  buildInputs = [
    pythonEnv
    pkgs.git
    # Uncomment for GPU support:
    # pkgs.cudaPackages.cudatoolkit
  ];

  shellHook = ''
    echo "AlexNet PyTorch Development Environment (shell.nix)"
    echo "Setting up Python environment..."
    
    # Create venv if it doesn't exist
    if [ ! -d .venv ]; then
      python -m venv .venv
      source .venv/bin/activate
      pip install --upgrade pip
      pip install -r requirements.txt
    else
      source .venv/bin/activate
    fi
    
    echo "Python: $(python --version)"
    echo "Ready to train!"
  '';

  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
