# Traditional Nix shell for compatibility
# Use this if you don't want to enable flakes
{ pkgs ? import <nixpkgs> {} }:

let
  # Python environment with all dependencies
  pythonEnv = pkgs.python311.withPackages (ps: with ps; [
    torch
    torchvision
    numpy
    matplotlib
    pillow
    tqdm
    tensorboard
    scikit-learn
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
    echo "Python: $(python --version)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    echo ""
    
    # Set Python path to include current directory
    export PYTHONPATH="$PWD:$PYTHONPATH"
  '';

  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
