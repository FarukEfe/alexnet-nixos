{
  description = "AlexNet PyTorch Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
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
      {
        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.git
            # pkgs.cudaPackages.cudatoolkit  # Optional: for GPU support
          ];

          shellHook = ''
            echo "AlexNet PyTorch Development Environment"
            echo "Python: $(python --version)"
            echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
            echo ""
            echo "Available directories:"
            echo "  - model/     : Neural network architecture"
            echo "  - data/      : Dataset storage"
            echo "  - results/   : Training results"
            echo "  - metrics/   : Performance metrics"
            echo ""
            
            # Set Python path to include current directory
            export PYTHONPATH="$PWD:$PYTHONPATH"
          '';

          # Environment variables
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        };

        # Package definition (optional, for building the project)
        packages.default = pkgs.python311Packages.buildPythonPackage {
          pname = "alexnet-nixos";
          version = "0.1.0";
          src = ./.;
          
          propagatedBuildInputs = with pkgs.python311Packages; [
            torch
            torchvision
            numpy
            matplotlib
            pillow
            tqdm
          ];

          # Skip tests for now (add when you have them)
          doCheck = false;
        };
      }
    );
}
