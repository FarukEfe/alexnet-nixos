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
      {
        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.git
            # pkgs.cudaPackages.cudatoolkit  # Optional: for GPU support
          ];

          shellHook = ''
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

          # Environment variables
          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
        };

        # Package definition (optional, for building the project)
        packages.default = pkgs.python311Packages.buildPythonPackage {
          pname = "alexnet-nixos";
          version = "0.1.0";
          src = ./.;
          
          propagatedBuildInputs = with pkgs.python311Packages; [
            # torch and torchvision installed via pip in venv
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
