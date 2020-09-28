#!/bin/sh


curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 
source $HOME/.cargo/env
rustup default nightly
pip install setuptools_rust
pip install torchex setuptools_rust
pip install git+https://github.com/0h-n0/thdbonas.git
pip install -r requirements.txt
pip install git+https://github.com/0h-n0/inferno.git

CUDA="cu102"
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
