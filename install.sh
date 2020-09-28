curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

rustup toolchain install nightly
rustup default nightly
pip install torchex setuptools_rust
pip install git+https://github.com/0h-n0/thdbonas.git
pip install -r requirements.txt
pip install git+https://github.com/0h-n0/inferno.git

