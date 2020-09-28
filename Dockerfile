FROM pytorch/pytorch:latest
RUN apt update
RUN apt upgrade -y
RUN apt install git curl build-essential -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH /root/.cargo/bin:$PATH
RUN rustup toolchain install nightly
RUN rustup default nightly
COPY requirements.txt /requirements.txt
RUN pip install torchex setuptools_rust
RUN pip install git+https://github.com/0h-n0/thdbonas.git
RUN pip install -r /requirements.txt
RUN pip install git+https://github.com/0h-n0/inferno.git
RUN apt remove git curl -y
RUN apt clean
WORKDIR /workspace
