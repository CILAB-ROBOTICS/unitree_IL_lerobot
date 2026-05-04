FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    software-properties-common \
    screen \
    tmux \
    gnupg \
    python3 \
    python3-pip \
    && apt-get clean

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN apt-get remove -y python3-blinker || true

WORKDIR /workspace
COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir "blinker>=1.7"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch \
        torchvision \
        pyyaml \
        pexpect \
        opencv-python \
        matplotlib \
        einops \
        packaging \
        h5py \
        ipython \
        tqdm

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1


RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

RUN pip uninstall -y torchcodec

RUN pip install torchcodec==0.2.0

RUN cd /workspace/unitree_lerobot/lerobot && \
    pip install -e .

RUN cd /workspace && \
    pip install -e .

RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && \
    cd unitree_sdk2_python && \
    pip install -e .

CMD ["bash"]