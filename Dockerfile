FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    software-properties-common \
    screen \
    tmux \
    gnupg \
    && apt-get clean

WORKDIR /workspace
COPY . .

# LeRobot
RUN cd unitree_lerobot/lerobot && pip install -e .

# unitree_lerobot
RUN cd unitree_lerobot && pip install -e .

# Unitree SDK2 Python
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && \
    cd unitree_sdk2_python && \
    pip install -e .

RUN apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN pip install -e .

CMD ["bash"]