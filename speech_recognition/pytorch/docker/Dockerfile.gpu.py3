FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

WORKDIR /tmp

# Generic python installations
# PyTorch Audio for DeepSpeech: https://github.com/SeanNaren/deepspeech.pytorch/releases
# Development environment installations
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  sox \
  libsox-dev \
  libsox-fmt-all \
  git \
  cmake \
  tree \
  htop \
  bmon \
  iotop \
  tmux \
  vim \
  apt-utils

# Make pip happy about itself.
#RUN pip3 install --upgrade pip

# Unlike apt-get, upgrading pip does not change which package gets installed,
# (since it checks pypi everytime regardless) so it's okay to cache pip.
# Install pytorch
# http://pytorch.org/
RUN pip3 install h5py \
                hickle \
                matplotlib \
                tqdm \
                torch==0.4.1 \
                torchvision \
                cffi \
                python-Levenshtein \
                librosa \
                wget \
                tensorboardX

RUN apt-get update && apt-get install --yes --no-install-recommends cmake \
                                                                    sudo

ENV CUDA_HOME "/usr/local/cuda"

# install warp-ctc
RUN git clone https://github.com/ahsueh1996/warp-ctc.git && \
    cd warp-ctc && git checkout pytorch_bindings && \
    mkdir -p build && cd build && cmake .. && make VERBOSE=1 && \
    cd ../pytorch_binding && python3 setup.py install

# install pytorch audio
RUN apt-get install -y sox libsox-dev libsox-fmt-all
RUN git clone https://github.com/pytorch/audio.git
# RUN cd audio; python setup.py install # Had troubles with pytorch
RUN cd audio; git reset --hard 67564173db19035329f21caa7d2be986c4c23797; python setup.py install

# install ctcdecode
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode; pip install .

ENV SHELL /bin/bash
