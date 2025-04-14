# Start from the CUDA Ubuntu base image
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt update -y && \
    apt install -y \
    vim \
    git \
    ca-certificates \
    parallel \
    latex2html \
    wget \
    unzip \
    bzip2 \
    gcc \
    make \
    emacs \
    zlib1g-dev \
    g++ \
    uuid-runtime \
    python3 \
    python3-pip \
    python-is-python3 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Install NumPy
RUN pip3 install --no-cache-dir numpy

WORKDIR /software/

RUN git clone https://github.com/vishnubk/dedisp.git && \
    cd dedisp && \
    make -j 32 && \
    make install

RUN git clone https://github.com/vishnubk/peasoup.git && \
    cd peasoup && \
    make -j 32 && \
    make install

# Update the dynamic linker run-time bindings
RUN ldconfig /usr/local/lib

