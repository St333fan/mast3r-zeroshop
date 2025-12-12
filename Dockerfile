FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL description="Docker container for MASt3R with dependencies installed. CUDA VERSION"
ENV DEVICE="cuda"
ENV MODEL="MASt3R_ViTLarge_BaseDecoder_512_dpt.pth"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git=1:2.34.1-1ubuntu1.10 \
    libglib2.0-0=2.72.4-0ubuntu2.2 \
    wget \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Add conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Initialize conda and disable auto-activation of base environment
RUN conda init bash && \
    conda config --set auto_activate_base false

# Accept conda Terms of Service for required channels
RUN conda config --set channel_priority flexible && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create colmap environment with Python 3.10 and install colmap
RUN conda create -n colmap python=3.10 -y && \
    bash -c "source activate colmap && conda install -c conda-forge colmap -y && conda deactivate"

RUN mkdir -p /home/appuser/mast3r
COPY . /home/appuser/mast3r/

WORKDIR /home/appuser/mast3r/

# Create mast3r environment with Python 3.11 and cmake
RUN conda create -n mast3r python=3.11 cmake=3.14.0 -y

# Install PyTorch with CUDA 12.1 support via pip (avoids Intel MKL symbol issues)
RUN bash -c "source activate mast3r && \
    pip install 'numpy<2.0' && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt && \
    pip install -r dust3r/requirements.txt && \
    pip install -r dust3r/requirements_optional.txt && \
    pip install 'opencv-python<4.10' --force-reinstall && \
    conda install -c conda-forge faiss-gpu -y && \
    pip install asmk && \
    pip install 'scipy<1.14' --force-reinstall && \
    pip install 'numpy<2.0' --force-reinstall"


# Make mast3r environment activate by default
RUN echo "conda activate mast3r" >> ~/.bashrc
ENV CONDA_DEFAULT_ENV=mast3r

# Disable Intel JIT profiling to avoid symbol errors
ENV DISABLE_ITTNOTIFY=1