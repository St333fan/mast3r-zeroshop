FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL description="Docker container for MASt3R with dependencies installed. CUDA VERSION"
ENV DEVICE="cuda"
ENV MODEL="MASt3R_ViTLarge_BaseDecoder_512_dpt.pth"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git=1:2.34.1-1ubuntu1.10 \
    libglib2.0-0=2.72.4-0ubuntu2.2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/appuser/mast3r
COPY . /home/appuser/mast3r/

WORKDIR /home/appuser/mast3r/

#RUN pip install -r requirements.txt
#RUN pip install -r requirements_optional.txt
#RUN pip install opencv-python==4.8.0.74

#WORKDIR /home/appuser/mast3r/dust3r/croco/models/curope/
#RUN python setup.py build_ext --inplace

#WORKDIR /home/appuser/mast3r
#RUN pip install -r requirements.txt