FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ARG PYTHON_VERSION=3.8

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV TORCH_HOME=/src/weights

RUN apt-get update --fix-missing && \
    apt-get install -y curl && \
    curl -o /tmp/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda install -y python=${PYTHON_VERSION} && \
    conda clean -ya

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget \
    unzip \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install runpod && \
    pip install numpy==1.21.6 && \
    pip install torch>=1.11 && \
    pip install opencv-python-headless==4.5.5.64 && \
    pip install scipy==1.10.0 && \
    pip install openpyxl==3.0.10 && \
    pip install psutil==5.9.4 && \
    pip install tqdm==4.64.1 && \
    pip install omegaconf==2.3.0 && \
    pip install salesforce-lavis

ADD handler.py .

CMD [ "python", "-u", "/handler.py" ]
