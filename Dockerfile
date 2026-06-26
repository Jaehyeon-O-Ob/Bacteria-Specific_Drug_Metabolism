FROM nvidia/cuda:13.3.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos "" lukas && \
    echo "lukas ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/lukas

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

RUN ln -s $CONDA_DIR/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /home/lukas/.bashrc && \
    echo "conda activate base" >> /home/lukas/.bashrc

USER lukas
WORKDIR /home/lukas/Bacteria-Specific_Drug_Metabolism

COPY --chown=lukas:lukas . /home/lukas/Bacteria-Specific_Drug_Metabolism

CMD [ "/bin/bash" ]

