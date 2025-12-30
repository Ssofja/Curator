# Dockerfile for NeMo Curator with MFA and Audio Processing
# Base: NVIDIA NeMo Curator 25.09 (Python 3.12)
# Strategy: Create Python 3.10.12 environment for all dependencies to resolve numpy<1.26.0 compatibility

FROM nvcr.io/nvidia/nemo-curator:25.09

# Metadata
LABEL maintainer="NeMo Curator Audio Pipeline"
LABEL description="NeMo Curator with MFA, vLLM, Chatterbox TTS in Python 3.10 environment"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV RAY_DASHBOARD_HOST=0.0.0.0
ENV CUDA_VISIBLE_DEVICES=all

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    vim \
    less \
    curl \
    sox \
    libsox-dev \
    libsox-fmt-all \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba \
    && mkdir -p /opt/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/usr/local/bin:$PATH

# Initialize micromamba
RUN /usr/local/bin/micromamba shell init -s bash --root-prefix=/opt/micromamba \
    && echo "eval \"\$(/usr/local/bin/micromamba shell hook -s bash)\"" >> /root/.bashrc

# Create main Python 3.10 environment with all dependencies
# This avoids numpy<1.26.0 incompatibility with Python 3.12
RUN /usr/local/bin/micromamba create -n py310 -c conda-forge -c nvidia -y \
    python=3.10 \
    pip \
    cuda-toolkit=12.1 \
    && /usr/local/bin/micromamba clean --all --yes

# Install numpy and cython first (required for building pkuseg in chatterbox-tts)
RUN /usr/local/bin/micromamba run -n py310 pip install --no-cache-dir \
    "numpy>=1.24.0,<1.26.0" \
    "cython>=0.29.0"

# Install Chatterbox-TTS with --no-build-isolation so pkuseg can find numpy
RUN /usr/local/bin/micromamba run -n py310 pip install --no-cache-dir --no-build-isolation chatterbox-tts==0.1.4

# Install NeMo Curator and dependencies
RUN /usr/local/bin/micromamba run -n py310 pip install --no-cache-dir \
    nemo-curator \
    loguru \
    librosa \
    soundfile \
    pydub \
    praatio \
    textgrid \
    nvidia-ml-py

# Install vLLM 0.8.5
RUN /usr/local/bin/micromamba run -n py310 pip install --no-cache-dir vllm==0.8.5

# Install Ray 2.31.0
RUN /usr/local/bin/micromamba run -n py310 pip install --no-cache-dir ray==2.31.0

# Install OpenTelemetry packages compatible with Ray 2.x
RUN /usr/local/bin/micromamba run -n py310 pip install --no-cache-dir \
    opentelemetry-api==1.27.0 \
    opentelemetry-sdk==1.27.0 \
    opentelemetry-exporter-otlp-proto-grpc==1.27.0 \
    opentelemetry-semantic-conventions==0.48b0

# Install Montreal Forced Aligner in separate environment
RUN /usr/local/bin/micromamba create -n mfa_aligner -c conda-forge -y \
    python=3.10 \
    montreal-forced-aligner=3.1.0 \
    "joblib<1.3.0" \
    && /usr/local/bin/micromamba clean --all --yes

# Copy MFA models from host (instead of downloading, since MFA server may be down)
# Models should be in the build context
COPY mfa_models/dictionary/english_us_arpa.dict /root/.local/share/montreal-forced-aligner/data/models/dictionary/
COPY mfa_models/acoustic/english_us_arpa.zip /root/.local/share/montreal-forced-aligner/data/models/acoustic/

# Create directories for pipeline data
RUN mkdir -p /workspace/data/input \
    /workspace/data/output \
    /workspace/data/voices \
    /workspace/logs

# Copy pipeline files
COPY vllm_inference.py /workspace/
COPY tts_generation.py /workspace/
COPY mfa_rttm_generation.py /workspace/
COPY merge_conversation_stage.py /workspace/
COPY topic_expander.py /workspace/
COPY pipeline_all.py /workspace/


# Create wrapper script for MFA
RUN echo '#!/bin/bash\n\
/usr/local/bin/micromamba run -n mfa_aligner mfa "$@"' > /usr/local/bin/mfa \
    && chmod +x /usr/local/bin/mfa

# Create wrapper for python (uses py310 environment by default)
RUN echo '#!/bin/bash\n\
/usr/local/bin/micromamba run -n py310 python "$@"' > /usr/local/bin/python-py310 \
    && chmod +x /usr/local/bin/python-py310

# Add py310 python to PATH - make it the default
ENV PATH=/opt/micromamba/envs/py310/bin:$PATH

# Create entrypoint script that sets up the py310 environment
RUN echo '#!/bin/bash\n\
\n\
# Ensure py310 environment is in PATH\n\
export PATH=/opt/micromamba/envs/py310/bin:$PATH\n\
\n\
# Set Ray dashboard to bind to all interfaces\n\
export RAY_DASHBOARD_HOST=0.0.0.0\n\
\n\
# Execute command\n\
exec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Set working directory
WORKDIR /workspace

# Expose Ray dashboard port
EXPOSE 8265

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["bash"]
