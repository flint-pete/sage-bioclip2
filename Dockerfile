# sage-bioclip2 — BioCLIP2 Species Classifier, v2 cache-consumer
# Default model: BioCLIP-2.5 Huge (ViT-H/14), TreeOfLife-200M embeddings.
# Target: 128GB unified memory ARM64 (Sage Thor / DGX Spark).
#
# Base: NVIDIA PyTorch 25.08 (CUDA 13.0, PyTorch 2.8, Python 3.12) — Blackwell
# (sm_110 Thor / sm_120/121 Spark). Same base + freeze strategy as sage-yolo2.
FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /app
COPY requirements.txt .

# Freeze the base image's torch/torchvision/numpy so pip (pulling pybioclip's
# deps) cannot overwrite the Blackwell-enabled build — same fix as sage-yolo2.
RUN pip install --no-cache-dir --upgrade pip && \
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)") && \
    TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)") && \
    NP_VER=$(python3 -c "import numpy; print(numpy.__version__)") && \
    echo "Freezing base stack: torch==${TORCH_VER} torchvision==${TV_VER} numpy==${NP_VER}" && \
    printf "torch==${TORCH_VER}\ntorchvision==${TV_VER}\nnumpy==${NP_VER}\n" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt

# Fresh opencv-headless matching the current numpy (base may ship a mismatched one).
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    rm -rf /usr/local/lib/python3.*/dist-packages/cv2* && \
    pip install --no-cache-dir -c /tmp/constraints.txt "opencv-python-headless==4.11.0.86"

# Enable BioCLIP-2.5 ViT-H/14 in pybioclip 2.1.5 (patches library internals).
# Applied ONCE here at build time — the patch is not idempotent, so app.py must
# NOT re-run it at runtime.
COPY patch_pybioclip.py /tmp/patch_pybioclip.py
RUN python3 /tmp/patch_pybioclip.py && rm /tmp/patch_pybioclip.py

# Pre-download the BioCLIP-2.5 model + TreeOfLife-200M text embeddings (~4-5 GB).
# BEFORE COPY app.py so code edits don't invalidate this expensive layer.
RUN python3 -c "\
from bioclip.predict import TreeOfLifeClassifier; \
TreeOfLifeClassifier(model_str='hf-hub:imageomics/bioclip-2.5-vith14')"

# --- Sage adjustment (not in upstream pybioclip) -------------------------------
# The model weights, config and taxonomy embeddings are now in the image's Hugging
# Face cache (the RUN above). Without this setting, huggingface_hub still makes HEAD
# requests to huggingface.co at every start-up to check for newer files. On a Sage
# node with no outbound internet that can stall or fail start-up. Offline mode makes
# it load only from the baked-in cache. To re-enable online checks for one run:
#   pluginctl run ... -e HF_HUB_OFFLINE=0 ...
ENV HF_HUB_OFFLINE=1

COPY save_match.py .
COPY consumer.py .
COPY selection.py .
COPY seenstore.py .
COPY node_info.py .
COPY app.py .

ENTRYPOINT ["python3", "/app/app.py"]
