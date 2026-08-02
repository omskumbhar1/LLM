# ---------------------------------------------------------------
# NOTE: install torch FIRST, separately. See README.md
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# Your RTX 5050 is Blackwell (sm_120) and needs the cu128 build.
# ---------------------------------------------------------------

diffusers>=0.31.0,<0.33.0   # InstantID pipeline is pinned to this API
transformers>=4.45.0
accelerate>=1.0.0
peft>=0.13.0
safetensors>=0.4.3
huggingface_hub>=0.25.0

# 8-bit optimizer (keeps training inside 8GB)
bitsandbytes>=0.44.0

# face detection + identity embedding
insightface==0.7.3
onnxruntime>=1.18.0

# imaging / misc
opencv-python>=4.9.0
pillow>=10.0.0
numpy>=1.26,<2.0
tqdm>=4.66.0
omegaconf>=2.3.0

# UI
gradio>=4.44.0
