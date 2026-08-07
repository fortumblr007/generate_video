# RunPod serverless worker: Wan2.2 Remix NSFW I2V + Lightning (Kijai wrapper)
# Base: fortumblr007/video-gen-base (CUDA 12.8 + PyTorch). Build base first:
#   docker build -f base.Dockerfile -t fortumblr007/video-gen-base:1.0 .
FROM fortumblr007/video-gen-base:1.0 AS runtime

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kael558/ComfyUI-GGUF-FantasyTalking && \
    cd ComfyUI-GGUF-FantasyTalking && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode && \
    git clone https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server && \
    git clone https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize && \
    cd ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize && \
    mv * ../

# Model dirs always present. Large HF weights are optional at build time.
# BAKE_MODELS=1 (default): bake weights into the image (large ~90GB, hard to push from low-RAM hosts).
# BAKE_MODELS=0: slim image; download_models.sh runs at container start (or mount a Network Volume).
RUN mkdir -p /ComfyUI/models/diffusion_models /ComfyUI/models/loras \
    /ComfyUI/models/text_encoders /ComfyUI/models/vae /ComfyUI/models/clip_vision

COPY download_models.sh /download_models.sh
RUN chmod +x /download_models.sh

ARG BAKE_MODELS=1
RUN if [ "$BAKE_MODELS" = "1" ]; then \
      /download_models.sh; \
    else \
      echo "BAKE_MODELS=0: skipping weight bake (runtime download / volume)"; \
    fi

COPY . .
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
RUN chmod +x /entrypoint.sh /download_models.sh

CMD ["/entrypoint.sh"]
