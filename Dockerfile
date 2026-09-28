# Use specific version of nvidia cuda image
# FROM wlsdml1114/my-comfy-models:v1 as model_provider
ARG BASE_IMAGE=wlsdml1114/multitalk-base:1.8
FROM ${BASE_IMAGE} AS runtime

ARG WAN_HIGH_NOISE_MODEL_URI=hf://Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
ARG WAN_LOW_NOISE_MODEL_URI=hf://Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
ARG WAN_HIGH_NOISE_MODEL_SHA256=6122e79d55e0f235698d11d657f3b196c5273c830da00b2b013c5a048d5e6a42
ARG WAN_LOW_NOISE_MODEL_SHA256=5471a457b6ac404202a5fbe6c11595a3d5641fc766b00f38763f72303fffc21e
# Rebuild the large model layers instead of restoring the slow remote cache.
ARG MODEL_REFRESH=2026-09-28-2

RUN pip install --no-cache-dir -U "huggingface_hub[hf_transfer]" && \
    hf --help > /dev/null
RUN pip install runpod websocket-client

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    cd ComfyUI-Frame-Interpolation && \
    git checkout 26545cc2dd95bc3d27f056016300673bdeee78f5

RUN cd /ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && \
    python install.py
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt

RUN hf download "hf://Comfy-Org/Wan_2.1_ComfyUI_repackaged/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" --local-dir /tmp/wan-text-encoder --force-download && \
    install -m 0644 /tmp/wan-text-encoder/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors /ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
    rm -rf /tmp/wan-text-encoder
RUN hf download "hf://Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/vae/wan_2.1_vae.safetensors" --local-dir /tmp/wan-vae --force-download && \
    install -m 0644 /tmp/wan-vae/split_files/vae/wan_2.1_vae.safetensors /ComfyUI/models/vae/wan_2.1_vae.safetensors && \
    rm -rf /tmp/wan-vae
RUN echo "Wan model refresh: ${MODEL_REFRESH}" && \
    hf download "${WAN_LOW_NOISE_MODEL_URI}" --local-dir /tmp/wan-low --force-download && \
    install -m 0644 /tmp/wan-low/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors /ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors && \
    echo "${WAN_LOW_NOISE_MODEL_SHA256}  /ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" | sha256sum -c - && \
    rm -rf /tmp/wan-low
RUN hf download "hf://Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" --local-dir /tmp/wan-high-lora --force-download && \
    install -m 0644 /tmp/wan-high-lora/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors /ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors && \
    rm -rf /tmp/wan-high-lora
RUN hf download "hf://Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" --local-dir /tmp/wan-low-lora --force-download && \
    install -m 0644 /tmp/wan-low-lora/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors /ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors && \
    rm -rf /tmp/wan-low-lora
RUN hf download "hf://Paolo222/DPaint@9f4f60c46049c67492dc9e2dbf3aa73fe2d7bbec/WAN22_assume_the_position_high.safetensors" --local-dir /tmp/assume-high --force-download && \
    install -m 0644 /tmp/assume-high/WAN22_assume_the_position_high.safetensors /ComfyUI/models/loras/WAN22_assume_the_position_high.safetensors && \
    echo "71c7dc04d49dbcf69da97ece2fe682fa9cfef24818b8fbd7c31d3825cab1543f  /ComfyUI/models/loras/WAN22_assume_the_position_high.safetensors" | sha256sum -c - && \
    rm -rf /tmp/assume-high
RUN hf download "hf://Paolo222/DPaint@9f4f60c46049c67492dc9e2dbf3aa73fe2d7bbec/WAN22_assume_the_position_low.safetensors" --local-dir /tmp/assume-low --force-download && \
    install -m 0644 /tmp/assume-low/WAN22_assume_the_position_low.safetensors /ComfyUI/models/loras/WAN22_assume_the_position_low.safetensors && \
    echo "7ddba644de1d6822ca7f83d69adce8862218e3841ee69fcc9227ecdbd8161a20  /ComfyUI/models/loras/WAN22_assume_the_position_low.safetensors" | sha256sum -c - && \
    rm -rf /tmp/assume-low
RUN hf download "hf://helpfff/airblow@73876392857cd0c3d397ec0656de7e79d2dbd213/Airb-high-80.safetensors" --local-dir /tmp/airblow-high --force-download && \
    install -m 0644 /tmp/airblow-high/Airb-high-80.safetensors /ComfyUI/models/loras/Airb-high-80.safetensors && \
    echo "38a40b1f87f2ebe07e85ad9228438eca177a49c5f53e6a8e05c8a88833108b84  /ComfyUI/models/loras/Airb-high-80.safetensors" | sha256sum -c - && \
    rm -rf /tmp/airblow-high
RUN hf download "hf://helpfff/airblow@73876392857cd0c3d397ec0656de7e79d2dbd213/Airb-low-70.safetensors" --local-dir /tmp/airblow-low --force-download && \
    install -m 0644 /tmp/airblow-low/Airb-low-70.safetensors /ComfyUI/models/loras/Airb-low-70.safetensors && \
    echo "6077e20aa161c9a8619a9aadbf63b5e7d4a737c8683048d7f6653c35a71c0c0e  /ComfyUI/models/loras/Airb-low-70.safetensors" | sha256sum -c - && \
    rm -rf /tmp/airblow-low
RUN echo "Wan model refresh: ${MODEL_REFRESH}" && \
    hf download "${WAN_HIGH_NOISE_MODEL_URI}" --local-dir /tmp/wan-high --force-download && \
    install -m 0644 /tmp/wan-high/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors /ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors && \
    echo "${WAN_HIGH_NOISE_MODEL_SHA256}  /ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" | sha256sum -c - && \
    rm -rf /tmp/wan-high

RUN mkdir -p /ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife && \
    hf download "hf://hfmaster/models-moved@cab6dcee2fbb05e190dbb8f536fbdaa489031a14/rife/rife49.pth" --local-dir /tmp/rife --force-download && \
    install -m 0644 /tmp/rife/rife/rife49.pth /ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth && \
    rm -rf /tmp/rife


COPY . .
RUN mkdir -p /ComfyUI/input && cp /example_image.png /ComfyUI/input/example_image.png
RUN mkdir -p /ComfyUI/user/default/ComfyUI-Manager
COPY config.ini /ComfyUI/user/default/ComfyUI-Manager/config.ini
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
