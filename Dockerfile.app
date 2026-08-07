# Thin app layer ON TOP of runtime. Change handler/workflow often — only these
# few files are pushed (MBs), not the ~24GB ComfyUI stack.
#
# One-time (or rare):
#   docker build -f Dockerfile.runtime -t fortumblr007/generate-video-runtime:1.0 .
#   docker push fortumblr007/generate-video-runtime:1.0
#
# Everyday code iterate:
#   docker build -f Dockerfile.app -t fortumblr007/generate-video-nsfw-i2v:app .
#   docker push fortumblr007/generate-video-nsfw-i2v:app
#
# Override runtime tag if needed:
#   docker build -f Dockerfile.app --build-arg RUNTIME_IMAGE=fortumblr007/generate-video-runtime:1.0 ...
ARG RUNTIME_IMAGE=fortumblr007/generate-video-runtime:1.0
FROM ${RUNTIME_IMAGE}

# App-only files (order stable for layer cache)
COPY download_models.sh /download_models.sh
COPY entrypoint.sh /entrypoint.sh
COPY handler.py /handler.py
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
COPY new_Wan22_api.json /new_Wan22_api.json
# Optional local example image used when no image_* in job
COPY example_image.png /example_image.png

RUN chmod +x /entrypoint.sh /download_models.sh

CMD ["/entrypoint.sh"]
