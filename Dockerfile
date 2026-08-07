# Default build = thin app image on top of published runtime.
# Prefer day-to-day:
#   docker build -f Dockerfile.app -t fortumblr007/generate-video-nsfw-i2v:app .
#
# First-time / heavy stack rebuild:
#   docker build -f Dockerfile.runtime -t fortumblr007/generate-video-runtime:1.0 .
#
# Legacy single-shot slim (runtime+app, no weights):
#   docker build -f Dockerfile.runtime -t fortumblr007/generate-video-runtime:1.0 .
#   docker build -f Dockerfile.app -t fortumblr007/generate-video-nsfw-i2v:1.0-slim .
#
# Full bake with weights still available via old flow:
#   Use Dockerfile.runtime then run download_models into a volume instead of baking.

ARG RUNTIME_IMAGE=fortumblr007/generate-video-runtime:1.0
FROM ${RUNTIME_IMAGE}

COPY download_models.sh /download_models.sh
COPY entrypoint.sh /entrypoint.sh
COPY handler.py /handler.py
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
COPY new_Wan22_api.json /new_Wan22_api.json
COPY example_image.png /example_image.png

RUN chmod +x /entrypoint.sh /download_models.sh

CMD ["/entrypoint.sh"]
