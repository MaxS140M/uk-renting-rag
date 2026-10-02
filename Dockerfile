# Production image for the UK Renting Guidance Assistant (API + demo page on one port).
#
#   docker build -t uk-renting-rag .
#   docker run -p 7860:7860 --env-file .env uk-renting-rag
#
# The models and the search index are built into the image, so the container starts
# without downloading anything. The API key is never baked in: it is passed at run time.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as an unprivileged user (UID 1000, which Hugging Face Spaces also expects). If the
# app were ever compromised, the attacker would not have root inside the container.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/home/user/app/src:/home/user/app \
    GRADIO_ANALYTICS_ENABLED=False
WORKDIR /home/user/app

# 1. Dependencies, in their own layer: rebuilt only when requirements.txt changes, not on
#    every code change. The CPU-only PyTorch build avoids several GB of unused CUDA files.
COPY --chown=user requirements.txt .
RUN TORCH_VERSION=$(grep -E '^torch==' requirements.txt | cut -d= -f3) \
    && pip install --user "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user -r requirements.txt

# 2. Only what the app needs: no tests, notebooks, evaluation results or secrets.
COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user scripts/deploy/prepare_deployment.py ./scripts/deploy/
COPY --chown=user eval/configs.yaml ./eval/
COPY --chown=user data/raw ./data/raw

# 3. Bake in the models and the index for the deployed configuration.
ARG RAG_CONFIG=hybrid_rerank_bge
ENV RAG_CONFIG=${RAG_CONFIG}
RUN python scripts/deploy/prepare_deployment.py

# Everything is local now: fail fast rather than silently downloading at run time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=7860
EXPOSE 7860

# --no-access-log: uvicorn's access log would record visitors' IP addresses.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
