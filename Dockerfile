# syntax=docker/dockerfile:1.6
#
# Multi-stage build.  The runtime image contains no compiler, no build cache and
# no test fixtures, and runs as a non-root user.
#
#   docker build -t ulrc3:1.0.0 .
#   docker run --rm -p 8000:8000 ulrc3:1.0.0
#
# Image size: ~180 MB with tiktoken + numpy, ~65 MB with --build-arg EXTRAS=""
# (the pure-stdlib core is fully functional; only token counts get approximate).

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG EXTRAS=server,tokenizers,fast
# tiktoken's cache location is controlled by TIKTOKEN_CACHE_DIR, *not* by
# ~/.cache -- setting it here is what makes the BPE table copyable into the
# runtime stage (and the build reproducible offline).
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY ulrc3 ./ulrc3
RUN pip install --upgrade pip setuptools wheel \
 && if [ -n "$EXTRAS" ]; then pip install ".[${EXTRAS}]"; else pip install .; fi

# Pre-download the BPE table so the first request is not slowed by a network
# round trip -- and so the container works in air-gapped deployments.  The
# directory is created unconditionally so the runtime COPY cannot fail when the
# build host has no network (the engine then falls back to the heuristic
# tokenizer, which is a supported configuration).
RUN mkdir -p /opt/tiktoken \
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || true

# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ULRC3_MODE=balanced \
    ULRC3_MAX_CONCURRENCY=32 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken /opt/tiktoken
COPY --from=builder /build/ulrc3 /app/ulrc3

RUN useradd --create-home --uid 10001 ulrc3 \
 && chown -R ulrc3:ulrc3 /app /opt/tiktoken 2>/dev/null || true
WORKDIR /app
USER 10001

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/v1/health\", timeout=2).status==200 else 1)"

# Render (and most PaaS) inject $PORT and require the process to bind to it.
# A shell form is used so the variable is expanded at runtime, not build time.
CMD ["sh", "-c", "exec python -m uvicorn ulrc3.server.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
