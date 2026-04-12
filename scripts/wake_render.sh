#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HEALTHCHECK_URL="https://fraudshield-cv7g.onrender.com/api/health"
HEALTHCHECK_URL="${1:-${RENDER_HEALTHCHECK_URL:-$DEFAULT_HEALTHCHECK_URL}}"

echo "Pinging ${HEALTHCHECK_URL}"

response="$(
  curl \
    --silent \
    --show-error \
    --fail \
    --location \
    --retry 3 \
    --retry-all-errors \
    --retry-delay 5 \
    --max-time 60 \
    "${HEALTHCHECK_URL}"
)"

echo "Wake-up ping succeeded."
echo "${response}"
