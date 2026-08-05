#!/usr/bin/env bash
# build-opt-prod.sh — Build the v0 opt-prod image on top of v0-base
#
# Usage:
#   ./v0/build-opt-prod.sh             # build only (local tag)
#   ./v0/build-opt-prod.sh --push      # build + push to GHCR
#
# Prerequisites:
#   - v0-base image built: ./v0/build-base.sh
#   - docker buildx available
#
# The opt-prod image adds performance tuning on top of v0-base:
#   - B12X performance env vars (WO_PROJECTION, MHC, FP8_GEMM, MOE, SPARSE_INDEXER)
#   - PCIe allreduce backend (b12x)
#   - CUDAGraph capture size optimization for DSpark
#   - AOT compile + breakable CUDAGraph disabled
#   - FlashInfer sampler enabled
#   - drop-caches cron job for NFS stability

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
GHCR_REPO="ghcr.io/${GHCR_OWNER}/gb10-kimi-3"
TAG="${TAG:-${GHCR_REPO}:v0-opt-prod}"
BASE_IMAGE="${BASE_IMAGE:-${GHCR_REPO}:v0-base}"

PUSH=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        --base) BASE_IMAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Building gb10-kimi-3 v0-opt-prod image ==="
echo "  Base:   ${BASE_IMAGE}"
echo "  Target: ${TAG}"
echo "  Push:   $([ "$PUSH" == true ] && echo yes || echo no)"
echo ""

# Ensure base image is present locally
docker pull "${BASE_IMAGE}" 2>/dev/null || true

docker build -f "$SCRIPT_DIR/Dockerfile.opt-prod" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${TAG}" \
    "$REPO_DIR"

echo ""
echo "Built: ${TAG}"

if [[ "$PUSH" == true ]]; then
    echo ""
    echo "=== Pushing to GHCR ==="
    docker push "$TAG"
    echo ""
    echo "Done. Published: ${TAG}"
    echo ""
    echo "Make the package public at:"
    echo "  https://github.com/users/${GHCR_OWNER}/packages/container/gb10-kimi-3/settings"
fi

echo ""
echo "Done."
