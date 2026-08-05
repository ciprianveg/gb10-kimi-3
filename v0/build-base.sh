#!/usr/bin/env bash
# build-base.sh — Build the v0 base image from the local-inference-lab/vllm fork
#
# Usage:
#   ./v0/build-base.sh                # build only (local tag)
#   ./v0/build-base.sh --push         # build + push to GHCR
#   ./v0/build-base.sh -j 8           # override build jobs (default 12)
#
# Prerequisites:
#   - spark-vllm-docker cloned as a sibling: ../spark-vllm-docker/
#   - mods/b12x-nvfp4/ present in spark-vllm-docker (for --install-b12x)
#   - docker buildx available
#
# The base image is built from the local-inference-lab/vllm fork
# (codex/hh-kimi-k3-dspark-dcp16-20260804 branch) which includes:
#   - B12X MLA attention backend (B12xMLASparseBackend)
#   - DSpark speculative decoding with DCP16 support
#   - InstantTensor loader
#   - K3-specific KDA kernels
#   - EP-aware MoE padding fix
#   - DSpark draft-disable-EP fix
# Plus the b12x 0.15.3 kernel package installed from local mods.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Paths
SPARK_VLLM_DIR="${SPARK_VLLM_DIR:-$(realpath "$REPO_DIR/../spark-vllm-docker" 2>/dev/null || echo "$HOME/eugr/spark-vllm-docker")"

# Fork ref
VLLM_REF="codex/hh-kimi-k3-dspark-dcp16-20260804"
VLLM_REPO="https://github.com/local-inference-lab/vllm.git"

# Image tags
GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
GHCR_REPO="ghcr.io/${GHCR_OWNER}/gb10-kimi-3"
TAG="${TAG:-${GHCR_REPO}:v0-base}"
BUILD_JOBS="${BUILD_JOBS:-12}"
GPU_ARCH="${GPU_ARCH:-12.0f}"

PUSH=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH=true; shift ;;
        -j) BUILD_JOBS="$2"; shift 2 ;;
        --gpu-arch) GPU_ARCH="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Building gb10-kimi-3 v0-base image ==="
echo "  Fork:     ${VLLM_REPO} @ ${VLLM_REF}"
echo "  GPU arch: ${GPU_ARCH}"
echo "  Jobs:     ${BUILD_JOBS}"
echo "  Tag:      ${TAG}"
echo "  Spark dir: ${SPARK_VLLM_DIR}"
echo ""

if [[ ! -d "$SPARK_VLLM_DIR" ]]; then
    echo "ERROR: spark-vllm-docker not found at $SPARK_VLLM_DIR"
    echo "  Set SPARK_VLLM_DIR env var to point to your spark-vllm-docker clone."
    exit 1
fi

cd "$SPARK_VLLM_DIR"

./build-and-copy.sh \
    --vllm-ref "$VLLM_REF" \
    --vllm-repo "$VLLM_REPO" \
    --install-b12x \
    --rebuild-vllm \
    --gpu-arch "$GPU_ARCH" \
    -t "${TAG##*:}" \
    -j "$BUILD_JOBS"

# Tag the local image with the full GHCR path
docker tag "${TAG##*:}" "$TAG"

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
