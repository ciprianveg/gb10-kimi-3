#!/usr/bin/env bash
# b12x-perf-env — set B12X performance env vars at container startup
#
# These enable the B12X custom kernel paths for MLA attention, MoE, and GEMM.
# Without these, vLLM falls back to stock Triton/Marlin paths (slower).
# The v0-opt-prod image bakes these in; this mod lets v0-base users opt in
# without rebuilding.
set -euo pipefail

cat >> /etc/environment <<'EOF'
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_USE_B12X_MHC=1
VLLM_USE_B12X_FP8_GEMM=1
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
VLLM_ENABLE_PCIE_ALLREDUCE=1
B12X_MLA_SM120_UNIFIED=1
B12X_DENSE_SPLITK_TURBO=1
B12X_W4A16_TC_DECODE=1
B12X_MOE_FORCE_A8=1
VLLM_USE_AOT_COMPILE=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPH=1
VLLM_USE_FLASHINFER_SAMPLER=1
EOF

echo "b12x-perf-env: 15 env vars written to /etc/environment"
