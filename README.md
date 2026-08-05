# gb10-kimi-3 — Full Kimi K3 on 16× GB10 / DGX Spark cluster

Full `moonshotai/Kimi-K3` (BF16 attention/dense-MLP + MXFP4-quantized experts, 896 experts, ~1453 GiB total) served via vLLM on a 16× GB10 / DGX Spark cluster with DSpark speculative decoding.

**This is v0 — a first working release.** Improvements are in progress (see [Future work](#future-work)). Contributions and ideas are welcome.

---

## What works

- Full Kimi-K3 loaded across 16× GB10 (TP=16, EP=16, ~91 GiB/node)
- DSpark speculative decoding with `Inferact/Kimi-K3-DSpark` (7 draft tokens)
- B12X MLA attention backend (`B12X_MLA_SPARSE`) with CUDAGraph `FULL_AND_PIECEWISE` mode
- InstantTensor loader (~3 min weight load vs ~20 min stock)
- ~250k context in `auto` mode (no DCP)
- Tool-call benchmark: **91/100, ★★★★★ Excellent**

## What doesn't work yet

- **At 100k context, decode drops to ~10 t/s** — the KV cache budget is tight at TP16 without DCP. A future version with DCP16 (decode-context-parallel) should fix this.
- CUDAGraph capture is functional but not yet tuned for optimal batch sizes.
- The 32K draft KV tail bound (from local-inference-lab/vllm#239) is not yet wired — the draft inherits the target's `max_model_len`, over-allocating draft KV by ~6×.

---

## Benchmarks

16× GB10 cluster, MikroTik Switch CRS804-4DDQ with 4× 400-to-4x100gbit breakout cables, RoCE v2 fabric. DSpark enabled with `Inferact/Kimi-K3-DSpark`, `num_speculative_tokens=7`. `llama-benchy` on coherent corpus, single-stream.

| Model | Test | t/s | Peak t/s | est PPT (ms) | e2e TTFT (ms) |
|---|---|---|---|---|---|
| KIMI-K3 | pp2048 @ d4000 | 654.89 | — | 8286.85 | 8288.20 |
| KIMI-K3 | tg1500 @ d4000 | 21.71 | 37.00 | — | — |
| KIMI-K3 | pp2048 @ d16000 | 758.78 | — | 21455.57 | 21460.88 |
| KIMI-K3 | tg1500 @ d16000 | 25.39 | 38.00 | — | — |

> **Note:** At 100k context, decode drops to ~10 t/s. This is expected given the KV cache budget at TP16 without DCP. A future version with DCP16 is planned to address this.

### Tool-call benchmark

```
Model: /root/models/Kimi-K3
Score: 91 / 100
Rating: ★★★★★ Excellent
```

Run via `tool-eval-bench --base-url http://localhost:5001 --parallel 2`.

### Power draw

~2.3 kW total for the 16-node cluster + switch under load.

---

## Images

Two images are published, both `linux/arm64` built for GB10 / sm_121:

| Image | Tag | Description |
|---|---|---|
| `ghcr.io/ciprianveg/gb10-kimi-3` | `v0-base` | Fork + b12x kernels, minimal config |
| `ghcr.io/ciprianveg/gb10-kimi-3` | `v0-opt-prod` | v0-base + B12X perf env vars + drop-caches cron + async scheduling |

### v0-base

Built from the [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm) fork (branch `codex/hh-kimi-k3-dspark-dcp16-20260804`) which includes:

- B12X MLA attention backend (`B12xMLASparseBackend`) — supports `FULL_AND_PIECEWISE` CUDAGraph with spec-decode (unlike stock `TritonMLABackend` which downgrades to `NONE`)
- DSpark speculative decoding with DCP16 support
- InstantTensor loader (pinned-memory weight load)
- K3-specific KDA kernels (fused decode + prefill)
- EP-aware MoE intermediate padding fix
- DSpark draft-disable-EP fix
- K3 truncated reasoning fix

Plus the `b12x` 0.15.3 kernel package (CuTe DSL kernels for NVFP4 GEMM and MoE).

### v0-opt-prod

Built on top of `v0-base` with performance env vars baked in:

- `VLLM_USE_B12X_WO_PROJECTION=1`, `VLLM_USE_B12X_MHC=1`, `VLLM_USE_B12X_FP8_GEMM=1`, `VLLM_USE_B12X_MOE=1`, `VLLM_USE_B12X_SPARSE_INDEXER=1`
- `VLLM_PCIE_ALLREDUCE_BACKEND=b12x`, `VLLM_ENABLE_PCIE_ALLREDUCE=1`
- `B12X_MLA_SM120_UNIFIED=1`, `B12X_DENSE_SPLITK_TURBO=1`, `B12X_W4A16_TC_DECODE=1`, `B12X_MOE_FORCE_A8=1`
- `VLLM_USE_AOT_COMPILE=1`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- drop-caches cron (clears FS page cache every 60s for NFS stability on UMA)

---

## Build

### Prerequisites

- [spark-vllm-docker](https://github.com/ciprianveg/spark-vllm-docker) cloned as a sibling directory
- Docker with buildx support
- 12+ CPU cores, 64+ GiB RAM (build is memory-intensive; use `-j 12` to avoid OOM)

### Build from source

```bash
# Clone
git clone https://github.com/ciprianveg/gb10-kimi-3.git
cd gb10-kimi-3

# Build v0-base (from fork + b12x, ~20-30 min)
./v0/build-base.sh -j 12

# Build v0-opt-prod (overlay on v0-base, ~1 min)
./v0/build-opt-prod.sh

# Push to GHCR (optional)
./v0/build-base.sh -j 12 --push
./v0/build-opt-prod.sh --push
```

### Pull prebuilt

```bash
docker pull ghcr.io/ciprianveg/gb10-kimi-3:v0-base
docker pull ghcr.io/ciprianveg/gb10-kimi-3:v0-opt-prod
```

---

## Serve

Recipes use the [spark-vllm-docker](https://github.com/ciprianveg/spark-vllm-docker) recipe format (eugr-based YAML). No API key is hardcoded — set your own.

```bash
# From spark-vllm-docker:
./run-recipe.sh ../gb10-kimi-3/v0/recipes/kimi-k3-full-tp16-base.yaml --setup

# Or opt-prod:
./run-recipe.sh ../gb10-kimi-3/v0/recipes/kimi-k3-full-tp16-opt-prod.yaml --setup
```

### Key settings

| Setting | Value | Notes |
|---|---|---|
| Tensor parallel | 16 | One rank per GB10 |
| Attention backend | `B12X_MLA_SPARSE` | Supports CUDAGraph + DSpark |
| MoE backend | `b12x` | B12X MXFP4 kernels |
| KV cache dtype | `fp8` | Default; NVFP4 opt-in for more context |
| GPU memory util | `0.90` | 0.88 + 0.02 → ~250k context in auto mode |
| Load format | `instanttensor` | ~3 min load vs ~20 min |
| Speculative | DSpark, 7 tokens | `Inferact/Kimi-K3-DSpark` |
| CUDAGraph | `FULL_AND_PIECEWISE` | Works with DSpark under B12X |
| max_model_len | `auto` | Auto-fits to ~250k at GMU 0.90 |

---

## Cluster setup

16× DGX Spark (GB10 SM121), connected via RoCE v2 fabric:

| Component | Spec |
|---|---|
| Nodes | 16× DGX Spark (1× GB10 SM121 each, 121 GiB UMA) |
| Switch | MikroTik CRS804-4DDQ |
| Cables | 4× 400G-to-4×100Gbit breakout |
| Fabric | RoCE v2 (NCCL over IB) |
| Model storage | NFS share on one node |
| Total power | ~2.3 kW |

---

## Future work

This is v0. Planned improvements for v1+:

- **DCP16** (decode-context-parallel) — should fix the 100k context decode drop (from ~10 t/s back to ~50+ t/s)
- **32K draft KV tail** — bound the DSpark draft's KV cache to a 32K replicated tail (saves ~6× draft KV memory)
- **Native sm_121 cubins** — the current build uses `12.0f` (family-specific sm_120 forward-compat); native sm_121 requires CMake patches to `CUDA_SUPPORTED_ARCHS`
- **Online MXFP8 quantization** — convert 69 KDA input projections to MXFP8/Marlin (~1.36 GiB/rank savings)
- **Tuned CUDAGraph capture sizes** — per-depth capture buckets for DSpark decode
- **PP=2, TP=8** exploration — may improve throughput for some workloads

### Contributing

Ideas, benchmarks, and PRs are welcome. Open an issue at [github.com/ciprianveg/gb10-kimi-3/issues](https://github.com/ciprianveg/gb10-kimi-3/issues).

---

## Credits

- **[local-inference-lab/vllm](https://github.com/local-inference-lab/vllm)** — the fork with B12X, DSpark DCP16, InstantTensor, and K3-specific kernels
- **[Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark)** — DSpark drafter model
- **[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)** — the model
- **[spark-vllm-docker](https://github.com/ciprianveg/spark-vllm-docker)** — the build system and recipe runner
- **[NVIDIA DGX Spark / GB10 User Forum](https://forums.developer.nvidia.com/t/full-kimi-k3-running-on-16x-gb10-cluster/379174)** — original post and community discussion

---

## License

The Dockerfiles, recipes, and scripts in this repo are MIT-licensed. The vLLM fork and b12x package have their own licenses — see the upstream repos.
