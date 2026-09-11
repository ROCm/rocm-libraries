# Replicating the gfx950 attention best-of-breed selection

This captures the state needed to reproduce hipDNN autotune picking among **flyDSL**,
**rocKE (Gfx950AttentionDense)**, and **AITER-ASM** for SDPA on gfx950 (MI355X), with
PyTorch SDPA forced through hipDNN. Throwaway PoC — hardcoded absolute paths, branch
`users/brpepers/hipdnn-flydsl-poc`. **No measured performance numbers live in the repo**
(compliance); methodology and levers only.

## Engine coverage in this snapshot

- **flyDSL** — `flash_attn_dualwave_swp_gfx950_kernel_0` (FlyDSL 0.3.2), head_dim 128,
  bf16 + fp16, causal + (h8/h16) noncausal. One HSACO per `(num_query_heads, num_kv_heads,
  causal, dtype)`; `seq_len`/`batch` are runtime scalars. 21 registered instances (see the
  `.kdp.json`), covering the causal-prefill head-configs the sweep exercises
  (8, 16, 28/4, 32/8, 32/32, 40/8, 40/40, 64/4, 64/8, 128/8) plus a `waves_per_eu=4` knob
  variant of 32/8 bf16. Baked knobs: `waves_per_eu=2`, dualwave-SWP stagger ON,
  `block_m=256`, `block_threads=512`.
- **rocKE dense** — shipped `gfx950_attention_dense` descriptor set, hd128, causal prefill,
  packed with the persistent + wide_lds_dma fast path (workload-relevant subset, ~1694 specs).
- **AITER-ASM** — non-causal + decode (`seq_q=1`), hd128/192. Causal is served by AITER-CK
  upstream, which is **not** pulled into hipDNN here, so AITER's causal wins are structurally
  unreachable in this build.

For a given SDPA shape, `kernel_match` filters each engine's family to the instance whose
baked config matches, and hipDNN's exhaustive autotune benchmarks the survivors and ranks
them by `robust_time_ms`. No matching instance -> the engine honestly reports unsupported.

## Regenerate the artifacts (fresh gfx950 checkout)

Binaries (flyDSL `*.hsaco`, the rocKE runtime kpack tree) are **not** committed — regenerate
them from the committed sources/scripts.

1. **Build provider + backend** (per `flydsl_poc_scratch/PHASE_A_NOTES.md`):
   `env -u ROCM_PATH cmake --preset hip-kernel-provider -G Ninja -DROCM_PATH=<sdk>
   -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPDNN_ENABLE_SDPA=ON
   -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=<local kpack python>` then `cmake --build build -j`.

2. **Regenerate the flyDSL attention HSACOs** (needs the flyDSL venv):
   ```
   /home/AMD/brpepers/flydsl-venv/bin/python \
       flydsl_poc_scratch/flydsl_build/build_all_flydsl_attention.py
   ```
   Rebuilds exactly the 21 HSACOs the `.kdp.json` references, next to the pack.

3. **Regenerate the rocKE runtime kpack** (needs the base ROCm SDK + kpack python):
   ```
   bash flydsl_poc_scratch/rocke_build/build_rocke_all_pack.sh
   ```
   Reads `rocke_pack_src_all/` (committed staged specs) -> writes `rocke_pack_out_all/gfx950`,
   which is what `run_py.sh` points `HIPDNN_DESCRIPTOR_RUNTIME_DIR` at by default.

## Drive selection through PyTorch SDPA

`flydsl_poc_scratch/app/run_py.sh` is the env wrapper (provider .so, frontend bindings,
flyDSL descriptor dir, rocKE runtime pack, AITER-ASM catalog). A harness that installs the
SDPA override and runs `hipdnn_torch.tuning(["sdpa"])` gets a per-shape ranked
`(engine_name, robust_time_ms)` list; `HIPDNN_TORCH_TUNE=tune` runs the exhaustive sweep.
Force-pin one engine with `HIPDNN_TORCH_SELECT=force HIPDNN_TORCH_ENGINE=hipkernel:FlydslAttention`.

## Files in this snapshot

- `flydsl_descriptors_attention/flydsl_attention/flydsl_attention.kdp.json` — the 21-instance
  flyDSL attention family (metadata-keyed match).
- `flydsl_build/build_all_flydsl_attention.py` — regenerate the referenced HSACOs from the pack.
- `flydsl_build/build_flash_attn_real.py` — single-config flyDSL attention builder (knobs:
  `--kv`, `--waves`, `--no-stagger`).
- `rocke_build/build_rocke_all_pack.sh` + `rocke_pack_src_all/` — rocKE dense runtime pack.
- `app/run_py.sh` — env wrapper for the hipdnn_torch harnesses.
