# MODEL_CATALOG — bringing real models up on hipDNN via the injection

Companion to the local build/wiring guide (`LOCAL_DEV.md`). That doc gets the stack *built
and wired*; this one is the living record of **which models we drive through `hipdnn_torch`,
how to stand each one up, and what it proves**. Every entry is a `samples/*_ab.py` harness
over the shared `samples/_census.py` device-time census.

> **gfx1151 note.** The AOT/rocKE catalog kernels are arch-specific; the numbers below were
> taken on an RDNA gfx1151 (Radeon 8060S) ROCm build on Windows. On other arches the routed
> set differs (see *Arch note*). Nothing here hardcodes a machine path — set `COMFYUI_PATH` /
> `HF_HOME` / the model knobs to your own locations.

## Why a catalog (three purposes, per model)

Each model serves one or more of:
1. **hipDNN coverage showcase** — A/B device-time: routed-through-hipDNN vs native PyTorch.
   Always run under `default` (all providers co-loaded); the census reports, per op, whether
   hipDNN routed it and which engine won. We do **not** use `force`: pinning
   `AOT_CATALOG_ENGINE` only gives AOT-specific attribution (a gfx1151-only story) and turns
   every non-AOT-served op into a false decline. The kernels are correctness-first references
   today, so absolute uplift is a later data-only tuning follow-on, not the headline.
2. **Coverage-gap backlog** — each model's fallback/decline report is the prioritized
   list of ops **no loaded engine** (AOT/hipblaslt/MIOpen) serves yet. `hipdnn_torch.report()`
   per model = "what hipDNN still needs."
3. **Correctness / regression** — parity vs native, and catching real corruption (e.g.
   the Wan 2.2 VAE-decode defect, ROCM-27995).

## Arch note (read this before comparing numbers)

The AOT/rocKE catalog kernels are **arch-specific**. The validated gfx1151 families are
RDNA/WMMA (`fmha_wmma`, `gemm_wmma[_universal]`, `conv2d_fprop`, `layernorm2d`,
`rmsnorm2d`, `activation`). On CDNA (gfx942/gfx950) the set differs or may be empty; there
the portable result is the **injection + `default` multi-provider routing** (MIOpen/
hipblaslt serve). This is exactly why the catalog runs everything in **`default`** and not
`force`: `default` asks the portable, arch-independent question ("did hipDNN route this op
at all, by whatever engine won"), whereas `force`-mode AOT attribution is a gfx1151-only
story. See `LOCAL_DEV.md`.

## Running any entry

1. Wire the env (`LOCAL_DEV.md`) — provider `.so/.dll`, frontend dir, 7-family
   `HIPDNN_AOT_CATALOG_DIR`, one-backend rule.
2. Set the model's `<MODEL>_*` knobs (see each entry) and, for ComfyUI-sourced models,
   `COMFYUI_PATH=<your clone>`.
3. `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` normally makes the *native* SDPA baseline a
   real fused kernel (not the math strawman). **gfx1151 caveat:** on the ROCm builds tested
   here the experimental aotriton fused SDPA **crashes for every tested shape** — it enqueues
   a bad launch that poisons the HIP context, so the *next* op dies with `HIP error: invalid
   argument` (independent of `is_causal`/GQA; reproduced with a pristine PATH, so it is not
   our DLL wiring). Leave it **off** until fixed; the math backend is a correct native
   baseline. Re-check per nightly.
4. Run under **`HIPDNN_TORCH_SELECT=default`** with all providers co-loaded (AOT/hipblaslt/
   MIOpen). This is the only mode we use — it measures general hipDNN coverage. Do **not** use
   `force`/`AOT_CATALOG_ENGINE`: it reports AOT-only attribution (gfx1151-specific) and
   mislabels every non-AOT-served op as a decline.
5. Read the per-op `aot=N native=M` census + the parity line + the fallback report. Here
   `aot=N` counts ops hipDNN routed by *any* winning engine; `native=M` is the coverage-gap
   backlog (no loaded engine served → fell back to native).

Random vs real weights: for **routing/coverage**, random weights are sufficient — the
checkpoint changes pixel values, not call sites/shapes/op mix. For **correctness/
regression** and **honest perf**, use real checkpoints (noted per entry).

> **Parity note (changed 2026-08-14).** Earlier snapshots reported `rel=0.0000` because
> almost everything fell back to native (injected == native, bit-identical). Now that GEMM
> (incl. fused bias), SDPA (D=64), and conv actually route through the AOT/WMMA and MIOpen
> kernels, parity is no longer bit-identical — it is **within bf16/f16 tolerance**
> (`rel ≈ 0.01–0.015` bf16, `≈ 0.002` f16). That is the expected, correct signature of ops
> genuinely leaving the native path; a `rel` far outside tolerance would be the regression
> signal.

## What changed 2026-08-14 (read before the older per-model snapshots)

Two catalog changes since the 2026-08-13 snapshots below invalidate the old "everything
declines" story for transformers:

- **SDPA now routes for `D==64`.** The `fmha_wmma_fwd` family was expanded from 2 kernels to
  **6** (f16/bf16 × {non-causal MHA, causal MHA, causal GQA `gqa_ratio=4`}), and — critically —
  **head count `H`/`H_kv` is now unbounded** (grid dim, not a compile-time constant). `D==64`
  is the *only* compile-time shape fact; `S_q/S_kv` need `%16`. So gpt2 (H12 causal), bert
  (H12 non-causal), whisper (H8 non-causal), flux (H24 non-causal), and llama (H32/KV8 GQA-4
  causal) **all route now**. This obsoletes the old "`H==32` exact-equality kills all SDPA"
  root-cause table (kept below, struck through, for history).
- **GEMM: N-D linears fold to 2-D and route, and bias fuses at rung 1.** `GemmAdapter` now
  collapses N-D `[…,K]·[K,N]` linears to 2-D and routes them when tile-aligned, and the AOT
  catalog gained **fused matmul+bias** kernels (24 `.co` = 12 non-bias + 12 bias, f16+bf16),
  gated by a `has_bias` capability fact. Biased `nn.Linear` now fuses in one kernel instead of
  matmul + separate native bias-add. Remaining GEMM declines are **partial-tile** (below).

## Op-family coverage matrix

Columns are the op families the injection routes. Legend: **✓** routes today · **◐** routes
for some shapes, declines others (see entry) · **⚠** exercised but **no loaded engine yet**
(a `default`-mode decline / backlog item) · blank = model doesn't exercise it.
`gemm` = linear/matmul, `conv` = conv2d/3d, `sdpa` = attention, `LN`/`RMS` = norms,
`act` = silu/gelu, `GN`/`BN` = group/batch norm.

| Model | Era | gemm | conv | sdpa | LN | RMS | act | GN/BN | Status |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **minimal_block** | — | ✓ | | ✓ | | ⚠ | ✓ | | ✅ working |
| **microbench_ab** | — | ✓ | ✓ | ✓ | ◐ | ⚠ | ✓ | | ✅ working |
| **sdpa_backends** | — | | | ✓ | | | | | ✅ working |
| **vit_ab** (ViT) | classic | ✓ | | ✓ | ⚠ | | ⚠ | | ⧗ pre-change snapshot |
| **ltx_video_ab** (LTX DiT) | modern | ✓ | | ✓ | ⚠ | ⚠ | ✓ | | ⧗ pre-change snapshot |
| **ltx_vae_ab** / **sd_vae_ab** | modern | | ✓ | | | | ✓ | ⚠ | ⧗ pre-change snapshot |
| **llama_block_ab** (Llama/Mistral) | modern | ✓ | | ✓ | | ⚠ | ✓ | | ✅ working |
| **wan22_vae_decode_ab** (ROCM-27995) | modern | | ✓ | | | | ◐ | ⚠ | ✅ working |
| **sdxl_unet_ab** | modern | ◐ | ✓ | ◐ | ⚠ | | ✓ | ⚠ | ✅ working |
| **gpt2_block_ab** | classic | ✓ | | ✓ | ⚠ | | ⚠ | | ✅ working |
| **resnet50_ab** | classic | ⚠ | ✓ | | | | | ⚠ | ✅ working |
| **bert_block_ab** | classic | ✓ | | ✓ | ⚠ | | ⚠ | | ✅ working |
| **sd15_unet_ab** | classic | ◐ | ✓ | ⚠ | | | ✓ | ⚠ | ✅ working |
| **convnext_ab** | modern | ◐ | ◐ | | ◐ | | ⚠ | | ✅ working |
| **whisper_enc_ab** | modern | ◐ | | ✓ | ⚠ | | ⚠ | | ✅ working |
| **flux_dit_ab** | modern | ◐ | | ✓ | | ⚠ | ◐ | | ✅ working |

The remaining ⚠ columns are the backlog: RMSNorm (no rank routes), most LayerNorm (only
rank-2 `[M,N]` routes), exact-erf GELU (variant gap), and GroupNorm/BatchNorm (no engine) —
the clearest signals of what hipDNN still needs on gfx1151.

## Per-model bring-up template

Each entry below follows this shape:

> **`<harness>.py`** — *<era>, <domain>*
> - **Exercises:** <op families> — *why it's in the catalog*
> - **Source:** ComfyUI `comfy.*` / `transformers` / `diffusers` / `torchvision`
> - **Checkpoint:** <real ckpt + size + gated?> · or *random weights OK for routing*
> - **Run:** `<MODEL>_* env knobs` + command
> - **Acceptance:** what should route, parity tol, known declines
> - **Status / notes**

---

## Working entries

> Entries marked **(re-validated 2026-08-14)** were re-run after the SDPA + bias-fusion
> changes. Entries marked **(pre-change snapshot)** carry 2026-08-13 numbers and are
> ComfyUI-dependent; their routed op mix is unchanged but SDPA/GEMM will now route more —
> re-run to refresh.

### `vit_ab.py` — *classic, vision transformer* — (pre-change snapshot)
- **Exercises:** gemm, sdpa, layernorm, gelu — clean transformer-encoder path.
- **Source:** self-contained. **Checkpoint:** random weights OK.
- **Run:** `VIT_IMG VIT_PATCH VIT_DIM VIT_DEPTH VIT_HEADS VIT_B VIT_ITERS VIT_WARMUP`.
- **Acceptance:** linear/sdpa/layernorm/gelu route; parity within dtype tol. Post-change,
  the `D=64` self-attention and folded linears are expected to route (like bert); re-run to
  confirm.

### `ltx_video_ab.py` — *modern, video DiT* — (pre-change snapshot)
- **Exercises:** gemm, sdpa, rmsnorm (+ layernorm) via `comfy.ldm.lightricks.model.LTXVModel`.
- **Source:** ComfyUI (`COMFYUI_PATH`). **Checkpoint:** random weights OK (op mix unchanged).
- **Run:** `COMFYUI_PATH=... LTX_LAYERS LTX_FRAMES LTX_H LTX_W LTX_CTX LTX_ITERS` +
  `--ops linear,rmsnorm,sdpa`.
- **Acceptance:** selected ops route; output not corrupted; per-op census + report.

### `ltx_vae_ab.py` / `sd_vae_ab.py` — *modern, VAE decode* — (pre-change snapshot)
- **Exercises:** conv2d, SiLU (+ ⚠ GroupNorm backlog).
- **Source:** ComfyUI. **Checkpoint:** random weights OK for routing.
- **Run:** `LTX_VAE_* ` / `SD_VAE_H SD_VAE_W SD_VAE_Z SD_ITERS` (+ `COMFYUI_PATH`).

### `llama_block_ab.py` — *modern, LLM decoder* — (re-validated 2026-08-14)
- **Exercises:** gemm (q/k/v/o + SwiGLU), causal sdpa (RoPE + GQA), rmsnorm, silu — the
  canonical modern-decoder recipe; the best 1:1 fit to the hipDNN op families.
- **Source:** self-contained (`LlamaDecoderLayer` stack). **Checkpoint:** random weights OK
  (nn.Linear default init + RMSNorm w=1 keep bf16 parity meaningful); a real small decoder
  (Qwen2.5-0.5B / TinyLlama) only changes values, not the routed shapes.
- **Run:** `LLAMA_DIM LLAMA_HEADS LLAMA_KV_HEADS LLAMA_MLP LLAMA_SEQ LLAMA_LAYERS LLAMA_B
  LLAMA_WARMUP LLAMA_ITERS` + `--ops linear,rmsnorm,sdpa,silu`. Set
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0` (see Running §3).
- **Validated (2026-08-14, gfx1151, dim=2048 H32/KV8 mlp8192 seq512 L4, bf16):** correctness
  **OK rel=0.0133**. **Now routes:** the **causal GQA SDPA** (H32/KV8 → `gqa_ratio=4`,
  `aot=4 native=0`) via the new fmha GQA kernel, and Llama's **bias-free linears** (28,
  `native=0`) via the `has_bias:{equals:false}` gate — folded to 2-D and served by the WMMA
  GEMM. **Still declines:** 3-D `rms_norm` (backlog, no engine) — matches the
  `test_parity.py` xfail. **silu routes** (AOT). A/B ≈ **0.62×** — this is the known
  WMMA-vs-rocBLAS GEMM-quality gap (native uses torch's own rocBLAS; hipBLASLt doesn't claim
  gfx1151 matmul on Windows), **not** a regression and not closable by selection or fusion.

### `wan22_vae_decode_ab.py` — *modern, VAE decode / ROCM-27995 defect locus* — (re-validated 2026-08-14)
- **Exercises:** the large VAE-decode conv stack — conv3d (CausalConv3d) + conv2d upsample
  (`{1,256/128,704,1280}→{1,3,704,1280}`) + SiLU (+ ⚠ GroupNorm). **This is the ROCM-27995
  defect locus**, carved out so it runs in seconds instead of the 11–14 h full 14B T2V job.
- **Source:** ComfyUI `comfy.ldm.wan.vae.WanVAE` (Wan 2.1 VAE = the one Wan 2.2 14B-T2V reuses;
  z_dim=4, 8× spatial / 4× temporal). **Checkpoint:** random weights OK for routing/coverage;
  the *actual* 0.209 corruption needs real Wan 2.1 VAE weights (`--vae`/`WAN_VAE`) + a real
  pre-VAE latent (`WAN_LATENT`) — Option 3, a deliberate download. Option 1 (synthetic latent
  + random weights) validated below. See `project_rocm27995` memory for shapes/solvers.
- **Run:** `COMFYUI_PATH=<clone> WAN_H=704 WAN_W=1280 WAN_F WAN_WARMUP WAN_ITERS` +
  `--ops conv2d,conv3d,silu`. No aotriton flag needed — this decode has no SDPA.
  **ComfyUI checkout note:** if your clone is AMD's AIMDO build, its
  `comfy.{model_management,ops,model_patcher}` `import comfy_aimdo.*` (an in-tree module, not
  on PyPI). Every AIMDO *runtime* use is gated behind `hasattr(s,"_v")` (set only by ComfyUI's
  own state-dict loader), so a directly-instantiated `WanVAE()` never hits it — the harness
  injects empty `comfy_aimdo.*` stubs into `sys.modules` to satisfy the imports. Deps pulled
  into the venv: `einops psutil safetensors pillow tqdm`.
- **Validated (2026-08-14, gfx1151, out 704×1280, 1 latent frame, random weights, bf16):**
  output `(1,3,1,704,1280)`; correctness **OK rel=0.0** (no divergence with random weights —
  expected; the 0.209 corruption is weight/latent-dependent, needs Option 3). **All convs
  route through hipDNN via `MIOPEN_ENGINE`** — incl. the ROCM-27995 fingerprint
  `Cpg=256,K=128,3×3`; AOT declines these (rank-4 NHWC-groups==1 / rank-5 out of its
  allowlist) so MIOpen serves under `default`. **SiLU splits**: AOT serves most, a few small
  numels decline (backlog). **A/B ≈ 0.012×** — a coverage result and a real perf-backlog
  signal: the injected conv path re-pays MIOpen's immediate-mode solver *find* per unique
  config per call (no plan cache across the wrapper). Device-time census (native): conv3d
  dominates (~91%).
- **Next (Option 3, needs user OK for the download):** real Wan 2.1 VAE weights + one saved
  pre-VAE latent → diff native (suspect MIOpen GemmFwdRest) vs injected for the silent
  0.209-vs-3e-5 corruption. Doubles as the model-level repro the ticket needs.

### `sdxl_unet_ab.py` — *modern, diffusion UNet* — (re-validated 2026-08-14)
- **Exercises:** conv2d + SiLU + GroupNorm (ResNet blocks) interleaved with gemm + non-causal
  SDPA + LayerNorm + GELU (Transformer2D blocks) — the canonical conv+attention net and the
  richest single-model op mix in the catalog. Strong ⚠ GroupNorm backlog signal.
- **Source:** `diffusers.UNet2DConditionModel`, built `from_config` with the **real SDXL base
  1.0 UNet config embedded in the harness** — no checkpoint download; random weights are fine
  for routing/coverage. SDXL's `addition_embed_type="text_time"` forward gets synthesized
  `added_cond_kwargs` (`text_embeds` [B,1280] + `time_ids` [B,6]) + `encoder_hidden_states`
  [B,77,2048]. Needs `diffusers` in the venv.
- **Run:** `SDXL_H SDXL_W` (latent spatial, default 64 = ~512px; native SDXL is 128=1024px)
  `SDXL_B SDXL_WARMUP SDXL_ITERS` + `--ops linear,conv2d,sdpa,silu`. Set
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0` (SDXL uses SDPA, see Running §3).
- **Validated (2026-08-14, gfx1151, latent 64×64 ~512px, random weights, bf16):** output
  `(1,4,64,64)`, correctness **OK rel=0.0124**. **conv2d fully routes via `MIOPEN_ENGINE`**;
  **SiLU fully routes via `AOT_CATALOG_ENGINE`**; **most biased linears now fuse** (`linear
  aot=571`) via the folded-2-D + bias-fusion path. **Residual declines** (`native≈204`):
  cross-attention SDPA (`S_kv=77`, not `%16`), non-aligned linear tiles, and GroupNorm (no
  engine). A/B ≈ **1.167×**. This is the richest single-model demonstration of the
  fold+fuse+MIOpen path all firing together.

### `resnet50_ab.py` — *classic, CNN* — (re-validated 2026-08-14)
- **Exercises:** pure **NCHW conv2d** (bottleneck 1×1/3×3/1×1 stack) + one genuinely 2-D
  `F.linear` classifier + BatchNorm (⚠ backlog) + ReLU (not a routed family). The NCHW +
  true-rank-2-GEMM combination is what makes it distinct from the diffusion nets.
- **Source:** self-contained (bottleneck `[3,4,6,3]` in plain `torch.nn`) — deliberately **no
  torchvision**, so it can't drag a mismatched CUDA/CPU torch into the ROCm nightly venv.
  **Checkpoint:** random weights OK (eval BatchNorm running_mean=0/var=1 keeps activations
  finite).
- **Run:** `RESNET_IMG (224) RESNET_B (8) RESNET_WARMUP RESNET_ITERS` + `--ops conv2d,linear`.
  No aotriton flag needed — no SDPA.
- **Validated (2026-08-14, gfx1151, B=8 224×224, random weights, bf16):** output `(8,1000)`,
  correctness **OK rel=0.0**. **conv2d fully routes via `MIOPEN_ENGINE`** (incl. the
  `Cpg=3,K=64,7×7` stem). **The 2-D classifier GEMM (`M=8, K=2048, N=1000`) still declines**
  and now falls to native **gracefully** (rung-1 bias declined → rung-2 matmul-only also
  declined → native, via the `base.py` try/except fix) — no crash. **Root cause** (see GEMM
  drill-down): the fc's `M=8` and `N=1000` are not tile-aligned and there is no partial-tile
  epilogue, so no WMMA GEMM kernel admits it; hipBLASLt is disabled on gfx115x/Windows
  (#9962). A/B ≈ **0.964×** (near parity — conv dominates device time).

### `gpt2_block_ab.py` — *classic, LLM decoder* — (re-validated 2026-08-14, bf16 + f16)
- **Exercises:** gemm (fused QKV / proj / MLP), causal SDPA (MHA, no GQA/RoPE), **LayerNorm**,
  **GELU** — the classic pre-RMSNorm decoder, complementing Llama on the **LayerNorm + GELU**
  axis. Geometry = GPT-2 base (dim=768, 12 heads, head_dim=64, mlp=3072).
- **Source:** self-contained (`nn.Linear` block stack) — **no `transformers`**, which also
  sidesteps GPT-2's `Conv1D` (a transposed matmul, not `nn.Linear`) so the routed op mix is
  unambiguous. **Checkpoint:** random weights OK.
- **Run:** `GPT2_DIM GPT2_HEADS GPT2_MLP GPT2_SEQ GPT2_LAYERS GPT2_B GPT2_WARMUP GPT2_ITERS` +
  `--ops linear,layernorm,sdpa,gelu` `--dtype {bf16,f16}`. Set
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0` (GPT-2 uses SDPA, see Running §3).
- **Validated (2026-08-14, gfx1151, dim=768 H12 mlp3072 seq512 L6, random weights):**
  - **bf16:** correctness **OK rel=0.0133**; **all 24 linears fuse** (`biased=24 fused=24
    native=0`, `AOT_CATALOG_ENGINE`); **causal SDPA routes** (`Sq=Skv=512,D=64, aot=6
    native=0`) via the new causal-MHA fmha kernel; **A/B 1.310×**. Device census: sdpa 44.4%,
    linear 39.2%, layernorm 10.8%, gelu 5.6%.
  - **f16:** correctness **OK rel=0.0017**; identical routing; **A/B 1.285×**.
  - **Still declines (backlog):** 3-D `layer_norm` (native=13) and exact-erf `gelu`
    (native=6) — the latter is a **variant** gap, not a coverage gap (drill-down below).
- **GELU variant drill-down (source + empirically confirmed):** the activation family ships
  **6 `gelu_tanh` kernels** (f16/bf16 × 3 tile configs) alongside 6 `silu` kernels — but **no
  exact-erf GELU**. `ActivationAdapter::activationToken` maps `SWISH_FWD→"silu"` and
  `GELU_APPROX_TANH_FWD→"gelu_tanh"`; exact-erf `GELU_FWD` returns `nullopt` — a **deliberate
  fail-closed decline** (routing erf to the tanh kernel would compute a *different function*).
  This sample calls `F.gelu` with the default `approximate='none'` (erf) → declines. **Proof
  of the flip side:** a copy with `approximate='tanh'` routes fully (`gelu aot=6 native=0`).
  Gap-closing lever = a rocKE exact-erf gelu builder op (+6 kernels); nothing in the adapter
  or selection path changes. (Real GPT-2/BERT use the tanh approximation — `gelu_new` — so a
  fidelity-matched harness would route today; exact-erf is deliberate to surface the gap.)

### `bert_block_ab.py` — *classic, bidirectional encoder* — (re-validated 2026-08-14)
- **Exercises:** gemm (fused QKV / proj / MLP), **non-causal** SDPA (full/bidirectional,
  `is_causal=False`), **LayerNorm** (post-norm), **GELU** — the non-causal counterpart to
  `gpt2_block_ab.py`, isolating whether dropping causality changes what SDPA will serve.
  Geometry = BERT-base (dim=768, 12 heads, head_dim=64, mlp=3072).
- **Source:** self-contained (`nn.Linear` block stack) — **no `transformers`**. **Checkpoint:**
  random weights OK. Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0`.
- **Validated (2026-08-14, gfx1151, dim=768 H12 mlp3072 seq512 L6, random weights, bf16):**
  correctness **OK rel=0.0114**. **All 24 linears fuse** (`native=0`); **non-causal SDPA
  routes** (`D=64, aot=6 native=0`) via the non-causal-MHA fmha kernel — the direct
  confirmation that **head count is no longer a blocker** (H12 ≠ 32 routes fine now).
  **Still declines:** 3-D `layer_norm`, exact-erf `gelu`. A/B ≈ **1.186×**.

### `sd15_unet_ab.py` — *classic diffusion UNet (conv-proj sibling of SDXL)* — (re-validated 2026-08-14)
- **Exercises:** ResNet-conv + SiLU + GroupNorm + Transformer2D (gemm + non-causal SDPA), at
  SD1.5 geometry (`cross_attention_dim=768`, no `text_time`). **Key contrast with SDXL:**
  `use_linear_projection` defaults **False**, so the Transformer2D in/out projections are 1×1
  **convs**, not `Linear` — a controlled conv-vs-gemm swap between the two UNet entries.
- **Source:** `diffusers.UNet2DConditionModel.from_config` (SD1.5 config embedded). Random
  weights OK. Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0`.
- **Validated (2026-08-14, gfx1151, latent 64×64, random weights, bf16):** output `(1,4,64,64)`,
  correctness **OK rel=0.0134**. **conv2d fully routes via `MIOPEN_ENGINE`** (aot=98 native=0,
  incl. every 1×1 projection conv — exactly the work SDXL declines as `Linear`). **SiLU fully
  routes to AOT** (aot=68). **Many linears fuse** (`aot=128`), the rest are non-aligned tiles
  (`native≈80`). **SDPA still 100% native** — SD1.5 head dims are **D=40/80/160, not 64**, so
  no fmha kernel matches (this is now the *only* SDPA blocker here; head count is fine). SDPA
  is the dominant device cost (~67%) → the single biggest remaining routing opportunity for
  this model, gated on compiling more head dims. GroupNorm all decline. A/B ≈ **0.973×**.

### `convnext_ab.py` — *modern CNN (conv × transformer-norm bridge)* — (re-validated 2026-08-14)
- **Exercises:** depthwise `Conv2d(k7,groups=dim)`, channels-last `LayerNorm`, pointwise MLP
  `Linear`, `GELU` — bridges the pure-CNN (resnet) and pure-transformer axes and probes two
  shapes nothing else does: **grouped/depthwise conv** and a **4-D `[B,H,W,C]` linear/LN**.
- **Source:** self-contained (`torch.nn`; no `torchvision`). Random weights OK. No SDPA → no
  aotriton flag needed.
- **Validated (2026-08-14, gfx1151, B=8 224², random weights, bf16):** output `(8,1000)`,
  correctness **OK rel=0.0038**. Four results: **(1)** depthwise convs (`Cpg=1`, 7×7) + the
  patchify stem route via `MIOPEN_ENGINE` (aot=10), **but the strided 2×2 `groups=1`
  downsample convs are refused by MIOpen itself** (`Requested convolution is not supported`) →
  native (aot=0/native=2) — a genuine *MIOpen* gfx1151 gap, not an AOT one. **(2)** pointwise
  MLP linears **partially fuse** (`aot=18`): stages whose folded `M` is tile-aligned fuse
  (`fused`), stages where `M` is `%16` but not `%64` fall to the reference GEMM kernel — which
  is bias-free — so the matmul routes but the bias goes native (`fused_declined`); the
  `N=1000` fc declines entirely (not `%16`). **(3)** the **2-D head `LayerNorm(384)` routes to
  AOT** (aot=1) — the first LayerNorm to route — while every channels-last **4-D** LayerNorm
  declines, pinning the AOT LN adapter to rank-2 `[M,N]`. **(4)** exact-erf `gelu` declines
  (variant gap). A/B ≈ **0.801×**; census conv2d 49.0%, layer_norm 22.9%, linear 21.5%.

### `whisper_enc_ab.py` — *modern audio encoder* — (re-validated 2026-08-14)
- **Exercises:** `Conv1d` mel frontend + non-causal transformer blocks (gemm, SDPA, LayerNorm,
  GELU) at Whisper-base geometry (dim=512, 8 heads, head_dim=64). Its unique data point: the
  frontend is **`Conv1d`, which is not a routed family** (only `F.conv2d`/`F.conv3d` are
  intercepted) — it documents a conv *rank* the catalog doesn't cover.
- **Source:** self-contained (`torch.nn`; no `transformers`). Random weights OK. Set
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0`.
- **Validated (2026-08-14, gfx1151, mel[1,80,800]→400 tokens, random weights, bf16):** output
  `(1,400,512)`, correctness **OK rel=0.0145**. **Non-causal SDPA now routes** (`Sq=Skv=400,
  D=64, aot=6 native=0`). **GEMM is the instructive case:** the folded token dim is **M=400,
  which is `%16` but not `%64`** — so the bias-capable *universal* WMMA kernel declines and the
  matmul falls to the **reference `gemm_wmma` kernel** (`%16`), which is `has_bias:false`. Net:
  the matmul routes to AOT but the bias adds go native → `linear aot=24 native=24
  fused_declined=24`. This is the cleanest example of the **reference-kernel-can't-fuse-bias**
  decline (distinct from a full tile-miss). LayerNorm (native=13) and exact-erf gelu
  (native=8) decline; the `Conv1d` frontend stays native by design. A/B ≈ **0.651×**.

### `flux_dit_ab.py` — *modern rmsnorm-heavy DiT* — (re-validated 2026-08-14)
- **Exercises:** a scaled-down Flux single-stream DiT — adaLN-Zero modulation (`SiLU`+`Linear`),
  **QK-RMSNorm** inside attention (`q`/`k` RMSNorm'd over head_dim before SDPA), gemm-heavy
  attn/MLP, non-causal SDPA, GELU. Its unique probe: **RMSNorm at a 4-D `[B,H,S,D]`** activation
  (head_dim=64), a different rank than Llama's 3-D `[B,S,D]` token RMSNorm.
- **Source:** self-contained (`torch.nn`; no `diffusers` — real FLUX.1 is ~24 GB). Random
  weights OK. Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0`.
- **Validated (2026-08-14, gfx1151, dim=1536 H24 seq256 L4, random weights, bf16):** output
  `(1,256,1536)`, correctness **OK rel=0.0077**. **Now routes:** non-causal SDPA (`D=64,
  aot=4`), the small adaLN `SiLU` (aot=4), and **most linears fuse** (`aot=16`, `fused`) — the
  token dim M=256 is `%64`. **Declines:** the adaLN modulation linear `K=1536,N=9216` declines
  entirely (`native=8`) — a per-sample conditioning vector whose `M` is not tile-aligned (full
  partial-tile miss, no reference fallback); **4-D QK `rms_norm` (N=64, native=8)** — RMSNorm
  doesn't route at head-dim rank any more than at token rank; exact-erf `gelu` (native=4).
  A/B ≈ **0.700×**; census linear 47.9%, sdpa 24.4%, rms_norm 14.9%.

### `sdpa_backends.py`, `microbench_ab.py`, `minimal_block.py`
- Micro/attention-backend A/B and the smoke block. `microbench_ab.py --dtype --iters`.

---

## Coverage synthesis (2026-08-14, gfx1151, `default`)

All entries pass correctness within dtype tolerance. Under `default` on gfx1151, the routed
set has grown substantially since 2026-08-13. **Routes today:**

- **conv2d / conv3d → `MIOPEN_ENGINE`** — all CNN/VAE/UNet convs, incl. 1×1 projection convs
  (sd15) and depthwise convs (convnext). *MIOpen* itself refuses a few shapes (convnext's
  strided 2×2 `groups=1` downsample) → native; that's a MIOpen gap, not an AOT one.
- **SiLU → `AOT_CATALOG_ENGINE`** — every model with SiLU (llama, wan, sdxl, sd15, flux).
- **GEMM (incl. fused bias) → WMMA** — tile-aligned linears fold to 2-D and route; biased
  linears **fuse matmul+bias in one kernel** (gpt2/bert fully; sdxl/sd15/flux/convnext
  partially). See the GEMM drill-down for what still declines.
- **SDPA (`D==64`) → `fmha_wmma_fwd`** — non-causal MHA (bert, whisper, flux), causal MHA
  (gpt2), and causal GQA-4 (llama) all route. Head count is unbounded now.
- **LayerNorm → AOT, but rank-2 `[M,N]` only** — the lone routed LN is convnext's 2-D head
  `LayerNorm(384)`; every 3-D/4-D LN declines.

**Remaining backlog** (no loaded engine serves → native):
- **GEMM partial-tile / bias declines** — non-aligned `M`/`N` with no partial-tile epilogue
  (resnet fc `M=8,N=1000`; flux adaLN `M=1`; convnext `N=1000`), and the
  reference-kernel-can't-fuse-bias case where `M` is `%16` but not `%64` (whisper `M=400`,
  convnext stage-2) → matmul routes, bias goes native. hipBLASLt stays disabled (#9962).
- **SDPA `D≠64`** — sd15's `D=40/80/160` heads, and masked/short-context cross-attention
  (`S_kv=77`, not `%16`, in sdxl/sd15). Each needs a new compiled geometry.
- **all RMSNorm** — 3-D token norm (llama) and 4-D QK-norm (flux) both decline.
- **exact-erf GELU** — a **variant** gap: 6 `gelu_tanh` kernels exist, no exact-erf builder op,
  so `ActivationAdapter` fail-closed declines `GELU_FWD`. (`approximate='tanh'` routes.)
- **Group/BatchNorm** — no engine.

## Why some GEMM and SDPA still decline — root-cause drill-down (2026-08-14, gfx1151, `default`)

The AOT `.co` kernels are produced with **static, tile-aligned constraints and no boundary
masking**, so a shape either matches a kernel or gets *no candidate* → `RuntimeError: Failed
to get ranked engine ids: No engine configurations available for the graph`.

### GEMM — three decline modes now (post fold+fuse)

1. **Full tile-miss.** Both `M` and/or `N` fail the *loosest* kernel's `multiple_of`. The bf16
   GEMM `.co` carry hard `multiple_of` (no partial-tile epilogue): reference `gemm_wmma` needs
   `M%16, N%16, K%16`; `gemm_wmma_universal` needs `M%64|128, N%64|128, K%32`. resnet fc
   (`M=8, N=1000`) misses both; flux adaLN (`M=1`) misses both → full native.
2. **Reference-only → bias goes native.** When `M` is `%16` but not `%64` (whisper `M=400`,
   convnext stage-2 `M=1568`), the bias-capable *universal* kernel declines, so the matmul
   falls to the reference `gemm_wmma` kernel — which is `has_bias:{equals:false}` — and the
   bias-add goes native. Shows as `aot>0 native>0 fused_declined>0`.
3. **hipBLASLt globally disabled (not a shape decision).** `HipblasltMatmulPlanBuilder::
   isApplicable` runs `REJECT_IF_WORKAROUND_ISSUE_9962(handle)` **first**: on gfx115x/Windows
   hipBLASLt segfaults *inside its own heuristic while building the plan* (ROCm/rocm-libraries
   #9962), and because `isApplicable()` probes support by constructing a plan, the probe itself
   would crash — so the workaround early-returns `false` for **every matmul** on `gfx115*` on
   Windows (compile-time gated to `_WIN32`, fail-closed on arch-query failure). No hipBLASLt
   GEMM path exists on this arch/OS regardless of shape.

- **Gap-closing levers:** partial-tile masking epilogues (relax `multiple_of` → 1) close modes
  1 and 2; adding bias to the reference kernel would also close mode 2. hipBLASLt stays
  disabled until #9962 is fixed upstream.

### SDPA — `D==64` routes; the residual is head-dim + masking geometry

The `fmha_wmma_fwd` family is now **6 kernels** (f16/bf16 × {non-causal MHA, causal MHA,
causal GQA `gqa_ratio=4`}). **`D==64` is the only compile-time shape fact; `H`/`H_kv` are
unbounded** (`min:1`, head count is the grid `y` dim), and `S_q/S_kv` need `%16`. So any
`D=64` attention routes if its (causal, gqa_ratio) pair has a kernel. What still declines:

| model | D | q/kv heads | causal | routes now? | if not, why |
|---|:--:|:--:|:--:|:--:|---|
| bert | 64 | 12/12 | no | ✓ non-causal MHA | — |
| whisper | 64 | 8/8 | no | ✓ non-causal MHA | — |
| flux | 64 | 24/24 | no | ✓ non-causal MHA | — |
| gpt2 | 64 | 12/12 | **yes** | ✓ causal MHA | — |
| llama | 64 | 32/**8** | **yes** | ✓ causal GQA-4 | — |
| sdxl | 64 | 5/10/20 | no | ◐ self-attn only | cross-attn `S_kv=77` not `%16` |
| sd15 | **40/80/160** | — | no | ✗ | **`D≠64`** (+ cross-attn `S_kv=77`) |

- **Residual SDPA blockers:** `D≠64` head dims (sd15), and short/masked cross-attention
  (`S_kv=77`). Each is a **distinct compiled kernel**, not a runtime knob — the gap-closing
  lever is *producing more geometries* (more head dims; a `%1` or padded `S_kv`), plus GQA
  ratios other than {1, 4} and non-causal GQA if a model needs them.
- **No fallback fused-attention path:** hipBLASLt is not a fused-attention provider, and the
  experimental aotriton fused SDPA is deliberately off (poisons the HIP context, Running §3).
  So SDPA is strictly **AOT-fmha-or-native**; a declining SDPA falls to PyTorch's unfused math.

> **Historical (2026-08-13, now obsolete) — the old SDPA root cause.**
> ~~The `fmha_wmma_fwd` family is 2 kernels (one f16, one bf16, identical geometry) gating on
> exact equality: `D==64, H==32, H_kv==32, causal==false, S%16`. `H==32` exact-equality was
> the dominant killer — 5 models had the right `D=64` and were non-causal yet all missed
> because real blocks run 8/12/24 heads, never 32.~~ This was fixed 2026-08-14 by unbounding
> `H`/`H_kv` and adding causal + GQA kernel variants; see the table above.

## Checkpoint provisioning

- Downloads are **deliberate and per-model** — confirm the source (e.g. Hugging Face)
  is reachable first, and prefer a shared cache dir (`HF_HOME`) off your system drive.
- **ComfyUI-sourced models** (`ltx_*`, `sd_vae`, `wan22_*`) need a ComfyUI clone; point
  `COMFYUI_PATH` at it. If it's AMD's AIMDO build, the harness's `comfy_aimdo` stubs handle the
  in-tree imports (see the wan entry).
- Most catalog entries run on **random weights** (routing/coverage is shape-driven). Real
  checkpoints are only needed for honest perf and for the ROCM-27995 corruption repro (wan,
  Option 3).
