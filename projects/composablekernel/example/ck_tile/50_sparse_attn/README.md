# sparse attention

This folder contains examples for sparse attention using ck_tile tile-programming. Four variants are exposed through one executable: `jenga`, `vsa`, `sparge`, and quantized `sparge_sage`.

## build
```
# 1. In the root of composable_kernel project, create the build directory.
[~/composable_kernel] mkdir build && cd build
# 2. In the build directory, run the CMake wrapper script to generate the build system files. Replace <arch> with the gfx architectures string.
[~/composable_kernel/build] ../script/cmake-ck-dev.sh .. <arch> -G Ninja
# 3. In the build directory, run the build system recipe.
[~/composable_kernel/build] ninja tile_example_sparse_attn_fwd
```
Running the build recipe will produce the executable `tile_example_sparse_attn_fwd`. The executable resides in the `bin` subdirectory of the build directory. Re-running ninja after editing kernel or codegen sources auto-detects `.py` codegen changes and regenerates the corresponding `.cpp` instances.

> [!NOTE]
> `cmake-ck-dev.sh` is a CMake wrapper.
>
> The first argument is the path to composable_kernel sources.
>
> The second argument is the gfx architectures string (e.g. "gfx950" or "gfx90a;gfx942").
>
> The remaining arguments are optional and are passed through to CMake.
> E.g. `-G Ninja` specifies ninja as the build system.

## kernel
Each sparse variant is realized as its own grid-wise kernel template under `include/ck_tile/ops/sparse_attn/kernel/`. They share the FMHA forward kernel skeleton from `01_fmha` and only differ in how K-block selection is encoded:

* `FmhaFwdJengaKernel` reads a one-hot `[B, H, Q_blks, K_blks]` mask and skips inactive K blocks.
* `FmhaFwdVsaKernel` reads a delta-encoded LUT of active K-block indices plus a valid-count buffer.
* `FmhaFwdSpargeKernel` is sparge's attention pass; an additional preprocess + mask-prediction pair produces the LUT at runtime, prefixed by a K-mean reduction when `-smooth_k=1`.
* `FmhaFwdSpargeSageKernel` is the quantized variant: the same sparge preprocess + mask-prediction, then an INT8/FP8-MFMA attention pass with per-channel FP8 V (SageAttention-style). Requires gfx950.

Each kernel takes a `FmhaPipeline` template parameter — one of the block tile pipelines under `include/ck_tile/ops/sparse_attn/pipeline/` — plus an `EpiloguePipeline` for the final store. Replacing the pipeline type is enough to switch scheduling strategy.

## codegen
To speed up compile time we instantiate the kernels into separate `.cpp` files so CMake/Make can compile them in parallel. This is driven by `generate.py`, with one per-variant module under `codegen/ops/`: `fmha_fwd_jenga.py`, `fmha_fwd_vsa.py`, `fmha_fwd_sparge.py`, `fmha_fwd_sparge_sage.py`. You can look into these scripts to learn how to instantiate a sparse FMHA kernel step by step.

## executable
`tile_example_sparse_attn_fwd` is the example executable, implemented in `example_sparse_attn_fwd.cpp`. You can type `./bin/tile_example_sparse_attn_fwd -?` to list all the arguments. Below is an example of the output (may subject to change):
```
        -api    sparse attention API: jenga (default) / vsa / sparge / sparge_sage
     -qscale    sparge_sage quant scale mode: perwarp|perblock|perthread|pertensor (default:perwarp)
    -qkdtype    sparge_sage Q/K quant dtype: int8 (i8fp8bf16) | fp8 (fp8bf16); V always fp8 (default:int8)
          -v    0:no validation, 1:validation (default:1)
       -mode    kernel mode. 0:batch, 1:group (jenga + vsa + sparge + sparge_sage) (default:0)
          -b    batch size (default:1)
          -h    num of head, for q (default:4)
        -h_k    num of head, for k/v, -1 means equal to h (default:-1)
                if not equal to h, then this is GQA/MQA case
          -s    seqlen_q (default:4096)
        -s_k    seqlen_k, -1 means equal to s (default:-1)
          -d    head dim for q, k (default:128)
        -d_v    head dim for v, -1 means equal to d (default:-1)
   -sparsity    target sparsity ratio [0,1). 0=dense (default:0.02 -> cdf 0.98)
                jenga / vsa: random-mask activation probability (skip ratio)
                sparge: passed to the algorithm chosen by -sparge_mode
-sparge_mode    sparge: block-selection algorithm (default:cdf)
                  cdf:  CDF threshold; greedy until cumulative softmax prob
                        reaches 1-sparsity (1-0.02 = 0.98, official default)
                  topk: deterministic 1-sparsity ratio of K-blocks per Q-block
-simthreshold  sparge & sparge_sage: cosine similarity threshold for fix blocks (default:0.6)
       -mask    0/1/2 or t/b[/:l,r]. Same syntax as 01_fmha.
                supported by all: jenga / vsa / sparge / sparge_sage.
       -sink    sparge & sparge_sage: 1 = always include first K block (default:0)
   -smooth_k    K smoothing: center K by per-channel mean; gates sparge pool/sim
                + sparge_sage K-quant. 0 disables (default:1)
-print_sparsity sparge: 1 = read back actual sparsity; needed for accurate TFlops/GB/s
  -pvthreshd    sparge & sparge_sage: P*V runtime block-skip threshold (log2 units);
                >0 enables Stage 2 (default:50)
-perhead_test   sparge & sparge_sage (batch & group): synthesize a per-head hyperparam pattern (default:0)
-{sparsity,sim,pvthreshd}_per_head
                sparge & sparge_sage (batch & group): comma-separated per-Q-head value, length=nhead_q
                (-sparsity_per_head is routed to topk or cdf field by -sparge_mode)
 -block_size    block size BLKQ=BLKK (default:128)
       -prec    fp16/bf16 (default:fp16)
    -scale_s    softmax scale factor; 0 ⇒ default 1/sqrt(d) (default:0)
                Override for fixed-scale eval / RoPE-aware models.
-logits_soft_cap Gemma-style logits soft cap; 0 ⇒ disabled (default:0)
                Pre-softmax: s = cap * tanh(s * scale / cap).
       -bias    n or 0, no bias (default:n)
                e(lementwise) or 1, elementwise bias 1*1*sq*sk. e:1, 1*h*sq*sk. e:2, b*h*sq*sk
                a(libi) or 2, alibi 1*h slope (needs causal mask). a:1, b*h
                (jenga / vsa / sparge / sparge_sage; batch + group)
      -iperm    permute input. 1: b*h*s*d, 0: b*s*h*d (default:1)
      -operm    permute output (default:1)
       -seed    random seed (default:42)
     -warmup    warmup iterations (default:5)
     -repeat    benchmark iterations (default:20)
      -kname    1: print kernel name (default:0)
    -vlayout    r: row-major (seqlen*hdim), c: col-major (hdim*seqlen) (default:r)
      -timer    timer type: gpu or cpu (default:gpu)
       -json    1: append per-run metrics as JSON (default:0)
   -jsonfile    path to write JSON metrics (default: empty)
```
Example 1: `./bin/tile_example_sparse_attn_fwd -api=jenga -b=1 -h=16 -s=4096 -d=128 -sparsity=0.5` runs jenga with batch=1, nhead=16, sequence length=4096, hdim=128, fp16, 50% sparsity.

Example 2: `./bin/tile_example_sparse_attn_fwd -api=vsa -b=1 -h=8 -h_k=2 -s=8192 -d=128 -sparsity=0.7` runs vsa with GQA (Q heads=8, KV heads=2), sequence length=8192, 70% sparsity.

Example 3: `./bin/tile_example_sparse_attn_fwd -api=sparge -b=1 -h=4 -s=4096 -d=128 -sparsity=0.6 -sparge_mode=topk -sink=1` runs sparge with deterministic 60% K-block skipping per Q-block (`-sparge_mode=topk`; the default `-sparge_mode` is `cdf`) plus an attention sink (always include the first K block).

Example 4: `./bin/tile_example_sparse_attn_fwd -api=sparge -b=1 -h=16 -s=16384 -d=128 -sparsity=0.6 -sparge_mode=cdf -print_sparsity=1` runs sparge under CDF block selection (`cdfthreshd = 1 - sparsity`) and reads back the realised sparsity for accurate TFlops / GB/s reporting (CDF mode's realised sparsity floats around the target).

## support features
Currently we are still in active development, so more features and optimizations will be coming soon.

### hdim
Currently only `hdim=128` is supported, with block size fixed at `128x128`. Wider hdim and other block sizes will be added later.

### group/batch mode
We support both `batch mode` and `group mode` (or `varlen` in FA's term) by setting `-mode=0` or `-mode=1`. Group mode packs per-sequence Q/K (lengths sampled in `[seqlen/2, seqlen]`, seed-stable) into a single buffer; mask and workspace layouts are fully packed with no K-padding waste. All four APIs support both modes, including per-head hyperparameters (`-perhead_test` / per-head CSV) in both batch and group.

### MQA/GQA
By setting `-h` (nhead for q) and `-h_k` (nhead for k/v) with different numbers you can achieve MQA/GQA. Please pay attention that `h % h_k == 0` when you set different numbers.

### input/output permute
All four APIs (`jenga` / `vsa` / `sparge` / `sparge_sage`) support both `b*h*s*d` (`-iperm=1`, BHSD) and `b*s*h*d` (`-iperm=0`, BSHD), in batch and group mode. Output permutation via `-operm=0/1` is also supported. Internally the kernels accept arbitrary `stride_q/k/v/o`, so other layouts can be exercised by editing the host args directly. (`sparge_sage` host-prequantizes V into the `-iperm` layout to match the kernel's strides.)

### attention bias
Attention bias is supported by all four APIs (`jenga` / `vsa` / `sparge` / `sparge_sage`) in batch + group mode, both elementwise and ALIBI.

`-bias=e[:rank]` adds an elementwise bias to the score matrix `S = Q·K^T·scale + bias` before softmax. Rank controls broadcast: `e:0` is `[1,1,sq,sk]` shared across batch and head, `e:1` is `[1,h,sq,sk]` shared across batch only, `e:2` is `[b,h,sq,sk]` per-batch with no broadcast. For sparge, bias is added at the **attention pass only** — sparge's mask-prediction step picks K-blocks based on `dot(q_means, k_means)` alone, independent of bias.

### alibi
ALiBi (Attention Linear Bias) is supported via `-bias=a[:rank]` for all four APIs (`jenga` / `vsa` / `sparge` / `sparge_sage`). Slopes are computed via the standard `get_alibi_slopes(nhead)` formula and require a causal mask (`-mask=t/b`). `a:0` uses `[1,h]` slopes shared across batch; `a:1` uses `[b,h]` per-batch slopes.

### scale_s and logits soft cap
`-scale_s=X` overrides the default `1/sqrt(d)` softmax scale (RoPE-aware models, fixed-scale eval, ablations). `-logits_soft_cap=C` enables Gemma-2/3 style soft cap `s = C * tanh(s * scale / C)` pre-softmax, required by the Gemma family. Supported by all four APIs (in `sparge_sage` the cap is applied on the descaled QK with the same tanh, before masking), and may be combined with bias (order `scale -> soft-cap -> +bias`, same as `01_fmha`).

### attention mask
We support `causal mask` and `sliding window attention (swa)` mask in both batch and group modes, either from top-left or bottom-right. Syntax mirrors `01_fmha`:

| mask case | cmdline | FA style |
|---|---|---|
| no mask | `-mask=0` (default) | |
| causal mask from top-left | `-mask=1` or `-mask=t` | `-mask=t:-1,0` |
| causal mask from bottom-right | `-mask=2` or `-mask=b` | `-mask=b:-1,0` |
| swa from top-left | | `-mask=t:3,5` |
| swa from bottom-right | | `-mask=b:10,11` |

VSA applies the mask at element level. Sparge additionally excludes blocks entirely outside the mask boundary at the prediction stage. **Jenga** applies the causal/SWA mask per element on edge tiles and via the causal tile range, on top of its one-hot block selection — supported in both batch and group mode (it reads the one-hot selection from global to stay race-free with the causal tile range).

### sparge — block selection mode
Sparge has one CLI input — `-sparsity=X` (target skip ratio in [0,1)) — paired with `-sparge_mode={topk|cdf}` to pick the algorithm:

* `-sparge_mode=topk` (default) → fixed-count: pick `max(1, round((1-X) * num_k_blocks))` top blocks per Q-block. Realised sparsity matches `-sparsity` exactly.
* `-sparge_mode=cdf` → CDF threshold: greedily add blocks until cumulative softmax probability ≥ `1 - X`. Realised sparsity floats around the target depending on the score distribution; pair with `-print_sparsity=1` for accurate TFlops / GB/s.

For `jenga` and `vsa`, `-sparsity` instead controls random-mask activation probability; diagonal blocks are always active. `-sparge_mode` is ignored for these variants.

At the C++ API level, `sparge_hyperparam_args` still exposes both `cdfthreshd` and `topk` fields independently (set the active one and leave the other at 0); the CLI's `-sparge_mode` only affects how the example wires the unified `-sparsity` value into those fields.

### sparge & sparge_sage — similarity threshold
`-simthreshold` enables Q/K-block self-similarity fix-up. Q blocks below the threshold attend to all causally-valid K blocks; K blocks below the threshold are force-included. The scalar doubles as a global switch — when `<= 0` the sim arrays are not allocated and any per-head pointer is silently ignored (with stderr warning).

### sparge — K smoothing
The `-smooth_k` flag (default `1`, matches upstream SpargeAttn `smooth_k=True`; set `-smooth_k=0` to disable) gates K-mean centering in two places:

* **non-quant sparge selection** — a no-op for block selection: subtracting the per-`(batch, kv_head)` K mean shifts every q@k_mean score within a row by a constant that softmax max-subtract removes, so `km_ptr` stays null in the attention.
* **sparge_sage quantization** — the per-`(batch, kv_head, channel)` global K mean (`km = k.mean` over seqlen) is subtracted from K **before** INT8/FP8 quantization (`round((k - km)/scale)`), reducing K's quantization error. Q is never centered. The attention pipeline is unchanged: the implied `-q@km^T` is a per-row constant absorbed by softmax. `km` is host-computed once and fed to both the device quant kernels and the reference, keeping faithful-dequant validation tight. With `-smooth_k=0`, `km_ptr` is null and both device and reference quantize raw K.

### sparge & sparge_sage — P·V skip threshold (Stage 2)
`-pvthreshd` matches upstream SpargeAttn `pv_threshold` (positive log2-units, default `50`). Per-iteration P·V is dropped when `(block_local_max - block_global_max) < -pvthreshd` in log2 space. Paper-typical range `[1, 5]`; `0` disables Stage 2. Supported by both `sparge` and `sparge_sage`, with or without bias (the block-peak is taken after bias is added, so the predicate is valid for NO_BIAS and ALIBI/ELEMENTWISE alike; soft-cap disables pv-skip). In `sparge_sage` the skip keeps all V/descale-follow-LUT bookkeeping unconditional, so the per-block k/v descale stays aligned with the non-skip path. Note: on uniform-random inputs the block-wide skip rarely fires (every K-block has at least one peaking Q-row).

### sparge & sparge_sage — per-head hyperparameters
At the C++ API level, `sparge_hyperparam_args` carries one device pointer per Q-head (length `nhead_q`, float) for `cdfthreshd` / `topk` / `simthreshold` / `pvthreshd`. `nullptr` falls back to the scalar. Both `sparge` and `sparge_sage` support this in **batch and group** mode (per-head is head-indexed, length `nhead_q`, independent of the group varlen packing). From the CLI, `-perhead_test=1` synthesizes a deterministic per-head pattern (requires `-h >= 2`), and `-{sparsity,sim,pvthreshd}_per_head=v0,v1,...` supplies per-head values explicitly (length must equal `nhead_q`). `-sparsity_per_head` is converted to `1 - sparsity[h]` and routed to either the `topk` or the `cdfthreshd` field according to `-sparge_mode`; the other field stays at 0. Per-field precedence: explicit CSV > `-perhead_test` > scalar.

### sparge — actual sparsity readout
`-print_sparsity=1` appends `sparsity=X` to the report and rescales TFlops / GB/s by the actual ratio. Without it the metrics use the input `-sparsity` threshold, which may diverge from the realised ratio under topk / sim / sink / CDF modes.

### SpargeAttention-Sage (quantized)
`-api=sparge_sage` fuses sparge's block selection with **SageAttention** low-bit quantization: Q/K are quantized to **INT8 or FP8** and V to **per-channel FP8**, so the masked-out blocks are skipped *and* the surviving blocks compute on quantized MFMA. Q/K quantization runs on-device; V's per-channel FP8 quantization runs on the host.

> [!IMPORTANT]
> Requires **gfx950 / MI350**. The pipeline uses transpose-load (`ds_read_tr`) plus FP8/INT8 MFMA; other architectures are not supported. `-prec=bf16` is mandatory (the wrapper rejects `fp16`).

Flags:
* `-prec=bf16` — required; the I/O dtype is bf16 (quant dtypes are `i8fp8bf16` / `fp8bf16`).
* `-qscale=perwarp|perblock|perthread|pertensor` — quantization scale granularity.
* `-qkdtype=int8|fp8` — Q/K quant dtype (default `int8`; pass `fp8` for FP8 Q/K).
* Inherits sparge's selection and feature flags: `-sparsity` / `-sparge_mode`, `-mode=1` (group), `-mask=t/b[:l,r]` (causal / SWA), `-bias=e:0/1/2` (elementwise), `-bias=a[:1]` (ALIBI, needs `-mask`), GQA (`-h_k`), `-sink=1`, `-simthreshold`, `-pvthreshd`, `-logits_soft_cap`, `-perhead_test` / per-head CSV, `-print_sparsity=1`.

Examples:
```
# INT8 Q/K, per-channel FP8 V, per-warp scale, top-left causal
./bin/tile_example_sparse_attn_fwd -api=sparge_sage -prec=bf16 -qkdtype=int8 -qscale=perwarp \
    -b=1 -h=16 -d=128 -s=16384 -sparsity=0.5 -mask=t -print_sparsity=1
# FP8 Q/K, group mode, GQA
./bin/tile_example_sparse_attn_fwd -api=sparge_sage -prec=bf16 -qkdtype=fp8 -qscale=pertensor \
    -mode=1 -b=2 -h=8 -h_k=2 -d=128 -s=1024 -sparsity=0.5
```

Performance (b=2 h=8 d=128 sparsity=0.5, s=16384): sparge_sage **2.34 ms** vs sparge 3.46 ms vs dense 6.2 ms — about **2.65x over dense**, and quantization nets roughly **1.5x over plain sparge** at long sequence length.

## C++ API integration
The high-level `sparge_sparse_attention` allocates and frees a workspace each call. For repeated calls, use the lower-level `fmha_sparge_fwd` directly: pre-allocate the workspace via `compute_sparge_workspace_layout(args).total_bytes` and assign it to `args.workspace_ptr` once.

cu_seqlens fields on all four args structs (`seqstart_*_ptr / seqlen_*_ptr / cu_seqlen_*_ptr`) mirror `01_fmha/fmha_fwd.hpp:fmha_fwd_args` in type and naming. Per-field semantics live in the in-struct docblock above `seqstart_q_ptr`. Sparse-specific block offset tables (`seqstart_q_block_ptr` etc.) have no fmha analogue and must be computed by the caller.

## benchmarking notes
Reported TFlops is `attention FLOP / total kernel time`. For sparge, total time includes K-mean / preprocess / mask-prediction passes that contribute no attention FLOP, so a direct TFlops comparison against a dense baseline understates sparge's attention pipeline rate. Prefer **latency** (or speedup vs dense at the same shape) as the primary metric. Pair with `-print_sparsity=1` for accurate ratio-based TFlops / GB/s.

## scripts
* `script/benchmark_sparse_attn.sh` — benchmark all variants (incl. quantized `sparge_sage`, bf16-only) across precisions, layouts, sparsity levels.
* `script/smoke_test_sparse_attn.sh` — correctness tests for all variants (incl. `sparge_sage` across `qkdtype`×`qscale`) under various masks.
