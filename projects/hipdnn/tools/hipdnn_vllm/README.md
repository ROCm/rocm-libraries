# hipdnn-vllm

A vLLM plugin that routes vLLM ops through hipDNN. It is installed into a vLLM
environment and needs no vLLM source or model changes. When hipDNN cannot serve
a call, the call runs vLLM's normal kernel and the plugin records the reason.

What this release provides:

- **Attention backend `HIPDNN`**, selected with `--attention-backend HIPDNN`.
  - It subclasses TRITON_ATTN and inherits its KV-cache layout, feature set and KV write.
  - **Encoder-only attention (P1e)**, as in BERT-style embedding models, runs on hipDNN SDPA in eager calls. Each run of equal-length requests is one dense, non-causal graph over zero-copy strided views of vLLM's packed query/key/value, and the result is written straight into vLLM's output buffer. Under HIP-graph capture these calls stay on TRITON_ATTN (`CAPTURE_INVARIANT:p1e_eager_only`). When no engine accepts a signature, the call falls back with the build-stage reason.
  - **Decoder attention (P2a/P2b)** runs as one hipDNN SDPA launch per layer call over the paged TRITON_ATTN cache, with no data movement. Q and O are vLLM's packed tensors, ragged through `query_start_loc` (one offsets tensor shared by Q and O). K and V are strided views of the interleaved cache with the same strides. The block table is the page table for both K and V (one tensor), and `seq_lens` is `seq_len_kv`. The causal mask is bottom-right, and a sliding window `W` is `left_bound = W-1`.
    - **P2a** serves uniform-decode batches (q_len 1, or k+1 with speculative decoding). **P2b** serves mixed, chunked-prefill and prefill batches, and also uniform decode when only its plan is accepted.
    - Plans stay within the process's scheduler bounds: batch `max_num_seqs`; query length (the decode length for P2a, `min(max_num_batched_tokens, max_model_len)` for P2b); and `max_model_len` rounded up to the block size. The metadata builder binds each step from device buffers with no host sync, so a plan serves every batch that fits its size, eager or replayed from a CUDA graph. The builder reports `AttentionCGSupport.ALWAYS`. Under capture, uniform-decode graphs hold P2a and every other graph holds P2b.
    - The engines' launch grids grow with the plan's batch and query-length bounds, so each call runs a plan sized for its batch rather than the upper bounds. A captured graph holds the plan of the batches it is replayed for. Under vLLM's V1 model runner that is its batch descriptor: the padded request count, and the decode length (uniform decode) or the padded token count (mixed). The V2 model runner passes no descriptor for FULL graphs (the forward context then holds one with no request count) and builds the capture batch from the graph's bounds, so the plan is the capture batch's request count and promised `max_query_len`. An eager call runs the power-of-two bucket of its request count and, for P2b, of its longest query. A smaller plan binds a prefix of the same persistent buffers. `max_seq_len_kv` stays at the bound, because the engines read each sequence's key count from `seq_len_kv`. The report's `attention_plans` lists the sizes built: one upper-bound plan per path, one per FULL capture size, and at most log2-many eager buckets.
    - Plans are probed at the upper bounds, the plans of every FULL capture size are built, and each new plan runs once on an empty batch, at a layer's first eager call for each KV cache (vLLM profiles with a small cache before allocating the real one). Nothing is built under capture. A layer whose first call on a cache is under capture keeps TRITON_ATTN in that graph (`CAPTURE_INVARIANT:not_prepared`). A signature no engine accepts (for example an unsupported head size or GQA ratio) falls back with the build-stage reason.
    - Engines today: `hipkernel:AttentionPagedUnified` on gfx942/gfx950, and the paged layouts of `hipkernel:Gfx1151WmmaAttention` on RDNA. Both come from the bundled providers.
    - Attention sinks (gpt-oss) are part of the graph: the layer's `sinks` (`[H_q]`, model dtype) are bound zero-copy as hipDNN's `sink_token` `[1, H_q, 1, 1]` in the Q dtype (a sink of another dtype is converted once and kept on the layer). The engine decides whether it takes them; a decline falls back with the build-stage reason (e.g. `NO_ENGINE`).
    - Some layer and batch features fall back before any graph is built, with a more specific reason: `softcap`, `alibi`, `fp8_kv`, `encoder_decoder`, `mm_prefix` or `rswa`.
  - Encoder-only layers stay on P1e. Neither paged engine accepts ragged Q over non-paged K/V, and the test suite records each arch's answer.
  - Layers that use `chunk_lookback` (today only Rnj1, whose `layer_types` contain `chunked_attention`) run on the inherited TRITON_ATTN kernel, with the reason `UNSUPPORTED_FEATURE:chunk_lookback`, because hipDNN has no chunk-lookback mask. Unpatched vLLM rejects such models under any backend other than TRITON_ATTN, because `Attention.__init__` checks the backend name. They need the local vLLM patch described under "Local vLLM patches". `probe` prints this requirement.
- **`torch.ops.hipdnn_vllm.rms_norm(x, weight, eps)`**: hipDNN `rmsnorm` (INFERENCE). It is not wired into vLLM yet.
- **`torch.ops.hipdnn_vllm.sdpa_fwd(q, k, v, out, scale, ...)`**: one parameterized hipDNN SDPA forward graph, written into `out`. It covers:
  - dense Q/K/V of any strides;
  - ragged Q/O: packed `[T, H, D]` with a token-unit `ragged_offset_q`, `seq_len_q` and `max_seq_len_q`. Q and O share one offset tensor, and each tensor's multiplier is its token stride;
  - ragged K/V;
  - paged K/V: `[num_blocks, H_kv, block_size, D]` containers, one `page_table` `[B, M]` bound as both the K and the V table, `seq_len_kv` and `max_seq_len_kv`;
  - the diagonal-band masks (`causal`, `bottom_right`, `left_bound`/`right_bound` in vLLM's `sliding_window` convention);
  - GQA;
  - an optional attention `sink` `[1, H_q, 1, 1]` (FLOAT or the Q dtype), cuDNN's `sink_token`: one logit per query head that joins the softmax max and denominator, not multiplied by `scale`. The plan key includes it, so sink and sink-free graphs never share a plan.

  The paged and ragged graphs follow cuDNN semantics. The ragged ones also need hipDNN API 1.4.0 for the ragged-offset multiplier, which the bundled backend provides.
- **`hipdnn-vllm probe --model <hf-id>`** builds every graph signature a model needs, at a few representative shapes, on this machine. For each signature it prints the backend's route and whether hipDNN accepts the graph, with the engine name or the decline reason.
- **ViT attention (`MMEncoderAttention`)**, on by default (`encoder_attention` in `HIPDNN_VLLM_OPS`).
  - It replaces vLLM's layer through `CustomOp.register_oot`. On ROCm each call runs `torch.ops.hipdnn_vllm.encoder_attention`, a dense non-causal hipDNN `sdpa` graph.
  - When every image in `cu_seqlens` has the same length, one batched graph covers all images. Otherwise there is one launch per image, and images of the same length share a plan.
  - Any decline runs vLLM's own layer for that call. The image lengths are read on the host, so a call made under HIP-graph capture (`cudagraph_mm_encoder`) declines with `CAPTURE_INVARIANT:cu_seqlens`. FP8 ViT attention declines with `UNSUPPORTED_FEATURE:fp8`.
  - Which engines serve which head sizes depends on the installed providers. The plugin probes each signature, so there is no engine table.
- **`LayerNorm` OOT override** (`layer_norm` in `HIPDNN_VLLM_OPS`): vLLM's `LayerNorm` runs `torch.ops.hipdnn_vllm.layer_norm`, hipDNN `layernorm` (INFERENCE) with optional weight and bias. Rows run as one exact plan when it was built (the `layer_norm_probe` call builds it), otherwise as power-of-two row blocks. An input whose hidden dim is not packed (e.g. a transposed patch-embedding output) is packed first, as ATen's `layer_norm` also does.
- **`Conv2dLayer` / `Conv3dLayer` OOT overrides** (`conv`): the parent's true convolution path runs `torch.ops.hipdnn_vllm.conv_fwd`, hipDNN `conv_fprop`. `Conv3dLayer`'s unfold + linear shortcut (kernel == stride, no padding, groups 1) stays with vLLM. With a bias, the op first tries MIOpen's conv-bias-activation fusion (with an identity clamp as the activation). If no engine takes it, the op runs the hipDNN conv, adds the bias with ATen, and counts the call as `conv_fwd` `hipdnn` `PARTIAL`; the inner conv is counted as `conv_fprop`.
- With these ops listed, the overrides run even under vLLM's Inductor default `custom_ops=["none"]`. `-layer_norm`/`-LayerNorm` or `-conv2d`/`-Conv2dLayer` (and the `conv3d` spellings) in `custom_ops` turn them off. On a decline, the ops count `fallback` and compute vLLM's native result, or raise under `HIPDNN_VLLM_STRICT=1`.
- **Opaque custom ops** (`custom_op`, opt-in): `torch.ops.hipdnn_vllm.custom_op_prepare(op_id, payload, inputs) -> str` and `custom_op_exec(op_id, payload, inputs, strict)` run one hipDNN `CUSTOM_OP` node (inputs bound in order, no outputs, the payload string as the node's data), for engines that claim an op id such as `megakernel.<model>` (the gfx94X wheel builds the megakernel engine, `ENABLE_MEGAKERNEL_ENGINE`). The plan is keyed by op id, payload and input shapes, not addresses. `prepare` builds the plan and returns `""` or the decline reason, never raising; call it before graph capture. `exec` counts a decline as `fallback` and returns (the caller runs its own path), or raises when `strict`, including a plan miss under capture. Without `custom_op` in `HIPDNN_VLLM_OPS` both decline `DISABLED` and build nothing.
- **Coverage report and `doctor` self-check.**

## Install

The wheel is built for one GPU family and must match the pinned vLLM, torch and
ROCm versions in `BUILD_PINS.toml`. `_C` is ABI-coupled to all three.

```
python3.12 -m venv venv && . venv/bin/activate
pip install "vllm==0.32.1rc1.dev117+gdf3bd094a.rocm100" "rocm[devel]==10.0.0" \
  --extra-index-url https://wheels.vllm.ai/rocm/df3bd094aeea829f27baa0a5282830464f601d9f/rocm100
pip install hipdnn_vllm-0.1.0+rocm10.0.0.<family>-cp312-cp312-linux_x86_64.whl
python -m hipdnn_vllm doctor
```

`doctor` prints the following, and exits non-zero if any check fails:

- the hipDNN backend path and version;
- the backend and frontend versions `_C` was built against;
- the device arch;
- the loaded providers and engines;
- the result of each check.

## Usage

```
vllm serve <model> --attention-backend HIPDNN
# offline
LLM(model=..., attention_backend="HIPDNN")
```

vLLM loads the plugin through the `vllm.general_plugins` entry point in every
process it starts. Registration does no GPU work. hipDNN loads lazily on the
first hipDNN op call in a worker.

Registration also adds `HIPDNN_VLLM_COMPILE_FACTOR` to
`vllm.envs.environment_variables`, which vLLM hashes into its torch.compile and
AOT-compile cache keys. Its value identifies the installed plugin files and
the dispatch settings (`HIPDNN_VLLM_OPS`, `_STRICT`, `_ATTN_DEBUG_ROUTE`,
`_PLUGIN_MODE`, `_PROVIDER_DIRS`). Without it, a run with the plugin could load
graphs that were compiled without it.

### RMSNorm (vLLM IR providers)

`rms_norm` and `fused_add_rms_norm` are vLLM IR ops. The plugin registers a
provider named `hipdnn` for each, and vLLM uses it only when the IR op priority
names it. Any of these forms selects it; they parse to the same config:

```
vllm serve <model> --ir-op-priority '{"rms_norm":["hipdnn","vllm_c","native"],"fused_add_rms_norm":["hipdnn","vllm_c","native"]}'
vllm serve <model> --ir-op-priority '{"rms_norm":"hipdnn,vllm_c,native","fused_add_rms_norm":"hipdnn,vllm_c,native"}'
vllm serve <model> --ir-op-priority.rms_norm hipdnn,vllm_c,native --ir-op-priority.fused_add_rms_norm hipdnn,vllm_c,native
# offline
LLM(model=..., ir_op_priority={"rms_norm": ["hipdnn", "vllm_c", "native"],
                               "fused_add_rms_norm": ["hipdnn", "vllm_c", "native"]})
```

vLLM appends its platform defaults to a user list, so `["hipdnn"]` alone also
works. vLLM falls back to the next provider for arguments the `hipdnn`
provider does not take. Those are: no weight, `variance_size`, dtypes other
than fp16/bf16/fp32, rows strided within the hidden dim, leading dims that do
not collapse into one row stride, and shapes no engine builds a plan for. Each
such fallback is counted.

No norm engine takes a runtime row count yet, so each plan has a fixed row
count. On the first eager call of a signature, the plugin pre-builds exact
plans for vLLM's cudagraph capture sizes. Those row counts run as one launch
of their exact plan; any other row count runs as power-of-two row chunks, one
launch per set bit, eager and captured alike, so a capture replays the plans
its eager warm-up built. `HIPDNN_VLLM_NORM_EXACT_ROWS` changes the pre-built set.
`fused_add_rms_norm` runs as an add plus the hipDNN rmsnorm, counted as
`PARTIAL`, unless an engine accepts the fused add→rmsnorm graph. A decline
inside a compiled graph runs vLLM's native formula, or raises under
`HIPDNN_VLLM_STRICT`.

To see what will run on hipDNN for a model on this machine:

```
hipdnn-vllm probe --model BAAI/bge-base-en-v1.5   # or python -m hipdnn_vllm probe ...
```

The `route` column shows what the backend does with the call today. The
`graph` column shows whether hipDNN accepts the graph for that path (`hipdnn`
or the decline reason). For decoder models it builds the backend's own
paged-decode (`p2a`) and ragged+paged (`p2b`) graphs for the bounds given with
`--max-num-seqs`, `--max-num-batched-tokens` and `--max-kv`.

## Local vLLM patches

`patches/vllm/` holds vLLM changes that are local to this plugin until they are
upstreamed. Apply them to the pinned vLLM install with:

```
scripts/apply_vllm_patches.sh /path/to/venv/bin/python
```

The script refuses any vLLM other than the pinned version, and skips patches
that are already applied. `docker/Dockerfile` applies the patches too.

- `0001-attention-capability-check-for-chunk-lookback.patch` replaces the
  `get_name() == "TRITON_ATTN"` assert for `chunk_lookback` layers with an
  `AttentionBackend.supports_chunk_lookback()` capability.
  - TRITON_ATTN returns True, and HIPDNN inherits that.
  - TRITON_ATTN_DIFFKV returns False.
  - Every in-tree backend behaves as before.
  - Without this patch, Rnj1-style models cannot select HIPDNN.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `HIPDNN_VLLM_DISABLE` | `0` | Register nothing. |
| `HIPDNN_VLLM_OPS` | `attention,encoder_attention,rms_norm,fused_add_rms_norm,layer_norm` | Ops routed to hipDNN. `conv`, `linear`, `linear_all` and `custom_op` are opt-in. `conv` is the only op that needs the MIOpen provider, so MIOpen is loaded only when `conv` is enabled: its handle initialization is costly and paid once per process, at the first hipDNN call (the time is logged at INFO). If `attention` is excluded, the backend still registers, and every call falls back with `DISABLED`. |
| `HIPDNN_VLLM_ATTN_DEBUG_ROUTE` | `0` | Log the attention route once per layer and batch class. |
| `HIPDNN_VLLM_STRICT` | `0` | Raise instead of falling back. For validation runs. |
| `HIPDNN_VLLM_REPORT_DIR` | unset | Write `report-<pid>.json` here (the JSON records the rank), at exit and on every summary interval. |
| `HIPDNN_VLLM_REPORT_INTERVAL` | `60` | Seconds between summary log lines. `0` disables them. |
| `HIPDNN_VLLM_PLAN_CACHE_SIZE` | `4096` | LRU bound on cached plans and declines per process. |
| `HIPDNN_VLLM_PLUGIN_MODE` | `auto` | `auto`, `additive` or `absolute` provider loading. See below. |
| `HIPDNN_VLLM_PROVIDER_DIRS` | unset | Extra provider paths. In `absolute` mode, these replace all providers. |
| `HIPDNN_VLLM_NORM_EXACT_ROWS` | `capture` | Row counts that get exact rms_norm plans: `capture` (vLLM's cudagraph capture sizes), `none`, or comma-separated counts. |

hipDNN's own variables are passed through unchanged. Examples are `HIPDNN_LOG_LEVEL`,
`HIPDNN_HEUR_FALLBACK_ENGINE_ORDER` and `HIPDNN_HEUR_CONFIG_PATH`. The engine order is
part of every plan-cache key.

`HIPDNN_VLLM_COMGR_CACHE` (`auto` or `keep`, default `auto`) controls where
comgr caches code objects. HIP unbundles compressed code objects, and hiprtc
compiles, through comgr. comgr's cache prunes with a timestamp file that NFS
never refreshes, so with the default directory (`~/.cache/comgr`, or
`$XDG_CACHE_HOME/comgr`) on an NFS home, every comgr action rescans the whole
directory and every hit is read back from NFS. Loading the MIOpen provider runs
hundreds of such actions. In `auto`, `register()` checks the default
directory's filesystem (statfs: NFS, SMB/CIFS, Lustre, GPFS, Ceph, BeeGFS, AFS,
Coda, 9P). If it is a network filesystem, it sets `AMD_COMGR_CACHE_DIR` to a
private node-local `<tmp>/hipdnn_vllm_comgr_<uid>` (mode 0700, owner checked)
and logs one INFO line. This mirrors upstream comgr
([ROCm/llvm-project#4247](https://github.com/ROCm/llvm-project/pull/4247)),
which disables the default directory on NFS, but keeps the cache. It runs in
`register()` because comgr reads its settings at HIP's first kernel launch.
Nothing changes if `AMD_COMGR_CACHE` or `AMD_COMGR_CACHE_DIR` is set, or with
`keep`.

### Backend and provider loading

Each wheel bundles its own hipDNN backend under `hipdnn_vllm/backend/<family>/`
and every engine provider (rocKE-enabled `hip_kernel_provider`, `miopen_plugin`,
`hipblaslt_plugin`, with their arch content) under
`hipdnn_vllm/providers/<family>/`, all built from one rocm-libraries commit.
The backend torch ships (`rocm-sdk-libraries`) has an older API and rejects
paged SDPA graphs, so the plugin does not use it.

`_C` is built in the hipDNN frontend's runtime-load mode and links no backend.
The bootstrap points the frontend at the bundled backend's directory before the
first hipDNN call, and the frontend loads that file by explicit path,
`RTLD_LOCAL`. torch may map its own copy too; `_C` never resolves a symbol from
it, and the bootstrap checks this with `dladdr`.

The bundled backend loads only the bundled providers. `auto` (default) and
`absolute` load them absolutely, leaving MIOpen out unless `conv` is enabled.
`HIPDNN_VLLM_PROVIDER_DIRS` adds paths; in `absolute` mode it replaces the
list. `additive` passes the same list on top of the backend's default plugin
directory. `doctor` reports both backends (the bundled one `_C` uses and
torch's), the mapped copies, the loaded plugin files and each engine's owning
plugin version.

### Unquantized linear (`linear`, `linear_all`)

Off by default. With `linear` in `HIPDNN_VLLM_OPS`, `register()` replaces
`dispatch_unquantized_gemm` in `vllm.model_executor.layers.linear`, so every
`UnquantizedLinearMethod` built afterwards calls `hipdnn_unquantized_gemm`.
`linear_all` also patches the LM head (`vocab_parallel_embedding`) and the fused
MoE router gate (`fused_moe/runner/moe_runner.py`). Before changing anything,
the patch checks every vLLM symbol and signature it relies on. A mismatch raises
`LinearPatchError` from `register()`, so a vLLM upgrade fails loudly. Without
`linear`, nothing is imported or patched.

Each call goes through `torch.ops.hipdnn_vllm.unquantized_gemm`. This op is opaque
to torch.compile and runs one of two paths:

- the hipDNN graph `matmul(x, weightᵀ)` with an optional pointwise-ADD bias epilogue, served by the hipBLASLt engine;
- vLLM's `rocm_unquantized_gemm`, when hipDNN declines. Declines are counted, and strict mode raises.

The token count M is dynamic, so each M gets its own plan in the LRU plan
cache. During HIP-graph capture, only plans that already exist are used;
vLLM's warm-up run of each capture size builds them. If capture meets an M with
no plan, the call falls back with `CAPTURE_INVARIANT` instead of building inside
the capture.

## Coverage report

Each counter row is keyed by `(op, route, reason, phase)`:

- `route` is `hipdnn` or `fallback`.
- `phase` is `capture` when the call was recorded during HIP-graph capture.
  Replays run no host code and are not counted.

`fallback_reasons` sums the rows by reason. Reasons follow this taxonomy:

| Reason | Meaning |
|---|---|
| `NO_BUILD` | hipDNN rejected the graph (validation or `build_operation_graph`). |
| `NO_ENGINE` | No engine is applicable. |
| `CHECK_SUPPORT` | The selected plan failed the support check. |
| `BUILD_PLANS` | Plan compilation failed. |
| `UNSUPPORTED_FEATURE:<name>` | Plugin-side pre-filter: no hipDNN mapping for the feature. |
| `DISABLED` | Excluded by `HIPDNN_VLLM_OPS`. |
| `CAPTURE_INVARIANT` | The route was forced to the capture-time variant, or the call met a signature with no plan under HIP-graph capture. |
| `EXECUTE` | A built plan failed to launch, or a HIP call inside the launch failed. |

Declines are cached with the plans, so a declined signature costs one build attempt per process.

No plan is ever built while the stream is capturing a HIP graph: graph build,
module load and allocation would run inside the capture. A call that needs a
plan not built before capture falls back with `CAPTURE_INVARIANT`, and that
decline is not cached, so the next eager call builds the plan. Plans a capture
uses are kept for the life of the process (`plan_cache_pinned` in the runtime
stats), because the captured graph replays their launches whatever the LRU
evicts.

## Building

`_C` is a torch C++ extension built with setuptools (`setup.py`). It is compiled
against the venv's torch and the frontend headers of a staged hipDNN install
(`HIPDNN_VLLM_HIPDNN_PREFIX`), with `HIPDNN_FRONTEND_RUNTIME_LOAD_BACKEND` and
`HIPDNN_ENABLE_SDPA`. `_C` refuses to run if the backend it resolved reports a
different version from the one it was built against.

`scripts/build_wheel.sh <family>` builds hipDNN (backend, frontend, and the
hip-kernel, MIOpen and hipBLASLt providers) from a rocm-libraries checkout
(`HIPDNN_SRC`, default: this one), stages it into the package, and builds the
wheel. It compiles, so run it on a build machine, never on a shared login node. It:

- pulls the DVC-stored rocKE descriptors first, the way hipdnn-superbuild-ci does;
- configures with the flags in `BUILD_PINS.toml`, which turn tests off and make
  a missing rocKE descriptor a configure error;
- ships only the production engine directory;
- checks that the backend and every provider resolve their dependencies from the
  pip ROCm install (MIOpen and hipBLASLt from rocm-sdk-libraries) and that no
  provider links `libhipdnn_backend` by SONAME.

Several families can build at once from one checkout: the steps that write into
the source tree run under a lock.

For a development build against a staged hipDNN install (the bootstrap takes the
backend and providers from the same prefix):

```
export HIPDNN_VLLM_HIPDNN_PREFIX=<stage>
CXX=$(rocm-sdk path --root)/lib/llvm/bin/clang++ python setup.py build_ext --inplace
python -m pytest tests/unit tests/gpu -m "not conv"
HIPDNN_VLLM_OPS=attention,encoder_attention,rms_norm,fused_add_rms_norm,layer_norm,conv \
  python -m pytest tests/gpu -m conv
```

The provider set is fixed when a process creates its first hipDNN handle, so
the `conv` tests, which need the MIOpen provider, run in their own process with
`conv` enabled. In a process without it they skip.

`docker/Dockerfile` wraps one wheel in an `ubuntu:24.04` image together with the pinned vLLM.
