# Non-paged KV input to tiled decode

The gfx942 and gfx950 unified 3D split-KV kernels accept non-paged KV through
`kv_layout="strided"`. This specialization changes global-memory addressing;
it reuses the tiled matrix-core computation, LDS layout, segment workspace,
and reducer. It does not copy KV or construct an identity page table.
The default `kv_layout="paged"` retains its existing ABI and addressing.

## Input contract

K and V have logical shape `[B, Hkv, capacity, D]`. Contiguous BHSD storage
works directly. For contiguous BSHD storage, pass a metadata-only
`cache.permute(0, 2, 1, 3)` view. K and V may have different outer strides,
including padding, with these bounds:

- fp16 or bf16, D in 64/128/256; one query per sequence;
- positive, nonoverlapping strides, unit D stride, 16-byte-aligned rows/base;
- per-head byte span and token stride at most `0x7fff0000`;
- query-to-KV head ratio in 1/2/4/8/16;
- contiguous Q and output `[B, Hq, D]`, on the same device as KV and metadata.

The selected segment and reducer must each fit the target's LDS capacity.
Admission uses the resolved tile dimensions and LDS allocation dtypes. In
particular, automatic gfx942 D256/block64 selection requires 74,752 bytes,
exceeding its 65,536-byte capacity, and is rejected before compilation. An
explicit smaller tile is checked using its actual footprint; selection does
not silently resize it. gfx950 D256/block64 fits its target capacity.

Capacity need not be a page or tile multiple. Device `seqused_k` is contiguous
int32 `[B]`, with each value in `[0, problem.max_seqlen_k]`, and
`problem.max_seqlen_k <= capacity`. Device `cu_seqlens_q` is contiguous int32
`[B+1]` with contents `[0,1,...,B]`. These **contents are caller contracts**:
validation checks tensor metadata without copying device values to the host.
The query position is `seqused_k[b] - 1`, supporting bottom-right causal
alignment and an optional sliding window. Empty sequences produce zero output.

The registered `attention_strided_decode` candidate declines bias, sinks, FP8,
top-left causal alignment, and multi-query prefill. Arbitrary D strides and
overlapping or unaligned views and K/V/output scaling are not supported. This change adds no LSE output.

## Usage and validation

Use the existing [runtime entry point](../library/kernels/common/attention_unified.py) with the following
arguments, after creating the problem and device tensors described above:

```python
run_unified_attention_torch(
    problem=problem,
    q=q,
    k=k_bhsd,  # or k_bshd.permute(0, 2, 1, 3)
    v=v_bhsd,
    out=out,
    cu_seqlens_q=query_offsets,
    seqused_k=kv_lengths,
    block_table=None,
    softmax_scale=problem.head_size**-0.5,
    softcap=0.0,
    backend="3d",
    kv_layout="strided",
    stream=stream,
)
```

For registry consumers, set `AttentionRequest.kv_layout="strided"`; the
[candidate](../library/dispatch/attention/strided_decode.py) provides spec building
and `bind_torch`. Direct and dispatched launches share the same spec policy.
Automatic strided selection keeps invariant hoisting disabled regardless of
`HIPDNN_GFX942_3D_HOIST`; an explicit kernel spec may enable it. A non-`None`
`problem.clamp_arch` must match the selected target (the active device at launch).
The binding accepts only `softmax_scale` and `stream` keyword overrides; it
rejects softcap, bias/sinks tensors, and unknown arguments. Explicit segment
and reducer specs must agree with the runtime problem's shapes, dtype, window,
and enabled features. The low-level [layout adapter](../library/kernels/common/attention_kv_cache.py) packs
independent K/V batch/head strides as i64 byte offsets and token strides/span
as i32 bytes. Separate C builder entry points preserve installed paged spec
structs; strided symbols carry `_stridedkv`.

## Streams and workspace ownership

The shared 3D runtime resolves `stream=0` to Torch's current stream on Q's
device. Scratch allocation, initialization, and segment/reduce launches all
use that effective stream. Callers must make input production visible to it
and wait before consuming outputs on another stream.

Compiled kernels are cached. Each eager invocation allocates independent
scratch, so overlapping calls cannot overwrite each other's partial results.
The asynchronous `no_fence()` path retains launch arguments until a runtime
completion drain such as `synchronize_and_release()`; callers must drain it
periodically. This adds eager allocator work; no performance claim is made.

Internal graph capture allocates scratch from Torch's graph-private pool;
cached replay performs no new scratch allocation. External captures use the
same ownership rule and may replay on a different stream. Warm up compilation
before capture. If callers explicitly share graph memory pools, they must
follow Torch's ordering and non-concurrency requirements. Replays of the same
graph and writes to caller-owned input/output buffers still require ordering.

## Test coverage

The [numeric tests](../library/tests/test_strided_kv_decode_numeric.py) compare
the direct runtime and registered binding against an independent CPU FP32
reference after input quantization. They include batched BHSD/BSHD, different
K/V strides, padding, unpadded capacity tails, changing valid lengths, empty
sequences, windows, paged regressions, and graph replay. Concurrency tests cover
eager calls, internal graphs, and separate external graphs captured on one
stream and replayed on different streams. They check scratch isolation and
replay after temporary launch references are drained. Set
`ROCKE_REQUIRE_DECODE_GPU=gfx942` or `gfx950` to require the target GPU rather
than skip. No performance claim follows from this correctness coverage.
