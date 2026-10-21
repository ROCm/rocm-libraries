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
A non-`None` `problem.clamp_arch` must match the selected target (the active device
at launch).
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
use that effective stream. When Q is on the active device and the effective
stream is already current, the runtime submits directly without rebuilding
Torch device/stream contexts. Other devices and streams keep the explicit
context boundary and restore the caller's state. Callers must make input
production visible to it and wait before consuming outputs on another stream.

Compiled kernels are cached. On gfx942/gfx950, sequential eager invocations
reuse bounded scratch owned by the problem and effective device/stream. A
whole-pair submission lock protects segment/reducer ordering. Unchanged pointers
and scalars reuse an immutable packed pair. Changed bindings rebuild it, retaining
previous input storage and argument buffers until completion.

Submit shared 3D attention calls and perform their completion cleanup on one
host thread. This applies across ordinary calls, dispatcher bindings, prepared
execution owners and graph-cache cleanup. Stop submitting before each stream
or device drain, including an external synchronization followed by retention
cleanup. Concurrent or reentrant host calls and cross-thread cleanup are
unsupported; the submission locks and private-scratch fallback do not coordinate
with completion drains. Queued asynchronous GPU launches and sequential host
submission to multiple explicit streams remain supported. Keep preparation,
launch, rebind, replay and close on that same thread.

The asynchronous `no_fence()` path retains launch arguments until a runtime
completion drain such as `synchronize_and_release()`; callers must drain it
periodically. Request-local input storage is released by that completion drain;
the eager cache owns only scratch and immutable packed addresses afterward.

For a sequential decoding loop, prepare an execution once outside timing. Keep
permuted K/V view objects stable, and update their storage and the device length
contents on the execution's stream. The fixed problem describes the maximum
length; changing geometry, dtype, layout or stream requires a new execution.

```python
from kernels import prepare_unified_attention_torch

k_view = k_bshd.permute(0, 2, 1, 3)
v_view = v_bshd.permute(0, 2, 1, 3)
with prepare_unified_attention_torch(
    problem=problem, q=q, k=k_view, v=v_view, out=out,
    cu_seqlens_q=cu_q, seqused_k=lengths, block_table=None,
    softmax_scale=scale, softcap=0.0, backend="3d",
    kv_layout="strided", stream=stream.cuda_stream, use_graph=False,
) as execution:
    # Preparation validates, compiles and launches once before this loop.
    # Queue Q/K/V/length updates on the same stream before each launch.
    for step in range(steps):
        execution.launch()
```

For a loop with fixed bindings on gfx942/gfx950, `replay_scope()` validates and
warms up once on entry, then yields an asynchronous operation with immutable
launch arguments. It establishes the execution's device and stream for the
whole scope. Queue content updates on that stream before each replay:

```python
with prepare_unified_attention_torch(**arguments) as execution:
    with execution.replay_scope() as replay:
        for step in range(steps):
            update_device_contents(step)  # Q/K/V and lengths, within the maximum
            replay()
    execution.rebind(q=next_q, out=next_out)
```

The callable is valid only inside the scope on its entering thread. Keep tensor
objects, storage addresses, layouts, scalar values and launch configuration fixed
throughout the scope; content-only updates need no rebind. Scope exit waits for
queued work before restoring the caller's stream and releasing bound arguments.
`rebind()` drains and validates new storage or scalar bindings between scopes;
it also launches once. Ordinary `rebind()` is available on gfx1250 independently
of the gfx942/gfx950 replay scope. Drain before changing tensor storage metadata. A different
geometry, dtype, layout, device or stream requires a new execution. The owner
rejects overlapping launch, rebind, close or nested scope, but this check does
not make shared completion cleanup safe for concurrent callers. Do not change the
active device/stream or begin an outer capture inside a scope; use `launch()`
for outer capture. Both prepared eager and internal graph execution support
this scope. A failed completion drain requires `close()` to retry cleanup.

Prepared eager launches reuse three scratch tensors. The complete segment/reducer
pair is submitted under one owner; concurrent or reentrant submission is rejected.
Separate streams need separate execution owners. Close on the submitting host
thread after capture and replay submission have stopped. Outer graph bindings
and replay completion remain the caller's responsibility; closing the execution
does not close the caller's graph or drain its capture-stream argument bucket.
Closing waits for the prepared stream's
queued work and releases scratch and captured bindings. For a dispatched strided
binding, the launch callable keeps the owner; call `binding.launch.close()` at the
completion boundary. A binding fixes the effective stream on its first launch.

Internal captures and arbitrary external captures allocate graph-private scratch,
independent of reusable eager storage. Prepared calls inside an outer capture
use the active capture stream instead of their saved eager stream. Warm up first. Internal
direct calls keep at most eight paired graph/resource entries, and a prepared
execution keeps at most one. A binding change that evicts an entry waits only for
that entry's stream before releasing it. Stable replay has no new allocation or
extra completion wait beyond the requested fence. `clear_attention_3d_graph_cache()`
explicitly drains and clears the direct cache. Tensor identity and scalar
parameters select the resident graph; live addresses and normalized
layout metadata are checked before each replay. Content-only updates reuse the
graph. Rebinding the same tensor objects to new storage drains and replaces
the captured graph; it does not recompile the cached kernels. Drain queued work
before mutating storage metadata. Prepared owners still reject geometry, dtype,
layout, device or stream changes.

The context shortcut and resident-key optimization apply to gfx942/gfx950.
gfx1250 retains explicit device/stream contexts and includes live storage
addresses in its graph key, preserving its existing cache eviction behavior.

Asynchronous graph replay retains its resource entry until the stream is drained,
even if the Python execution owner is discarded. Asynchronous eager launches also
retain argument buffers and references until completion; scratch reuse bounds
storage allocations, while argument bookkeeping still grows until the explicit
drain. If callers share graph memory pools, Torch's ordering requirements apply.
Replay of one graph and writes to caller-owned inputs/outputs require ordering.

## Test coverage

The [numeric tests](../library/tests/test_strided_kv_decode_numeric.py) compare
the direct runtime and registered binding against an independent CPU FP32
reference after input quantization. They include batched BHSD/BSHD, different
K/V strides, padding, unpadded capacity tails, changing valid lengths, empty
sequences, windows, paged regressions, graph replay, content updates, and
same-object Q/output storage rebinding for direct and prepared graph calls.
Concurrency tests cover eager calls, internal graphs, and separate external graphs captured on one
stream and replayed on different streams. They check scratch isolation and
replay after temporary launch references are drained. Set
`ROCKE_REQUIRE_DECODE_GPU=gfx942` or `gfx950` to require the target GPU rather
than skip. No performance claim follows from this correctness coverage.
