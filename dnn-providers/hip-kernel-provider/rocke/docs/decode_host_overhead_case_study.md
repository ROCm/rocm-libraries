# Ordinary split-KV decode host submission case study

This change reduces repeated Python host work for ordinary gfx942/gfx950 split-KV attention calls with stable executable bindings. Kernel computation, segment/reducer geometry and the native provider launch path are preserved.

An ordinary call previously allocated three scratch tensors and rebuilt both argument buffers. The runtime now caches scratch by resolved executable pipeline, launch geometry, segment count, device, effective stream and submitting host thread. Compatible maximum-KV changes share the same owner when the resolved geometry remains identical. Changed live pointers or scalars rebuild the immutable packed pair; admission and binding checks still run on each call.

The cache owns scratch and packed addresses. Request storage is snapshotted and retained separately before submission, once per immutable operation per completion batch. The caller submits shared attention work and performs cleanup on one host thread, with submission stopped before completion drains. Failed submission or completion leaves in-flight resources reachable for later cleanup. Ordinary asynchronous calls acquire no new HIP wait or event.

The eight-entry limit is an untuned heuristic matching the existing direct graph-cache default. A small LRU retains recent geometry/device/stream owners while bounding resident entries. Correctness does not depend on the value eight. The current performance validation uses one resident owner and does not tune cache capacity; this is not a byte budget. Each owner uses `4 * total_q * num_query_heads * num_segments * (head_size + 2)` scratch bytes. Explicit eager-cache clearing drains resident streams before releasing scratch. Evicted in-flight owners remain retained until their submitted streams drain. Capture uses private scratch, and prepared executions and internal graphs preserve their separate ownership.

The device/stream context shortcut applies only when the effective device and stream are already current on gfx942/gfx950. Other contexts are explicitly entered and restored. Existing gfx1250 context and graph-cache behavior is preserved; gfx1250 hardware validation is outside this change's scope.

## Validation contract

Host regression tests exercise the public saved executable-candidate binding, reuse counts, changed pointers/scalars and geometry, capture isolation, cache clearing, failed drains, pair ownership and per-thread default-stream ownership. Adjacent graph/prepared tests verify live storage checks and existing ownership behavior. The numerical suite compares retained runtime paths against independent references on matching hardware.

Performance revalidation compares final source against the pinned base with the same executable bindings, compiled work and launch geometry. Setup, compilation and drains stay outside host enqueue timing; HIP-event intervals are measured separately. Numerical checks run before and after timing, and paired repeated trials alternate source-arm order. The validation harness is maintained outside this PR. No new measured result is established by this explanation; report completed evidence and its limits with the final PR validation.

See the [canonical caller and ownership contract](strided-kv-decode.md#streams-and-workspace-ownership), [eager host regressions](../library/tests/test_attention_3d_eager_reuse.py) and [runtime retention contract](../platform/dsl_docs/runtime/comgr_and_hipmodule.md#bound-owner-retention).
