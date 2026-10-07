# Single-token attention decode

`hipkernel:AttentionDecode` is a generic-ingestor pack for the existing gfx942 and
gfx950 tiled 2D unified-attention kernels. It performs one attention step per
hipDNN graph execution. The application owns the KV cache, appends the current
K/V token, and updates a device length scalar before executing on its stream.
The pack does not project activations or update the cache.

## Initial supported slice

| Tensor | hipDNN logical dimensions | Element strides | Type |
|---|---|---|---|
| Q, O | `[1, 8, 1, 128]` | `[1024, 128, 1024, 1]` | BF16 |
| K, V | `[1, 2, C, 128]` | `[256*C, 128, 256, 1]` | BF16 |
| `seq_len_kv` | `[1, 1, 1, 1]` | `[1, 1, 1, 1]` | INT32 |

`C` is cache capacity, a multiple of 64 from 64 through 65536. K/V storage is
contiguous and token-major (BSHD). Strides of unit-extent axes are immaterial.
Q/K/V/O buffers must be 16-byte aligned; the length and workspace must be at
least 4-byte aligned. Operands and scratch must not overlap. All tensors are
nonvirtual device buffers. Accumulation is FLOAT.

Before the first append, zero-initialize the full K and V cache allocations on
(or ordered before) the execution stream. Repeat this initialization when
starting a new sequence or reusing cache storage, then append the first token.
Do not clear the cache between appends within a sequence.

The kernels load complete 64-token tiles. The length masks attention scores,
but does not prevent reads of unused K/V slots; in particular, zero probability
times a NaN V operand still produces NaN. Uninitialized or nonfinite unused
storage is therefore outside this initial interface's supported input domain.
The adapter does not initialize or sanitize the caller's cache.

The caller must keep `1 <= seq_len_kv <= C` and order cache/length writes before
execution on the hipDNN stream. Length includes the current token. Device length
values are not copied to the host or validated at launch. An invalid length is
outside this interface's supported input domain. Concurrent executions require
separate workspace and appropriate application synchronization of shared cache
storage.

Set a finite scalar attention scale, `set_seq_len_kv(length)`, and
`set_generate_stats(false)`. Both unmasked and bottom-right causal attention
attend every valid KV entry for a single query. Top-left causal attention,
windows, bias, stats/LSE, external page tables, quantization, dropout, query
length tensors, batching and multiple query tokens are declined.

Select this engine explicitly when validating reachability:

```cpp
graph.build_operation_graph(handle);
graph.create_execution_plan_ext(
    hipdnn_data_sdk::utilities::engineNameToId("hipkernel:AttentionDecode"), {});
graph.check_support();
graph.build_plans();
// Allocate graph.get_workspace_size(...). Zero the full K/V cache before each
// new sequence, then reuse this plan within the sequence:
// append K/V, write Q and seq_len_kv, graph.execute(handle, buffers, workspace).
```

Check every returned status in application code. The integration test
`IntegrationGpuAttentionDecode` demonstrates complete error-checked execution.
Set `HIPDNN_TEST_EXPECTED_ARCH=gfx942` (or `gfx950`) in required GPU runs to
fail if the allocated device is missing or has the wrong architecture.

The adapter uses `4 * (4 + C/64)` workspace bytes: query offsets `[0,1]`, two
padding words, and an identity page table. It copies this small immutable table
on the caller stream before launch; it never copies the KV cache. Grid `(2,1,1)`
and block `(64,1,1)` match both compiled builders. No new rocKE emitter or Python
`bind_decode.py` is required. Production packaging lowers the two authored rocKE
descriptors to per-architecture kpack archives.

This first slice makes no graph-capture or performance guarantee. Split-KV,
additional shapes/dtypes, paged frontend inputs, speculative decoding and LSE
remain separate extensions. The kernel interface is checked against the recorded
argument names, sizes and kinds during preparation.
