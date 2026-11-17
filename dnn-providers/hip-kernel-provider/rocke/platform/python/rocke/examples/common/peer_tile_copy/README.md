# Peer tile-copy validation methodology

The [spec-driven builder](../../../instances/common/peer_tile_copy.py) validates
ordinary memory accesses before introducing distributed GEMM scheduling. Its
Python and C++ implementations use the same scalar, masked global-memory IR.

**Current evidence:** CPU logical-access tests, independent builder/binding
parity, and offline COMGR compilation. **Not yet established:** actual HIP peer
reads/writes, runtime cross-device visibility, bandwidth, or compute overlap.

## Operation and placement

The logical operation is a strided rectangle copy:

```text
Y[dst_origin + (row, col)] = X[src_origin + (row, col)]
```

A coordinate participates only if it lies in the copy rectangle and both
source/destination views. Otherwise no load or store occurs. Destination guards
and padding are unchanged. Each CTA owns one output tile; the final thread chunk
also masks elements outside the compile-time tile area.

The same kernel implements two hardware probes:

- Peer source, local destination: test peer reads.
- Local source, peer destination: test peer writes.

Ownership is not encoded in the kernel. HIP establishes directed mappings and
supplies addresses. Source and destination allocations must be disjoint and
remain alive across every consumer. No LDS, atomics, matrix instructions,
transport calls, or asynchronous DMA submissions are involved.

This probe is not a GEMM input loader: invalid positions are left unchanged,
not zero-filled. A later GEMM loader must establish its own padding semantics
before consuming a staged tile, especially for a partial reduction dimension.

## Spec and launch arguments

- `PeerTileCopySpec`: tile rows/columns, block size, dtype, and stable kernel name.
- `PeerTileCopyProblem`: copy extents, source/destination view extents, origins,
  and positive element strides. Origins may be negative for clipping tests.
- `peer_tile_copy_signature`: two global pointers plus signed i64 scalar fields.
- `peer_tile_copy_grid`: one CTA per tile; skip launch when the rectangle is empty.

Host validation rejects coordinate/byte-span overflow and overlapping destination
rows. Raw-pointer validity, allocation size, peer mappings, and allocation
lifetimes remain the caller's responsibility. Use one loaded module per device;
a function from a different device is not an interchangeable launch handle.

Synthetic CPU-only authoring example:

```python
from rocke.instances import (
    PeerTileCopyProblem, PeerTileCopySpec, build_peer_tile_copy,
    peer_tile_copy_grid, peer_tile_copy_signature,
)

spec = PeerTileCopySpec(tile_rows=3, tile_cols=7, dtype="bf16")
problem = PeerTileCopyProblem(
    rows=7, cols=9,
    src_rows=6, src_cols=9, dst_rows=8, dst_cols=10,
    src_stride_row=16, dst_stride_row=16,
    src_row=-1, dst_row=1,
)
kernel = build_peer_tile_copy(spec, arch="gfx950")
signature = peer_tile_copy_signature(spec)
grid = peer_tile_copy_grid(problem.rows, problem.cols, spec)
# problem.values(source_pointer, destination_pointer) supplies packed ABI values.
# Building this IR does not resolve HIP or launch any GPU work.
```

Scalar typed-memory lowering preserves i32 or i64 GEP indices. The copy kernel
uses i64 address arithmetic so a large view stride does not silently truncate.
Values are loaded/stored in their native dtype without conversion; tests include
raw signed-zero, infinity, and NaN payloads.

## CPU and compiler checks

Run from `rocke/platform/` using the project development environment:

```text
python -m pytest tests/instances/test_peer_tile_copy.py tests/core/test_global_memory_indices.py
python tools/check_byte_identity.py --only peer_tile_copy --build-root <build-directory>
```

Repeat the byte-identity gate for the supported LLVM flavors. Build the optional
C++ binding from the same archive, then run
`tests/instances/test_peer_tile_copy_binding.py`. A skipped binding lane is not
binding validation.

The CPU tests interpret the actual scalar IR, rather than only duplicating the
address formula. They compare exact read/write sets, bit-preserving results,
unchanged guards, unique writers, clipped origins, extra CTAs, partial thread
chunks, and offsets beyond the i32 range. They do not model GPU caches or links.

The standalone parity family covers both supported targets. Dedicated tests
check its LLVM and canonical-IR snapshots. Preserve existing golden entries:
`run_diff.py --record-golden --only ...` replaces an entire snapshot key rather
than merging a family and must not be used to add this family.

## Hardware gate to run next

1. Obtain an explicitly authorized allocation with two GPUs on the target.
2. Record device identities, directed peer capability, partition modes, and the
   resolved HIP/COMGR stack without assuming topology from the gfx target.
3. Allocate/publish distinct source payloads and guarded destinations under each
   owner device. Load modules under their execution devices.
4. Establish only the directed mappings needed by the probe with
   `peer_access_scope`; retain all buffers and modules until the scope exits.
5. Exercise peer reads and peer writes in both directions, along with local-only
   controls. Test aligned and clipped/tail cases and repeated launches.
6. Observe successful completion before readback or freeing memory. Scope exit
   drains participating execution devices before removing owned mappings. If a
   drain fails, recover before freeing allocations or disabling preserved paths.
7. Compare copied-back bits and guards on CPU. Classify compilation, mapping,
   launch, completion, and result failures separately.

Do not introduce a readiness/reuse protocol or claim communication/compute overlap
from these copy probes. Those require later, independently validated mechanisms.
Keep internal topology, raw logs, and measured software-performance values out of
the repository, following the
[platform plan and evidence policy](../../../../../dsl_docs/development/distributed_tile_gemm_poc.md).
