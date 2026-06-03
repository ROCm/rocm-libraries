# gfx950 Async and Direct-Footprint Probe Results

Status snapshot from 2026-06-02 17:25.

This note extends the deep-fusion experiment records with the latest schedule
experiments on the gfx950 fused conv0 -> conv1 -> pool prototype.

## Kernel and Target

Prototype:

```text
conv0 3x3, C=8 -> K0=32
-> ReLU
-> 1x1 conv1, K0=32 -> K1=24
-> ReLU
-> 2x2 stride-2 maxpool
```

Full target shape:

```text
input:       [1, 2160, 3840, 8]
conv0 out:   [1, 2160, 3840, 32]
conv1 out:   [1, 2160, 3840, 24]
pool out:    [1, 1080, 1920, 24]
```

Current tuned base:

```text
pool_tile=4x8
tile_m=128
tile_n=32
tile_k=16
warp_m=2
warp_n=1
block_size=128
pipeline=mem
async_dma=False
```

## Async / Ping-Pong Experiment

The existing conv `compv4` / async-DMA machinery was exposed through the fused
spec as `pipeline` and `async_dma`.

Correctness:

```text
pipeline=compv4, async_dma=False, tile_k=16: correct
pipeline=compv4, async_dma=True,  tile_k=16: incorrect
pipeline=compv4, async_dma=True,  tile_k=32: incorrect
```

Full-target timing, including invalid-output async variants:

```text
pipeline=mem,    async=False, tile_k=16: 0.369873 ms
pipeline=compv4, async=False, tile_k=16: 0.368769 ms
pipeline=compv4, async=True,  tile_k=16: 0.369600 ms  (incorrect output)
pipeline=compv4, async=True,  tile_k=32: 0.400059 ms  (incorrect output)
```

Conclusion: async does not currently show a performance upside, even if
correctness is ignored. The failure appears specific to `async_dma`; `compv4`
without async is correct. Keep async disabled until the async loader contract is
adapted to the custom rectangular `m_index_fn` / epilogue override path.

## Hot-Loop Unroll Experiment

The existing conv `unroll_k` path was exposed as an opt-in experiment. It failed
correctness on the default target-channel smoke shape:

```text
pipeline=mem, tile_k=16, unroll_k=True: incorrect
max_abs_diff ~= 0.881, bad_count=112/1536
```

The likely issue is that the unrolled path removes the post-MFMA synchronization
between K tiles. In the standalone conv path this was intended to expose overlap,
but in the fused rectangular carrier the next K-tile load can race with MFMA
consumption of the current LDS tile. The flag is now rejected by validation for
this fused prototype. A correct software pipeline should use true ping/pong
buffers with a wait/sync discipline proven for the custom `m_index_fn` path,
rather than simply dropping the barrier.

## Static Probe Summary

The following utility probes were run from `dsl_docs/optimization/utilities`:

```text
probe_occupancy.py
probe_intrinsic_counts.py
probe_isa_inspect.py
probe_config_sweep.py
```

### Occupancy

```text
variant              VGPR  AGPR  SGPR  LDS bytes  waves/CU  wg/CU  limiter
default_tk16          100    32    50     26112        12      6    LDS
materialized_cache    136    32    56     28992        10      5    LDS
direct_footprint       76    32    37     22848        14      7    LDS
old_tk32              104    32    50     31232        10      5    LDS
```

All inspected variants are LDS-limited by the static estimate. Direct-footprint
has the best static occupancy, but still loses at runtime, so occupancy is not
the deciding factor.

### Intrinsic Counts

Lowered LLVM intrinsic counts:

```text
variant              mfma32x32x16  raw buffer stores  s.barrier  s.waitcnt
default_tk16              7                7              6          6
materialized_cache        7                7              7          7
direct_footprint          7                7              7          7
old_tk32                  9                7              6          6
```

`tile_k=16` reduces MFMA calls from 9 to 7 relative to the older `tile_k=32`
variant. Cache-oriented variants do not reduce MFMA count; they add input
footprint staging work around the same MFMA structure.

### ISA Mix

Static ISA category counts:

```text
variant              VALU  ds_read  ds_write  waitcnt  barrier  vmem_load  MFMA
default_tk16          654      39       70       40        5        6        6
materialized_cache    928      55       94       57        6       14        6
direct_footprint      955     109       82       94       14       18       14
old_tk32              635      42       70       40        5        6        8
```

This explains the cache/direct regressions:

- materialized cache adds more VALU, LDS reads/writes, waitcnts, and VMEM loads;
- direct footprint improves static occupancy but greatly increases LDS reads,
  waitcnts, barriers, and scalar address arithmetic;
- the default path keeps MFMA operand delivery more regular via vectorized LDS
  reads from the materialized A/B tiles.

## Config Sweep Result

`probe_config_sweep.py` was wired to the HIP event timing harness. One run showed
some process/order sensitivity, but the stable takeaway matches the standalone
timing:

```text
default path is faster than materialized input cache and direct footprint.
tile_k=16 and tile_k=32 are close; tile_k=16 has better padded-work accounting.
```

Follow-up alternating repeat timing gave:

```text
tk32:       ~361 us
tk16:       ~358-370 us
direct16:   ~462 us, one noisy run at 723 us
cache16:    ~499 us
```

## Interpretation

The latest evidence points away from HBM input bandwidth as the primary limiter.
If raw input bandwidth were the bottleneck, footprint caching should have helped.
Instead, the static probes show that footprint variants increase on-chip
overhead enough to dominate any reduction in global input traffic.

Current likely gaps:

```text
1. LDS operand delivery and synchronization.
2. Scalar coordinate arithmetic in direct-footprint fragment loading.
3. C-shuffle staging cost between conv0, conv1, and pool.
4. Need for an MFMA-fragment-friendly footprint LDS layout before direct-conv
   loading can become profitable.
```

## Next Steps

Recommended next experiments:

1. Keep `pipeline=mem`, `async_dma=False` as the default.
2. Keep materialized-cache and direct-footprint paths opt-in only.
3. Design a footprint LDS layout whose rows/columns match the actual
   `32x32x16` MFMA A-fragment access pattern, then retest direct footprint.
4. Capture rocprof counters for default vs direct-footprint:
   LDS busy/conflict metrics, MFMA utilization, VALU utilization, and memory
   bandwidth.
5. Isolate C-shuffle cost by timing variants that stop after conv0 or conv1
   staging before adding pool.
