# gfx1151 deep-fusion native-int: tile-shape optimization & profiling verdict

**Date:** 2026-06-05
**Kernel:** `concat -> conv0 3x3 -> ReLU -> conv1 1x1 -> ReLU -> 2x2/s2 maxpool`,
native integer pipeline (conv0 iu8 WMMA, conv1 iu4 WMMA, integer requant + maxpool).
**Shape:** `(1,8,2160,3840) -> (1,24,1080,1920)`, full board run (gfx11-generic
ELF on the Strix Halo iGPU, 4 WGP / 8 CUs), warmup=100, iters=100.

## TL;DR

- The native-int pipeline (iu8 conv0 + iu4 conv1) is **bit-exact**
  (`max_abs_diff=0`, `bad=0/49,766,400`) and now the default path.
- The kernel is **memory/latency-bound, not VALU- or occupancy-bound**. The
  prior session's static-ISA conclusion ("86% VALU -> cut VALU to go faster")
  did not hold up under measurement.
- The live free lever is **pool-tile shape**, not instruction count. A
  wide-short **2x32** tile (`tile_m=256`) is optimal at **16.1 ms**, vs **18.5 ms**
  for the previously-assumed-best 8x8 -- a **~13% win for zero extra code**.
- `2x32` is now baked in as the default config in the example
  (`--pool-tile-h 2 --pool-tile-w 32 --sched compv4`).

## Why the static-VALU premise was wrong

Static ISA analysis of the best config showed ~86% of instructions were VALU
(integer bit-manip for requant/pack/maxpool) and concluded the path to <10 ms
was "emit fewer VALU instructions" (packed-int16 `v_pk_*`, lane-local C0 pack).
Measurement overturned this:

1. **More waves are slower.** `--waves-per-eu 2` -> **24.26 ms** (worse). If we
   were VALU-throughput-bound, extra waves would hide latency and help. They hurt
   -> we are latency-bound, not throughput-bound.
2. **Cutting static VALU did not help.** Lever 2 (lane-local C0 repack,
   `--repack-c0`) removed ~12.5% of static VALU (v_and 84->4, no `ds_bpermute`)
   yet ran **18.11 -> 20.00 ms (~10% slower)**. The VALU it removed lived
   *inside* the conv1 WMMA k-loop where it was already overlapped with WMMA/LDS
   latency; the repack replaced hidden work with a mandatory full-workgroup
   barrier + an extra C0 LDS round-trip on the critical path. Kept behind the
   `--repack-c0` flag as a documented negative result; do not enable.
3. **Schedule policy saturates.** `compv3`/`compv4`/`intrawave` all land ~18.18 ms
   at 8x8; `mem` is worse (20.70 ms). Reordering instructions cannot buy more
   once latency, not issue, is the wall.
4. **Tile *shape* moves the needle at fixed occupancy** -> the cost is in the
   memory access pattern, not the math.

## Pool-tile sweep (native-int direct, sched compv4, full shape)

| pool tile (h x w) | tile_m | latency (ms) | note |
|-------------------|--------|--------------|------|
| 2 x 32            | 256    | **16.14**    | **best, new default** |
| 4 x 16            | 256    | 16.48        | |
| 4 x 24            | 384    | 16.87        | |
| 4 x 32            | 512    | 17.03        | |
| 6 x 16            | 384    | 17.12        | |
| 8 x 8             | 256    | 18.51        | prior assumed-best |
| 4 x 8             | 128    | 19.83        | under-filled |

All sweep configs verified bit-exact. The pattern: **wide W extent (32) at
`tile_m=256`** wins. Wide-short tiles make conv output writes and the maxpool
read footprint coalesce along the fast (W) axis; tall-square tiles fragment the
same traffic. Note `tile_m=256` at *both* 2x32 and 8x8 -- shape, not size,
is what differs, isolating access pattern as the cause.

## What this means for <10 ms

The remaining ~1.6x must come from the **memory/latency** side, not VALU:
- reduce global+LDS round-trips on the critical path (the conv0->conv1 C0 handoff
  and the maxpool read footprint);
- improve coalescing / reuse of the input footprint cache across the pool tile;
- overlap the conv0 epilogue requant with conv1 staging rather than separating
  them with a barrier.
Packed-`v_pk_*` requant (old "lever 1") and lane-local C0 packing (old "lever 2")
are **deprioritized**: they target throughput the kernel is not bound on.

## How to reproduce

Build locally for the board (gfx11-generic), run the manifest on the board:

```
# build (dev host): emits hsaco + manifest.json
python ck_dsl/examples/gfx1151/deep_conv_fusion/deep_fused_conv_pool_verify.py \
    --arch gfx11-generic --native-int --direct \
    --n 1 --h 2160 --w 3840 \
    --emit-hsaco /tmp/deep/deep.hsaco

# run (board): verify (integer-exact) + timing
python -m ck_dsl.run_manifest deep.hsaco manifest.json --verify
```

`--pool-tile-h 2 --pool-tile-w 32 --sched compv4` are now defaults, so the
plain `--native-int --direct` build is the best config.
