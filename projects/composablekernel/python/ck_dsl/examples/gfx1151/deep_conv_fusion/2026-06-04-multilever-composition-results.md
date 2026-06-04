# gfx1151 Deep-Fusion: multi-lever + composition campaign

Status snapshot 2026-06-04. Box: AMD Strix Halo, **gfx1151** (RDNA3.5), wave32,
`wmma_f32_16x16x16_f16` only. Kernel:
`instances/gfx1151/deep_fused_conv_pool.py` at the full encoder_0 target shape
(N=1, H=2160, W=3840, C=8, K0=32, K1=24 -> pooled (1080,1920,24), ~50.9 GFLOP).

Continues `2026-06-03-2230-...-results.md`, which left the kernel at **7.54
useful TFLOP/s (~12.8% of the ~59 TF f16 WMMA peak)** via the direct-conv0
footprint cache (now the default). That campaign isolated each lever one-at-a-
time. This one drives the *remaining* candidate levers, **tests their
compositions** (not just one-at-a-time), and gives the bold latency-hiding idea
(butterfly register-fusion) an honest lever/non-lever verdict.

Discipline unchanged (runbook S8.6): every lever behind a correctness-neutral
spec toggle; verified **bit-exact (bad_count=0)** on toy + multi-CTA shapes
before any speed quote; speed only from **same-process interleaved A/B** at the
full shape; rocprofv3 unavailable on Windows, so interpretation is bench +
instruction-shape reasoning.

## Levers under test

| Lever | Spec toggle | Idea |
|---|---|---|
| L1 | `waves_per_eu=2` | force a 2nd WG/CU (direct-conv0 freed LDS ~43->~26 KB) for latency-hiding |
| L2 | `sched_policy="compv3"/"compv4"` | `sched_group_barrier` DS_READ/MFMA interleave hints around each WMMA k-atom |
| L3 | `mask_maxpool=True` | predicate the maxpool tail instead of `scf_if` (cut branch/divergence) |
| L4 | `butterfly_conv01=True` | delete c0_smem + barrier 2: transpose conv0 C-frag -> conv1 A-frag in-register |

All of L1/L2/L3 verified bad_count=0 on the toy (1536 elems) and multi-CTA
(H32W64 grid (1,4,4), 12288 elems) shapes, and in the compare harness at the
full shape, before timing. L4 was rejected by `is_valid_spec` before codegen
(see verdict below).

## Composition matrix (full interleaved A/B)

`compare_configs.py --h 2160 --w 3840 --rounds 8 --iters 50 --warmup 200`, all
bad_count=0:

| Config | med (ms) | spread | TFLOP/s | vs base |
|---|---|---|---|---|
| base direct-conv0 | 6.774 | 1.8% | 7.52 | +0.0% |
| L1 wpe2 | 7.197 | 1.5% | 7.08 | **-5.9%** |
| L2 sch compv3 | 6.623 | 9.0% | 7.69 | +2.3% |
| L2 sch compv4 | 6.655 | 5.8% | 7.66 | +1.8% |
| L3 maskpool | 7.650 | 13.9% | 6.66 | **-11.5%** |
| L1+L2 | 6.937 | 5.8% | 7.35 | -2.3% |
| L1+L3 | 8.792 | 43.5% | 5.80 | -22.9% |
| L2+L3 | 7.552 | 5.3% | 6.75 | -10.3% |
| L1+L2+L3 | 8.880 | 9.3% | 5.74 | -23.7% |

L2's delta was inside its spread, so a focused tighter confirmation
(`--l2only --rounds 12 --warmup 300`, spreads 3.1-3.6%):

| Config | med (ms) | spread | TFLOP/s | vs base |
|---|---|---|---|---|
| base direct-conv0 | 6.872 | 3.6% | 7.42 | +0.0% |
| L2 sch compv3 | 6.735 | 3.1% | 7.57 | **+2.0%** |
| L2 sch compv4 | 6.750 | 3.1% | 7.55 | +1.8% |

## Verdicts

- **L2 (scheduler hints) is the only positive lever: ~+2% (7.42 -> 7.57 TF),
  compv3 ~= compv4.** Confirmed across two sessions (+2.3% then +2.0%, tight
  spreads). Small because these are tiny GEMMs (conv0 K=72/5 k-atoms, conv1
  K=32/2 k-atoms) with few ds/mma per k-atom, so the interleave hint has little
  schedule to reshape. Correctness-neutral and free; adoptable as default
  (`sched_policy="compv3"`). Left as a non-default toggle here to keep the
  established 7.5x baseline stable.
- **L1 (2 WGs/CU) is a non-lever: -5.9%.** The plan's premise was that direct-
  conv0's freed LDS (~26 KB/CTA, 2x26<64) would admit a free 2nd resident WG.
  But occupancy here is **VGPR-bound, not LDS-bound**: forcing
  `waves-per-eu=2` makes the compiler cap VGPRs, and the lost per-wave registers
  cost more latency-hiding than the second WG buys. The textbook "free 2nd WG"
  lever does not apply once registers, not LDS, are the binding occupancy limit.
- **L3 (mask maxpool) is a non-lever: -11.5%.** The maxpool tile has exactly
  `pool_tile_h*pool_tile_w = 32` pooled pixels = one wave, so the `scf_if` guard
  was already warp-uniform (no real divergence to remove). Masking instead
  activates the other 7 warps to compute redundant, discarded work -> pure loss.
- **L4 (butterfly register-fusion) is an analyzed non-lever; rejected before
  codegen.** Chaining WMMA where the producer's N becomes the consumer's K is a
  genuine **cross-lane 16x16 transpose**: the conv0 C-fragment scatters N across
  lanes (`col = lane%16`) while the conv1 A-fragment needs that same N in the
  per-lane fragment slots (`k = slot`). The only wave32 cross-lane vehicle,
  `ds_bpermute`, is **itself an LDS-unit instruction** ("uses LDS as the shuffle
  vehicle") and broadcasts a single register per source lane -- it cannot hand
  different slots to the different destination lanes that read the same source.
  So a correct transpose needs ~8 bpermutes per output slot (~64-128 LDS-unit
  ops/warp) to replace the c0_smem path's ~4 ds_reads + one **warp-uniform
  (single-WG) barrier**. On this LDS/latency-bound kernel that is a guaranteed
  large regression: **here the LDS round-trip is the cheap path.** Same anti-
  staging thesis as the rest of the campaign, opposite sign -- staging only
  loses when it adds *redundant* LDS traffic (im2col R*S re-staging); the conv0
  ->conv1 hand-off moves each value exactly once, which is already minimal, and
  a register transpose would move it many times through the same LDS unit.
  (This matches the community pattern: RDNA3 flash-attention stages the P@V
  operand through LDS for the same reason.)

## Net

**No lever or composition beats the direct-conv0 baseline meaningfully.** The
only positive is L2 sched hints at ~+2% (7.57 TF, ~12.9% of peak). The kernel
remains issue/latency-bound and now appears close to the practical ceiling for
this tiny-GEMM fused shape on gfx1151: warp-count and the anti-redundant-staging
levers (direct-conv0) were the big wins in the prior campaign; the remaining
occupancy/schedule/transpose levers are bounded by VGPR pressure (L1), tiny
k-atom counts (L2), already-uniform control flow (L3), and the LDS-unit cost of
cross-lane shuffles (L4).

### No composition super-additivity

The compositions showed **no super-additive interaction** -- every stack simply
inherited the regressions of its worst member. L1xL2 (the hypothesized "more
warps change the ds/mma ratio the hints target") was -2.3%, i.e. L2's +2% could
not offset L1's -6%. Anything containing L3 landed -10% to -24%. There is no
combination of these levers that recovers more than they individually cost.

## Reproduce

```text
# correctness per toggle (toy + multi-CTA), from projects/composablekernel:
PYTHONPATH=python ../../.venv-ckdsl/Scripts/python.exe \
  -m ck_dsl.examples.gfx1151.deep_conv_fusion.deep_fused_conv_pool_verify \
  --arch gfx1151 --verify --direct --sched compv3 --h 16 --w 16 --c 8 --k0 32 --k1 24
#   (also --waves-per-eu 2, --mask-maxpool; --h 32 --w 64 for grid (1,4,4))

# full composition matrix:
PYTHONPATH=python ../../.venv-ckdsl/Scripts/python.exe \
  -m ck_dsl.examples.gfx1151.deep_conv_fusion.compare_configs \
  --h 2160 --w 3840 --rounds 8 --iters 50 --warmup 200

# focused L2 confirmation:
PYTHONPATH=python ../../.venv-ckdsl/Scripts/python.exe \
  -m ck_dsl.examples.gfx1151.deep_conv_fusion.compare_configs \
  --h 2160 --w 3840 --rounds 12 --iters 50 --warmup 300 --l2only
```
