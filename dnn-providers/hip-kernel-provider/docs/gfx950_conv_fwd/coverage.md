# gfx950 forward convolution verification and coverage

The [graph contract](graph_contract.md) defines the accepted operation and
storage semantics.

## Shipped catalog

The shipped catalog contains 118 variants for 44 synthetic verification
requests. Every request has implicit-GEMM variants with `tile_k=64` and
`tile_k=128` (88 variants); 18 depthwise requests also have direct depthwise
variants (30 variants):

- 10 ungrouped requests: a padded 3x3 smoke shape, pointwise 1x1, strided,
  dilated and non-square cases, each in FP16 and BF16;
- 14 grouped and depthwise requests: two grouped 3x3 cases, depthwise 3x3 at
  stride 1 and 2, odd-channel depthwise 7x7 (default epilogue), a
  channel-multiplier depthwise 3x3 and a dilated grouped 5x5, each in FP16 and
  BF16. The three plain depthwise shapes also have a spatial and a standard
  direct variant;
- 12 depthwise requests with direct variants: 96 channels at stride 1 (with a
  second standard variant with a wider, partly empty channel tile) and at stride 2 on
  odd and even input heights, 64 channels on a 70x40 input at stride 1 and 2
  (with a `block_w=32` variant on rocKE's runtime row loop), and a 5-channel
  17x17 filter (a spatial variant on the runtime row loop), each in FP16 and
  BF16;
- 8 depthwise requests outside the direct family's guard, served by implicit
  GEMM only: 3x3 without padding and with padding 2, unequal strides, and
  dilation 2, each in FP16 and BF16.

Every direct request uses N=2, so a write past the last output row lands in
the second image and fails the reference comparison.

Workload shapes are not part of the shipped catalog. Catalogs built for real
workloads follow the [private workload catalog](README.md#private-workload-catalogs)
procedure, and their verification and coverage evidence stays with them,
outside the source tree.

## Verification

The shipped catalog is checked on gfx950 by:

| Check | What it establishes |
|---|---|
| Adapter and miner tests | The rocKE adapter accepts every catalog spec, refuses grouped pointwise, non-divisible group counts and grid-z overflow, and emits the same IR as the original builder for both families; implicit-GEMM symbols are unchanged by the family fields; the direct guard refuses the row-coverage cases, unequal stride or padding, dilation and channel multipliers; the miner admits grouped records and excludes grouped pointwise. |
| Provider unit tests (`*Gfx950ConvFwd*`) | Graph refusal, every baked constraint of both families, grouped geometry (filter bytes over C/G, grid z over groups), the per-group epilogue rule, the direct row-coverage table and launch geometry, family consistency and placeholder checks, ranking, output validation and buffer nonaliasing. |
| Census (`TestGfx950ConvFwdPacks`) | The packed gfx950 shard registers exactly the generated kernel inventory. |
| Descriptor checks | `hkp_desk_check` and `verify_variant_sets` in full mode: every packaged binary agrees with its descriptor's compiled specialization, with one distinct TOC entry per variant. |
| Provider GPU integration test | Every implicit-GEMM catalog variant, grouped and depthwise included, is forced for both `tile_k` values and compared with the CPU reference. Every direct variant is served from a winner record that ranks it first and compared with the CPU reference; an unforced plan serves the direct default; benchmarking measures both families and reuses its winner; the guarded depthwise graphs offer only `kernel_family=0` and stay correct. |
| Shared integration bundles | `quick/Gfx950ConvFwd/Smoke`, `standard/Gfx950ConvFwd/Spatial` and `standard/Gfx950ConvFwd/Depthwise` run through the exact engine name with `--fail-on-unsupported`. |

Engine attribution is explicit: `hipkernel:Gfx950ConvFwd`
(`0xAD075A5EA86DD563`). Selecting another engine cannot satisfy these checks.
The installed descriptor loader emits its existing warnings for the optional
`provenance` extension.

## Known gaps

- Grouped pointwise convolution is declined: rocKE's pointwise shortcut
  indexes the input and output without the group offset.
- The direct depthwise family covers pure depthwise convolution only. Channel
  multipliers, unequal strides, unequal or asymmetric padding, dilation, and
  padding that leaves an output row or column outside the input stream all fall
  back to implicit GEMM, which leaves most lanes of each tile idle when every
  group has a single channel. The row-coverage cases are rocKE kernel bugs
  (unproduced output rows; writes past the last output row on the runtime row
  loop) that the engine guards against rather than fixes.
- Depthwise 1x1 at stride 1 without padding is declined by both families: the
  shared grouped-pointwise rule refuses it before the direct family is
  considered.
- The unbenchmarked default direct variant is ranked first wherever it applies
  and can be slower than implicit GEMM, for example at stride 2 or once the
  standard kernel takes its runtime row loop. Benchmarking corrects the order.
- 3D, fused, channels-first and other dtypes are integration gaps, not claims
  about rocKE's broader capabilities.

## Performance evidence

The integration probe checks the installed descriptor UUID and all compiled
spec fields of either family, rebuilds the kernel with the family's original
rocKE builder from the equivalent spec,
and uses the same gfx950 device, LLVM flavor, and COMGR library. Compilation,
plan creation, first execution/search, correctness, capture, and warmup are
excluded from steady-state kernel comparison. HIP graph replay and ordinary
submission timing are reported separately.
Selection-log callbacks and backend/plugin logging are suspended during timing
and restored for subsequent cache assertions. The report checks the logging
state and zero callback invocations within that interval.

Candidate ranking uses the existing runtime's one warmup and seven timed
executions with `robustMean`; the probe's steady-state report uses the median
of sample batches. These are separate measurements. A winning variant is the
fastest valid candidate measured during that search, with no claim that a close
ordering is invariant across runs. Persistent cache checks compare UUIDs and
ranking hashes across different process IDs.

Measured performance and the workload shapes it was measured on are retained as
private evidence outside the source tree.
