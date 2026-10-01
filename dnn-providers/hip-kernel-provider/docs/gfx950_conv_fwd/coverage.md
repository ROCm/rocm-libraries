# gfx950 forward convolution verification and coverage

The [graph contract](graph_contract.md) defines the accepted operation and
storage semantics.

## Shipped catalog

The shipped catalog contains 48 variants for 24 synthetic verification
requests, every request with both `tile_k=64` and `tile_k=128`:

- 10 ungrouped requests: a padded 3x3 smoke shape, pointwise 1x1, strided,
  dilated and non-square cases, each in FP16 and BF16;
- 14 grouped and depthwise requests: two grouped 3x3 cases, depthwise 3x3 at
  stride 1 and 2, odd-channel depthwise 7x7 (default epilogue), a
  channel-multiplier depthwise 3x3 and a dilated grouped 5x5, each in FP16 and
  BF16.

Workload shapes are not part of the shipped catalog. Catalogs built for real
workloads follow the [private workload catalog](README.md#private-workload-catalogs)
procedure, and their verification and coverage evidence stays with them,
outside the source tree.

## Verification

The shipped catalog is checked on gfx950 by:

| Check | What it establishes |
|---|---|
| Adapter and miner tests | The rocKE adapter accepts every catalog spec, refuses grouped pointwise, non-divisible group counts and grid-z overflow, and emits the same IR as the original builder; the miner admits grouped records and excludes grouped pointwise. |
| Provider unit tests (`*Gfx950ConvFwd*`) | Graph refusal, every baked constraint, grouped geometry (filter bytes over C/G, grid z over groups), the per-group epilogue rule, output validation and buffer nonaliasing. |
| Census (`TestGfx950ConvFwdPacks`) | The packed gfx950 shard registers exactly the generated kernel inventory. |
| Descriptor checks | `hkp_desk_check` and `verify_variant_sets` in full mode: every packaged binary agrees with its descriptor's compiled specialization, with one distinct TOC entry per variant. |
| Provider GPU integration test | Every catalog variant, grouped and depthwise included, is forced for both `tile_k` values and compared with the CPU reference. |
| Shared integration bundles | `quick/Gfx950ConvFwd/Smoke` and `standard/Gfx950ConvFwd/Spatial` run through the exact engine name with `--fail-on-unsupported`. |

Engine attribution is explicit: `hipkernel:Gfx950ConvFwd`
(`0xAD075A5EA86DD563`). Selecting another engine cannot satisfy these checks.
The installed descriptor loader emits its existing warnings for the optional
`provenance` extension.

## Known gaps

- Grouped pointwise convolution is declined: rocKE's pointwise shortcut
  indexes the input and output without the group offset.
- Depthwise convolution is served by the implicit-GEMM kernel, which leaves
  most lanes of each tile idle when every group has a single channel. rocKE's
  direct depthwise kernel is not packaged.
- 3D, fused, channels-first and other dtypes are integration gaps, not claims
  about rocKE's broader capabilities.

## Performance evidence

The integration probe checks the installed descriptor UUID and all compiled
spec fields, rebuilds direct rocKE from the equivalent original-builder spec,
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
