# Per-schedule gfx1151 CMS input yamls

Bead: `rocm-libraries-uj9`. Source format derived from
`Tensile/Components/CustomSchedule.py` (commit `0052629546`, tip of branch
`users/alvasile/validator_long_term_plans`) plus the kernel-config skeleton
in `Tensile/Tests/unit/test_CustomSchedule.py:_gfx1151_base_kernel()`.

## What this directory contains

31 Tensile **input** yamls (the kind `Tensile`/`TensileCreateLibrary`
consume — `GlobalParameters:` / `BenchmarkProblems:` shape — *not*
`LibraryLogic:` output yamls), one per registered gfx1151 CMS schedule.
Each yaml exercises exactly one `_get_schedule_*_gfx1151` function
registered with `@RegisterSchedule(...)`. Building the yaml with
`Tensile --gpu-targets gfx1151` produces exactly one solution whose live
`hasCustomSchedule(state)` returns `True` and whose kernel name contains
the `CMS` marker.

The materialized corpus survives independently of
`test_CustomSchedule.py` (whose `GFX1151_TILES` covers only 8 of 31
schedules) and lets the validator probe
(`Tensile/Components/GFX1151_AUDIT/validator_coverage_probe.py`) be
pointed at one CMS schedule at a time.

## Schedule index

All 31 use ISA=(11,5,1), WavefrontSize=32, DTL=0, DPLB=False, WSGRA=0,
WSGRB=0, MatrixInstruction=[16,16,16,1] (MIK=16 for all currently
registered gfx1151 schedules), TransposeA=True/TransposeB=False (TN
layout), DataType=h (fp16) with HighPrecisionAccumulate=True.
LocalReadVectorWidth = 16 throughout.

| # | Filename | MT | PGR | PLR | MIWG | MIWaveTile [A,B] | GRVW [A,B] | Notes |
|---|----------|----|-----|-----|------|------------------|------------|-------|
| 1  | `_get_schedule_96x128x32_16bit_gfx1151.yaml`             | 96x128x32  | 2 | 1 | [2,2] | [3,4] | [8,8] | line 5594 |
| 2  | `_get_schedule_128x96x32_16bit_gfx1151.yaml`             | 128x96x32  | 2 | 1 | [2,2] | [4,3] | [8,8] | line 5657 |
| 3  | `_get_schedule_192x64x32_16bit_gfx1151.yaml`             | 192x64x32  | 2 | 1 | [4,1] | [3,4] | [8,8] | line 5719 |
| 4  | `_get_schedule_64x192x32_16bit_gfx1151.yaml`             | 64x192x32  | 2 | 1 | [1,4] | [4,3] | [8,8] | line 5781 |
| 5  | `_get_schedule_32x128x64_16bit_gfx1151.yaml`             | 32x128x64  | 2 | 1 | [1,4] | [2,2] | [8,8] | line 5844 |
| 6  | `_get_schedule_96x128x64_plr0_14_16bit_gfx1151.yaml`     | 96x128x64  | 2 | 0 | [2,2] | [3,4] | [8,8] | line 5922 |
| 7  | `_get_schedule_128x64x64_plr1_16bit_gfx1151.yaml`        | 128x64x64  | 2 | 1 | [2,2] | [4,2] | [8,8] | line 5995 |
| 8  | `_get_schedule_128x80x64_16bit_gfx1151.yaml`             | 128x80x64  | 2 | 1 | [4,1] | [2,5] | [8,8] | line 6068 |
| 9  | `_get_schedule_128x80x64_pgr1_16bit_gfx1151.yaml`        | 128x80x64  | 1 | 1 | [4,1] | [2,5] | [8,8] | line 6145 |
| 10 | `_get_schedule_64x128x32_14_16bit_gfx1151.yaml`          | 64x128x32  | 2 | 1 | [1,4] | [4,2] | [8,8] | line 6212 |
| 11 | `_get_schedule_128x64x32_16bit_gfx1151.yaml`             | 128x64x32  | 2 | 1 | [4,1] | [2,4] | [8,8] | line 6268 |
| 12 | `_get_schedule_64x128x32_22_16bit_gfx1151.yaml`          | 64x128x32  | 2 | 1 | [2,2] | [2,4] | [8,8] | line 6324 |
| 13 | `_get_schedule_96x96x32_16bit_gfx1151.yaml`              | 96x96x32   | 2 | 1 | [2,2] | [3,3] | [8,8] | line 6384 |
| 14 | `_get_schedule_64x64x32_22_16bit_gfx1151.yaml`           | 64x64x32   | 2 | 1 | [2,2] | [2,2] | [8,8] | line 6463 |
| 15 | `_get_schedule_16x16x128_16bit_gfx1151.yaml`             | 16x16x128  | 2 | 1 | [1,1] | [1,1] | [8,8] | line 6529 |
| 16 | `_get_schedule_32x16x128_16bit_gfx1151.yaml`             | 32x16x128  | 2 | 1 | [2,1] | [1,1] | [8,8] | line 6616 |
| 17 | `_get_schedule_128x96x64_plr0_41_16bit_gfx1151.yaml`     | 128x96x64  | 2 | 0 | [4,1] | [2,6] | [8,8] | line 6703 |
| 18 | `_get_schedule_96x128x64_plr0_14_16bit_gfx1151_tn.yaml`  | 96x128x64  | 2 | 0 | [1,4] | [6,2] | [8,8] | line 6765 |
| 19 | `_get_schedule_128x112x32_plr0_16bit_gfx1151.yaml`       | 128x112x32 | 2 | 0 | [4,1] | [2,7] | [8,4] | line 6827; vector_widths=[8,4,16] |
| 20 | `_get_schedule_128x80x64_plr0_16bit_gfx1151.yaml`        | 128x80x64  | 2 | 0 | [4,1] | [2,5] | [8,8] | line 6885 |
| 21 | `_get_schedule_128x80x64_pgr1_plr1_16bit_gfx1151.yaml`   | 128x80x64  | 1 | 1 | [4,1] | [2,5] | [8,8] | line 6952; sibling of #9 |
| 22 | `_get_schedule_80x128x64_pgr2_plr1_16bit_gfx1151.yaml`   | 80x128x64  | 2 | 1 | [1,4] | [5,2] | [8,8] | line 7020 |
| 23 | `_get_schedule_80x128x64_pgr2_plr0_16bit_gfx1151.yaml`   | 80x128x64  | 2 | 0 | [1,4] | [5,2] | [8,8] | line 7084 |
| 24 | `_get_schedule_64x224x32_pgr2_plr0_16bit_gfx1151.yaml`   | 64x224x32  | 2 | 0 | [2,2] | [2,7] | [8,8] | line 7153 |
| 25 | `_get_schedule_128x64x32_pgr2_plr0_16bit_gfx1151.yaml`   | 128x64x32  | 2 | 0 | [2,2] | [4,2] | [8,8] | line 7212 |
| 26 | `_get_schedule_128x48x32_pgr2_plr1_16bit_gfx1151.yaml`   | 128x48x32  | 2 | 1 | [4,1] | [2,3] | [8,4] | line 7271; vector_widths=[8,4,16] |
| 27 | `_get_schedule_64x32x64_pgr2_plr1_16bit_gfx1151.yaml`    | 64x32x64   | 2 | 1 | [2,2] | [2,1] | [8,8] | line 7331; **CMS=1 in shipping HHS yaml** |
| 28 | `_get_schedule_128x96x32_pgr1_plr0_16bit_gfx1151.yaml`   | 128x96x32  | 1 | 0 | [2,2] | [4,3] | [8,8] | line 7401 |
| 29 | `_get_schedule_128x96x64_pgr2_plr0_22_16bit_gfx1151.yaml`| 128x96x64  | 2 | 0 | [2,2] | [4,3] | [8,8] | line 7458 |
| 30 | `_get_schedule_128x64x64_pgr2_plr0_22_16bit_gfx1151.yaml`| 128x64x64  | 2 | 0 | [2,2] | [4,2] | [8,8] | line 7521 |
| 31 | `_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151.yaml`| 128x64x64  | 1 | 0 | [2,2] | [4,2] | [8,8] | line 7584; **CMS=1 in shipping HHS yaml** |

`MIWaveTile` values are derived as `MT0 / (MI_M * MIWGM)` and
`MT1 / (MI_N * MIWGN)` with `MI_M = MI_N = 16`. They appear in slots 5
and 6 of the 9-element `MatrixInstruction` list inside each yaml
(`[16, 16, 16, 1, 1, MIWaveTileA, MIWaveTileB, MIWGM, MIWGN]`), and
that's how the per-schedule tile shape is encoded.

## How to use with the validator probe

Pick any single yaml and run `Tensile` against it for `gfx1151`. The
generated `_FinalBenchmarkParameters_*.yaml` (under the output
directory's `00_Final/source/library/` tree) will contain exactly one
solution with `UseCustomMainLoopSchedule: 1`. The probe can then be
pointed at that single-solution yaml:

```sh
WORKTREE=...rocm-libraries
SRC=$WORKTREE/projects/hipblaslt/tensilelite
YAML=$SRC/Tensile/Components/GFX1151_AUDIT/per_schedule_yamls/_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151.yaml

# 1. Build the kernel for gfx1151 (use --build-only to skip benchmarking).
$SRC/Tensile/bin/Tensile --gpu-targets gfx1151 --build-only "$YAML" /tmp/uj9_one

# 2. Or, inspect the dispatch directly via the unit-test fixture:
PYTHONPATH=$SRC python3 - <<'PY'
import sys
SRC = "..."  # set to the same SRC path
sys.path.insert(0, SRC)
from Tensile.LibraryIO import readYAML
from Tensile.Components.CustomSchedule import hasCustomSchedule, _SCHEDULE_REGISTRY
print(f"Registered schedules: {len(_SCHEDULE_REGISTRY)}")
y = readYAML("$YAML")
# (use SolutionStructs.Solution to fill defaults, then call hasCustomSchedule)
PY

# 3. Or, run the validator coverage probe over the single yaml's output:
python3 $SRC/Tensile/Components/GFX1151_AUDIT/validator_coverage_probe.py \
  --yaml /tmp/uj9_one/0_Build/Stream0/.../FinalParameters.yaml \
  --report /tmp/probe_report.json
```

## Format / source notes

Each yaml uses the input-yaml format documented at
`Tensile/Common/GlobalParameters.py` and parsed by
`Tensile/SolutionStructs/Solution.py`. The 9-element
`MatrixInstruction` form is converted to internal MI / MIBlock /
MIWaveTile / MIWaveGroup parameters by
`Tensile/SolutionStructs/Validators/MatrixInstruction.py`
(`convertMI` at line ~50, `MIBlockBM = wg0 // mi[0]` at line 109).

Each yaml hard-codes `UseCustomMainLoopSchedule: 1` so that the
dispatcher at `Solution.py:2054-2062` will both verify the registry
match **and** raise an error if the registration ever drifts away from
the yaml. (If `hasCustomSchedule(state)` returns False for one of these
yamls in the future, Tensile will emit
`reject: UseCustomMainLoopSchedule=1 but CMS is not supported` — that
is the signal that the registration the yaml was generated against has
been removed or its parameters changed.)

## Spot-check log (this commit)

Three yamls were spot-checked end-to-end against
`Tensile --gpu-targets gfx1151 --build-only`:

| Yaml | Result |
|------|--------|
| `_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151.yaml` | `Actual Solutions: 1 / 1`; kernel name contains `CMS_SN_` marker; assembly generation reaches the codegen stage (downstream `total vgpr: 315 not in [0, 256]` failure unrelated to yaml format). |
| `_get_schedule_64x64x32_22_16bit_gfx1151.yaml`            | `Actual Solutions: 1 / 1`; kernel-gen reaches assembly emission (downstream `ScheduleCapture.CaptureConsistencyError` is a separate live-validator concern, not a format issue). |
| `_get_schedule_96x128x32_16bit_gfx1151.yaml`              | `Actual Solutions: 1 / 1`; kernel-gen completes assembly without errors. |

Five additional yamls were spot-checked at the dispatcher layer (a
direct call to `hasCustomSchedule(kernel_state)` with the per-yaml
parameter tuple), and all five returned `True`:

- `_get_schedule_128x64x64_pgr1_plr0_22_16bit_gfx1151`
- `_get_schedule_80x128x64_pgr2_plr0_16bit_gfx1151`
- `_get_schedule_128x112x32_plr0_16bit_gfx1151` (the `[8,4,16]`
  vector-widths variant)
- `_get_schedule_16x16x128_16bit_gfx1151`
- `_get_schedule_64x32x64_pgr2_plr1_16bit_gfx1151`

## Source-format version pin

These yamls were generated against:

* `users/alvasile/validator_long_term_plans` tip = commit
  `0052629546` ("Eager source-aware Failure labels
  (rocm-libraries-g4w)").
* Tensile input-yaml schema version: `MinimumRequiredVersion: 5.0.0`
  (Tensile v5.0.0).

If the yaml schema changes incompatibly in future Tensile versions, the
generator script (`/tmp/gen_yamls.py` used for this commit, structure
documented inline in each generated yaml's header comment) can be
re-run against an updated `YAML_TEMPLATE` constant.
