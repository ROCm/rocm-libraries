# 6hk3 UNKNOWN-Instruction Investigation Artifacts

**Bead:** `rocm-libraries-6hk3` — VPermB32 / unknown-instruction in CMS variant
of TF32 TN 128x160x64 kernel (BPG#11 of `custom_mainloop_scheduling_tf32.yaml`).

**Worktree:** `/home/alvasile/rocm-libraries/.claude/worktrees/agent-a6cf69172d8397f93`
**Branch:** `agent-6hk3-reproducer`

## Kernel descriptor

- **Source:** `Tensile/Tests/common/gemm/gfx950/custom_mainloop_scheduling_tf32.yaml`
  BPG#11 (lines 415-452 of that file).
- **TF32 TN:** `DataType=S`, `F32XdlMathOp=X`, `TransposeA=1`, `TransposeB=0`.
- **MatrixInstruction:** `[16, 16, 32, 1, 1, 4, 5, 2, 2]`
  - MT0 = 16 * 4 * 2 = 128
  - MT1 = 16 * 5 * 2 = 160
- **DepthU:** 64
- **MacroTile:** 128 x 160 x 64
- **Cross-product:** `UseCustomMainLoopSchedule: [0, 1]` (both variants built).

## What's here

| File | Content |
|---|---|
| `6hk3_cms.s` | Full raw assembly for the CMS variant (10085 lines; kernel name contains `_CMS_SN_`). MAINLOOP macro body at lines 2140-3045. |
| `6hk3_noncms.s` | Full raw assembly for the non-CMS variant (10724 lines; no `_CMS_` marker). Inline mainloop at lines 2031-2667. |
| `6hk3_mainloop_cms.txt` | MAINLOOP macro body extracted from `6hk3_cms.s` (lines 2140-3045, 906 lines). |
| `6hk3_mainloop_noncms.txt` | Inline mainloop extracted from `6hk3_noncms.s` (lines 2031-2667, 637 lines). |
| `build.log` | Tensile `--build-only` log. |

## How this was produced

Standalone YAML at:
```
Tensile/Components/CustomSchedule/gfx950/test_yamls/6hk3_tf32_128x160x64_tn.yaml
```
Mirrors BPG#11 of `custom_mainloop_scheduling_tf32.yaml` (lines 415-452)
under the TF32 TN ProblemType (lines 29-37 of the source), but with
`UseCustomMainLoopSchedule: [0, 1]` cross-product so both variants build.

The xj16 inline assertion + shadow inline assertion were LOCALLY disabled in
`Tensile/KernelWriter.py:4985-5001` (uncommitted change in the worktree, kept
the conditional but gated it with `if False and ...`) to let both kernels emit.
Without this disable, the CMS variant would die on the inline
`compare_graphs` / `validate_edge_wait_coverage` assertions before producing
final assembly.

Reproduce:
```bash
cd <worktree>/projects/hipblaslt/tensilelite
rm -rf /tmp/6hk3_out
mkdir -p 6hk3_artifacts
PYTHONPATH=$PWD /home/alvasile/venv/bin/python3 Tensile/bin/Tensile \
    Tensile/Components/CustomSchedule/gfx950/test_yamls/6hk3_tf32_128x160x64_tn.yaml \
    /tmp/6hk3_out --build-only --gpu-targets gfx950 > 6hk3_artifacts/build.log 2>&1
```

The two `MT128x160x64` `.s` files land under:
```
/tmp/6hk3_out/1_BenchmarkProblems/Cijk_Alik_Bljk_S_MX_B_UserArgs_00/00_Final/source/build_tmp/SOURCE/assembly/
```
One contains `_CMS_SN_` in its kernel comment block (CMS variant); the other
does not (non-CMS variant).

## 4x4 grep table — target instructions

Per-file occurrence counts for the four "VPermB32-trigger" instructions the
bead flagged:

| instruction | cms (full) | non-cms (full) | cms (mainloop) | non-cms (mainloop) |
|---|---:|---:|---:|---:|
| `v_swap_b32` | 0 | 0 | 0 | 0 |
| `v_perm_b32` | 0 | 0 | 0 | 0 |
| `v_or_b32`   | 0 | 0 | 0 | 0 |
| `v_mov_b64`  | 2 | 82 | 0 | 20 |

### Headline finding

**None of `v_swap_b32`, `v_perm_b32`, or `v_or_b32` appear anywhere in either
emitted .s file** — not in the full assembly, not in the extracted mainloops.
`v_mov_b64` appears in the non-CMS mainloop (20 occurrences) but is entirely
absent from the CMS mainloop. The CMS variant therefore does **not** reproduce
the VPermB32 / unknown-instruction issue described in the bead at this
MatrixInstruction shape (`[16,16,32,1,1,4,5,2,2]`, MT 128x160x64). Either the
trigger is in a different BPG of `custom_mainloop_scheduling_tf32.yaml` or
the issue is gated by a configuration not exercised by this fixture (e.g.
a different `MatrixInstruction` shape, `LDSTrInst=True`, or another DataType).
