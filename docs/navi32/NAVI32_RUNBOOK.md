# Runbook — developing a navi32 (gfx1101) catalog on a navi31 (gfx1100) card

Method only. Findings live in `REPORT.md`. Every trap below was actually hit.

---

## 0. What this setup can and cannot measure

navi32 = RX 7800 XT = **60 CUs / 30 WGPs**. This card is 96 CUs / 48 WGPs. Emulation has two
independent halves and they are **not** interchangeable:

| half | mechanism | what it buys | cost |
|---|---|---|---|
| selection | `--sm_count_target 60` | kernel choice + StreamK grid as a 60-CU part | none |
| execution | `HIPBLASLT_BENCH_CU_MASK=60` | real 60-CU timing (measured 62.4% vs ideal 62.5%) | **~37% of runs hang** |

Neither emulates navi32's **memory system**: this card keeps ~960 GB/s and 96 MB Infinity
Cache against navi32's 624 GB/s / 64 MB. Memory-bound shapes therefore look better than real
navi32 no matter which half you use. Say so next to every memory-bound number.

**Practical protocol:** run the full sweep with selection fidelity only (fast, 0 hangs), and
validate the ranking on a smaller masked subset. Because every arm then runs on the same
96 CUs, the execution error is common-mode and **arm ratios stay meaningful** — only absolute
throughput is optimistic.

## 1. CU masking — three traps, in the order you will hit them

**Trap 1: `ROC_GLOBAL_CU_MASK` looks like it works and does nothing.** It changes
`hipGetDeviceProperties().multiProcessorCount` (48 -> 30), which is exactly the number you
would check to confirm it worked. It does **not** restrict execution (compute-bound
throughput measured 100.4% of unmasked) and does **not** reach hipBLASLt's selector
(`TENSILE_STREAMK_DYNAMIC_GRID=0`, whose grid *is* `computeUnitCount`, still reads 48).

**Trap 2: each mask bit is a WGP, not a CU.** This is why trap 1 looks total. A 60-bit mask
requests 60 WGPs = 120 CUs, which on a 48-WGP part is no restriction at all. Halve your CU
number before building the mask.

*A single measurement cannot distinguish "inert" from "mis-scaled".* Sweep the knob:

| mask bits | measured | implies |
|---|---|---|
| 16 | 34.1% | 32 CUs |
| 32 | 64.4% | 64 CUs |
| 60 | 92.9% | ~96 CUs, clamped |

Every row is 2x the CU count the bit count implies — that slope is what identifies the units.

**Trap 3: an over-range mask hangs the GPU.** Requesting more WGPs than the device has does
not error; the process wedges at 1% GPU and needs `kill -9`. Clamp and warn.

**Verify the mask by throughput, never by a reported count.** A real N/M restriction must
show ~N/M on a compute-bound GEMM. Ours reads 62.4% against an ideal 62.5%.

## 2. The masked stream hangs 2% of runs — USABLE, despite what this section used to say

`hipExtStreamCreateWithCUMask` intermittently wedges: the run emits its result row and then
never exits, holding the GPU lock. It is a teardown race, not a bad shape — isolated retries
always pass — and it is **position-dependent, not library-dependent**: in a 5-arm interleave
the *same library* had 0 timeouts as the first arm and 2 as the last. Always give each run a
tight `timeout` plus a `pkill` of the hung child, or one hang stalls every later arm.

**The rate, measured properly:**

| sample | masked timeouts | rate |
|---|---|---|
| original 8-run probe | 3 of 8 | 37% — *this section's original claim* |
| **1 242-run sweep** | **25 of 1 242** | **2.0%** |

**An 8-run sample cannot tell 37% from 2%** (the 95% CI on 3/8 reaches down to ~8%), and the
first masked run after an idle GPU reliably hangs — which is exactly what a short probe
oversamples. On that 8-run basis the campaign fell back to selection fidelity and left its
central premise untested for the whole run.

**Execution fidelity is affordable: a full 998-shape masked sweep costs ~3 hours.** Prefer it.
See [`MASKED_60CU_VALIDATION.md`](MASKED_60CU_VALIDATION.md), which used it to confirm the
catalog win at genuine 60 CUs (+22.7% wall-clock against a 0.11 pt A/A floor).

**Rule: measure a failure rate at the scale you intend to run at.** A rate estimated from a
handful of runs is a decision about the whole campaign made on almost no data.

## 2b. `HIPBLASLT_TENSILE_LIBPATH` must point at the arch subdirectory

`.../libs/<arm>/library/gfx1100` — not the arm root, not `.../library`. Point it one level up
and *every* row comes back `status=error`, at full speed, with rows appearing at the normal
rate. Check `status` counts before reading any number out of a sweep.

## 3. Retargeting a logic file has TWO ISA sites

Missing either yields a file that looks retargeted but is not:

* top level element `[2]` — the string the build filters on;
* **every solution in `[5]`** — its own `ISA: [11,0,N]`.

`retarget_logic.py` rewrites both and hard-fails if any solution lacks the key. Element `[1]`
is the arch *name* and must match the directory the build enumerates.

Solution names do **not** encode the ISA, so they survive a retarget — which is what makes
the gfx1101 build gate a like-for-like check.

## 4. The gfx1101 build gate

Kernel generation is host-side; no navi32 GPU is needed.

```bash
python3 retarget_logic.py <src>.yaml gate/logic/navi32/<x>.yaml --isa gfx1101 --name navi32
cd src/projects/hipblaslt/tensilelite
PYTHONPATH=<build>/tensilelite/rocisa:. python3 Tensile/bin/TensileCreateLibrary \
    gate/logic gate/out HIP --architecture gfx1101 --jobs $(nproc) --logic-format yaml
```

Pass conditions: 0 `not a valid operand`, 0 `overflowedResources` (1 = >256 VGPR,
4 = occupancy), and the produced ELF reports `Flags: 0x46, gfx1101`. **Check the ELF flags** —
a successful-looking run that emitted a gfx1100 object proves nothing.

Worth running even though gfx1100 and gfx1101 have identical VGPR/LDS/wave caps: instruction
selection can still differ between targets. Here it does not (298/298 passed), and now that
is measured rather than assumed.

## 5. Origami: adding an architecture

Nine sites, all mechanical (`shared/origami/`):

`hardware.hpp` — enum, `arch_name_to_enum`, `arch_enum_to_name`, `get_arch_constants`,
MI-latency map. `hardware.cpp` — `cus_per_multiProcessorCount` (**the RDNA x2 list; this is
what makes 30 WGPs read as 60 CUs**), XCD count, and two capability predicates.

Copying a sibling architecture's constants is safe for CU count, because the CU count is not
baked into them — it arrives at runtime as `multiProcessorCount * cus_per_multiProcessorCount`.
The constants that *would* need recalibration are the memory ones (`mem2`/`mem3`).

Origami's switches are exhaustive (they carry an explicit `case architecture_t::Count`), so a
missed site fails the build. Cross-check anyway: `grep -c` the sibling arch and the new one
should return the same number.

## 6. GridBased -> Prediction is a small transformation

A Prediction library has **no shape table** (`[7] = None`); Origami ranks `[5]` per shape.
`to_prediction.py` nulls `[6]`–`[9]` and sets `[11] = Prediction`.

Confirm the Prediction path is actually live by checking it selects a **different kernel**
than the GridBased arm on the same shape — with no table to look up, a difference can only
come from the model.

## 7. Benchmarking

- **`--fixed-iters` is mandatory when arms differ in library size** (here 60 vs 238 kernels):
  tiered iteration counts charge one-time library init unevenly.
- **A/A arm placed LAST** so it brackets the whole interleave and sees maximum position drift.
- **Report geomean AND flops-weighted wall-clock.** They have disagreed in sign on this
  workload.
- **Jackknife every wall-clock claim.** Here the top 5 shapes hold ~59% of kernel time.
- **Break out by size and geometry.** The headline number hid a 2.3–2.7x collapse confined to
  tiny/GEMV shapes.
- Batch mode (`--yaml`) is ~10x fewer process launches but **hangs partway through a
  1000-shape file**; per-invocation with a tight timeout proved more reliable.

## 7b. `--logic-filter` does not recurse — a wrong glob builds NOTHING and reports success

`invoke build --logic-filter 'navi32/*'` exits 0 with no errors and produces a library
directory. It also processes **zero kernels**: the glob matches the `GridBased` *directory*,
not the YAMLs inside it. The correct form is `navi32/GridBased/*`.

| filter | LibraryLogicFiles | kernels | errors |
|---|---|---|---|
| `navi32/*` | 0 | **0** | 0 |
| `navi32/GridBased/*` | 38 | **1 743** | 0 |

A clean exit code proved nothing here. **Verify a build by its artifacts** — kernel count,
`LibraryLogicFiles`, the size of `Tensile/library/<arch>/`, and the ELF flags of a produced
`.co`. The vacuous build left a 356 KB directory containing only extop/transform helpers and
no GEMM libraries at all; the real one is 9.8 MB with 39 code objects.

## 7c. The Aux ProblemTypes need extra bench flags, and fail silently without them

`*_Bias_AuxH_*` / `*_Bias_AuxB_*` carry an auxiliary output. Benchmarked with the same flags
as their non-Aux siblings, the library reports **`NO solution found`** and every row lands as
`status=error` with `gflops=0.00` and an empty kernel name — 231 rows of it before I looked
at the status column rather than the row count.

Required: `--use_e --aux_type f16_r --activation_type gelu`.

`--use_e` alone is not enough; it fails with *"The activation type 1 does not support
'--use_e'"*. Any of `gelu`/`relu` satisfies it.

**Check `status` counts, not row counts.** A sweep producing rows at the expected rate can be
producing nothing but errors, and the harness will happily run to completion.

## 7d. Two ways a build failure lies about itself

**`grep -i error` matches `ImportError`, `error_code` and `stderr`.** A gfx1102 gate run was
reported here as "2 assembler errors, 0 code objects" on the strength of that grep. There
were no assembler errors: the log said
*"rocisa C++ sources are newer than the built `_rocisa.so` — bindings are stale."* Trusting
the grep would have produced a confident, entirely wrong conclusion about a different
architecture. **Grep the specific diagnostic** — `not a valid operand`,
`overflowedResources`, `Total kernels processed` — never the substring `error`.

**A scratch-branch merge test leaves environment residue.** Checking out back to the original
branch restores file *contents* but not *timestamps*, and rocisa's staleness guard keys on
mtime. So a merge test two hours earlier can break an unrelated build later. Symptom is the
ImportError above; fix is `invoke rocisa`, or rebuild `_rocisa` in the build tree. Verify with
`python3 -c "import rocisa"` before blaming the thing you are actually testing.

## 8. Operational

- `flock` every GPU action. A hung child holds it — kill by PID, and remember
  `pgrep -f <pattern>` **matches its own command line** (use `pgrep -f '[p]attern'`).
- Never launch a second sweep while one is running; they fight over the lock and both stall
  in a way that looks like a benchmark bug.
- Flush result CSVs **per shape**, not per batch, or a long run is uninspectable and a crash
  loses everything.
- The harness resumes: it skips `(shape, arm, rep)` triples already present in the CSV.
