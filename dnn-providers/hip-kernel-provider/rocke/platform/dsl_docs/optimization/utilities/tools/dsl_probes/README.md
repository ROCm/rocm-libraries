# CK DSL Static Inspection Probes

Small, focused scripts that turn a `KernelDef` (or a spec dataclass that
builds one) into a concrete, actionable optimization signal in **under
one second per variant** and **without launching the kernel**.

They are vendored from a working set of one-off probes after
generalization: removed hardcoded user paths, replaced attention-only
specs with a `--demo` argument plus public programmatic entry points,
and standardized output formatting so the results compose with the
rest of `dsl_docs/optimization/utilities/`.

## When to use these

The optimization runbook (`../../../optimization_runbook.md`) tells you
to (1) verify correctness,
(2) measure with hygiene, then (3) **inspect the generated artifact
before chasing performance**. These probes are step (3).

| Want to answer | Use |
|---|---|
| "Will the kernel fit at the occupancy I expect?" | `probe_occupancy.py` |
| "Are the right MFMA / async-DMA / ds.read.tr16 intrinsics emitted?" | `probe_intrinsic_counts.py` |
| "What is the ISA opcode mix between two variants?" | `probe_isa_inspect.py` |
| "Does the HIP-debug backend agree with LLVM-direct on this kernel?" | `probe_lowering_compare.py` |
| "Which `(num_warps, tile_size, …)` variant is fastest for my shapes?" | `probe_config_sweep.py` (+ your `run_fn`) |
| "Best rocke vs. baseline (Triton / AITER) latency, single-shape pair?" | `probe_targeted_bench.py` |
| "Profile one shape with rocprof, with the build/warm cost amortized out" | `probe_rocprof_single.py` |

The rocprof-counter-driven probes (`stage4_analyze/analyze_lds_conflicts.py`,
`stage5_compare/compare_rocprof_stats.py`, `utils/profile_register_usage.py`)
are complementary: they need a live launch and a rocprof CSV. These
DSL probes feed *into* the rocprof workflow by narrowing the variant
list before paying the rocprof cost.

## Scripts

### `probe_occupancy.py`
**Input**: list of `(label, KernelDef, waves_per_workgroup)`.
**Output**: per-variant VGPR / AGPR / SGPR / spill / LDS bytes, plus a
coarse estimate of waves/CU and workgroups/CU, with the apparent
limiter (`VGPR`, `LDS`, `WAVES_PER_EU_HINT`, …) annotated.

How it works: compiles each kernel via `compile_kernel`, dumps the
HSACO notes with `llvm-readelf --notes`, and applies the gfx950 / gfx942 /
gfx90a per-CU resource caps. The occupancy math matches what `rocprofv3`
reports as `OccupancyPct` for typical kernels within a few percent.

Useful for: tile / block-size sweeps where the wall-clock difference
is on the order of LDS or VGPR bumps. If the limiter switches from
`LDS` to `VGPR` across the sweep, the optimal tile size will sit at the
boundary.

### `probe_isa_inspect.py`
**Input**: list of `(label, KernelDef)`.
**Output**: per-variant ISA opcode histogram (MFMA, `ds_read{_tr}`,
`ds_write`, `vmem_load`, `vmem_store`, `waitcnt`, `barrier`, VALU /
SALU sub-buckets), plus the most frequent `s_waitcnt` operand patterns.

How it works: compiles via `compile_kernel`, runs
`llvm-objdump -d --mcpu=gfx950`, parses the output with the same opcode
regex `analysis/isa.py` uses internally. Adds the SALU/VALU
sub-buckets that are missing from `parse_isa` because they only matter
during optimization, not at runtime.

Useful for: confirming that an MFMA-atom switch actually emitted the
new shape, that the cshuffle epilogue uses wide `buffer_store_dwordx{2,4}`
instead of scalar stores, that the `s_waitcnt` count went down after
adding sched groups, etc.

### `probe_intrinsic_counts.py`
**Input**: list of `(label, KernelDef)`.
**Output**: per-variant histogram of AMDGCN LLVM intrinsics
(`mfma.f32.*`, `ds.read.tr16.*`, `ds.bpermute`, `ds.swizzle`,
`raw.ptr.buffer.load.lds`, `cvt.pk.f32.fp8`, `s.waitcnt`, …) plus
structural counts (load, store, fmul, fadd, fmuladd, br, phi).

How it works: just `lower_kernel_to_llvm(kdef)` plus a small regex
table. Faster than `probe_isa_inspect` because it doesn't shell out to
`llvm-objdump` and doesn't pay the COMGR HSACO step.

Useful for: pre-COMGR sanity checks — did the spec switch flip the
right intrinsic call? — and for catching DSL-level miscompiles where
an `if` statement is silently elided by an AST rewriter (the canonical
symptom: stores went from N to 0, see the optimization runbook §10).

### `probe_lowering_compare.py`
**Input**: list of `(label, KernelDef)`, `--arch=gfx950`.
**Output**: side-by-side HSACO sizes from the LLVM-direct and HIP-debug
backends; optional VGPR / LDS deltas from `llvm-readelf` on both.

How it works: production path is `compile_kernel`; debug path is
`lower_kernel_to_hip(kdef)` written to disk and compiled with
`hipcc --offload-arch=<arch> --genco`. The two backends should agree
on a healthy kernel; large size or resource deltas usually mean one
backend is missing an op lowering (the HIP-debug backend's op coverage
is narrower than LLVM-direct's).

Useful for: triaging "kernel works in LLVM but the HIP debug source
won't compile" parity issues. The
`examples/common/hip_lowering_parity.py` harness exercises the same path on a
matrix of all shipped instance specs.

### `probe_config_sweep.py`
**Input**: `build_fn`, a `base_spec` dataclass, a list of
`replace()` override dicts, an optional `run_fn(KernelArtifact, label)
-> latency_us`.
**Output**: build status, HSACO size, optional latency per variant,
and the "best" variant.

How it works: `dataclasses.replace(base, **override)` for each
override, then `build_fn` then `compile_kernel`, then your `run_fn`.
This script intentionally does **not** own the timing — wire it to
`time_launches`, `torch.cuda.Event`, or `probe_targeted_bench` so the
methodology lives in one place.

Useful for: building one canonical sweep table from a single source
of truth instead of duplicating the boilerplate every time a new lever
is added.

### `probe_targeted_bench.py`
**Input**: list of `(name, shape_dict)`, a `candidate_fn(shape)`, an
optional `baseline_fn(shape)`.
**Output**: per-shape `candidate_us`, `baseline_us`, and speedup.

How it works: a single CUDA-event window per shape with a fixed
`warmup` and `iters`. The CUDA-event timer does not require
`torch.cuda.synchronize()` between iterations the way `time_launches`
does, which avoids a known Triton autotune-vs-sync interaction.

Useful for: production-trace-shaped end-to-end bench tables that need
to be honest about per-shape variance. See `optimization_runbook.md` §2.3
section "Benchmark Hygiene" — discard the first run, report median,
and bench from a fresh process where it matters.

### `probe_rocprof_single.py`
**Input**: `--builder=pkg.mod:fn`, `--problem-json=...`,
`--iters=10 --warmup=5`.
**Output**: per-iter wall time after warmup.

The script's job is to put **only the kernel-execution window** inside
the rocprof trace. Build / COMGR / warmup happen before the timed
loop, so a `rocprofv3 -i metrics.txt -- python probe_rocprof_single.py
…` invocation captures hardware counters for the steady-state
kernel and not for the warmup.

Useful for: pairing with the rocprof-based tools under
`utilities/tools/stage4_analyze/` and `stage5_compare/`.

## Layout summary

```text
dsl_probes/
├── README.md                         # this file
├── probe_occupancy.py                # llvm-readelf notes → occupancy
├── probe_isa_inspect.py              # llvm-objdump → opcode histogram
├── probe_intrinsic_counts.py         # lower_kernel_to_llvm → intrinsic histogram
├── probe_lowering_compare.py         # LLVM-direct vs HIP-debug HSACO compare
├── probe_config_sweep.py             # dataclasses.replace sweep over a spec
├── probe_targeted_bench.py           # one-window CUDA-event bench across shapes
└── probe_rocprof_single.py           # single-process rocprof-friendly harness
```

## Conventions

- **No hardcoded user paths.** Every script bootstraps `rocke` from
  the package layout, falling back to the canonical workspace path
  only if all else fails. To pin a custom location, set `PYTHONPATH`
  before invoking.
- **No GPU launch in the smoke `--demo`.** Demos only need `compile_kernel`
  and inspection. This makes the probes safe to run in CI without a
  device attached (other than `probe_rocprof_single.py`, which by
  definition requires a GPU).
- **One probe per question.** Composable Python entry points
  (`probe_occupancy(entries)`, `probe_isa_inspect(entries)`,
  `count_intrinsics(ir_text)`, `bench_shapes(shapes, …)`,
  `probe_config_sweep(...)`, …) so you can call any of them from a
  one-off `.py` exploration script without re-implementing the parser.

## Running

Set `PYTHONPATH` so `rocke` is importable, then invoke the probe
directly. From the `composablekernel/python` directory:

```bash
export PYTHONPATH=.
python dsl_docs/optimization/utilities/tools/dsl_probes/probe_occupancy.py \
  --demo attention_tiled_2d
```

Each script also exposes a Python entry point so you can wire it into
a one-off exploration script — see the per-script docstring.
