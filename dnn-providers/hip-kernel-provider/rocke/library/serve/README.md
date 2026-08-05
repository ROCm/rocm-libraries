# `rocke-serve` — JSON entry point for external kernel tooling

An external kernel-optimization orchestrator profiles a serving workload, finds
the hot attention kernel, and wants a faster one. It cannot build a kernel;
rocKE can. This package is the seam between them: JSON in, a planned — and where
a GPU is present, verified and measured — kernel out.

It is invoked as a subprocess rather than imported, because rocKE runs under its
own ROCm and `ROCKE_LLVM_FLAVOR` environment, which is not necessarily the
caller's.

## Usage

```bash
# What can rocKE serve here? Needs no request and no device.
python -m serve probe --arch gfx950

# Dispatch only. Reproducible for an arch that is not attached.
python -m serve plan request.json [result.json]

# Plan, then verify and measure whatever the machine allows.
python -m serve run request.json result.json
```

Both source roots must be importable, as for every other entry point in this
tree (see `rocke/BUILDING.md`):

```bash
export PYTHONPATH=library:platform/python
```

Exit codes are `0` served, `2` declined, `1` malformed. They are for the shell;
the result file is the actual answer, and `run` always writes one — including on
failure, so a caller can tell a declined shape from a crash.

## Why a request carries two views of one shape

Each entry in `requests` has both an `attention_request` and a `problem`, and
they disagree on purpose.

| view | `total_q` | used for |
|---|---|---|
| `attention_request` | `batch * seqlen_q`, the padded upper bound | dispatch |
| `problem` | what was actually observed | measurement |

Under continuous batching, sequences in one launch have different query lengths,
so the observed total is strictly below the padded product. Planning on the
padded bound is right because that is what the kernel must cover; measuring on
it would overstate the work. Entries where the two differ are flagged `ragged`.

## Why the caller sends `num_seqs` at all

Tensor geometry alone does not determine which kernel rocKE selects. Holding
head geometry and query-row count fixed and varying only the batch
decomposition moves the selected path:

| `num_seqs` | selected |
|---|---|
| few | `attention_unified_3d` (split-KV) |
| many | `attention_unified_2d` (tiled) |

A tracing profiler sees tensor shapes, which do not record that split. That is
why the caller instruments the serving process to capture `num_seqs`,
`max_seqlen_q`, `max_seqlen_k`, and the paged `block_size`, and why a request
missing them is rejected rather than guessed at. `tests/serve/test_serve_planner.py`
pins the sensitivity, so if it ever stops holding, the capture requirement gets
re-examined instead of silently becoming dead weight.

## Modules

| module | needs | role |
|---|---|---|
| `protocol.py` | nothing | wire format; validates anywhere |
| `planner.py` | the library | runs the production dispatch registry |
| `runner.py` | torch + a GPU | verifies and times |

The split is what makes the degraded mode useful: with no GPU, planning still
answers whether rocKE serves a shape, which is what the caller needs before it
spends a node finding out.

## What the measured lanes claim

Correctness compares the planned fast path against rocKE's scalar attention
kernel. That catches tiling, geometry, and codegen faults in the fast path — the
errors that actually happen — but not a misreading of paged-KV semantics, which
both kernels share. The independent torch reference lives in the parity
harnesses under `builders/*/attention/`.

Speedup is reported against AITER's Triton `unified_attention` when it is
importable, because that is what the traced workload runs. With no baseline
present the lane reports `null`; it does not substitute something easier to
beat. `micro_speedup` and `correctness_passed` are `null` whenever their lane did
not run, so "not measured" never arrives looking like a measurement.

Where several shapes are measured, the reported speedup is total baseline time
over total rocKE time, weighted by observed call counts — not a mean of ratios,
which would let a rare shape outvote the one the workload lives in.

Per AGENTS.md, measured numbers stay in the run's output directory and never in
this repository.
