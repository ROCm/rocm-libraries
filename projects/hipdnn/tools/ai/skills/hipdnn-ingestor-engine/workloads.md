<!-- skill-paths: external-repo ROCm/dnn-benchmarking -->

# Real workloads — deciding what to compile, and proving it runs

**You are sent here from RUNBOOK step 2a (which shapes exist), step 3 (which to ship) and
step 8e (proving it).** You owe two things before generating: the list of real shapes your
kernel can serve, and a *count* of how many your proposed variant set covers.

The tool is **`ROCm/dnn-benchmarking`** — a separate repository, not part of this tree.
It is the project's inventory of what callers actually run, and it is the only source
here that answers *"will anyone use this kernel?"*

> Prefer its own docs over this page: `README.md` for the CLI and setup,
> `docs/troubleshooting.md`, and each workload's `MANIFEST.md` for provenance. This file
> covers only what an ingestor integration needs and the traps that cost real time.

---

## Why this is not optional

An AOT variant set is a **compile-time commitment**. A kernel that bakes its extents
serves the shapes you compiled and declines everything else, so "capable but no variant"
and "unsupported" are the same thing to a caller.

Everything else the runbook points you at for sizing — the dispatcher, the spec's knob
comments, the `supports_*` predicate — is **kernel-side**. It tells you what the kernel
*can* be built for. None of it tells you what anyone will *ask* for. That gap is not
theoretical:

| Attempt | How the set was sized | Real graphs served |
|---|---|---|
| 1 | kernel-side sources only | **0 / 38** |
| 2 | model traces, `microbench/` assumed synthetic | **33 / 118** |
| 3 | every servable graph enumerated and counted | **118 / 118** |

Both failures passed every mechanical gate in this runbook — legal specs, clean
descriptors, green desk check, green validator, passing tests. The set was wrong and
nothing in the integration could see it.

---

## The corpus

```bash
git clone https://github.com/ROCm/dnn-benchmarking && cd dnn-benchmarking
ls Workloads/models/       # real model traces
ls Workloads/microbench/   # per-library shape collections
dvc pull Workloads/models/*.dvc Workloads/microbench/*.dvc
```

Graphs are the **same JSON your matcher already walks** — same node types, same tensor
`dims`/`strides` — so they load with the reader you used at step 2a, and your own bundles
run through `dnn-benchmark` unchanged.

**`microbench/` does NOT mean synthetic.** It is a provenance label: these suites are
rendered from real shapes found in a library's source. The `aiter` MANIFEST is explicit —
*"Every JSON here was rendered by the committed hipDNN emitters from a real shape found in
the aiter source. No shapes were invented."* AITER, hipBLASLt and the rest serve real
customers. Discarding a `microbench/` suite on the assumption that it is a synthetic sweep
is exactly how attempt 2 above missed 72 shapes it could have served. **Read the MANIFEST
before you exclude anything.**

---

## Step 2a — enumerate what exists

Extract every graph for your op and classify it with **the same predicate your matcher
uses**. A triage that checks two or three attributes will call a graph servable that your
`graph_match` declines on the fourth, and the error is always optimistic.

```python
# For each graph: parse, apply your full Tier-3 decline list, then ask the kernel.
ok, why = supports_<op>(Spec(**derived_fields))
```

Record three buckets, and put them in `graph_contract.md`:

1. **Servable** — your kernel can build for it. This is your candidate shape list.
2. **Declined** — outside scope. Each should map to a named row of your step-2b rejection
   checklist. *A decline you cannot name is a bug in your understanding, not a scope call.*
3. **Declined but shippable** — the kernel could build it and you simply have no variant.
   **This bucket is invisible from inside your own bundles** and is the one that matters.

Also read what the corpus tells you that the in-tree bundles cannot:

- **Layout.** Real traces and in-tree test bundles can disagree on stride order for the
  same logical dims. Whichever you read first silently decides what your matcher enforces.
  *(In one run the split was total: every one of 3654 real graphs used one order, every
  in-tree bundle the other.)*
- **Feature spelling.** Where a schema carries both a modern field set and the deprecated
  one it replaced, real traces and authored bundles routinely pick opposite spellings. A
  matcher handling only its own bundles' convention passes CI and mis-serves production.
- **Shape magnitude.** Authored tests are small on purpose; traces are not. Here the gap
  was two orders of magnitude.

---

## Step 3 — count coverage before you generate

**The gate is a number, not a judgement.** For every servable graph, does a proposed
variant match it? Report `covered / servable`.

```
servable real graphs : 118
covered by your set  : 118   (100%)
```

A set that covers a small fraction is a test matrix, not a shipping set. Widen it, or
scope the remainder out **deliberately and in writing** — scoping out is legitimate,
never having looked is not.

Two things this count will not tell you, so decide them explicitly:

- **Deliberately synthetic variants are sometimes correct.** If every real shape happens
  to satisfy a knob's constraint, that knob's other values are unreachable by any real
  graph and `score` never chooses them. Shipping a synthetic shape that *forces* the other
  value is how you keep the tuning axis alive. Say why it exists.
- **Compile cost scales with shape, not variant count.** In this run 56 toy kernels packed
  in 17 s; 65 real-shape kernels took 233 s — same order of kernels, **~8× per kernel**.
  The emitted IR was byte-identical in size; the cost was in the backend, because the
  kernel bakes its loop trip counts and LLVM sees 4 iterations at a toy length and 512 at
  a production one. Budget by shape, not by count.

---

## Step 8e — run it

Required, and **never wired into CI** — that repo's README says not to use it in build or
CI pipelines. Run it by hand, carry the findings into step 9.

### Setting up against YOUR engine

The trap that costs the most time: `setup_env.py` builds hipDNN, the providers and the
Python bindings **from its `rocm-libraries` submodule**, which tracks `develop`. Left
alone, it benchmarks an engine that does not include your work.

**Point the submodule at your branch and let it build one coherent stack:**

```bash
git -C rocm-libraries fetch --depth 1 origin <your-branch>
git -C rocm-libraries checkout FETCH_HEAD
python3 setup_env.py --workspace .workspace --torch-mode rocm --gpu-arch <arch> \
  --torch-index-url https://rocm.nightlies.amd.com/whl-multi-arch/ -y
source .workspace/.venv/bin/activate
```

- The `--torch-index-url` is TheRock's nightly channel, and it is in that repo's own
  Dockerfile. **PyTorch must match the container's ROCm.** A mismatch fails at
  `import hipdnn_frontend` *after* `import torch` with an undefined HSA symbol — a
  confusing error, because the bindings import fine on their own.
- Do **not** hand-build the bindings wheel and combine it with `--reuse-artifacts`. That
  mixes two independently-built stacks and fails with an undefined `libhipdnn_backend`
  symbol. The submodule route exists to prevent exactly this.
- If you must reuse an existing install, set `ROCM_PATH` to it and pass
  `--plugin-path $ROCM_PATH/lib/hipdnn_plugins/engines` — otherwise your engine is simply
  absent and every graph reports "no engines applicable".

### What to run

```bash
# your own bundles, against an INDEPENDENT reference
dnn-benchmark --graph '<your bundles>/*.json' --validate pytorch -v

# the corpus your engine claims to serve
dnn-benchmark --graph 'Workloads/models/<model>.tar.gz' --validate pytorch -o model.json
```

`--validate pytorch` is the point: stage 8's in-tree reference and your matcher can share
a misunderstanding, and PyTorch cannot participate in it.

### Performance, including against PyTorch

`--engine <id>` selects by **numeric id**, which is how you attribute a run to your engine
while engine-name exposure is still pending.

For a best-case comparison, benchmark twice — once to find the best variant, once to use
it:

```bash
# 1. oracle sweep: times every applicable candidate and persists the winner
HIPDNN_FORCE_BENCHMARKING=1 dnn-benchmark --graph "$G" --plugin-path $P --iters 10 -o /dev/null

# 2. your best, record reused
dnn-benchmark --graph "$G" --plugin-path $P --warmup 10 --iters 50 -o mine.json

# 3. PyTorch, both a named backend and normal dispatch
# a named backend/library, where the op has selectors (see that repo's README)
dnn-benchmark --graph "$G" --backend pytorch [--pytorch-<op>-backend <sel>] \
  [--pytorch-rocm-fa-library <lib>] -o named_backend.json
dnn-benchmark --graph "$G" --backend pytorch -o pytorch_default.json
```

Compare against **default dispatch** as well as a named library: default is what a real
caller gets. Also check the kernel's own docstring claims — rocKE modules frequently
record measured results, and your `score` encodes some of them as a ranking. Verify the
ranking against a measurement instead of shipping the assertion; a `score` that ranks
backwards passes every correctness test.

---

## Two limits that bound stage 8 itself

**The in-tree reference cannot verify production shapes.** `gpu-ref` is one thread per
output element looping over every KV position — untiled, O(problem size). Measured here:
milliseconds at a toy length, ~20 minutes at 8192, **~8 hours at 32768**, against tier
budgets of 600 s and 1800 s. Verify correctness at shapes the reference can evaluate; run
the production shapes through `dnn-benchmark`, which validates against PyTorch. Do not
author a bundle you cannot afford to verify.

**Detach long device jobs properly.** `nohup … &` dies with the session and leaves an
empty log that reads like a job still running. Use `setsid nohup … < /dev/null &`.

**And distinguish infrastructure from defects.** A job that fails with
`No space left on device` or `pyxis: failed to create container filesystem` never started
your payload. Re-run excluding that node before debugging your engine.

---

## GATE

- `graph_contract.md` carries the three buckets, with counts.
- The step-3 batch states `covered / servable` and names anything scoped out.
- Step 8e has run: your bundles under `--validate pytorch`, the corpus triaged, and any
  *declined-but-shippable* rows either added to the set or listed at step 9 with a reason.
