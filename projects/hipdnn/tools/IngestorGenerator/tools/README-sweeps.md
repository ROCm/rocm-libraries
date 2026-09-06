# Running the coverage, correctness and performance sweeps

Everything here needs a machine with the target GPU. Nothing here needs a particular
cluster, scheduler or site: if you can run `rocminfo` and see your arch, you can run
all of it.

The runbook (`hipdnn-ingestor-engine/RUNBOOK.md`, steps 8e and 9) says *when* to do
this and what the gates are. This page is the *how*.

> **Prefer the source projects' own documentation over this page.** It goes stale;
> theirs does not. For the graph corpora, the CLI and its setup: the
> `ROCm/dnn-benchmarking` repository and its `docs/`, plus each workload's
> `MANIFEST.md`. For the kernel library — what the dispatcher decides, how kernels are
> authored, what its own gates prove and where they have known gaps:
> `<provider>/rocke/library/dispatch/AGENTS.md`, `rocke/AGENTS.md`,
> `rocke/KERNEL_AUTHORING.md` and `rocke/TESTING.md`. This page covers only what an
> *ingestor integration* has to do with them, and the measurement discipline the
> harness encodes.

---

## What you need

| | |
|---|---|
| a machine with the target device | `rocminfo \| grep -m1 -E "Name:\s+gfx"` matches your `$ARCH` |
| the provider built and installed | `cmake --install $BUILD --prefix $INSTALL` |
| `dnn-benchmark` on `PATH` | ships with the provider build |
| graph corpora, staged | one directory per corpus name (below) |
| a writable path **the device machine can see** | not merely one your shell can see |

That last row is the one people get wrong. If the machine that runs is not the machine
you are typing on, verify the path from *there* before you queue anything expensive —
a job whose output path is invisible on the compute node dies before your payload runs,
and the failure does not look like a path problem.

---

## The three questions, and why they are not one

A sweep is one run and three results. Conflating them is how a variant set ships that
is fast on the shapes it covers and covers almost nothing.

| | question | how | fails when |
|---|---|---|---|
| **coverage** | which graphs do we serve, and is every decline defensible? | served/declined counts, reconciled against rocKE | rocKE serves something you decline |
| **correctness** | are the ones we serve *right*? | `--validate` against an independent reference | any mismatch, or an unwritten output |
| **performance** | how does the shipped package land? | timings, split by corpus | you cannot say which population a number describes |

---

## Corpora: two of them, measured separately

**Never merge these into one number.** A geomean over a mixed corpus reports one
population's result as everyone's.

1. **What real callers send.** The `ROCm/dnn-benchmarking` repository's workload
   graphs. Its own `docs/` explain the dvc pull; each workload carries a `MANIFEST.md`
   describing provenance. A `microbench/` path is a provenance label, **not** a
   synthetic-data warning — check the manifest before discarding a suite, because one
   dismissed on its directory name alone turned out to be entirely real.

2. **What the kernel owners judge themselves against.** `rocke/library/benchmarks/`
   in this repository — the per-arch benchmark scripts and, when you can get it, the
   results CSV they emit. The CSV is the better artifact: it is the shape list already
   resolved, and it carries a priority column that exists nowhere else. Ask for it
   before mining anything.

Stage each as its own directory of graph JSON. The directory names are not free:
`sweep.sh` reads `CORPORA` from the env config and gates each one on a matching
`EXPECT_GRAPHS_<name>`, so these must be the names the config declares. The shipped
`sweep-isolation.env.example` uses `published` and `servable`:

```bash
mkdir -p $CORPUS_DIR/servable $CORPUS_DIR/published
cp <dnn-benchmarking graphs>/*.json $CORPUS_DIR/servable/
cp <the owners' sweep graphs>/*.json $CORPUS_DIR/published/
```

---

## 1. Coverage

Mine a shape corpus, resolve it through the dispatcher, and reconcile every decline:

```bash
$GEN/.venv/bin/python $GEN/tools/mine_shapes.py \
    --published <results.csv> --graphs $CORPUS_DIR/servable \
    --arch $ARCH --out $SHAPES

$GEN/.venv/bin/python $GEN/tools/reconcile_applicability.py \
    --profile $GEN/configs/$SLUG.profile.yaml --shapes $SHAPES
```

**Gate: zero `ONLY THE REFERENCE` rows.** If rocKE serves a shape and you decline it,
that is missing coverage or a matcher bug — add the variant, fix the matcher, or show
that rocKE computes it *incorrectly* and report that as a rocKE defect. Choosing not
to serve it is not one of the three.

This part needs no GPU. Do it before you book device time.

---

## 2. Correctness

```bash
dnn-benchmark --graph "$CORPUS_DIR/servable/*.json" \
    --plugin-path $INSTALL/lib/hipdnn_plugins/engines \
    --validate pytorch --warmup 1 --iters 3 \
    -o /tmp/validate.json
```

`--validate` against an *independent* reference is the point: your matcher and
hipDNN's in-tree reference can share a misunderstanding and cancel it out. A third
implementation cannot participate in that.

**Gate: zero failures among served graphs.** Read `allClose=false` with **zero finite
mismatches** as *"an output element was never written"* — never as a tolerance
problem. Outputs are NaN-sentinel-filled precisely so an unwritten element cannot
pass, and the diff report counts only finite mismatches, so it prints `Mismatched: 0`
while failing.

Run correctness as a **separate pass** from timing. A reference execution per graph
distorts the very timings you are about to measure.

---

## 3. Performance

`tools/sweep.sh` runs the timed phases. It takes a config —
`configs/sweep-isolation.env.example` is the worked one, and
`configs/gfx950_attention_dense.sweep.env` is the gfx950 attention_dense one — rather
than being forked per comparison:

```bash
SWEEP_CONFIG=my-sweep.env $GEN/tools/sweep.sh
```

**`EXCLUDE_TENSORS` must name the exact backward-gradient tensor set
`tools/mine_shapes.py` filters on** (`BACKWARD_GRADIENT_TENSOR_NAMES`), never a
hand-copied subset — `tests/test_sweep_configs.py` checks every committed
`configs/*.sweep.env` and `configs/*.env.example` against that one source of truth,
so a value that drifts from it fails the test suite rather than silently gating
nothing on a real backward graph.

Read its header before changing anything. Every structural choice exists because
**clocks usually cannot be pinned on a shared machine**, so the harness controls for
drift instead of pretending it is absent:

- **one machine, one session, one job** — cross-machine absolute numbers are meaningless;
- **a warmup pass, discarded**, so phase 1 does not measure a cold device;
- **several rounds**, so round 1 versus round N *measures* drift rather than assuming none;
- **fixed arm order, never rotated.** Drift penalises whichever arm runs later, so an
  effect moving the *opposite* way to the confound cannot be explained by position. Put
  the baseline first. This sign check is worth more than rotation would buy;
- **a known-identical control.** Include shapes whose arms can only pick a byte-identical
  binary: they must read exactly 1.000x, and whatever they actually read is your noise
  floor. Build that set from descriptor `sha256`, **never** from timing — selecting
  graphs *because* they timed alike is circular.

### Reporting

Report **geomean-of-ratios and time-weighted (sum baseline / sum arm) side by side**,
and split by corpus provenance. Those two statistics can differ by more than an order
of magnitude on the same data when the shapes driving the geomean hold a small share of
total time. Both are true; publishing only the flattering one implies wall-clock savings
the data does not support.

---

## Sizing: how long this actually takes

Two costs dominate, and neither is the kernel:

- **The reference executor.** The shared references are untiled — roughly one thread per
  output element, looping the full contraction. Cost scales with the whole problem, so a
  production-sized shape can take orders of magnitude longer to *verify* than to *run*.
  Size this before booking device time; it is the usual reason a "quick" run is not.
- **Packing.** Compile cost scales with the shape you compile, not just the variant
  count, and the packer may not saturate the machine. Time one pack before assuming a
  large set is affordable.

---

## A note on what you can publish

Keep measured numbers out of the public repository — commit messages, code comments and
docs alike. State the *method* and the *shape* of a result ("the pinned arm measured
worse on the shapes where the policy would have chosen otherwise"), not the digits.
The lesson transfers; the number does not, because it belongs to one machine on one day.
