# Runbook — comparing StreamK modes on a new SKU

How to repeat the gfx1100 StreamK campaign somewhere else. Written from the gfx1100 run;
the traps listed are ones that were actually hit, not hypothetical.

Findings live in `REPORT.md`. This file is method only.

---

## 0. What this measures, and what it cannot

**Every comparison is internal** — one harness, one ProblemType, one session. That is what
makes SK3-vs-SK4, CLR, DepthU and prefetch contrasts trustworthy.

It does **not** support statements about the shipped library's absolute performance. That
needs a different experiment (the `~/ab1100`-style A/B: same tool, same ProblemType
including bias/scaleAlphaVec, library's own selector). Do not compare a Tensile-client
number against a `hipblaslt-bench` number.

---

## 1. Environment (30 min, and two traps)

```bash
cd <tree>/projects/hipblaslt/tensilelite
export PYTHONPATH=<tree>/build/release/tensilelite/rocisa:<tree>/projects/hipblaslt/tensilelite
invoke build-client --gpu-targets <arch> --no-rebuild-rocisa
```

**Trap 1 — `PYTHONPATH` is mandatory.** A stale `rocisa-dev.pth` in
`~/.local/lib/pythonX/site-packages` can point at an unbuilt rocisa tree, so `import rocisa`
fails with a confusing circular-import error. Point `PYTHONPATH` at a tree whose
`_rocisa*.so` actually exists.

**Trap 2 — `--no-rebuild-rocisa` is deliberate.** The task defaults it to `True`, which
re-installs the editable rocisa and disturbs exactly the `.pth` situation you just routed
around. An unattended run should not mutate its own interpreter setup.

**Gate:** before anything else, build *and benchmark* one known-good SK3 config end to end.
Kernel generation succeeding is not enough — the client is a separate stage and can be
missing entirely.

---

## 2. Shape selection — the step that decides whether you measure anything

StreamK is **inert** on most shapes: it is only doing something when

```
skTiles != skGrid   AND   itersPerTile > 1
```

Both conjuncts matter. The second is not optional — a tile with one k-iteration cannot be
split, and omitting that check silently admits shapes where nothing streams.

On gfx1100 only **337 of 1500** census shapes (22.5%) qualified. Pick shapes from a census
that reads `skGrid` / `skTiles` / `itersPerTile` **out of the kernel arguments** with
`TENSILE_DB=0x40`, not from a model.

Stratify by kernel duration and **report per band** — a single geomean reverses sign against
the banded view on this workload.

---

## 3. Timing protocol — measure the floor before believing anything

The first protocol used here (`NumWarmups 4`, `EnqueuesPerSync 4`, `SyncsPerBenchmark 2` =
**8 timed enqueues**) had a **541% p95 noise floor** in the `<0.1 ms` band. It would have
produced a confident 12% "win" that does not exist.

Use adaptive iteration counts:

```yaml
MinFlopsPerSync: 2000000000      # scales enqueues per problem: 66 kflop -> ~30k, 1.3 Tflop -> 1
EnqueuesPerSync: 20              # floor
MaxEnqueuesPerSync: 40000        # cap
SyncsPerBenchmark: 5
NumWarmups: 10
SleepPercent: 100
NumElementsToValidate: -1        # NOT optional, see §4
```

That took the floor to 19.1% / 19.4% / **1.8%** across the three bands — 28× tighter in the
worst one.

**Then measure the floor for real:** run one arm **twice** and compute the p95 of
`|ratio − 1|` per band. Bootstrap CIs resample shapes and **cannot see** this; only an A/A
repeat can. Nothing smaller than that floor is a result.

---

## 4. Two disciplines that caught real errors

**Validation is not optional.** `NumElementsToValidate: -1`, and report validation status
alongside every number. A mode that assembles and computes garbage looks *identical* in a
throughput table. Arms failing validation are reported as failures, never silently dropped.

**Include a negative control.** An arm you expect to do nothing. Here it was
`StreamKXCCMapping` (no XCDs on a monolithic part). It is what turns "we saw no effect" into
"the harness can see a 3% effect and this knob shows 0.5%". Without it a null is
unfalsifiable — and in this campaign the control *fired*, which is how the first bad
protocol was caught.

**Prefer an inert PARTITION to an inert ARM — and use both.** Across the StreamK and gate
campaigns every error was in the optimistic direction, and every one was caught by something
built to be inert:

| control | what it caught |
|---|---|
| A/A arm (same config, run twice) | a 1% "side effect" on shapes the code cannot touch — actually small-sample noise |
| below-threshold partition | that a *global* A/A correction over-corrects: the systematic was ~0.2% on one partition, ~0.8% on the other |
| **common partition** (shapes where every arm is *definitionally identical*) | **0.28 pt of arm-POSITION drift that an A/A arm cannot see** |

The third is the one worth stealing. When comparing N variants of a threshold or knob there
is usually a region where they all behave identically *by construction*. Score it. Any
spread there is measurement artefact — and if that spread is **monotone in the arms'
interleave order**, it is position drift: later-measured arms run warmer and score higher.
An A/A arm is structurally blind to this, because it occupies one fixed slot while the
variants occupy others.

The same partition is then the calibration: divide each arm by its own value there. Here
that collapsed the cross-threshold spread from 0.34 pt to 0.06 pt and changed which
threshold looked best. **Automate the check** — `plateau_analyze.py` prints
`MONOTONE IN ARM ORDER` — because by eye a monotone sequence reads as signal, not as drift.

---

## 5. Sweep order — cheapest-and-most-certain first

Run the axis that needs **no source patch** before any patching work. If the patched arms
fail you still have a result.

Axes that mattered here, and **three of them inverted the SK4/SK3 ratio**:

| axis | effect |
|---|---|
| `ClusterLocalRead` | inverts the ratio; its own preference then inverts with DepthU |
| `DepthU` | SK4/SK3 ran 155% → 115% → 93% across DU 16/32/64 |
| `PrefetchGlobalRead` / `PrefetchLocalRead` | inverts it; **check the catalog's distribution first** |
| duration band | 43% at `<0.1 ms` vs 112% at `>=1 ms` |

**Check your recipe against the shipped catalog before trusting a sweep.** The gfx1100 run
used `PGR0/PLR0` — which is **8%** of shipped solutions. Everything had to be re-measured.

```python
# per-parameter distribution across a shipped logic file
sols = yaml.safe_load(open(logic_yaml))[5]
Counter(s.get("PrefetchGlobalRead") for s in sols)
```

---

## 6. Ratio vs absolute — the trap that nearly produced the wrong recommendation

**A ratio computed *within* a configuration can invert the ranking *across* configurations.**

SK4 beat SK3 by up to 155%. Best SK4 anywhere was still **95.0% of best SK3** — the wins were
against a baseline that other settings had already weakened, on geometries 10–30% slower to
begin with.

Always rank every configuration **absolutely** on the same shapes before recommending
anything. A tuner optimising the per-recipe ratio would have picked the slower mode.

---

## 7. Operational notes

- **`setsid --fork`** for long-lived background services. Plain `setsid ... &` leaves the
  process in a group that gets killed when the launching command exits.
- **Verify a service by its listener** (`ss -lntp | grep <port>`), not by a `curl` that
  immediately follows the launch in the same block — the sleep masks a failed start.
- **`pgrep -f <pattern>` matches its own command line.** A check for `pgrep -f my_script.py`
  run inside a shell wrapper reports the job alive forever, because the pattern appears in
  the wrapper's own argv. Bracket the first character (`pgrep -f '[m]y_script\.py'`) and
  corroborate with something the check cannot itself produce — `flock -n` on the job's lock
  is ideal, since a free lock proves the holder exited. Same failure mode as the curl trap:
  a liveness check must not be able to observe itself.
- **Shell cwd persists between commands.** `cd` in one block silently changes where the next
  runs. Use absolute paths or re-`cd` explicitly.
- **`flock`** every GPU action; never benchmark while building.
- Record a `git diff` revert point before touching a vendored tree, and check the final diff
  against it.
- **Patching a cited source file shifts line numbers.** If a deck or doc cites `file:line`,
  re-verify after patching — and fix against the pre-patch backup, since another line may
  contain the same token and produce a citation that passes but points at the wrong code.

---

## 8. Analysis

`analyze.py` in this directory. Two notes:

- Join validation status to throughput by **matching against the known solution list**. Do
  not regex the solution name out of a client log line: every row begins
  `Contraction_l_Alik_Bljk_Cijk_Dijk`, which contains `Cijk_Dijk`, so a naive `Cijk_\w+`
  match collapses every record onto one fake solution (192 → 24 here).
- Band by a **fixed** duration source (the census), not by the run being analysed, so
  banding cannot drift with the thing being measured.
