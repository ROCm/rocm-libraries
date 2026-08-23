# gfx1100 HHS-TN: v6 Prediction catalog vs current develop GridBased

> **Reconstructed 2026-08-22 from the session transcript.** The original lived in the plan-mode
> scratch slot `~/.claude/plans/modular-spinning-sunbeam.md`, which is reused per session and
> was later overwritten by an unrelated StreamK campaign plan. Text below is the approved
> plan as executed. Results: [`REPORT.md`](REPORT.md).

## Context

Compare the **v6 branch** (`users/vmijovic/gfx1100-hhs-tn-v6-stock-ship`) against **current
develop** on fp16 HHS TN, using the AIHPBLAS-4345 sweep kit
(`~/aihpblas-4345-rdna-bf16tn-sweeps-20260818.zip`) converted from bf16 to fp16.

This plan borrows heavily from the gfx1201 A/B plan — the two experiments are the same
shape, and several of its findings transfer directly (noted inline).

### Structural facts established during planning

1. **`~/TheRock/rocm-libraries` is the working repo**, and `~/exp/{shipv6,stock,tuned}` are
   already **worktrees of it**. It carries both `develop` and the v6 branch, plus
   `projects/hipblaslt/tasks.py` (so `invoke build --logic-filter` is available; `invoke` is
   on PATH). `~/rocm-libraries` is a separate, **older** standalone checkout with no
   `tasks.py` — **ignore it**.
2. **Build both arms fresh, identically.** Reusing the existing `~/exp/shipv6` build would
   confound the catalog difference with build configuration (it was built unfiltered, at a
   different time). Worktrees also materialise *committed* state, which automatically
   excludes the uncommitted `ORIGAMI_*` experiment patches sitting in `~/exp/stock`.
3. **VRAM is the binding constraint, and it is tighter here than on gfx1201.** This card has
   **25.75 GB**; the suite has **16 problems over 18 GB**, topping out at ~29 GB
   (`#198 80×3616×16 b=23009`, `#189`, `#185`, `#39`, `#40`, `#190`). Several **will OOM**.
   Plus ~514 MB of rotating buffers.
4. **Conversion is dtype-only.** `bf16_r` appears in the sweep YAMLs only as
   `a/b/c/d_type`; `compute_type: c_f32_r` and `scale_type: f32_r` do not contain the
   substring, so a global substitution is safe. `c_f32_r` is valid (hipBLASLt's own test
   data uses it) and **must stay** — it is what the S in HHS means.
5. **`ANALYTICAL_GEMM_PREDICTION_MODE` is dead** — absent from the codebase. Origami's live
   knobs are `ANALYTICAL_GEMM_DEBUG`, `ANALYTICAL_GEMM_HEURISTICS`. Harmless to leave set.
6. **`TENSILE_PREDICTION_LIB` is load-bearing and must be identical on both arms.** v6's
   HHS-TN logic is `Prediction` type; develop's is `GridBased`. Per the gfx1201 finding,
   `ExactLogicLibrary` skips only Equality/Range rows when the prediction library is active —
   **GridBased rows are not skipped** — so if an arm ends up with both a GridBased and a
   Prediction file for the same ProblemType, arm assignment depends on glob order.

---

## Steps

### 1 — Worktrees (pristine, identical treatment)

```bash
R=/home/vmijovic/TheRock/rocm-libraries
git -C $R fetch origin
mkdir -p ~/ab1100/{yaml,parsers,logs,results}
git -C $R worktree add --detach $R/.claude/worktrees/ab-dev    origin/develop
git -C $R worktree add --detach $R/.claude/worktrees/ab-v6     users/vmijovic/gfx1100-hhs-tn-v6-stock-ship
```

**Gate:** in each worktree, list `.../Logic/asm_full/navi31/*/navi31_Cijk_Alik_Bljk_HHS_*`
and record file → LibraryType (element `[11]`) → solution count. Confirm the **v6 arm has
exactly one** file for the `HA_S_SAV` ProblemType and it is `Prediction`; if a GridBased
file for the same ProblemType is also present, `git rm` it on a branch (fact 6).

### 2 — Build both arms (filtered)

```bash
cd <worktree>/projects/hipblaslt
invoke build --architecture gfx1100 --jobs $(nproc) --tensile-threads $(nproc) \
  --clients --skip-rocroller \
  --logic-filter 'navi31/*/navi31_Cijk_Alik_Bljk_HHS_*' \
  2>&1 | tee ~/ab1100/logs/build_<arm>.log
```

The filter cuts navi31 from ~150 logic files to a handful — this is what keeps the build
near an hour. **The filter string must be byte-identical on both arms**: it defines the
candidate pool and therefore the oracle. Build **develop first**, resolve any toolchain
issue, then v6. Never build while a benchmark runs.

**Verify:** `sha256sum <arm>/build/release/Tensile/library/gfx1100/TensileLibrary_*.dat*`
differs between arms; both `clients/hipblaslt-bench` exist. Benchmarks read
`build/release/`, never any `*-install/` tree.

**Fallback if the build fails:** use the existing `~/exp/shipv6` build for v6 and
`~/exp/devlib_stock_grid` (298-sol GridBased, stock binary) as a clearly-labelled *proxy*
for develop — and state plainly that it is not develop.

### 3 — Convert the harness to fp16

Extract the kit to `~/ab1100/harness/`. Convert the three sweep YAMLs:

```bash
for f in bf16tn bf16tn_adaptive_heuristic bf16tn_allalgos_adaptive_method1; do
  sed 's/bf16_r/f16_r/g' harness/yaml/$f.yaml > yaml/${f/bf16tn/f16tn}.yaml
done
diff <(sed 's/f16_r/bf16_r/g' yaml/f16tn.yaml) harness/yaml/bf16tn.yaml   # must be empty
```

`grid_sweep_adaptive.sh` / `wgm_sweep_adaptive.sh` hardcode the dtype block. **Do not
`sed -i` them** — add a defaults block after `source setup_env.sh` (`DTYPE=${DTYPE:-bf16_r}`
etc.) and substitute the literals, so bf16 reproducibility survives. Always pass the output
CSV explicitly; both scripts default to `results/bf16tn_*.csv` and would mislabel fp16 output.

**Patch the kit's step-5 "force DP":** it uses `TENSILE_STREAMK_DATA_PARALLEL=1`, whose store
is immediately overwritten by the `skDynamicGrid > 0` branch — verified on this tree during
the prior campaign. Use `TENSILE_STREAMK_DYNAMIC_GRID=4` (`data_parallel`) and note the
deviation in the results README.

**Megagrid (step 7) excluded:** 69 MB of bf16 problems, multi-day, low value here. Say so in
the report so it does not read as an oversight.

### 4 — Memory pre-flight (before anything long)

Run the 16 problems over 18 GB **alone**, both arms. Whatever OOMs is dropped **symmetrically
from both arms** and recorded. Consider lowering `--rotating` for that subset instead of
dropping. This is the single most likely cause of a wasted overnight run on a 24 GB card.

### 5 — Routing gate (hard gate, ~20 min)

Build `solution_allowlists.json` from each arm's logic (`SolutionNameMin` sets), then over a
20-problem subset assert:

1. **Partition** — each arm's observed `--Solution name:` ⊆ its own allow-list, and the two
   observed sets are **disjoint**. ~100% identical kernels means one arm did not load what we
   think.
2. **Origami actually ran** on v6 — `ANALYTICAL_GEMM_DEBUG=1` produces Origami output. Direct
   evidence, not inference from names.
3. **Negative control** — v6 with `TENSILE_PREDICTION_LIB=0` yields *different* kernels. If
   identical, fact 6 is wrong for this build and every routing conclusion is void.
4. **SK-mode mix** per arm from `_SK<n>_`, and the observed `skGrid` via `TENSILE_DB=0x40`
   (reads the launch grid with no statistical uncertainty).

**Do not proceed if the observed solution sets overlap.**

### 6 — Rank-1 headline sweep, ABBA (~3–6 h)

`run_bench_yaml.sh yaml/f16tn_adaptive_heuristic.yaml` in order **A₁, B₁, B₂, A₂**. The
repeats are the point: `|A₁/A₂ − 1|` at p95 is the in-session noise floor, and the tie
deadband is derived from it (2× p95) rather than assumed. (Prior campaign measured ~0.5% on
this rig; this re-measures it under the actual protocol.)

### 7 — Cross-arm comparison (the headline)

**Metric trap, straight from the gfx1201 plan and it applies here:**
`selection_efficiency = heuristic ÷ own-oracle-best` is computed against *each arm's own*
candidate pool. v6 enumerates ~58 kernels, develop ~73 — **v6's efficiency is inflated by
construction**. Efficiency is valid **within** an arm only and must never appear in a
cross-arm table without that caveat inline.

The cross-arm metric is **absolute rank-1 GFlop/s** per `(transA, transB, batch, M, N, K)`,
straight from the two rank-1 logs, no oracle dependency. New
`~/ab1100/parsers/ab_rank1_compare.py`, kept outside the vendored harness; reuse
`parse_bench_log` / `problem_key` from `harness/parsers/bf16tn_extract_db_csvs.py` so parsing
is bit-identical.

Report: **geomean** speedup (never the arithmetic mean of ratios), median/p10/p90,
throughput-weighted `Σa_us/Σb_us`, bucket counts
(`severe <0.75 | regression <0.90 | loss | tie ±d | win | strong >1.10`), `kernel_change_rate`,
per-`label` breakdown (the 20 Guillermo OOB categories come from the YAML), and worst-20 /
best-20 with both solution names side by side. **Also band by kernel duration** — a single
geomean hides reversals.

### 8 — Per-arm oracle and triage (unattended, overnight)

Per arm into `results/<arm>/`: `bf16tn_extract_db_csvs.py` → `bf16tn_heuristic_efficiency_parse.py`
→ `pick_worst_performers.py --top 15` → `DTYPE=f16_r grid_sweep_adaptive.sh` →
`DTYPE=f16_r wgm_sweep_adaptive.sh`. Run **v6's oracle first** (smaller pool, fast sanity
check), then develop's. Worst-15 sets will differ between arms — expected, this is within-arm
triage.

### 9 — Standing constraints

- **One GPU job at a time:** `flock -w 86400 ~/ab1100/.gpu.lock <cmd>`.
- **Stock clocks only.** Capture `rocm-smi --showclocks --showtemp --showpower` before/after
  every run.
- Pin `HIP_VISIBLE_DEVICES=0` and `ROCR_VISIBLE_DEVICES=0`; start each log with
  `capture_rocminfo.sh`.
- Per-run JSON manifest sidecar: worktree SHA, `HIPBLASLT_BUILD`,
  `HIPBLASLT_TENSILE_LIBPATH`, `TENSILE_PREDICTION_LIB`, all `ANALYTICAL_GEMM_*`, rocm-smi
  before/after, and `sha256sum` of the arm's `TensileLibrary_*.dat`.
- **Clean restart over append-and-resume** for the A/B logs — the parser keeps the longest
  converged suffix per key, so a partial re-run can silently outrank the original.
- Run under `tmux` / `setsid nohup`; a driver script records step completion in a JSON ledger
  and skips completed steps on restart.

---

## Verification checklist

1. Both worktrees clean; the only difference between arms is the HHS-TN logic file.
2. `--logic-filter` identical in both build logs; same file count resolved.
3. Library `.dat` sha256 differs between arms.
4. `HIPBLASLT_TENSILE_LIBPATH` in each manifest points into the *correct* worktree — the
   easiest thing to get wrong, since both arms have identically-named subpaths.
5. Solution-name partition holds across all 200 problems; sets disjoint.
6. `kernel_change_rate` non-trivial.
7. `ANALYTICAL_GEMM_DEBUG=1` shows Origami ran on v6; `TENSILE_PREDICTION_LIB=0` control
   yields different kernels.
8. A₁-vs-A₂ p95 spread reported and small relative to the claimed effect.
9. Dropped OOM problems identical across arms.

## Deliverables

`~/ab1100/results/`: `ab_rank1_compare.{csv,md}` (**the headline**), per-arm
`{heuristic,oracle,heuristic_efficiency}.csv/md`, `{worst_performers,grid_sweep,wgm_sweep}.csv`,
`routing_audit.md`, `rocminfo/`, and `REPORT.md`.

`REPORT.md` order: noise floor and derived deadband → geomean + throughput-weighted speedup
with the bucket table → per-label and per-duration-band breakdown → worst 20 regressions with
both solution names → per-arm `selection_efficiency` **with the non-comparability caveat
inline** → grid/WGM findings → verification checklist ticked.

## Most likely failure modes

1. **OOM on the 16 large problems** on this 24 GB card → step 4.
2. Both arms identical because a GridBased file shadowed the Prediction one, or
   `TENSILE_PREDICTION_LIB` drifted → checks 5, 6, 7.
3. Both arms pointed at the same `HIPBLASLT_TENSILE_LIBPATH` → check 4.
4. `selection_efficiency` leaking into a cross-arm table → a confident wrong conclusion.
5. Build exceeds budget → fallback in step 2, clearly labelled.
