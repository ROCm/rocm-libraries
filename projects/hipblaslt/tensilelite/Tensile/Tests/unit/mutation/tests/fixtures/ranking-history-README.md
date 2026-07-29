# Ranking history + refresh process

Versioned snapshots of the Phase-0-global ranking/disposition (Issue 17), plus how
the ranking feeds the CI allowlist ratchet. Scaffolding/dry-run: refreshing runs no
mutation testing and edits no production source.

## What lives here

`<date>-<pin>.md` snapshots produced by `rank-refresh.sh`, which re-runs
`rank-modules.py` and stores its output. Each snapshot is the full disposition
table + formula + metric-availability at a given source pin.

## Refresh (dry/read-only)

```bash
bash projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/rank-refresh.sh \
  [--out-dir <dir>] [--pin <sha>] [--metrics <json>]
```
- snapshots to `ranking-history/<date>-<pin>.md`;
- diffs the refresh against the most recent prior snapshot;
- deterministic: `rank-modules.py` output has no embedded timestamp, so for a fixed
  working tree (and same `--metrics`) it reproduces byte-identical content and the
  wrapper reports "no change" / "no diff". NOTE: `rank-modules.py` analyzes the LIVE
  working tree under `projects/hipblaslt/tensilelite`; it does not check out the
  pinned sha. `--pin` is only a filename label for the snapshot - keep it in sync
  with the tree's actual base.

`--metrics <json>` is forwarded to `rank-modules.py` to fill the PENDING
`cyclomatic` / `no_test_fraction` inputs and compute scores; without it, scores stay
PENDING (the wrapper never invents measured scores).

## Formula is preserved verbatim (never re-derived here)

The wrapper only invokes `rank-modules.py`, which enforces the Issue-17 formula:

```
score = 0.40*importers_norm + 0.25*cyclomatic_norm
      + 0.15*LOC_norm       + 0.20*no_test_fraction_norm
```

weights `0.40/0.25/0.15/0.20`, all inputs min-max normalized to [0,1], **no log10**
(LOC enters linearly), **no subtraction** of `no_test_fraction`. `rank-refresh.sh`
does no scoring of its own, so it cannot drift from these.

## When to refresh

- **pin changes** - the campaign re-pins the base per slice; a new base changes LOC
  and the mutant population, so re-snapshot at the new pin.
- **slices complete** - a certified slice moves a dir to `certified` and its files
  into the allowlist; refresh so the disposition/ordering reflects it.
- **metrics become available** - once `cyclomatic` (`lizard`) and `no_test_fraction`
  (`pytest --cov` in `tl-mut`) are measured, re-run with `--metrics` to replace the
  interim hand-ranked order with the computed order.
- **broadening beyond the current tranche** - when scheduling the next tranche of
  `deferred-high-value` modules, refresh to rank the newly-in-scope candidates.

## How the ranking feeds the CI ratchet / allowlist ordering

The ranking sets the ORDER in which modules enter the `only_mutate` allowlist; the
CI ladder (`../ci/mutation-ci-ladder.md`) then gates the allowlisted set:

1. `disposition.md` classifies every char dir (certified / scheduled /
   deferred-high-value / deferred-coverage-gap / out-of-scope).
2. Highest-scored `scheduled` modules are certified next and their files are
   appended to the `only_mutate` allowlist (rung 4, allowlist ratchet). CI only ever
   gates files in the allowlist - never a repo-wide global threshold. NOTE: until
   `cyclomatic`/`no_test_fraction` are measured (via `--metrics`), scores are PENDING
   and this order is the plan's INTERIM hand-ranking, not a computed order.
3. Each newly-certified slice's killed-set is added to the CI baseline
   (`../ci/killed-set.example.json` shape), and the fail-on-regression rung guards
   it by stable mutant identity. (Certification is tracked at file granularity -
   a multi-module char dir may be only partially certified; see disposition.md.)
4. `SLICE_FLOOR` stays **UNSET** until >=2 more slices give a stable
   actionable-survivor rate (per the plan); refreshing the ranking does NOT set a
   floor. Ordering/allowlist growth is what the ranking drives, not a floor.

So: ranking -> allowlist order -> per-slice certification -> CI baseline growth,
with no premature `SLICE_FLOOR` and no global threshold.
