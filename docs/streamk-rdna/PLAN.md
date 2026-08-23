# 10-hour UNATTENDED StreamK campaign — TN HHS priority

## Context

gfx1100 runs exactly one StreamK mode (SK3). Its fixup costs ~11% extra memory traffic on a large
shape, and the two paths that might beat it are blocked: **SK4/SK5 will not assemble** (a gfx12
`th:TH_ATOMIC_RETURN` fallback on an arch that spells it `glc`), and **`StreamKAtomic` is gated to
SGEMM**. This campaign unblocks what it can, improves SK3 where it can't, and measures everything.

**TN HHS is the priority** (user-directed). Runs unattended for ~10 h; user checks back after.

### The atomic blocker is structural — verified, and it decides the ordering

```python
Solution.py:1686   reject if not EnableMatrixInstruction    # StreamK needs an MI
Solution.py:1713   reject if not DataType.isSingle()        # atomic needs fp32 INPUT
```

gfx1100 has **no fp32 MatrixInstruction** (WMMA is f16/bf16/i8/i4 → f32 accumulate), so the two
gates are **jointly unsatisfiable**. True HHS is doubly out: it writes fp16 D, the emitted instruction
is `BufferAtomicAddF32` (`GlobalWriteBatch.py:2601`), and gfx1100 has no `global_atomic_pk_add_f16`.

The gate tests the wrong thing — the atomic writes **D**, so the real predicate is the *destination*
type, and the machinery already keys on compute type (`KernelWriter.py:7453`: `useAtomicAdd =
HasAtomicAdd and ComputeDataType.isSingle()`). For **HSS** (fp16 in → **fp32** out) accumulator,
compute type and D are all fp32 and `HasAtomicAdd` is true here. That is the only legal route to the
atomic path on this part — **adjacent to TN HHS, not it**, and every number from it must say so.

**Consequence for an unattended run:** the atomic workstream is the least certain and the least
on-priority, so it goes last. The SK3 knob sweep needs **no patch at all**, is pure TN HHS, and is
near-certain to produce a result — so it goes **first**, before the risky patch work. If everything
after P1 fails, the run still delivers a TN HHS finding.

---

## Phases — ordered by (certainty × priority), not by narrative

Each is independent and fail-open: a failure is recorded and the driver proceeds to the next.

| # | phase | patch needed? | workload | cap |
|---|---|---|---|---|
| P0 | infra + **baseline gate** | no | TN HHS | 0:45 |
| P1 | **SK3 knob sweep** | **no** | **TN HHS** | 2:00 |
| P2 | SK4/SK5 `glc` patch + validate | yes (1 line) | TN HHS | 1:30 |
| P3 | SK3 vs SK4 vs SK5 benchmark | — | TN HHS | 2:30 |
| P4 | atomic path | yes (1 line) | **HSS, not HHS** | 1:30 |
| P5 | report + revert | — | — | 0:45 |
| — | slack | | | 1:00 |

### P0 — Infrastructure and the baseline gate (0:45)

```bash
cd <stock>/projects/hipblaslt/tensilelite
export PYTHONPATH=<stock>/build/release/tensilelite/rocisa:<stock>/projects/hipblaslt/tensilelite
invoke build-client --gpu-targets gfx1100 --no-rebuild-rocisa
```

- `PYTHONPATH` is **mandatory** — a stale `rocisa-dev.pth` in `~/.local/.../site-packages` points at
  an unbuilt tree. Verified working.
- `--no-rebuild-rocisa` is **deliberate**: the task defaults it to `True`, which would re-install the
  editable rocisa and disturb the exact `.pth` situation the `PYTHONPATH` workaround already routes
  around. Do not let an unattended run mutate its own interpreter setup.
- Fallback on failure: `--prebuilt-client /home/vmijovic/tuning/temp/origami_test/0_Build/client/tensile_client`
  (Sept 2025 — **ABI risk**, so smoke-test on one shape before trusting it).

**Record a revert point first.** `exp/stock` is a TheRock worktree and is *already dirty* (three
uncommitted `ORIGAMI_*` patches). Capture `git -C <stock> diff > P0_baseline.patch` plus
`git status --porcelain`, so every later change is attributable and revertible.

**Gate:** a known-good SK3 TN HHS config must build **and benchmark end-to-end** before any patch.
The earlier probe never reached the client; until that is disproved nothing downstream is
interpretable. **If this gate fails, skip P2/P3/P4 and go straight to P5** — do not burn 8 h on a
broken harness.

### P1 — SK3 knob sweep · TN HHS · no patch (2:00)

The shipped catalog leaves two StreamK-family knobs nearly untouched, and both act on the fixup — the
thing measured as the cost:

| knob | shipped on navi31 | role |
|---|---|---|
| `StreamKFixupTreeReduction` | 180×0, **12×1** | tree vs linear reduction chain |
| `StreamKXCCMapping` | 192×0 | **negative control** — expected inert on monolithic Navi |

Fork `StreamKFixupTreeReduction: [0, 1]` on the pinned geometry. The XCC arm exists so the harness
can demonstrate it detects "no effect"; without it a positive result on the other arm is unfalsifiable.

### P2 — Unblock SK4/SK5 · TN HHS (1:30)

One line, `Components/StreamK.py:363-368`. `GLOBALModifiers` already has a `glc` field; the renderer
emits `glc` only when `HasGLCModifier` (true gfx11 / false gfx12) and `scope:`/`th:` only under the
gfx12 caps — so setting **both** is this codebase's own idiom (cf.
`StreamKMemoryOrderingDefault.flagBufferMubuf()`: `glc=True, dlc=True, scope=SCOPE_DEV`).

```python
has_th = bool(writer.states.asmCaps.get("HasTHModifier", False))   # False gfx11, True gfx12
modifier=GLOBALModifiers(glc=not has_th, scope=CacheScope.SCOPE_DEV)
```

Assembler-verified: `global_atomic_inc_u32 v3, v1, v2, s[46:47] glc` assembles on gfx1100, SADDR form
valid. Blast radius nil — SK4/SK5 ship nowhere, gfx1100 has no scalar atomics, and GSU's analogous
fallback uses `FlatAtomicDecU32`.

**Correctness is the deliverable, not the build.** `NumElementsToValidate: -1`. An SK4 that assembles
and computes garbage is a worse outcome than one that doesn't assemble, and only validation separates
them. **If validation fails, record it and skip P3's SK4/SK5 arms** — still run SK3.

### P3 — Benchmark SK3 vs SK4 vs SK5 · TN HHS (2:30)

Reuse `~/sk_modes/configs/probe_sk345.yaml` (already pinned to a real shipped navi31 SK3 solution).
MT128x128 needs 266 VGPRs against a 256 budget — use the MT128x64 / 64x64 / 64x32 geometries that
survived.

- ~24 census shapes where SK3 **actually streams**: `skTiles != skGrid` **and** `itersPerTile > 1`.
  The second conjunct is not optional — a tile with `ipt == 1` cannot be split.
- Stratify <0.1 ms / 0.1–1 ms / ≥1 ms and **report per band**. A single geomean reverses sign against
  the banded view on this workload.
- DP floor via `TENSILE_STREAMK_DYNAMIC_GRID=4`.
- **Measure the noise floor by repeating one arm** before interpreting any contrast. Bootstrap CIs
  resample shapes and cannot see it.

### P4 — Atomic path · HSS, not HHS (1:30, first to be dropped)

Patch `Solution.py:1713` to the **destination** predicate (what `BufferAtomicAddF32` actually
requires), then build TN **HSS** (`DataType: h`, `DestDataType: s`, `ComputeDataType: s`) with
`StreamK: 3` + `StreamKAtomic: 1`. `storeBranchesCommon` (`:748`) and `writePartialsCommon` (`:1049`)
both early-return under `StreamKAtomic`, so this genuinely removes workspace, flag and spin-wait.

**`NumElementsToValidate: -1` is mandatory and results are reported only with validation status.** A
float reduction can be *nearly* right; "it ran and was fast" is not a result. Label every number
**HSS**.

### P5 — Report and revert (0:45)

`~/sk_modes/REPORT.md` — but written **incrementally after every phase**, not at the end, so a crash
still leaves findings. Then `git -C <stock> checkout` the two patched files, and confirm the final
`git diff` equals `P0_baseline.patch`.

---

## Unattended driver

`~/sk_modes/run_campaign.sh`, launched under `setsid`, with `~/sk_modes/ledger.json` recording
`{phase: {status, started, ended, artifacts, note}}`.

- **Every phase wrapped in `timeout <cap>`** — one hang must not eat the budget.
- **Fail-open**: non-zero exit records `failed` and continues. Only the P0 gate can skip later phases.
- **Resumable**: on restart, phases marked `done` are skipped.
- **`flock ~/sk_modes/.gpu.lock`** around every GPU action — one job at a time, never benchmark
  while building.
- **No interactive anything**; all output tee'd to `~/sk_modes/logs/<phase>.log`.
- `rocm-smi --showclocks --showtemp --showpower` before/after each timed phase; stock clocks only.
- Patches applied per-phase and reverted in P5 — never committed.

## Verification

1. P0 baseline SK3 TN HHS build+benchmark reproduces before any patch.
2. Each variant's kernel name carries the expected `_SK<n>_` token. Name-matching against logic YAMLs
   does **not** work — built kernels carry an `SKWS` token the YAMLs predate.
3. `NumElementsToValidate: -1` passes for every arm that reports a number; arms that fail validation
   are reported as failures, not omitted.
4. Noise floor measured in-session by arm repetition; only differences above it are claimed.
5. Final `git -C <stock> diff` == `P0_baseline.patch` (i.e. only the pre-existing `ORIGAMI_*` remain).

## Most likely failure modes

1. **Client build fails** → prebuilt fallback, ABI-smoke-tested; if both fail, P0 gate skips to P5 and
   the report says the harness was the blocker.
2. **SK4/SK5 assemble but mis-compute** → validation catches it; that is the finding.
3. **All arms tie** because shapes chosen where StreamK is inert → `ipt > 1` conjunct guards it, and
   the DP floor proves the harness can see a difference at all.
4. **Atomic validates but is slower** (contention on C) → still a result.
5. Overrun → P4 drops first; P1 alone guarantees a TN HHS deliverable.

---

*Not in scope: deck tasks #19–26 (`s5.modes`, `s5.owner`, `s5.gra`, `s5.dpswitch`, `s5.trace`,
`s5.fixup`, `iteration_carve`) remain pending from the earlier plan.*
