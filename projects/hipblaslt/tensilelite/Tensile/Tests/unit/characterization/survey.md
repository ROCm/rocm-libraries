# Characterization-test tooling survey

Goal: pick a tool to pin the *current* behaviour of a tensilelite module
with snapshot-style tests that (a) fit the existing pytest/tox setup, (b)
tolerate internal refactors, and (c) handle determinism. One module is
characterized first (`Tensile/SolutionStructs/Validators/`, see
[target.md](target.md)); the tool should also scale to later modules whose
output is structured (solution dicts, MI-parameter dicts, YAML logic).

## Candidates

| Tool | What it is |
|---|---|
| **syrupy** | pytest plugin; `assert x == snapshot`. Stores snapshots in `__snapshots__/` (amber) or single files. `--snapshot-update`. |
| **approvaltests** | `verify(x)`; writes `*.approved.txt` / `*.received.txt`; needs a diff "reporter" and a `.gitignore` for received files. |
| **pytest-snapshot** | pytest plugin; `snapshot.assert_match(value, "name")`. One file per snapshot. `--snapshot-update`. |
| **pytest-recording** (vcr) | Records HTTP interactions to cassettes. Not a value-snapshot tool. |
| **plain golden-file** | Hand-rolled: serialize value, compare to a committed file, env-var to regenerate. |

## Comparison

| Criterion | syrupy | approvaltests | pytest-snapshot | pytest-recording | golden-file |
|---|---|---|---|---|---|
| **Setup cost** | low — `pip install syrupy`, use `snapshot` fixture | medium — install + configure a reporter; manage received/approved + gitignore | low — plugin + `snapshot` fixture | low to install | medium — write+maintain the harness |
| **Refactor tolerance** | high — asserts on the *value* you pass; internal refactor that preserves output keeps snapshots green | high — same value-based idea | high | n/a (only HTTP) | high but DIY |
| **Determinism handling** | you normalize the value before asserting; supports custom serializers/matchers; pairs with seeded RNG/frozen time | you normalize via `scrub`/options | you normalize before `assert_match` | records real I/O — *adds* nondeterminism unless you scrub | fully manual |
| **Diff UX** | rich pytest diff on the structured object; `--snapshot-update`; `--snapshot-details` | external diff tool / received-vs-approved files | plain file diff; `--snapshot-update` | cassette diff | whatever you build |
| **Fit with pytest/tox** | native pytest plugin → works under existing `pytest.ini` markers and the `coverage*` tox envs with zero glue | works but adds received-file lifecycle + reporter config to manage in CI | native plugin, fine | irrelevant — no network here | works but is more code to own |

## Decision: **syrupy**

Rationale:
- **Native pytest plugin** → drops into the existing `pytest.ini` /
  `-m unit` flow and the `coverage`/`coverage-unit` tox envs with no extra
  CI plumbing (no approved/received file lifecycle, no external reporter).
- **Value-based, refactor-tolerant**: the target validators return
  structured Python (`matrixInstructionToMIParameters` → a dict of MI
  params; the `validate*` fns → a bool plus a `state["Valid"]` mutation).
  Snapshotting the *returned structure* pins behaviour while leaving the
  implementation free to change — the core of characterization testing.
- **Determinism is in our control**: the chosen target is pure (no RNG,
  time, file, or network — see target.md), so snapshots are naturally
  stable. Where a value carries an env-dependent field (e.g. an `IsaInfo`'s
  `asmCaps` derived from the live assembler), we snapshot the *normalized*
  structured form or pin a synthetic `isaInfoMap`, never a raw blob. syrupy
  lets us pass exactly the normalized object to `== snapshot`.
- **pytest-recording is the wrong category** (HTTP cassettes; nothing here
  does network I/O). **approvaltests** and **golden-file** are viable but
  cost more lifecycle/harness management for no benefit over syrupy here.
  **pytest-snapshot** is a fine second choice but is less ergonomic for
  rich nested structures than syrupy's amber serializer.

Verified present in the dev env: `syrupy==5.3.1` (see
`work/tensilelite-characterization/env/Dockerfile`).

### How determinism is handled with syrupy here
- Target modules are pure → no RNG seeding / time freezing needed for the
  values themselves (confirmed: no `random`/`time`/`open`/`global` in the
  three files).
- `reject()` writes to stdout and can raise on LibraryLogic states; tests
  pass `printRejectionReason=False` to keep it silent and assert on the
  returned bool + `state["Valid"]`, never on captured stdout.
- For MI-parameter snapshots that embed an ISA tuple / caps, we drive with
  a fixed, explicit `isaInfoMap` (synthetic or pinned ISAs) so the snapshot
  is reproducible regardless of the host compiler version.
