# Resistance log — `Tensile/TensileLogic/`

Functions/lines that resisted characterization, with the reason and the
workaround. Kept as a **new file** in the `TensileLogic/` test dir rather than
editing the shared `../resistance.md`, per the add-only rule.

Coverage after the suite: see `coverage-after.txt`. Every file reaches 100%
line **except `ValidChipId.py`**, whose two remaining missing lines (129, 155)
are provably-unreachable defensive code (below). Under add-only we cannot add a
`# pragma: no cover` (that edits source), so they remain "missing" in the
report and are documented here instead.

## Unreachable lines (cannot be exercised by any input)

### ValidChipId.py:129 — base-dir out-of-arch chip ID rejection
```python
if not chip_id_dir.hasChipIdDir:                 # L127
    if not device_ids.issubset(arch_ids):        # L128 (always False)
        return (                                  # L129  <- unreachable
            f"base {gfx} logic may only declare chip IDs available for {gfx} ..."
        )
    return None                                   # L133 (reached)
```
Before `_validateChipIdPlacement` is reached, `_validateChipId` runs
`_verifyPredicate(device_id, gfx)` for **every** device ID (L192-193).
`_verifyPredicate` raises unless `SUPPORTED_BUILD_CHIP_IDS[id] == gfx`, i.e.
unless the chip ID is one of `GFX_CHIP_IDS[gfx]`. So by the time placement runs,
`device_ids ⊆ _archChipIds(gfx) == arch_ids` is guaranteed, making L128 always
False and L129 dead. The check is defensive against a future caller that skips
the predicate loop. **Workaround:** none without source edits; documented here.

### ValidChipId.py:155 — variant-dir chip ID outside the fallback family
```python
family = _fallbackFamily(chip_id_dir.chipId, gfx)    # L153
if not device_ids.issubset(family):                  # L154 (always False today)
    return (                                          # L155  <- unreachable
        f"{chip_id_dir.chipId} directory may only declare chip IDs in fallback ..."
    )
```
`supportsChipIdPredicate` gates only `gfx950` today. For `gfx950`, every source
chip ID in `SUPPORTED_CHIP_ID_FALLBACKS` shares the same direct fallback
(`id=75a0`), so `_fallbackFamily(source, "gfx950")` expands to *all* gfx950
sources plus `id=75a0`. The only gfx950 chip IDs **not** in that family are the
defaults (`id=75a0`, `id=75b0`) — and a default in a variant directory is
already rejected one block earlier at L146-147 (`declared default fallback chip
IDs`). So no input reaches L154 with a non-family ID, and L155 is dead given the
current registry. It would become reachable only if a future arch had sources
with *disjoint* fallback roots. **Workaround:** none without source edits.

> Reachable placement rejections (non-source ID, missing matching ID, default
> in variant, malformed dir) and the base/variant accept paths are all covered
> by the existing `test_ValidChipId.py`; `_chipIdDirFromPath`'s no-arch
> fallback (L114) and `_fallbackFamily`'s skip arm (branch 72->71) are covered
> by `test_validchipid_char.py`.

## Behavioural hazards handled (did not block)

### `reject()` raises on a valid SolutionIndex (inherited from the validators)
`_validateMatrixInstruction` calls `validateMIParameters` with the **default**
`printSolutionRejectionReason=True` (the wrapper exposes no override). When a
solution carries a real `SolutionIndex`, `reject()` *raises* ("rejection of a
LibraryLogic is not expected") rather than returning — an `Exception`, not the
`AssertionError` the wrapper catches. **Workaround:** the reject-path test sets
`SolutionIndex=-1`, so `reject` instead sets `Valid=False` and returns; the
wrapper's own `assert solution["Valid"]` then produces the caught
`AssertionError`. (See `test_validmatrixinstruction_char.py`.)

### `validateMIParameters`/`validateWorkGroup` never set `Valid` on success
Both wrappers `assert solution["Valid"]` after a passing validate, but the
validators only write `Valid` on *reject*. A passing solution must therefore
already carry `Valid` (logic-file solutions do); the accept-path tests pre-set
`Valid=True`. A solution missing the key would `KeyError` — that is the
contract, pinned by the accept tests.

### Module-global failure state in `ValidWorkGroupMappingXCC`
`_xcc_failures_by_file` accumulates per-file reject counts across calls. Every
XCC test calls `reset_reported_failures()` first and snapshots
`{returned, reported_failures}`, so results never depend on test order; a
dedicated 3-solution test pins the one-message-per-file accounting.

### `Run.py` — threading / time / subprocess / multiprocessing
`_progress_loop` (a `threading.Event` + `time.time` loop) and `main`'s
background progress thread are **exercised for line coverage but never
snapshotted**: their stdout is carriage-return progress text with elapsed
seconds. `_progress_loop`'s body is driven by a fake event that returns
`False` once then `True`; `main`'s deterministic Total/Keep/Reject stdout is
snapshotted only with the progress thread disabled (`Verbose >= 2`).

`_setup` (shells out via `validateToolchain`, builds caps via `makeIsaInfoMap`)
and `main` (fans out via `ParallelMap2`, calls `exit()`) are characterized by
**injecting** their collaborators in the `Run` namespace — the validators,
toolchain/caps builders, `ParallelMap2`, and `_setup` itself are monkeypatched —
so the orchestration logic (keep/total/known-bug/chip-id counts, batching,
exit codes) is pinned deterministically with no live toolchain or fan-out.
`_runChecks` is fully covered this way; `Run.py` reaches 100% line.

## Environment note (not a code limitation)

The suite uses `syrupy` (the `snapshot` fixture). The dev image
`tensilelite-char:dev` needs `pip install syrupy` if the container is recreated
fresh — otherwise the `Validators` *and* `TensileLogic` characterization tests
error at setup with "fixture 'snapshot' not found". Same path-mode `--cov`
rule as before: pass a directory, never a dotted module (rocisa nanobind
re-init → SIGABRT).
