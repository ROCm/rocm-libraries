# Resisting branches — `SolutionStructs/Utilities.py` + `LdsPadding.py`

`Utilities.py` reaches **100% line and branch**. `LdsPadding.py` reaches
**100% line** with **2 residual partial branches**. The notable item is a class
of defensive code in `LdsPadding` that the real access patterns never exercise
through the public selectors — characterized via direct private-helper tests
rather than left uncovered. New file in the per-target dir per the add-only rule.

## `LdsPadding.py` — defensive code unreachable via the public selectors

The public selectors (`get_fp4/fp8/fp16/fp32/mxs_mt_config`) feed the
bank-conflict checkers the *real* `ds_load_tr*` access patterns, for which the
closed-form / tiered search **always** finds a passing `(B, P)` config. A
container probe over a wide (mt, wave, vector-width) grid found **zero** inputs
that drive the fp16 closed-form to fail. So the following branches are dead via
the public API and are instead pinned by **direct tests of the private helpers**
(a legitimate characterization of their documented contract):

| Code | What | How covered |
|---|---|---|
| `_b128_check` L243 (not 16-aligned → False) | b128 dword-alignment reject | `test_b128_check_not_16_aligned_false` (crafted address) |
| `_b128_check` L250 (bank conflict → False) | b128 conflict reject | `test_b128_check_bank_conflict_false` (two threads same base) |
| `_b64_check` L113 (not 4-aligned → False) | b64 dword-alignment reject | `test_b64_check_not_dword_aligned_false` (crafted address) |
| `_b64_compute_config` L169 (no valid config → all-zero) | empty candidate set | `test_b64_compute_config_no_valid_block_returns_zero` (`minB` > all blocks) |
| `_compute_fp16_config` L301-313 (search fallback) | runs only if the closed-form check fails | `test_fp16_search_fallback_finds_config` (monkeypatch `_b128_check` to reject the closed-form pick, accept `B=16,P=4`) |
| `_compute_fp16_config` L314 (search finds nothing) | all candidates fail | `test_fp16_search_fallback_no_config` (monkeypatch `_b128_check` → always False) |

The closed-form **success** path (`_b128_check` returns True, L252) and the b64
tier search are covered transitively by the public-selector grid.

## `LdsPadding.py` — residual partial branches (line coverage still 100%)

| Branch | Meaning | Why not taken |
|---|---|---|
| `122->exit` | `_b64_check` final `if require_some_full_wave and not any_full_wave` falling through to `return True` | The grid inputs that reach this checker always have `require_some_full_wave=False` or do see a full-wave instruction; the combined false-condition fall-through to the loop's end isn't exercised. |
| `309->311` | `_compute_fp16_config` search: the `if best is None or overhead < best[0]` false side (a later `B` ties/loses to the existing best) | The forced-search test accepts only a single `(B, P)`, so `best` is set exactly once; no second candidate competes. Cosmetic; line 311 (the `break`) is covered. |

Both affect blended % only; module **line coverage is 100%**.

## `Utilities.py` — nothing resisted

All branches reachable with plain inputs: `getMiInputType`'s 3 cases;
`reject`'s `NoReject` / quiet / `None` / print-no-index / valid-index-raise /
ProblemType-name-fallback paths (the raise pinned via `pytest.raises`, the quiet
paths driven with `printSolutionRejectionReason=False`); `pvar`; `roundupRatio`;
both `getRealDataType` mappers across the 4 mix dtypes + passthrough.

## Determinism technique (not a gap)

- All outputs are pure; snapshot directly. `reject`'s stdout is suppressed from
  the snapshot (only return + `Valid` captured); its raise message is pinned.
- The `LdsPadding` selectors are `lru_cache`d and deterministic; the
  monkeypatched fp16 search tests use unique `mt` values (20000 / 20032) to
  avoid cache collisions with the real grid.
