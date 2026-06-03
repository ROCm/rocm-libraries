# Resisting functions / lines — `Tensile/Common/ValidParameters.py`

**Nothing resisted.** The suite reaches **100.00% line and 100.00% branch**
(169 stmts, 0 miss, 84 branches, 0 partial). As with the `DataType` target,
this is a pure, self-contained module (imports only `math` / `functools` /
`SUPPORTED_ISA` / `IsaVersion`), so every branch was reachable with plain
inputs. This file records the items that *could* have resisted and how each was
handled. New file in the per-target dir per the add-only rule.

## Paths that could have resisted, and how they were reached

| Code | Line(s) | Why it could have resisted | How it was reached |
|---|---|---|---|
| `checkParametersAreValid` unknown-name raise | 1117-1122 | The message interpolates `sorted(validParameters.keys())` (138 names) — snapshotting it whole would be huge and would churn whenever the table changes. | Drove the branch with a synthetic `validParams`, and snapshotted only `str(exc).split("\n")[0]` — the behaviour is pinned without coupling to the full roster. |
| value-not-in-list, `>32`-combos message | 1126-1132 | The truncated-message (`msgExt`) variant only fires when `len(validParams[name]) > 32`. | `test_check_params_value_not_in_long_list_raises` uses a synthetic `{"P": list(range(40))}`; the short-list sibling uses `[1,2,3]` to pin the no-suffix variant. |
| `-1` any-value accept | 1125 (short-circuit) | `validParams[name] != -1` short-circuits only when the value is the literal `-1` sentinel. | `test_check_params_accept_any_value_sentinel` passes `{"P": -1}`. |
| `SpaceFillingAlgo` / `SFCWGM` dispatch | 1133-1136 | The two `elif` arms require the value to pass the membership check first *and* the name to match. | Synthetic `{"<name>": -1}` bypasses the membership raise so control reaches the dispatch; both valid (→ `None`) and invalid (→ propagated sub-validator raise) cases are pinned. |
| sub-validator reject branches | 1078-1089, 1092-1106 | Each `raise` needs a specifically-malformed value. | Parametrized directly over not-a-list / too-many-levels / bad-element inputs for both `checkSpaceFillAlgoIsValid` and `checkSpaceFillAlgoWGMIsValid`. |

## Large structures — snapshotted as normalised summaries (technique, not a gap)

| Builder | Output size | What was snapshotted |
|---|---|---|
| `makeValidMatrixInstructions` | ~746,550 entries | `{len, starts_with [[],[-1]], head[2:10], tail[-3:]}` — pins shape + boundaries, not the full list. |
| `makeValidMFMA` / `makeValidSMFMA` | dict incl. large `_format9` | per-key `{len, head[:3]}` over sorted keys — pins every dtype-combo key's size and content start. |
| `makeValidWorkGroups` | hundreds | `{len, head[:5], tail[-3:]}`. |
| `validParameters` | 138 keys, some multi-100k-entry value lists | `sorted(keys())` roster + per-entry `{type, len/value, head}` structural summary. |

The builders' loops execute fully when called regardless of how much of the
return is snapshotted, so summarising costs no coverage — it only keeps the
`.ambr` reviewable. The choice is documented in `target.md`.

## Determinism notes

- All outputs are pure functions of the inputs — no paths/time/version/GPU — so
  **no normalisation was needed** beyond the size-driven summaries above.
- The `lru_cache` builders are snapshot-safe (same input → same cached output);
  `test_matrix_instructions_lru_cache_identity` pins the caching behaviour.
- The realistic accept (`test_check_params_accept_against_real_table`) selects
  its key dynamically (first list-valued entry, insertion-order-deterministic)
  rather than hard-coding a parameter name, so it survives table edits.
