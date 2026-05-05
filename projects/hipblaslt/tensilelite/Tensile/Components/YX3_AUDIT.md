# Bead `yx3` audit: decorative headers, tombstones, commented-out code

Scope per bead instructions:
- Inline `#` comments and decorative `#` blocks only.
- NOT docstrings (ekz scope, parallel agent).
- NOT signature lines (fnf done).
- NOT trivial restate-next-line inline comments (f80 done).
- NOT bead-citation removals from docstrings (ont done).

## Per-file disposition

### `Tensile/Components/CMSValidator.py`  (1697 → 1488, -209 lines)

| location (orig) | category | verdict | reasoning |
|---|---|---|---|
| 1-21 | decorative | KEEP | Standard MIT license banner |
| 244-258 | tombstone | TRIM | Pack-machinery removal narrative; trimmed to a 9-line "where the invariants live now" pointer (load-bearing for navigators) |
| 270 | tombstone (misindented) | DELETE | One-line stale tombstone with broken indentation; pure historical chatter |
| 272-279 | tombstone | TRIM | "estimate_quad_cycles removed" narrative; trimmed to a 5-line "where the constants live now" pointer |
| 351-355 | tombstone (in-class) | DELETE | `mfma_finish_cycles` removal — purely historical |
| 365-371 | docstring | SKIP | ekz scope |
| 374-377 | tombstone (in-class field) | TRIM | Pack.issue_index "no longer consumed" — kept the "retained as a construction-time tag" justification, dropped the removed-rule narrative |
| 412-415 | tombstone (in-class) | DELETE | MFMAPack ClassVar removal — purely historical |
| 424-433 | tombstone (in-class field) | DELETE | GlobalRead removed-fields narrative; barriered_at field directly above is still used by _validate_needed_by, no reader navigation aid lost |
| 1153-1158 | tombstone | DELETE | apply_barriers removal narrative |
| 1160-1166 | tombstone | DELETE | apply_must_start_after_barriers removal narrative |
| 1171-1179 | tombstone | DELETE | set_lr_needed_by_for_VMFMA removal narrative |
| 1182-1185 | tombstone | DELETE | set_gr_needed_by_from_lrs removal narrative |
| 1187-1194 | tombstone | DELETE | set_gr_must_start_after_* removal narrative |
| 1199-1203 | tombstone | DELETE | _handle_min_pack_quad_cycles removal narrative |
| 1232-1244 | tombstone | DELETE | estimate_quad_cycles removal narrative |
| 1351-1358 | tombstone | DELETE | _transform_index_* + lr_needed_by_mfma removal narrative |
| 1430-1437 | tombstone | DELETE | add_local_read_constraints removal narrative |
| 1442-1452 | tombstone | DELETE | add_gr_not_too_early_constraints removal narrative |
| 1455-1460 | tombstone | DELETE | add_gr_finish_before_lr_constraints removal narrative |
| 1463-1468 | tombstone | DELETE | index_for_force_unroll_sub_iter removal narrative |
| 1471-1483 | tombstone | DELETE | InstructionCountRule removal narrative |
| 1486-1490 | tombstone | DELETE | LRDataReadyRule removal narrative |
| 1495-1501 | tombstone | DELETE | GRAfterLRRule removal narrative |
| 1504-1511 | tombstone | DELETE | GRBeforeLRRule removal narrative |
| 1515-1519 | tombstone (in-list) | TRIM | Replaced inline list-body tombstones with a 6-line module-level "intentionally empty; preserves dispatch contract" comment above both lists |
| 1523-1526 | tombstone (in-list) | TRIM | Same — folded into the above |

### `Tensile/Components/ScheduleCapture.py`  (4555 → 4535, -20 lines)

| location (orig) | category | verdict | reasoning |
|---|---|---|---|
| 1-21 | decorative | KEEP | License banner |
| All `# ====` major-region dividers (53/121/320/459/904/1166/1295/1670/1827/2124/3014/3996/4209/4338) | decorative | KEEP | Each delimits a real architectural region with substantial code on either side and an informative descriptive block under the divider |
| All `# ----` per-Failure-class dividers (567/590/626/644/669/707/726/747/768/783) | decorative | KEEP | Each carries unique Failure-class descriptive content; not pure filler |
| Intra-function `# ----` markers (2756/2794) | decorative | KEEP | Phase-1 / Phase-2 markers inside `build_dataflow_graph` |
| 2858-2864 | decorative + tombstone | TRIM | Removed "(both removed; ...)" parenthetical; kept the "sole source" + algorithm description |
| 3496-3500 | tombstone | TRIM | "structural-side ... have been removed" line; kept the "single source of truth" + ISA reference |
| 3997-4011 | tombstone | TRIM | Bullet list of "lint that replaces:" rules with stale CMSValidator.py:3461 / :3464 line refs and two `(removed)` entries; reduced to "Same Failure types..." paragraph |
| 4046-4047 | tombstone | DELETE | "(now-removed) structural-side MiddlePack.validate" — pure historical |
| 4203-4216 | tombstone + stale-line-refs | TRIM | MiddlePack pair-interleaving header; dropped "the structural rule wires pack.pair_consumer..." paragraph and stale "Mirrors `_hook_up_middle_16_pairs` line 1944" / "Mirrors `MiddlePack.validate` line 426-433" line refs (those line numbers no longer exist post-deletion) |
| Other `# X was removed` inline comments inside dispatch branches (3247/3283) | tombstone | KEEP | The "structural-side mirror was removed; this dispatch is now the only enforcement path" continues to justify dispatch order — load-bearing for "why does this `if` come before that `if`?" |

### `Tensile/Components/CustomSchedule.py`  (7763 → 7637, -126 lines)

| location (orig) | category | verdict | reasoning |
|---|---|---|---|
| 1-21 | decorative | KEEP | License banner |
| All `############` architectural dividers (5698/6036/6182/6498/6577/6817/7065/7571) | decorative | KEEP | Each marks a major schedule-family boundary with substantial code below |
| 5008-5133 | commented-out code | DELETE | 126-line commented draft body inside an `elif` that already returns `False, None`; pure refactor debris |
| 2170-2171 | comment (legitimate) | KEEP | Explanatory cross-reference, not commented-out code |

### Test helpers (`Tensile/Tests/unit/cms_test_utils.py`, `cms_validation_base.py`, `dataflow_fixtures.py`, `graph_native_validation_base.py`, `conftest.py`)

| disposition | reasoning |
|---|---|
| All decorative `# =====` dividers | KEEP — every one delimits a logical section with substantive distinct content (Stand-in instruction classes vs Helpers vs TaggedInstruction builders, Capture wrapping vs Failure-list assertions, etc.) |
| Tombstones | NONE FOUND — only "removed" / "deleted" mentions are SWait scenario descriptions in test code, not removed-symbol references |
| Commented-out code | NONE FOUND |

## Verification

```
pytest Tensile/Tests/unit/  →  583 passed, 2 skipped, 1 xfailed
```

Run from worktree (sparse checkout) with the editable Tensile install
overridden via `sys.modules` priming.

## Counts

Per bead-instructions output format:

| category | KEEP | DELETE / TRIM |
|---|---|---|
| Decorative (`# ====` etc.) | ~60 | 0 |
| Tombstones (CMSValidator) | 0 | ~17 (deleted), 4 (trimmed) |
| Tombstones (ScheduleCapture) | ~3 (in-dispatch) | 4 (trimmed) |
| Commented-out code | 0 | 1 large block (126 lines) + the misindented one-liner |

`wc -l` deltas: CMSValidator −12.3%, CustomSchedule −1.6%, ScheduleCapture −0.4%.
Total: 14015 → 13660 (−355, −2.5%).

CMSValidator hit the bead's 5–15% target range. The other two were already
in good shape per the bead's "high-quality baseline" expectation; the
ScheduleCapture deletions were of stale line references and dispatch-order
narratives, not bulk decorative noise.
