# n7og Correctness & Soundness Report

## Verdict
**HAS-BUGS** — the investigation correctly filed a probe test and a carry-forward bead, but the carry-forward bead's mechanism description is empirically wrong. The bead's load-bearing claim that Approach E "does NOT fix this" is FALSE.

Filed P0 bead `rocm-libraries-32tg` documenting the defect (blocks udqg).

## Per-Q

### Q1 — Mechanism diagnosis correct?
**NO. The udqg bead's mechanism is empirically refuted on both failing fixtures.**

Direct introspection of `_last_default_capture` and `_last_cms_capture` on `bpg11-tf32-4x4-tn`:

| metric | SHADOW | CMS |
|---|---|---|
| total edges | 647 | 471 |
| edges with `(-1)` sentinel producer byte-key | **9** | **1** |
| pack→MFMA edges total | 416 | 224 |
| extra-in-SHADOW pack→MFMA edges | 192 | — |
| extra edges with **numeric** byte-keys | **184** (96%) | — |
| extra edges with **sentinel** byte-keys | **8** (4%) | — |

**Critical finding**: `LoopBodyCapture.name_to_idx` is **IDENTICAL** between SHADOW and CMS on every body (PRO/ML-1/ML/NGL/NLL). Both have `ValuA_X0_I0=-1` (sentinel), `ValuA_T0_I0=76`, `ValuB_T0_I0=92`, `ValuB_X0_I0=31`. The bead's claim that "SHADOW's `LoopBodyCapture.name_to_idx` is MISSING the bindings for `ValuA_X0_I0`, `ValuA_T0_I0`, `ValuB_X0_I0`, `ValuB_T0_I0`" is empirically false — those bindings are present in SHADOW (and the only sentinel one, `ValuA_X0_I0=-1`, is ALSO `-1` in CMS).

`collect_regset_stream` (ScheduleCapture.py:2293 — bead misnames it `harvest_name_to_idx`) produces the SAME table for both pipelines.

### Q2 — "Approach E byte-key wouldn't fix this" correct?
**NO. Approach E genuinely fixes both failing fixtures.**

Direct test (compare graphs on byte-key basis `(prod_byte_key, cons_byte_key, edge_kind)` instead of identity-tuple basis):

| fixture | SHADOW set | CMS set | SHADOW \ CMS | CMS \ SHADOW |
|---|---|---|---|---|
| bpg11-tf32-4x4-tn | 139 | 139 | **0** | **0** |
| oplb-tf32-6x8-tn | 255 | 255 | **0** | **0** |

All 192 "extra-in-SHADOW" edges on bpg11 have a byte-key equivalent in CMS (matched=192, unmatched=0). The udqg bead's claim that Approach E "ALSO collapses on the bad sentinel key" is directly refuted.

The real divergence: SHADOW and CMS emit the same logical edges with the same byte-key footprints but **different `canonical_render` strings** (different operand-numbering syntactic conventions for the same physical register). The identity-tuple in `edge_keys()` (CMSValidator.py:1300-1303) embeds `producer.identity` which embeds `canonical_render` — exactly the divergence n7og's ORIGINAL Approach-E hypothesis predicted.

### Q3 — xfail strict catching regressions?
**YES, the xfail mechanism works correctly.**

- `--runxfail` invocation shows the underlying assertion DOES fire on both failing fixtures (208 / 624 mismatches).
- `strict=True` is set on both xfail marks (`test_n7og_edge_keys_multifixture.py:225-247`).
- The assertion compares SHADOW edge_keys to CMS edge_keys via set symmetric difference — not self-fulfilling.
- If udqg is fixed (probe passes), strict=True causes XPASS to fail loud → regression caught.

The test scaffolding is sound. The defect is in the bead's narrative, not the test machinery.

### Q4 — bf16-256x256x64-tn passing pin meaningful?
**YES.**

Verified via `--runxfail`: bf16 fixture genuinely passes (1 passed). The fixture has `UsePLRPack=False` (not specified → default False; no `UseMFMAF32XEmulation`), so it doesn't exercise the pack-buffer rotation path that triggers the canonical_render divergence on the TF32+UsePLRPack fixtures. It is a legitimate negative-control pin.

If SHADOW regressed on bf16 (e.g. introduced a new canonical_render divergence in the non-pack path), the test would fail loud.

### Q5 — udqg description complete?
**NO — major mechanism error.**

What udqg correctly captures:
- Per-fixture mismatch counts (208 / 624 / 0) — accurate.
- File:line citations to `CMSValidator.py:1300` and `ScheduleCapture.py:1428` — correct.
- Status, priority, label, blocker chain to r62g — correct.

What udqg gets wrong:
- "SHADOW capture's `LoopBodyCapture.name_to_idx` is MISSING the bindings" — false (identical to CMS).
- "Approach-E byte-key matching does NOT fix this" — false (fixes it completely).
- "(b) Drop the broken-byte-keys edges from edge_keys" would mask 9/192 = 4.7% of the failing edges, not the bulk of them.
- The acceptance criterion "SHADOW pack-MFMA edge byte_keys resolve to numeric VGPR indices" is misaimed — 184/192 of the failing edges ALREADY have numeric byte-keys.
- Function name `harvest_name_to_idx` at `ScheduleCapture.py:2319` doesn't exist; actual function is `collect_regset_stream` at line 2293 (plan-adherence verifier also flagged this).

### Q6 — Test coverage gap acknowledgment?
**Minor concern, not blocking.**

udqg acceptance criterion only requires three fixtures to pass; StreamK/GSU/sparse-MX are not exercised here (no CMS schedules registered for those classes on gfx950). Plan-adherence noted this. Since udqg's mechanism is wrong, the acceptance criterion is also wrong — the actual fix (Approach E) is general and would cover these classes uniformly once their schedules land.

### Q7 — Other discoveries?
The diagnostic harness revealed:

1. SHADOW has 9 sentinel byte-keys total; CMS has 1. Both pipelines emit at least one sentinel. The "(-1) sentinel emission" is a real (small) defect orthogonal to the main divergence. Could be a separate bead but is currently subsumed by 32tg's narrative.

2. The probe test docstring (lines 64-80) documents "Outcome B mechanism" — those lines are wrong per the same evidence as udqg. When 32tg/udqg are fixed, the docstring should be corrected. No code change needed now.

3. The probe test currently passes 1/3 fixtures and xfails 2/3, with the xfail reason embedding the udqg ID. When the actual fix (Approach E) is applied, all 3 fixtures pass and the xfail flips XPASS → fails strictly → marks must be removed. This is the correct surfacing pattern, just with wrong narrative.

### Q8 — Baseline?
**Match.** `1063 passed, 4 skipped, 4 xfailed, 1 failed in 24.87s`. Pre-existing failure is `test_prologue_capture.py::test_whole_kernel_cms_prologue_matches_non_cms_reference` (tracked as `rocm-libraries-l1l6`), unrelated.

## Bugs found

**Filed:** `rocm-libraries-32tg` (P0, type=bug) — comprehensive defect report against udqg's mechanism description. Linked via `udqg blocks 32tg` (so 32tg must be resolved before udqg can be acted on — otherwise the fix will be aimed at the wrong root cause).

Full evidence:
- SHADOW/CMS name_to_idx tables are identical (per-body printout in diagnostic).
- 184/192 SHADOW-extra edges have NUMERIC byte-keys, not sentinel.
- Approach E byte-key matching produces 0 mismatches on both failing fixtures (BPG#11 and oplb).
- Plan-adherence verifier's flag that `harvest_name_to_idx` is mis-named is corroborated.

## Recommendation

**HAS-BUGS — do NOT proceed to Step 9 squash-merge as-is.**

Options for the parent investigation orchestrator:

1. **Preferred**: Update the udqg bead description in-place to reflect the correct mechanism (allocation-variant canonical_render in identity tuples, Approach E IS the fix), citing 32tg as the empirical correction. Then proceed to squash-merge. The probe test scaffolding remains valid; only the docstring narrative needs a follow-up correction.

2. **Alternative**: Squash-merge the worktree as-is (the probe test is correctly scaffolded and the xfail mechanism works), accepting that udqg + the probe-test docstring carry a now-known-wrong narrative. 32tg ensures the wrong narrative is tracked and will be corrected before any actual fix lands.

Either way, **anyone who picks up udqg will go down the wrong path** if they don't read 32tg first. The dependency `udqg blocks 32tg` is set so that ordering is forced.

The plan-adherence verifier was correct that the investigation produced a CLEAN deliverable structure (probe test, xfail markers, bead filing). The substantive defect is in WHAT the bead says, not HOW the investigation packaged it.
