# n7og Plan-Adherence Verification

## Verdict
CLEAN

## Per-Q

### Q1 — Empirical claim correct?
**YES, independently reproduced.**

Reran the multi-fixture probe — `2 xfailed, 1 passed` matching the agent's report (pytest output, `test_n7og_edge_keys_multifixture.py::test_shadow_vs_cms_edge_keys_match`).

Independently reproduced the SHADOW-vs-CMS divergence on `bpg11-tf32-4x4-tn` outside pytest:
- SHADOW edges = 647, CMS edges = 471
- `set(graph.edge_keys())` symmetric difference: **208 = 16 (in CMS not SHADOW) + 192 (in SHADOW not CMS)** — exact match with udqg bead's description.
- Edge-kind breakdown: all 647 SHADOW edges and all 471 CMS edges are `raw_intrawave` — the divergence is in edge cardinality, not in edge kind, confirming the "structural edge-count divergence, not just register-naming drift" claim.
- Confirmed `producer_write_byte_key = (('v', -1),)` sentinel pattern is present on SHADOW edges (small count in surface scan, but the structural divergence is consistent with `_resolve_producers` overmatching on the sentinel).

`_byte_keys_for_resource` at `Tensile/Components/ScheduleCapture.py:1466-1478`: when `name_to_idx` lacks the bare name, falls through to symbolic-tuple keying — the agent's claim that this collapses to a `-1` sentinel is approximately correct (the `-1` comes from `RegisterContainer.regIdx=-1` paths combined with unresolved symbolic names; the divergence mechanism is real either way).

### Q2 — xfail (not skip) acceptable?
**YES.**

`Tensile/Tests/unit/test_n7og_edge_keys_multifixture.py:225-247`: both failing fixtures use `pytest.mark.xfail(strict=True, reason="rocm-libraries-udqg: ...")` with explicit per-fixture reasons citing the bead ID and the root cause (sentinel byte-key + missing rotating pack-buffer bindings).

Verified `strict=True` is set on both xfail marks. The cautionary comment at lines 216-223 explicitly forbids "skip" markers and explains the XPASS-on-fix signal mechanism. This is the legitimate "known-broken-tracked" pattern, not the skip-with-todo punt.

Pytest output confirms `XFAIL` (not `SKIPPED`).

### Q3 — udqg properly filed + n7og properly closed?
**YES.**

`rocm-libraries-udqg`:
- exists, status=open, priority=0 (P0), type=bug
- title accurately describes the defect ("SHADOW capture has unresolved name_to_idx for pack-MFMA registers — byte-keys collapse to `(('v',-1),)` sentinel and edge_keys mismatch CMS by hundreds on TF32+UsePLRPack fixtures")
- description carries the per-fixture mismatch counts (208 / 624 / 0), the byte-key sentinel mechanism, the explicit Approach-E rejection with reasoning, and the principled-fix path (extend SHADOW capture harvest)
- `dependents: [{id: rocm-libraries-r62g, dependency_type: blocks}]` — confirmed `udqg blocks r62g`
- labels include `phase-3-blocker`, `shadow-capture`, `validator`, `from-n7og-investigation`

`rocm-libraries-n7og`:
- closed with detailed `close_reason` citing Outcome B, per-fixture mismatch counts, structural mechanism, Approach-E rejection evidence, and udqg as carry-forward
- dependency from r62g to n7og remains (r62g→n7og status=closed) — harmless because closed deps don't block. The active block is now r62g→udqg.

### Q4 — Approach-E-wouldn't-fix-it claim correct?
**YES.**

`Tensile/Components/CMSValidator.py:1229-1303` (`edge_keys`): the matching basis is `(producer.identity, consumer.identity, edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot)` — identity-based, which embeds canonical_render. Approach E would swap this to byte-keys.

`Tensile/Components/ScheduleCapture.py:1428-1478` (`_byte_keys_for_resource`): when `name_to_idx.get(bare)` returns None (line 1475-1476), falls through to `(rt, name, base+i)` — symbolic-tuple key. The agent's claim is that SHADOW's `name_to_idx` is missing the rotating ValuA/B_T0/X0_I0 bindings, so even byte-key matching would collapse SHADOW's pack-MFMA edges to a single sentinel-equivalent key set distinct from CMS's numeric keys. The reasoning holds: switching the matching basis at `edge_keys` doesn't fix a broken upstream byte-key resolver.

### Q5 — Right standing-rule disposition?
**YES — file-new is correct here, not update-existing.**

n7og's original framing was register-naming drift surfacing through identity tuples; the fix was speculated to be Approach-E byte-key matching. Empirically the defect is structurally different (SHADOW's `name_to_idx` is missing pack-buffer bindings, byte-keys themselves are broken, Approach-E doesn't help). The two defects share a symptom but have different root causes and different fix surfaces (edge-layer matching basis vs. SHADOW capture harvest pipeline). Spinning out udqg with the correct framing is cleaner than retrofitting n7og's description and obscuring its history. The r62g blocker chain is preserved (now via udqg).

### Q6 — Test coverage gaps documented?
**YES — gap is real and the bead acknowledges it implicitly.**

`Tensile/Components/CustomSchedule/gfx950/`: confirmed no `streamk`, `gsu`, `sparse`, or `mx` schedule files exist — only TF32 and 16bit (BF16/FP16) schedules. The agent's "not testable on this branch" claim is correct.

Minor concern: udqg's acceptance criteria (1) only requires the three current fixtures to pass; it does not explicitly call out that the same defect may apply to StreamK/GSU/sparse-MX fixtures when CMS schedules are added for them. Not a blocker — the principled fix (extend SHADOW harvest to capture all rotating ValuA/B bindings) would naturally cover those classes if/when they arrive. Could be noted in a future udqg comment.

### Q7 — No new red flags?
**YES.**

Diff scope: 2 files, +401/-28 — `CMSValidator.py` (comment-only update at the 3626 note pointing to udqg) and the new probe test file. No code-behavior changes.

Red-flag grep: only matches are the test file's own cautionary comments warning AGAINST `setdefault` / `skip` / defensive classifications — the test enforces those rules, doesn't violate them.

No new bare `pytest.skip`, no `setdefault`, no defensive `try/except` swallows, no new TODOs without bead references, no feature flags or backwards-compat shims.

### Q8 — Baseline match?
**YES.**

Ran broader suite: `1063 passed, 4 skipped, 4 xfailed, 1 failed in 26.40s`.

Pre-n7og baseline (per instructions): `1062 passed, 4 skipped, 2 xfailed, 1 failed`.
Expected post-n7og: `1063 passed, 4 skipped, 4 xfailed, 1 failed` (+1 passing pin, +2 xfail).

Exact match. The pre-existing 1 failure is `test_prologue_capture.py::test_whole_kernel_cms_prologue_matches_non_cms_reference` (Solution missing `pre_cms_state` snapshot — tracked under `rocm-libraries-l1l6`), unrelated to n7og.

## Concerns

- (Minor) udqg's bead description says the harvest entry point is `harvest_name_to_idx` at `ScheduleCapture.py:2319`; the actual function name at that line is `collect_regset_stream` (`ScheduleCapture.py:2293`). Not load-bearing — the file:line is correct, the function reference can be fixed in a comment update when udqg is worked.
- (Minor) udqg acceptance does not call out StreamK/GSU/sparse-MX fixture classes. Implicit but worth a comment when those CMS schedules land.

## Recommendation
**CLEAN → proceed to correctness verifier.**

All claims independently reproduced; xfail is strict + reason-citing; udqg is correctly filed with active block on r62g; n7og close is appropriate; no architectural red flags introduced; baseline matches exactly.
