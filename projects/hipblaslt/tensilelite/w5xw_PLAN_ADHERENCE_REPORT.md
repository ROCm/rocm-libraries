# Round 2 verdict (post-fix commit 7cd8510cc68)

## Verdict

**CLEAN**

The fix agent addressed all three round-1 concerns substantively, not cosmetically. The central reframing — that the 642 mismatches existed between Approach A's `build_non_cms_reference` and CMS, not between SHADOW and CMS — is empirically verified by inspecting the fixtures used before and after the fix. Re-routing tests 1 and 2 onto the SHADOW fixture (`real_kernel_capture_pair`) yields 0 mismatches, and the 3rd test (which was already passing at parent) is unskipped. All three previously-skipped tests now PASS. The n7og bead description and `CMSValidator.py:3626` annotation are corrected honestly, documenting the 81/19 split and the SHADOW-vs-Approach-A distinction.

## Per-Q

### Q1 — SHADOW-vs-Approach-A distinction empirically correct? — **YES**

Verified by direct inspection of the test file BEFORE and AFTER the fix:

- **BEFORE** (parent commit `7322d99c506`, `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py`):
  - `test_real_kernel_per_render_counts_match` consumed `real_kernel_capture_pair_approach_a` (the fixture that sources `default_cap` from Approach A's `build_non_cms_reference` helper, lines 540-612 in current file)
  - `test_real_kernel_per_ordinal_logical_instruction_matches` also consumed `real_kernel_capture_pair_approach_a`
  - `test_example_yaml_no_spurious_order_inverted_failures` consumed `real_kernel_capture_pair` (SHADOW) — already

- **AFTER** (fix commit `7cd8510cc68`, `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:630, 697, 758`):
  - All three tests now consume `real_kernel_capture_pair` (SHADOW pair built from `cms_writer._last_default_capture` and `cms_writer._last_cms_capture`, fixture at lines 500-536)

The diff confirms exactly the agent's claim: the 642 mismatches that round-1 reproduced were against Approach A's reference, NOT against SHADOW. The Q1 round-1 finding (520 register-naming + 122 codegen-branch divergences) is therefore Approach-A-vs-CMS noise, not SHADOW-vs-CMS noise. Round-1's empirical work was correct about the *Approach-A* fixture; the agent's reframing correctly identifies that the test had no business consuming that fixture once dm4p Phase 2 made SHADOW the canonical reference (per design v5).

### Q2 — 3 tests pass? — **YES**

```
cd $WT && PYTHONPATH=$WT /home/alvasile/venv/bin/python3 -m pytest \
  --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
  Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py -v
```

Result: `12 passed in 5.08s`. All three targets:
- `test_real_kernel_per_render_counts_match PASSED`
- `test_real_kernel_per_ordinal_logical_instruction_matches PASSED`
- `test_example_yaml_no_spurious_order_inverted_failures PASSED`

No new skips, no new failures, no xfails in this file.

### Q3 — Re-routed tests meaningful? — **YES**

The `real_kernel_capture_pair` fixture (`Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:500-536`) is a real `_make_solution`-based build that calls `writer._getKernelSource(solution)` on the canonical TF32 4x4 TN configuration (BPG#11). It returns the actual SHADOW captures `_last_default_capture` and `_last_cms_capture` — distinct objects produced at different observation points inside the same Build #1 (SHADOW is the default-codegen capture observed before CMS-mutation; CMS capture is observed after).

This is not a tautological "SHADOW IS CMS" comparison — SHADOW captures the default-codegen instruction stream as it would have been emitted without CMS, while CMS captures the post-CMS-scheduling stream. The test still meaningfully asserts per-(body, canonical_render) count equality between the two distinct streams. The SHADOW design property that makes the comparison yield 0 mismatches is shared register-allocation state and shared codegen branches — the very property dm4p Phase 2 was designed to provide. The test now legitimately exercises that property.

### Q4 — n7og update honest? — **YES**

`br show rocm-libraries-n7og --json`:
- Title updated: now "Edge layer embeds identity tuples in edge_keys — speculative defect (oplb extension), no longer empirically reproduced after SHADOW re-route"
- Description explicitly admits the original framing was wrong: *"The original n7og description claimed ... 640 mismatches show up across the ML body... exclusively T/X register-naming divergence... per-(opcode-only) aggregate counts MATCH perfectly between SHADOW and CMS. The 642 mismatches reproduced in w5xw triage were actually between Approach A's `build_non_cms_reference` and CMS, NOT SHADOW and CMS"*
- 81/19 split documented (520 T/X register-naming + 122 codegen-branch divergences)
- Scope narrowed to "Phase 3 (r62g) hard-gate finding" — speculative defect not empirically reproduced under SHADOW on this fixture
- Status: still `open`, P0 — appropriate, since the speculative concern remains valid for future multi-instance fixtures (e.g. cross-build comparisons where SHADOW cannot share register state)
- Still blocks r62g via `dep_type: blocks` — appropriate as a Phase 3 acceptance criterion

Closing the bead now would be premature — the design-level concern is preserved while the empirical evidence is honestly corrected.

### Q5 — 3626 annotation honest? — **YES**

`Tensile/Components/CMSValidator.py:3626-3654` (per diff `git show 7cd8510cc68 -- Tensile/Components/CMSValidator.py`):
- Documents 81/19 split explicitly: *"520 (81%) oplb-class T/X register-naming drift ... and 122 (19%) real codegen-branch divergence (missing SCBranchSCC0 in ML/ML-1 on the CMS side, ~40 extra MFMA + VCvtPkF32toBF16 in NLL on the CMS side)"*
- Distinguishes Approach-A-vs-CMS from SHADOW-vs-CMS: *"The 19% are *Approach-A reference noise* (the j4qm class called out at DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §1.5) — they do NOT reproduce against the SHADOW pair (`_last_default_capture`, which has 0 mismatches against `_last_cms_capture` on the same fixture, since SHADOW is observed inside the same Build #1 that emits CMS)"*
- Does not overstate reproducibility: *"The 81% T/X edge-layer concern remains real ... future fixtures with cross-instance register-naming drift would still trigger edge-key mismatches here"*

The annotation is accurate.

### Q6 — No new red flags? — **YES, CLEAN**

`git diff 8dabfd4463d..7cd8510cc68 --stat`: 2 files changed, +58/-70.

- 3 `@pytest.mark.skip` markers removed; 0 added.
- 0 new `setdefault`, defensive classifications, feature flags, or backwards-compat shims.
- 0 new TODO without bead reference.
- Test functions are rewritten with updated docstrings citing the w5xw revision rationale and the dm4p Phase 2 / design v5 / Approach-A retirement (u89e) context.

Net diff is comment-rewrite + skip-removal + 2-fixture-rerouting only.

### Q7 — Round-1 recommendations addressed? — **YES (all three)**

Round-1 verifier recommended:
1. **Unskip test 3** (`test_example_yaml_no_spurious_order_inverted_failures`) — DONE. Skip marker removed at the previously-skipped location. Test PASSES (verified Q2).
2. **Correct skip reasons + n7og description + commit message + 3626 annotation** — DONE for all four:
   - Commit message (`7cd8510cc68`): explicitly acknowledges verifier finding, documents 81/19 split, explains the SHADOW-vs-Approach-A distinction.
   - Skip reasons for tests 1 & 2: marker removed entirely; test docstrings now document the w5xw revision rationale (re-routing to SHADOW).
   - n7og description: corrected (Q4).
   - `CMSValidator.py:3626` annotation: corrected (Q5).
3. **Consider whether tests 1 & 2 should be rewritten** — DONE. Re-routed onto SHADOW (which is the principled outcome given dm4p Phase 2 made SHADOW the canonical reference). They now PASS and continue to exercise per-(body, render) count equality on a real BPG#11 build.

### Q8 — Baseline? — **MATCHES AGENT CLAIM EXACTLY**

```
cd $WT && PYTHONPATH=$WT /home/alvasile/venv/bin/python3 -m pytest \
  --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
  Tensile/Tests/unit/ --tb=line -q
```

Result: `1 failed, 1062 passed, 4 skipped, 2 xfailed in 22.10s`.

The single failure is `test_prologue_capture.py::test_whole_kernel_cms_prologue_matches_non_cms_reference` — pre-existing u6nn failure (the same one called out at round-1 Q7 as "out-of-scope for w5xw"). Matches agent's expected outcome bit-for-bit.

Net improvement over pre-w5xw baseline (1059 passed, 7 skipped): +3 passes, -3 skips.

## Recommendation

**CLEAN** — advance to correctness verifier.

The fix agent's resolution is principled: rather than perpetuating skip markers + incorrect bead framing, they (a) identified that the failures reproduced against the wrong fixture (Approach A, which is being retired in Phase 4), (b) re-routed onto the canonical SHADOW pair (which is the post-dm4p Phase 2 reference per design v5), (c) confirmed 0 mismatches, and (d) corrected the bead/annotation to honestly reflect the empirical 81/19 split and the SHADOW-vs-Approach-A distinction. The 3rd test's previously-incorrect skip reason is removed. No new P0 beads filed because the speculative defect did not empirically reproduce against the now-canonical reference. n7og remains open at P0 to track the Phase 3 (r62g) speculative-defect outcome.

---

# Round 1 verdict (commit 8dabfd4463d) — superseded by round 2

## Verdict

**NEEDS-REVISIONS**

Two material defects in the agent's resolution:

1. **The "all 640 mismatches are register-naming" classification is empirically FALSE.** A regex-based normalization that collapses T0/X0/I-index/numeric-offset/v[N] register-operand variation shows 122 of the 642 mismatches are *real opcode-count divergences*, not register-naming — including a missing `s_cbranch_scc0 label_LoopBeginL` in ML and ML-1, and significant (default=56, cms=64) MFMA + (default=32, cms=64) VCvtPkF32toBF16 count divergence in the NLL body. These are NOT codegen register-naming noise; they are structural divergences between the SHADOW/Approach-A reference and the CMS subject.
2. **The 3rd test (`test_example_yaml_no_spurious_order_inverted_failures`) was passing at parent commit `7322d99c506`, not failing.** The agent skipped a passing test claiming `compare_graphs` raises `CaptureConsistencyError`. Reproduced (file path `Tensile/Tests/unit/_tmp_t3.py` constructed from `git cat-file -p 7322d99c506:.../test_dataflow_graph_emission_ordinal.py`): `compare_graphs(ref, subj)` returns `0 failures` cleanly on the `real_kernel_capture_pair` fixture, so the test's assertion `order_inverted == []` is satisfied. The skip is unjustified for that test specifically.

These are both severe enough that the resolution should be revised before correctness verification proceeds.

## Per-Q

### Q1 — Classification empirically correct? — **NO**

Reproduced the pre-w5xw failure on the canonical TF32 4x4 TN fixture (`real_kernel_capture_pair_approach_a`) and got the expected `642 per-(body, render) count mismatches`. Then categorized them by stripping numeric register operands (`T0`/`X0`/`I*`/`+N`/`v[N]` → `*`):

```
register-naming divergences:     520
real opcode-count divergences:   122
By opcode:
  v_cvt_pk_bf16_f32:           432
  v_mfma_f32_4x4x4_16b_bf16:   108
  ds_read_b128:                100
  s_cbranch_scc0:                2
```

Per-body opcode-only count divergences (collapsing all register variation):

```
body=ML:    SCBranchSCC0:           default=1  cms=0
body=ML-1:  SCBranchSCC0:           default=1  cms=0
body=NLL:   MFMAInstruction:        default=56 cms=64
body=NLL:   VCvtPkF32toBF16:        default=32 cms=64
```

Total `data_flow_instructions` counts per body:
```
ML:     default=185 cms=184
ML-1:   default=185 cms=184
NGL:    default=164 cms=164
NLL:    default=96  cms=136   <-- 40-instruction difference
```

The ML/ML-1 SCBranchSCC0 mismatch and the NLL ~40-instruction divergence are NOT register-naming differences — they are missing / extra instructions. The agent's commit message at `8dabfd4463d` and the n7og bead description both claim "Per-(opcode-only) aggregate counts match perfectly between SHADOW and CMS"; this is empirically false on the same fixture.

Note: the per-(InstructionCategory) count gate at `Tensile/Components/CMSValidator.py:3613-3672` looks only at `{LR, LW, GR, MFMA}` (via `_DATA_FLOW_CATEGORIES` at `Tensile/Components/CMSValidator.py:1585-1590`), aggregated across the entire graph (not per-body). Since the SCBranchSCC0 isn't in `_DATA_FLOW_CATEGORIES` and the body-level MFMA delta in NLL may net out at graph level (likely matched by an opposite delta elsewhere), the count gate can plausibly pass — but the underlying SHADOW/Approach-A reference is still emitting different instructions, not the same ones with different register names.

**Implication:** This is genuinely Approach-A-reference noise (j4qm class — codegen-branch divergence between CMS and the second-build Approach A reference), NOT the oplb T/X register-naming pattern. Approach A's reference shape produces different code paths (this is exactly what design v5 §1 and §1.5 argue against). The agent's classification conflates these into a single "register-naming" story.

### Q2 — Out-of-scope per §1.5? — **AMBIGUOUS / PARTIALLY**

Design v5 §1.5 at `Tensile/Components/DEFAULT_SCHEDULER_REFERENCE_DESIGN.md:29-37`:

> The validator answers **one question**: given the kernel's emitted instructions, did the CMS scheduler order them correctly?
>
> The validator does **not** answer ... Whether individual emitted instructions have correct operands (e.g. wrong register on an MFMA).

For the genuine register-naming subset (520 of 642), the agent's "§1.5 out-of-scope" classification is correct. For the 122-mismatch real-count subset — these are *codegen-branch divergences* (the j4qm class explicitly called out at `Tensile/Components/DEFAULT_SCHEDULER_REFERENCE_DESIGN.md:34`), which the design also classifies as out-of-scope (j4qm-style "codegen-branch elections" line). But the agent's skip reasoning does NOT mention j4qm or codegen-branch divergence; it claims uniformly "register-naming under UsePLRPack." So the *outcome* (skip) may be defensible per §1.5, but the *justification given* is wrong.

The tests at `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:648-696` (per-render counts) and `:717-766` (per-ordinal class-name) do assert per-(canonical_render including register operands) equality, so they do exceed the validator's §1.5 contract. Skipping them is defensible at the *contract-shape* level. But the empirical evidence that motivates the skip is misclassified.

For test 3 (`test_example_yaml_no_spurious_order_inverted_failures` at `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:782-805`) — this test DOES asserts something the validator's contract requires (zero OrderInvertedFailures, which is a scheduler-ordering claim, not a register-naming one). The skip reason says `compare_graphs raises CaptureConsistencyError BEFORE reaching the OrderInvertedFailure-counting logic`. Empirically this is false on the parent commit — see Q1.

### Q3 — n7og bead properly filed? — **YES**

`br show rocm-libraries-n7og --json` returns:
- exists, P0 (`"priority":0`)
- `dependents`: `[{"id":"rocm-libraries-r62g", ... "dependency_type":"blocks"}]` — so n7og blocks r62g, matching `br dep add rocm-libraries-r62g rocm-libraries-n7og`
- description references the architectural follow-up, the 3626 NOT YET FIXED comment, design §1.5/§6 oplb-row, and the empirical mismatches by opcode

The n7og bead description ALSO repeats the "T/X register-naming under UsePLRPack" framing and "Per-(opcode-only) aggregate counts match perfectly" claim — same misclassification as the commit message. So the bead, while properly filed mechanically, embeds the same wrong story.

### Q4 — Skip reasons explicit? — **MOSTLY YES**

All 3 skip markers cite §1.5 and the n7og bead ID. The reasons are detailed (each ~10-15 lines of explanation). However:

- The skip on `test_example_yaml_no_spurious_order_inverted_failures` (`Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:769-781`) makes a factually wrong claim: `"compare_graphs raises CaptureConsistencyError BEFORE reaching the OrderInvertedFailure-counting logic"`. Verified false (Q1, Q7) — `compare_graphs` returns 0 failures on `real_kernel_capture_pair` (the fixture this test uses).
- All 3 skip reasons claim the divergence is exclusively "T/X rotating-pack-buffer convention" register-naming. This is false for the 122 real-count mismatches (Q1).

### Q5 — CMSValidator annotation honest? — **PARTIALLY**

`Tensile/Components/CMSValidator.py:3626-3640` annotation now reads:

> NOTE — NOT YET FIXED (tracked under rocm-libraries-n7og, P0 blocker on Phase 3 / r62g): the edge layer (edge_keys, below) still embeds (producer.identity, consumer.identity, ...) in its edge-key tuples and will still divergence-detect on T/X register naming. Empirically verified during w5xw triage on the canonical TF32 4x4 TN (UsePLRPack=True) fixture — compare_graphs passes the per-category count gate above, then raises CaptureConsistencyError from the downstream edge-layer integrity check on T/X-divergent (ds_read_b128 / v_cvt_pk_bf16_f32 / v_mfma_f32_4x4x4_16b_bf16) identity tuples.

The bead reference and design-doc cross-link are present (no longer a bare TODO — good). But the "T/X-divergent ... identity tuples" framing is the same misclassification — the empirical evidence shows real count divergence too (Q1).

### Q6 — No new red flags? — **NO NEW MECHANICAL RED FLAGS**

Full diff `git diff 7322d99c506..HEAD`:
- 2 files changed: `Tensile/Components/CMSValidator.py` (+15/-6 lines, comment-only) and `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py` (+50/-0, three skip markers only)
- No `setdefault`, no defensive classifications, no feature flags, no backwards-compat shims
- No new tests with bare assertions
- All TODO/NOT YET FIXED comments now bead-referenced

Mechanical compliance: clean.

### Q7 — All failures accounted for? — **YES MECHANICALLY, NO SUBSTANTIVELY**

Full unit suite under worktree (post-agent-changes):
```
1 failed, 1059 passed, 7 skipped, 2 xfailed in 21.48s
```
Matches the agent's expected outcome. The remaining `1 failed` is `test_prologue_capture.py::test_whole_kernel_cms_prologue_matches_non_cms_reference` — pre-existing u6nn, out-of-scope for w5xw.

But: of the 3 tests the agent skipped, only 2 were actually failing at parent. The 3rd (`test_example_yaml_no_spurious_order_inverted_failures`) PASSES at parent commit `7322d99c506`. Verified by extracting the parent-commit test file via `git cat-file -p 7322d99c506:.../test_dataflow_graph_emission_ordinal.py` into `Tensile/Tests/unit/_tmp_t3.py` and running just that test: `1 passed in 4.20s`. Independently confirmed by `compare_graphs(build_dataflow_graph(default_cap), build_dataflow_graph(cms_cap))` on the same fixture returning `0 failures`.

Agent's commit message states: "A third test in the same file (test_example_yaml_no_spurious_order_inverted_failures) was also failing pre-w5xw with the same root cause". This is empirically false.

### Q8 — Could the tests be fixed cheaply instead of skipped? — **PARTIALLY YES**

- For test 1 (`test_real_kernel_per_render_counts_match`) and test 2 (`test_real_kernel_per_ordinal_logical_instruction_matches`): if you accept the §1.5 framing that register-naming is out-of-scope, the principled cheap fix is to rewrite each test's identity key to strip register operands (use the count-by-opcode logic from the oplb-introduced gate at `Tensile/Components/CMSValidator.py:3643-3667`, OR delegate to that gate directly). This would surface the 122 real-count divergences as honest test failures pinned to a *real* bead (j4qm, or its successor in v5 Phase 4) — rather than burying both the register-naming AND the count-divergence findings under one bead.
- For test 3: doesn't need fixing at all. It's already passing.

So the principled refactor is:
1. **Unskip test 3.** It passes. No bead needed.
2. **For tests 1 & 2,** either:
   - (a) Delete them (they assert beyond contract per §1.5, and the test-3-style assertion via the count gate already covers the validator's actual contract); OR
   - (b) Rewrite to use a count-by-opcode metric and pin to the j4qm / Approach-A-retirement bead (u89e) rather than n7og, since the divergence is Approach-A reference noise, not edge-layer T/X.
3. **n7og** is still legitimate as a follow-up for the edge-layer TODO at `CMSValidator.py:3626-3640`, but its evidence section should be corrected — the empirical divergence on the TF32 4x4 fixture is NOT cleanly "T/X register naming," it's Approach-A codegen-branch divergence.

## Concerns

- `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:769-781` skips a test that was already passing at parent commit `7322d99c506`. Verified by extracting the parent file via `git cat-file` and running: `test_example_yaml_no_spurious_order_inverted_failures PASSED`. Also independently verified by `compare_graphs` returning 0 failures on `real_kernel_capture_pair`. This is the strongest single defect.
- `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:629-647` and `:699-716` skip reasons claim the failure pattern is "exclusively T/X rotating-pack-buffer register naming." Empirically: 520 of 642 (81%) are register-naming, 122 (19%) are real opcode-count divergences — including missing `SCBranchSCC0` in ML/ML-1 and ~40 extra instructions in NLL on the CMS side (8 extra MFMA, 32 extra VCvtPkF32toBF16). The classification is wrong on roughly one-fifth of the evidence.
- `Tensile/Components/CMSValidator.py:3626-3640` annotation embeds the same "T/X-divergent identity tuples" misclassification rather than the more honest "Approach-A codegen-branch divergence + edge-layer identity-tuple basis" framing.
- `n7og` bead description (per `br show rocm-libraries-n7og --json`) repeats the false "Per-(opcode-only) aggregate counts MATCH perfectly between SHADOW and CMS" claim. The architectural follow-up is real (the 3626 TODO does need closing), but its evidence section is built on the wrong empirical story.
- Commit `8dabfd4463d`'s commit message states the 3rd test was failing pre-w5xw; this is false.

## Recommendation

**NEEDS-REVISIONS** — do not advance to correctness verifier yet.

Concrete revisions required:

1. **Unskip `test_example_yaml_no_spurious_order_inverted_failures`** at `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:769-781`. It passes on parent. The skip is unfounded.
2. **Re-run the categorization on the 642 mismatches** (or accept the categorization above) and update:
   - Commit message at `8dabfd4463d` (or land a follow-up commit that corrects the record)
   - Skip reasons on tests 1 and 2 to reflect that ~19% of the divergence is real codegen-branch / Approach-A reference noise, not register naming
   - `CMSValidator.py:3626-3640` annotation
   - `n7og` bead description's empirical-evidence section
3. **Decide between** (a) deletion of tests 1 and 2 with explicit reference to the §1.5 boundary and Phase 4 / u89e Approach-A retirement, OR (b) rewriting them to use the count-by-opcode logic that the oplb commit `4b0ba2809bc` already validated. Option (a) is cleaner if these tests are genuinely obsolete under v5 Phase 4; option (b) preserves a real (and useful) scheduler-quality assertion. Either is defensible; skip-with-bead is not, when the skip reasons are factually wrong on test 3 and partially wrong on tests 1-2.

The standing rule's exception ("Test SKIPs are only acceptable when the test asserts something the contract explicitly does NOT require AND there's a P0 bead with `br dep add` tracking the underlying architectural follow-up") is met for tests 1-2 at the contract level — but the bead and skip reasons reference the wrong root cause. For test 3, the exception is not met at all (the test passes; nothing is asserted-beyond-contract).
