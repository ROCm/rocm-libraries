# Review: `input_yaml_validation_plan.md`

## Top of file

### Biggest holes

- **Worker-process exception swallowing.** `_generate_single_solution` (`tensilelite/Tensile/BenchmarkProblems.py:155-194`) wraps the full `Solution(...)` body in `try: ... except Exception as e: print(...); return None`. `Solution.__init__` (`Solution.py:451`) constructs a fresh `ProblemType` at `Solution.py:473` *and* runs `validateParameterTypes` at `Solution.py:491` inside the worker. Step 4 promotes `validateProblemTypeParameterTypes` to raise; Step 8 conditionally drops the Solution-side call. Any `ConfigTypeError` raised inside the worker is caught, printed as a one-line "Error processing permutation", and the bad solution becomes `None`. The plan does not address this swallow at all, and "fail loudly at the earliest validation step" is undermined.
- **Open Q §9.5 is a phantom.** Every `globalParameters["X"] = (\n    value  # comment\n)` line in `GlobalParameters.py:44-326` is a single value in line-continuation parentheses, not a tuple. No `,` ever appears inside the parens. The answer is "there are no tuple defaults; do nothing" — this should have been resolved before producing the plan.
- **Open Q §9.3 (`LibraryType` enum set) is half-resolved and the proposed enum is wrong.** `LibraryIO.py:639-656` shows that `LibraryType` is a *distance metric label* for anything not in `{"FreeSize", "Prediction"}` — the value falls into the `Matching` branch and becomes `Library["distance"]`. The default `"GridBased"` is itself an arbitrary distance label. Enforcing `LibraryType ∈ {"GridBased","Matching","FreeSize","Prediction"}` (the plan's recommendation) would reject valid configs that pass e.g. `LibraryType: Equality` to set the distance label. This is open-ended by design; the plan's instinct ("if the set turns out to be open-ended, drop the enum check") is correct, but the conclusion is already reachable.
- **The `validateParameterTypes` "keep warn-only" decision is a tactical fix.** §1 keeps it warn-mode on the input-YAML path "to avoid threading a strict flag through `Solution.__init__`" and "avoid changing library-logic behaviour." The first reason is exactly the principled-vs-tactical choice the user's "no tactical fixes" rule prohibits — option (a) in §9.1 (a `validateTypes` kwarg) is the principled fix and is trivial to thread. The "library-logic strictness is out of v1 scope" framing dresses up the deferral as scope discipline, but the input-path coverage is what the plan otherwise claims to fix.
- **The plan calls for "single PR that lands strict + tree fix in one shot" but acknowledges step ordering that breaks bisect.** Step 1 fixes YAMLs; Steps 3-7 flip validators to raise. Until all steps in the same PR are merged the tree is broken if anyone bisects past Step 3 to before Step 1 within the PR's history. The plan papers over this by saying "each step is independently mergeable assuming Step 1 has landed" — but that's a within-PR ordering constraint, not independence.

### Biggest strengths

- The in-place-extension shape is correct: each validator already owns its section's dispatch, and a central module would duplicate section detection. The `_expectedParamTypes`/`_expectedProblemTypeParamTypes` registries and the `_skipTypeCheck` set already exist (`Solution.py:165-201`) so the BenchmarkCommon/Fork extension genuinely is "import the map and call it."
- §5 correctly mandates `type() is T` over `isinstance` for the bool/int trap. The existing collector code at `Solution.py:268-287` and `Problem.py:780-799` follows this rule consistently and the extensions can copy it.
- Identifying `StrictTypeLoader` (`LibraryIO.py:81-103`) as the precondition for any of this working at all is not explicit in the plan but is implicitly relied on. The strict loader is the default for `readYAML`, so the plan's strategy of catching `0`/`1` vs `True`/`False` at validators downstream actually works.

### Overall verdict

**Revise.** The architectural shape is sound. The execution plan has three substantive bugs (worker swallowing, phantom open Q on tuples, wrong enum for `LibraryType`), one substantive deferral disguised as scope (warn-only `validateParameterTypes`), and several missed entry points / consumers. Address those before implementing.

---

## 1. Architectural soundness

### Finding 1.1 — In-place shape leaks a coupling the plan denies

**Severity:** MAJOR
**Claim:** §2 says the BenchmarkCommon/Fork extension "imports `_expectedParamTypes` from `Solution.py`." This is a section→section import (`Tensile.Common.ValidParameters` would import from `Tensile.SolutionStructs.Solution`). The current import graph runs the other direction: `Naming → Problem → Solution → Naming` (per the comment at `Problem.py:776-777` explaining a circular-import workaround). Adding `ValidParameters → Solution` may create a new cycle since `Solution.py:39-50` imports from `Tensile.Common`.
**Evidence:** `Solution.py:39-50` imports `Tensile.Common.GlobalParameters`, `Tensile.Common.ValidParameters` (transitively via other Common modules). A reverse import from `ValidParameters` back into `Solution` to grab `_expectedParamTypes` is likely to bite.
**Recommendation:** Move `_getExpectedTypes` and `_skipTypeCheck` into `Tensile/Common/ValidParameters.py` (where `validParameters` already lives). `Solution.py` then imports them from there — same direction as the rest of the import graph. The plan's "no re-exports needed" claim becomes true in the other direction.

### Finding 1.2 — Missing validation contexts: `TensileClientConfig`, `TensileRetuneLibrary`, `TensileUpdateLibrary`, `GenerateSummations`, `BenchmarkSplitter`, `CustomKernels`

**Severity:** MAJOR
**Claim:** §1 enumerates entry points but only covers the `Tensile.Tensile()` flow. Other tools read YAML and construct `ProblemType` / call `assignGlobalParameters`:
- `TensileClientConfig.py:132` reads user YAMLs and at line 172 calls `ProblemType(problemDict)` — note the wrong arity (missing `printIndexAssignmentInfo`), which suggests this path is dead or recently broken. Either way, if Step 4 promotes `validateProblemTypeParameterTypes` to raise, this tool's behaviour changes.
- `TensileRetuneLibrary.py:194` calls `assignGlobalParameters({"LibraryFormat": libraryFormat, "OutputPath": outputPath})` — `outputPath` is a `Path`, not a `str`. After Step 5 promotes type checks on `globalParameters`, this raises unless `OutputPath` lives in `ignoreKeys` (it does, `GlobalParameters.py:760`). Fine, but the plan should call out that programmatic dict callers (non-YAML) also hit the validator.
- `TensileUpdateLibrary.py:42` reads YAML via `LibraryIO.readYAML` — out-of-scope (logic YAML) but worth listing.
- `BenchmarkSplitter.py:42` uses `yaml.safe_load` (not `StrictTypeLoader`) — bypasses the strict loader entirely, so any bool/int mismatches in those YAMLs convert at parse time and never reach the validators.
**Evidence:** grep hits listed above.
**Recommendation:** Add an explicit "Entry points covered" subsection. List every consumer of `LibraryIO.read` and decide per consumer whether it enters the extended validators. Repair `TensileClientConfig.py:172` arity in the same PR.

### Finding 1.3 — Worker-process exception swallowing breaks fail-fast

**Severity:** BLOCKER
**Claim:** `BenchmarkProblems.py:193` swallows every exception from `Solution(...)` construction, including the `ConfigTypeError` Step 4 will raise from `validateProblemTypeParameterTypes` (which runs inside `Solution.__init__` at `Solution.py:473`). Step 8 deletes the post-construction `validateParameterTypes` invocation on this path but does NOT touch the ProblemType-validator call site at `Solution.py:473`, which fires inside the same try/except. Workers run via `ParallelMap2` (joblib); the user gets one error line per bad permutation and a silent count drop, not the structured fail-fast message the plan promises.
**Evidence:** `BenchmarkProblems.py:155-195`. `Solution.__init__` constructs `ProblemType(config["ProblemType"], printIndexAssignmentInfo)` at line 473, which runs `validateProblemTypeParameterTypes` at `Problem.py:816`.
**Recommendation:** Either (a) make `ConfigTypeError` a non-`Exception` subclass (e.g. `BaseException` directly) so the broad `except Exception` doesn't catch it, or (b) replace the broad except with a typed one that re-raises `ConfigTypeError`. Either fix belongs in Step 4. A counter-argument: the worker is fed `deepcopy(problemType.state)` from line 157 of the same file — a state already validated by the *outer* `ProblemType` construction in `BenchmarkProcess.__init__` (`BenchmarkStructs.py:128`). If true, the inner re-validation never fires for input YAMLs. But the plan never makes this argument; it should, or fix the swallow.

### Finding 1.4 — Central module benefits the plan must now re-solve

**Severity:** MINOR
**Claim:** §2 rejects the central module. The user's framing on uniform error formatting, "one place to flip warn↔raise," and cross-section consistency are partly addressed by `TypeValidationErrors.py` (§4) — `ConfigTypeError`, `formatMismatch`, `getStrictMode`. Cross-section consistency checks (e.g. a `GlobalParameter` referenced in a `ProblemType` key path) are not addressed because no validator owns the cross-section view.
**Evidence:** §4 declares the shared module; §3 explicitly says each validator derives its own expected-types map.
**Recommendation:** Either accept that cross-section consistency is out of scope (call it out) or add a Step 9 for it.

### Finding 1.5 — Parameter-name collisions across registries

**Severity:** MAJOR
**Claim:** `validParameters` and `globalParameters` and `_defaultProblemType` are nominally disjoint, but the plan does not assert this. Concretely, `ISA` appears in `globalParameters` (`GlobalParameters.py:146`) and also as a Solution parameter (added at `assignGlobalParameters` line 719: `validParameters["ISA"] = [IsaVersion(0,0,0), *isaList]`). The `_skipTypeCheck` set at `Solution.py:193` already lists `"ISA"`, which suggests the disjointness has been broken before. `CodeObjectVersion` appears in both `globalParameters` (`GlobalParameters.py:270`) and the Solution flow (`Solution.py:505`).
**Evidence:** referenced lines.
**Recommendation:** Add the property test from §7 ("every key in `globalParameters`, `_defaultProblemType`, `validParameters`, `defaultAnalysisParameters` appears in exactly one expected-types map or in a skip set") and promote it from "optional" to required. Decide and document where each collision is owned.

---

## 2. Migration / rollout risk

### Finding 2.1 — "Strict + fix in one PR" is not atomic on a per-validator basis

**Severity:** MAJOR
**Claim:** §6 says "the extended validators run in strict (raise) mode from the first commit" and "every input YAML in the tree is fixed in the same PR." But §8 spells out the steps as separate commits (Step 1 cleans the tree; Steps 3-7 flip validators). Within the PR, after Step 1 but before Step 3, the tree is fine; after Step 3 but before all yamls fixed for it, the tree may not be — and if a reviewer bisects through the PR (or if the PR is split during review), parts of the history are red.
**Evidence:** §6 vs §8 are in tension.
**Recommendation:** Either squash the entire PR into one commit (no commit ever fails CI) or invert step ordering: validators added warn-only first (or behind env flag), then tree-fix, then flip the flag default — but the plan rejects this for the right reasons. Squash-and-merge is the clean answer; state it explicitly.

### Finding 2.2 — Escape-hatch env var: where does it die?

**Severity:** MINOR
**Claim:** §6 introduces `TENSILE_STRICT_TYPE_CHECK={strict,warn,off}` with `strict` as the default. It does not say when this knob is removed. A permanent `off` switch is a permanent loophole; absent a removal date, downstream consumers will rely on it and the design rule erodes.
**Evidence:** §6 "Opt-out" paragraph; no removal step in §8.
**Recommendation:** Either declare the env var temporary with a removal date (e.g. one release cycle after the PR merges) or drop the `off` value and keep only `warn` for transitional use. Putting the answer in §8 as a final cleanup step makes the intent explicit.

### Finding 2.3 — `StrictTypeLoader` is an unwritten precondition

**Severity:** MINOR
**Claim:** The whole plan assumes that `0`/`1` parsed from YAML stay as `int` and `true`/`false` stay as `bool`. This only holds because `LibraryIO.read` defaults to `StrictTypeLoader` (`LibraryIO.py:345-358`). `BenchmarkSplitter.py:42` uses bare `yaml.safe_load` — bool/int collapse happens there. The plan doesn't mention `StrictTypeLoader` or audit non-`LibraryIO` readers.
**Evidence:** `LibraryIO.py:81-103` defines `StrictTypeLoader`; `BenchmarkSplitter.py:42` bypasses it.
**Recommendation:** Add a precondition section: "All YAML reads go through `StrictTypeLoader`." Audit and convert offending readers; file blockers for any that can't be converted.

---

## 3. Correctness of the validator design

### Finding 3.1 — bool/int trap consistently handled in *new* sites, but the rule must propagate

**Severity:** MINOR
**Claim:** Existing collector sites at `Solution.py:273-274`, `Problem.py:785-786` use `type() in expectedTypeSet`. The plan's §5 reaffirms the rule. The risk is in derived rules added for the new sites:
- `assignGlobalParameters` extension: §3 says `expected = {type(default_value)}`. Correct.
- `LibraryLogic.main` extension: §3 says same. Correct.
- `InternalSupportParams` extension: §3 says same. Correct.
- The `LibraryType` enum check is set-membership of strings; no type interplay.
- The `SolutionImportanceMin` range check is numeric comparison. Risk: if the validator allows `int` for a float-defaulted parameter (because `type(0.01) is float` but a user could pass `0` and Python's `0 < 0.5 < 1` works for ints too), the membership check is silently strict — `type(0) is float` is `False`. The plan never says whether `int` should be accepted where `float` is expected.
**Evidence:** §3 "globalParameters (single defaults)" rule; absence of int-for-float discussion.
**Recommendation:** Decide once and write it down: are numeric defaults accepting their sibling numeric type (`int` accepts `float` and vice versa), or strictly the default's type? `GlobalReadPerMfma` is one such case (`fix_yaml_types.py:65-67`). The current `_getExpectedTypes` union over allowed values handles this naturally for `validParameters`-typed parameters but NOT for `defaultAnalysisParameters`-typed ones.

### Finding 3.2 — Sentinel `-1` rule is undertested

**Severity:** MINOR
**Claim:** §5 names `LocalWritePerMfma: [-1]` as the canonical sentinel case. Spot-check `ValidParameters.py:371` confirms the allowed values are `[i/100 for i in range(1, 3200)] + [-1]` → expected type set `{float, int}`. But other params have similar sentinel patterns (`MaxVgprNumber`, `LdsBlockSizePerPad*`, `LdsPad*`), and none are enumerated in the test plan.
**Evidence:** `ValidParameters.py:371` and surrounding sentinel-bearing entries; `GlobalParameters.py:399-410` shows several `[-1]` defaults.
**Recommendation:** Enumerate sentinel-bearing parameters explicitly in §3 (not just `LocalWritePerMfma`) and add at least one test per *class* of sentinel.

### Finding 3.3 — `None`-default handling: skip set is incomplete

**Severity:** MAJOR
**Claim:** §3 says "skip entries whose default is `None`." This applies to e.g. `CmakeCxxCompiler` (`GlobalParameters.py:690`), `RocProfCounter` (`:326`), `ClientExecutionLockPath` (`:250-252`), `ROCmSMIPath` (`:261`). But the plan handles `None` only by skipping — it never validates these keys at all. A user typing `RocProfCounter: 42` (an int) gets through silently. The semantics matter: `None` here means "string, default unset" in most cases, not "type unknown." A `Union[NoneType, str]` declaration would catch the int.
**Evidence:** referenced lines; `GlobalParameters.py:261` comment says `# /opt/rocm/bin/rocm-smi` — i.e., the value is a str path. Same for `ClientExecutionLockPath`. `RocProfCounter` is `None` or a counter spec.
**Recommendation:** Per-key annotation override map. The plan's §3 already invites "Special cases"; just write the table now. Skipping `None` defaults wholesale is a coverage gap.

### Finding 3.4 — Tuple-default open question is a phantom

**Severity:** MINOR (process)
**Claim:** §9.5 and §3 flag "tuple-defaulted GlobalParameters" as unresolved. There are none. Every `globalParameters["X"] = (\n    value  # comment\n)` line in `GlobalParameters.py` between lines 44 and 326 has a single value with no comma inside the parens.
**Evidence:** confirmed by spot-check of every `( ... )` line in `GlobalParameters.py:40-340` — no trailing comma, no second element. Specifically: lines 44, 47, 50, 55, 60, 63, 66, 69, 74, 86, 89, 101, 104, 107, 111, 114, 118, 121, 151, 200, 206, 223, 226, 229, 236, 239, 250, 253, 256, 295, 299, 303, 312, 318.
**Recommendation:** Delete the open question. Note in §3 that no tuple defaults exist; no handling needed.

### Finding 3.5 — Free-form list params do fall out naturally (spot-check passes)

**Severity:** NIT
**Claim:** §1 claims `MatrixInstruction`, `WorkGroup`, `ThreadTile`, etc. are skipped because their `validParameters` entry is `-1`. Spot-checked at `ValidParameters.py:1086-1097`:
- `MatrixInstruction: -1` ✓
- `WorkGroup: -1` ✓
- `ThreadTile: -1` ✓
- `MIWaveGroup: -1`, `MIWaveTile: -1` ✓

`SpaceFillingAlgo` and `SFCWGM` are handled by their own checkers (`ValidParameters.py:1099-1128`). The `[[]]` placeholder pattern claim in §3 deserves verification; I did not see `[[]]` in `validParameters` — the placeholder for `MatrixInstruction` etc. is `-1`, not `[[]]`. Either the plan is wrong about `[[]]` or there are entries I missed.

**Recommendation:** Verify the `[[]]` pattern claim with a single grep before implementation. If it doesn't exist, remove the `[[]]` skip rule.

---

## 4. The "redundant warn-only validator" question

### Finding 4.1 — Keeping `validateParameterTypes` warn-only on input path is a tactical deferral

**Severity:** MAJOR
**Claim:** §1 in-scope discussion of Solution post-construction `validateParameterTypes`:
> "Leaving it warn-mode in v1 avoids threading a strict flag through `Solution.__init__` and avoids changing library-logic behaviour."

The first half is exactly the "less work" justification the user's "no tactical fixes" rule prohibits. Option (a) in §9.1 (a `validateTypes: bool = True` kwarg) is principled and 4 lines of code. Option (b) (split constructors) is also principled. The plan rejects neither and just defers the decision.

The second half — "avoids changing library-logic behaviour" — is real, but it conflates two things: keeping the validator *callable* in collector mode (for the library-logic path) and keeping it *invoked* in collector mode on the input-YAML path. These can be split: the validator can keep its collector behaviour while the input-YAML callers (`BenchmarkProblems.py:179, 246`) stop invoking it (or invoke it with a flag that flips to raise). Step 8 in §8 does eventually argue for this, but conditions it on "verification first" — which is fine — and then on user decision in §9.1 — which is the deferral.

**Evidence:** §1 lines on `validateParameterTypes`; §9.1 options.
**Recommendation:** Resolve §9.1 in the plan. The user has been clear: lead with the principled option. Pick (a), spell it out in Step 8, and remove the §9.1 deferral.

### Finding 4.2 — `printTypeMismatchSummary` deletion is conditioned but the condition is checkable now

**Severity:** MINOR
**Claim:** §6 step 8 says "if verification shows it can never fire on the input-YAML path … delete the call." Verification: after Step 4 promotes `validateProblemTypeParameterTypes` to raise on the first per-instance mismatch, and Step 8 either drops or flag-disables the Solution-side validator, the collector dict for the input-YAML path is never populated. The summary call at `BenchmarkProblems.py:762` becomes dead. This is decidable from the design now; the "verification first" framing defers an answer the planner already has.
**Evidence:** flow as described.
**Recommendation:** State the conclusion in Step 8 instead of conditioning it.

---

## 5. Test strategy

### Finding 5.1 — Each extension point has tests; integration test is weak

**Severity:** MINOR
**Claim:** §7 lists per-validator unit tests. The integration test described ("load a known-good real test config; mutate one key") only exercises one section per run. A cross-section integration test — a YAML with a bad `GlobalParameter` AND a bad `ProblemType` key — would assert the correct validator fires first (the one closest to the load) and the error message is the GlobalParameters one. This kind of test catches future refactors that reorder validators.
**Evidence:** §7 "Integration test" paragraph.
**Recommendation:** Add a "validator ordering" integration test: prove `assignGlobalParameters` fires before `ProblemType.__init__`, before `checkParametersAreValid`, before `LibraryLogic.main`'s merge check, on the standard `Tensile.Tensile()` path.

### Finding 5.2 — "Corpus clean" regression gate not mentioned

**Severity:** MAJOR
**Claim:** §7 lists no test that walks every YAML in `tensilelite/Tensile/Tests/common/` and `Tensile/Configs/` and asserts that *none* of them now produces a `ConfigTypeError`. Without it, the tree-cleanliness invariant after Step 1 is unenforced — any subsequent commit can re-introduce a bad value and only fail when the specific config is selected by a downstream test that happens to hit the affected key.
**Evidence:** §7 lacks the gate.
**Recommendation:** Add a corpus-clean test: parametrize over every YAML under the in-tree YAML roots, load via the full `Tensile.Tensile()` entry path on a `tmp_path` copy (or just the load+validate prefix), assert no exception. Mandatory in CI.

### Finding 5.3 — Bool/int trap unit-only

**Severity:** MINOR
**Claim:** §5's mandatory tests are unit-level. None of them assert that an integration run (e.g. running BenchmarkProblems on a config with `BoundsCheck: False`) fires the trap and not some other validator. Spurious-trap risk is real because `bool` is a subclass of `int`.
**Evidence:** §7 test enumeration.
**Recommendation:** Add one integration test that runs a config with a bool-where-int and asserts the failure path identifies the right validator, the right key, and the right key path.

---

## 6. Open questions that aren't really open

### §9.1 — `validateParameterTypes` conditioning

**Severity:** MAJOR (process)
**Verdict:** Decidable. Pick (a), the kwarg threading. Don't ask the user.

### §9.2 — Promote unknown-global-key to raise

**Severity:** MINOR (process)
**Verdict:** Decidable — the user's "no tactical fixes" rule says raise. Out-of-tree YAMLs that carry extra keys are exactly the silent-corruption class this work addresses. The escape-hatch env var covers them.

### §9.3 — `LibraryType` allowed enum

**Severity:** MAJOR (correctness)
**Verdict:** Decidable. The code at `LibraryIO.py:651-656` shows `LibraryType` is a distance-metric label by design (anything not in `{"FreeSize","Prediction"}` becomes the distance label of a `Matching` library). Drop the enum check. The plan's §9.3 hedge ("if the set turns out to be open-ended, drop the enum check") is the right instinct; the answer is just "yes, drop it."

### §9.4 — `InternalSupportParams` fold vs sibling

**Severity:** MINOR (process)
**Verdict:** Decidable. Plan already recommends fold. State it.

### §9.5 — Tuple-typed globalParameters

**Severity:** MAJOR (process)
**Verdict:** Phantom. No tuples exist. Delete the question.

### §9.6 — Env var vs CLI flag

**Severity:** NIT
**Verdict:** Plan's recommendation is fine. State it.

### §9.7 — One PR vs split

**Severity:** MINOR
**Verdict:** State the recommendation. The user's "squash at merge" memory is the relevant constraint here — within a development branch, the steps are reviewable as separate commits, but the final merge is one commit. That resolves the bisect concern.

### §9.8 — Cached YAML path

**Severity:** MINOR
**Verdict:** The cache path at `BenchmarkProblems.py:107` reads via `LibraryIO.read` and never reaches `assignGlobalParameters`/`checkParametersAreValid` (it just compares to a previously-cached config). Decidable now — no special-case needed.

---

## 7. Scope creep / scope gaps

### Finding 7.1 — Dead/legacy parameter knobs surfaced but not filed as blockers

**Severity:** MINOR
**Claim:** `LibraryLogic.py:174` reads `inputParameters["SmoothOutliers"]` and `:849, 858, 860` read `BranchPenalty` — neither is in `defaultAnalysisParameters`. The `assignParameterWithDefault` loop never populates them, so the code paths `KeyError` if reached. Per the user's "no deferred discoveries" rule, this should be filed as a blocker on Step 6 (LibraryLogic validation) — adding `LibraryType` strictness while leaving `KeyError`-prone code paths in the same file untouched is incoherent.
**Evidence:** `LibraryLogic.py:174, 849, 858, 860` reference keys not in `defaultAnalysisParameters` (`GlobalParameters.py:588-594`).
**Recommendation:** File the dead/legacy keys as a separate blocker task (delete the code paths or add the keys to `defaultAnalysisParameters` with sensible defaults).

### Finding 7.2 — `TensileClientConfig.py:172` arity bug surfaced but not filed

**Severity:** MAJOR
**Claim:** `ProblemType(problemDict)` at `TensileClientConfig.py:172` is missing the `printIndexAssignmentInfo: bool` positional argument required by the current signature (`Problem.py:809`). This is a latent crash, and Step 4 makes it worse by raising additional errors before that point. Either this code path is dead (delete it) or it's broken (fix it). File as a blocker.
**Evidence:** `Problem.py:809` requires `printIndexAssignmentInfo`; `TensileClientConfig.py:172` doesn't pass it.

### Finding 7.3 — `validateParameterTypes` deferral is the tactical-fix instance

**Severity:** MAJOR (already covered in §4.1)
**Recommendation:** See §4.1.

---

## 8. Anything else

### Finding 8.1 — `LibraryLogic.main()` line number/function name slip

**Severity:** NIT
**Claim:** §1 and §2 refer to "LibraryLogic.main() merge loop (`LibraryLogic.py:1444-1447`)." Lines 1444-1447 are in `generateLogic` (defined at `LibraryLogic.py:1427`), not `main` (defined at `:1567`). `main` just delegates to `generateLogic`. The plan's line numbers are right, the function name is wrong.
**Evidence:** `LibraryLogic.py:1427` (`generateLogic`) vs `:1567` (`main`); merge loop at `:1445-1447`.
**Recommendation:** s/main()/generateLogic()/ where appropriate.

### Finding 8.2 — `_typeMismatchCollector` is module-global; multiprocess merge is fragile

**Severity:** MINOR
**Claim:** The collector at `Solution.py:210` is a module dict. Workers spawned via `ParallelMap2` (joblib) populate their own copies and the merge functions (`getTypeMismatchCollector`, `mergeTypeMismatchCollector`) marshal across processes. The plan keeps this machinery alive for the library-logic path. After Step 4 makes ProblemType validation raise inside the same workers, the collector is half-used (Solution-side still populates; ProblemType-side raises). This split is workable but subtle and the plan doesn't call it out.
**Evidence:** `Solution.py:206-247`.
**Recommendation:** Mention in Step 4 that the collector is no longer the primary signal on the input-YAML path; it's belt-and-braces for the library-logic path until v2.

### Finding 8.3 — `BenchmarkSplitter.py:42` bypasses `StrictTypeLoader`

**Severity:** MAJOR
**Claim:** `BenchmarkSplitter.py:42` uses `yaml.safe_load(f)` directly. Anything it loads goes through PyYAML's default bool resolver, so `0`/`1` become `False`/`True`. If that loaded data ever reaches a validator, the bool/int mismatch is invisible.
**Evidence:** grep hit cited above.
**Recommendation:** Switch to `LibraryIO.read` (or `yaml.load(..., StrictTypeLoader)`) in the same PR. File as Step 0 prerequisite if it changes the surface area of the validators.

### Finding 8.4 — Plan doesn't audit `MasterSolutionLibrary` / `SolutionLibrary` consumers downstream of `LibraryLogic`

**Severity:** MINOR
**Claim:** §1 notes hipBLASLt consumes YAMLs downstream. Per the hipBLASLt CLAUDE.md, the C++ runtime reads `.hsaco`/`.co` and library logic YAMLs at hipBLASLt-call time. The plan validates the *input* YAML that produces library-logic output but never confirms that strict validation on input YAMLs doesn't itself produce a library-logic output whose consumer expects the older loose typing. This is unlikely (the same `fix_yaml_types.py` proves the consumer wants strict), but worth a one-line confirmation.
**Recommendation:** Add a line under Migration noting that hipBLASLt-side consumers benefit (msgpack `std::bad_cast` goes away) and there's no behavioural risk for them.

### Finding 8.5 — `[[]]` placeholder claim in §3 not verified

**Severity:** MINOR
**Claim:** §3 says some entries' `validParameters` value is `[[]]` ("placeholder for free-form list payloads — e.g. `MatrixInstruction`, `SpaceFillingAlgo`, `SFCWGM`, `ClusterDim`"). Spot-checked `ValidParameters.py:1086-1097` — all those keys use `-1`, not `[[]]`. I did not find `[[]]` in `validParameters` at all. The skip rule in `_getExpectedTypes` (`Solution.py:178-184`) only handles `-1` and `len(list) > 0`; it doesn't explicitly handle `[[]]`. If `[[]]` exists somewhere I missed, it's covered by the `len(list) > 0` check (which would set the type set to `{list}` — almost certainly not what's intended).
**Evidence:** `Solution.py:178-184`, `ValidParameters.py:1086-1097`.
**Recommendation:** Drop the `[[]]` mention from §3, or grep and cite a concrete line.

### Finding 8.6 — `InternalSupportParams` is dict-typed, not list-typed

**Severity:** MAJOR
**Claim:** §1 says `checkParametersAreValid` "skips it with an early return (`ValidParameters.py:1136-1137`). Add an in-place check against `defaultInternalSupportParams`." But `checkParametersAreValid` takes `(name, values)` where `values` is a list (each invocation iterates `for value in values`). `InternalSupportParams` arrives as a dict (`BenchmarkStructs.py:184: self.internalSupportParams = getNonNoneFromConfig("InternalSupportParams", {})`), passed into `checkParametersAreValid` as `(\"InternalSupportParams\", configValue)` — but actually it's never passed in today; the early return at line 1136 is dead code from a prior call path. Step 7's fold-into-`checkParametersAreValid` will need to either (a) detect `name == "InternalSupportParams"` and switch to dict-iteration mode (changing the function's contract) or (b) leave the early return and add a sibling validator (the §9.4 (b) option the plan deprecates).
**Evidence:** `BenchmarkStructs.py:184, 227-235` show what gets passed in.
**Recommendation:** Pick (b) in §9.4 — sibling validator — for `InternalSupportParams`. The fold breaks the function's `(name, list)` contract. The plan's recommendation here is wrong.

### Finding 8.7 — `defaultInternalSupportParams` semantics: `KernArgsVersion`

**Severity:** MINOR
**Claim:** `defaultInternalSupportParams = {"KernArgsVersion": 2, "SupportUserGSU": True, ...}` — `KernArgsVersion` is an int. The "type check against `type(default)`" rule gives `{int}`. Fine, but is this version field allowed to be a string in YAML? CustomKernels are user-authored files. Worth a one-line confirmation that no in-tree custom kernel sets it to a string.
**Evidence:** `GlobalParameters.py:381-393`.
**Recommendation:** Run `fix_yaml_types.py` against the custom kernels directory before flipping strict mode.
