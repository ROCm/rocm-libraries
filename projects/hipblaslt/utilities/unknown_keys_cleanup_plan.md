# Unknown GlobalParameters Keys — Findings & Cleanup Plan

Audit-flagged blocker B-AUD-1: 15 unknown global keys polluting in-tree YAMLs would crash `Tensile.Tensile()` under the new strict gate (commit `0ce0829c`, Step 5 of the input-yaml validation work). 12 keys investigated in worktree-isolated parallel agents; verdicts and cleanup actions below.

Two of the 15 were already-benign ignoreKeys (`Device`, `PrintLevel`); `NewClient` was investigated and cleaned separately (commit `380fa11124d8`).

## Verdict matrix

| Key | YAMLs | Verdict | Mechanism | Action |
|---|---|---|---|---|
| `NewClient` | 183 | C | Removed `dc2c963c` Mar 2025; YAMLs copy-pasted forward | **Cleaned in `380fa11124d8`** |
| `PrintSolutionRejectionReason` | 186 | **A** | Live via `DebugConfig.printSolutionRejectionReason` (`Types.py:98-99`); not in `ignoreKeys` | Add to `ignoreKeys` |
| `ForceGenerateKernel` | 9 | **A** | Live via `DebugConfig.forceGenerateKernel` (`Types.py:96-97`, `KernelWriter.py:6649, 9632`); not in `ignoreKeys` | Add to `ignoreKeys`; fix 2 type mismatches (see below) |
| `MaxLDS` | 44 | **A** | Live *solution* parameter in `defaultBenchmarkCommonParameters` (`GlobalParameters.py:412`); placed in wrong YAML block | **Delete from `GlobalParameters:` blocks** (correct placement is `ForkParameters:`; one file already does this) |
| `PrintIndexAssignments` | 1 | C | Renamed to `PrintIndexAssignmentInfo` in `dc2c963c`; successor is itself a DebugConfig key not in `ignoreKeys` | **Rename** in `sgemm_xf32_asm.yaml` to `PrintIndexAssignmentInfo: true`; ensure successor also covered by `ignoreKeys` fix |
| `DeviceLDS` | 45 | C | Removed `7770c97e13a3` May 2025 (moved to `archCaps`); 13+ copy-paste re-introductions through 2026-06-03 | Delete from YAMLs |
| `ROCmAgentEnumeratorPath` | 8 | C | Removed `4a5aa3cb7fbe` Mar 2026 (emulator revert); re-added by post-revert gfx1250 PRs | Delete from YAMLs |
| `AMDGPUArchPath` | 1 | C | Removed `dc2c963c` Mar 2025; the lone YAML occurrence was authored Mar 2026 with a wrong value (`rocm_agent_enumerator`, not amdgpu-arch path) | Delete from YAML |
| `MergeFiles` | 16 | C | Removed `2d2e1496a9d7` Jan 2025; cleanup commit missed `largeMT.yaml`; 4 later commits re-introduced via copy-paste | Delete from YAMLs (incl. 4 commented-out) |
| `MaxFileName` | 18 | C | Replaced by hardcoded `MAX_FILENAME_LENGTH = 64` in `d170037bd4fe` Feb 2025 | Delete from YAMLs **+ delete dead deprecation warning at `Tensile/Tensile.py:664-665`** |
| `MinKForGSU` | 102 | C | Replaced by hardcoded `MIN_K_FOR_GSU = 32` in `dc2c963c` Mar 2025 | Delete from YAMLs; **see test-correctness concern below** |
| `DataInitTypeeScaleE` | 2 | B | Typo of `DataInitTypeScaleE`; both spellings have no implementation | Delete from YAMLs |
| `UseGPUTimer` | 5 | B | Never registered; every occurrence sits alongside the correct `KernelTime` key in the same YAML | Delete from YAMLs |

**Verdict legend:** A = live reader; B = dead-on-arrival (never had a reader); C = once-live, now dead.

## The two recurring failure modes

### Mode 1 — DebugConfig routing bug (verdict A, 2 keys + 1 implied)

Commit `dc2c963c` (Mar 2025, "Remove global variables") moved several keys out of `globalParameters` and into a new `DebugConfig` named-tuple built by `makeDebugConfig()` in `Tensile/Common/Types.py`. The function reads the keys *directly out of the raw `GlobalParameters:` config dict*, bypassing the registry. The migration was correct on the reader side, but **`ignoreKeys` in `assignGlobalParameters` was never updated** to skip the migrated keys. Pre-strict-gate, the unknown-key path was a `printWarning`, so the validator's blind spot was invisible.

Affected keys: `PrintSolutionRejectionReason`, `ForceGenerateKernel`, `PrintIndexAssignmentInfo` (the renamed successor of `PrintIndexAssignments`). Likely also: `EnableAsserts`, `DebugKernel`, `SplitGSU` — any other field on `DebugConfig` that `makeDebugConfig()` reads from the raw dict. Audit all `DebugConfig` fields when implementing the fix.

### Mode 2 — copy-paste cargo cult (verdict C, 5 keys)

When a registry entry + reader are deleted, YAML cleanup is often partial or skipped. Subsequent PRs then copy from stale fixtures and re-introduce the dead keys. Documented re-introductions after removal:

- `DeviceLDS`: removed May 2025; 13+ copy-paste re-adds through 2026-06-03
- `ROCmAgentEnumeratorPath`: removed Mar 2026; re-added by 4 gfx1250 PRs in April-May 2026
- `AMDGPUArchPath`: removed Mar 2025; the sole occurrence was authored Mar 2026
- `MergeFiles`: removed Jan 2025; 4 commits re-introduced Apr-Jun 2026
- `MinKForGSU`: removed Mar 2025; 102 stale YAMLs

The strict gate is exactly what catches this — surfacing copy-paste at author-time instead of letting silently-ignored config rot the corpus. **This is an argument for keeping the strict gate strong, not relaxing it.**

## Cleanup PR shape

Three logical units. Can be one PR with three commits or three separate PRs depending on review preference.

### Commit/PR 1 — Validator fix: extend `ignoreKeys` for the DebugConfig family

- **File:** `Tensile/Common/GlobalParameters.py` (the `ignoreKeys` list near `assignGlobalParameters`)
- **Action:** add every key consumed by `makeDebugConfig()` in `Tensile/Common/Types.py`. At minimum: `PrintSolutionRejectionReason`, `ForceGenerateKernel`, `PrintIndexAssignmentInfo`. Audit `DebugConfig` for the full list before committing.
- **Tests:** add a property test that every key `makeDebugConfig()` reads from the raw config dict is in `ignoreKeys`. This prevents the next DebugConfig addition from re-introducing the same blind spot.
- **Rationale:** Mode 1 above. These keys are live; the validator is wrong, not the YAMLs.

### Commit/PR 2 — YAML corpus cleanup

Single commit covering all the (B) and (C) deletions plus the one (A)-wrong-block and the one rename:

| Action | Scope | Notes |
|---|---|---|
| Delete `DeviceLDS:` | 45 YAMLs | Don't confuse with live `archCaps["DeviceLDS"]` in rocisa |
| Delete `ROCmAgentEnumeratorPath:` | 8 YAMLs | Tool-path now owned by `Toolchain/Validators.py` |
| Delete `AMDGPUArchPath:` | 1 YAML (`streamk/gfx1250/sk_mxf8f4gemm_quick.yaml:32`) | Value was also semantically wrong |
| Delete `MergeFiles:` | 16 active + 4 commented YAMLs | |
| Delete `MaxFileName:` | 18 YAMLs | **Also delete dead deprecation warning at `Tensile/Tensile.py:664-665`** in same commit (delete-hacks-immediately) |
| Delete `MinKForGSU:` | 102 YAMLs | See test-correctness concern below |
| Delete `DataInitTypeeScaleE:` | 2 YAMLs (`gemm/fp8nfp16mix_hfp8ns.yaml:18`, `gemm/gfx950/f8f16mix_f8s.yaml:18`) | |
| Delete `UseGPUTimer:` | 5 YAMLs (+ commented variants) | Sibling `KernelTime:` already does the work |
| Move `MaxLDS:` from `GlobalParameters:` to `ForkParameters:` | 44 YAMLs | OR delete and rely on auto-detection (`-1` default → `archCaps["DeviceLDS"]`); pick one |
| Rename `PrintIndexAssignments:` → `PrintIndexAssignmentInfo: true` | 1 YAML (`sgemm_xf32_asm.yaml`) | Requires Commit/PR 1 to land first or in same PR |
| Fix `general_wgm.yaml` `ForceGenerateKernel: 1` → `true` | 1 YAML | Type mismatch surfaced by ForceGenerateKernel investigator |
| Fix `tdm_multicast_gfx1250.yaml` `ForceGenerateKernel: [True]` → `true` | 1 YAML | Same |

### Commit/PR 3 — `MinKForGSU` test-correctness decision (separate)

This is a real semantic question, not janitorial:

- The YAML override `MinKForGSU: 1` was *meaningful* before the March 2025 removal — it lowered the GSU eligibility threshold below the default 32 so small-K test problems would actually exercise GSU code paths.
- With the hardcoded `MIN_K_FOR_GSU = 32` in `Contractions.py:39`, any test that authored `MinKForGSU: 1` to exercise sub-32-K GSU is now silently NOT exercising what its author intended.
- 102 YAMLs are affected.
- Possible outcomes: (a) tests still pass coincidentally because 32 is low enough → safe to delete; (b) tests are silently miscalibrated → either lower `MIN_K_FOR_GSU` or restructure the predicate; (c) the GSU code path is no longer reachable from small-K tests at all, in which case the test coverage gap should be filed and addressed.

**Recommendation:** investigate before deleting. A separate worktree agent could enumerate which of the 102 tests pass *despite* the change vs. *because* the GSU branch is no longer hit.

## Bonus findings (already fixed or to be fixed in passing)

- **`BenchmarkStructs.py:305` NameError** — introduced by Step 3 (`e27dbb135df1`) of the original validation implementation. The refactor removed the `configParams` dict but left a stray `params.update(configParams)`. The error was masked by `NewClient` ConfigTypeError firing earlier in the stack. **Already fixed in commit `380fa11124d8`** as part of the NewClient cleanup. The commit message doesn't mention it; consider amending or noting in the PR description.
- **Dead deprecation warning at `Tensile/Tensile.py:664-665`** — checks `"MaxFileName" in config` against the wrong dict level (top-level instead of `config["GlobalParameters"]`) AND runs after `assignGlobalParameters` which now raises first. Doubly inert. Delete in the `MaxFileName` cleanup commit.
- **`DebugConfig` key audit needed** before the Commit/PR 1 fix lands — confirm the full set of keys `makeDebugConfig()` consumes so the `ignoreKeys` extension is exhaustive, not just the three surfaced here.
- **Audit/implementer blind spots** that the verifier missed: the corpus-clean test's "Be permissive" shortcut (audit's B-AUD-2) caused the audit to underestimate the corpus pollution. With B-AUD-2 fixed, the test would have surfaced all 15 unknown keys directly. The fix for B-AUD-2 — extracting the type-check logic into a function both `assignGlobalParameters` and the corpus test call — should land alongside Commit/PR 1 so the test actually exercises the validator going forward.

## Open question for the user

1. **`MinKForGSU` test correctness.** Bulk-delete the 102 occurrences and accept the silent miscalibration, or investigate per-test first? (Recommendation: investigate.)
2. **`MaxLDS` migration.** Move the 44 occurrences to `ForkParameters:` (preserves the test author's evident intent) or delete and rely on auto-detection (simpler, but loses the explicit override for any test that genuinely wanted a specific cap)?
3. **`NewClient` commit `380fa11124d8` rolls in the BenchmarkStructs NameError fix** despite the commit message saying "remove dead NewClient config". Amend message, split commit, or leave?
