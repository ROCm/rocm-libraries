# XYV — `test_tf32_4x4_tn_capture_shape` flake investigation

Bead: `rocm-libraries-xyv` (P0)
Investigator: assistant (worktree `/home/alvasile/rocm-libraries/.worktrees/xyv`, branch `xyv-investigate`)
Worktree base: forked from `users/alvasile/validator_long_term_plans` at `a2287d8d12` (br4.7)

## TL;DR

**Flake could not be reproduced at the current worktree tip (a2287d8d12) nor at the bead's nominal repro commit (`18cbf1e59b`, br4.2).** All variants of the bead-described invocations now produce `602 passed, 2 skipped, 1 xfailed` — including the variant the bead claimed was failing.

The most likely root cause, identified by code inspection, is **shared mutable state between the conftest's session-scope `isaInfoMap` and a module-level `makeIsaInfoMap(SUPPORTED_ISA, ...)` call in `test_MatrixInstructionConversion.py:37`** (executed at pytest collection time). Empirical probing of the rocisa singleton showed that `asmCaps` dicts returned by separate `makeIsaInfoMap` calls are NOT aliased today, but the module-level call still pre-pollutes the rocisa singleton state and re-invokes init for every entry in `SUPPORTED_ISA`. If aliasing is ever reintroduced (or if any code mutates an entry in `IsaInfo.asmCaps`), the order/cwd-sensitive collection would expose it as exactly the symptom the bead describes.

**This bead's scope: investigation only.** Fix is deferred to a follow-up bead because the flake is currently latent and a defensive fix here would land speculatively.

## Verified invocations (current state)

All tested at `a2287d8d12` (worktree tip) AND at `18cbf1e59b` (bead-claimed repro commit). Each ran to completion:

| Cmd | Cwd | Result |
|---|---|---|
| `pytest -q .worktrees/xyv/projects/hipblaslt/tensilelite/Tensile/Tests/unit -x` | `/home/alvasile/rocm-libraries` | 602 passed |
| `pytest -q /home/alvasile/rocm-libraries/.worktrees/xyv/projects/hipblaslt/tensilelite/Tensile/Tests/unit` | `/home/alvasile/rocm-libraries` | 602 passed |
| `pytest -q rocm-libraries/.worktrees/xyv/projects/hipblaslt/tensilelite/Tensile/Tests/unit -x` | `/home/alvasile` | 602 passed |
| `pytest -q .worktrees/xyv/.../unit -x` | `/home/alvasile/rocm-libraries` | 602 passed |
| `pytest -q Tensile/Tests/unit -x` | `.../xyv/projects/hipblaslt/tensilelite` | 602 passed |
| `pytest -q .../unit -x` | `/tmp` | 602 passed |
| `pytest -q .../unit/test_ScheduleCapture.py::TestRealKernelCapture::test_tf32_4x4_tn_capture_shape` | any | 1 passed |

Bead-described outcome (108 passed + 1 xfailed + 1 failed at position 110 with `-x`) matches the *position* of `test_tf32_4x4_tn_capture_shape` in collection order (verified — it is item 110).

## Collection-order diff

`pytest -v --collect-only` was captured for both the failing-form invocation (relative path with `-x`) and the passing-form invocation (absolute path no `-x`). The diff is **empty** apart from the timing line:

    607c607
    < 605 tests collected in 76.17s (0:01:16)
    ---
    > 605 tests collected in 78.22s (0:01:18)

So collection order is identical between the two invocations the bead distinguishes. **The cause is NOT collection.** Per the bead's own hypothesis matrix, this points to per-test/module-import state mutation.

## Suspect: rocisa singleton + module-level init in test_MatrixInstructionConversion.py

`Tensile/Tests/unit/test_MatrixInstructionConversion.py:37`:

    isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, cxxCompiler)

This runs at **module import time**, i.e. during pytest collection — for every collected pytest worker, regardless of whether any test in this file is selected. `SUPPORTED_ISA` is the full list of 22 ISAs. `makeIsaInfoMap` does, per ISA:

    ti = rocisa.rocIsa.getInstance()  # process-wide singleton
    ti.init(v, cxxCompiler, False)
    isaInfoMap[v] = IsaInfo(asmCaps=..., archCaps=..., ...)

After import, the rocisa singleton was last `init`'d to `IsaVersion(12, 5, 0)` (last entry in `SUPPORTED_ISA`).

Then `Tensile/Tests/unit/conftest.py:48`:

    isaInfoMap = makeIsaInfoMap([IsaVersion(9, 5, 0)], compiler)

— this re-inits the singleton to gfx950. The session-scoped fixture caches this map.

`test_tf32_4x4_tn_capture_shape` builds a `Solution` via `_make_solution` in `cms_test_utils.py:636`, which calls into `Solution.__init__` at `Tensile/SolutionStructs/Solution.py:374-375`:

    isa = IsaVersion(isa[0], isa[1], isa[2])
    assert self.isaInfoMap[isa].asmCaps["SupportedISA"]

For `isa = IsaVersion(9,5,0)` against the conftest's gfx950-only map, `SupportedISA` should be `1` (verified via direct probe).

### Empirical probe of the singleton

    m1 = makeIsaInfoMap([IsaVersion(9,5,0)], cxx)
    asmcaps_950 = m1[IsaVersion(9,5,0)].asmCaps
    # SupportedISA = 1, id = 127267100492992
    m2 = makeIsaInfoMap([IsaVersion(8,0,3)], cxx)
    asmcaps_803 = m2[IsaVersion(8,0,3)].asmCaps
    # asmcaps_950 is asmcaps_803 -> False
    # SupportedISA(9,5,0) STILL = 1
    # SupportedISA(8,0,3) = 1

So **today** the dicts are not aliased — re-init returns a fresh dict — and the gfx950 entry survives a subsequent re-init unchanged. That explains why the failure does not currently reproduce.

If the rocisa C++ binding ever changes to share the dict (or if a future caller mutates `IsaInfo.asmCaps` in place), the symptom would re-emerge: `test_tf32_4x4_tn_capture_shape` would assert `SupportedISA` against a dict whose contents have been repointed at whatever ISA was init'd last.

The cwd/`-x` sensitivity in the bead's repro is consistent with this hypothesis if we assume *some* secondary differentiator (e.g. a worker spawning order in pytest-xdist, or a sys.path fragment that controls whether `test_MatrixInstructionConversion` even imports cleanly). Without a live failing run we cannot pin the secondary differentiator.

## Why the failure does not currently reproduce

Two plausible explanations, neither conclusively verified:

1. **The flake was real but transient at the moment the bead was filed.** The bead snapshot was taken on `users/alvasile/validator_long_term_plans` at `18cbf1e59b` with possible uncommitted local state (e.g. an earlier `.pyc` from a prior gfx1151-probing conftest still in `__pycache__`). After cache invalidation and a clean import path, the flake stops. We saw no failure at the same commit hash today.

2. **The aliasing condition was eliminated by an unrelated change between the original observation and today.** Most likely candidate: a rocisa rebuild that changed dict allocation behavior. The `Tensile/SolutionStructs/Solution.py` source has not changed since `3809cfb7af` per `git log --oneline` of that file.

Either way, the structural risk remains: `test_MatrixInstructionConversion.py` mutates global rocisa state at module-import time, and the test suite relies on the gfx950 entry of a session-scoped map remaining valid across that mutation.

## Files implicated

- `projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_MatrixInstructionConversion.py` (module-level `makeIsaInfoMap(SUPPORTED_ISA, ...)` at line 37)
- `projects/hipblaslt/tensilelite/Tensile/Tests/unit/conftest.py` (session-scope `isa_infrastructure` fixture, single-ISA gfx950 probe at line 48 — already documented as deliberate)
- `projects/hipblaslt/tensilelite/Tensile/Common/Capabilities.py` (rocisa singleton init in `makeIsaInfoMap`)
- `projects/hipblaslt/tensilelite/Tensile/SolutionStructs/Solution.py` line 375 (the assertion that fails)

## Proposed fix sketch (deferred — see below)

Defensive: refactor `test_MatrixInstructionConversion.py` to lazily build its `isaInfoMap` inside a session-scoped fixture (mirroring conftest), instead of at module import. This eliminates the import-time singleton mutation. Estimated change: ~10 lines, plus one fixture parameter on each test in that file.

Stronger: change `IsaInfo` to deep-copy its dicts in `__post_init__` so future aliasing accidents in the rocisa binding cannot leak. ~5 lines in `Tensile/Common/Types.py`.

Both are one-bead changes. Neither touches CMSValidator.py or ScheduleCapture.py (per xyv hard rule).

## Scope decision

- **In scope here (closing this bead):** the investigation report above, plus collection-order verification, plus singleton aliasing probe.
- **Deferred:** the actual fix. Without a live repro, landing a "fix" would be speculative — we cannot verify the fix actually closes the flake. Recommended path: file follow-up bead (P1, not P0, because flake is latent today) to apply the lazy-fixture refactor in `test_MatrixInstructionConversion.py` AND add a CI invariant that runs the unit suite from two cwds with `-x` and compares results. If the invariant ever flakes, the follow-up bead's fix can be validated.

## Hard-rule compliance

- No changes to `CMSValidator.py` or `ScheduleCapture.py` (validator surface is locked while br4.x lands). Verified by `git diff` — no source edits in this bead.
- No `git push`.
- No subagent spawning.
- Report is under 400 lines.
