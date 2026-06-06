# GPU-mock switch — standalone prerequisite PR (`--cpu-only` / `--mock-gpu`)

**What this is.** A small, self-contained, **source-changing** PR that makes the TensileLite
**client perf-run** path exercisable **without a GPU**, by adding one switch that returns
synthetic perf metrics and spoofs the system device-probes per architecture. It is the
prerequisite **P0.5** for the codegen-coverage campaign (`PLAN-CODEGEN-WORKFLOW.md`), but it
stands on its own and is independently useful (GPU-less CI).

> **Not part of the ADD-ONLY characterization branch, and not a fan-out workflow.** This is
> ordinary source work that needs human review and conventional landing. Treat it as
> single-agent / goal-file work (shared context, iterative, source edits). The coverage
> campaign stays strictly add-only and merely *consumes* this switch once it is merged.

---

## Why it's needed (and how narrow it really is)

The campaign metric is **whole-project line coverage**. Almost the entire pipeline is already
CPU-reachable — the GPU boundary is much smaller than it first appears:

| Component | GPU? | Notes |
| --- | --- | --- |
| Codegen emit (`KernelWriterAssembly`, `KernelWriter`, `Components/*`, `Asm*`) | **No** | Emits assembly *text*; proven CPU-only in P0. |
| Solution derivation (`_generateForkedSolutions`, `parseLibraryLogicFile`) | **No** | Pure `Solution` construction from params; uses the cached CPU `assembler`+`isaInfoMap`. |
| **TensileCreateLibrary** | **No** | A *cross-compiler* — host `amdclang++` for the target ISA; no device. |
| **Client perf-run** (`ClientWriter.runClient` / `getClientExecutablePath`) | **YES** | Launches the compiled client on a device to collect GEMM perf metrics. ← mock this |
| System probes (`amd-smi`, `rocm_agent_enumerator`) | **YES** | Shell out to detect/describe the device. ← spoof these |
| **ISA detection** (`detectGlobalCurrentISA` → `_detectGlobalCurrentISA`) | **YES** | Runs `amdgpu-arch` (default) / `rocm_agent_enumerator` to read the *current* device ISA; raises `Exception("Failed to detect currect ISA")` on a GPU-less host. ← spoof this (NEW, see §"What P2 taught us") |

So the switch only has to cover the **last three rows**. Everything above them is covered by the
campaign without any mock (P1–P3 and the codegen/CreateLibrary P4 rounds).

---

---

## What P2 taught us (NEW — 2026-06-06)

Two concrete findings from the coverage campaign that sharpen this PR's scope and motivation:

**1. ISA detection is a third device probe, and it's the one that actually blocks config-driven
coverage today.** When P2 added designed benchmark `*.yaml` configs under `Tensile/Tests`, the
`Tests/common/test_config.py::test_config` suite (parametrized over every YAML it discovers) ran
them through `Tensile.Tensile([config])` → `detectGlobalCurrentISA` (`Tensile/Tensile.py:647`)
→ `Exception("Failed to detect currect ISA")` in the GPU-less `tl-char` container. The campaign
worked around it add-only (configs relocated under a `test_data/` path that `findConfigs` skips).
But the *root cause is exactly this missing switch*: with ISA detection spoofed per target arch,
`Tensile.Tensile()` runs CPU-only, `test_config` (and any config-driven entry point) becomes
exercisable on a GPU-less host, and a large slice of currently-unreachable orchestration coverage
opens up — well beyond just the client perf-run. **Add ISA-detection spoofing to this PR's scope.**

**2. There is already a mock precedent for ISA detection — reuse its style.**
`Tensile/Tests/unit/characterization/Architectures/test_architectures_char.py:66,76,85`
monkeypatches `_detectGlobalCurrentISA("amdgpu-arch", 0)` and snapshots
`detectGlobalCurrentISA(0, "amdgpu-arch")`. Follow that pattern (and the existing
`ProblemSizesMock*`) so the production switch mirrors how the tests already fake these probes.

**3. Why it matters quantitatively (see `BASELINE-AND-PROGRESS.md`).** develop `-m unit`
whole-project coverage is **22.47%**; the campaign's CPU-only codegen seed set reaches
**35.89%**. The jump to ≥80% is large, and a material share of the remainder is precisely the
client perf-run + the device/ISA probes this PR unblocks. Without this PR the honest Stage-2
ceiling is expected to sit **well below 80%** — landing it is what makes the upper rounds of P4
reachable CPU-only.

---

## Relevant files (start here)

Inside the worktree
`/home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage/projects/hipblaslt/tensilelite`:

- `Tensile/ClientWriter.py` — `runClient`, `getClientExecutablePath`, `writeClientConfig*`;
  already uses `ProblemSizesMock` / `ProblemSizesMockDummy` at line ~143. **The perf-run mock
  lands here.**
- `Tensile/SolutionStructs/Problem.py:271` — `ProblemSizesMock` / `ProblemSizesMockDummy`
  (existing mock precedent to follow for style/placement).
- `Tensile/TensileCreateLibrary/ParseArguments.py:104` — the existing `--no-enumerate` flag
  (skips `rocm_agent_enumerator`). **Add the new switch alongside it**, and extend it to also
  cover `amd-smi`.
- `Tensile/Common/Architectures.py:256` (`detectGlobalCurrentISA`) / `:239`
  (`_detectGlobalCurrentISA`) — the **ISA-detection probe** (`amdgpu-arch` default, falls back to
  `rocm_agent_enumerator`); raises `"Failed to detect currect ISA"` GPU-less. Called from
  `Tensile/Tensile.py:647` (and `GenerateSummations.py:74`, `Tensile/Tests/conftest.py:204`).
  **Spoof this per target arch behind the switch** — it is what blocks `Tensile.Tensile()` /
  `test_config` CPU-only (see §"What P2 taught us").
- `Tensile/Toolchain/Validators.py:121` — `DEVICE_ENUMERATOR` resolves to `amdgpu-arch`
  (non-RHEL8) else `rocm_agent_enumerator`; the spoof must cover whichever is selected.
- `Tensile/Tests/unit/characterization/Architectures/test_architectures_char.py:66,76,85` —
  **existing mock precedent**: monkeypatches `_detectGlobalCurrentISA` and snapshots
  `detectGlobalCurrentISA`. Mirror this style.
- `Tensile/BenchmarkProblems.py:51` — imports `runClient`; line ~447 builds the
  `BenchmarkProcess`. The caller that the switch short-circuits.
- `Tensile/Common/GlobalParameters.py` — where a `GlobalParameters["CpuOnly"]` (or env)
  default would live if you back the CLI flag with a global.
- Wherever `amd-smi` / `rocm_agent_enumerator` are invoked — grep:
  `grep -rn "rocm_agent_enumerator\|amd-smi\|amdsmi" Tensile --include=*.py`

---

## GOAL

One switch (CLI flag `--cpu-only`, optionally backed by `GlobalParameters["CpuOnly"]` / an env
var) that, when **on**, routes the client perf-run and the system device-probes through
deterministic CPU-only mocks whose output is **spoofable per target architecture**; when
**off**, behavior is byte-identical to today.

**Achieved when (each provable by output you surface):**
- `git diff --stat` shows changes confined to the client-run + probe boundary (the files above)
  — **no edits** to codegen, solution derivation, or TensileCreateLibrary cross-compile.
- With the switch **on**, a CPU-only invocation runs the full
  TensileCreateLibrary → client flow for `gfx942`, `gfx950`, `gfx90a` and prints **synthetic
  perf metrics**, on a host with no GPU (verify in the `tl-char` container).
- With the switch **off**, an existing test's before/after output diff is empty (no behavior
  change).
- New tests cover the switch's on-path; they run under `-m unit` and pass.

## CONSTRAINTS (hard)

- This PR **may edit source** (that is its whole point) — but keep the diff minimal and at the
  mock boundary only. Do **not** fold any coverage-campaign (add-only) changes into it.
- Mocks must be **deterministic** and **per-arch parameterized** (spoof `amd-smi` /
  `rocm_agent_enumerator` output captured from a real run, keyed by target arch).
- **NEVER push.** Local atomic commits; land via the normal review process when ready.
- Synthetic perf metrics are a stub — see the caveat below; document it in the PR description.

## Approach (smallest-first)

1. Capture real `amd-smi`, `rocm_agent_enumerator`, **and `amdgpu-arch`** output once (any
   available host), store per-arch fixtures; add the spoof behind the switch next to
   `--no-enumerate`, including `_detectGlobalCurrentISA` (return a spoofed `IsaVersion` for the
   target arch so `detectGlobalCurrentISA` no longer raises). Reuse the monkeypatch style in
   `test_architectures_char.py`.
2. Add the client-run mock in `ClientWriter.runClient`: when the switch is on, skip
   `getClientExecutablePath`/launch and return a deterministic synthetic perf result in the
   same shape the real path returns. Follow the `ProblemSizesMock*` style.
3. Thread the switch from CLI → `GlobalParameters`/env → the two call sites; default off.
4. Add unit tests: switch-on completes the flow CPU-only for the three archs; switch-off diff
   is empty.
5. Write the PR description, including the synthetic-metrics caveat.

## Caveat to record in the PR

The synthetic perf stub means any code that **branches on measured performance** (winner
selection, retuning decisions) follows the stub's values, not real measurements. Coverage of
those branches is real, but the *decisions* are synthetic — do not mistake mocked tuning
results for meaningful ones. This is acceptable for line coverage; flag it so downstream users
don't misread results produced under `--cpu-only`.

---

## Gating relationship to the campaign

P0.5 gates **only** the P4 expansion rounds whose gap-targets sit in the client/perf-run path.
In `WORKFLOW-SPECS.md`, P4's Rank phase tags each target `needs_cpu_only_switch`; gated
targets are skipped until this PR is merged, at which point the round is re-run with
`args.haveSwitch = true`. P1–P3 and all codegen/CreateLibrary rounds do not depend on it.

---

## Start here in a NEW session (kickoff)

This work is **separate from the add-only coverage branch** — it edits source and lands via
normal review. Suggested opening prompt for a fresh session:

```
Implement the TensileLite --cpu-only / --mock-gpu switch (P0.5 prerequisite). Read, in order:
  work/tensilelite-characterization/GPU-MOCK-PR.md        (this plan — scope, files, constraints)
  work/tensilelite-characterization/BASELINE-AND-PROGRESS.md  (why it matters: 22.47% -> needs this for >=80%)
Then implement per the "Approach (smallest-first)" section. Single-agent/source work (NOT a
fan-out workflow, NOT add-only). Container tl-char, in-container project
/work/projects/hipblaslt/tensilelite, cp312 pytest/coverage on PATH. NEVER push; local atomic
commits; land via review. Scope = client perf-run + device probes (amd-smi, rocm_agent_enumerator,
amdgpu-arch) + ISA detection (_detectGlobalCurrentISA) ONLY — keep the diff at that boundary,
mirror the existing ProblemSizesMock* / test_architectures_char.py monkeypatch precedents.
Done-criteria and the synthetic-perf caveat are in this doc.
```

Decisions to confirm with the author before/while implementing:
- Switch surface: CLI `--cpu-only` only, or also `GlobalParameters["CpuOnly"]` + an env var
  (`TENSILE_CPU_ONLY`)? (Recommended: all three, env + global backing the CLI, default off.)
- Should the switch also short-circuit `test_config`'s discovered configs (i.e. is the goal to
  make the *whole* `Tensile.Tensile()` flow CPU-only, beyond just perf-run)? The P2 finding says
  yes — confirm scope explicitly so the ISA-detection spoof is in-scope, not just perf-run.
- Are real per-arch `amd-smi` / `rocm_agent_enumerator` / `amdgpu-arch` captures available, or
  should fixtures be authored from documented expected output? (Affects fixture provenance.)
