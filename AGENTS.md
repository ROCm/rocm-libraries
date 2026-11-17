# AI Agent Guidance

This file is the repo-level entry point for AI coding agents (and for people) working in the ROCm Libraries super-repo. It does not replace component guidance. It tells you where guidance lives, how the PR-quality skills are layered, and which skills exist.

## Orientation

- **What this repo is.** A super-repo that consolidates ROCm libraries and shared components for unified build, test, and CI. See [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
- **Default branch:** `develop`. Branches follow `users/<github-username>/<branch-name>` (see CONTRIBUTING.md).
- **Layout:**
  - `projects/` holds the libraries (hipBLASLt, hipDNN, rocBLAS, MIOpen, and others).
  - `shared/` holds shared components (StinkyTofu, rocRoller, Origami, and others).
  - `dnn-providers/` holds the hipDNN providers.
  - `.github/` holds workflows, composite actions, CI scripts, and Copilot instructions.
  - `skills/` holds repo-wide skills.

## Where agent guidance lives

Start with the closest guidance to the files you are changing, then come back here.

| Where | What it is |
| --- | --- |
| Component `AGENTS.md`, or `docs/ai-rules.md` | The component's own agent rules: layout, build, conventions. `CLAUDE.md` and tool-specific files are usually symlinks to it. |
| `<component>/skills/` or `<component>/tools/ai/skills/` | Skills for that component. A skill sits next to the code it governs. |
| `skills/` (repo root) | Skills that apply across projects. |
| `.github/instructions/*.instructions.md` | Copilot code-review instructions, each scoped by an `applyTo` glob. They usually point at a skill. |
| `TESTING.md` / `testing.md` in each project | The project's testing strategy and CI gates. Read it before changing tests or the CI that runs them. |

Component guidance wins over this file for that component. If two disagree, fix the disagreement in a PR.

## PR-quality skills

PR quality is a layered set of skills: one library-agnostic base, and thin overlays on top.

```text
rocm-pr-quality (base, in ROCm/TheRock)
 |-- hipblaslt-pr-quality   projects/hipblaslt/**
 |-- hipdnn-pr-quality      projects/hipdnn/**, dnn-providers/**
 `-- github-ci-pr-quality   .github/**
```

- **The base is not in this repo.** `rocm-pr-quality` lives in ROCm/TheRock at [`skills/rocm-pr-quality/`](https://github.com/ROCm/TheRock/tree/main/skills/rocm-pr-quality) (`SKILL.md` plus `reference.md`). Each overlay's "Dependency" section says how to get it. Do not vendor a copy here.
- **Overlays add and tighten; they never relax.** An overlay can add rules, add change classes, and raise thresholds for its paths. It cannot weaken a base MUST rule (M1 to M5). On any conflict the base wins.
- **Four actions.** Author assist, review assist, pre-merge gate, and check attestation. Pick one per request.
- **Advisory only.** The skills never approve or block a merge and never post to GitHub or Jira without a human approving the exact text.
- **Machine gates outrank the skills.** The Libraries PR Bot (see [docs/LIBRARIES_PR_BOT_FAQ.md](docs/LIBRARIES_PR_BOT_FAQ.md)) is the first-pass gate. Conform to it before anything else.

When a PR touches several areas, apply the base once and every overlay whose scope the diff touches.

## Skill catalog

Keep this table current. See [Adding a skill](#adding-a-skill-or-overlay).

### PR quality

| Skill | Path | Use it when |
| --- | --- | --- |
| `rocm-pr-quality` (base) | [ROCm/TheRock `skills/rocm-pr-quality/`](https://github.com/ROCm/TheRock/tree/main/skills/rocm-pr-quality) | Authoring, reviewing, or merge-gating any ROCm library PR. Read first. |
| `hipblaslt-pr-quality` | [`projects/hipblaslt/skills/hipblaslt-pr-quality/`](projects/hipblaslt/skills/hipblaslt-pr-quality/SKILL.md) | A PR touches `projects/hipblaslt/**` (including `tensilelite/`). |
| `hipdnn-pr-quality` | [`projects/hipdnn/tools/ai/skills/hipdnn-pr-quality/`](projects/hipdnn/tools/ai/skills/hipdnn-pr-quality/SKILL.md) | A PR touches `projects/hipdnn/**` or `dnn-providers/**`. |
| `github-ci-pr-quality` | [`skills/github-ci-pr-quality/`](skills/github-ci-pr-quality/SKILL.md) | A PR touches `.github/**` (workflows, actions, scripts). Cross-reads each affected project's `TESTING.md`. |

`hipdnn-review` and `pr-summary` (under `projects/hipdnn/tools/ai/skills/`) are deprecated stubs that redirect to `hipdnn-pr-quality`.

### hipBLASLt and TensileLite

| Skill | Path | Use it when |
| --- | --- | --- |
| `tensilelite-mutation-rerun` | [`projects/hipblaslt/skills/tensilelite-mutation-rerun/`](projects/hipblaslt/skills/tensilelite-mutation-rerun/SKILL.md) | Planning, running, or auditing offline TensileLite mutation analysis to find coverage gaps. |

### hipDNN

All under [`projects/hipdnn/tools/ai/skills/`](projects/hipdnn/tools/ai/skills/) unless noted.

| Skill | Use it when |
| --- | --- |
| `hipdnn-superbuild` | Building hipDNN with its providers through the repo superbuild. |
| `hipdnn-superbuild-test` | Running tests against an existing hipDNN superbuild. |
| `hipdnn-tidy` | Running clang-tidy on hipDNN, scoped to the files you changed. |
| `hipdnn-ingestor-engine` | Creating or extending a generic-kernel-ingestor integration. |
| `hipdnn-uhd` | Taking a hipDNN engine through the UHD heuristic lifecycle on the local GPU: generating, training, judging, validating, installing, and triaging heuristics and their training data. |
| `hipdnn-codegen` (at [`projects/hipdnn/tools/DescriptorGenerator/.claude/skills/`](projects/hipdnn/tools/DescriptorGenerator/.claude/skills/hipdnn-codegen/SKILL.md)) | Generating operation boilerplate from a YAML config. |
| `rfc-review` | Reviewing a technical RFC across the structured perspectives. Follow-ups: `rfc-review-compatibility`, `rfc-review-ops`, `rfc-review-security`. |
| `rfc-backlog` | Turning an RFC into a backlog of Jira stories and tasks. |

### StinkyTofu

| Skill | Path | Use it when |
| --- | --- | --- |
| `stinkytofu-guide` | [`shared/stinkytofu/.claude/skills/stinkytofu-guide/`](shared/stinkytofu/.claude/skills/stinkytofu-guide/SKILL.md) | Finding where a feature lives in StinkyTofu, or where a new one should go. |
| `add-logical-ir-instruction` | [`shared/stinkytofu/.cursor/skills/add-logical-ir-instruction/`](shared/stinkytofu/.cursor/skills/add-logical-ir-instruction/SKILL.md) | Adding an instruction to the logical IR path. |
| `worktree-venv` | [`shared/stinkytofu/.claude/skills/worktree-venv/`](shared/stinkytofu/.claude/skills/worktree-venv/SKILL.md) | Same skill as the repo-wide one below, kept for StinkyTofu's Claude setup. |

### rocKE

| Skill | Path | Use it when |
| --- | --- | --- |
| `rocke-tiling-api` | [`.../rocke/platform/python/rocke/helpers/tiling/.claude/skills/rocke-tiling-api/`](dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/helpers/tiling/.claude/skills/rocke-tiling-api/SKILL.md) | Designing and writing a GPU kernel with the rocKE tiling API. |
| `layout-viz` | [`.../tiling/.claude/skills/layout-viz/`](dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/helpers/tiling/.claude/skills/layout-viz/SKILL.md) | Rendering and explaining MMA layout visualizations. |
| `bank-conflict` | [`.../tiling/.claude/skills/bank-conflict/`](dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/helpers/tiling/.claude/skills/bank-conflict/SKILL.md) | Analyzing LDS bank conflicts for a kernel or layout. |
| `lds-bank-conflict-expert` | [`.../utilities/skills/lds-bank-conflict-expert/`](dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/optimization/utilities/skills/lds-bank-conflict-expert/SKILL.md) | Predicting LDS access conflicts from a rocKE architecture profile. |

### Repo-wide

| Skill | Path | Use it when |
| --- | --- | --- |
| `worktree-venv` | [`skills/worktree-venv/`](skills/worktree-venv/SKILL.md) | Giving each git worktree its own Python venv for a rocm-libraries package. |

## Component entry points

| Component | Entry point |
| --- | --- |
| hipBLASLt | [`projects/hipblaslt/AGENTS.md`](projects/hipblaslt/AGENTS.md) (reference material in `AGENTS_reference.md`) |
| TensileLite | [`projects/hipblaslt/tensilelite/AGENTS.md`](projects/hipblaslt/tensilelite/AGENTS.md) |
| hipDNN | [`projects/hipdnn/docs/ai-rules.md`](projects/hipdnn/docs/ai-rules.md) |
| MIOpen provider | [`dnn-providers/miopen-provider/docs/ai-rules.md`](dnn-providers/miopen-provider/docs/ai-rules.md) |
| rocKE | [`dnn-providers/hip-kernel-provider/rocke/AGENTS.md`](dnn-providers/hip-kernel-provider/rocke/AGENTS.md) |
| rocRoller | [`shared/rocroller/docs/ai-rules.md`](shared/rocroller/docs/ai-rules.md) |
| StinkyTofu DAG scheduler | [`shared/stinkytofu/src/transforms/asm/dag/AGENTS.md`](shared/stinkytofu/src/transforms/asm/dag/AGENTS.md) |
| rocBLAS | [`projects/rocblas/.github/copilot-instructions.md`](projects/rocblas/.github/copilot-instructions.md) and `projects/rocblas/.cursor/rules/` |

### Copilot review instructions

| File | Scope |
| --- | --- |
| [`hipblaslt-pr-quality.instructions.md`](.github/instructions/hipblaslt-pr-quality.instructions.md) | `projects/hipblaslt/**` |
| [`projects-rocblas-testing.instructions.md`](.github/instructions/projects-rocblas-testing.instructions.md) | `projects/rocblas/**` |
| [`github-ci-pr-quality.instructions.md`](.github/instructions/github-ci-pr-quality.instructions.md) | `.github/**` |
| [`rocke-pr-review.instructions.md`](.github/instructions/rocke-pr-review.instructions.md) | `dnn-providers/hip-kernel-provider/rocke/**` |

## Adding a skill or overlay

1. **Place it next to what it governs.** A component skill goes under that component (`projects/<name>/skills/<skill-name>/SKILL.md`, or the component's existing skills directory). A skill that spans projects goes under the root `skills/`.
1. **Write the frontmatter.** `name` matches the directory. `description` says what it does and when to use it. For a PR-quality overlay, also set `extends: rocm-pr-quality`.
1. **Keep an overlay thin.** Include a Dependency section (read the base first), a Scope section (paths and target branch), the supplements (what it adds or tightens), and a "What the overlay cannot do" section. Never relax a base MUST rule. Follow the existing overlays as templates.
1. **Make it discoverable.** Link it from the owning component's `AGENTS.md` or `docs/ai-rules.md`. If Copilot review should follow it, add a `.github/instructions/<name>.instructions.md` file with an `applyTo` glob that points at the skill.
1. **Register it in the [Skill catalog](#skill-catalog)** in the same PR.
1. **Author the PR with the PR-quality skill**, like any other change.

## Working rules

- Prefer the repo's own standards and scripts over inventing a process: [CONTRIBUTING.md](CONTRIBUTING.md), each project's `TESTING.md`, and the pre-commit hooks in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) (`pre-commit run --files <changed files>`).
- Do not post to GitHub or Jira on someone's behalf without showing them the exact text first.
- Do not skip, disable, or weaken a test just to turn CI green.
- When you move, rename, or delete files, search `.github/`, tox files, and `CODEOWNERS` for the old paths and update them in the same PR. CI path filters and trigger maps do not follow a rename on their own.
