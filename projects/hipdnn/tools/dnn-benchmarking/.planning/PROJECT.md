# hipDNN Automated Benchmarking System

## What This Is

An automated performance benchmarking system for hipDNN that runs curated graph suites against all available providers/engines, captures timing and correctness results, and integrates into CI via GitHub Actions. Built on top of the existing `dnn-benchmarking` tool, extending it from a single-graph CLI into a suite-based automation system.

## Core Value

Reliable, automated detection of performance regressions and correctness failures across hipDNN providers — run weekly, results available without manual effort.

## Requirements

### Validated

<!-- Existing capabilities from the current benchmarking tool -->

- ✓ Single-graph benchmark execution via CLI — existing
- ✓ A/B testing between two engine configurations — existing
- ✓ GPU kernel timing via PyTorch CUDA/ROCm events — existing
- ✓ CPU wall-clock timing for graph build and execution — existing
- ✓ Reference validation via PyTorch CPU provider — existing
- ✓ Tolerance-based output comparison (rtol/atol) — existing
- ✓ JSON result export from single benchmark runs — existing
- ✓ Random seed management for reproducible inputs — existing
- ✓ BFloat16/Float16/Float32 data type support — existing
- ✓ PyTorch CUDA backend as alternative executor — existing

### Active

- [ ] Sequential multi-graph suite execution
- [ ] Structured result output (JSON/CSV) for automation consumption
- [ ] Curated set of hand-authored benchmark graphs covering broad operations
- [ ] Per-provider/engine performance and correctness capture
- [ ] Weekly GitHub Actions workflow on self-hosted GPU runner
- [ ] Results stored as GH Actions artifacts
- [ ] Separated CPU (graph build) vs GPU (kernel execution) timing per provider/engine
- [ ] Correctness tracking: execution success + output tolerance match
- [ ] Regression detection over time (performance drops, correctness failures)
- [ ] Declarative configuration for adding new benchmark graphs
- [ ] Minimal boilerplate for new benchmark addition

### Out of Scope

- Dashboard visualization — owned by another team
- Workload-level aggregation (Llama training/inference grouping) — future milestone (M5)
- Parallel batch execution across multiple GPUs — GPU parallelism not feasible yet
- PR-commenting GitHub Action — later milestone, not M1-M3
- Mobile or cloud deployment — CI runner is self-hosted

## Context

- **Existing tool:** `tools/dnn-benchmarking/` is a working Python CLI that benchmarks single hipDNN graphs with timing, validation, and A/B testing
- **hipDNN project:** Located at `../../` relative to this tool; provides Python bindings (`hipdnn_frontend`) and C++ source
- **CI environment:** Self-hosted GPU runner with AMD GPU + ROCm already exists in the org
- **Graph format:** Hand-authored JSON files representing hipDNN graph operations — no model export pipeline
- **Correctness definition:** Graph must execute without error AND output must match reference within tolerance (rtol/atol)
- **Ownership:** Tool code and basic CI automation owned by user; dashboards and advanced CI owned by others

## Constraints

- **Sequential execution:** Graphs must run one at a time on a single GPU — no parallel batch execution
- **Platform:** AMD GPU + ROCm required for hipDNN execution; PyTorch ROCm nightly builds
- **Dependencies:** `hipdnn_frontend` must be installed separately from hipDNN build
- **CI runner:** Self-hosted with GPU access; GH Actions artifacts for result storage (ephemeral)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hand-authored JSON graphs (not model exports) | Targeted operation coverage, no model export pipeline needed | — Pending |
| Sequential graph execution only | Single GPU, parallelism not feasible yet | — Pending |
| GH Actions artifacts for results | Simple, no external storage needed; dashboards team can pull later | — Pending |
| Scope M1-M3 only (tool work) | User owns tool + basic CI; dashboards/aggregation are other teams | — Pending |
| All available providers at runtime | Dynamic discovery, no hardcoded provider list | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-08 after initialization*
