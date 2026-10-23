# ADR 0034: Distinguish code-changing kernel modes

Status: Accepted
Bulk-Snapshot-Update: yes
Related: [PR #11202](https://github.com/ROCm/rocm-libraries/pull/11202), AIHPBLAS-4388

## Context

Kernel compilation groups records by their emitted symbol. Controls that change
the generated assembly must therefore distinguish that symbol, even when other
solution parameters match. Non-default `StreamKAtomic` and `MbskPrefetchMethod`
values previously lacked tags in the minimal name. Adding those tags changes
the shortened basenames recorded by four code-generation snapshot files:

- `test_emit_bigfiles_char.ambr`: the `equality_gfx950_HSS_big` node.
- `test_emit_gfx942_char.ambr`: the BBS bias/activation and DTV entries.
- `test_r2_streamk_char.ambr`: the affected StreamK entry.
- `test_seed_gfx942_gg_char.ambr`: the affected grouped-GEMM entries.

These are intended naming changes, including on stable gfx942, caused by the
newly encoded controls. Only basename strings change; the successful `err`
values remain unchanged. The snapshot guard counts four changed files and
requires an explicit override above its limit of three.

## Decision

Accept the basename changes in those four files together because they record
the same naming correction. Preserve every `err` value and every unaffected
node. Keep non-default code-changing controls in the symbol so different
kernels cannot share a compiled artifact merely because those tags were absent.

The snapshots were updated only at the affected nodes and passed two subsequent
runs without snapshot updates. The complete characterization run then passed
3,564 tests and 795 snapshots. Validation used AMD clang 23.0.0git, ROCm
llvm-project `aa451e1fe6a7`. The naming tests additionally require distinct
symbols and compile groups for each supported non-default control.

## Consequences

The four reviewed files satisfy the snapshot guard through this decision record;
the guard's limit and all numerical tolerances remain unchanged. Benchmark cache
naming version 2 prevents reuse of artifacts generated with the older names.
Further naming changes require their own explanation and targeted validation.
Passing these snapshots establishes expected naming and code-generation status;
device execution and the earlier MIOpen numerical failures still require GPU CI.
