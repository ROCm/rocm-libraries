---
name: hipdnn-ingestor-engine
description: "Create or extend a hipDNN generic-kernel-ingestor integration: graph/kernel contracts, descriptors, native hooks, packaging, host checks, exact-engine device correctness and final corpus coverage."
argument-hint: "[create|extend] [<kernel-source-path-or-dir> | <existing-descriptor-dir>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN Ingestor Engine

Execute [RUNBOOK.md](RUNBOOK.md), the **only ordered create/extend workflow**.
These pages supply contracts, not alternate procedures:

| Reference | Owns |
|---|---|
| [extend.md](extend.md) | Existing identities and addition-only splices |
| [graph-contract.md](graph-contract.md) | Graph semantics, UID edges, fields and reference capability |
| [rocke-mining.md](rocke-mining.md) | Applicability, specialization, layout, geometry and ABI |
| [native-pack.md](native-pack.md) | Native hooks, ownership, registration and census scope |
| [workloads.md](workloads.md) | Corpus identity, coverage and runtime accounting |
| [Sweep reference](../../../IngestorGenerator/tools/README-sweeps.md) | Python CLI, YAML, measurement and resume |

## Entry contract

Record create/extend, source revision, target architecture, exact UED engine name,
source/builder and requested scope. Extensions also record the known-good installed
baseline and retained identities. Missing source, dependencies, representability or
a capable numerical reference blocks the corresponding gate; scope changes require
explicit approval. Keep local scheduling and experiment evidence outside product
source according to workspace policy.

| | `direct_load` | `packaged` |
|---|---|---|
| Authored source | `embedded_source` | `rocke` or `hip` |
| Runtime descriptors | Authored direct-load tree | Lowered per-arch tree, `kind: kpack` |
| Source staging | Provider embedding/descriptor lists | Production packager's authored source root |

rocKE is always packaged. Direct-load engines need neither a fictitious rocKE
profile nor a compiled-specialization claim their path cannot supply.

## Completion and handoff

Identify the **final installation**, not a replaced baseline or isolation arm.
Report the last completed RUNBOOK stage, blocked gates, source/config and final
descriptor/payload/plugin identities, paths, commands, exit codes and evidence.
Completion requires:

- Approved feature/shape scope and complete corpus denominators with source provenance.
- Implemented referenced hooks, no reachable placeholders, applied source/test/CMake
  splices, and preserved extension identities and unchanged inventory.
- Separately stated structural, applicable compiler/artifact, real native-loading and
  packaged-census results; direct-load inventory is covered by its unit suites.
  Packaged census is the direct native per-suite/per-arch obligation in RUNBOOK
  stage 4; tests built OFF or no declared suite is absence of evidence, not evidence.
  Preserve every `NOT VERIFIED HERE` limitation. Host loading does not prove dispatch.
- Exact-engine quick/standard numerics and required negative cases, with selected,
  served, skipped and failed counts. Zero selected, all-skipped or another engine's
  work is not correctness evidence. An extension must dispatch its new candidate.
- Rebuilt/reinstalled artifacts after tuning or regeneration, then fresh final gates.
  Every final corpus input has an attributable runtime outcome and the complete join
  required by [workloads.md](workloads.md); missing/ambiguous/error outcomes block
  runtime acceptance. Without that join, reconciliation is offline only.

State what each observation proves and does not prove, including device, reference,
graph and architecture limits. Generation, enumeration, proposed commands and queued
jobs cannot be reported as completed integration.
