---
name: hipdnn-ingestor-engine
description: "Create or extend a hipDNN generic-kernel-ingestor integration: graph/kernel contracts, descriptors, native hooks, packaging, host checks, exact-engine device correctness and final corpus coverage."
argument-hint: "[create|extend] [<kernel-source-path-or-dir> | <existing-descriptor-dir>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN Ingestor Engine

Execute [RUNBOOK.md](RUNBOOK.md), the **only ordered create/extend workflow**. These
pages supply contracts, not alternate procedures:

| Reference | Owns |
|---|---|
| [extend.md](extend.md) | Existing identities and addition-only splices |
| [graph-contract.md](graph-contract.md) | Graph semantics, UID edges, fields and reference capability |
| [rocke-mining.md](rocke-mining.md) | Applicability, specialization, layout, geometry and ABI |
| [native-pack.md](native-pack.md) | Native hooks, ownership, registration and census scope |
| [workloads.md](workloads.md) | Corpus identity, coverage and runtime accounting |
| Sweep reference, from the repository root in `projects/hipdnn/tools/IngestorGenerator/tools/README-sweeps.md` (RUNBOOK's `$GEN/tools/README-sweeps.md`) | Python CLI, YAML, measurement and resume |

## Entry contract

Record create/extend, source revision, target architecture, exact UED engine name,
source/builder and requested scope. Extensions also record the known-good installed
baseline and retained identities. Missing source, dependencies, representability or a
capable numerical reference blocks the corresponding gate; scope changes require
explicit approval. Keep local scheduling and experiment artifacts outside product source.
Follow a configured local evidence policy when one exists; otherwise use a user-selected
per-run evidence directory as described in [RUNBOOK.md](RUNBOOK.md#paths-and-interpreters).

| | `direct_load` | `packaged` |
|---|---|---|
| Authored source | `embedded_source` | `rocke` or `hip` |
| Authored under | `test_descriptors/<set>/<slug>`, `<set>` one of `shared`, `unit`, `integration`, `archive_fixture`; staged, not shipped | `descriptors/<subpath>`, defaulting to `<kind>/<slug>`; ships |
| Runtime descriptors | Per-arch shard, kind unchanged (passthrough) | Per-arch shard, rewritten to `kind: kpack` |
| Kernel source | `add_kernels_for_embedding()` key table, compiled into the binary | Lowered at pack time into one archive per arch |

Both dialects are packed and staged per architecture, and **neither registers a
descriptor in CMake**: the authored subpath is the whole mechanism. Only
`embedded_source` needs a kernel-source registration, and rocKE is always packaged.
Direct-load engines need neither a fictitious rocKE profile nor a
compiled-specialization claim their path cannot supply.

## Completion and handoff

Identify the **final installation**, not a replaced baseline or isolation arm. Report
the last completed RUNBOOK stage, blocked gates, source/config and final
descriptor/payload/plugin identities, paths, commands, exit codes and evidence.
Completion requires:

- Approved feature/shape scope and complete corpus denominators with source provenance.
- Implemented referenced hooks, no reachable placeholders, applied source/test/CMake
  splices, and preserved extension identities and unchanged inventory.
- Separately stated structural, applicable compiler/artifact, real native-loading and
  census results, under RUNBOOK stage 4's rules. Tests built OFF or an empty `SUITES` is
  absence of evidence; a green embedded-source verification is not reachability evidence
  — name the shard that appeared under the pack's `OUT_ROOT`. Preserve every `NOT
  VERIFIED HERE` limitation. Host loading does not prove dispatch.
- Exact-engine quick/standard numerics and required negative cases, with selected,
  served, skipped and failed counts. Zero selected, all-skipped or another engine's work
  is not correctness evidence. An extension must dispatch its new candidate.
- Rebuilt/reinstalled artifacts after tuning or regeneration, then fresh final gates.
  Every final corpus input has an attributable runtime outcome and the complete join
  required by [workloads.md](workloads.md); missing, ambiguous or error outcomes block
  runtime acceptance, and without that join reconciliation is offline only.

State what each observation proves and does not prove, including device, reference,
graph and architecture limits. Generation, enumeration, proposed commands and queued
jobs are not completed integration.
