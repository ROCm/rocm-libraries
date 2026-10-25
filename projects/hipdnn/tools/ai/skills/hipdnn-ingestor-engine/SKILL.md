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
source/builder (for rocKE, its registry candidate) and requested scope. Extensions also
record the known-good installed baseline and retained identities. Missing source,
dependencies, representability or a capable numerical reference blocks the corresponding
gate; scope changes require explicit approval. Keep local scheduling and experiment
artifacts outside product source.
Follow a configured local evidence policy when one exists; otherwise use a user-selected
per-run evidence directory as described in [RUNBOOK.md](RUNBOOK.md#paths-and-interpreters).

| | `direct_load` | `packaged` |
|---|---|---|
| Authored source | `embedded_source` | `rocke`, `hip` or `hsaco` |
| Authored under | `test_descriptors/<set>/<slug>`, `<set>` one of `shared`, `unit`, `integration`, `archive_fixture`; staged, not shipped | `descriptors/<subpath>`, defaulting to `<kind>/<slug>`; ships |
| Runtime descriptors | Per-arch shard, kind unchanged (passthrough) | Per-arch shard, rewritten to `kind: kpack` |
| Kernel source | `add_kernels_for_embedding()` key table, compiled into the binary | `rocke` and `hip` are lowered at pack time; `hsaco` is packed as authored; one archive per arch |

An authored `hsaco` source is `{kind: "hsaco", file, symbol}`: `file` is a prebuilt
AMDGPU code object resolved relative to the descriptor that names it (inside the source
root, no root-relative fallback) and `symbol` is authored. The packer copies the bytes
as-is into the arch's archive with no compile step, reads the signature from the
object's metadata, and declares `metadata_fields: []`. It does not check the object's
format or target processor; `arch` follows "hsaco kernels and `arch`" in `dnn-providers/hip-kernel-provider/descriptor-packaging/README.md`. `hsaco_file` (direct load) remains unsupported.

Both dialects are packed and staged per architecture, and **neither registers a
descriptor in CMake**: the authored subpath is the whole mechanism. Only
`embedded_source` needs a kernel-source registration, and rocKE is always packaged.
Direct-load engines need neither a fictitious rocKE engine manifest nor a
compiled-specialization claim their path cannot supply.

## Completion and handoff

Identify the **final installation**, not a replaced baseline or reduction iteration. Report
the last completed RUNBOOK stage, blocked gates, source/config and final
descriptor/payload/plugin identities, paths, commands, exit codes and evidence.
Completion requires:

- Approved feature/shape scope and complete corpus denominators with source provenance.
- Every RUNBOOK user checkpoint (corpus, ground truth, knobs, build and sweep plan)
  asked and its answer recorded.
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
- For an engine with a `reduction:` block, the `variant_reduce.py` report, the applied
  rules and a user-accepted instance count; a remaining AOT-budget WARN is justified.
- Packaging probes run and passing (RUNBOOK's final step): the automatic
  `hkp-probe-<arch>` for the bundle's architecture (no edit), plus any named `UKDS` probe
  the integration adds in `descriptor-packaging/probes/probes.cmake`.
- Rebuilt/reinstalled artifacts after reduction or regeneration, then fresh final gates.
  Every final corpus input has an attributable runtime outcome and the complete join
  required by [workloads.md](workloads.md); missing, ambiguous or error outcomes block
  runtime acceptance, and without that join the accounting is incomplete. Every
  supplied ground-truth row is within the confirmed margin or its exception user-accepted.

State what each observation proves and does not prove, including device, reference,
graph and architecture limits. Generation, enumeration, proposed commands and queued
jobs are not completed integration.
