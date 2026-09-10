---
name: hipdnn-ingestor-engine
description: "Create or extend a complete hipDNN generic-kernel-ingestor integration: graph and kernel contracts, descriptors, applicable native hooks, CMake integration, artifact agreement, installed host checks, engine-attributed device correctness and final corpus coverage. Use for new engines, new packs or variants, and rocKE builder integrations."
argument-hint: "[create|extend] [<kernel-source-path-or-dir> | <existing-descriptor-dir>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN Ingestor Engine

Read this entry/completion contract, then execute [RUNBOOK.md](RUNBOOK.md).
**RUNBOOK is the only ordered workflow for both create and extend.** A generated
bundle is not a completed integration, and extending an engine does not end at a
whole-directory structural check.

| Reference | Owns |
|---|---|
| [RUNBOOK.md](RUNBOOK.md) | Ordered execution, commands, gates and final handoff |
| [extend.md](extend.md) | Identity preservation, scratch generation and addition-only splicing |
| [graph-contract.md](graph-contract.md) | Graph semantics, topology/UID edges, fields and reference capability |
| [rocke-mining.md](rocke-mining.md) | Kernel restrictions, effective specialization, layout, geometry and ABI |
| [native-pack.md](native-pack.md) | Applicable native hooks, ownership and registration contracts |
| [workloads.md](workloads.md) | Corpus identity/provenance, coverage and runtime outcome accounting |
| [Sweep reference](../../../IngestorGenerator/tools/README-sweeps.md) | Python CLI, declarative YAML, measurement and resume contracts |

## Entry contract

Record whether this is **create** or **extend**, the source revision, target
architecture, exact UED engine name, kernel source/builder and requested scope.
For extend, record the existing installed baseline and all identities/references
before generating anything. Find the actual source and CMake consumers; filenames
and historical examples are clues, not substitutes for discovery. An unresolved
source, dependency, reference capability or requested feature is a blocker to the
corresponding gate, not permission to guess or silently narrow the request.

Establish real-device and workspace feasibility early, **without an install
prerequisite**. An inherited `INSTALL` variable is not an input to early mode.
Installed-tree probing belongs after build/install. Run probes on the machine that
will execute the tests, not a login host; a listed or schedulable GPU is not proof
that the payload can see the requested device and paths.

Read the current [integration-test reference limits](../../../../../../dnn-providers/integration-tests/README.md#what-the-reference-executors-cannot-verify).
Both current SDPA reference plans reject `sink_token_tensor_uid`; **CPU is not a
sink fallback**. A missing capable numerical reference blocks the claimed feature.
Do not fabricate golden output, copy the implementation under test into a private
reference, count a skip as correctness, or implement a new CPU path as an implicit
part of this skill. Present an explicit scope/reference decision to the user.

## Dialect contract

| | `direct_load` | `packaged` |
|---|---|---|
| Typical source | Provider-embedded `.cpp`/`.hip` | rocKE builder or build-time HIP source |
| Authored source kind | `embedded_source` | `rocke` or `hip` |
| Consumer | Runtime descriptor loader | `hkp_pack`, then runtime loader |
| Runtime descriptor tree | Authored direct-load tree | Packed per-architecture tree (`kind: kpack`) |
| Embedded-source splice lists | Applicable | Never install the unlowered authored tree through these lists |

rocKE always uses the packaged path. Generic generation stays toolchain-free;
compiler evidence comes from the producing build, not generation or a later
verifier's rocKE installation. Direct-load integrations do not need a fictitious
rocKE profile, and must not be described as having compiled-specialization proof
that their path cannot supply.

## Completion contract

A complete result identifies the **final installation**, not a baseline or an
isolation arm that was replaced later. It includes:

- The approved graph/feature/shape boundary, corpus denominators and source
  provenance; all unexplained in-scope gaps are resolved.
- Real hook implementations for every referenced role, with no reachable
  placeholders. Graph criteria exist only for genuine pack narrowing; a scorer
  and UHD exist only when a heuristic is configured. A `heuristic: none` engine
  must have no score declaration, body, registration or descriptor reference.
- Applied CMake/registration changes and the correct authored/packed staging.
  Extensions preserve old UUIDs, references, hooks and unchanged inventories.
- Separate evidence for structural checks, applicable compiler/artifact agreement,
  actual typed host registration/loading and emitted-bundle census. None of these
  is proof of dispatch. Full artifact agreement cannot be replaced by a structural
  descriptor check, a policy callback or source-text symbol inspection.
- Quick-tier functional breadth, bounded standard-tier numerical depth and
  applicable negative cases against the **exact** intended engine. Record selected,
  served, skipped and failed counts; a green process with zero selected cases or
  another engine serving them fails this contract.
- Independent numerical comparison for the served corpus, and coverage/performance
  reported separately. Every final corpus input has an attributable runtime outcome:
  served, explicitly declined, execution error, missing or ambiguous. Absence of a
  timing row is not a decline. Missing/ambiguous/error outcomes block acceptance.
- After tuning or regeneration: rebuilt/reinstalled final artifacts, fresh
  agreement/native/device/corpus gates, and a complete runtime outcome join before
  `reconcile_applicability.py --declines`. Without that join, reconciliation is
  **offline only**.

## Handoff

Name the last completed RUNBOOK stage and any blocked gate. List source/config and
final artifact identities, installed paths, actual commands, exit codes and evidence
locations. State exactly what each observation proves and does not prove. Include
per-corpus denominators, exact expected/observed engine identities, correctness
results, decline reasons backed by runtime evidence, exclusions and remaining
limitations. An unexecuted command or proposed test is not evidence.

For extend, also identify the existing files/IDs retained and the new variant that
actually dispatched. Old tests passing without exercising the addition is not
extension acceptance. For partial work, report the precise boundary honestly; never
present descriptor generation, enumeration or queued device work as completion.

This skill integrates kernels through existing generator, packaging, runtime and
build/test interfaces. It does not introduce a runtime/plugin API, a build runner,
a reference implementation or a generic workflow framework. Keep local scheduling
and experiment evidence outside product source according to workspace policy; do
not publish site-specific paths or claims for unexecuted architectures.
