---
name: hipdnn-ingestor-engine
description: "Production mining and lowering for a hipDNN generic-kernel-ingestor engine: corpus and workloads, sweeps and tuning, packaging, host checks, exact-engine device correctness, final corpus coverage, and adding a hiprtc_file variant to a pack whose symbols are already installed."
argument-hint: "[<existing-descriptor-dir> | <profile-or-config-path>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN Ingestor Engine

Execute [RUNBOOK.md](RUNBOOK.md), the **only ordered workflow for production mining and
lowering** — corpus and baseline approval, packaging, tuning and final corpus proof.

**The create path is not here.**
[hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md) owns it — it consumes
the four handover facts (the entry-point signature, the bundle's file set, the macros the
source requires bound with their legal values, and the launch geometry and workspace the
kernel assumes) and lands new symbols, descriptors, registration and graphs; this skill
owns production mining and lowering. Taking a plain HIP integration through the eight
stages below makes it pay for corpus approval, packaging and tuning it does not owe.

These pages supply contracts, not alternate procedures:

| Reference | Owns |
|---|---|
| [extend.md](extend.md) | Existing identities and addition-only splices |
| [graph-contract.md](graph-contract.md) | Graph semantics, UID edges, fields and reference capability |
| [rocke-mining.md](rocke-mining.md) | Applicability, specialization, layout, geometry and ABI |
| [hiprtc-mining.md](hiprtc-mining.md) | Adding a `hiprtc_file` variant to a pack whose symbols are already installed: scope, entry point, defines and bundles |
| [native-pack.md](native-pack.md) | Native hooks, ownership, registration and census scope |
| [workloads.md](workloads.md) | Corpus identity, coverage and runtime accounting |
| [Sweep reference](../../../IngestorGenerator/tools/README-sweeps.md) | Python CLI, YAML, measurement and resume |
| [hipdnn-kernel-authoring](../hipdnn-kernel-authoring/SKILL.md) | Upstream: authoring the kernel source and proving its numerics |
| [hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md) | The create path: native symbols, descriptors, registration with the testing system, tests and graphs |

## Entry contract

Record create/extend, source revision, target architecture, exact UED engine name,
source/builder and requested scope. Extensions also record the known-good installed
baseline and retained identities. Missing source, dependencies, representability or
a capable numerical reference blocks the corresponding gate; scope changes require
explicit approval. Keep local scheduling and experiment evidence outside product
source according to workspace policy.

| | `direct_load` / `embedded_source` | `direct_load` / `hiprtc_file` | `packaged` |
|---|---|---|---|
| Authored source | HIP, embedded at configure time | HIP, compiled at `prepare()` | `rocke` or `hip` |
| Runtime descriptors | Authored direct-load tree | Authored tree, usually a drop-in root | Lowered per-arch tree, `kind: kpack` |
| Source staging | Provider embedding/descriptor lists | Bundle directory beside the descriptors | Production packager's authored source root |
| Adding a variant | Reconfigure, rebuild, reinstall | Copy files, restart the process — the cheapest exhaustive-sweep vehicle there is | Repack, reinstall |

rocKE is always packaged. Direct-load engines need neither a fictitious rocKE
profile nor a compiled-specialization claim their path cannot supply. `hiprtc_file`
adds variants to an **already-installed** pack only, reusing its registered symbols
and its handler's launch ABI. Where it applies, that is what makes it the cheapest way
to run an exhaustive sweep — a new variant costs a descriptor entry and a file rather
than a rebuild, so the descriptor-cost tier of a tuning space can be enumerated against
one installed provider — and its two constraints are the same sentence: it adds no
native symbol, so the pack must already be installed and be one you shipped, and its
handler must route `kernel_source.kind`. **A genuinely new native symbol is a rebuild,
and it is create-path work** — take it to
[hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md) rather than through
the stages below, whichever dialect it ends up using.

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

**A zero exit status means a tool completed, not that it checked anything.** Commands
that select no work, skip every case, or report their checks as not-run exit zero
routinely, which is why every requirement above is stated as a count rather than as an
outcome. Every reported result carries the count of what actually executed, and a
result whose denominator is zero is absence of evidence — it blocks the gate it was
offered for instead of passing it. The in-tree form of this discipline is
`IngestorGenerator/tools/coverage_gate.py`: it runs the static and loader rungs, then
prints the device rung as `NOT RUN`, stated rather than inferred from the two that
passed (`IngestorGenerator/tools/coverage_gate.py:32-34`,
`IngestorGenerator/tools/coverage_gate.py:236-239`), and its structural mode reports
compiled agreement as `NOT CHECKED` rather than printing a stronger claim than it
made. Report the same way. Summarising a `NOT RUN` or `NOT VERIFIED HERE` line into a
pass is the failure this rule exists to prevent.

State what each observation proves and does not prove, including device, reference,
graph and architecture limits. Generation, enumeration, proposed commands and queued
jobs cannot be reported as completed integration.
