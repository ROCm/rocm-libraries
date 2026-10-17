# RFC 0021: Kernel Slots: Several Independently Selected Kernels per Engine

- Contributors: Brian Harrison
- **Status**: Draft
- **Implementation Version**: TBD
- **Follows**: [RFC 0017](0017_UniversalKernelDescriptor.md) § 14.2, the multi-kernel half of the
  "Composition" follow-up. Graph decomposition across engines (the composite engine, or UCD, RFC
  0017 § 15.2) is out of scope.
- **Builds on**: [RFC 0019](0019_UniversalHeuristicDescriptor.md) and
  [RFC 0020](0020_UniversalEngineDescriptor.md).

## Table of Contents

1. [Summary](#1-summary)
2. [Motivation](#2-motivation)
3. [The Slot Model](#3-the-slot-model)
4. [Descriptor Format](#4-descriptor-format)
5. [Selection and Execution](#5-selection-and-execution)
6. [Shared Workspace and Arguments](#6-shared-workspace-and-arguments)
7. [Measurement and Heuristics](#7-measurement-and-heuristics)
8. [Validation](#8-validation)
9. [Compatibility](#9-compatibility)
10. [Amendments to Other RFCs](#10-amendments-to-other-rfcs)
11. [Deferred Work and Open Questions](#11-deferred-work-and-open-questions)

Terms follow [RFC 0017 § 2](0017_UniversalKernelDescriptor.md#2-the-descriptors): a **UED** is an
engine, a **KDP** (pack) binds matchers and a dispatch to a set of kernels, a **UKD** is one
kernel, and a **KMD** is the metadata schema its kernels fill in. An engine's **catalog** is the
kernels applicable to a graph; its **ranker** orders the catalog; a **winner record** stores the
order measurement found.

---

## 1. Summary

An engine may run several kernels in a fixed order to serve one graph. Each position in that order
is a **slot**, and each slot chooses its kernel independently: it has its own KMD, catalog, ranker,
knobs and winner records. The slots share the engine's identity, its graph match, and a workspace
region for the data they hand each other.

Slots sit between the engine and its packs: **UED → slots → KDPs → UKDs**. Everything below a slot
is unchanged; a UKD is still one kernel. An engine that declares no slots has one implicit slot and
behaves as it does today. To a caller, a slotted engine looks like any other: a configuration is an
engine plus knob settings.

Independent slots let each stage be tuned on its own: for split SDPA backward, 253 per-kernel
configurations instead of 613 fixed pairs (§ 2). The cost is that a slot has no time of its own, so
measurement times whole programs, varying one slot at a time (§ 7). This RFC proposes the slot model
and a 1.1 revision of the UED and KDP formats; split SDPA backward is the first user.

## 2. Motivation

Split SDPA backward runs three kernels on one stream:

| Slot | Computes | Selected by |
|---|---|---|
| `prep` | Delta (`rowsum(dO ∘ O)`, a per-row term of the softmax gradient) and a sanitized copy of the forward log-sum-exp | the graph alone |
| `dkdv` | dK and dV | its own ranker |
| `dq` | dQ | its own ranker |

The best `dkdv` and `dq` kernels are only weakly coupled. AOTriton keeps separate gfx950 tuning
databases for dK/dV and dQ
([`bwd_kernel_dk_dv`, `bwd_kernel_dq`](https://github.com/ROCm/aotriton/tree/f65658a69f/modules/flash/database/amd/gfx950)).
Covering 80% of their tuned problem shapes takes 253 distinct configurations chosen per kernel,
against 613 distinct (dK/dV, dQ) pairs.

RFC 0017 offers two shapes for multi-kernel work, and neither fits:

- **Multi-launch pack (§ 15.1).** One UKD holds every launch and is selected as a unit. Each
  (dK/dV, dQ) pair must be authored as its own UKD, with `prep` duplicated into each.
- **Composite engine (UCD, § 15.2).** Each stage is a separate engine answering a child graph. No
  graph node means "dQ only", and Delta is not a graph tensor.

## 3. The Slot Model

```
UED (engine)            identity, graph match, shared workspace size, engine estimate
 ├── slot "prep"        KMD
 │    └── KDP(s)        matchers, dispatch, UKDs
 ├── slot "dkdv"        KMD, ranker, knobs
 │    └── KDP(s)
 └── slot "dq"          KMD, ranker, knobs
      └── KDP(s)
```

A slot holds everything an engine uses to select a kernel. The engine keeps what identifies it and
what reads the graph. Four rules define the model:

1. **Order is the contract.** Slots launch in declaration order. A slot may read anything an
   earlier slot wrote.
2. **Every slot is required.** The engine accepts a graph only if every slot has an applicable
   kernel, the shared-workspace token is valid, and the shared region fits the workspace limit (§ 6).
   Optional slots are deferred (§ 11).
3. **Slots are selected independently.** Nothing in one slot constrains the choice in another.
4. **Hand-off conventions belong to the engine.** The format and layout of data passed between
   slots, and any convention two slots must agree on (such as the dropout random-number layout), are
   fixed for the engine and met by every kernel in every slot. Every KMD field that changes a
   kernel's result is pinned to the graph by a matcher. A variant that needs a different convention
   is a different engine. This rule is what makes rule 3 safe: every combination of per-slot picks
   computes the same result.

Kernels whose choices depend on each other belong in **one** slot, launched together by that slot's
dispatch handler. Examples are a backward kernel with an optional output-conversion kernel, or a
split-KV kernel with the reduction that consumes its splits.

## 4. Descriptor Format

The UED gains two members, and each slot carries the selection members the engine carried before:

| Member | Meaning |
|---|---|
| `slots` | Ordered list of slots. When present, the engine-level `metadata`, `sort_kernel_catalog`, `knobs` and `predict_applicable_kernels` are absent. |
| `slots[].name` | A lowercase identifier (`^[a-z][a-z0-9_]*$`), unique within the engine. `global` is reserved, because [RFC 0004](0004_EngineConfigKnobs.md) owns `global.` knobs. |
| `slots[].metadata` | The slot's KMD. |
| `slots[].sort_kernel_catalog` | The slot's ranker (optional, metric-less; § 7). |
| `slots[].knobs` | KMD fields the slot exposes as knobs (optional). |
| `shared_workspace` | The bound token holding the shared region's size (optional; § 6). |

A KDP gains `slot`, naming the slot its kernels serve. It is required for a slotted engine and
forbidden otherwise. A slot may have several packs. A UKD is unchanged; it fills the KMD of its
pack's slot. A kernel serves at most one slot of an engine, so logs and records can name it without
its slot.

```jsonc
// UED
{
  "version": "1.1",
  "id": "5d2f…",
  "name": "flydsl:sdpa_bwd_cdna",
  "graph_match": {"native": "flydsl.sdpa_bwd_cdna.graph_match"},
  "shared_workspace": "ws.bytes",
  "predict_engine": {"gfx950": ["7b1e…"]},
  "slots": [
    {"name": "prep", "metadata": "a10c…"},
    {"name": "dkdv", "metadata": "b27e…", "sort_kernel_catalog": {"gfx950": "c3f1…"},
     "knobs": ["block_m", "block_n", "num_warps"]},
    {"name": "dq",   "metadata": "d48a…", "sort_kernel_catalog": {"gfx950": "e59b…"},
     "knobs": ["block_m", "block_n", "num_warps"]}
  ]
}

// KDP: names the slot its kernels serve
{"version": "1.1", "engine": "5d2f…", "slot": "dkdv", "arch": ["gfx950"] /* matchers, dispatch, kernels */}
```

## 5. Selection and Execution

**Matching.** The engine's graph match runs once for all slots. Each slot's packs are then matched
as an engine's packs are today, producing one catalog per slot (rule 2 applies).

**Knobs.** A knob `f` on slot `s` is advertised as `s.f`, with legal values and a default computed
over that slot's catalog. Setting `s.f` filters slot `s` only; a setting that leaves the slot empty
is rejected, naming the knob. An unslotted engine keeps bare knob ids.

**Candidate enumeration** lists every combination of per-slot kernels, so its size is the product of
the slot catalog sizes. Entries are computed on demand in a fixed order, independent of rankers and
records, so pages stay stable. The enumeration schema is unchanged, and a candidate's knob settings
cover every slot. A candidate is named by its per-slot kernel ids joined in slot order
(`prep=<id>|dkdv=<id>|dq=<id>`).

**Plan build.** Each slot resolves its kernel as an engine does today: order the catalog (winner
record, else ranker, else static order), apply the slot's knob filter, and take the first kernel
that builds. A kernel that fails to build costs only its slot's pick; a slot with no buildable
kernel fails the plan. The plan runs the picks in slot order on the plan stream.

**Engine selection is unchanged.** Default engine order stays static. A fused single-kernel
alternative is a separate single-slot engine, and engine selection chooses between the two as it
does between any engines.

## 6. Shared Workspace and Arguments

**Shared region.** Slots pass data through a region at the start of the plan's workspace. Its layout
depends only on the graph, so the engine's native graph match (provider code, not a declarative
pattern) computes it and publishes it as bound tokens, named values written `ws.*`. The UED's
`shared_workspace` names the token holding the total size. The match also publishes
`ws.scratch_offset`, the shared size rounded up to 256 B, and an offset token per shared object
(for example `ws.delta.offset`). The runtime checks the token at catalog build: a missing, non-integer
or negative value makes the engine decline the graph with an error naming the token. For a slotted
engine the `ws.` root is reserved: its tokens are excluded from ranker features, problem features and
training rows. For split SDPA backward, the region
holds Delta and the sanitized log-sum-exp, sized once for the maximum sequence length.

**Workspace size** is the aligned shared region plus the largest scratch any one slot needs. Slots run
one at a time, so they reuse the same scratch. A slotted kernel's workspace query reports its scratch
only. Every handler receives the unshifted workspace base; in v1 each slot dispatches through a native
handler, which reads the offset tokens in `prepare` and adds them to the base at launch. The plan
rejects a null workspace when shared plus scratch is non-zero. The workspace limit applies to the
total: a kernel is pruned from its slot when the aligned shared region plus its scratch exceeds the
limit, and the engine declines when the shared region alone exceeds it. The shared region does not
depend on any pick, so slots stay independent.

**Shared arguments.** Every runtime argument more than one slot uses (scale, dropout parameters and
random-number seed, ragged sequence information, maximum sequence lengths) is computed once in the
graph match. Slots read it rather than recompute it, so they cannot disagree on, for example, the
dropout mask.

## 7. Measurement and Heuristics

**Measurement times whole programs, varying one slot at a time,** because slots share one stream and
have no time of their own. With benchmarking enabled, the plan times:

1. The **reference program**: each slot at its first kernel in plan-build order (§ 5), its
   **reference pick**. If a reference pick fails at launch while benchmarking, that slot moves to
   its next kernel.
2. Each other kernel in each slot, with the remaining slots at their reference picks.

This takes 1 + Σₛ(Kₛ − 1) runs instead of ∏ₛ Kₛ. The plan then times the **winners program**, each
slot at its measured winner, and serves it. If the winners program fails, the plan serves the
reference program and writes no program record, so the next run re-times only the winners
program. A slot whose winner record already covers its catalog is not re-measured. While a knob
filter or workspace limit is active, the plan times the filtered catalogs, serves the winner and
writes no records.

**Winner records are per slot**, ranking that slot's kernels by the whole-program times above. A
separate **program record** stores the winners program, as a slot-to-kernel-id map, and its time,
which is the engine's measured time. A program record is valid only if every kernel it names is in
the current catalogs and equals the slot winners; validity is checked on read. Slot records and the
program record are written in one locked append.

**Heuristics.**

- A slot's ranker orders only its own slot. It predicts no time, since a slot has none.
- The engine's estimate comes only from its engine-level model (`predict_engine`), which models the
  whole program.
- A configuration-level prediction comes only from measurement. It exists while a valid program
  record exists, and only for knob settings that leave that program unchanged; otherwise it is
  unavailable.

**Training data.** Rankers are trained on benchmark data. In v1 the benchmark tool times every
enumerated candidate, which for a slotted engine is the full product of slot catalogs; a reference
kernel that fails to launch fails the sweep. Training reference picks are each slot's first kernel in
kernel-id order, marked by a `slot_reference` column; slot rankers train on the rows where the other
slots hold their training reference picks. These can differ from the runtime reference picks, which
is acceptable under the separability assumption (§ 11).

## 8. Validation

At load, the runtime applies RFC 0020 § 13 and rejects a slotted engine, or a pack, that breaks a
rule in § 4: engine-level selection members beside `slots`, an empty or misnamed slot list, a slot
ranker that declares a metric, `shared_workspace` without a native graph match, a pack with a
missing or unknown `slot`, or one kernel in two slots. The existing per-engine checks (RFC 0020
§ 13.2, RFC 0019 § 6.3, RFC 0018 § A.5) run per slot. A slot with no loadable pack is a warning:
the engine stays registered and declines every graph. An unslotted engine with no loadable pack is
still dropped, as today. Within a slotted engine, a duplicate metadata tuple in a slot or a duplicate
kernel id in a slot's arch group drops the offending pack; unslotted engines keep today's behavior.

## 9. Compatibility

UED and KDP take a minor bump to 1.1, adding `slots`, `shared_workspace` and `slot`; UHD takes 1.1
for a slot ranker, whose `trained_against` adds `slot`, and a slotted `predict_engine` model, whose
`trained_against` carries `kmds: [{slot, id, revision}]` instead of `kmd` (RFC 0019 § 8.1). Only
files that use slots are written as 1.1. A 1.0 runtime skips 1.1 files with a warning, and a 1.1 KDP
beside a skipped UED is reported as orphaned; a 1.1 runtime reads 1.0 files unchanged. A 1.0-labelled
file that contains slot members is rejected.

An engine without slots keeps its knob ids, trained models and winner records, so nothing trained or
stored is invalidated. Adding, removing, renaming or reordering slots is a breaking engine revision,
because knob ids and record keys change. For a slotted engine, winner records carry the slot name,
the cache shard path uses the UED id and the hash of the slot rankers' model hashes in slot order,
and a slot ranker with a model but no hash disables persistence, as for an unslotted engine
(RFC 0019 § 4.1, § 9.2). Winner-record lines move to format v3 for slotted engines only; v2 lines
remain readable.

## 10. Amendments to Other RFCs

For an unslotted engine, every statement in RFCs 0017-0020 reads as written. For a slotted engine,
statements about an engine's kernel selection apply per slot. The edit list:

| RFC | Change |
|---|---|
| 0017 | Kernel slots replace the multi-launch pack (§ 15.1) and its intermediate buffers (§ 15.3) for multi-kernel engines; the UCD (§ 15.2) is unaffected. KDPs gain `slot` (§ 4). Applicability requires a kernel in every slot, and workspace includes the shared region (§ 8.1, § 8.4, § 8.6). Declarative workspace arguments are not offset past the shared region, so v1 slots use native handlers that read offset tokens (§ 6). Metadata-tuple uniqueness is per slot; a kernel id belongs to one slot (§ 10). |
| 0018 | Kernel-scoped matchers check against the KMD of the pack's slot (§ 8, § A.5). |
| 0019 | KMD, ranker, catalog, winner records, caches and contract checks are per slot; slot rankers are metric-less and train on whole-program times; a slotted engine has no model-based configuration estimate; its benchmark cost is the product of slot catalogs (§ 3, § 4.1, § 5, § 6.3, § 8.1, § 9.2, § 11, § 13.2). |
| 0019.13 | Slot rankers train per slot (§ 3.5, § 7.3, § 8.3). |
| 0020 | UED gains `slots` and `shared_workspace`; `metadata` is required only without `slots` (§ 4.2, § 10). Selection members and knobs move to slots, with knob ids `<slot>.<field>` (§ 4.2, § 4.6, § 8). A native graph match may publish `ws.*` tokens (§ 6.1). Load checks run per slot (§ 13). |

## 11. Deferred Work and Open Questions

| Deferred | Path forward |
|---|---|
| Concurrent slots | A `phase` per slot; slots in the same phase run concurrently. Needs helper streams or graph capture. |
| Optional slots | A `required` flag, once a consumer needs one. A slot whose need depends on another slot's pick stays inside that slot. |
| Cross-slot constraints | Constraint fields on the KMD; selection becomes a constrained search. |
| Per-slot timing | Per-kernel timing inside the program would allow calibrated slot rankers and model-based configuration estimates. |
| Cheaper training collection | Collect from benchmarking mode instead of the full product, once winner records carry the benchmark tool's per-run statistics. |
| `predict_applicable_kernels` per slot | `slots[].predict_applicable_kernels`. |
| Shared-region tokens from declarative graph matches | Declarative publication of the `ws` token root, which this RFC reserves for slotted engines, in RFC 0020 § 6.1. |

**Open questions.**

1. **Separability.** Measurement assumes one slot's pick does not change another slot's time;
   shared cache state could break this. Comparing the winners program's time with the reference
   time detects a large effect. The first implementation (split SDPA backward) should also time the
   top 2×2 slot pairs end to end; if pairs reorder, measurement needs a pairwise confirmation step.
2. **Shared kernels across engines.** A standalone `prep` UKD can serve several engines, but each
   pack loads its own copy. Is one load per pack acceptable?
