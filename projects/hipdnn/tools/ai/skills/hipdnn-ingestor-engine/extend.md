# Extending an existing engine

This page owns the identity and splice rules used by [RUNBOOK.md](RUNBOOK.md).
It is not a second workflow. Both create and extend finish at the RUNBOOK's final
installed artifact, device and corpus gates.

## Existing identity is the boundary

Before generation, inventory the existing engine's complete descriptor tree and
known-good installation: UED, referenced KMD, dispatch, optional heuristic, shared
matchers, per-pack matchers/KDPs and inline or standalone UKDs. Resolve references
by UUID, not filename. Pointwise ADD/MUL/SUB, for example, share a KMD even though
none of their pack filenames names it.

Record names, UUIDs, references, effective architectures and complete metadata
(including KMD defaults/types). Keep existing IDs, symbols and hooks unchanged.
Only genuinely new objects receive new UUIDs. A same-valued metadata tuple on
disjoint architectures is legal; conflicting tuples on overlapping effective
architectures are not. Check across the whole engine, not just the added pack.

A new pack adds a KDP and any genuine pack-specific graph criterion. A new variant
adds an entry in an existing pack, or a new standalone UKD plus its reference; it
need not add an engine-level object. Confirm new architecture, metadata and knobs
against the existing contract rather than re-deriving that contract by accident.

Revisit [graph-contract.md](graph-contract.md) whenever the addition changes accepted
node types, topology, layouts, attributes or feature combinations, even if the
addition is called only a "variant". Reuse unchanged contracts explicitly. A
single-pack engine becoming multi-pack requires a discriminator for every pack,
including the original, while retaining its existing IDs and engine-wide hooks.

## Scratch generation, not live regeneration

The generator has no incremental mode. A generation mints engine-level identities
as well as new pack/kernel identities. Never point `--force` at a live descriptor
or hand-filled native tree to "add" a variant: it can remint shared IDs and orphan
old packs.

RUNBOOK generation targets an empty scratch directory. Review the generated
addition against the baseline inventory. Remap every scratch reference to an
existing object to that object's real UUID, including engine, metadata schema,
dispatch, heuristic and shared matcher references wherever present. Preserve the
new IDs for genuinely new KDPs, UKDs and operation matchers. Do not replace a live
KMD/UED/UDD/UHD merely because scratch generation emitted one.

Embedded specialization declarations also contain engine/KMD IDs. Identity
remapping must keep those declarations consistent with the final tree. For
packaged output, stage the finalized authored addition and rebuild; never edit a
packed descriptor's IDs/metadata and reuse its old compiler evidence. That evidence
is bound to the exact descriptor, schema, architecture and payload.

Copy only the addition and deliberately changed existing references. For a variant
inside an existing KDP, append that reviewed entry rather than overwriting the
pack with a scratch copy missing its siblings. Preserve all pre-existing entries
and native bodies. Keep source/config, finalized authored tree and installation
inventories together in the evidence so stale staging is visible.

## Append-only integration

Apply the emitted fragments only to the applicable lists described in RUNBOOK:
source targets, registration declaration/table, test targets, and direct-load
embedding/descriptor lists when applicable. Do not replace whole lists with the
scratch fragment. A new pack under an existing engine reuses its engine-level
registration entry; add only the registration work the new pack needs.

Packaged authored descriptors never enter `HIPDNN_DESCRIPTOR_FILES` or
`HIPDNN_INGESTOR_PACK_KERNELS`. Direct-load source embedding is keyed by the actual
source-file stem, not an assumption that it equals the exported entry point.

After splicing, placeholder checks must address the selected engine's exact native
and test relative paths: pass one `--emitted-root` per spliced tree to
`generate.py --check-placeholders`, as RUNBOOK stage 3 shows. An unrelated
same-basename file cannot satisfy a missing target. Field auditing is per schema
across all relevant native sources; it is an inventory aid, not native semantic
proof.

## Acceptance specific to an addition

Whole-engine structural/catalog checks include shared KMDs and old variants.
Compiled host registration/census must include both old and new emitted inventory,
with explicit architecture selection. Actual device correctness must select the
addition, not merely leave the old default candidate running. All final corpus
and post-regeneration requirements in RUNBOOK still apply.

The concrete disposable extension walkthrough adds **HALF / block_size=256** to
the existing pointwise ADD pack. Use a **one-element graph**, because the shipped
pointwise source computes one element. Preserve MUL/SUB inventories and all old
identities; update the copied census's expected ADD candidates from three to four.
Use an explicit-knob frontend smoke selecting HALF/256 and compare its numeric
output. An unchanged default-FLOAT case cannot prove this addition. Do not infer
arbitrary-size pointwise coverage from that smoke.

The handoff names retained IDs/references, all changed/new files, the baseline and
final installations, whole-engine results, and evidence that the new variant
actually served. A structural-only pass is an intermediate result, never "extend
complete".
