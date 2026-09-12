# Extension contract

[RUNBOOK.md](RUNBOOK.md) owns execution. This page defines what an addition may
change and what it must preserve.

## Identity and scope

The baseline inventory includes the known-good installation and complete engine
tree: UED, referenced KMD, dispatch, optional heuristic, shared/per-pack matchers,
KDPs and inline/standalone UKDs. Resolve by UUID, not filename; pointwise ADD/MUL/SUB
share a KMD whose filename does not follow the individual packs.

Preserve old names, UUIDs, symbols, references and native hooks. Only genuinely new
objects get new IDs. Compare completed, KMD-typed metadata across the whole engine;
conflicting tuples on overlapping effective architectures fail, while equal tuples
on disjoint architectures are legal. New topology, layouts, attributes or feature
combinations reopen [graph-contract.md](graph-contract.md), even for a "variant."

A new variant can be one added KDP entry or a standalone UKD reference, without new
engine-level objects. A new pack adds its KDP and genuine pack-specific criterion.
The generator requires a discriminator for every pack when an engine becomes
multi-pack; retain existing engine-wide identities/hooks.

## Addition-only splicing

The generator has no incremental mode. Scratch generation emits engine-level
identities too; `--force` against a live tree can replace shared objects and
hand-filled bodies. An extension therefore copies only reviewed additions and
explicitly changed references, never the whole scratch tree or an entire CMake list.

Every scratch reference to an existing UED/KMD/UDD/UHD/shared matcher must use the
real retained UUID. The same applies to consumer IDs in the enclosing KDP's
`provenance.specialization_contract` and any per-UKD override. A standalone UKD
carries its own declaration. Do not overwrite a live shared descriptor just because
scratch generation emitted one.

Append new entries to an existing KDP without dropping its siblings. Preserve new
IDs only for genuinely new descriptors. Packaged changes enter the authored
production source root and are repacked; a packed descriptor cannot be edited and
paired with its old descriptor/schema/arch/payload-bound evidence.

A new pack under an existing engine reuses its engine registration row. Append
only required source/test/registration work. Packaged authored descriptors never
enter `HIPDNN_DESCRIPTOR_FILES` or `HIPDNN_INGESTOR_PACK_KERNELS`; direct-load
embedding is keyed by source-file stem, not exported entry point. The selected-path
placeholder check and all splice commands are in RUNBOOK stage 3.

## Addition-specific evidence

Whole-engine checks include shared KMDs and unchanged variants, plus the new
inventory. The packaged per-arch census and direct-load unit inventory are different
gates; a packaged addition must run its
`hip-kernel-provider-hkp-census-<arch>-<suite>` entry for every requested packaging
arch, and a direct-load addition runs its ordinary host suites instead. See
[native-pack.md](native-pack.md). Device proof must explicitly select and
numerically verify the new candidate. Passing the unchanged default is not
extension acceptance.

The handoff identifies retained IDs/references, changed/new files, baseline/final
installations, whole-engine results and the addition's actual dispatch. RUNBOOK
stage 5 contains the bounded HALF/256 pointwise example; its one-element source
must not be presented as arbitrary-size pointwise coverage.
