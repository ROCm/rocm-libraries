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
only required source/test/registration work. The selected-path placeholder check and
all splice commands are in RUNBOOK stage 3.

## Authoring a direct-load bundle

**Descriptors get no registration, because there is none.** The packer walks a
`SOURCE_ROOT` recursively and no descriptor is ever named in CMake. Adding one is
dropping files in a folder — `test_descriptors/<set>/<slug>/` for a test bundle,
`descriptors/<producer>/<bundle>/` for a shipped one. The set is the whole mechanism:
each is a separate pack target walked by directory, so a bundle written into the
wrong set installs cleanly and is then invisible to the binary meant to read it.

Two packer rules govern what that walk accepts:

- **Hidden paths are skipped, and said so.** Any dot-prefixed path segment or
  dot-prefixed filename is warned and skipped, as is a `*.json` whose name carries no
  type token, so an incidental file or a `.git/` under a user-supplied root is
  tolerated rather than aborting the pack — and nothing the walk passes over is
  invisible. The production content gate applies the same rule, so a KDP under a
  hidden path does not wire packaging.
- **An `embedded_source` `source_file` must be able to act as an identity.** It is
  never normalised, so a `..` segment is rejected (one file would take two identities
  under two spellings) and an absolute path is rejected (it names a location on one
  machine, while the emitted key must be the same on every machine). Write it
  relative to the descriptor's own folder, for example `kernels/MyKernel.cpp`.

The one registration a generated engine does need is the **kernel source**, and only
for `kernel_source.kind == "embedded_source"`:

```cmake
# In P/src/tests/CMakeLists.txt, beside base's existing calls. Required ONLY for
# kernel_source.kind == "embedded_source": that kind resolves source_file against a
# key table the build compiles into the binary. hip / rocke / hsaco / kpack lower at
# pack time and need nothing here.
set(_my_kernel_dir "${CMAKE_CURRENT_SOURCE_DIR}/../engines/kernel_ingestor_engine/test_descriptors/<set>/<slug>/kernels")
add_kernels_for_embedding(
    TARGET hip_kernel_provider_tests
    FILES "${_my_kernel_dir}/MyKernel.cpp"
    KEYS  kernels/MyKernel.cpp)
```

Each `KEYS` entry must equal the `source_file` string the descriptor authors, because
`hkp_verify_embedded_sources` compares staged descriptors against exactly this
manifest. The runtime reads `source_file` as a key into that table and opens no file
for this kind; the packer copies none of these sources into the staged tree.

## Addition-specific evidence

Whole-engine checks include shared KMDs and unchanged variants, plus the new
inventory. An addition whose suite is censused must run its
`hip-kernel-provider-hkp-census-<arch>-<suite>` entry for **every** arch the owning
pack target was wired for; census eligibility follows the shard count, not the
authored dialect, so an `embedded_source` addition can be censused and a multi-shard
suite cannot. An addition under an uncensused suite states its inventory through that
suite's ordinary host run instead. See [native-pack.md](native-pack.md). Device proof
must explicitly select and numerically verify the new candidate. Passing the
unchanged default is not extension acceptance.

The handoff identifies retained IDs/references, changed/new files, baseline/final
installations, whole-engine results and the addition's actual dispatch. RUNBOOK
stage 5 contains the bounded HALF/256 pointwise example; its one-element source
must not be presented as arbitrary-size pointwise coverage.
