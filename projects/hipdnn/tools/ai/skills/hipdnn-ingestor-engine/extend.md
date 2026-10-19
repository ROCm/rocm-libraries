# Extension contract

[RUNBOOK.md](RUNBOOK.md) owns execution. This page defines what an addition may change
and what it must preserve.

## Identity and scope

The baseline inventory is the known-good installation plus the complete engine tree:
UED, referenced KMD, dispatch, optional heuristic, shared/per-pack matchers, KDPs and
inline/standalone UKDs. Resolve by UUID, not filename — pointwise ADD/MUL/SUB share a
KMD whose filename does not follow the individual packs.

Take the installed half with `tools/inventory.py` from the IngestorGenerator: kernels per
KDP (from `kernelDescriptors`), source kinds, distinct metadata tuples (`--group-by
<fields>`, for attention `head_size,num_query_heads,num_kv_heads`), the catalog digest
RUNBOOK stage 8 uses, and the `hipdnn_validate_descriptors` verdict:

```bash
"$PY" "$GEN/tools/inventory.py" \
  "$INSTALL/lib/hipdnn_plugins/engines/arch_content/hip-kernel-provider/$ARCH" \
  --engine "$ENGINE" --group-by <fields> \
  --validator "$BUILD/bin/hipdnn_validate_descriptors" --json "$EVIDENCE/inventory_base.json"
```

Run it again on the extended installation; the two reports are the before/after record.

Preserve old names, UUIDs, symbols, references and native hooks; only genuinely new
objects get new IDs. The generator does this for descriptor UUIDs when it is told where
the previous bundle is (below). Compare completed, KMD-typed metadata across the whole engine:
equal tuples fail only when they compete at the same arch tier on a device; disjoint
architectures or different tiers (explicit over generic over empty) are legal. New topology, layouts, attributes or feature combinations
reopen [graph-contract.md](graph-contract.md), even for a "variant".

A new variant can be one added KDP entry or a standalone UKD reference, with no new
engine-level objects. A new pack adds its KDP and a genuine pack-specific criterion. The
generator requires a discriminator for every pack when an engine becomes multi-pack;
retain existing engine-wide identities and hooks.

## Addition-only splicing

Generating an existing engine reuses its descriptor UUIDs by default: with no flag,
`generate.py` reads the bundle installed in the provider tree and prints `Reusing ids from
the installed bundle <path>`. Every descriptor the config still produces keeps its UUID
(UED, KMD, UHD, UDD, UMDs, KDPs, kernels; a retuned kernel is the exception, below) and only added descriptors get new ones, so the
scratch references to the retained UED/KMD/UDD/UHD/shared matcher, and the consumer IDs in
the KDP's `provenance.specialization_contract`, already carry the real UUIDs. Check the
result against the pre-splice KDP with `catalog_diff.py --expect-ids`.

The generator still has no in-place mode. Scratch generation emits engine-level files too,
and `--force` against a live tree, or a copy of a scratch descriptor directory, replaces
shared objects and hand-filled bodies. An extension therefore copies only reviewed
additions and explicitly changed references, never the whole scratch tree or an entire
CMake list.

`tools/splice_additions.py --scratch <scratch bundle dir> --live <live bundle dir>` does
this for descriptors: it maps every scratch UUID to its live twin (documents by file
name, kernel entries by `name`; with the installed ids reused the map is the identity for
everything retained), refuses and writes nothing if any retained document or
kernel differs beyond that map or is missing from scratch, and appends only new kernel
entries after the live ones. A refusal means the config change is not addition-only:
review it as a change, not a splice. The generated census test carries no UUIDs and is
copied whole. Review KDP diffs with `git diff --diff-algorithm=histogram`; the default
algorithm can show an interleaved addition as thousands of deletions.

- The previous version is, in order: `--ids-from <dir>`, the installed bundle, then the
  descriptor directory under `--output-dir` when it holds a bundle. `--new-ids` reads
  nothing and gives every descriptor a new ID (an engine being replaced or renamed); it
  contradicts `--ids-from` and `--engine-ids-from` and turns off sibling discovery. When
  the installed bundle is used by default and the output directory also holds a bundle (a
  previous scratch pass), it is read as a second, gap-filling source (`Also reading <dir>
  for the ids of descriptors added since installation; installed ids win wherever both
  have one.`); a scratch bundle made with different engine ids, or one that cannot be read,
  is ignored with a warning. An explicit `--ids-from` replaces the default and the
  installed bundle is then not consulted.
- A kernel is matched on its completed metadata tuple plus its effective arch first, not
  by name or position. A kernel left over then takes a left-over prior ID when each side
  has exactly one kernel with that tuple, so widening or narrowing its arch keeps the ID;
  a tuple shipped at several arch tiers is told apart by arch alone. Reordering,
  inserting, renaming, or moving a kernel between packs keeps its ID, and so does a
  priority-only change. A metadata (tuple) change or an ambiguous tuple gets a new ID.
  A reused ID also needs the new `kernel_source` to still say everything the prior one said,
  at the same values: a value the prior stated that the new source changes or drops is a
  **retune** (the same tuple built from different compiled bytes) and gets a new ID (retired
  once, minted once), because RFC 0017 (Identity collisions) leaves open whether an ID may
  stay with changed compiled bytes, and a plan that records only which kernel was picked
  would resolve a reused ID to different code on replay. An arch move combined with such a
  change also gets a new ID. A field only the new source carries is not a retune (it cannot
  be told from a knob newly stated at its default), so a knob newly stated at a
  **non-default** value under an unchanged tuple keeps the ID; to force new IDs for a
  deliberate retune, change the metadata tuple or pass `--new-ids`. Retuning retained
  kernels is not reliably ID-stable, and `splice_additions.py` refuses it as a changed
  retained kernel.
- A removed or replaced ID is never reused for another descriptor. Prior kernels that
  share one tuple under the current KMD are not reused.
- The summary line `Ids: reused N from <dir> (...), minted M, retired R: <names>` lists what
  was dropped, and gains `; N kept an id across an arch change: <names>` when a kernel kept
  its id although its effective arch changed (the tuple-only second pass can also pair two
  different kernels this way, so look there). `--dry-run` prints the same account as `Ids
  (would be): ...` and fails (exit 2) on an unusable previous bundle, including an
  installed bundle whose DVC KDP has not been pulled: `dvc pull` first, or pass `--new-ids`.
- A directory with a UED but no KDP is an error, not an empty history. A DVC-stored KDP
  (a `.dvc` file beside the descriptors) must be fetched first.
- Stderr warnings, none fatal: a DVC pointer whose tracked output does not hash to its
  recorded md5 (`dvc pull` or `dvc add`); a directory beside the bundle that holds no UED
  and no KDP but does hold a `*.dvc` pointer whose artifact is absent (an unpulled satellite
  KDP is invisible to sibling discovery and to the duplicate check: `dvc pull`; a directory
  already identified by a UED or KDP is not repeated, and a home whose KDP is unfetched
  errors when read);
  with `--engine-ids-from`, another bundle of the same engine in the same family folder
  that is not named; descriptor files
  in the output descriptor directory that this run will not write (a removed pack's KDP:
  delete it, or the next run reads it as part of the previous bundle; the warning comes
  before anything is written).
- An engine rename makes the output-directory fallback an error with a hint: use
  `--new-ids` or clear the directory.
- Iterating on an **extension** within a ticket: regenerate into the **same** output
  directory with `--force`, and the ids of the installed kernels and of the kernels added
  since are both kept. A **new** directory on each pass has nothing to read for the added
  kernels, so they are reminted (a warning says `N descriptor(s) got new ids, and <dir>
  holds an earlier generation of this engine that was not read` when one sits beside
  `--output-dir`); pass `--ids-from <previous
  scratch>/descriptors/<subpath>` instead.
- A **new engine** that is not installed yet is iterated in scratch: each pass generates into
  a **new** directory with `--ids-from <previous scratch>/descriptors/<subpath>` (or
  `test_descriptors/<set>/<slug>`). The **same** directory needs `--force` and keeps the
  ids. Either way, a same-directory run regenerates the C++ stubs and overwrites
  hand-filled bodies, so hand-fill and splice only once the descriptors settle, or keep
  hand-filled C++ outside the scratch directory.

Generation still emits the C++ stubs and fragments from scratch, and `--force` against a
live tree replaces hand-filled bodies. An extension therefore copies only reviewed additions
and explicitly changed references, never the whole scratch tree or an entire CMake list.
A standalone UKD carries its own declaration. Do not overwrite a live shared descriptor
because scratch generation emitted one.

Append new entries to an existing KDP without dropping its siblings. Packaged changes
enter the authored production source root and are repacked; a packed descriptor cannot
be edited and paired with its old descriptor/schema/arch/payload-bound evidence. A new
pack under an existing engine reuses its engine registration row and appends only
required source/test/registration work. The selected-path placeholder check and all
splice commands are in RUNBOOK stage 3.

## Sibling bundles

A second bundle of an existing engine (more kernels under the same UED name) is a
**satellite**; the bundle holding the engine-level files is the **home**. Two UEDs with one
name and different UUIDs make the loader drop both, so a satellite must reuse the home's
engine-level UUIDs:

- Sibling bundles are **packaged** and **single-pack** only. A satellite's descriptor
  directory sits under the same family folder of `descriptors/` as the home's, with its
  own `authored_subpath`; it must not write into the home's folder.
- Installed siblings are found automatically. For a packaged config, the other bundles
  beside this bundle's installed directory (same family folder under the provider's
  `descriptors/`) are used when they hold the engine's UED or a KDP naming its id, so a
  home finds its satellites and a new satellite finds its home (`Reusing the engine's ids
  from the installed bundle(s) <dirs>`). Pass `--engine-ids-from <dir>` only for bundles
  outside the provider tree, such as scratch; it is the override, it turns discovery off,
  and then **every** other bundle of the engine, the home included, must be named. A bundle
  left out is missing from the engine-wide duplicate check, and a tuple that competes at
  the same arch tier makes the loader drop the whole engine. `--new-ids` also turns
  discovery off.
- The satellite's engine block (UED, KMD, UHD, UDD, kernel matcher) must generate exactly
  what the home holds. Change the engine block in the home, then regenerate every
  satellite.
- Splice **only the satellite's KDP**. The packer rejects a repeated descriptor ID even
  with identical content, and the satellite's native stub, tests and fragments duplicate
  the engine's existing classes and `s_packs` row. A kernel equal to the home's at the
  same arch is dropped from the satellite; a competing different one is an error.
- The engine's census test asserts its own emitted kernel and pack names only: merge the
  satellite's kernel and pack names into it by hand.

## Authoring a direct-load bundle

**Descriptors get no registration, because there is none.** The packer walks a
`SOURCE_ROOT` recursively and no descriptor is named in CMake, so adding one is dropping
files in a folder — `test_descriptors/<set>/<slug>/` for a test bundle,
`descriptors/<producer>/<bundle>/` for a shipped one. The set is the whole mechanism:
each is a separate pack target walked by directory, so a bundle written into the wrong
set installs cleanly and is then invisible to the binary meant to read it.

Two packer rules govern what that walk accepts:

- **Hidden paths are skipped, with a warning.** Any dot-prefixed path segment or
  filename is warned and skipped, as is a `*.json` whose name carries no type token, so
  an incidental file or a `.git/` under a user-supplied root is tolerated rather than
  aborting the pack. The production content gate applies the same rule, so a KDP under a
  hidden path does not wire packaging.
- **An `embedded_source` `source_file` must be able to act as an identity.** It is never
  normalised, so a `..` segment is rejected (one file would take two identities) and an
  absolute path is rejected (the emitted key must be the same on every machine). Write
  it relative to the descriptor's own folder, for example `kernels/MyKernel.cpp`.

The one registration a generated engine needs is the **kernel source**, and only for
`kernel_source.kind == "embedded_source"`:

| Group | Kinds | What happens |
|---|---|---|
| Compilation inputs | `hip`, `rocke` | Authored, lowered to a code object at pack time. No CMake registration. |
| Prebuilt input | `hsaco` | Authored as `{file, symbol}`, packed byte-for-byte: no compile, no CMake registration. `arch` is mandatory; see [hsaco kernels and `arch`](../../../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md#hsaco-kernels-and-arch). |
| Passthrough | `embedded_source` | Shipped as authored, no producer runs, no archive entry. **Needs the registration below.** |
| Runtime descriptor form | `kpack` | Written *by* the packer and shipped; the runtime sees only this. The packer's intermediate also uses `kind: hsaco` for compiled variants. |

```cmake
# In P/src/tests/CMakeLists.txt, beside base's existing calls. Required ONLY for
# kernel_source.kind == "embedded_source": that kind resolves source_file against a
# key table the build compiles into the binary.
set(_my_kernel_dir "${CMAKE_CURRENT_SOURCE_DIR}/../engines/kernel_ingestor_engine/test_descriptors/<set>/<slug>/kernels")
add_kernels_for_embedding(
    TARGET hip_kernel_provider_tests
    FILES "${_my_kernel_dir}/MyKernel.cpp"
    KEYS  kernels/MyKernel.cpp)
```

Each `KEYS` entry must equal the `source_file` string the descriptor authors, because
`hkp_verify_embedded_sources` compares staged descriptors against exactly this manifest.
The runtime reads `source_file` as a key into that table and opens no file for this
kind; the packer copies none of these sources into the staged tree.

## Addition-specific evidence

Whole-engine checks cover shared KMDs and unchanged variants plus the new inventory. An
addition whose suite is censused must run its
`hip-kernel-provider-hkp-census-<arch>-<suite>` entry for **every** arch that suite is
registered at: the owning pack target's wired arches, narrowed by the call's `ARCHES`
when it names any. The entry runs `hip_kernel_provider_census_tests`, the binary census
suites are compiled into. An addition under an uncensused suite states its inventory
through that suite's ordinary host run. [native-pack.md](native-pack.md) owns
eligibility. Device proof must explicitly select and numerically verify the new
candidate; passing the unchanged default is not extension acceptance. RUNBOOK stage 5
(*Where an extension's explicit selection runs*) names where those cases go and the
command that runs only them.

A new architecture is probed automatically, and a new compile path or `spec` value is
packed by the automatic probe's default sample. When the addition needs a specific UKD
(a value combination the sweep does not guarantee) packed on every PR, add its name to the
integration's named probe `UKDS`. Run the `hkp-probe-*` entries as in RUNBOOK's final step.

The handoff identifies retained IDs/references, changed/new files, baseline/final
installations, whole-engine results and the addition's actual dispatch. RUNBOOK stage 5
contains the bounded HALF/256 pointwise example; its one-element source is not
arbitrary-size pointwise coverage.
