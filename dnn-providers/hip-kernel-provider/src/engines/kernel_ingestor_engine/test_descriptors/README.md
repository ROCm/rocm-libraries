# Test descriptors

Authored descriptor sets for the hip-kernel-provider test suites. The build packs each set
into one of two discovery roots. The unit binary reads the `unit` root. The integration
binary reads the `integration` root. The two roots stay disjoint.

| Source folder | Dialect | Consumer |
|---|---|---|
| `shared/conv_fwd/` | `hip` | both binaries |
| `unit/pointwise/` | `embedded_source` | the unit binary |
| `integration/pointwise/` | `hip` | the integration binary |
| `integration/archive_fixture/` | `hip` | the integration binary |

The packer runs five times over these sets. `shared/conv_fwd/` feeds both binaries, and each
other folder feeds one binary. Every descriptor in these sets applies to each packed
architecture. Each set therefore stages one architecture folder for each packed architecture.

## `shared/conv_fwd/`

The descriptors for the engine `hipkernel:ConvFwd`. Author this set one time. The build runs
the packer over it two times, one time into each discovery root. Both binaries then read the
same authored descriptors.

## `unit/pointwise/` and `integration/pointwise/`

Both folders declare the engine `hipkernel:Pointwise`. `unit/pointwise/` uses the
`embedded_source` dialect. `integration/pointwise/` uses the `hip` dialect.

Author the two sets one time each, one set per dialect. One engine id in two dialects
collides on the completed metadata tuple. The collision removes that engine from the whole
suite. Each set therefore feeds a different discovery root, and the two roots never merge.

The two sets are not a matched pair. The dialect collision is the reason for two sets. Edit a
set for the binary that reads it, and do not mirror the edit into the other set.

Each kernel descriptor in an architecture folder declares that architecture. The packer
compiles no kernel for the `embedded_source` dialect. The architecture folders of
`unit/pointwise/` therefore hold no archive.

## Kernel sources

`shared/conv_fwd/kernels/` and `integration/pointwise/kernels/` hold the sources that the
packer compiles. The packer resolves each `source` value as `source_root / rel_dir / source`,
so the folder controls resolution here.

`unit/pointwise/kernels/` holds the sources that the build embeds in the unit-test binary.
The build embeds a table of sources in that binary. The build lists each source and its key
in `src/tests/CMakeLists.txt`. The packer copies none of these sources into the staged tree.

Each `embedded_source` descriptor names one key in its `source_file` value. Each descriptor
writes this value relative to its own folder, for example `kernels/PointwiseAdd.cpp`. The
packer emits the value unchanged. The runtime reads `source_file` as a key into the table.
The runtime opens no file for this dialect.

A build step checks each staged `embedded_source` descriptor against the table. The step
fails the build when the table holds no entry for a `source_file` value. The step also checks
that the registered file sits at the location the descriptor records. Add each source to
`src/tests/CMakeLists.txt` under the key its descriptor names.

Each staged `embedded_source` descriptor holds a `provenance` block beside its
`kernel_source` block. The block records the authored values. The `rewritten` field names the
architecture stamp. Read the block to find the origin of a staged descriptor. The runtime
does not read the block.

## `integration/archive_fixture/`

Descriptors in the `hip` dialect, which the build packs into a per-arch archive.

Keep the descriptors in the `pointwise/` child folder. The packer preserves the authored
subpath of each descriptor. A nested descriptor reaches the archive through a climb out of
its own folder. A flat fixture does not emit that step and cannot test it.

The fixture declares its own engine, `hipkernel:pointwise_packed`. That identity is what
keeps the fixture clear of a collision with `hipkernel:Pointwise`.

Each matcher symbol that the fixture names is the symbol that `hipkernel:Pointwise` also
names. The fixture therefore shares the claim space of `hipkernel:Pointwise` by design, and
both engines rank for a pointwise ADD/f32 graph.

The overlap is a requirement. The broken-archive test corrupts the archive of the fixture,
and then asserts that the workload still runs. `hipkernel:Pointwise` serves that workload. A
suite with only one applicable engine has no fallback and cannot make that assertion.

Two rules follow for a test author:

- Name the engine id when you assert which engine served a graph.
- Do not assert the size of the applicable-engine list for a pointwise ADD/f32 graph.
