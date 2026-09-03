# Packaged (kpack) integration fixture

The authored `hip`-form source root the build-time packager compiles, prunes and packs so
`IntegrationGpuKernelIngestorKpack` has a **real** `.kpack` artifact to load, dispatch and
verify against the CPU reference.

This is a **test fixture, not product surface.** It is wired through the same
`hkp_wire_root()` every source root goes through, declared with no `INSTALL_BASE`, so it
stages into the build tree's descriptor directory and nothing here reaches a release.

It packs under its own archive group (`HKP_GROUP_TESTFIXTURE`), so its
`<arch>/kpack/<group>_<arch>.kpack` cannot collide with the product root's when both stage
as siblings into one descriptor tree.

Layout rules the packager imposes (`descriptor-packaging/python/hkp_pack/descriptors.py`):

- Descriptors live in a **child folder** (`pointwise/`), not at the root. That is
  deliberate: the packer preserves each descriptor's authored subpath, so a nested
  descriptor's `library` climbs back out to the arch root to reach the archive. A flat
  fixture emits no `..` and cannot exercise that path -- which is precisely how a runtime
  containment bug reached on-device testing with every suite green.
- Every `<name>.<type>.json` plus the HIP sources the UKDs name. The type token is the
  second-to-last dot-separated segment of the filename, never a field.
- The UKDs are authored in `kind:"hip"` form (`source` / `entry` / `build`); the packager
  rewrites them to `kind:"kpack"` with `library` / `toc_key` / `symbol` / `sha256`.
- Ids are UUIDs. `requireId()` in `DescriptorLoader.hpp` rejects the packager's own
  `descriptor-packaging/tests/fixtures/` ids (`ukd-...`), which are shaped for the packager's
  pytest suite and never loaded by the runtime.

Two things here are load-bearing and fail silently if changed:

- **The engine is its own engine** (`hipkernel:pointwise_packed`, its own UED/UHD/KMD/UDD/UMD
  ids). Shipping these kernels under the shipped `hipkernel:Pointwise` engine would collide on
  the completed metadata tuple with its `embedded_source` kernels and
  `KernelIngestorStateManager` would throw, taking that engine down for the whole suite.
- **The matchers are the tightest available set.** A new engine is visible to every ingestor
  test that walks the descriptor tree, and a loose matcher would turn
  `IntegrationGpuKernelIngestor.DeclinesATwoNodeGraph`'s `EXPECT_TRUE(rankedEngineIds.empty())`
  red in a file this fixture never touched.

`arch` on the KDP is `[]` — the wildcard. The packager narrows it to the single shard arch it
is packing for, so the fixture packs for whatever `GPU_TARGETS` the developer configured
rather than pinning one machine's device.
