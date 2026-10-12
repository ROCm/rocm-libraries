# Shipped descriptors

This root holds the descriptors the provider **ships**. Its sibling `test_descriptors/`
stages into the build tree for the unit and integration binaries and is installed only
under `HIPKERNELPROVIDER_ENABLE_TESTS`. It holds two producers' bundles:

- `rocKE/gfx950_attention_dense/`, whose KDP declares gfx950 only and whose kernels are
  compiled at pack time; it packs for a build whose GPU targets include gfx950 and that
  has rocKE enabled (`HIPKERNELPROVIDER_ENABLE_ROCKE=ON`, which keeps the `rocKE/`
  folder in the walk).
- `FlyDSL/<op>/`, one bundle per op, whose kernels are **pre-built** and checked in
  beside their descriptors as `kind: "hsaco"` UKDs under `FlyDSL/<op>/<target>/`. The
  target is the generic `gfx11-generic`, whose UKDs list all eight member arches
  (gfx1100–gfx1103, gfx1150–gfx1153), so the same objects pack into each member's shard.
  It packs for a build whose GPU targets include one of those arches and that has
  FlyDSL enabled (`HIPKERNELPROVIDER_ENABLE_FLYDSL=ON`). The objects and descriptors are
  generated, not hand-written: see `flydsl/REGEN.md` in the provider tree.

For any other build the root has nothing to pack and production packaging is dormant,
unless the cache variable below is pointed elsewhere.

## Authoring a bundle

```
descriptors/<producer>/<bundle>/
```

A bundle is authored under the producer that builds its kernels. The subpath is a
convention: `kernel_source.kind` selects the behaviour. `rocKE/` is spelled with
that capitalization because the packer preserves an authored subpath verbatim into the
staged and installed trees. A bundle sits one level under its producer, so every
descriptor lands in a child of its shard root and the archive can be written at the root.

Nothing here is registered in CMake: the packer walks this root recursively, so adding a
bundle is dropping files in a folder. Kernel-source *embedding* is a separate mechanism,
required only for `kernel_source.kind == "embedded_source"`.

A `hip` or `rocke` bundle here is compiled at pack time, one comgr invocation per variant,
on every build that has this root wired — CI included. An `hsaco` bundle is packed as-is:
no comgr and no hipcc run. Trim an authored variant set to a covering subset before it
lands, and register the symbols its UKDs name in a native pack, or the loader refuses the
engine at provider load and every lowered kernel is wasted build time.

The packer does not check an `hsaco` object's format or target processor. Every
`hsaco` UKD must list its `arch`(es) (non-empty; a generic-target object lists every arch
it runs on). `hkp_pack` rejects one without, because an unrestricted `hsaco` UKD would ship
the same bytes into every shard, and they fail at module load on the wrong device.

## Root selection and dormancy

`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` is a `CACHE PATH` defaulting to this
directory; a consumer needing a different root overrides that variable rather than adding
CMake. Configure asks the packer what the root would ship for this build's architectures,
with the build's excluded family folders and disabled kinds applied. A root that would
ship nothing -- no KDP at all, or KDPs that prune on every architecture or belong to an
excluded family -- leaves packaging dormant and removes any stale product tree, whether
the root was named or inherited. That is not an error. A root that is not a directory is
fatal.

Do not delete this README: git tracks no empty directory, and the cache variable's
set-but-not-a-directory check is fatal.

## Relationship to the examples tree

`descriptor-packaging/examples/descriptors/` is a **test fixture** tree. Bundles are
authored and proved there against the packaging suite, then relocated onto this root once
a native pack registers the symbols their UKDs name. The Linux superbuild CI lane
overrides `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` to that fixture tree, so that lane
packs the fixtures and never this root.
