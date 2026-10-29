# Shipped descriptors

This root holds the descriptors the provider **ships**. Its sibling `test_descriptors/`
stages into the build tree for the unit and integration binaries and is installed only
under `HIPKERNELPROVIDER_ENABLE_TESTS`. It holds one bundle,
`rocKE/gfx950_attention_dense/` (its KDP is stored in DVC, see
[Large rocKE files live in DVC](#large-rocke-files-live-in-dvc) below), whose KDP declares gfx950 only, so production packaging
runs for a build whose GPU targets include gfx950 and that has rocKE enabled
(`HIPKERNELPROVIDER_ENABLE_ROCKE=ON`, which keeps the `rocKE/` folder in the walk). For
any other build the root has nothing to pack and production packaging is dormant, unless
the cache variable below is pointed elsewhere.

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

Packaging probes read this root too (superbuild-only,
`HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES`, `OFF` by default). Every architecture an
explicit `arch` of a KDP or UKD here names is probed automatically in each build that does
not target it, with the default sample (one UKD per compile group, or for rocKE a sweep
packing each value of each varying `spec` field once); wildcard descriptors get no probe.
An integration that needs specific UKDs packed adds a named probe,
`hkp_add_packaging_probe(ARCH <gfx> NAME <integration> UKDS <ukd-name>...)`, naming UKDs of
this root by `name`. A KDP a probe keeps must
keep its `kernelDescriptors` inline: a standalone-UKD reference fails probe configure.
See "Packaging probes" in `descriptor-packaging/README.md`.

Do not delete this README: git tracks no empty directory, and the cache variable's
set-but-not-a-directory check is fatal.

## Relationship to the examples tree

`descriptor-packaging/examples/descriptors/` is a **test fixture** tree. Bundles are
authored and proved there against the packaging suite, then relocated onto this root once
a native pack registers the symbols their UKDs name. The Linux superbuild CI lane
overrides `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` to that fixture tree, so that lane
packs the fixtures and never this root.

## Large rocKE files live in DVC

Large and binary files of a rocKE bundle are in DVC, not git: the KDPs (`*.kdp.json`), kernel
objects (`*.co`, `*.hsaco`), and heuristic model files (for example `model.bin` under
`heuristics/`). The other descriptors are small JSON and stay in git. The blobs are in
`s3://therock-dvc/rocm-libraries/hipdnn/ingestor`, which allows anonymous read.

**Recommended layout: one DVC artifact per bundle.** Put the bundle's KDPs and kernel
objects together in one folder and track that folder with a single pointer:

```
rocKE/<bundle>/
  <name>.kmd.json, .udd.json, .ued.json, .uhd.json, *.umd.json   in git
  artifacts.dvc                                                  in git: md5, nfiles, remote
  artifacts/                                                     git-ignored, from DVC
    <name>.kdp.json
    *.co
  heuristics.dvc                                                 in git (only if the bundle has heuristic models)
  heuristics/                                                    git-ignored, from DVC
    <id>/.../model.bin
```

One pointer per bundle keeps the pointer count and the pull small, and the whole bundle
updates atomically. Files are stored uncompressed, one blob each. `gfx950_attention_dense`
uses this layout. A single large file with its own `<name>.kdp.json.dvc` pointer is also
accepted.

**Heuristic models.** A heuristic model file is binary and does not belong in git. Track it
like the other large files: keep the `heuristics/` tree where the bundle expects it and add
one pointer for that folder (`dvc add heuristics`), or place the files in `artifacts/` when
nothing refers to them by path. A `.dvc` pointer anywhere under `rocKE/` is checked at
configure.

A KDP in `artifacts/` packs correctly: the packer walks the root recursively and computes
each packed kernel's archive path from the descriptor's depth (`../../../kpack/...` here).
The small descriptors name each other by id, not by path, so moving a KDP needs no edit to them.

### Using the bundles

- **Fetch:** TheRock's `fetch_sources.py` pulls every `*.dvc` pointer. By hand:
  `dvc pull -r ingestor <path-to-pointer>.dvc`.
- **`HIPKERNELPROVIDER_ENABLE_ROCKE=ON`:** if a DVC output is missing or differs from its
  pointer (file size, or file count for a folder), configure prints a warning and the
  shipped bundle is not packed. Builds and tests still run, so a build without a `dvc pull`
  works, for example a shallow clone or an ASAN run. The warning prints the `dvc pull`
  command. A build that sets `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` elsewhere is not
  checked. Set `-DHIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON` (default OFF) to make this
  an error. Release builds should set it.
- **`OFF`:** rocKE descriptors are not consumed. DVC is not needed.

### Adding or changing a large file

```
cd <this dir>/rocKE/<bundle>
dvc add artifacts                       # a folder; for a single file: dvc add <name>.kdp.json
printf '  remote: ingestor\n' >> artifacts.dvc   # new pointers only
dvc remote modify --local ingestor allow_anonymous_login false
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... dvc push -r ingestor
git add artifacts.dvc .gitignore
```

- Commit the `.dvc` file in the same change as the push. The pointer diff shows only the new
  md5 and size, so describe the content change in the commit message.
- The committed `.dvc/config` sets `allow_anonymous_login = true` so readers need no
  credentials. DVC then sends unsigned requests and ignores your credentials, so every write
  fails with `AccessDenied`. The `--local` override goes in the git-ignored
  `.dvc/config.local`. The identity needs `s3:PutObject` and `s3:ListBucket` on the
  `ingestor/` prefix. Pass credentials as environment variables, never in a config file.
- `dvc push -r ingestor` uploads everything in your local cache. Do not pull other remotes'
  data into the same worktree first.
