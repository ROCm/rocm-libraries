# Shipped descriptors

This root holds the descriptors the provider **ships**. Its sibling `test_descriptors/`
stages into the build tree for the unit and integration binaries and is installed only
under `HIPKERNELPROVIDER_ENABLE_TESTS`. It holds the rocKE bundles below, plus
`FlyDSL/<op>/` and `conv/<family>/` bundles.

| Bundle | Engine | Native pack |
|---|---|---|
| `rocKE/gfx950_attention_dense/` | `hipkernel:Gfx950AttentionDense` | `packs/Gfx950AttentionDenseNative.cpp` |
| `rocKE/gfx950_attention_unified_2d/` | `hipkernel:Gfx950AttentionUnified2D` | `packs/Gfx950AttentionUnified2DNative.cpp` |
| `rocKE/gfx950_attention_unified_3d/` | `hipkernel:Gfx950AttentionUnified3D` | `packs/Gfx950AttentionUnified3DNative.cpp` |
| `rocKE/gfx942_attention_unified_3d/` | `hipkernel:Gfx942AttentionUnified3d` | `packs/Gfx942AttentionUnified3dNative.cpp` |
| `rocKE/gfx942_attention_dense/` | `hipkernel:Gfx942AttentionDense` | `packs/Gfx942AttentionDenseNative.cpp` (shares `packs/AttentionDenseNative.cpp` with gfx950) |
| `rocKE/gfx1151_wmma_attention/` | `hipkernel:Gfx1151WmmaAttention` | `packs/Gfx1151WmmaAttentionNative.cpp` |
| `rocKE/gfx1151_wmma_attention_decode/` | `hipkernel:Gfx1151WmmaAttentionDecode` | `packs/Gfx1151WmmaAttentionNative.cpp` |
| `rocKE/gfx950_conv_dgrad_aot/` | `hipkernel:Gfx950ConvDgradAot` | `packs/ConvDgradAotNative.cpp` |
| `rocKE/gfx1250_conv_dgrad_aot/` | `hipkernel:Gfx1250ConvDgradAot` | `packs/ConvDgradAotNative.cpp` (shared with gfx950) |

The `gfx950_attention_dense`, `gfx950_attention_unified_2d`, `gfx950_attention_unified_3d`,
`gfx942_attention_unified_3d`, `gfx942_attention_dense`, `gfx950_conv_dgrad_aot` and
`gfx1250_conv_dgrad_aot` KDPs are stored in DVC (see [Large rocKE files live in DVC](#large-rocke-files-live-in-dvc) below)
and declare gfx950, gfx942 or gfx1250 only; `gfx1151_wmma_attention` and `gfx1151_wmma_attention_decode` KDPs are in
git and declare the gfx11 and gfx12 RDNA targets. Each bundle packs for a build that has
rocKE enabled (`HIPKERNELPROVIDER_ENABLE_ROCKE=ON`, which keeps the `rocKE/` folder in the
walk) and whose GPU targets include an arch its KDP declares; for gfx950, gfx942 and
gfx1250 the DVC KDPs must also have been pulled.

The two gfx1151 engines declare engine knobs in their UEDs (set in the IngestorGenerator
configs, not by hand): the prefill bundle `rocKE/gfx1151_wmma_attention/` declares
`head_size` and `head_size_v`, and the decode bundle `rocKE/gfx1151_wmma_attention_decode/`
declares `head_size`. A knob value is the bucket width a kernel serves, not an exact size,
which keeps the knob tuple of every co-admitted kernel distinct. A pin that matches no
kernel is refused. The tile size (`block_m`) is not a knob.

`FlyDSL/<op>/`, one bundle per op, whose kernels are **pre-built** and stored beside
their descriptors under `FlyDSL/<op>/<target>/`, as `kind: "hsaco"` UKDs inline in
that target's KDP. Each `<target>/` directory is a DVC output on the same `ingestor`
remote (`<target>.dvc` in git), fetched and checked like rocKE's (below). The
target is a concrete arch or an LLVM generic target such as `gfx11-generic`, whose UKDs
list every member arch, so the same objects pack into each member's shard. A bundle
packs for a build whose GPU targets include one of its arches and that has FlyDSL
enabled (`HIPKERNELPROVIDER_ENABLE_FLYDSL`, OFF by default; TheRock turns it on). The objects and descriptors are
generated, not hand-written: see `flydsl/REGEN.md` in the provider tree.

`conv/<family>/`, the conv engine group, one bundle per convolution engine family. Its
kernels are HIP source, `kind: "embedded_source"`, stored beside their descriptors under
`conv/<family>/kernels/`, embedded in the plugin and compiled at runtime through HIPRTC, so
its KDPs declare no arch and pack into every shard. A bundle packs for a build that has the
group enabled (`HIPKERNELPROVIDER_ENABLE_CONV`, OFF by default). The generated instance
catalogs (`conv/<family>/*.kdp.json`) are in DVC, one `<catalog>.dvc` pointer each (see
[Conv catalogs live in DVC](#conv-catalogs-live-in-dvc) below); naive's three hand-written
KDPs and every other descriptor stay in git. See `conv/README.md` in the provider tree.

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
  heuristics/<id>/<role>/<arch>/<metric>/                        one folder per trained model (only if the bundle has any)
    heuristic.uhd.json, model.bin.dvc, .gitignore                in git
    model.bin                                                    git-ignored, from DVC
```

One pointer per bundle keeps the pointer count and the pull small, and the whole bundle
updates atomically. Files are stored uncompressed, one blob each. `gfx950_attention_dense`
and `gfx942_attention_unified_3d` use this layout. A single large file with its own `<name>.kdp.json.dvc` pointer is also
accepted.

**Heuristic models.** A heuristic model file is binary and does not belong in git. Each
one is its own DVC output: `heuristics/<ued-id>/<role>/<arch>/<metric>/model.bin` has a
`model.bin.dvc` pointer and a one-line `.gitignore` (`/model.bin`) beside its
`heuristic.uhd.json`, the layout FlyDSL uses. `uhd_gen promote` copies the model there;
add the pointer with `dvc add model.bin` in that folder, then append `remote: ingestor`
to it as for any new pointer (below). A model file that nothing refers to by path can
instead sit in `artifacts/`. Any `.dvc` pointer under `rocKE/` is checked at configure.

A KDP in `artifacts/` packs correctly: the packer walks the root recursively and computes
each packed kernel's archive path from the descriptor's depth (`../../../kpack/...` here).
The small descriptors name each other by id, not by path, so moving a KDP needs no edit to them.

### Using the bundles

- **Fetch:** TheRock's `fetch_sources.py` pulls every `*.dvc` pointer. By hand:
  `dvc pull -r ingestor <path-to-pointer>.dvc`.
- **`HIPKERNELPROVIDER_ENABLE_ROCKE=ON`:** if a DVC output is missing or differs from its
  pointer (file size, or file count for a folder), configure prints a warning and the
  whole bundle is left out of the pack, git-tracked descriptors included (a UED in git may
  name a UHD that lives in `heuristics/`). Builds and tests still run, so a build without a
  `dvc pull` works, for example a shallow clone or an ASAN run. A heuristic model is the
  same, one `model.bin.dvc` pointer each, but finer-grained: configure warns, and the pack
  ships the rest of the bundle without that model's UHD and logs `warning: not shipping
  UHD ...` with the `dvc pull` command. The UED still names the model; the loader reads
  the dangling id as that role being off for the architecture, so the engine stays
  available and falls back to its other rankers. A missing model with no `.dvc` pointer
  beside it is a bug and still fails the pack. Every warning prints the `dvc pull`
  command. A build that sets `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` elsewhere is not
  checked. Set `-DHIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON` (default OFF) to make this
  an error. Release builds should set it.
- **`OFF`:** rocKE descriptors are not consumed. DVC is not needed.
- **FlyDSL:** each `FlyDSL/<op>/<target>.dvc` pointer is checked the same way when
  `HIPKERNELPROVIDER_ENABLE_FLYDSL` is ON (with the kernel ingestor); a target not pulled
  ships no FlyDSL kernels. `flydsl/REGEN.md` covers updating them.
- **conv:** each `conv/<family>/<catalog>.kdp.json.dvc` pointer is checked the same way
  when `HIPKERNELPROVIDER_ENABLE_CONV` is ON (with the kernel ingestor). A family with a
  catalog not pulled is not served: its folder is withdrawn from the pack and its native
  pack is not registered (`conv/ConvContent.cmake`). `conv/REGEN.md` covers updating them.

## Conv catalogs live in DVC

A conv family's KDPs are generated instance catalogs (`conv/generators/`), tens of
thousands of lines each, so they are DVC outputs. They stay **beside** the family's other
descriptors rather than in an `artifacts/` folder: an `embedded_source` UKD names its
source relative to its own folder (`kernels/<File>.cpp`), the packer refuses a `..`
segment, and `hkp_verify_embedded_sources` checks the source sits at the KDP's folder plus
`source_file`. A KDP in a DVC folder could not have the git-tracked `kernels/` below it, so
each catalog is a single-file output with its own pointer:

```
conv/<family>/
  *.ued.json, *.udd.json, *.kmd.json, *.umd.json   in git
  kernels/                                         in git
  <catalog>.kdp.json.dvc                           in git: md5, size, remote: ingestor
  <catalog>.kdp.json                               git-ignored (.gitignore), from DVC
```

`conv/generators/regen_all.py` regenerates every family and records the pointers;
`conv/REGEN.md` is the procedure.

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
