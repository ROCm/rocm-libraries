<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# FlyDSL kernels in the hip-kernel-provider

Ahead-of-time RMSNorm and attention kernels written in
[FlyDSL](https://github.com/ROCm/FlyDSL), compiled once and stored in
the repository's DVC remote, reaching hipDNN through the kernel ingestor.

This directory holds the kernels, the generators that produce them, and the
descriptors that make them selectable. The matching C++ — the code that decides
whether a graph may run on them and how to launch them — lives one level up, one
file per engine:
[`FlydslRmsNormNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp)
(RMSNorm forward) and
[`FlydslRmsNormBwdNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormBwdNative.cpp)
(backward), with the rules they share in
[`FlydslRmsNormCommon.hpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormCommon.hpp);
[`FlydslSdpaRdnaNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaNative.cpp),
[`FlydslSdpaRdnaDecodeNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaDecodeNative.cpp),
[`FlydslSdpaAiwNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaAiwNative.cpp),
[`FlydslSdpaCdnaNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaCdnaNative.cpp),
[`FlydslSdpaCdnaDualwaveNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaCdnaDualwaveNative.cpp)
and
[`FlydslSdpaCdnaDecodeNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaCdnaDecodeNative.cpp)
(and [`FlydslSdpaDropoutMaskNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaDropoutMaskNative.cpp),
the RNG dump's keep-mask writer's engine, which serves no graph; the dropping engines launch its
objects, `FlydslSdpaRngDump.hpp`),
with what the SDPA engines share in `FlydslSdpaRdnaCommon.hpp` and `FlydslSdpaCdnaCommon.hpp`,
and the rules RDNA and CDNA apply alike (the mask's bounds, the sink, the scale tensor,
output aliasing) in `FlydslSdpaShared.hpp`.

**One kernel family per engine.** An engine's objects are one kernel — one algorithm, one
argument list, one launch rule, one set of knobs every object fills. Two approaches are
two engines, never a knob that picks between them, so each engine's ranker compares like
with like and the choice between approaches is hipDNN's engine-level heuristic's (UHD), or
autotune's. A variation that only bakes a runtime value or changes a schedule — a tile
size, a generic head_dim tier — stays a knob of its family.

**Nothing here is compiled by the build.** The `.hsaco` objects are prebuilt
artifacts; the provider's shared packer packs and stages them. (The build does run
Python 3, for the checks below — `gen_descriptors.py --check` and
`tools/check_shards.py` — but never FlyDSL or a kernel compiler.) That is the central fact about this
directory, and it is what the rest of this file is mostly about: an artifact
whose build is not reproduced on every `ninja` has to carry its provenance and a
way to check it, or it is just bytes someone once vouched for.

## Fetching the objects

Each `<op>/<target>/` content directory under
`src/engines/kernel_ingestor_engine/descriptors/FlyDSL/` (the objects, their
`manifest.json` and the target's KDP) is a DVC output on the `ingestor` remote, the
one rocKE's large files use. Git holds only the
`<op>/<target>.dvc` sidecar, which records the directory's content hash, and the
human-facing files beside it (`<target>.SOURCE.md` and the op's descriptors).
Fetch the directories before configuring, from the repository root (TheRock's
`fetch_sources` pulls every pointer for you):

```bash
dvc pull -r ingestor dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/FlyDSL/*/*.dvc
```

Reads are anonymous (`.dvc/config`). The provider checks every pointer at configure
exactly as it checks rocKE's: with `HIPKERNELPROVIDER_ENABLE_FLYDSL` ON, a directory
that is missing or incomplete gets a warning naming its `dvc pull` command, and its
target ships no FlyDSL kernels (FlyDSL reports dormant if nothing was pulled); builds
and tests still run. `-DHIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON` makes it an
error, as release builds and the superbuild CI set. Inside each directory,
`manifest.json` still records every object's SHA256, which the build checks (below).

## Where to go

| | |
|---|---|
| **What is covered, and what is refused** | [COVERAGE.md](COVERAGE.md) |
| **Rebuilding or verifying the kernels** | [REGEN.md](REGEN.md) |

## What ships

For RDNA, four engines -- RMSNorm forward and backward, and SDPA prefill and decode, 417
kernel objects per family, 465 on gfx11-generic with its 48 dropout objects -- plus, on gfx12-generic only, AOTriton's gfx1201 attention
kernel as a fifth (196 more, 613 in all), and on both the RNG dump's keep-mask writer (one object
each, 466 and 614 in all), built once for each of two LLVM generic targets
and shipped to every part each covers:

- **`gfx11-generic`** -- RDNA3 / RDNA3.5: gfx1100, gfx1101, gfx1102, gfx1103,
  gfx1150, gfx1151, gfx1152 and gfx1153. Verified on gfx1151.
- **`gfx12-generic`** -- RDNA4: gfx1200 and gfx1201. Built from the same kernel
  sources, which carry both WMMA operand layouts. Verified on gfx1201.

Each member's shard carries its family's bytes (see [REGEN.md §3](REGEN.md) for
why one object set per family is enough, and what it needs). RMSNorm is also built
for each of the CDNA targets gfx90a, gfx942 and gfx950.

**RMSNorm forward** (`hipkernel:flydsl_rmsnorm`), 90 objects: bf16, f16 and f32, and bf16
and f16 with an f32 gamma; inference, and training that also saves inv_rms; epsilon is
a kernel argument, so any value is served:

- a **specialized** tier with `N` baked in at 2560 / 3072 / 3584 / 4096 / 5120 / 6144 /
  7168 / 8192 / 12288 / 16384, chosen by measurement (each 1.2–4.2× over the generic
  tier on five GPUs)
- a **generic** tier that reads `N` from the tensor descriptor at runtime, so
  every other width is served too — more slowly, but served. f32 ships this tier
  only: the measurement that chose the baked widths covered bf16, and f32 baked
  objects were never measured.

**RMSNorm backward** (`hipkernel:flydsl_rmsnorm_bwd`), 63 objects: dx and dgamma, the same
dtypes and gamma forms, a deterministic two-stage dgamma reduction, baked widths from
1536 to 16384 and generic tiers bounded at 1024 / 8192 / 16384 ([COVERAGE.md §2a](COVERAGE.md)).

Widths outside the specialized lists are a **performance** boundary, not a
coverage one (bar backward above 16384). The things that are genuinely out of scope —
bias and dbias, other dtypes, non-packed operands, fusion — are enumerated in
[COVERAGE.md §3](COVERAGE.md), each with the reason the shipped objects cannot
compute it.

**SDPA forward on RDNA**: two engines, 264 objects (312 on gfx11-generic). The prefill kernel
(`hipkernel:flydsl_sdpa_rdna`, 156 objects, 204 on gfx11-generic) and the split-KV decode kernel
(`hipkernel:flydsl_sdpa_rdna_decode`, 108 objects), bf16 and f16 ×
`head_dim` 64, 96, 128 and 256 × causal on and off × with and without an
additive bias (`attn_mask`, f32 or the query's dtype, broadcast over any axis), plus a generic tier
that serves every other `head_dim` that is a multiple of 8 up to 256, with or without a bias, and a
decode path (split-KV, the GQA group packed into 16-row tiles, with and without a bias, and a generic tier
up to 512 without one) for decode steps, multi-head latent attention's un-absorbed prefill (Q/K 192, V 128), and decode objects at Q/K 576 / V 512 (absorbed MLA) and 512/512; attention sinks (gpt-oss), FLOAT or in the query's dtype, at `head_dim` 64 and 128, prefill and decode, with or without a bias. Batch, both sequence lengths,
head counts (MHA, GQA, MQA), every stride, the scale, the mask bounds (causal at
either corner, right-side bands, sliding windows) and the LSE output are runtime
arguments, so each object serves every shape and layout of its class. Inference
only; padding (per-batch lengths under `padding_mask`) is a runtime argument every object reads;
on gfx11, 48 prefill dropout objects (bf16/f16 × causal × `head_dim` 64, 96, 128, 256, without a
bias, with an f32 one or one in the query's dtype) apply dropout with AOTriton's Philox mask (on gfx12 the aiw objects below serve it);
packed (THD) sequences, compact or with per-batch lengths (cuDNN's form), run on the prefill
objects (a runtime argument they read; the decode objects decline them). Any `head_dim` pair of
multiples of 8 up to 512 runs on the aiw objects below. Status against common SDPA feature
requirements and every declined feature with its effort are in
[COVERAGE.md §5](COVERAGE.md).

**SDPA forward on RDNA4, AOTriton's kernel** (`hipkernel:flydsl_sdpa_aiw`, gfx12-generic, 196
objects): AOTriton's FlyDSL gfx1201 forward, vendored under
[`kernels_src/kernels/attention_rdna4/`](kernels_src/kernels/attention_rdna4/README.md) with the
modifications that bring it to the RDNA kernel's coverage (sinks, an additive mask with a
causal mask or window, f32 masks, V's own head count, a strided LSE). bf16 and f16 × exact
`head_dim` 64, 128, 256 (plain, a mask in the query's dtype, an f32 mask), sinks (FLOAT or the
query's dtype) at 64 and 128 (alone or with either mask), MLA 192/128, and padded tiers up to 64, 96, 128, 160, 224, 256, 384 and 512 for any other
`head_dim` that is a multiple of 8 (so up to 512, which no other RDNA engine serves at a prefill
tile past 256), each plain or with either additive mask, and each with and
without a causal mask or window. Every object also serves padding (`padding_mask` with
per-batch `seq_len_q`/`seq_len_kv`, read at runtime) and packed (THD) sequences (ragged Q/O
and K/V through cumulative token tables; COVERAGE §5.9), the padded tiers any Q/K and V head-dim
pair (V narrower or wider, the wider within the tile), and 36 dropout objects
(bf16/f16 × causal × 64, 128, 256, without a mask or with either) apply dropout with AOTriton's Philox mask (COVERAGE §5.10). Faster than the
RDNA kernel on medium and long sequences, slower on short ones; which runs a graph both
serve is the heuristic's (or autotune's) to decide. `generators/gen_sdpa_aiw.py` builds each
row through the kernel's own tuning table and records the schedule it chose as knobs.

**SDPA forward on CDNA**: three engines with objects built per arch — FlyDSL's generic
MFMA kernel (`hipkernel:flydsl_sdpa_cdna`, **gfx90a**, **gfx942** and **gfx950**), its dual-wave MFMA
kernel (`hipkernel:flydsl_sdpa_cdna_dualwave`, **gfx950** only), and the generic kernel's
split-KV decode objects (`hipkernel:flydsl_sdpa_cdna_decode`): bf16 and f16 at
`head_dim` 64, 128 and 256, causal and non-causal; both sequence
lengths, the head counts, every operand's strides (BSHD, BHSD, packed QKV), the
softmax scale, the causal diagonal (top-left or bottom-right, with a band of any width),
sliding windows and the LSE output are runtime
arguments. A bias in the query's dtype is served at `head_dim` 64, 128 and 256 by the
generic kernel on all three arches, and at 64 and 128 by gfx950's dual-wave kernel;
attention sinks (FLOAT or the query's dtype) at 64 and 128 (prefill and decode), with or without a bias; every other
`head_dim` that is a multiple of 8 up to 256 by bucket objects, with or without a bias
(gfx90a adds 96, 160, 192 and 224 buckets); an f32 bias, or one broadcast along the keys
or laid out off the 16-byte rows, by bucket objects that read it one key at a time;
MLA prefill with V 128 and Q/K up to 256, exact 512/512 prefill (the VAE mid-block's
attention), and absorbed-MLA decode at V 512 (Q/K 576).
Padding (per-batch lengths, padded rows O = 0 and LSE = −inf) runs on the generic prefill
and decode objects, packed THD sequences (ragged offsets, compact) on the generic prefill
objects, and dropout (AOTriton's Philox mask, bit for bit) on 12 generic prefill objects
per arch, with a bias on 12 more; exact f32 on 6 generic prefill objects per arch: every
multiple of 8 up to 128 (to 256 on gfx950) and 256.
174 objects on gfx950, 176 on gfx942 and 192 on gfx90a, plus the RNG dump's keep-mask writer on
each (175, 177 and 193 in all). Each class ships a default
object and measured companions -- other tiles for short queries and large problems --
chosen by timing every candidate on the dnn-benchmarking attention graphs
([COVERAGE.md §7.6](COVERAGE.md)). It is the first set that carries
these kernels through the whole path -- generation, packing, selection and
launch -- and is verified on all three arches; the order in which it widens is in
[COVERAGE.md §7](COVERAGE.md). The RDNA SDPA kernels stay RDNA-only.

## How it reaches hipDNN

The objects and their descriptors are a **bundle of the provider's production
descriptor root**, beside rocKE's, and ship through the same packer:

```
src/engines/kernel_ingestor_engine/descriptors/
  rocKE/…                         authored rocKE bundles (compiled at pack time)
  FlyDSL/<op>/*.json              descriptors shared by every target of the op
  FlyDSL/<op>/<target>.SOURCE.md  readable summary of one target's object set
  FlyDSL/<op>/<target>/           the target's pack (KDP) holding one inline
                                  `hsaco` UKD per object, the objects,
                                  manifest.json; `arch` lists the members
                                  (gfx11-generic, gfx12-generic)
        │
        │  shared packer (descriptor-packaging), product root
        ▼
lib/hipdnn_plugins/engines/arch_content/hip-kernel-provider/<arch>/   one per member
  FlyDSL/…                        descriptors, rewritten to kind "kpack"
  kpack/hip_kernel_provider_<arch>.kpack   every producer's objects, one archive
```

Each target's KDP holds its UKDs inline, one per object: no UKD is shared
between packs, the specialization contract is declared once on the KDP, and
the packaging probes take only inline UKDs. Each UKD is `kind: "hsaco"` and
names its object by file name, beside the KDP. The packer resolves the object, packs it into the archive of every arch the UKD
lists, reads the argument signature out of the object and rewrites
the UKD to `kind: "kpack"` — the same path a rocKE kernel takes after it is
compiled. The shard is installed with everything else under
`arch_content/hip-kernel-provider/`, which the plugin finds beside its own
module, so an installed plugin loads FlyDSL's kernels with **no environment
variable** and nothing FlyDSL-specific at runtime.

The ingestor reads the shard's descriptors to learn what kernels exist and what
they cost, and calls into the four native symbols each pack registers —
`hipkernel.flydsl_<op>.{graph_match,kernel_match,score,dispatch}` — to match,
rank and launch. No FlyDSL, no Python and no compiler is present at runtime;
by then these are ordinary code objects in an archive.

The descriptors are **generated from the objects they describe**
(`gen_descriptors.py`, from each `manifest.json`), so a descriptor cannot drift
from its kernel. The build re-runs `gen_descriptors.py --check` before the pack
-- every object must match its manifest's SHA256 and be built for the target its
directory names -- and `tools/check_shards.py` after it, which shows every
member's shard ships exactly the checked-in objects.

## Building it

Two flags, and the difference between them matters:

- **`HIPKERNELPROVIDER_ENABLE_FLYDSL`** — the option. States *intent*. **OFF by
  default**, like rocKE; switch it on with `-DHIPKERNELPROVIDER_ENABLE_FLYDSL=ON`.
  TheRock enables it through its `HIPKERNELPROVIDER_ENABLE_FLYDSL` feature flag
  (TheRock's flags can only turn an option ON, so the flag is also how TheRock
  switches FlyDSL off). It takes effect only with
  `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`, since only the ingestor loads FlyDSL's
  shard; with the ingestor OFF, configure reports FlyDSL as not built and
  carries on.
- **`HIPKERNELPROVIDER_FLYDSL_ACTIVE`** — an internal cache variable, raised by
  `flydsl/CMakeLists.txt` whenever FlyDSL content exists (the `<op>/<target>.dvc`
  pointers or pulled objects), independent of `GPU_TARGETS`. The engine and test
  TUs key on it. TheRock's multi-arch build ships one shard's host half to every
  arch, so host code must not depend on the requested architectures: the tests
  carry each op's content arch list and decide at run time whether the device's
  shard is staged.

Packing, `check_shards.py` and the census follow the requested architectures. With
`GPU_TARGETS=gfx908` the engines compile but no shard ships FlyDSL objects, because
no gfx908 kernels are checked in; that is not an error.

```bash
cmake -B build -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPKERNELPROVIDER_ENABLE_FLYDSL=ON \
      -DHIPDNN_ENABLE_SDPA=ON
cmake --build build -j
ctest --test-dir build -R hip-kernel-provider
```

`HIPDNN_ENABLE_SDPA` is hipDNN's own switch for attention, OFF by default; it
gates `graph->sdpa()` in the frontend. The SDPA pack builds and stages without
it, but no graph can then contain an SDPA node to reach it.

`HIPKERNELPROVIDER_ENABLE_FLYDSL` shapes exactly this integration's surface —
one engine TU and one test TU per op, one census suite per op, and the `FlyDSL/`
family folder, which the packer excludes with the option OFF
(`HKP_DESCRIPTOR_FAMILIES`). No other integration's flag changes
anything FlyDSL compiles, registers or asserts, and no FlyDSL suite names another
engine. That isolation is a property to preserve, not an accident. No build or CI
job checks it; to check it by hand, build with `ENABLE_HIP_MLOPS_ENGINE=OFF` and diff
the FlyDSL case list, which must come out empty.

## File map

`<content>` is `src/engines/kernel_ingestor_engine/descriptors/FlyDSL/` in the
provider tree; everything else is relative to this directory.

| Path | |
|---|---|
| `<content>/<op>/*.json` | descriptors shared by every target of an op: KMD (metadata schema), UED (engine), UHD (heuristic), UDD (dispatch), UMD (matcher) |
| `<content>/<op>/<target>.SOURCE.md` | the target's readable summary: toolchain, input digest, one row per object |
| `<content>/<op>/<target>/*.hsaco` | the committed code objects |
| `<content>/<op>/<target>/*.kdp.json` | the target's pack, with one inline `hsaco` UKD per object |
| `<content>/<op>/<target>/manifest.json` | per-object SHA256 + knobs + argument layout + the toolchain and input digest that built them |
| `kernels_src/kernels/**` | vendored kernel sources, each with its upstream's SPDX and copyright lines. A modified file carries a header naming its upstream (FlyDSL or AITER), commit, path and modifications; the `attention_rdna4/` modules inherit their package's (AOTriton), and a file written for this provider says so in its own. The FlyDSL and AITER commits are pinned in `generators/_flydsl_env.py` and recorded in each `<target>.SOURCE.md` |
| `generators/_instances.py` | **the instance table** — one row per shipped object, every op |
| `generators/_flydsl_env.py` | the pins: wheel version, upstream commits, ROCm recording |
| `generators/_codeobject.py` | reads and verifies a compiled object (target, ELF generic machine), shared by every generator |
| `generators/arch_families.json`, `_arch_families.py` | the generic targets and their member arches; read by the generators and `CMakeLists.txt` |
| `generators/_generic_targets.py` | the FlyDSL 0.3.4 shim that lets MLIR lower for a generic target |
| `generators/gen_<op>.py` | compile one op's (one engine's) rows into objects + manifest + `<target>.SOURCE.md`: `gen_rmsnorm`, `gen_rmsnorm_bwd`, `gen_sdpa_rdna`, `gen_sdpa_rdna_decode`, `gen_sdpa_aiw`, `gen_sdpa_cdna`, `gen_sdpa_cdna_dualwave`, `gen_sdpa_cdna_decode` |
| `generators/_parallel.py` | compiles a generator's objects in a pool of worker processes (`FLYDSL_GEN_JOBS`) |
| `generators/_sdpa_rdna_build.py`, `_sdpa_cdna_build.py` | what each architecture's SDPA generators share |
| `gen_descriptors.py` | derives descriptors from the objects; `--check` re-verifies |
| `tools/diff_upstream.py` | vendored sources vs. their upstream checkouts, black-normalized |
| `tools/check_shards.py` | after the pack: every member's shard ships exactly the checked-in objects |

Adding an instance is a **row in `_instances.py`** plus a regeneration — the
compiler, the packer and the descriptor emitter all read that table, so there is
no second place that can disagree with it.

## What is readable, and what is generated output

The content tree is split by one rule: **what a reviewer reads, and what does
not grow with the object count, sits at the op level; everything that grows with
it is generated output inside the per-target folders.**

```
<content>/<op>/
  flydsl_<op>.{kmd,ued,uhd,udd}.json, flydsl_<op>_kernel_match.umd.json
                              a fixed handful per op -- however many objects,
                              knobs, targets or arches it gains
  <target>.SOURCE.md          one per target: the toolchain, the input digest,
                              and one short row per object (name, priority,
                              VGPRs, LDS, SHA256 prefix) -- the diff a review
                              of a regeneration reads
  <target>/                   generated output: the pack (KDP) with one inline
                              UKD per object, the objects, manifest.json --
                              binaries and long lists nobody reviews line by line
```

`<target>` is a generic family (`gfx11-generic`, `gfx12-generic`) or a concrete
arch (`gfx950`, `gfx942`, `gfx90a` for the CDNA SDPA ops); both can sit side by side under
one op, each its own folder and summary, as the provider grows arch-specific
objects beside a family set.

**Who changes what.** A contribution changes the *inputs*: the kernel sources
under `kernels_src/`, the generators and their instance table, the docs. The
per-target folders and summaries are *outputs*, written only by the generators
(REGEN.md §3). Every manifest records a digest of the inputs its op was generated
from (`inputs_sha256`): its `gen_<op>.py` and every `generators/` module it
reaches through imports, the op's own kernel directories, the shared kernel
helpers (`kernels/common`), and its own rows of the three tables -- the op
registry (`generators/_ops.py` `OP_KERNEL_DIRS`, where a new op adds its row), the
instance table, and its target's row of `arch_families.json`. Registering an op,
or adding a module only some ops import, leaves every other op's objects current;
`tests/test_inputs_digest.py` refuses an input the digest could miss (a dynamic
import, a data file, a kernel import outside the op's declared directories). The
build's `gen_descriptors.py --check` fails when
the tree's inputs no longer match -- so a source change without a regeneration
cannot ship objects built from other sources. Ops go stale together when they share
inputs: `rmsnorm` and `rmsnorm_bwd` both digest the whole `norm/` directory, so an
edit to the forward-only `rmsnorm_kernel.py` invalidates the backward objects too,
and every op digests the shared generator machinery and kernel helpers. The digest hashes bytes, so
a comment-only edit to any digested file also needs a regeneration (which then
reproduces byte-identical objects). Regeneration is
byte-reproducible against the pinned toolchain: a contributor who regenerates
gets exactly the bytes a maintainer would, and a maintainer can regenerate a
contribution to confirm it.

## The provenance chain

Each link is checkable by a command, and [REGEN.md](REGEN.md) gives each command:

```
upstream FlyDSL @ 89ad52fbbb9e, AITER @ 8253efc40595
   │  tools/diff_upstream.py          — vendored copy == upstream, but for recorded modifications
kernels_src/
   │  gen_<op>.py (flydsl 0.3.4)      — byte-reproducible: 465/465 gfx11-generic, 613/613 gfx12-generic, 327/327 gfx950, 329/329 gfx942, 345/345 gfx90a
<content>/<op>/<arch>/*.hsaco + manifest.json
   │  gen_descriptors.py --check      — objects match manifest SHA256 and target; descriptors agree
<content>/<op>/**/*.json (KDPs of inline hsaco UKDs)
   │  shared packer, then tools/check_shards.py — every member shard ships those bytes
arch_content/hip-kernel-provider/<arch>/
   │  TestFlydsl*Packs + census       — the shard is really there, and is what the suite read
```

The last link is the one that is easy to leave out and expensive to be without:
a test suite that builds its graphs in-process stays green against an **empty**
shard. The census exists to fail in exactly that case.

Byte-reproducibility is what lets the whole chain be re-derived rather than
trusted. Two consequences worth stating plainly:

- Objects that regenerate to **different bytes** are a *change to what this
  provider ships*, not a refresh — commit them as such, with objects, manifest,
  `<target>.SOURCE.md` and descriptors moving together ([REGEN.md §4](REGEN.md)).
- The repo's `black` hook reformats `kernels_src/`. That is accepted because the
  reformat was verified codegen-neutral; re-run that verification after any bulk
  reformat ([REGEN.md §2](REGEN.md)).
