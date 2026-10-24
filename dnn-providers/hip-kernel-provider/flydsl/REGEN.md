<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# Regenerating the shipped FlyDSL kernels

The `.hsaco` code objects, their `manifest.json` and `<arch>.SOURCE.md`, and the
descriptor JSON beside them are **generated and committed**, as FlyDSL's bundle
of the provider's production descriptor root:
`src/engines/kernel_ingestor_engine/descriptors/FlyDSL/<op>/<arch>/` (and
`FlyDSL/<op>/` for the descriptors shared across arches). Below, `$CONTENT`
names that `FlyDSL/` folder. The build's shared packer packs and stages them
like every other bundle in that root; it never compiles them.
This file is how you reproduce them, and how you check that a tree you did not
build still matches the toolchain it claims.

Regeneration is **reproducible**: the pinned toolchain in
[`generators/_flydsl_env.py`](generators/_flydsl_env.py) produces byte-identical
objects. Verifying that is the first procedure below, and it is the one to run
after touching anything under `kernels_src/`.

Every manifest records a digest of the bytes of its op's inputs (`inputs_sha256`), and
the build's `--check` fails when they no longer match. So even a **comment-only** edit
to a digested file -- anything under `kernels_src/`, or a shared module under
`generators/` -- needs a regeneration per §3; it should reproduce byte-identical
objects, with only the digest recorded in `manifest.json` and `<arch>.SOURCE.md` moving. Edits to `_instances.py` count only
through the op's own rows, so a comment there does not.

> An object that regenerates to *different* bytes is a **change** to what this
> provider ships, not a refresh. See [When the bytes differ](#when-the-bytes-differ).

---

## 1. Environment

Everything below runs from the provider's `flydsl/` directory.

| Component | Pinned value | Where the pin lives |
|---|---|---|
| Python | 3.12 | — |
| `flydsl` | **0.3.4** (pip wheel) | `generators/_flydsl_env.py` `FLYDSL_VERSION` |
| FlyDSL kernel sources | commit `89ad52fbbb9e…` (`v0.3.4.1-19-g89ad52f`) | `_flydsl_env.py` `FLYDSL_KERNELS_*`, and vendored under `kernels_src/` |
| ROCm | 7.13.0 | read at runtime from `$ROCM_PATH/.info/version` |
| `torch` | any ROCm build | imported by the vendored kernel sources |
| `msgpack` | any (the tree was built with 1.2.2) | reads the AMDGPU metadata note out of each object |

`torch` and `msgpack` are **not** transitive dependencies of the `flydsl`
wheel — install them explicitly. Regeneration packs nothing: the archive is
written by the build's shared packer, so `rocm_kpack` is not needed here.

```bash
python3 -m venv /path/to/flydsl-regen-venv
/path/to/flydsl-regen-venv/bin/pip install 'flydsl==0.3.4' torch msgpack
```

`flydsl` compiles but does not *dispatch* here, so **no matching GPU is
required** — `prepare()` sets `COMPILE_ONLY=1`, which is what lets a gfx942
object be produced on a machine with an RDNA GPU, or none.

### The three settings every command below reads

Set these once per shell. The rest of this file uses them verbatim, so a
procedure you paste runs against your toolchain rather than someone else's:

```bash
PY=${PY:?the interpreter holding the pinned flydsl wheel}
CONTENT=../src/engines/kernel_ingestor_engine/descriptors/FlyDSL
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export REGEN_OUT=${REGEN_OUT:-/tmp/flydsl-regen}
```

`PY` has no default on purpose: the generators need the pinned `flydsl` wheel,
and a bare `python3` without it would fail later, at the first import, rather
than here. `assert_flydsl_version()` then refuses any wheel but the pinned one.

The tree as committed was produced with Python 3.12.3, `flydsl` 0.3.4,
`torch` 2.10.0+rocm7.13.0a20260513 and ROCm 7.13.0. Those versions are the
contract; where they live on disk is not.

`ROCM_PATH` matters beyond finding the compiler: `rocm_version()` reads
`$ROCM_PATH/.info/version` and writes it into `manifest.json` and `<arch>.SOURCE.md`.
With neither `ROCM_PATH` nor `ROCM_VERSION` set it records the literal
`"unknown"` rather than guessing — an unrecorded toolchain is better than a
wrong one, because a wrong one reads as verified.

### Generators run as modules

`generators/gen_rmsnorm.py` opens `from . import _flydsl_env as env`, so running
it as a script dies with `ImportError: attempted relative import with no known
parent package`. Invoke it from `flydsl/` as `python -m generators.gen_rmsnorm`.
`gen_descriptors.py` is a plain script and takes either form.

### Generators compile in parallel

Every generator compiles its objects in a pool of fresh worker processes
(`generators/_parallel.py`), each prepared as the generator's own process is; the
objects come back in table order and are checked, written and recorded serially, so
the bytes are a serial build's. `FLYDSL_GEN_JOBS` sets the worker count (default:
half the CPUs, since each worker holds a compile in memory; `1` compiles in the
generator's own process). On a 32-thread machine every op and arch regenerates in
about five minutes instead of twenty-five.

---

## 2. Verify the shipped objects reproduce

This writes nothing into the tree. It needs the content directories pulled
([README.md](README.md) *Fetching the objects*). It is the check that backs every claim this
directory makes about its own provenance.

Each target has its own ops; pick one row and set `ARCH` and `OPS` from it:

| `ARCH` | `OPS` | Objects |
|---|---|---|
| `gfx11-generic` | `rmsnorm rmsnorm_bwd sdpa_rdna sdpa_rdna_decode` | 465 (90 + 63 + 204 + 108) |
| `gfx12-generic` | `rmsnorm rmsnorm_bwd sdpa_rdna sdpa_rdna_decode sdpa_aiw` | 613 (90 + 63 + 156 + 108 + 196) |
| `gfx950` | `rmsnorm rmsnorm_bwd sdpa_cdna sdpa_cdna_dualwave sdpa_cdna_decode` | 327 (90 + 63 + 130 + 20 + 24) |
| `gfx942` | `rmsnorm rmsnorm_bwd sdpa_cdna sdpa_cdna_decode` | 329 (90 + 63 + 152 + 24) |
| `gfx90a` | `rmsnorm rmsnorm_bwd sdpa_cdna sdpa_cdna_decode` | 345 (90 + 63 + 168 + 24) |

```bash
ARCH=gfx11-generic OPS="rmsnorm rmsnorm_bwd sdpa_rdna sdpa_rdna_decode"

# Compile into a scratch directory -- must be empty or nonexistent, since the
# generator writes files rather than clearing the directory, so an object left
# by an earlier run with a different instance list would survive and be compared.
for op in $OPS; do
  PYTHONDONTWRITEBYTECODE=1 \
    $PY -m generators.gen_$op --arch $ARCH --out-dir "$REGEN_OUT"
done
```

Then compare every object against the manifests the tree ships:

```bash
CONTENT=$CONTENT ARCH=$ARCH OPS="$OPS" $PY - <<'PY'
import hashlib, json, os, pathlib, sys

bad = total = 0
arch = os.environ["ARCH"]
for op in os.environ["OPS"].split():
    manifest = json.loads((pathlib.Path(os.environ["CONTENT"]) / f"{op}/{arch}/manifest.json").read_text())
    regen = pathlib.Path(os.environ["REGEN_OUT"]) / f"{op}/{arch}"
    for instance in manifest["instances"]:
        total += 1
        obj = regen / instance["file"]
        digest = hashlib.sha256(obj.read_bytes()).hexdigest() if obj.is_file() else "<missing>"
        if digest != instance["sha256"]:
            print(f"DIFFERS {op}/{instance['name']}\n  manifest {instance['sha256']}\n  regen    {digest}")
            bad += 1
print(f"{total - bad}/{total} byte-identical")
sys.exit(1 if bad else 0)
PY
```

The manifests and the provenance record must also come out identical:

```bash
for op in $OPS; do
  diff "$CONTENT/$op/$ARCH/manifest.json" "$REGEN_OUT/$op/$ARCH/manifest.json"
  diff "$CONTENT/$op/$ARCH.SOURCE.md"     "$REGEN_OUT/$op/$ARCH.SOURCE.md"
done
```

Expected, and the state of the tree as committed: every object byte-identical -- the
table's count for each target -- and every diff empty.

`manifest.json` agreeing is the stronger of the two checks — it carries the
`toolchain` block, so an identical manifest means the *recorded* toolchain and
the *actual* one are the same. Matching objects under a mismatched manifest
would mean the record is wrong.

### Re-run this after any bulk reformat of `kernels_src/`

The repo's `black` hook rewrites the vendored sources, and that is accepted; the
reformat was verified codegen-neutral by exactly the procedure above. The
checked-in binaries and their manifest are only trustworthy while that holds, so
the verification is the thing that keeps the acceptance honest.

---

## 3. Regenerate for real

Only when the kernels actually change: a `kernels_src/` edit, a new instance row,
a new arch, a deliberate toolchain bump.

```bash
# (1) Compile. Writes $CONTENT/<op>/<arch>/*.hsaco + manifest.json, and
#     $CONTENT/<op>/<arch>.SOURCE.md
#     for each op. The SDPA kernels carry the gfx11 (RDNA3 / RDNA3.5) and gfx12
#     (RDNA4) WMMA ABIs; gen_sdpa_rdna refuses any other arch before compiling.
for ARCH in gfx11-generic gfx12-generic; do
  $PY -m generators.gen_rmsnorm --arch $ARCH
  $PY -m generators.gen_rmsnorm_bwd --arch $ARCH
  $PY -m generators.gen_sdpa_rdna --arch $ARCH
  $PY -m generators.gen_sdpa_rdna_decode --arch $ARCH
  # AOTriton's gfx1201 kernel, its own engine (gfx12-generic only).
  [ $ARCH = gfx12-generic ] && $PY -m generators.gen_sdpa_aiw --arch $ARCH

  # (2) Descriptors. Every field is derived from the manifest and the objects it
  #     names, so a descriptor cannot disagree with the object it describes.
  $PY gen_descriptors.py --arch $ARCH

  # (3) Verify the set. This is the same invocation the build runs before packing.
  $PY gen_descriptors.py --arch $ARCH --check
done

# The CDNA SDPA kernels are MFMA code, built per arch, one engine per kernel family:
# FlyDSL's generic kernel (every arch), its dual-wave kernel (gfx950 only; the
# generator builds nothing for gfx942 or gfx90a) and the generic kernel's split-KV decode
# objects. Each refuses any other arch before compiling. RMSNorm builds for every arch.
for ARCH in gfx950 gfx942 gfx90a; do
  $PY -m generators.gen_rmsnorm --arch $ARCH
  $PY -m generators.gen_rmsnorm_bwd --arch $ARCH
  $PY -m generators.gen_sdpa_cdna --arch $ARCH
  $PY -m generators.gen_sdpa_cdna_dualwave --arch $ARCH
  $PY -m generators.gen_sdpa_cdna_decode --arch $ARCH
  $PY gen_descriptors.py --arch $ARCH
  $PY gen_descriptors.py --arch $ARCH --check
done
```

The CDNA kernels take FlyDSL's `fx.Tensor` arguments, so the CDNA generators trace
each from torch `meta` tensors -- shapes and strides with no storage. No GPU is
touched, and the object is identical to one traced from device tensors, because
every tensor is lowered with a dynamic layout.

**Why `gfx11-generic`.** It is an LLVM *generic* target: one object set that the
loader accepts on every RDNA3 and RDNA3.5 part -- gfx1100, gfx1101, gfx1102,
gfx1103, gfx1150, gfx1151, gfx1152 and gfx1153 (`generators/arch_families.json`;
gfx1170/gfx1171 belong to `gfx11-7-generic` and are not covered). The
descriptors list all eight, and the packer copies the same bytes into each of
those arches' shards, so every device finds them under its own name with no
runtime support for generic names. It needs code object v6 and ROCm 6.4 or
later at load time.

FlyDSL 0.3.4 cannot name a generic target to MLIR, so `prepare()` installs
`generators/_generic_targets.py`, a shim pinned to that release, which hands
MLIR the family's lowest member (`gfx1100`) as its chipset while the object is
still built for, and stamped as, `gfx11-generic`. The manifest's toolchain block
records it (`generic_target_shim`). `gen_sdpa_rdna` also builds the generic objects
with `amdgpu-use-amdgpu-trackers`, added to the kernel's own LLVM options (the
effective set is recorded per instance as `llvm_options`):
without it the d128 causal objects spill a few VGPRs, because the generic ISA
has no gfx115x scalar-float instructions.

**`gfx12-generic`** (gfx1200, gfx1201) works the same way, with one more step.
FlyDSL 0.3.4's own arch checks classify RDNA by name prefix (`gfx120`), which
the generic name fails: told `gfx12-generic` it picks no WMMA atom and CDNA
buffer-descriptor flags -- silently, for RMSNorm. The family's row in
`arch_families.json` therefore names a `flydsl_gpu_arch` (gfx1200) that
`prepare()` gives those checks, while the object is still built for and stamped
as `gfx12-generic`; `prepare()` refuses a build whose checks would not see RDNA.
The kernels pick their WMMA operand layout from that same arch (attention-kernel
modification 17). gfx12 needs no `amdgpu-use-amdgpu-trackers`: every instance
fits without a spill under the default scheduler.

A concrete gfx11 or gfx120x arch still works (`--arch gfx1151`) if a part ever
needs its own objects; its directory then ships only to that arch.

Steps 1 and 2 are per-arch and **one arch per invocation** — FlyDSL reads `ARCH`
from the environment at each compile, so a multi-arch run could file an object
under a name it was not built for. The generator additionally re-reads each
object's own `amdhsa.target` and refuses a mismatch, and for a generic target
checks the ELF header carries the generic machine too.

The objects, the manifest, the KDP, `<arch>.SOURCE.md` and the op's descriptors are
**one unit**, committed in one change. The `<op>/<arch>/` directory is a DVC output,
so re-record it and upload it to the `ingestor` remote before committing, from the
repository root (`dvc push` needs write credentials, set with
`dvc remote modify --local ingestor ...`; reads are anonymous):

```bash
dvc add  $CONTENT/<op>/<arch>                    # rewrites <op>/<arch>.dvc with the new content hash
dvc push -r ingestor $CONTENT/<op>/<arch>.dvc    # uploads the directory
git add  $CONTENT/<op>/<arch>.dvc $CONTENT/<op>/<arch>.SOURCE.md $CONTENT/<op>/*.json
```

A new `<op>/<arch>/` needs the same `dvc add`, which also lists the directory in
`<op>/.gitignore`, and then `printf '  remote: ingestor\n' >> $CONTENT/<op>/<arch>.dvc`
so a bare pull finds it. Push before the PR goes up: CI pulls by the hash in the
sidecar, and a sidecar whose content was never pushed fails every pull.

They must agree: the build's `--check` fails on a descriptor that disagrees
with its manifest, on an object whose bytes no longer match the SHA256 its
manifest records, and on one built for a processor other than its directory's.
The shared packer itself checks neither bytes nor target; that is why `--check`
runs before every pack.

### Packing

Packing is a **build step**, done by the provider's shared packer
(`descriptor-packaging/`) over the whole production root, FlyDSL bundle
included; the archive is not committed. To check it, build the product pack and
look in the staged shard:

```bash
cmake --build build --target hkp_packaging_product
ls build/lib/hipdnn_plugins/engines/arch_content/hip-kernel-provider/gfx1151/kpack/
```

Every producer's objects for an arch share that one
`hip_kernel_provider_<arch>.kpack`, and the `gfx11-generic` objects land in the
shard of every member arch the build packs for. The packer reads the argument
signature out of the object itself, so the descriptors never carry a
hand-written one. After the pack, the build runs `tools/check_shards.py`
(target `flydsl_shard_check`): every packed arch's shard must hold one shipped
UKD per checked-in object, inline in its target's KDP, packed from that object and carrying its SHA256.

---

## 4. When the bytes differ

Work through these in order; the first three are environment, the last is a real
change.

1. **`flydsl` version.** `assert_flydsl_version()` hard-fails on anything other
   than the pin, naming this file. If you hit that error, you have the wrong
   wheel — installing the pinned one is the fix, *not* editing `FLYDSL_VERSION`.
2. **ROCm version.** Not asserted, only recorded — so this one fails silently
   into the manifest diff rather than loudly. FlyDSL's own release notes and this
   tree's Flash2 precedent both record ~1.4× performance swings from the ROCm
   version alone on identical source, so a ROCm delta is a real difference in
   what ships, not noise. Check the `toolchain` block in the manifest diff first.
3. **`kernels_src/` drift.** Either a local edit, or an upstream re-vendor that
   did not update the pins. `tools/diff_upstream.py` answers this against an
   upstream checkout.
4. **A genuine kernel change.** Then the new bytes are the point: regenerate per
   §3, `dvc add` and `dvc push` the directory, commit its sidecar + `<arch>.SOURCE.md` +
   descriptors in one change, and
   say in the message what moved. A regenerated object committed as a refresh
   reviews as a no-op diff while changing what every user runs.

Bumping a pin is a deliberate act with a cost attached: `FLYDSL_VERSION` is
per-provider, not per-arch, so changing it obliges regenerating **every** target
under `$CONTENT/<op>/`. Leaving one behind ships a tree whose `<arch>.SOURCE.md` files disagree
about which compiler built it.

---

## 5. Checking the vendored sources against upstream

`kernels_src/` is a vendored copy of twenty kernel modules from three upstreams.
Thirteen come from FlyDSL; the RDNA attention kernel comes from AITER, which carries
FlyDSL kernels upstream FlyDSL does not; and the RDNA4 attention package
(`attention_rdna4/`, six modules) comes from AOTriton. Deliberate modifications are
recorded beside the code: `kernels_src/kernels/norm/rmsnorm_kernel.py` (four),
`kernels_src/kernels/norm/rmsnorm_bwd_kernel.py` (two) and
`kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py` (sixteen — the gfx11
port and the runtime arguments that let a few objects cover many shapes) in their
own headers, the RDNA4 package's in `kernels_src/kernels/attention_rdna4/README.md`, and the
CDNA attention files' in `kernels_src/kernels/attention_cdna/README.md`.
A modified file's header also names the upstream and path it came from -- the renamed
`attention_rdna/` and `attention_cdna/` packages both map back to upstream's
`kernels/attention/`; the modules of `attention_rdna4/` carry none and inherit their
package's, which names AOTriton's flat `modules/flash/flyc/` directory -- and the tool
reads that header to decide what to diff it against. A file with no provenance header
(the unmodified FlyDSL modules carry only FlyDSL's SPDX and copyright lines) is diffed
against the same relative path in the FlyDSL checkout. The FlyDSL and AITER commits are
pinned in `generators/_flydsl_env.py`. To see what has drifted:

```bash
python3 tools/diff_upstream.py --upstream /path/to/FlyDSL --aiter /path/to/aiter \
    --aotriton /path/to/aotriton --commit
```

Expected against the pinned checkouts, and the state of the tree as committed:

```
FlyDSL (v0.3.4.1-19-g89ad52f) at the pinned commit 89ad52fbbb9e
AITER at the pinned commit 8253efc40595
AOTriton at the pinned commit b5e8cfb909e5
=== kernels/attention_cdna/__init__.py
  …
=== kernels/attention_cdna/flash_attn_generic.py
  …
=== kernels/attention_cdna/flash_attn_gfx950.py
  …
=== kernels/attention_cdna/flash_attn_utils.py
  …
=== kernels/attention_rdna/__init__.py
  …
=== kernels/attention_rdna/flash_attn_func_rdna.py
  …
=== kernels/attention_rdna4/flash_attn_func_gfx1201_aiw.py
  …
=== kernels/attention_rdna4/flyc_polyfill.py
  …
=== kernels/attention_rdna4/fmha_abi_gfx1201.py
  …
=== kernels/attention_rdna4/fmha_common_gfx1201.py
  …
=== kernels/attention_rdna4/fmha_tuning_gfx1201.py
  …
=== kernels/norm/rmsnorm_bwd_kernel.py
  …
=== kernels/norm/rmsnorm_kernel.py
  …
7/20 vendored files identical to upstream after normalization; 13 differ, 0 absent upstream, 0 not compared
```

Without `--aiter` or `--aotriton`, that upstream's files are reported as not compared
rather than as absent. The RDNA4 package's `__init__.py` is this tree's own (upstream's
directory has none) and is not compared. The differing files are the point of the
tool, not a failure — read each one's hunks against its recorded modifications. Every
one of them differs at least by the provenance header it carries, or in
`attention_rdna4/` by the package imports of its modification 0; the two `__init__.py` files differ by nothing
else. Anything those records do not account for is drift: either a local edit nobody
wrote down, or an upstream re-vendor that did not update the pins in
`generators/_flydsl_env.py` (AOTriton's commit is recorded in its package's
`__init__.py`, since no generator reads it). Exit status reports whether the
comparison held together, not whether files matched: non-zero means a vendored file
had **no upstream counterpart**, i.e. this is not the checkout the tree was taken from
and no hunk can be trusted.

It **normalizes both sides with `black` before diffing**. This is a requirement,
not a nicety: the repo's hook reformats our copy, so an un-normalized diff shows
a whole-file delta on every vendored file and the real semantic drift is
invisible inside it. The tool refuses to run if `black` is unavailable rather
than producing that diff. It needs no flydsl and no GPU — any Python 3.10+ with
`black` on `PATH` (or importable as `python -m black`) will do, which is why the
command above is not `$PY`.

Pass `--commit` to assert the checkout sits at the pinned commit. Without it the
tool warns with the actual `HEAD` and continues, since comparing against a
*different* commit is sometimes exactly what you want — it shows what upstream
has done since — but then the hunks mix local modifications with upstream's own
movement and cannot be read as drift.
