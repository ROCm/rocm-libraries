<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# Regenerating the checked-in FlyDSL kernels

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
| `msgpack` | 1.2.2 | reads the AMDGPU metadata note out of each object |

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

---

## 2. Verify the checked-in objects reproduce

This writes nothing into the tree. It is the check that backs every claim this
directory makes about its own provenance.

```bash
cd dnn-providers/hip-kernel-provider/flydsl

# Compile into a scratch directory -- must be empty or nonexistent, since the
# generator writes files rather than clearing the directory, so an object left
# by an earlier run with a different instance list would survive and be compared.
for op in rmsnorm sdpa_rdna; do
  PYTHONDONTWRITEBYTECODE=1 \
    $PY -m generators.gen_$op --arch gfx11-generic --out-dir "$REGEN_OUT"
done
```

Then compare every object against the manifests the tree ships:

```bash
CONTENT=$CONTENT $PY - <<'PY'
import hashlib, json, os, pathlib, sys

bad = total = 0
for op in ("rmsnorm", "sdpa_rdna"):
    manifest = json.loads((pathlib.Path(os.environ["CONTENT"]) / f"{op}/gfx11-generic/manifest.json").read_text())
    regen = pathlib.Path(os.environ["REGEN_OUT"]) / f"{op}/gfx11-generic"
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
for op in rmsnorm sdpa_rdna; do
  diff "$CONTENT/$op/gfx11-generic/manifest.json" "$REGEN_OUT/$op/gfx11-generic/manifest.json"
  diff "$CONTENT/$op/gfx11-generic.SOURCE.md"     "$REGEN_OUT/$op/gfx11-generic.SOURCE.md"
done
```

Expected, and the state of the tree as committed: **140/140 byte-identical** (12
RMSNorm, 128 SDPA), all four diffs empty. Repeat with `gfx12-generic` in place of
`gfx11-generic` throughout: **140/140** there too.

The CDNA SDPA op is built per arch rather than per family. The same three steps,
with `sdpa_cdna` as the only op and `gfx950`, then `gfx942`, as the target:
**2/2** for each.

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
cd dnn-providers/hip-kernel-provider/flydsl

# (1) Compile. Writes $CONTENT/<op>/<arch>/*.hsaco + manifest.json, and
#     $CONTENT/<op>/<arch>.SOURCE.md
#     for each op. The SDPA kernels carry the gfx11 (RDNA3 / RDNA3.5) and gfx12
#     (RDNA4) WMMA ABIs; gen_sdpa_rdna refuses any other arch before compiling.
for ARCH in gfx11-generic gfx12-generic; do
  $PY -m generators.gen_rmsnorm --arch $ARCH
  $PY -m generators.gen_sdpa_rdna --arch $ARCH

  # (2) Descriptors. Every field is derived from the manifest and the objects it
  #     names, so a descriptor cannot disagree with the object it describes.
  $PY gen_descriptors.py --arch $ARCH

  # (3) Verify the set. This is the same invocation the build runs before packing.
  $PY gen_descriptors.py --arch $ARCH --check
done

# The CDNA SDPA kernels are MFMA code, built per arch: FlyDSL's dual-wave kernel
# for gfx950 and its generic kernel for gfx942. gen_sdpa_cdna refuses any other
# arch before compiling.
for ARCH in gfx950 gfx942; do
  $PY -m generators.gen_sdpa_cdna --arch $ARCH
  $PY gen_descriptors.py --arch $ARCH
  $PY gen_descriptors.py --arch $ARCH --check
done
```

The CDNA kernels take FlyDSL's `fx.Tensor` arguments, so `gen_sdpa_cdna` traces
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

Commit the objects, the manifest, `<arch>.SOURCE.md` and the descriptors **together**.
They are one unit: the build's `--check` fails on a descriptor that disagrees
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
UKD per checked-in object, packed from that object and carrying its SHA256.

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
   §3, commit objects + manifest + `<arch>.SOURCE.md` + descriptors in one change, and
   say in the message what moved. A regenerated object committed as a refresh
   reviews as a no-op diff while changing what every user runs.

Bumping a pin is a deliberate act with a cost attached: `FLYDSL_VERSION` is
per-provider, not per-arch, so changing it obliges regenerating **every** arch in
`kernels/`. Leaving one behind ships a tree whose `<arch>.SOURCE.md` files disagree
about which compiler built it.

---

## 5. Checking the vendored sources against upstream

`kernels_src/` is a vendored copy of fourteen kernel modules from two upstreams.
Thirteen come from FlyDSL; the RDNA attention kernel comes from AITER, which
carries FlyDSL kernels upstream FlyDSL does not. Deliberate modifications are
recorded beside the code: `kernels_src/kernels/norm/rmsnorm_kernel.py` (three) and
`kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py` (sixteen — the gfx11
port and the runtime arguments that let a few objects cover many shapes) in their
own headers, and the CDNA attention files in
`kernels_src/kernels/attention_cdna/README.md` (an import path, and a runtime row
guard). Each file's header names the upstream and path it came from -- the
renamed `attention_rdna/` and `attention_cdna/` packages both map back to
upstream's `kernels/attention/` -- and the tool reads that header to decide what
to diff it against. To see what has drifted:

```bash
python3 tools/diff_upstream.py --upstream /path/to/FlyDSL --aiter /path/to/aiter --commit
```

Expected against the pinned checkouts, and the state of the tree as committed:

```
FlyDSL (v0.3.4.1-19-g89ad52f) at the pinned commit 89ad52fbbb9e
AITER at the pinned commit 8253efc40595
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
=== kernels/norm/rmsnorm_kernel.py
  …
7/14 vendored files identical to upstream after normalization; 7 differ, 0 absent upstream, 0 not compared
```

Without `--aiter`, the RDNA attention file is reported as not compared rather
than as absent. The differing files are the point of the tool, not a failure —
read each one's hunks against its recorded modifications. Every one of them
differs at least by the provenance header it carries; the two `__init__.py`
files differ by nothing else. Anything that header does not
account for is drift: either a local edit nobody wrote down, or an upstream
re-vendor that did not update the pins in `generators/_flydsl_env.py`. Exit
status reports whether the comparison held together, not whether files matched:
non-zero means a vendored file had **no upstream counterpart**, i.e. this is not
the checkout the tree was taken from and no hunk can be trusted.

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
