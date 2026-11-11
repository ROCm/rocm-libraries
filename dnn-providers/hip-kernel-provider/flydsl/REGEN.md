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
| ROCm | 7.13.0; the `sdpa_bwd_*` ops are built with 10.2.0, so they need a separate `ROCM_PATH`. FlyDSL generates code with its own LLVM and takes only `ld.lld` from ROCm, which writes its version into each object's `.comment` section: another ROCm release changes every `sdpa_bwd_*` object's sha256, and the pin gate refuses it. The objects run on newer ROCm runtimes unchanged | read at runtime from `$ROCM_PATH/.info/version` |
| LLVM tools (`llvm-objdump`) | the `sdpa_bwd_*` gate and `tools/analyze.py` take them from `$LLVM_BIN`, else `$ROCM_PATH/llvm/bin` | `tools/analyze.py` `llvm_bin()` |
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
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}   # 7.13.0 install, for every op but sdpa_bwd_*
export ROCM_PATH_BWD=${ROCM_PATH_BWD:-$ROCM_PATH}   # 10.2.0 install, for the sdpa_bwd_* ops
export REGEN_OUT=${REGEN_OUT:-/tmp/flydsl-regen}
export REF_ROOT=${REF_ROOT:-/tmp/flydsl-regen-pins}   # the pinned objects, one directory per arch
```

The `sdpa_bwd_{prep,dkdv,dq,merged}` generators also take the pinned objects, one directory
per arch (`$REF_ROOT/<arch>`). `tools/sdpa_bwd_s1.py --from-content` builds it from the pulled
content and the tree's data, with no other input (§2).

`PY` has no default on purpose: the generators need the pinned `flydsl` wheel,
and a bare `python3` without it would fail later, at the first import, rather
than here. `assert_flydsl_version()` then refuses any wheel but the pinned one.

The tree as committed was produced with Python 3.12.3, `flydsl` 0.3.4 and
`torch` 2.10.0+rocm7.13.0a20260513, with ROCm 7.13.0 for every op except the
`sdpa_bwd_*` ones, which were built with ROCm 10.2.0 (their manifests record it). Those
versions are the contract; where they live on disk is not. A `sdpa_bwd_*` command below
therefore runs with `ROCM_PATH` naming a 10.2.0 install, and every other op with a 7.13.0 one.

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
| `gfx11-generic` | `rmsnorm rmsnorm_bwd sdpa_rdna sdpa_rdna_decode sdpa_dropout_mask sdpa_head_pad sdpa_bwd_prep sdpa_bwd_dkdv sdpa_bwd_dq` | 1,084 (90 + 63 + 312 + 116 + 1 + 1 + 33 + 238 + 230; the dQ op also writes 3 standalone reduce objects, its manifest's `companions`, which the comparison below does not count) |
| `gfx12-generic` | `rmsnorm rmsnorm_bwd sdpa_rdna sdpa_rdna_decode sdpa_aiw sdpa_dropout_mask sdpa_head_pad` | 719 (90 + 63 + 192 + 116 + 256 + 1 + 1) |
| `gfx950` | `rmsnorm rmsnorm_bwd sdpa_cdna sdpa_cdna_dualwave sdpa_cdna_decode sdpa_dropout_mask sdpa_head_pad sdpa_bwd_prep sdpa_bwd_dkdv sdpa_bwd_dq sdpa_bwd_merged` | 3,139 (90 + 63 + 248 + 26 + 24 + 1 + 1 + 86 + 1,329 + 962 + 309) |
| `gfx942` | `rmsnorm rmsnorm_bwd sdpa_cdna sdpa_cdna_decode sdpa_dropout_mask sdpa_head_pad` | 457 (90 + 63 + 278 + 24 + 1 + 1) |
| `gfx90a` | `rmsnorm rmsnorm_bwd sdpa_cdna sdpa_cdna_decode sdpa_dropout_mask sdpa_head_pad` | 465 (90 + 63 + 286 + 24 + 1 + 1) |

`sdpa_bwd_dbias_reduce` and `sdpa_bwd_dsink_reduce` ship no content (both reduce kernels ride
inside the dQ and merged objects), so no `OPS` list names them; their generators only build the
standalone reduce pins that the S1 gates compare against.

`sdpa_bwd_tailzero` ships no content of its own either: its six objects (gfx950 and gfx11-generic, each f16, bf16, fp32) and its two
V-head objects (role `tailzero_hv`, one per arch; the f16 and bf16 entries name the same bytes) are
companions built by the dQ generator and listed by the merged one (§3, *Companions*).

The f32-bias, 16-bit direct-bias dQ and split K/V head rows (79 on gfx950, 72 on gfx11-generic) are ordinary rows of these ops: each is a sweep pin or ships under the
nondeterminism policy below. The rows with separate V heads (`u2`) are their own packs, `<arch>.u2.kdp.json` (dK-only and dV-only dK/dV) and
`<arch>.dq_u2.kdp.json` (V-head dQ), and the rows that store dB through a reduce are `<arch>.reduce.kdp.json` of dQ and merged, written by `gen_descriptors.py`; what they serve is in [COVERAGE.md §9.2](COVERAGE.md).

```bash
ARCH=gfx11-generic OPS="rmsnorm rmsnorm_bwd sdpa_rdna sdpa_rdna_decode sdpa_dropout_mask sdpa_head_pad sdpa_bwd_prep sdpa_bwd_dkdv sdpa_bwd_dq"

# The pins of the backward ops: the shipped objects the arch's reference CSV names, copied out of
# $CONTENT. The directory must not exist (the tool refuses to replace one), so a repeated run removes it first;
# the tool fails on a pin it can neither copy nor name by digest.
case "$OPS" in *sdpa_bwd_*)
  rm -rf "$REF_ROOT/$ARCH"
  $PY tools/sdpa_bwd_s1.py --from-content "$CONTENT" --arch $ARCH --reference-dir "$REF_ROOT/$ARCH" ;;
esac

# Compile into a scratch directory -- must be empty or nonexistent, since the
# generator writes files rather than clearing the directory, so an object left
# by an earlier run with a different instance list would survive and be compared.
for op in $OPS; do
  case $op in
    sdpa_bwd_prep|sdpa_bwd_dkdv|sdpa_bwd_dq|sdpa_bwd_merged)
      pins=(--reference-dir "$REF_ROOT/$ARCH" --stored-dir "$CONTENT"); op_rocm=$ROCM_PATH_BWD ;;
    *) pins=(); op_rocm=$ROCM_PATH ;;
  esac
  ROCM_PATH=$op_rocm PYTHONDONTWRITEBYTECODE=1 \
    $PY -m generators.gen_$op --arch $ARCH --out-dir "$REGEN_OUT" "${pins[@]}"
done
```

The four `sdpa_bwd_{prep,dkdv,dq,merged}` generators exit 1 without `--reference-dir`, the
directory of the pinned `<hash>.hsaco` objects. They ship their own build, which
matches the pin byte for byte on every row but the rows of the nondeterministic list
(`tests/data/sdpa_bwd_nondeterministic.csv`, below) and the rows of `tests/data/sdpa_bwd_pinned_exceptions.csv`
(known-bad rebuilds, append-only): those ship a stored pin (for a combined row the combined object of the shipped content,
read through `--stored-dir`, the content root, below), and their `manifest.json` entry and the `Pinned objects`
table of `<arch>.SOURCE.md` say so. A row outside both lists that does not build, whose build is not its pin, or
whose build spills (R-MIR) or was not scanned, stops the run. The R-MIR policy on gfx950: a freshly built object is always
scanned and `not scanned` fails; `tools/rmir_known.csv` exempts no merged spec; a stored pin is recorded `not scanned` and ships only
when its row of `tests/data/sdpa_bwd_reference.csv` records the pin's own scan in `rmir_scan` (`clean`: a sweep scan whose dump build
is the pinned object, with no finding; `exception`: a row of `sdpa_bwd_pinned_exceptions.csv`), else the run stops; a nondeterministic pin that
is its own red rebuild stops the run; a pinned exception is covered by a device screen of the pin, cited in its `evidence`. The S1 tool
(`--from-content`, and the sweep mode that writes the column) takes a gfx950 pin only with that record. The detector reads gfx950 MIR, so gfx11 is out of scope: a gfx11
object has no R-MIR scan and is held to its pin byte for byte. The backward generators refuse an unrecorded ROCm
version, so `ROCM_PATH` must be set. Every other `OPS` entry takes no reference dir.

**The pins come from the tree.** A pin is a row of the arch's reference CSV (`tests/data/sdpa_bwd_reference.csv` for gfx950,
`tests/data/sdpa_bwd_gfx11_reference.csv` for gfx11-generic): Spec hash, Spec key, the sha256 of the pinned object and its resource row.
The shipped content holds the object of every pin but two kinds, because a shipped pin row *is* its pin: the main kernel of a combined
object (a dB-storing bias row; the shipped object also carries the dBias reduce) and gfx950's standalone reduce kernels. `tools/sdpa_bwd_s1.py
--from-content "$CONTENT" --arch $ARCH --reference-dir <new dir>` copies each pin's object out of the content by sha256 and verifies it. For
the two exceptions, `tests/data/sdpa_bwd_pin_functions.csv` records the sha256 of the pin's function bytes, which the per-kernel gate and
`tools/sdpa_bwd_compare.py` read in place of the object. The tool fails on any other pin without an object. The same S1 tool also has a
sweep mode (`--sweep-repo`, `--selection`), which the sweep's maintainers use to compose a new catalogue from the sweep (an AMD-internal
repository) and its selection policy; it writes the `_instances.py` tables, the reference CSVs and the digest file this tree holds, and a
regeneration needs none of it. A gfx11 object is not gated like a gfx950 one: it must be its pin byte for byte (a row of the nondeterministic list ships its pin), with no exception rows, and the dQ generator also writes the three standalone reduce objects (`companions` of its manifest).

`tools/sdpa_bwd_tables_from_sweep.py` installs a pruned catalogue: a pin whose object another pin of the same arch and op also pins, with a Spec equal apart from `floor` and a served head-dim range that contains its own, is not installed (equal ranges keep the smaller hash). The tables and the reference CSV hold the pins that remain, and the tool prints the dropped rows. The tool installs no nondeterministic list.

**Nondeterministic builds.** LLVM's IR optimisation is nondeterministic for some kernels, and the 16-bit dQ and dK/dV modules are compiled for a second fast-math target, so the same bug shows as a different object or as `dual_object`. One list, `tests/data/sdpa_bwd_nondeterministic.csv` (both arches; columns in `generators/_sdpa_bwd_nondet.py`), and one policy, the "Nondeterministic builds" section of the `generators/_sdpa_bwd_cdna_build.py` docstring:

- every listed row ships its stored pin; a recorded variant is a build-gate tolerance and is never shipped (no variant needs device validation);
- a listed standalone row ships the sweep's pin; it is built once, for the manifest, never retried, and a `dual_object` build is no error, but a rebuild that exists must be the pin or a recorded variant (whole object, or main function), else the run stops;
- a listed combined row (a dB-storing bias row) ships the combined object its shipped `manifest.json` records (`--stored-dir <content root>`: the content as it was before the run; a missing object, or one whose sha256 disagrees with the manifest, stops the run). Its rebuild must still pass the per-kernel gate on the pin's main function or any of its `variant_function_sha256s` (resources unchanged), and a `dual_object` build is retried up to `_ATTEMPTS`;
- an unlisted row whose build differs from its pin, a listed row whose rebuild is neither its pin nor a recorded variant, or a row that is still `dual_object` after `_ATTEMPTS`, stops the run: the list is incomplete. Sample the row (N = 6 fresh-process builds under the regeneration conditions) and list it; never retry harder.

Merge a sampling run into the list (union of variants, evidence appended): `cd flydsl && $PY -m generators._sdpa_bwd_nondet merge <nondeterministic-sampled.csv>`.

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
  # The RNG dump's keep-mask writer (COVERAGE.md §5.10), one object per target.
  $PY -m generators.gen_sdpa_dropout_mask --arch $ARCH
  # The head-dim padding copy (COVERAGE.md §7.12), one object per target.
  $PY -m generators.gen_sdpa_head_pad --arch $ARCH

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
  $PY -m generators.gen_sdpa_dropout_mask --arch $ARCH
  $PY -m generators.gen_sdpa_head_pad --arch $ARCH
  $PY gen_descriptors.py --arch $ARCH
  $PY gen_descriptors.py --arch $ARCH --check
done

# (4) No transcendental result read by the very next VALU op on gfx942/gfx950: the
#     backend adds the wait state for its own instructions but not for inline asm, so
#     a finding is an inline-asm hazard (CDNA kernel modification 46). Exit 1 on one.
$PY tools/check_trans_hazards.py --content-dir $CONTENT
```

The SDPA backward ops (prep, dK/dV, dQ and, on gfx950, merged) build with ROCm 10.2.0 (`ROCM_PATH` naming it)
and take the arch's pinned-object dir (built from the content before it moves, as §2 does). A generator writes
`<op>/<arch>/` under `--out-dir` and requires that directory to be empty, so move the old content aside to `$STORED` (`dvc checkout`
restores it): every listed combined row ships the combined object that content records (`--stored-dir "$STORED"`), and the run stops if it is
missing or altered. Run dQ with `FLYDSL_GEN_JOBS=8`: at wider settings rows
outside `tests/data/sdpa_bwd_nondeterministic.csv` flip too, and an unlisted flipped row fails the run.

A catalogue change (a new or repinned row) has no shipped object to copy: the maintainers compose it with the S1 tool's sweep mode (§2), whose `--reference-dir` then replaces the `$REFERENCE_DIR` below.

```bash
STORED=$(mktemp -d)   # keep it until the regenerated content is committed
for ARCH in gfx950 gfx11-generic; do
  case $ARCH in
    gfx950)        OPS="sdpa_bwd_prep sdpa_bwd_dkdv sdpa_bwd_dq sdpa_bwd_merged" ;;
    gfx11-generic) OPS="sdpa_bwd_prep sdpa_bwd_dkdv sdpa_bwd_dq" ;;
  esac
  REFERENCE_DIR=$(mktemp -d)/$ARCH   # a fresh path: the S1 tool refuses an existing directory, and §2 already built $REF_ROOT/$ARCH
  $PY tools/sdpa_bwd_s1.py --from-content "$CONTENT" --arch $ARCH --reference-dir "$REFERENCE_DIR"   # before the content moves
  for op in $OPS; do
    mkdir -p "$STORED/$op"; mv "$CONTENT/$op/$ARCH" "$STORED/$op/$ARCH"   # the run needs an empty output directory; the old content is the stored pins' source
    jobs=${FLYDSL_GEN_JOBS:-28}; [ $op = sdpa_bwd_dq ] && jobs=8
    ROCM_PATH=$ROCM_PATH_BWD FLYDSL_GEN_JOBS=$jobs PYTHONDONTWRITEBYTECODE=1 \
      $PY -m generators.gen_$op --arch $ARCH --out-dir "$CONTENT" --reference-dir "$REFERENCE_DIR" --stored-dir "$STORED"
  done
  $PY gen_descriptors.py --arch $ARCH
  $PY gen_descriptors.py --arch $ARCH --check
done
```

**Companions.** A dQ or merged row launches a tail-zero object after its gradient launches when the graph has a
`padding_mask` (`generators/gen_sdpa_bwd_tailzero.py`; the kernels are `fmha_bwd_tailzero_gfx950.py` and
`fmha_bwd_tailzero_gfx11.py`). The dQ generator builds the objects through
`gen_sdpa_bwd_tailzero.companion_entries(arch, folder)` and lists them under `companions` in the dQ manifest; the merged
generator builds nothing and lists the dQ manifest's entries (it stops if they or their objects are missing or altered), so
**generate `sdpa_bwd_dq` before `sdpa_bwd_merged`** into the same content root (the `OPS` order above does); the
`spec` of an entry is `<role>,<dtype>` (`tailzero,f16`, with `;arch=gfx11-generic` on gfx11), and
`gen_descriptors.py` keys companions by `(dtype, role)`. Each tail-zero object is kept once, in the dQ arch folder (gfx11
dQ holds `dbias_reduce` and `tailzero` companions, gfx950 dQ holds `tailzero`); the merged manifest's `tailzero.file`
points to `../../sdpa_bwd_dq/gfx950/<file>` and the packer deduplicates by resolved path. A gfx950 row that names a
`reduce_symbol` has no reduce companion: its `dbias_reduce` entry names the row's own object and that symbol. The
launch-name sets of the rows follow: dQ `{dq, tailzero}`, dB-storing dQ `{dq, dbias_reduce, tailzero}` (the `reduce`
pack), merged `{merged, tailzero}`, dB-storing merged `{merged, dbias_reduce, tailzero}` (its `reduce` pack);
`gen_descriptors.py` refuses a dQ or merged row whose dtype has no tail-zero companion.

A V-head tail-zero object (`tailzero_hv,<dtype>`, built by the same call) launches after a V-head dQ row (`u2=dq`) instead of the standard one.

A tail-zero object has no pin. Its provenance is the generator: each object is built twice in fresh processes and the two
builds must be byte-identical, the kernarg layout must be `PARAMS`, and on gfx950 the R-MIR scan must be clean; the gfx950
resource row is recorded in the companion entry. The pin steps above (reference rows, pin
exceptions) apply to every other object and are skipped for tail-zero rows. To build the tail-zero objects alone, for inspection:

```bash
ROCM_PATH=$ROCM_PATH_BWD PYTHONDONTWRITEBYTECODE=1 \
  $PY -m generators.gen_sdpa_bwd_tailzero --arch gfx950 --out-dir "$SCRATCH"   # also gfx11-generic
```

The gfx950 dQ op builds at `FLYDSL_GEN_JOBS=8` (wider worker counts flip nondeterministic dQ rows); the tail-zero objects
build in the same run, so they inherit the dQ worker count.

**Packed tile.** `gen_descriptors.py` writes the resolved `block_q` of every packed (`gqa_pack` 1) dK/dV and merged row into
its KMD and KDP and refuses a row whose value is missing or above `PACK_MAX_BLOCK_Q`, which it reads from
`FlydslSdpaBwdNativeShared.hpp`. The generators write the resolved tile into the row knobs; a tile above the bound is a
change of the bound and of the engines' packed-span proof, not of a table.

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

`kernels_src/` is a vendored copy of 24 kernel modules from three upstreams.
Fourteen come from FlyDSL; the RDNA attention kernel comes from AITER, which carries
FlyDSL kernels upstream FlyDSL does not; and nine come from AOTriton: the RDNA4 attention package
(`attention_rdna4/`, six modules) and three SDPA backward modules
(`attention_bwd_cdna/fmha_{common,abi}_gfx1201.py` and `attention_bwd_rdna/fmha_common_gfx1201.py`). The other files of
`attention_bwd_cdna/` are written for this provider and say `Not vendored` in their headers. Deliberate modifications are
recorded beside the code: `kernels_src/kernels/norm/rmsnorm_kernel.py` (four),
`kernels_src/kernels/norm/rmsnorm_bwd_kernel.py` (two) and
`kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py` (sixteen — the gfx11
port and the runtime arguments that let a few objects cover many shapes) in their
own headers, the RDNA4 package's in `kernels_src/kernels/attention_rdna4/README.md`, and the
CDNA attention files' in `kernels_src/kernels/attention_cdna/README.md`.
A modified file's header also names the upstream and path it came from -- the renamed
`attention_rdna/` and `attention_cdna/` packages both map back to upstream's
`kernels/attention/`; the modules of `attention_rdna4/` carry none and inherit their
package's, which names AOTriton's flat `modules/flash/flyc/` directory, while the
three `attention_bwd_*` modules carry their own block naming that directory's file -- and the tool
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
=== kernels/attention_bwd_cdna/fmha_abi_gfx1201.py
  …
=== kernels/attention_bwd_cdna/fmha_common_gfx1201.py
  …
=== kernels/attention_bwd_rdna/fmha_common_gfx1201.py
  …
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
8/24 vendored files identical to upstream after normalization; 16 differ, 0 absent upstream, 0 not compared
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
