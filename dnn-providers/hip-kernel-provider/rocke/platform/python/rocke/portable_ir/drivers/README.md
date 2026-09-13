# Drivers

Runnable entry points for the portable IR: the gates that defend it, the
surveys that measure it, and the tools you point at your own kernel.

Everything here runs from `platform/` with the engine importable:

```bash
cd .../rocke/platform
export PYTHONPATH="$PWD/python:$PWD/../library${PYTHONPATH:+:$PYTHONPATH}"
```

Anything that replays through the C engine also needs a shared `librocke`:

```bash
cmake -S . -B /tmp/rocke/core -DCMAKE_BUILD_TYPE=Release -DROCKE_BUILD_SHARED_ENGINE=ON
cmake --build /tmp/rocke/core --target rocke_shared -j"$(nproc)"
export ROCKE_ONLINE_LIB=/tmp/rocke/core/librocke.so
```

A `librocke.so` built before the ABI contract landed is **rejected at load**
with a message telling you to rebuild, rather than being used and producing
garbage. If `online.load()` complains about a missing `rocke_abi_version`, the
build is stale — rebuild it, do not hunt for an older binding.

## Which driver

**Gates.** Pinned expectations, run together by
`python3 tools/run_portable_ir_gates.py`. Each defends one claim and fails
loudly when it stops being true.

| Driver | Answers | Key flags |
|---|---|---|
| `record_coverage` | Does the live recorder agree with the post-hoc oracle, for every parity emitter? | `--verbose` |
| `parity_matrix` | Is the `.ll` identical across Python, the C importer and the recipe VM, for every instance × arch? | `--arches`, `--flavor`, `--verbose` |
| `hsaco_parity` | …and are the HSACO bytes identical through comgr? | `--arch`, `--flavor`, `--cap-gb`, `--baseline` / `--no-baseline` / `--update-baseline` |
| `roll_hsaco_parity` | Does a *rolled* recipe reproduce HSACO at sampled **and** held-out axis values? | `--families`, `--no-hsaco`, `--flavor`, `--expect-points` |

**Surveys.** Exploratory, not gating; they report coverage rather than pass/fail.

| Driver | Answers | Key flags |
|---|---|---|
| `roll_nd_coverage` | Can one recipe cover a family's axis cross product? | `--ll`, `--slow` |
| `roll_coverage` | Tiered single-axis roll status | — |
| `roll_gfx950_sweep` | Every family in `kernels/gfx950`, every axis, every flag, plus CBOR bytes | `--family`, `--phase`, `--samples` |
| | *both roll through `roll_kernel`; see [one way in](#one-way-in)* | |
| `verify_recording_production` | Recorder against hand-picked production kernels | — |

**Tools.** Point these at your own work.

| Driver | Answers | Key flags |
|---|---|---|
| `roll_kernel` | **The generic one.** Record, roll, verify and ship *a kernel you name*. | see below |
| `derive_guards` | Guard derivation end to end on real families | `--family`, `--axes`, `--probe`, `--pool-cap`, `--samples`, `--roll`, `--bundle` |
| `gpu_replay` | record → CBOR → C replay → comgr → launch → numerics, on a device | `--device`, `--arch`, `--only`, `--verbose` |
| `launch_from_bundle` | The pure-C launch path: geometry and kernarg offsets from the engine | `--op`, `--dtype`, `--n` |

**Benchmarks.** `bench_jit_validation` (`--n`, `--no-comgr`) times cold JIT from
CBOR with the admission checks in place; `bench_online` (env-configured)
attributes the compile timeline stage by stage.

`roll_recipe` (`--emit`) is the legacy bespoke `head_size` roller for
unified-attention 2D, kept for reference.

> Every roll **gate** and survey carries a hard-coded family list pinned to
> `ARCH = "gfx950"` and `kernels/gfx950`, because a gate has to defend a fixed
> claim. They cannot be aimed at another arch without editing them. The
> concrete-path drivers discover their kernels automatically instead of naming
> them: `parity_matrix` and `hsaco_parity` take an arch (`--arches` / `--arch`),
> while `record_coverage` probes each emitter's arch itself. For a kernel of
> your own on any arch, use `roll_kernel`.

## Recording, rolling, shipping a new kernel

A *recipe* is the recorded log of the ops your builder emitted. Recording is
interception, not instrumentation: `record_kernel` temporarily rebinds
`IRBuilder` and runs your **unmodified** production builder, so a kernel needs
no code changes to be recorded.

*Rolling* turns several recordings into one **parametric** recipe covering a
whole axis, by fitting each varying constant as an expression over the axis and,
where the program's shape itself changes, folding repeated blocks into a loop.
The roller proves its fit by replaying at values it never saw. When it cannot
prove one it **declines and says why** — a refusal costs coverage, never
correctness, and the concrete per-point recipes remain valid.

`roll_kernel` runs that whole pipeline against a kernel you name:

```
record -> roll (N axes) -> verify .ll/HSACO vs the Python oracle
       -> derive guard -> stamp ABI -> write a CBOR bundle
```

<a name="one-way-in"></a>
It is also the one way in. `roll_coverage` and `roll_gfx950_sweep` used to carry
their own copies of this plumbing — make the spec from a base plus a point, ask
the kernel's gate, build, record, roll — and each copy answered slightly
differently. In particular a point the *kernel* refused and an axis the *roller*
declined both surfaced as one undifferentiated "refused", which is exactly the
distinction those surveys exist to report. They now call `roll()` with
`quiet=True` and print their own tables from the `RollResult`.

| Flag | Meaning |
|---|---|
| `--kernel`, `--arch` | Module path (e.g. `kernels.gfx1151.wmma_fmha_fwd`) and target arch |
| `--build`, `--spec` | Only needed if the module exposes more than one builder or spec class |
| `--fixed PATH=V` | Spec fields held constant and baked into the recipe |
| `--axis PATH=V1,V2` | A free integer axis and its sample values (≥2). Repeatable — that is the multi-axis case |
| `--holdout PATH=V` | Values never used for fitting, verified afterwards |
| `--structural NAME` | The one axis allowed to change the program's *shape* |
| `--domain NAME=V1,..` | Every value the guard should admit (see the pitfalls below) |
| `--probe` | Per-axis triage, then stop |
| `--verify`, `--hsaco` | Replay each point and diff `.ll`, then HSACO bytes |
| `--guard`, `--out` | Derive an admission guard; write the bundle |

### From Python, with dictionaries

The command line and `roll()` run the same code. The CLI is quicker for a flat
spec; the function is the sane one for a nested spec, where the equivalent
invocation is a wall of `--fixed` flags:

```python
from rocke.portable_ir.drivers.roll_kernel import roll

r = roll(
    kernel="rocke.instances.common.gemm_universal", arch="gfx950",
    fixed={"name": "gemm", "tile": {"tile_m": 16, "tile_k": 16},
           "data.dtype_a": "bf16", "block_size": 64},
    axes={"tile": {"tile_n": [32, 64]}}, structural="tile_n",
    holdout={"tile_n": [128]}, verify=True, guard=True, out="gemm.cbor",
)
if r:                      # truthy only when everything asked for passed
    publish(r.cbor)
elif r.refused:            # code 3: not this arch, and it names the ones it is
    skip(f"{r.reason} — builds on {', '.join(r.elsewhere)}")
```

`fixed` and `axes` take nested dicts, dotted keys, or both mixed. `holdout` and
`domain` are keyed by axis and take a list (or a lone value). `fixed` also takes
a spec **instance**, which is usually what a kernel author already has — build
it the normal way, then name the fields to vary and they override that base:

```python
base = UniversalGemmSpec(name="g", tile=TileSpec(tile_m=16, tile_n=32, ...), ...)
r = roll(kernel=GEMM, arch="gfx950", fixed=base, axes={"tile.tile_n": [32, 64]},
         structural="tile_n")
```

`RollResult` carries `code` (the exit status), `recipe`, `points`, `cbor`,
`n_recorded`, `trace_bytes`, `parity`, and for a refusal `reason`, `refusals`
and `elsewhere`. A malformed request raises `UsageError` rather than exiting the
interpreter underneath you; a refusal by the kernel comes back as a value, since
it is an answer rather than an accident.

### Kernels that do not follow the conventions

`--kernel some.module` works by looking for a spec dataclass, a `build*`
function and an `is_valid_spec` / `supports_*` gate. Plenty of real kernels do
not fit: `attention_dense` gates through `supports_*` taking a dozen keyword
arguments rather than a spec, `fastkv_regp` builds its spec out of another
kernel's spec, and the examples `qk_block` and `export_mha` have no spec
dataclass at all. Those describe themselves with a `Kernel` instead:

```python
from rocke.portable_ir.drivers.roll_kernel import Kernel, roll

fastkv = Kernel(
    label="fastkv_regp",
    make_spec=lambda **kw: make_fastkv_register_p_spec(Tiled2DSpec(**{**T2D, **kw})),
    build_at=lambda **point: build_fastkv(fastkv_spec(**point), arch="gfx950"),
    gate=lambda spec: supports_tiled_2d(head_size=spec.head_size, ...),
    coherent=lambda point: point["num_query_heads"] % point["num_kv_heads"] == 0,
)
r = roll(kernel=fastkv, arch="gfx950", axes={"num_seqs": [16, 32]}, quiet=True)
```

A bare callable is shorthand for `Kernel(build_at=...)`, so an example kernel or
an ad-hoc closure can be rolled directly: `roll(kernel=lambda D:
build_qk_block(D, "f16"), axes={"D": [64, 128]}, structural="D", ...)`.

`coherent` is for a constraint the kernel's own gate does not enforce but its
emitted code depends on — the tiled kernels accept a `num_kv_heads` that does
not divide `num_query_heads` and then bake in a group size that means nothing.
Points it rejects are reported as refusals like any other.

### Flat and nested specs

`PATH` is a field name for a flat spec and a **dotted path** for a nested one.
`WmmaFmhaFwdSpec` is flat, so `--fixed head_size=64`. `UniversalGemmSpec` nests
`TileSpec`, `TraitSpec` and `DataSpec`, so `--fixed tile.tile_m=16` and
`--axis tile.tile_n=32,64`.

Values are converted using the field's *declared* type rather than guessed, so a
`str` field holding digits stays a string. A nested spec with no required fields
of its own is filled in from its defaults, so reaching `tile` does not oblige
you to spell out `trait` and `data`. An axis is named by its **leaf** — `tile.tile_n`
rolls as `tile_n` — which matches what the gates already declare, and is forced
anyway: an axis name reaches `kernel_name_fmt` as a `{placeholder}`, and a dot
there means attribute access to Python's formatter. Two axes with the same leaf
are rejected rather than silently merged.

```bash
python3 -m rocke.portable_ir.drivers.roll_kernel \
    --kernel rocke.instances.common.gemm_universal --arch gfx950 \
    --spec UniversalGemmSpec --fixed name=gemm_probe \
    --fixed tile.tile_m=16 --fixed tile.tile_k=16 \
    --fixed tile.warp_m=1 --fixed tile.warp_n=1 --fixed block_size=64 \
    --axis tile.tile_n=32,64 --structural tile_n --holdout tile.tile_n=128 \
    --verify --hsaco
```

```
name_fmt : gemm_probe_fp16_t16x{tile_n}x16_w1x1x1_wt16x16x16_compv4_intrawave_cshuffle
CBOR     : 17.5 KiB parametric vs 95.3 KiB for the same points concrete
tile_n=32    EXACT  8667c33e9fd7   715504063bba (6336 B)
tile_n=64    EXACT  cb1c0bc0c89b   aa9a882a650c (6848 B)
tile_n=128   EXACT  0f0ddf07bcb5   463466a9b0c1 (7880 B)
```

### Targeting an architecture

**Choosing the target is the caller's job; deciding whether the kernel serves it
is the kernel's.** Before recording anything, `roll_kernel` puts every point it
is about to touch through the module's own `is_valid_spec` / `supports_*` and
relays the answer verbatim:

```
REFUSED on gfx950: the kernel's own gate rejects every point.

   {'num_query_heads': 8}
      WMMA wmma_f32_16x16x16_f16 atom absent on gfx950 (WMMA is an RDNA
      gfx11/gfx12 instruction; this kernel needs a wave32 RDNA target)
```

A partial refusal is reported the same way (`rejects 1/2`) and names the
offending values, so `head_size=100` is an early "must be a multiple of 16"
rather than a traceback from inside the builder. If the module exposes no gate
at all, the run says so instead of implying the arch was checked.

The arch reaches the gate **by keyword**, because the two conventions disagree
about it: `is_valid_spec(spec, arch)` leaves it positional while
`supports_attention_dense(spec, *, arch=)` makes it keyword-only. Passing it
positionally and retrying on `TypeError` looks like it handles both and does not
— the retry drops the arch, so the gate answers about its *default* target while
the recipe is built for the requested one. That silently admitted a gfx950-only
kernel for gfx942.

A `supports_*` taking only keyword arguments (`head_size=`, `block_size=`, ...)
describes a shape rather than a spec, so it cannot be asked about the one being
built. Those degrade to no gate, with a note saying so and pointing at
`Kernel(gate=...)`, rather than failing the run.

A refusal alone does not say *which* mistake you made, so the driver re-asks the
gate on the other `known_arches()`. A spec accepted somewhere else is a target
you aimed wrong at, and the report names the arches that would take it. A spec
refused everywhere is not about the target at all:

```
   {'tile_n': 32}
      tile_m not divisible by warp_m * warp_tile_m

This is the spec, not the target: the same points are refused on
every other known arch too. Fix the flags rather than the matrix.
```

That distinction is the whole point of the exit codes below — a mistyped tile
config must not quietly "skip" every arch in the matrix.

### Exit status

| | |
|---|---|
| `0` | every requested stage passed |
| `1` | a stage failed: parity mismatch, or `--probe` found a vacuous axis |
| `2` | usage error: unknown arch, missing or misspelled field, or a spec refused on every known arch |
| `3` | refused: the kernel does not serve *this* arch, though it serves others |

`3` is not a failure, which is what makes a per-arch matrix workable: a kernel
that only exists on RDNA should *skip* on gfx950, not go red. `--probe` draws
the same distinction internally — a **declining** axis exits 0, because a
refusal costs coverage rather than correctness and some axes will decline until
the roller grows, while a **vacuous** axis exits 1 because that is an authoring
mistake claiming coverage it does not have.

### 1. Triage the axes first

```bash
python3 -m rocke.portable_ir.drivers.roll_kernel \
    --kernel kernels.gfx1151.wmma_fmha_fwd --arch gfx1151 \
    --fixed head_size=64 --fixed mask_mode=causal \
    --axis num_query_heads=8,16 --axis sliding_window=64,128 --axis head_size=64,128 \
    --probe
```

```
axis                   verdict    detail
num_query_heads        rolls      2 points verified
sliding_window         VACUOUS    identical program at 64 and 128 — rolling it proves nothing
head_size              declines   not constants-only: program: 672 instructions at base vs 1296 at probe 1
```

`--probe` answers two independent questions per axis: *does it roll*, and *does
it matter*. The second is the one people skip. An axis the emitted program does
not depend on rolls trivially and verifies at every point, because nothing
varies — the coverage it appears to buy is not real. Only comparing recorded
programs detects that, which is why triage runs before the roll.

### 2. Roll what survived, and verify it

```bash
python3 -m rocke.portable_ir.drivers.roll_kernel \
    --kernel kernels.gfx1151.wmma_fmha_fwd --arch gfx1151 \
    --fixed head_size=64 --fixed mask_mode=causal \
    --axis num_query_heads=8,16 --holdout num_query_heads=32 \
    --verify --hsaco
```

```
ROLLED in 101 ms
   recorded 2 trace(s), verified 3 point(s)
   CBOR     : 177.8 KiB parametric vs 533.1 KiB for the same points concrete

point                     .ll      ll sha         HSACO
num_query_heads=8         EXACT    af78af6a48fa   fc07e7ed2087 (12760 B)
num_query_heads=16        EXACT    bf1890b486d9   0822d0caa862 (12776 B)
num_query_heads=32        EXACT    015c0762e660   965366d292c4 (12776 B)
```

Two recordings, one recipe, byte-identical output at all three points —
including the held-out 32, which nothing in the fit ever saw.

### 3. Ship it

```bash
python3 -m rocke.portable_ir.drivers.roll_kernel \
    --kernel kernels.gfx1151.wmma_fmha_fwd --arch gfx1151 \
    --fixed head_size=64 --fixed mask_mode=causal \
    --axis num_query_heads=8,16 --holdout num_query_heads=32 \
    --domain num_query_heads=1,2,4,8,16,32,64,128 \
    --verify --guard --out /tmp/wmma_fmha_gfx1151.cbor
```

The guard is derived from your kernel's own `is_valid_spec` / `supports_*`, so
the bundle refuses illegal shapes before anything is compiled. The ABI stamp
records which reader level the artifact needs.

## Worked example, in full: `gfx1151 wmma_fmha_fwd`

Useful because most of its axes *don't* roll, and each declines for a different
and instructive reason.

| Axis | Verdict | Why |
|---|---|---|
| `num_query_heads` | **rolls** | Enters only as constants and in the kernel name |
| `sliding_window` | **vacuous** | Identical recorded program at 0, 64 and 128 — the WMMA path never reads it |
| `head_size` | declines | 672 → 1296 ops; structurally, the KV loop's `scf.for` iter-arg arity goes 20 → 24 as the softmax accumulators scale with `head_size / 16` |
| `num_kv_heads` | declines | The GQA divisor is `num_query_heads // num_kv_heads`, a ratio the affine fitter cannot express |

`head_size` is the known limitation tracked as gap 12 (parametric `scf.for`
iter-args). The workaround is one recipe per head size, with the other axes
rolled inside each — the concrete path is unaffected.

Its concrete path is already gated on this arch and passes today:

```bash
python3 -m rocke.portable_ir.drivers.parity_matrix --arches gfx1151 --verbose
python3 -m rocke.portable_ir.drivers.hsaco_parity  --arch gfx1151 --no-baseline
```

Use `--no-baseline` for gfx1151: `hsaco_baseline.json` only carries gfx942 and
gfx950. The `gfx1151_wmma_fmha_fwd` entries listed there under `refused` are
`is_valid_spec` correctly rejecting WMMA on CDNA, not a compile failure.

## Three ways this goes wrong

**A vacuous axis looks like success.** Covered above; `--probe` is the answer.
An axis that reports `rolls` with an identical program at every value has taught
the recipe nothing.

**A guard derived from the samples is over-strict.** The guard's candidate set
is the domain you intend to *serve*, not the points you fitted from. Derive it
from `--axis num_query_heads=8,16` and you get the rule
`num_query_heads must be one of {8, 16}` — so the shipped bundle refuses
`num_query_heads=32`, a shape the same run just verified byte-identical. Always
pass `--domain`; without it the driver warns, and the default of samples plus
holdouts is almost certainly too narrow.

**A stale engine is refused, not tolerated.** See the setup note above.

Note also that `check_recipe_guard` / `check_bundle_guard` return a
`(verdict, reason)` **tuple**, not a bool. Truthiness-testing the return value
makes every shape look admitted, including refused ones.

## What gates a pull request

`python3 tools/run_portable_ir_gates.py` builds the shared engine and runs the
unit tests plus `parity_matrix`, `hsaco_parity` and
`roll_hsaco_parity --expect-points 22`, owning the pinned expectations so a
local run and CI cannot disagree. It takes about 90 seconds and needs no GPU.
Useful flags: `--lib` to reuse an existing `librocke.so`, `--arch` for the
HSACO gate (default `gfx950,gfx942`), `--skip-tests`, `--no-hsaco` for a
faster and deliberately weaker pass, and `--log-dir` for a log per gate.

Two gates carry a pinned expectation because both can pass while measuring
nothing: `hsaco_parity` reads `hsaco_baseline.json` (a **new** non-compiling
kernel fails; known ones do not), and `roll_hsaco_parity --expect-points N`
fails when fewer than N points were verified, so an axis that quietly stops
rolling is a failure rather than a shorter table.

The surveys and `gpu_replay` are not wired in — the first are exploratory, the
last needs a GPU runner. There is currently **no GitHub Actions workflow**
invoking any of this; the gate script has to be run deliberately.

`roll_kernel --probe` is worth adding to a per-kernel job even while its axes
decline: it pins the current verdict, so the day someone makes `sliding_window`
real, or the roller learns parametric `scf.for` iter-args and `head_size`
starts rolling, the table changes and you find out.

## Building artifacts per architecture

Recording rebinds `IRBuilder` process-wide, so **fan out by process, not by
thread** — eight concurrent recordings in a thread pool lose six to that race,
while the same eight in a process pool all pass. A per-arch matrix is the
natural unit, and one job per arch is what the artifact layout wants anyway:

```yaml
strategy:
  matrix: { arch: [gfx942, gfx950, gfx1151, gfx1201] }
steps:
  - run: |
      python3 -m rocke.portable_ir.drivers.roll_kernel \
        --kernel $KERNEL --arch ${{ matrix.arch }} $AXES \
        --verify --hsaco --guard --out out/${{ matrix.arch }}.cbor
      status=$?
      [ $status -eq 3 ] && echo "not applicable on ${{ matrix.arch }}" && exit 0
      exit $status
```

Artifacts are byte-reproducible for identical inputs, so they cache and diff
cleanly. No target hardware is needed: comgr cross-compiles, and a gfx950 host
produces verified HSACO for gfx1151 and gfx1201. Only `gpu_replay`, which checks
numerics, needs a real device.

Per-arch bundles merge into one multi-arch artifact by concatenating entries and
rebuilding, which re-derives the ABI `min_reader` over the merged set:

```python
entries = []
for arch in arches:
    entries += recipe_bundle.cbor_decode(open(f"out/{arch}.cbor","rb").read())["entries"]
merged = recipe_bundle.build_bundle(entries)
```

Merging is also the *safe* way to combine them. A recipe carries no arch of its
own — the bundle entry is what binds one to an architecture, and
`bundle_contains(key, arch)` is what checks it. Replaying a recipe under the
wrong arch does not raise; it silently returns `.ll` that is neither build. So
keep recipes inside bundles rather than pairing loose CBORs by filename.
