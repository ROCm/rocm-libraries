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

| Flag | Meaning |
|---|---|
| `--kernel`, `--arch` | Module path (e.g. `kernels.gfx1151.wmma_fmha_fwd`) and target arch |
| `--build`, `--spec` | Only needed if the module exposes more than one builder or spec class |
| `--fixed NAME=V` | Spec fields held constant and baked into the recipe |
| `--axis NAME=V1,V2` | A free axis and its sample values (≥2). Repeatable — that is the multi-axis case |
| `--holdout NAME=V` | Values never used for fitting, verified afterwards |
| `--structural NAME` | The one axis allowed to change the program's *shape* |
| `--domain NAME=V1,..` | Every value the guard should admit (see the pitfalls below) |
| `--probe` | Per-axis triage, then stop |
| `--verify`, `--hsaco` | Replay each point and diff `.ll`, then HSACO bytes |
| `--guard`, `--out` | Derive an admission guard; write the bundle |

It exits non-zero if a requested stage fails, so it drops straight into CI.
`--probe` is the one deliberate exception: a **declining** axis exits 0, because
a refusal costs coverage rather than correctness and some axes will decline
until the roller grows — failing there would leave a job permanently red for a
known gap. A **vacuous** axis exits 1, because that is an authoring mistake
claiming coverage it does not have.

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
