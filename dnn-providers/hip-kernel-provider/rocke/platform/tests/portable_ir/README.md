# portable-IR replay tests

Tests for the replay path in `cpp/portable_ir/` — the code that turns a
serialized artifact back into a `rocke_kernel_def_t` and lowers it, with no
CPython in the process.

Two artifact forms, two entry points:

| artifact | schema | entry point |
| --- | --- | --- |
| concrete IR graph | `rocke.ir/v1` (JSON) | `rocke_import_kernel_from_json` |
| builder recipe | `rocke.recipe/v1` (JSON or CBOR, bare or bundled) | `rocke_recipe_run_from_*` |

A recipe is the interesting one: it encodes the *builder algorithm* including
its compile-time control flow, so one small artifact covers a whole kernel
family and the shape specialization happens in C at JIT time.

## What is here

`dom_decoders.cpp` (ctest) — unit tests for the JSON and CBOR decoders. The
load-bearing case is that CBOR decodes to the same `jd_val_t` DOM as the
equivalent JSON, which is what lets the compact shipping format reuse the JSON
consumers unchanged. Also pins clean failure on truncated input.

`recipe_vm_replay.cpp` (ctest) — runs an embedded toy recipe through the VM at
two different spec values and checks that the spec drives *structure* (the
`static_for` expands to N multiply-accumulates for spec `D=N`), that spec
strings reach the kernel name, and that replay is deterministic. Hermetic: no
artifact files, no Python.

`replay_cli.cpp` (build-only) — a standalone `artifact -> .ll` binary. This is
the deployment shape made concrete, and the thing to reach for when bisecting a
parity failure without a Python stack in the way. Sibling of
`tests/core/ir_lower_cli.cpp`, which reads rocke's own `ckdsl.ir/v1` text
format rather than the front end's JSON.

```
cmake --build <build> --target rocke_portable_ir_replay_cli

rocke_portable_ir_replay_cli --ir kernel.ir.json --arch gfx950
rocke_portable_ir_replay_cli --recipe r.cbor --cbor --int D=128 --str dtype=f32
rocke_portable_ir_replay_cli --bundle b.cbor --key gemm_f16 --arch gfx950
```

`test_portable_ir.py` (pytest) — CI wiring for the Python-side drivers, which
live in `python/rocke/portable_ir/drivers/` because they need the Python front
end to author artifacts. The lanes are deliberately separated:

| lane | marker/artifact policy | claim |
| --- | --- | --- |
| in-package unit modules | always; never builds native artifacts | codecs, ABI, guards, launch plans, freshness policy |
| recorder coverage | always; pure Python | every buildable family records without gaps |
| shared-engine parity | `integration`; source-matched library | C++ replay LLVM is byte-identical for each supported, buildable family/arch cell |
| standalone replay | explicit source-matched or installed-manifest-validated CLI | deployment-shaped replay is byte-identical |
| recipe rolling | `integration` for shared-engine replay; explicit CLI for standalone replay | sampled and held-out recipes replay exactly |
| CTest decoder/VM units | configured CMake build | hermetic native decoder and VM behavior |

Direct pytest builds source/build-recipe-keyed artifacts on demand: CLI-only
lanes resolve only the CLI, while integration lanes validate the complete
shared-engine/CLI bundle. The normal `tests/run_all.py` path configures,
preflights, and supplies both binaries from its build root, so an unrelated
local cache cannot affect that gate. A deliberately supplied foreign CMake build
root cannot own rocKE targets; in that exceptional case pytest resolves the same
source-keyed artifacts used by direct invocation.

## The gate that matters

Byte-identity, not equivalence: for every supported, buildable production
kernel/target-arch cell, the `.ll` from replaying the artifact must equal the
Python lowerer's `.ll` byte for byte. Concrete recipes carry `exact_names`, so the VM reproduces
Python's SSA naming rather than merely an isomorphic graph. That is strong
enough to catch reordering and naming drift that an HSACO comparison would
hide.

```
export ROCKE_ONLINE_LIB=<path>/librocke.so
python -m rocke.portable_ir.drivers.parity_matrix [--verbose]
```

An explicitly selected library or replay CLI must have the content build ID of
the current portable-IR native bundle: the engine/CLI sources and their CMake
build recipes. A stale or mixed build fails once with both IDs and a rebuild
command instead of producing one mismatch per kernel. Without environment overrides,
local discovery uses a content-addressed temporary cache and builds the current
bundle ID when it is absent.

The full local path owns all of its artifacts and is the closest CI equivalent:

```
python tests/run_all.py
```

Cases that require a shared engine or compile replayed artifacts carry the
`integration` marker. Standalone CLI consumers remain in the installed quick
lane: CTest supplies the prebuilt CLI and its build-generated provenance
manifest, so those tests neither discover `librocke` nor build native code. The
remaining unit, recorder, roller, codec, and coverage tier never constructs
native artifacts; a few unit modules opportunistically exercise an
already-supplied source-matched engine:

```
python -m pytest tests/portable_ir -m "not integration"
```

Device coverage is separate, since it needs a GPU:

```
python -m rocke.portable_ir.drivers.gpu_replay --device 2 --verbose
```

That one splits author time from run time across a real file on disk — record
to CBOR, then load the CBOR, replay it in C, compile with comgr, and launch —
and checks the device output against a numpy reference. The elementwise linear
and non-transcendental unary kernels are gated bit-exact.
