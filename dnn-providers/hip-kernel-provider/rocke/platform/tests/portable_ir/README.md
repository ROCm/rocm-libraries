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
end to author artifacts. Runs the in-package unit tests, the recorder-coverage
sweep, the byte-identity parity matrix, and a standalone-binary byte-identity
check. The last two skip with an actionable reason when `librocke.so` or the
CLI has not been built.

## The gate that matters

Byte-identity, not equivalence: for every production kernel and every target
arch, the `.ll` from replaying the artifact must equal the Python lowerer's
`.ll` byte for byte. Concrete recipes carry `exact_names`, so the VM reproduces
Python's SSA naming rather than merely an isomorphic graph. That is strong
enough to catch reordering and naming drift that an HSACO comparison would
hide.

```
export ROCKE_ONLINE_LIB=<path>/librocke.so
python -m rocke.portable_ir.drivers.parity_matrix [--verbose]
```

Device coverage is separate, since it needs a GPU:

```
python -m rocke.portable_ir.drivers.gpu_replay --device 2 --verbose
```

That one splits author time from run time across a real file on disk — record
to CBOR, then load the CBOR, replay it in C, compile with comgr, and launch —
and checks the device output against a numpy reference. The elementwise linear
and non-transcendental unary kernels are gated bit-exact.
