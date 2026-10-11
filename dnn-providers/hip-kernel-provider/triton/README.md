# Triton AOT generator (RMSNorm)

Compiles the RMSNorm `@triton.jit` kernel in `kernels_src/rmsnorm.py` to AMDGPU code
objects for `gfx942` and `gfx950`, then emits the hip-kernel-provider descriptors for them.
No GPU is needed. The output is committed under
`src/engines/kernel_ingestor_engine/descriptors/Triton/rmsnorm/`.

## Environment

The generator needs `triton==3.8.0` (a CPython 3.12 wheel) and `msgpack==1.2.3`, pinned
with wheel hashes in `requirements.txt`:

```
python3.12 -m venv /path/to/triton-aot-gen-venv
/path/to/triton-aot-gen-venv/bin/python -m pip install --no-deps --require-hashes \
    -r dnn-providers/hip-kernel-provider/triton/requirements.txt
```

pip enforces the hashes. At run time the generator checks only that `triton.__version__`
is `3.8.0`, that `triton` is imported from site-packages, and that `requirements.txt` and
`generators/_contract.py` carry the same Triton hash. The `triton_wheel_sha256` in each
manifest is that pinned hash, not a hash of the installed wheel.

To change a pin, update `requirements.txt`, `TRITON_VERSION` and `TRITON_WHEEL_SHA256` in
`_contract.py`, and the copy in `projects/hipdnn/dockerfiles/Dockerfile.ubuntu24`. The
`triton-aot-pin-sync` pre-commit hook checks the Dockerfile copy.

The hipDNN dev image provides this venv at `/opt/triton-aot-gen-venv`. It is not on `PATH`.

`gen_descriptors.py` does not import `triton`; any interpreter with `msgpack` runs it.

### `llvm-objdump`

The generator disassembles every object with `llvm-objdump`, searched for in
`$ROCM_PATH/llvm/bin`, `/opt/rocm/llvm/bin`, then `PATH`.

### Environment variables

Before importing `triton`, the generator:

- sets `TRITON_DISABLE_LINE_INFO=1` and a fresh `TRITON_CACHE_DIR` for each compile;
- fails with `TRITONGEN-ENV` if `TRITON_INTERPRET` or `TRITON_KERNEL_OVERRIDE` is set;
- unsets `TRITON_OVERRIDE_ARCH`, `TRITON_ALWAYS_COMPILE`, `TRITON_HIP_*`, `AMDGCN_*` and
  `*_FP_FUSION`.

Other `TRITON_*` variables, and other variables `libtriton` reads (e.g.
`DISABLE_LLVM_OPT`), are left set and can change the output; the manifest records
`TRITON_F32_DEFAULT`. Regenerate in a clean environment.

## Regenerate

Run from this directory. `<root>` is the folder that contains `rmsnorm/`; it defaults to
the committed `descriptors/Triton/`. One command regenerates the objects and the
descriptors:

```
cd dnn-providers/hip-kernel-provider/triton
VENV=/opt/triton-aot-gen-venv       # or the venv created above
$VENV/bin/python -m generators.gen_rmsnorm --arch gfx942 --arch gfx950 [--out <root>]
```

For each arch in turn, `gen_rmsnorm` compiles and checks every instance, writes the
objects, `manifest.json` and `<arch>.SOURCE.md`, emits the op-level descriptors and the
arch's KDP and UKDs, validates the tree with `hkp_pack`, and swaps the arch into
`<root>/rmsnorm/`. A failure leaves the failing arch and every later one untouched;
earlier arches are complete.

Each arch's swap first validates the whole op tree, including the other arches as they
are in `<root>`. If an arch already in `<root>` holds an invalid descriptor, regenerating
any other arch fails with `TRITONGEN-INPUT` until it is repaired: regenerate the broken
arch first (`--arch <broken> --arch <other>`), or rebuild its descriptors with
`gen_descriptors.py`.

`gen_descriptors.py` re-derives the descriptors from the committed manifests and
objects, without compiling:

```
$VENV/bin/python gen_descriptors.py [--arch ...] [--in <root>] [--out <root>]
```

It builds the selected arches from the manifests, objects and `SOURCE.md` under `--in`,
takes every other arch as it is under `--out`, validates the complete tree with
`hkp_pack`, and swaps it in under `--out`, removing stale descriptors. An input or validation failure leaves `--out` untouched.

### Verify

In the pinned environment, regeneration into an empty folder reproduces the committed
tree byte for byte:

```
$VENV/bin/python -m generators.gen_rmsnorm --arch gfx942 --arch gfx950 --out <scratch>
diff -r <scratch> ../src/engines/kernel_ingestor_engine/descriptors/Triton
```

A difference in a `.hsaco` or `manifest.json` points to a different wheel, LLVM, or
process environment.

## Manifest

`rmsnorm/<arch>/manifest.json` records the toolchain (Triton version, pinned wheel hash,
the LLVM commit from the objects' `.comment`, the environment variables above and the
`HIPOptions`) and, per object, its launch metadata, kernarg layout and SHA256. It holds
no paths or timestamps. `<arch>.SOURCE.md` summarizes it.

## Error codes

Once its arguments parse, every failure of either command prints
`error: TRITONGEN-<CODE>: <detail>` to stderr and exits with status 1:

| Code | Meaning |
| --- | --- |
| `TRITONGEN-ENV` | The environment cannot generate: a refused environment variable; `triton` not importable, the wrong version or shadowed; a `HIPOptions` pin that differs from the wheel default; `requirements.txt` and `_contract.py` pinning different hashes; `llvm-objdump`, `hkp_pack` or `msgpack` missing or failing; an object with no LLVM commit in its `.comment`, or one arch's objects carrying different LLVM commits |
| `TRITONGEN-ARCH` | The object's `amdhsa.target` is unreadable or not the arch it is filed under |
| `TRITONGEN-ABI` | The object is not a readable single-kernel code object, or its kernarg tuple (including both scratch slots), kernel symbol, kernarg segment size or workgroup size differs from the contract |
| `TRITONGEN-SCRATCH` | Non-zero global or profile scratch |
| `TRITONGEN-ERRATUM` | Packed-FP32 instructions (`v_pk_{fma,mul,add}_f32`) in the same object as `v_mfma*`/`v_smfmac*`, a pattern a gfx950 hardware erratum makes unsafe and the pinned wheel does not avoid |
| `TRITONGEN-COMPILE` | `triton.compile` raised |
| `TRITONGEN-INPUT` | `gen_descriptors` input is unusable: a missing or malformed manifest, a missing object or `SOURCE.md`, an object whose SHA256 differs from its manifest, a manifest that disagrees with `_contract.py` or whose block size differs from its object's workgroup size, or an emitted tree that fails `hkp_pack` validation |
| `TRITONGEN-INTERNAL` | Any other exception; the detail names its type |

## Tests

```
$VENV/bin/python -m unittest discover -s tests
```
