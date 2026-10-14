# Triton AOT generator

Compiles `@triton.jit` kernels to AMDGPU code objects (`.hsaco`) and emits the
hip-kernel-provider descriptors for them. No GPU is needed. The output is committed under
`src/engines/kernel_ingestor_engine/descriptors/Triton/<op>/`.

**hipDNN's build, image and CI never install or import Triton.** Triton is needed only to
regenerate the committed objects. Everything else (`--check`, the build gates, the unit
tests) runs with Python and `msgpack` alone.

| Path | Role |
| --- | --- |
| `kernels_src/<op>.py` | Kernel source |
| `tritongen/ops/<op>.py` | Op contract: instances, ABI, descriptor fields |
| `tritongen/gen.py` | Compiles, verifies and emits (`python -m tritongen.gen`) |
| `gen_descriptors.py` | Re-derives or checks descriptors from the committed objects |
| `requirements.txt` | Hash-pinned `triton` and `msgpack` |
| `tools/check_shards.py` | Build-time check of the packed shards |

Ops: `rmsnorm` (`gfx942`, `gfx950`).

## Regenerate

Create the venv once from the hash-pinned requirements (CPython 3.12):

```
python3.12 -m venv <dir>
<dir>/bin/pip install --no-deps --require-hashes -r requirements.txt
```

Regenerate from `dnn-providers/hip-kernel-provider/triton`:

```
<dir>/bin/python -m tritongen.gen --op rmsnorm --arch gfx942 --arch gfx950
```

This rewrites the objects, `manifest.json`, `<arch>.SOURCE.md` and descriptors under
`descriptors/Triton/rmsnorm/`. `--out <root>` writes to another content root instead. Each
arch is built into a staging folder, validated with `hkp_pack` and swapped in; a failure
leaves that arch and every later one untouched.

### Reproducibility contract

Regeneration is byte-identical across hosts when all of these hold:

- the wheel is the one pinned by hash in `requirements.txt`;
- the interpreter is CPython 3.12 on x86_64 Linux;
- the process environment is clean: the generator deletes every variable except
  `HOME`, `PATH` and `ROCM_PATH`, sets `TRITON_DISABLE_LINE_INFO=1`, and uses a fresh
  `TRITON_CACHE_DIR` per compile; it refuses to run with `TRITON_INTERPRET` or
  `TRITON_KERNEL_OVERRIDE` set;
- `llvm-objdump` is on `$ROCM_PATH/llvm/bin`, `/opt/rocm/llvm/bin` or `PATH`.

Each `<arch>/manifest.json` records the evidence:

| Field | Meaning |
| --- | --- |
| `toolchain.triton_version`, `triton_wheel_sha256` | The pinned wheel |
| `toolchain.libtriton_sha256` | Digest of the installed `libtriton.so` |
| `toolchain.llvm_commit`, `hip_options` | LLVM commit read from the objects; fixed `HIPOptions` |
| `toolchain.env` | The environment allowlist and fixed values |
| `toolchain.inputs_sha256` | Digest of the generator inputs: `tritongen/*.py`, the op contract, the kernel source, `requirements.txt` and the instance table |
| `instances[]` | Per object: `knobs`, launch metadata, kernarg layout, SHA256 |

The manifest holds no paths or timestamps. To verify, regenerate into an empty folder and
compare:

```
<dir>/bin/python -m tritongen.gen --op rmsnorm --arch gfx942 --arch gfx950 --out <scratch>
diff -r <scratch> ../src/engines/kernel_ingestor_engine/descriptors/Triton
```

A difference in a `.hsaco` or `manifest.json` points to a different wheel, LLVM or
environment.

## Check

```
python gen_descriptors.py --check
```

`--check` writes nothing and needs no Triton. It recomputes every committed descriptor and
manifest from the kernel source, op contract, generator code and `requirements.txt`, and
verifies each object's SHA256, ABI and arch against its manifest. It prints one
`TRITONGEN-CHECK: <path>: <reason>` line per fault and exits 1, or one summary line and
exits 0. Edit a generator input without regenerating, and `--check` fails.

The build runs it as the `triton_descriptor_check` stamp before the descriptors are
packed, so a stale committed tree fails the build. `gen_descriptors.py` without `--check`
re-emits descriptors from the committed manifests and objects, without compiling.

## Build gates

| Gate | Runs | Verifies |
| --- | --- | --- |
| `triton_descriptor_check` | Before the product pack | The committed tree matches the generator (`--check`) |
| `triton_shard_check` | After the product pack | Each shipped shard holds one packed object per committed `.hsaco`, with its SHA256 |
| `hip-kernel-provider-triton-generator` (CTest) | With the unit tests | The generator's unit tests |

Run the unit tests directly from this directory:

```
python -m unittest discover -s tests -t .
```

## Add an op

1. Add the kernel source `kernels_src/<op>.py`.
2. Add the op contract `tritongen/ops/<op>.py` and register it in `tritongen/ops/__init__.py`.
3. Add the native pack that registers the op's kernel symbols (see
   `src/engines/kernel_ingestor_engine/packs/TritonRmsNormNative.cpp`).
4. Regenerate, commit the output, and confirm `--check` passes.

## Error codes

Every failure of either command prints `error: TRITONGEN-<CODE>: <detail>` to stderr and
exits 1. The codes (`ENV`, `ARCH`, `ABI`, `SCRATCH`, `ERRATUM`, `COMPILE`, `INPUT`,
`CHECK`, `INTERNAL`) are defined in `tritongen/_family.py`; the detail names the cause.
