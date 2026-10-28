# `builders/hiptensor`: the Python reference of hipTensor's rocKE pack builder

hipTensor's rocKE backend (`projects/hiptensor`, `HIPTENSOR_ENABLE_ROCKE_BACKEND=ON`) ships a
kernel pack per GPU target (`rocke.htpk`, with `rocke.index.json`). hipTensor's build writes it
with a C++ host tool, `library/src/backends/rocke/pack/rocke_pack_builder.cpp`, which builds the
kernels with rocKE's C++ engine (`platform/cpp`, the builders of
`rocke/instance_hiptensor_aot.h`, `rocke/instance_tensor_gett.h`, the GEMM and gett_f64
instances) and compiles them with the COMGR library. hipTensor's build runs no Python.

`make_rocke_pack.py` is the same pack writer with rocKE's Python engine (the builders of
`kernels/common/hiptensor_aot` and `rocke.instances`). It is rocKE's cross-check of the C++
builder, never part of hipTensor's build:

```bash
python library/builders/hiptensor/make_rocke_pack.py --arch gfx950 --out /path/to/out \
    [--select-models <hipTensor>/library/src/backends/rocke/select] [--jobs N] [--only REGEX]
```

For the same rocKE sources and COMGR library both write the same packs: the same kernels, code
objects, kernel records, routes and selection models. Only the two header strings that name the
toolchain differ: the C++ builder writes the JIT compiler id and `rocke <version>; build <id>`,
this script `llvm-<version>;comgr=<library>` and `rocke <version>; hiptensor_aot <hash>`. Given
this script's strings (`--compiler-id`, `--rocke-version`), the C++ builder writes byte-identical
`rocke.htpk` and `rocke.index.json` files.

`library/tests/test_hiptensor_pack_cross_check.py` asserts that equality. It needs a built C++
builder (hipTensor's build target `hiptensor_rocke_pack_builder`) and the COMGR library it links:

| Variable | Meaning |
|---|---|
| `ROCKE_HIPTENSOR_PACK_BUILDER` | path of the built `hiptensor_rocke_pack_builder` (the test skips without it) |
| `ROCKE_HIPTENSOR_PACK_COMGR` | COMGR library this script compiles with (default: the one the builder links, from `ldd`) |
| `ROCKE_HIPTENSOR_PACK_ARCHS` | `:`-separated targets (default: hipTensor's pack targets, `gfx90a:gfx942:gfx950`) |
| `ROCKE_HIPTENSOR_PACK_ONLY` | family filter, e.g. `ewt\|dgett` (default: every family) |
| `ROCKE_HIPTENSOR_PACK_SELECT_MODELS` | selection model directory written into both packs (default: none) |
| `ROCKE_HIPTENSOR_PACK_JOBS` | compile jobs of both writers (default: the CPUs of the process) |

With the builder, the test also runs its `--list --check-manifests manifests` (below).

## Manifests

`manifests/<arch>.manifest` holds, for each of hipTensor's pack targets, what the pack contains
apart from its code objects: every kernel with its attributes and argument names, the routes
(without selection models) and the kernel classes of hipTensor's JIT. Both writers list it
without lowering or compiling anything (`make_rocke_pack.py --list`, `hiptensor_rocke_pack_builder
--list`). Two default tests keep the two sides together:

- rocKE: `library/tests/test_hiptensor_pack_cross_check.py` (`test_manifest_is_current`) fails
  when this tree's Python writer lists something else than the committed manifests;
- hipTensor: `rocke_pack_manifest_test` (ctest, no GPU) fails when hipTensor's C++ builder lists
  something else than this tree's committed manifests.

After changing a kernel row, attribute, route or JIT class, regenerate them

```bash
python library/builders/hiptensor/make_rocke_pack.py --list \
    --arch gfx90a --arch gfx942 --arch gfx950 --out library/builders/hiptensor/manifests
```

and make hipTensor's builder (`library/src/backends/rocke/pack/rocke_pack_builder.cpp`,
`rocke_ew_rows.hpp`) agree.

Neither writer produces anything for hipTensor's JIT (`HIPTENSOR_ENABLE_ROCKE_JIT`): hipTensor
builds its problem-specialized kernels at run time with the C++ twin of
`kernels/common/hiptensor_aot/jit_ew.py` (`rocke_build_htaot_ew_jit`), which it links.
