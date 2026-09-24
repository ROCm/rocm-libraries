# Run an explicit TensileLite recipe through hipBLASLt

This example supplies one YAML recipe to
`hipblaslt_ext::experimental::jit::tensilelite::getGemmAlgo`, then executes the
returned algorithm through both `hipblasLtMatmul` and `hipblaslt_ext::Gemm`.
The application owns its descriptors, buffers, workspace and output checks;
hipBLASLt owns generation, module loading and retained algorithm state.

The application follows four steps:

1. Create ordinary GEMM descriptors and initialize the input buffers.
2. Set the Python/compiler paths, explicit YAML recipe and fresh output directory
   in `tensilelite::Options`, then call `tensilelite::getGemmAlgo`.
3. Allocate the returned workspace and execute with `selected.algo` through the
   existing C and C++ APIs.
4. Compare the output with the CPU reference and synchronize before cleanup.

The [direct API guide](../../../docs/jit-tensilelite.md) explains the build
requirements, generation options, failure diagnostics and lifetime rules.

## Build and use

Enable `HIPBLASLT_ENABLE_JIT` in the existing build and use a Python environment
with the TensileLite dependencies and built `rocisa` extension. Build the sample
with:

```bash
cmake --build "$project_build" --target _rocisa hipblaslt-jit-gemm --parallel
```

Choose a recipe and architecture that support the sample's GEMM. The output
directory must be new and its parent must exist. Generation diagnostics remain
in `<output>.log`; generated files are retained under `<output>/bundle`.

For the FP16 NN sample (M=256, N=128, K=512), run from the repository root:

```bash
project_build=projects/hipblaslt/build/release
tensile_source="$PWD/projects/hipblaslt/tensilelite"
python_path="$PWD/$project_build/tensilelite/rocisa:$PWD/$project_build/tensilelite:$tensile_source"
# Use single_solution_splitk_gfx1250.yaml for gfx1250.
"$project_build/clients/staging/hipblaslt-jit-gemm" \
  "$VIRTUAL_ENV/bin/python" "$tensile_source" "$python_path" \
  "$tensile_source/Tensile/Tests/unit/test_data/single_solution_splitk.yaml" \
  "$PWD/jit-sample-output" gfx950 /opt/rocm/bin/amdclang++
```

Replace the Python, compiler and architecture arguments with those for the
configured build and current GPU. A passing run reports both
`hipblasLtMatmul PASS` and `hipblaslt_ext::Gemm PASS` with zero maximum error.
The sample requires no prebuilt device library.

Repeated GEMM calls can reuse the returned algorithm in the same process and on
the device that created it. Retaining its bytes or index does not create a
persistent JIT library entry.

This is the direct TensileLite usage example. The separate generic API stack
provides its own usage example while retaining this direct path.
