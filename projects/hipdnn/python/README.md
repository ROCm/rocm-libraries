# hipDNN Python Bindings

> [!CAUTION]
> **This is a POC of python bindings for hipdnn.  It likely has bugs and features missing.  Making this not a POC has been planned for a future date**


This project provides Python bindings for the hipDNN frontend library using the nanobind library. The bindings allow users to access the functionalities of the hipDNN library directly from Python, enabling seamless integration of deep learning operations.

## Project Structure

```
python/
├── README.md
├── download_third_party_deps.py             # Downloads pinned CI third-party source archives
├── frontend_bindings/
│   ├── CMakeLists.txt                     # CMake build; superbuild component or standalone
│   └── src/
│       ├── module.cpp                     # Main nanobind module entry point
│       ├── bindings.hpp                   # Shared binding declarations
│       ├── device_buffer.hpp              # hipdnn.DeviceBuffer
│       ├── dlpack_adapter.cpp             # DLPack (__dlpack__) import for variant packs and tensor_like
│       ├── dlpack_adapter.hpp
│       ├── autotune_bindings.cpp
│       ├── graph_bindings.cpp
│       ├── graph_kwargs_bindings.cpp      # cuDNN-style keyword overloads for Graph ops
│       ├── pointwise_bindings.cpp         # cuDNN-style named pointwise methods
│       ├── pygraph_bindings.cpp           # hipdnn.pygraph (cuDNN lifecycle semantics)
│       ├── cudnn_compat.cpp               # Shared helpers for the cuDNN-style API
│       ├── cudnn_compat.hpp
│       ├── handle_bindings.cpp
│       ├── handle_wrapper.hpp             # hipdnn.Handle implementation
│       ├── memory_bindings.cpp
│       ├── tensor_bindings.cpp
│       ├── attributes_bindings.cpp
│       ├── hip_bindings.cpp
│       └── types_bindings.cpp
└── frontend_wheel_package/
    ├── src/
    │   └── hipdnn_frontend/
    │       └── __init__.py                # Runtime package initializer
    ├── samples/                           # Source-tree sample scripts
    ├── tests/                             # Source-tree tests
    ├── pyproject.toml                     # Wheel metadata and pytest config
    └── pack_frontend_wheel.py             # Stages and packs the wheel package
```

## Prerequisites

- CMake 3.26 or higher
- Ninja or another CMake generator
- A C++ compiler with C++17 support (e.g. clang++)
- Python 3.12 or higher, including development headers
- ROCm/HIP runtime and libraries
- Installed hipDNN development artifacts with `hipdnn_frontendConfig.cmake`,
  `hipdnn_backendConfig.cmake`, headers, and the backend shared library
- `nanobind` and `tsl-robin-map` CMake packages, or network access during
  configure so CMake can fetch the pinned source archives
- The `build` Python package when creating a wheel
- The `numpy` and `pytest` Python packages when running source-tree tests or samples

## Building

The recommended way to build the bindings is as a component of the
rocm-libraries superbuild, alongside hipDNN itself — no separate install step
or `CMAKE_PREFIX_PATH` needed since the extension links the in-tree
`hipdnn_frontend`/`hipdnn_backend` targets directly. From the repository root:

```bash
cmake --preset hipdnn-python -GNinja -DROCM_PATH=/path/to/rocm
cmake --build build --target hipdnn_frontend_bindings
```

See [Superbuild](../docs/Building.md#superbuild) for preset details; e.g.
`hipdnn-dev-all` builds the bindings alongside every provider.

The bindings can also be built standalone against an already-installed hipDNN
(e.g. a ROCm/hipDNN nightly artifact), which doesn't require a full source
checkout:

```bash
cmake -S projects/hipdnn/python/frontend_bindings -B build/hipdnn-python -GNinja \
    -DCMAKE_PREFIX_PATH=/path/to/hipdnn/install
cmake --build build/hipdnn-python
```

`CMAKE_PREFIX_PATH` is required for the standalone build: point it at the
installed hipDNN artifact prefix, or set the `CMAKE_PREFIX_PATH` environment
variable before configuring.

Either way, the build only produces the nanobind extension in the CMake build
tree. CMake does not know about wheel packaging, does not configure
`hipdnn_frontend/__init__.py`, and has no install rules.

Run the wheel packer to create the staged import package under
`<build-dir>/wheel_package/hipdnn_frontend`; downstream environment wiring
should use that staged package tree.

For a source-tree development import after staging the package, put the staged
wheel package root on `PYTHONPATH`:

```bash
PYTHONPATH=build/wheel_package python -c "import hipdnn_frontend"
```

The backend shared library must also be discoverable at runtime: use
`LD_LIBRARY_PATH=/path/to/hipdnn/install/lib` on Linux, or set `ROCM_PATH` to the
artifact prefix on Windows so `hipdnn_frontend/__init__.py` can register `bin/`.

## Creating a Wheel

After building the bindings, run the packer script:

```bash
python projects/hipdnn/python/frontend_wheel_package/pack_frontend_wheel.py \
    --build-dir <build-dir> \
    --wheel-dir <build-dir>/wheel_package
```

`<build-dir>` is `build` for the superbuild or `build/hipdnn-python` for the
standalone build; the packer searches the whole tree for the extension, so
the exact nesting doesn't matter. The wheel is written to
`<build-dir>/wheel_package/`, beside the `hipdnn_frontend/` package directory.
The script packs `wheel_package/hipdnn_frontend` into a temporary setuptools
project. The wheel contains only `hipdnn_frontend/__init__.py` and the native
extension. It does not include samples or tests, and it does not bundle
`libhipdnn_backend`; users still need ROCm and hipDNN runtime libraries
discoverable through ROCm wheels, `ROCM_PATH`, or the platform loader path.

## Testing the Wheel

The `hipDNN Superbuild CI` workflow validates the wheel end-to-end inside the
matching Linux and Windows superbuild jobs after installing the superbuild
outputs into the ROCm SDK path. The workflow calls
`projects/hipdnn/python/download_third_party_deps.py` to download and verify
pinned third-party source archives from `rocm-third-party-deps`, then passes
those source directories to CMake FetchContent as part of the single
superbuild configure. It then builds hipDNN and the bindings together, packs
the wheel, installs that wheel into the same venv, and runs:

```bash
python -m pytest -q projects/hipdnn/python/frontend_wheel_package/tests
```

The wheel package uses a `src/` layout, so running pytest from
`frontend_wheel_package/` does not accidentally import the source package.

## DLPack Interoperability

`Graph.execute()`, `Graph.execute_timed_ext()`,
`Graph.execute_plan_at_index()`, `Graph.autotune()`, and
`Graph.autotune_exhaustive_sweep()` take a `variant_pack` keyed by tensor UID
or by `Tensor`. Each value, and `workspace`, may be one of these kinds, checked
in this order:

1. An `int` pointer.
2. A `hipdnn.DeviceBuffer`.
3. An object with a `data_ptr()` method (for example a PyTorch tensor).
4. An object that implements `__dlpack__` in host (`cpu`), ROCm (`rocm`), or
   pinned host (`rocm_host`) memory. The bindings use the DLPack data pointer
   plus its byte offset.

No data is copied. The caller must keep each object alive until the HIP work
completes. The bindings call `__dlpack__(copy=False)`, so a producer that can
export only a copy raises its own error instead of returning memory that nothing
keeps alive. Errors raised by `__dlpack__` reach the caller unchanged.

`Graph.tensor_like(obj, name="")` also accepts a `__dlpack__` producer. It
copies the dims and the element strides (row-major when the producer reports
none) and maps the data type. A host (`cpu`) producer becomes a
runtime pass-by-value tensor: pass a host tensor for it in the variant pack of
each execute call. Runtime pass-by-value tensors need an engine plugin that
reports plugin API 1.2.0 or later. Sub-byte types follow the DLPack
definition (`lanes=1`); packed exports such as `torch.float4_e2m1fn_x2`
(`lanes=2`) raise `ValueError`.

```python
x = torch.randn(8, 16, device="cuda")
scale = np.full((1, 1), 0.5, np.float32)
x_t = graph.tensor_like(x, "x")
scale_t = graph.tensor_like(scale, "scale")  # runtime pass-by-value
# ... build the graph and plans with x_t and scale_t, producing y_t ...
graph.execute(handle, {x_t: x, scale_t: scale, y_t: y}, workspace)
```

For a compile-time constant (plugin API 1.0.0), call `set_value` on the tensor
instead. It bakes the value into the graph and clears the runtime flag.
`set_value(value, data_type=None)` supports `FLOAT`, `DOUBLE`, `HALF`,
`BFLOAT16`, `UINT8`, `INT32`, `INT64`, and `BOOLEAN`. Without `data_type`, it
keeps the tensor's data type, or infers `BOOLEAN`, `INT64`, or `FLOAT` from a
Python `bool`, `int`, or `float`. It resets dims and strides to `[1]`, so the
tensor must have one element.

```python
scale_t = graph.tensor_like(scale, "scale").set_value(0.5)  # FLOAT constant
eps_t = hipdnn.Tensor().set_value(1e-5, hipdnn.DataType.DOUBLE)
```

## cuDNN-style API

`hipdnn.Graph` follows the hipDNN C++ API: op methods take a `*Attributes`
object, and lifecycle methods (`validate`, `build_operation_graph`, `execute`,
...) return an `Error`. That surface is unchanged. Code written for
`cudnn.pygraph` can use the cuDNN-shaped additions below instead; all of them
are extra overloads, methods, or names, so existing `hipdnn.Graph` code keeps
working as before.

On `hipdnn.Graph`:

- Every op with a cuDNN counterpart (`conv_fprop`, `conv_dgrad`, `conv_wgrad`,
  `matmul`, `batchnorm*`, `layernorm*`, `rmsnorm*`, `block_scale_*`,
  `reduction`, `moe_grouped_matmul`, `sdpa*`) also accepts cuDNN's keyword
  signature, for example `graph.conv_fprop(image=x, weight=w, padding=[1, 1])`.
  Passing an attributes object together with cuDNN keywords raises `TypeError`.
- cuDNN's named pointwise methods (`add`, `bias`, `mul`, `relu`, `gelu`,
  `sigmoid`, ...). `mod`, `pow` and `cos` are not available: hipDNN has no
  pointwise mode for them.
- `Graph.tensor(dim=, stride=, data_type=, is_virtual=, is_pass_by_value=,
  name=, uid=)`. `data_type` here and in `Tensor.set_data_type` also takes a
  torch or numpy dtype.
- cuDNN's keyword names next to hipDNN's: `index` for `plan_index`, and `input`
  for `tensor_like`'s first argument.

`hipdnn.pygraph` is a `Graph` subclass with cuDNN's lifecycle semantics:

- The constructor takes `name`, `io_data_type`, `intermediate_data_type`,
  `compute_data_type` and `handle`. Without a handle, the graph creates one the
  first time a call needs it.
- `build()` runs `validate` and `build_operation_graph`; `build(heur_modes)`
  also creates, checks and builds the execution plans.
- `execute(tensor_to_device_buffer, workspace, handle=None)` and
  `execute_plan_at_index(tensor_to_device_buffer, workspace, index,
  handle=None)` take cuDNN's argument order. The hipDNN forms remain available.
- `serialize()` and `deserialize([handle_,] pyobj)`.
- Failures raise instead of returning an `Error`: `ValueError` for invalid
  attributes, shapes, tensor names or variant packs;
  `hipdnn.GraphNotSupportedError` (a `RuntimeError`) when no engine supports
  the graph; `RuntimeError` otherwise. `execute_timed_ext` still returns
  `(Error, ExecutionTiming)`.

The package also exports cuDNN's module-level names for the hipDNN types:
`tensor`, `data_type`, `heur_mode`, `norm_forward_phase`, `reduction_mode`,
`convolution_mode`, `build_plan_policy`, `behavior_note`, `diagonal_alignment`,
`attention_implementation` and `moe_grouped_matmul_mode`.

```python
import hipdnn_frontend as cudnn

graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.FLOAT,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)
X = graph.tensor(name="X", dim=x.shape, stride=x_strides, data_type=x.dtype)
W = graph.tensor(name="W", dim=w.shape, stride=w_strides, data_type=w.dtype)
Y = graph.relu(input=graph.conv_fprop(image=X, weight=W, padding=[1, 1]))
Y.set_output(True)
graph.build([cudnn.heur_mode.FALLBACK])
graph.execute({X: x_dev, W: w_dev, Y: y_dev}, workspace)
```

Remaining differences from `cudnn.pygraph`:

- cuDNN parameters hipDNN cannot honour keep their position but accept only
  cuDNN's default value; anything else raises `NotImplementedError`. Examples
  are `matmul(padding=)`, `tensor(ragged_offset=, reordering_type=)`, the
  `pygraph` constructor's `sm_count`, `sm_version`, `kernel_cache` and
  `device_property`, the SDPA `score_mod` callbacks, and runtime shape
  overrides outside `execute` (and, without SDPA support, in `execute` too).
- Operations hipDNN does not implement are absent, for example
  `instancenorm`, `adalayernorm`, `rope`, `slice`, `concatenate`, `reshape`,
  `transpose`, `genstats`, the FP8/MXFP8 SDPA variants, and the CUDA-graph
  methods.
- Enum values differ where the libraries differ: `data_type` has no
  `FAST_FLOAT_FOR_FP8`, `INT8x4`, `UINT8x4` or `INT8x32`, `heur_mode` has no
  `OPENSOURCE`, and `behavior_note` values are vendor-specific.
- `create_handle()` returns a `hipdnn.Handle`, not an integer.
- The `cudnn.graph` context manager and the `cudnn.jit` / `cudnn.graph_cache`
  decorators are not provided.

## Running the Samples

Sample scripts are source-tree utilities and are not included in the wheel.

```bash
python projects/hipdnn/python/frontend_wheel_package/samples/conv_fprop.py
python projects/hipdnn/python/frontend_wheel_package/samples/conv_dgrad.py
python projects/hipdnn/python/frontend_wheel_package/samples/conv_wgrad.py
python projects/hipdnn/python/frontend_wheel_package/samples/matmul.py
python projects/hipdnn/python/frontend_wheel_package/samples/autotune.py
python projects/hipdnn/python/frontend_wheel_package/samples/cudnn_style_conv_bias_relu.py
```
