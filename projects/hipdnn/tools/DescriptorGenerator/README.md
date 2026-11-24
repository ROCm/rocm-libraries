# hipDNN Descriptor Code Generator

Generates hipDNN operation descriptor boilerplate (C++ source, tests, and integration fragments) from YAML configuration files.

## Prerequisites

- Python 3.10+
- PyYAML >= 6.0
- Jinja2 >= 3.1

## Setup

```bash
cd projects/hipdnn/tools/DescriptorGenerator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# Generate files for a single operation
.venv/bin/python generate.py \
    --config configs/convolution_fwd.yaml \
    --output-dir /tmp/codegen-output

# Generate directly into the hipdnn project tree
.venv/bin/python generate.py \
    --config configs/matmul.yaml \
    --output-dir ../../

# Preview the files a mode would write, without writing them
.venv/bin/python generate.py \
    --config configs/matmul.yaml \
    --output-dir /tmp/codegen-output \
    --mode full --dry-run
```

Options:

| Option | Description |
|--------|-------------|
| `--config PATH` | YAML config file (required) |
| `--output-dir PATH` | Output root directory (required) |
| `--mode {backend,frontend,full}` | `backend` (default): descriptor, packer, unpacker, backend and integration tests, backend fragments. `frontend`: Attributes header, Node header, frontend tests, frontend fragments. `full`: both. |
| `--dry-run` | List the files that the mode would generate. Do not write files. |

## Output

Each `--mode backend` run generates:

| File | Purpose |
|------|---------|
| `backend/src/descriptors/<Op>OperationDescriptor.hpp` | Class declaration |
| `backend/src/descriptors/<Op>OperationDescriptor.cpp` | setAttribute/getAttribute, finalize, buildNode, toString |
| `frontend/include/hipdnn_frontend/detail/<Op>Packer.hpp` | Frontend packer function |
| `backend/tests/descriptors/Test<Op>OperationDescriptor.cpp` | Descriptor unit tests |
| `backend/tests/descriptors/TestGraphDescriptor<Op>.cpp` | Graph building tests |
| `tests/frontend/Integration<Op>DescriptorLowering.cpp` | Lowering round-trip + per-scalar preservation tests (operation-specific tests can be added on top) |
| `tests/frontend/Integration<Op>DescriptorLifting.cpp` | Lifting round-trip, tensor sharing, no-finalization, and per-scalar tests |
| `frontend/include/hipdnn_frontend/detail/<Op>Unpacker.hpp` | Frontend unpacker (inverse of packer) |
| `backend/tests/descriptors/Test<Op>OperationFromNode.cpp` | fromNode() round-trip tests |
| `fragments/*.txt` | Code snippets for manual insertion into existing files, including the work-model fragments: `fbs_tensor_fields.txt` (schema annotations), `node_flops_overload.txt`, `logical_flops_case.txt` and `work_model_test_case.txt` |
| `test_sdk/include/hipdnn_test_sdk/constants/<Op>Constants.hpp` | Shared test constants (only when `test_data.constants_include` is not set) |
| `backend/include/<header>.h`, `fragments/mode_*_<field>.txt` | Mode enum header and plumbing fragments (only for `mode` fields with an `enum_def`) |

A `--mode frontend` run generates the Attributes header (`frontend/include/hipdnn_frontend/attributes/`), the Node header (`frontend/include/hipdnn_frontend/node/`, unless `frontend.generate_node: false`), frontend tests (`frontend/tests/`), and the frontend fragments.

See `CLAUDE.md` for post-generation integration steps.

## Adding a New Operation

The end-to-end workflow for adding a new operation type:

1. **Create the FBS schema** in `flatbuffers_sdk/schemas/` and generate the C++ header with `flatc`
2. **Create a YAML config** in `configs/` (use `convolution_fwd.yaml` as a reference)
3. **Run the generator** to produce source files, tests, and fragments
4. **Add enums** to backend headers and string utilities (using the generated fragments)
5. **Add enum test coverage** to `TestBackendEnumStringUtils.cpp`
6. **Place generated files** into the project tree
7. **Update CMake** build files
8. **Build and test** to verify everything compiles and passes
9. **Review the generated integration tests** — both `Integration<Op>DescriptorLowering.cpp` and `Integration<Op>DescriptorLifting.cpp` now ship with full round-trip + per-scalar coverage; add operation-specific tests (multi-input variants, multi-op graphs) on top following the ConvFprop reference at `tests/frontend/IntegrationConvFpropDescriptorLowering.cpp`
10. **Add the work model** — annotate the schema (`fbs_tensor_fields.txt`), regenerate `node_operands_generated.h`, and give the op a FLOP formula or a `NO_COUNTING_CONVENTION` entry (`node_flops_overload.txt`, `logical_flops_case.txt`, `work_model_test_case.txt`)

See `CLAUDE.md` for detailed integration steps (especially steps 4-7 and 12).

### Creating a YAML Config from an FBS Schema

The YAML config maps FBS schema fields to hipDNN backend API concepts. To create a new config:

1. Read the FBS schema in `flatbuffers_sdk/schemas/`
2. Identify tensor UID fields (`*_tensor_uid: long`) → `tensor_fields`; a field annotated `work_data_dependent` sets `work_data_dependent: true`
3. Identify data fields (vectors, enums, scalars) → `data_fields`
4. Look at existing frontend node/attributes classes for naming conventions
5. Use `convolution_fwd.yaml` as the reference config

See `CLAUDE.md` for a detailed mapping guide from FBS field types to YAML `type` values.

### YAML Config Format

```yaml
operation:
  name: "MyOp"                              # PascalCase operation name
  class_name: "MyOpOperationDescriptor"      # Full class name
  fbs_table: "MyOpAttributes"               # FlatBuffer table name
  fbs_generated_header: "my_op_attributes_generated.h"

  descriptor_type:
    enum_name: "HIPDNN_BACKEND_OPERATION_MY_OP_DESCRIPTOR"

  operation_attr_prefix: "HIPDNN_ATTR_OPERATION_MY_OP"

  frontend:
    packer_function: "createMyOpOperation"   # Generated packer function name
    node_class: "MyOpNode"                    # Frontend node class (if it exists)
    attributes_class: "MyOpAttributes"        # Frontend attributes class (if it exists)
    # attributes_filename: "MyOp"             # Optional: override basename of generated
                                              # Attributes header / test files when the
                                              # in-tree filename differs from the class
                                              # name (e.g., ConvFpropAttributes lives in
                                              # ConvolutionFpropAttributes.hpp).

  tensor_fields:
    - name: "input"
      fbs_field: "input_tensor_uid"
      attr_suffix: "INPUT"
      required: true
      # frontend_getter (optional): override the default getter resolution.
      # When unset, the loader matches by frontend.inputs[].name / outputs[].name,
      # then falls back to abbreviation-aware matching. Set it to a bare
      # accessor name (a trailing "()" is stripped) when the backend tensor
      # name does not map to any frontend tensor — e.g., SDPA's `attn_mask`
      # uses `frontend_getter: "get_bias"`. See "Recent Changes" for details.
      work_data_dependent: false      # true when the tensor's contents, not its shape,
                                      # decide the node's work (valid lengths, page tables,
                                      # routing offsets); emits the schema annotation

  data_fields:
    - name: "alpha"
      fbs_field: "alpha"
      attr_name: "HIPDNN_ATTR_MY_OP_ALPHA"
      type: "scalar_float"            # vector_int64 | mode | enum (legacy) | scalar_float | scalar_int32 | scalar_int64 | bool
      required: true
      frontend_getter: "get_alpha()"  # data_fields[].frontend_getter is still honored
      shared: false                   # true if attribute enum already exists from another op
      fbs_optional: false             # true for FBS optional scalars (frontend getter returns std::optional<T>)
      # For enum fields only:
      # test_enum_value: "ADD"        # Required for enum type - value used in generated tests
      # For mode fields with new enums, see CLAUDE.md "Mode Field Properties"
      # and "enum_def" sections; mode_sentinel controls finalize sentinel checks.

  tensor_array_fields:                # For tensor arrays (e.g., peer_stats)
    - name: "peer_stats"
      fbs_field: "peer_stats_tensor_uid"
      attr_name: "HIPDNN_ATTR_OPERATION_MY_OP_PEER_STATS"
      frontend_getter: "get_peer_stats()"
      required: false
      test_uids: [100, 101]           # UIDs for test tensor descriptors
      test_label: "PeerStats"          # Label used in generated test case names
      work_data_dependent: false      # as for tensor_fields

  has_compute_data_type: true
  compute_data_type_attr: "HIPDNN_ATTR_MY_OP_COMP_TYPE"
  compute_data_type_shared: false     # true if compute type attr enum already exists

  test_data:
    tensor_uids: { input: 1 }
    tensor_configs:
      input: { dims: [1, 3, 32, 32], strides: [3072, 1024, 32, 1] }
    field_values:
      alpha: [1.0]
```

## Existing Configs

| Config | Operation |
|--------|-----------|
| `convolution_fwd.yaml` | Convolution forward (reference, validated against POC) |
| `convolution_bwd.yaml` | Convolution backward data |
| `convolution_wrw.yaml` | Convolution backward weights |
| `matmul.yaml` | Matrix multiplication |
| `pointwise.yaml` | Pointwise operations |
| `batchnorm.yaml` | Batch normalization (training forward) |
| `batchnorm_backward.yaml` | Batch normalization backward |
| `batchnorm_inference.yaml` | Batch normalization inference |
| `batchnorm_inference_variance_ext.yaml` | Batch normalization inference (variance extension) |
| `reduction.yaml` | Reduction operations |
| `layernorm_backward.yaml` | Layer normalization backward |
| `rmsnorm_backward.yaml` | RMS normalization backward |
| `resample_fwd.yaml` | Resample forward |
| `resample_bwd.yaml` | Resample backward |
| `moe_grouped_matmul.yaml` | MoE grouped matrix multiplication (checked-in `finalize()` is hand-maintained — see `CLAUDE.md` "Hand-maintained ops") |
| `moe_grouped_matmul_bwd.yaml` | MoE grouped matrix multiplication backward |
| `sdpa.yaml` | Scaled dot-product attention forward (packer/unpacker only; Attributes class is hand-maintained — see `CLAUDE.md` "Hand-maintained ops") |

## Third-Party Libraries

This tool uses the following third-party Python libraries (installed via `pip`):

| Library | License | Purpose |
|---------|---------|---------|
| [Jinja2](https://jinja.palletsprojects.com/) | BSD 3-Clause | Template engine for rendering C++ source from `.j2` templates |
| [MarkupSafe](https://markupsafe.palletsprojects.com/) | BSD 3-Clause | Transitive dependency of Jinja2 (safe string escaping) |
| [PyYAML](https://pyyaml.org/) | MIT | YAML parser for loading operation config files |

See `THIRD_PARTY_LICENSES.md` for full license texts.

## Recent Changes

The codegen overhaul added some YAML keys, removed others, and folded `--mode lift-only` into `--mode backend`. See [`CLAUDE.md`](./CLAUDE.md#recent-changes) for the full breakdown of added/removed surfaces, replacement keys, and current loader behavior.

---

## Field Type Reference

| YAML `type` | C++ Storage | Backend API Type | Use For |
|-------------|-------------|------------------|---------|
| `vector_int64` | `std::vector<int64_t>` | `HIPDNN_TYPE_INT64` | Padding, strides, dimensions |
| `mode` | FBS enum type | Dedicated mode type tag (e.g., `HIPDNN_TYPE_CONVOLUTION_MODE`) | Enum fields in new operations (see `CLAUDE.md` "Enum Field Types") |
| `enum` | FBS enum type | `HIPDNN_TYPE_INT64` | Legacy enum fields only; do not use for new operations |
| `scalar_float` | `float` | `HIPDNN_TYPE_FLOAT` | Alpha/beta scaling factors |
| `scalar_int32` | `int32_t` | `HIPDNN_TYPE_INT32` | 32-bit scalars |
| `scalar_int64` | `int64_t` | `HIPDNN_TYPE_INT64` | Axis indices, counts |
| `bool` | `bool` | `HIPDNN_TYPE_BOOLEAN` | Feature flags |
