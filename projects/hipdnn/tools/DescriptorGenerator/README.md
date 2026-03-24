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
```

## Output

Each run generates:

| File | Purpose |
|------|---------|
| `backend/src/descriptors/<Op>OperationDescriptor.hpp` | Class declaration |
| `backend/src/descriptors/<Op>OperationDescriptor.cpp` | setAttribute/getAttribute, finalize, buildNode, toString |
| `frontend/include/hipdnn_frontend/detail/<Op>Packer.hpp` | Frontend packer function |
| `backend/tests/descriptors/Test<Op>OperationDescriptor.cpp` | Descriptor unit tests |
| `backend/tests/descriptors/TestGraphDescriptor<Op>.cpp` | Graph building tests |
| `tests/frontend/Integration<Op>DescriptorLowering.cpp` | Integration test **stub** (must be implemented) |
| `fragments/*.txt` | Code snippets for manual insertion into existing files |

See `CLAUDE.md` for post-generation integration steps.

## Adding a New Operation

The end-to-end workflow for adding a new operation type:

1. **Create the FBS schema** in `data_sdk/schemas/` and generate the C++ header with `flatc`
2. **Create a YAML config** in `configs/` (use `convolution_fwd.yaml` as a reference)
3. **Run the generator** to produce source files, tests, and fragments
4. **Add enums** to backend headers and string utilities (using the generated fragments)
5. **Add enum test coverage** to `TestBackendEnumStringUtils.cpp`
6. **Place generated files** into the project tree
7. **Update CMake** build files
8. **Build and test** to verify everything compiles and passes
9. **Implement integration test** — the generated integration test is a stub; implement full E2E round-trip tests following the ConvFprop reference at `tests/frontend/IntegrationConvFpropDescriptorLowering.cpp`

See `CLAUDE.md` for detailed integration steps (especially steps 4-7).

### Creating a YAML Config from an FBS Schema

The YAML config maps FBS schema fields to hipDNN backend API concepts. To create a new config:

1. Read the FBS schema in `data_sdk/schemas/`
2. Identify tensor UID fields (`*_tensor_uid: long`) → `tensor_fields`
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

  tensor_fields:
    - name: "input"
      fbs_field: "input_tensor_uid"
      attr_suffix: "INPUT"
      required: true
      frontend_getter: "get_input()"         # Method on the frontend attributes class

  data_fields:
    - name: "alpha"
      fbs_field: "alpha"
      attr_name: "HIPDNN_ATTR_MY_OP_ALPHA"
      type: "scalar_float"            # vector_int64 | enum | scalar_float | scalar_int64 | bool
      required: true
      frontend_getter: "get_alpha()"
      shared: false                   # true if attribute enum already exists from another op
      # For enum fields only:
      # test_enum_value: "ADD"        # Required for enum type - value used in generated tests

  tensor_array_fields:                # For tensor arrays (e.g., peer_stats)
    - name: "peer_stats"
      fbs_field: "peer_stats_tensor_uid"
      attr_name: "HIPDNN_ATTR_OPERATION_MY_OP_PEER_STATS"
      frontend_getter: "get_peer_stats()"
      required: false
      test_uids: [100, 101]           # UIDs for test tensor descriptors
      test_label: "PeerStats"          # Label used in generated test case names

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

## Third-Party Libraries

This tool uses the following third-party Python libraries (installed via `pip`):

| Library | License | Purpose |
|---------|---------|---------|
| [Jinja2](https://jinja.palletsprojects.com/) | BSD 3-Clause | Template engine for rendering C++ source from `.j2` templates |
| [MarkupSafe](https://markupsafe.palletsprojects.com/) | BSD 3-Clause | Transitive dependency of Jinja2 (safe string escaping) |
| [PyYAML](https://pyyaml.org/) | MIT | YAML parser for loading operation config files |

See `THIRD_PARTY_LICENSES.md` for full license texts.

## Field Type Reference

| YAML `type` | C++ Storage | Backend API Type | Use For |
|-------------|-------------|------------------|---------|
| `vector_int64` | `std::vector<int64_t>` | `HIPDNN_TYPE_INT64` | Padding, strides, dimensions |
| `enum` | FBS enum type | `HIPDNN_TYPE_INT64` | Mode selections (ConvMode, etc.) |
| `scalar_float` | `float` | `HIPDNN_TYPE_FLOAT` | Alpha/beta scaling factors |
| `scalar_int64` | `int64_t` | `HIPDNN_TYPE_INT64` | Axis indices, counts |
| `bool` | `bool` | `HIPDNN_TYPE_BOOLEAN` | Feature flags |
