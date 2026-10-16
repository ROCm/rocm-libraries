# RFC 0021 — Supporting Reference: Verified cuDNN FE v0.x and backend C API surface

This document is the long-form companion to
[RFC 0021: cuDNN v8 shim for hipDNN](./0021_CuDNN_V8_Shim.md). It collects the verified tables the
main RFC summarises: the cuDNN backend C API identifiers the v0.x frontend uses and what hipDNN offers
for each, the v0.x class inventory, per-operation setters, the consumer surfaces, NVIDIA's legacy
sample catalogue, and the `CUDNN_VERSION` gates.

Citations are against the local copy of
[NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend) at tag **v1.24.0**
(`CUDNN_FRONTEND_VERSION = 12400`) and the hipDNN tree at the time of writing. Upstream is cited as
`upstream:include/<file>:<line>` (or `upstream:samples/...`); hipDNN as a repo-relative `path:line`.
Inside a table, a bare `:N` refers to the file named in that table's caption.

No numeric value of any cuDNN enum appears here: cuDNN's C headers are proprietary, and the shim
assigns its own values (RFC §4.8). Enums are compared by name only (§1.4). Consumer sources (§4) are
pinned to a commit per repository.

## Table of Contents

1. [Identifier mapping: cuDNN backend C API to hipDNN](#1-identifier-mapping-cudnn-backend-c-api-to-hipdnn)
2. [v0.x class and builder inventory](#2-v0x-class-and-builder-inventory)
3. [Operation setters and attributes per op type](#3-operation-setters-and-attributes-per-op-type)
4. [Consumer surfaces](#4-consumer-surfaces)
5. [Legacy sample catalogue](#5-legacy-sample-catalogue)
6. [Version gates in the v0.x headers](#6-version-gates-in-the-v0x-headers)

---

## 1. Identifier mapping: cuDNN backend C API to hipDNN

Legend for the "hipDNN" columns:

| Mark | Meaning |
|---|---|
| `=` | Same identifier modulo the `CUDNN_` / `HIPDNN_` prefix |
| `≈` | Renamed equivalent (named in the cell) |
| **MISSING** | No hipDNN identifier |
| NS | hipDNN identifier exists, but its setter or getter returns `HIPDNN_STATUS_NOT_SUPPORTED` |
| T! | hipDNN identifier exists but expects a different attribute type |

The "Shim" column states what RFC 0021 does with the row; section references are to the main RFC.

### 1.1 C entry points

hipDNN declarations are in `projects/hipdnn/backend/include/hipdnn_backend.h` (caption file for
`:N`) and implemented in `projects/hipdnn/backend/src/HipdnnBackend.cpp`. The v9 shim's existing
functions are in `projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/cudnn.h`.

| cuDNN entry point | hipDNN | Shim |
|---|---|---|
| `cudnnCreate` | `=` `hipdnnCreate` :88 | exists (v9 shim `cudnn.h:76`) |
| `cudnnDestroy` | `=` :98 | exists (`cudnn.h:84`) |
| `cudnnSetStream` | `=` :109 | exists (`cudnn.h:91`) |
| `cudnnGetStream` | `=` :126 | exists (`cudnn.h:98`) |
| `cudnnGetErrorString` | `=` :282 | exists (`cudnn.h:105`) |
| `cudnnGetVersion`, `cudnnGetCudartVersion` | **MISSING** as integers (`hipdnnVersionString_ext` :995 returns a string) | exist; return the claimed `CUDNN_VERSION` 92200 and `CUDNN_CUDART_VERSION` (`cudnn.h:112,118`) |
| `cudnnBackendCreateDescriptor` | `=` :141-142; unknown type → `NOT_SUPPORTED` (`projects/hipdnn/backend/src/descriptors/DescriptorFactory.cpp:160-161`) | new; creates a shim descriptor for **every** cuDNN type, including ones hipDNN lacks (§4.4) |
| `cudnnBackendDestroyDescriptor` | `=` :160-161 | releases the caller's reference only; shim descriptors are reference-counted (§4.4) |
| `cudnnBackendSetAttribute` | `=` :266-271, same parameter order | typed store with cuDNN attribute-type and count rules (§4.4) |
| `cudnnBackendGetAttribute` | `=` :232-238, same parameter order | descriptor-valued gets fill **caller-created** descriptors in place (§4.4) |
| `cudnnBackendFinalize` | `=` :200 | validates eagerly; hipDNN descriptors are created lazily at operation-graph finalize (§4.5) |
| `cudnnBackendExecute` | `=` :182-184 | one `hipdnnBackendExecute` call; a handle other than the plan's stored context handle → `BAD_PARAM` (§4.7) |
| `cudnnGetLastErrorString` | `=` `hipdnnGetLastErrorString` :297 | new; per-thread buffer merged with hipDNN's message (§4.4) |
| `cudnnBackendInitialize` | **MISSING** (named only in a comment, :190) | not provided: deprecated in cuDNN 9.2 and called nowhere in upstream `include/` or `legacy_samples` (§4.8) |
| `cudnnBackendPopulateCudaGraph`, `cudnnBackendUpdateCudaGraph` | **MISSING** | not provided; v9-only, dropped from the `cudnn_frontend_shim.h` port (§4.9) |
| `cudnnReorderFilterAndBias` (legacy v7, behind `cudnnReorderFilterAndBiasInt8x32`) | **MISSING** | `cudnnReorderFilterAndBiasInt8x32` returns `NOT_SUPPORTED` (§4.9) |

hipDNN extension entry points the shim builds on: `hipdnnGetEngineCount_ext` :753,
`hipdnnGetEngineInfo_ext` :786, `hipdnnGetEngineIdByName_ext` :817, `hipdnnGetEngineNameById_ext` :843,
`hipdnnBackendGetSerializedExecutionPlan_ext` :432, `hipdnnBackendCreateAndDeserializeExecutionPlan_ext`
:454. All shim calls go through `hipdnn_frontend::detail::hipdnnBackend()`, so they are mockable and
work with the dynamically loaded frontend.

### 1.2 Descriptor types

hipDNN's source of truth for "implemented" is the `switch` in `DescriptorFactory::create`
(`projects/hipdnn/backend/src/descriptors/DescriptorFactory.cpp:59-165`, caption file for `:N`); enum
declarations are in `projects/hipdnn/backend/include/HipdnnBackendDescriptorType.h:27-352`.

| cuDNN descriptor type | hipDNN | hipDNN status | Shim |
|---|---|---|---|
| `CUDNN_BACKEND_TENSOR_DESCRIPTOR` | `=` | implemented (:79-80) | lowered at graph finalize, one hipDNN tensor per UID (§4.5) |
| `CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR` | **MISSING** | — | shim parameter descriptor; folded onto the convolution op at graph finalize |
| `CUDNN_BACKEND_POINTWISE_DESCRIPTOR` | **MISSING** | — | folded onto `OPERATION_POINTWISE` |
| `CUDNN_BACKEND_MATMUL_DESCRIPTOR` | **MISSING** | — | folded onto `OPERATION_MATMUL` |
| `CUDNN_BACKEND_REDUCTION_DESCRIPTOR` | **MISSING** | — | folded onto `OPERATION_REDUCTION` |
| `CUDNN_BACKEND_RESAMPLE_DESCRIPTOR` | **MISSING** | — | folded onto the resample ops |
| `CUDNN_BACKEND_RNG_DESCRIPTOR` | **MISSING** | — | creatable and validated; any op using it is `NOT_SUPPORTED` |
| `CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR` | `=` | implemented (:82-83) | lowered, params folded (§4.5) |
| `CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR` | `=` | implemented (:100-101) | lowered, params folded |
| `CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR` | `=` | implemented (:85-86) | lowered, params folded |
| `CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR` | `=` | implemented (:97-98) | lowered with slot mapping and default elision |
| `CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR` | `=` | implemented (:112-113) | lowered |
| `CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR` | `=` | implemented (:145-146) | lowered |
| `CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR` | `=` | implemented (:148-149) | lowered |
| `CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR` | `=` | implemented (:151-152) | lowered |
| `CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR` | **MISSING**; `≈` per-norm `BATCHNORM_DESCRIPTOR_EXT` (:133-134), `BATCHNORM_INFERENCE_DESCRIPTOR_EXT` (:88-89), `LAYERNORM_DESCRIPTOR_EXT` (:127-128), `RMSNORM_DESCRIPTOR_EXT` (:121-122) | implemented (ext) | dispatched on `NORM_FWD_MODE` and phase |
| `CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR` | **MISSING**; `≈` `BATCHNORM_BACKWARD_DESCRIPTOR_EXT` (:103-104), `LAYERNORM_BACKWARD_DESCRIPTOR_EXT` (:130-131), `RMSNORM_BACKWARD_DESCRIPTOR_EXT` (:154-155) | implemented (ext) | dispatched on `NORM_BWD_MODE` |
| `CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR` | `=` | declared only | `NOT_SUPPORTED` at finalize, naming the op |
| `CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR` | **MISSING** | — | `NOT_SUPPORTED` at finalize |
| `CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR` | **MISSING** | — | `NOT_SUPPORTED` at finalize |
| `CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR` | **MISSING** | — | `NOT_SUPPORTED` at finalize |
| `CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR` | **MISSING** | — | `NOT_SUPPORTED` at finalize |
| `CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR` | `=` | declared only | `NOT_SUPPORTED` at finalize |
| `CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR` | `=` | implemented (:70-71) | `HANDLE` required; canonicalize, stable topological sort, lower through `hipdnn_frontend` nodes and packers (§4.5) |
| `CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR` | `=` | implemented (:76-77) | every mode → FALLBACK (§4.6) |
| `CUDNN_BACKEND_ENGINE_DESCRIPTOR` | `=` | implemented (:64-65) | dense per-graph index over the enumerated engines, sorted by signed engine ID (§4.6) |
| `CUDNN_BACKEND_ENGINECFG_DESCRIPTOR` | `=` | implemented (:61-62) | shim-finalized at once; reuses the engine's cached default hipDNN config, or builds its own lazily when knob choices are set (§4.6) |
| `CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR` | `=` | implemented (:94-95) | projected onto `cudnnBackendKnobType_t` (§4.6) |
| `CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR` | `=` | implemented (:91-92) | reverse-mapped to hipDNN string knob IDs (§4.6) |
| `CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR` | `=` | implemented (:67-68) | §4.7 |
| `CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR` | `=` | implemented (:73-74) | §4.7 |
| `CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR` | `=` | declared only | `setKernelCache` with a non-null cache → `NOT_SUPPORTED` |
| `CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR` | **MISSING** | — | `setDeviceProperties` with non-null properties → `NOT_SUPPORTED` |
| `CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR`, `CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR` | `=` | declared only | creatable; no v0.x class uses them; every attribute that references them is `NOT_SUPPORTED` |

hipDNN-only operation types (SDPA, block-scale (de)quantize, MoE grouped matmul, custom op, profiling
control) are unreachable from the v0.x surface; the upstream `OperationBuilder_v8` rejects those modes
itself (§3.1).

### 1.3 Identifier tables per descriptor

In the "Shim" columns, *forwarded*, *folded* and *set on the op* mean the value is placed on the
matching `hipdnn_frontend` attribute object (`TensorAttributes`, `ConvFpropAttributes`,
`PointwiseAttributes`, ...) of a standalone `hipdnn_frontend` node, whose validation, inference and
packer (`projects/hipdnn/frontend/include/hipdnn_frontend/detail/*Packer.hpp`) emit the hipDNN
attributes (RFC §4.5). The shim issues no hand-written `hipdnnBackendSetAttribute` sequences for
operations or tensors.

**Tensor.** Caption: `upstream:include/cudnn_frontend_Tensor.h`.

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_ATTR_TENSOR_DATA_TYPE` :372 | `=` | translated by name; unmappable types (`INT8x4`, `UINT8x4`, `INT8x32`, `FAST_FLOAT_FOR_FP8`, `COMPLEX_*`) fail closed |
| `CUDNN_ATTR_TENSOR_DIMENSIONS` :382, `STRIDES` :392, `UNIQUE_ID` :402, `IS_VIRTUAL` :422 | `=` | forwarded; one hipDNN tensor per UID, conflicting definitions → `BAD_PARAM` |
| `CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT` :409 | `=` | ≥ 16 not set; < 16 forwarded (the graph then needs plugin API ≥ 1.3.0, which no current provider reports — divergence); never forwards > 16 |
| `CUDNN_ATTR_TENSOR_IS_BY_VALUE` :436 (set) | NS on set (read-only) | `≈` `HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT`; variant-pack entries are host pointers |
| `CUDNN_ATTR_TENSOR_CONSTANT_VALUE` :527-557 | `≈` `HIPDNN_ATTR_TENSOR_VALUE_EXT` | forwarded; bytes copied at tensor finalize (upstream passes a pointer into builder-owned storage); takes precedence over `IS_BY_VALUE` |
| `CUDNN_ATTR_TENSOR_VECTOR_COUNT` :451, `VECTORIZED_DIMENSION` :465 | **MISSING** | `VECTOR_COUNT` ≠ 1 → `NOT_SUPPORTED` |
| `CUDNN_ATTR_TENSOR_REORDERING_MODE` :514 | **MISSING** (the attribute type exists) | ≠ NONE → `NOT_SUPPORTED` |
| `CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC` :488 | `=` | forwarded; the graph then needs plugin API ≥ 1.3.0 (`K_RAGGED_TENSOR_MIN_API_VERSION`, `projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp:32`), which no current provider reports — divergence |

Evidence: hipDNN's tensor setter switch has no `IS_BY_VALUE` case
(`projects/hipdnn/backend/src/descriptors/TensorDescriptor.cpp:168-250`); the attribute is documented
read-only (`projects/hipdnn/backend/include/HipdnnBackendAttributeName.h:477-478`). The graph's minimum
plugin API version is computed from ragged tensors and non-default alignment
(`projects/hipdnn/backend/src/descriptors/GraphDescriptor.cpp:161-169`; detection `:670-686`); both
floors are 1.3.0 (`projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp:32-39`).

**Convolution.** Caption: `upstream:include/cudnn_frontend_Operation.h` unless prefixed `ConvDesc.h`
(= `upstream:include/cudnn_frontend_ConvDesc.h`).

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR` | **MISSING** | parameter descriptor held by the shim |
| `CUDNN_ATTR_CONVOLUTION_COMP_TYPE` / `CONV_MODE` / `DILATIONS` / `FILTER_STRIDES` / `PRE_PADDINGS` / `POST_PADDINGS` (ConvDesc.h:307-390) | `=`, but set on the **operation** descriptor | folded onto the hipDNN conv op at graph finalize |
| `CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS` (ConvDesc.h:335) | **MISSING** (implied by array lengths) | checked against the array lengths |
| `CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X` / `W` / `Y` (:1970-1997) | `=` | forwarded |
| `CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX` / `W` / `DY` (:596-626) | `=` | forwarded |
| `CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X` / `DW` / `DY` (:976-1007) | `=` | forwarded |
| `CUDNN_ATTR_OPERATION_CONVOLUTION_{FORWARD,BWD_DATA,BWD_FILTER}_CONV_DESC` (:2006, :636, :1017) | **MISSING** | consumed by the shim; parameters folded |
| `CUDNN_ATTR_OPERATION_CONVOLUTION_{FORWARD,BWD_DATA,BWD_FILTER}_ALPHA` / `BETA` (:2022-2037, :653-668, :1033-1048) | **MISSING** | alpha ≠ 1 or beta ≠ 0 → `NOT_SUPPORTED` at op finalize, naming the attribute |

Evidence: conv ops route the parameter attributes to `setConvolutionAttribute`
(`projects/hipdnn/backend/src/descriptors/DescriptorAttributeUtils.hpp:519-575`).

**Pointwise.** Caption: `upstream:include/cudnn_frontend_Operation.h` unless prefixed `PW.h`
(= `upstream:include/cudnn_frontend_PointWiseDesc.h`).

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_BACKEND_POINTWISE_DESCRIPTOR` | **MISSING** | parameter descriptor held by the shim |
| `CUDNN_ATTR_POINTWISE_MODE` (PW.h:231) | `=`, on the op | translated by name; `COS`, `MOD`, `POW` have no hipDNN mode and fail closed |
| `CUDNN_ATTR_POINTWISE_MATH_PREC` (PW.h:253) | `=`, on the op | forwarded |
| `CUDNN_ATTR_POINTWISE_NAN_PROPAGATION` (PW.h:267) | **MISSING** | stored and ignored (divergence) |
| `CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP` / `RELU_UPPER_CLIP` / `RELU_LOWER_CLIP_SLOPE` / `ELU_ALPHA` / `SOFTPLUS_BETA` / `SWISH_BETA` (PW.h:280-362, double) | `=`, T! (hipDNN requires float) | value equal to cuDNN's default for the mode → not set; otherwise narrowed to float, clips saturate at ±`FLT_MAX`, non-finite slope/beta → `BAD_PARAM` |
| `CUDNN_ATTR_POINTWISE_AXIS` (PW.h:372) | `=` | default (−1) not set; otherwise forwarded |
| `CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR` (:455) | **MISSING** | consumed by the shim |
| `CUDNN_ATTR_OPERATION_POINTWISE_XDESC` / `BDESC` / `TDESC` / `YDESC` (:468, :552, :567, :482) | `≈` `HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT` / `IN_1_EXT` / `IN_2_EXT` / `OUT_0_EXT` | X→IN_0, B→IN_1, T→IN_2, Y→OUT_0 |
| `CUDNN_ATTR_OPERATION_POINTWISE_DXDESC` / `DYDESC` (:508, :495) | **MISSING** | backward activations: DY→IN_0, X→IN_1, DX→OUT_0 |
| `CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1` / `ALPHA2` (:526, :538) | **MISSING** | ≠ 1 → `NOT_SUPPORTED` at op finalize |

Evidence: float-typed parameter setters at
`projects/hipdnn/backend/src/descriptors/PointwiseOperationDescriptor.cpp:94-141`.

**Matmul.** Caption: `upstream:include/cudnn_frontend_Operation.h` unless prefixed `MM.h`
(= `upstream:include/cudnn_frontend_MatMulDesc.h`).

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_BACKEND_MATMUL_DESCRIPTOR` | **MISSING** | parameter descriptor held by the shim |
| `CUDNN_ATTR_MATMUL_COMP_TYPE` (MM.h:140) | `=`, on the op | folded onto the op |
| `CUDNN_ATTR_MATMUL_PADDING_VALUE` (MM.h:160) | **MISSING** | `NOT_SUPPORTED` when set |
| `CUDNN_ATTR_OPERATION_MATMUL_ADESC` / `BDESC` / `CDESC` (:350-373) | `=` | forwarded |
| `CUDNN_ATTR_OPERATION_MATMUL_DESC` (:426) | **MISSING** | consumed by the shim |
| `CUDNN_ATTR_OPERATION_MATMUL_GEMM_{M,N,K}_OVERRIDE_DESC` (:384-415) | **MISSING** | `NOT_SUPPORTED` when set |

**Reduction.** Caption: `upstream:include/cudnn_frontend_Operation.h` unless prefixed `RD.h`
(= `upstream:include/cudnn_frontend_ReductionDesc.h`).

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_BACKEND_REDUCTION_DESCRIPTOR` | **MISSING** | parameter descriptor held by the shim |
| `CUDNN_ATTR_REDUCTION_OPERATOR` / `COMP_TYPE` / `IS_DETERMINISTIC` (RD.h:177, :155, :205) | `=`, on the op | folded; operator translated by name |
| `CUDNN_ATTR_OPERATION_REDUCTION_XDESC` / `YDESC` (:314, :326) | `=` | forwarded |
| `CUDNN_ATTR_OPERATION_REDUCTION_DESC` (:301) | **MISSING** | consumed by the shim |

**Resample.** Caption: `upstream:include/cudnn_frontend_Operation.h` unless prefixed `RS.h`
(= `upstream:include/cudnn_frontend_Resample.h`).

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_BACKEND_RESAMPLE_DESCRIPTOR` | **MISSING** | parameter descriptor held by the shim |
| `CUDNN_ATTR_RESAMPLE_MODE` / `COMP_TYPE` / `PADDING_MODE` (RS.h:339, :361, :396) | `=`, on the op | folded; NEAREST and BILINEAR → `NOT_SUPPORTED`; `EDGE_VAL_PAD` has no hipDNN mode and fails closed |
| `CUDNN_ATTR_RESAMPLE_WINDOW_DIMS` / `PRE_PADDINGS` / `POST_PADDINGS` / `STRIDES` (RS.h:422-464, `cudnnFraction_t`) | `=`, T! (hipDNN stores `int64` vectors) | integer fractions converted; non-integer → `NOT_SUPPORTED` |
| `CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION` (RS.h:374) | **MISSING** | stored and ignored (divergence) |
| `CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS` (RS.h:409) | **MISSING** | checked against the array lengths |
| `CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC` / `YDESC` / `IDXDESC` (:1385-1451) | `=` | forwarded |
| `CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC` / `DYDESC` / `IDXDESC` (:1477, :1527, :1578) | `=` | forwarded |
| `CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC` / `YDESC` (:1495, :1513) | **MISSING** | `NOT_SUPPORTED` when set (hipDNN's backward resample has no such inputs) |
| `CUDNN_ATTR_OPERATION_RESAMPLE_{FWD,BWD}_ALPHA` / `BETA` (:1409-1436, :1539-1566) | **MISSING** | alpha ≠ 1 or beta ≠ 0 → `NOT_SUPPORTED` |
| `CUDNN_ATTR_OPERATION_RESAMPLE_{FWD,BWD}_DESC` | **MISSING** | consumed by the shim |

**Norm, BN, GenStats, RNG, Reshape, paged cache.** Caption: `upstream:include/cudnn_frontend_Operation.h`
unless prefixed `Rng.h` (= `upstream:include/cudnn_frontend_Rng.h`).

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_ATTR_OPERATION_NORM_FWD_*` (`MODE`, `PHASE`, `XDESC`, `MEAN_DESC`, `INV_VARIANCE_DESC`, `SCALE_DESC`, `BIAS_DESC`, `EPSILON_DESC`, `EXP_AVG_FACTOR_DESC`, `INPUT/OUTPUT_RUNNING_MEAN/VAR_DESC`, `YDESC`, `PEER_STAT_DESCS`; :1094-1238) | **MISSING** as one op; `≈` split per norm (`BATCHNORM*_EXT`, `LAYERNORM_EXT`, `RMSNORM_EXT`); no `NORM_MODE` type | BATCH → `BatchnormAttributes` (training) or `BatchnormInferenceAttributes` (mean + inverse variance); LAYER → `LayernormAttributes` (normalized-dimension count inferred by the hipDNN node); RMS → `RMSNormAttributes` (inverse variance → inverse RMS); `EXP_AVG_FACTOR_DESC` → hipDNN `MOMENTUM`; epsilon / momentum become runtime pass-by-value scalars when the caller marks them by-value; compute type synthesized as FLOAT (cuDNN norm ops carry none, hipDNN requires one — divergence); INSTANCE, GROUP, ADA_LAYER, `PEER_STAT_DESCS` → `NOT_SUPPORTED` |
| `CUDNN_ATTR_OPERATION_NORM_BWD_*` (:1273-1365) | **MISSING**; `≈` `BATCHNORM_BACKWARD_DESCRIPTOR_EXT`, `LAYERNORM_BACKWARD_DESCRIPTOR_EXT`, `RMSNORM_BACKWARD_DESCRIPTOR_EXT` | BATCH / LAYER / RMS → `BatchnormBackwardAttributes` / `LayernormBackwardAttributes` / `RMSNormBackwardAttributes`; compute type FLOAT as forward; peer stats and other modes → `NOT_SUPPORTED` |
| `CUDNN_ATTR_OPERATION_BN_FINALIZE_*` (:705-876) | **MISSING** | settable; op `NOT_SUPPORTED` at finalize |
| `CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_*` (:1836-1951) | **MISSING** | settable; op `NOT_SUPPORTED` at finalize |
| `CUDNN_ATTR_OPERATION_GENSTATS_*` (:896-951) | descriptor type declared, no class or attributes | settable; op `NOT_SUPPORTED` at finalize |
| `CUDNN_ATTR_RNG_*` (Rng.h:236-302) | **MISSING** | stored and validated |
| `CUDNN_ATTR_OPERATION_RNG_YDESC` / `SEED` / `DESC` / `OFFSET_DESC` (:1612-1675) | **MISSING** | settable; op `NOT_SUPPORTED` at finalize |
| `CUDNN_ATTR_OPERATION_RESHAPE_XDESC` / `YDESC` / `MODE` (:1771-1805) | **MISSING** | settable; op `NOT_SUPPORTED` at finalize |
| `CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_*` (:1731-1752) | descriptor type declared only | settable; op `NOT_SUPPORTED` at finalize |

**Operation graph.** Caption: `upstream:include/cudnn_frontend_OperationGraph.h`.

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_ATTR_OPERATIONGRAPH_OPS` :265 | `=`; hipDNN **appends** on repeated sets and requires finalized ops | replace semantics in the shim; hipDNN `OPS` set once, after a stable topological sort |
| `CUDNN_ATTR_OPERATIONGRAPH_HANDLE` :279 | `=`, set-only; hipDNN graph finalize rejects a null handle (`projects/hipdnn/backend/src/descriptors/GraphDescriptor.cpp:34`) | **required**: missing → `BAD_PARAM` at finalize, as the v9 shim's handle-less build; cuDNN ≥ 9.8 accepts a handle-less graph (divergence, §6) |
| `CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT` :77 (get) | NS | synthesized: N = number of enumerated engines; the first engine query runs the enumeration and needs a HIP device (§4.6) |
| `CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED` :299 | NS | `true` → `NOT_SUPPORTED` |
| `CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED` :320 | `≈` `HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT` | `true` → `NOT_SUPPORTED` (the v8 variant pack has no override attributes) |

Evidence: `setOperations` requires finalized ops and appends
(`projects/hipdnn/backend/src/descriptors/GraphDescriptor.cpp:348-388`); setter switch `:403-426`;
getter switch `:248-287` has no `HANDLE` or `ENGINE_GLOBAL_COUNT` case.

**Engine heuristics.** Caption: `upstream:include/cudnn_frontend_Heuristics.h`.

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH` :210 | `=` | forwarded |
| `CUDNN_ATTR_ENGINEHEUR_MODE` :223; `CUDNN_HEUR_MODE_INSTANT` / `A` / `B` / `FALLBACK` | `=`, but only `HIPDNN_HEUR_MODE_FALLBACK` is accepted | every mode → FALLBACK; first use of each other mode warns once per process |
| `CUDNN_ATTR_ENGINEHEUR_RESULTS` :92, :112 (get) | `=`; fills caller ENGINECFGs in place but leaves them **unfinalized** | count query honored; fills caller-created shim configs in rank order, shim-finalized at once, each reusing the engine's cached default hipDNN config |
| `CUDNN_ATTR_ENGINEHEUR_DEVICEPROP` :238, `SM_COUNT_TARGET` :260 | NS | `NOT_SUPPORTED` when set |

Evidence: FALLBACK-only check at
`projects/hipdnn/backend/src/descriptors/EngineHeuristicDescriptor.cpp:457-463`.

**Engine and knobs.** Caption: `upstream:include/cudnn_frontend_Engine.h` unless prefixed
`EC.h` (= `cudnn_frontend_EngineConfig.h`), `EP.h` (= `cudnn_frontend_ExecutionPlan.h`) or `F.h`
(= `cudnn_frontend_Filters.h`), all under `upstream:include/`.

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_ATTR_ENGINE_OPERATION_GRAPH` :266 | `=` | forwarded |
| `CUDNN_ATTR_ENGINE_GLOBAL_INDEX` :279 | `=`, but the value is a 64-bit hash of the engine name | dense `0..N-1` index ↔ hipDNN ID. The engines are those returned by hipDNN's FALLBACK heuristic whose default config finalizes, sorted by **signed** engine ID; enumerated lazily on the first engine query of a graph (needs a HIP device) and cached per graph (§4.6) |
| `CUDNN_ATTR_ENGINE_KNOB_INFO` :315 (get) | `=`; get allocates new descriptors | in-place fill, capped at `CUDNN_KNOB_TYPE_COUNTS` |
| `CUDNN_ATTR_ENGINE_NUMERICAL_NOTE` (EP.h:167, F.h:56) | NS | synthesized: `NONDETERMINISTIC` unless the engine claims determinism; nothing else (§4.6) |
| `CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE` (EP.h:193, F.h:84) | `=`; enumerators differ | mapped by name; only `RUNTIME_COMPILATION` exists on both sides |
| `CUDNN_ATTR_ENGINE_LAYOUT_INFO`, `SM_COUNT_TARGET`, `DEVICEPROP` | NS (`SM_COUNT_TARGET` `≈` `CU_COUNT_TARGET_EXT`) | `NOT_SUPPORTED` |
| `CUDNN_ATTR_KNOB_INFO_TYPE` / `MAXIMUM_VALUE` / `MINIMUM_VALUE` / `STRIDE` :126-150 | `=`; `TYPE` is a string, T! | projected through the v9 shim's `detail/knob_wrapper.h`; a knob with no cuDNN type, or whose valid values cannot be expressed as min / max / stride, is hidden |
| `CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE` / `KNOB_VALUE` (EC.h:122, :134) | `=`; `KNOB_TYPE` is a string, T! | reverse-mapped; a hidden type → `NOT_SUPPORTED`; a duplicate type → `BAD_PARAM`; values checked with hipDNN's knob validation |
| `cudnnBackendKnobType_t`, `CUDNN_KNOB_TYPE_*`, `CUDNN_KNOB_TYPE_COUNTS` | **MISSING** (knobs are plugin-defined string IDs) | shim-defined enum, values shim-assigned |
| `CUDNN_ATTR_ENGINECFG_ENGINE` (EC.h:184; get EP.h:447, F.h:50) | `=`; get allocates | in-place fill |
| `CUDNN_ATTR_ENGINECFG_KNOB_CHOICES` (EC.h:202 set; EP.h:253 get) | `=`; set supported, get NS | answered from shim state |
| `CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE` | `=`, get-only | forwarded from the cached default hipDNN config; a config with knob choices creates and finalizes its own on the first query or at plan finalize |
| `CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO` | NS | `NOT_SUPPORTED` |

Evidence: `projects/hipdnn/backend/src/descriptors/EngineDescriptor.cpp:146-155` (numerical note,
layout info, CU count, device properties → `NOT_SUPPORTED`);
`projects/hipdnn/backend/src/descriptors/EngineConfigDescriptor.cpp:73-77` (`KNOB_CHOICES` get →
`NOT_SUPPORTED`).

**Execution plan.** Caption: `upstream:include/cudnn_frontend_ExecutionPlan.h`.

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_ATTR_EXECUTION_PLAN_HANDLE` :505 | `=`; accepted and ignored | **required**; must equal the operation graph's handle (`BAD_PARAM` otherwise); stored as the context handle that `cudnnBackendExecute` compares against |
| `CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG` :399 (get :537) | `=`; set requires a finalized config; get allocates | must be finalized; get fills in place |
| `CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE` :298 | `=` | forwarded |
| `CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION` :128, :492 | NS both ways | get: shim-generated JSON object; `"engine"` is always an object with `globalIndex`, `id` (a decimal string, since hashed IDs exceed JSON's exact integer range), optional `name` and `behaviorNotes`; `payload` (base64) only when `hipdnnBackendGetSerializedExecutionPlan_ext` succeeds (§4.7). Set (`loadFromJson`): parsed at plan finalize by a fixed-format reader; any other format → `BAD_PARAM`; no `payload` → `NOT_SUPPORTED`; otherwise deserialized via `hipdnnBackendCreateAndDeserializeExecutionPlan_ext` |
| `CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE` :414 | NS | `NOT_SUPPORTED` |
| `CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS` / `RUN_ONLY_INTERMEDIATE_UIDS` / `DEVICEPROP` | NS | `NOT_SUPPORTED` |

Evidence: `projects/hipdnn/backend/src/descriptors/ExecutionPlanDescriptor.cpp:152-160` (get NS),
`:210-222` (set NS), `:248` (handle ignored), `:270-273` (unfinalized config rejected).

**Variant pack.** Caption: `upstream:include/cudnn_frontend_VariantPack.h`.

| cuDNN identifier | hipDNN | Shim |
|---|---|---|
| `CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS` :139, `UNIQUE_IDS` :152, `WORKSPACE` :165 | `=` | forwarded; the hipDNN variant pack is built at shim variant-pack finalize |
| `CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES` | NS | `NOT_SUPPORTED` |

**Attribute types.** Every `CUDNN_TYPE_*` the v0.x headers use exists by name in hipDNN except
`CUDNN_TYPE_NORM_MODE` and `CUDNN_TYPE_RESHAPE_MODE` (**MISSING**); the shim declares both and checks
attribute types against cuDNN's rules, not hipDNN's.

### 1.4 Enum name comparison

Only names are compared. The design never relies on NVIDIA's numeric values: the shim assigns its own
(RFC §4.8), every translation between cuDNN and hipDNN enums is a switch on names, never a cast, and
unknown values fail closed. Status codes are translated by the v9 shim's existing switch
(`projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/detail/status_translation.h`).

| Enum family | Name differences (cuDNN v0.x usage vs hipDNN) |
|---|---|
| Attribute names | prefix rename for the shared core; many hipDNN `_EXT` additions; see §1.3 for missing names |
| Descriptor types | prefix rename; see §1.2 for missing types |
| Attribute types | all v0.x types exist except `NORM_MODE`, `RESHAPE_MODE` |
| Data types | hipDNN lacks `INT8x4`, `UINT8x4`, `INT8x32`, `FAST_FLOAT_FOR_FP8`, `COMPLEX_FP32`, `COMPLEX_FP64`; hipDNN adds `FP6_*_EXT`, `FP8_*_FNUZ` |
| Pointwise modes | hipDNN lacks `COS`, `MOD`, `POW` |
| Convolution mode | same names |
| Heuristic modes | hipDNN has only `FALLBACK` |
| Behavior notes | shared: `RUNTIME_COMPILATION`. cuDNN-only: `REQUIRES_FILTER_INT8x32_REORDER`, `REQUIRES_BIAS_INT8x32_REORDER`, `SUPPORTS_CUDA_GRAPH_NATIVE_API`, `CUBLASLT_DEPENDENCY`. hipDNN-only: `REQUIRES_LAYOUT_TRANSFORM`, `SUPPORTS_GRAPH_CAPTURE`, `EXTERNAL_LIBRARY_DEPENDENCY`, `SUPPORTS_EXECUTION_PLAN_SERIALIZATION` |
| Numerical notes | **MISSING** in hipDNN |
| Reduce-tensor ops | same names |
| Resample modes | hipDNN lacks `NEAREST`, `BILINEAR` |
| Padding modes | hipDNN lacks `EDGE_VAL_PAD`; adds `NOT_SET` |
| Norm forward phase | same names |
| Status codes | the shim keeps the v9 shim's single `cudnnStatus_t` (`cudnn_status.h`) |

---

## 2. v0.x class and builder inventory

### 2.1 Header inventory

Line counts are `wc -l`. Caption directory: `upstream:include/`. "Shim" is the RFC 0021 disposition
(RFC §4.1).

| Header | Lines | Main public contents | Shim |
|---|---:|---|---|
| `cudnn_frontend.h` (umbrella) | 152 | includes every v0.x header and the v1 graph API; `_v8` alias block :137-152 | extend (existing v9 umbrella) |
| `cudnn_backend_base.h` | 165 | `OpaqueBackendPointer`, `ManagedOpaqueDescriptor`, `make_shared_backend_pointer`, `BackendDescriptor` | port |
| `cudnn_frontend_utils.h` (shared with v1) | 2616 | JSON glue, `cudnnException`, `set_error_and_throw_exception`, `AllowAll`, `to_string`, FE enums, `detail::convert_*` | extend (existing) |
| `cudnn_frontend_shim.h` (shared with v1) | 722 | `detail::create_descriptor` / `set_attribute` / `get_attribute` / `finalize` / `execute`, CUDA wrappers, dynamic loading | port subset |
| `cudnn_frontend_Logging.h` | 174 | env-driven logging, `CUDNN_FE_LOG*` | port subset |
| `cudnn_frontend_Tensor.h` | 583 | `Tensor_v8`, `TensorBuilder_v8` | port |
| `cudnn_frontend_ConvDesc.h` | 425 | `ConvDesc_v8`, `ConvDescBuilder_v8` | port |
| `cudnn_frontend_PointWiseDesc.h` | 411 | `PointWiseDesc_v8`, `PointWiseDescBuilder_v8` | port |
| `cudnn_frontend_MatMulDesc.h` | 198 | `MatMulDesc_v8`, `MatMulDescBuilder_v8` | port |
| `cudnn_frontend_ReductionDesc.h` | 245 | `ReductionDesc_v8`, `ReductionDescBuilder_v8` | port |
| `cudnn_frontend_Resample.h` | 494 | `ResampleDesc_v8`, `ResampleDescBuilder_v8` | port |
| `cudnn_frontend_Rng.h` | 339 | `RngDesc_v8`, `RngDescBuilder_v8` | port |
| `cudnn_frontend_Operation.h` | 3115 | `Operation_v8`, `OperationBuilder_v8` | port |
| `cudnn_frontend_OperationGraph.h` | 360 | `OperationGraph_v8`, `OperationGraphBuilder_v8`, `MAX_OPGRAPH_OPS = 250` (:36) | port |
| `cudnn_frontend_Heuristics.h` | 491 | `EngineHeuristics_v8`, `EngineHeuristicsBuilder_v8`, `get_heuristics_list` | port |
| `cudnn_frontend_Engine.h` | 341 | `Engine_v8` (nested `Knob`), `EngineBuilder_v8` | port |
| `cudnn_frontend_EngineConfig.h` | 246 | `EngineConfig_v8`, `EngineConfigBuilder_v8`, `EngineConfigList` | port |
| `cudnn_frontend_EngineFallbackList.h` | 180 | `get_fallback_engine_list`, `EngineFallbackList_v8`, `EngineFallbackListBuilder_v8` | port |
| `cudnn_frontend_Filters.h` | 96 | `filter`, `hasNumericalNote<>`, `hasBehaviorNote<>` | port |
| `cudnn_frontend_ExecutionPlan.h` | 599 | `ExecutionPlan_v8`, `ExecutionPlanBuilder_v8` | port |
| `cudnn_frontend_VariantPack.h` | 202 | `VariantPack_v8`, `VariantPackBuilder_v8` | port |
| `cudnn_frontend_Errata.h` | 293 | `load_from_config`, `check_shape`, `check_rule`, `check_errata` | port |
| `cudnn_frontend_ExecutionPlanCache.h` | 183 | `ExecutionPlanCache_v1`, `ExecutionPlanCache_v2`, `ExecutionPlanCache` | port |
| `cudnn_frontend_Reorder_Tensor.h` | 105 | `cudnnReorderFilterAndBiasInt8x32` | port (`NOT_SUPPORTED`) |
| `cudnn_frontend_EngineConfigGenerator.h` | 123 | `executionPlans_t`, `Predicate`, `GeneratorSource`, `CudnnFindSamplingTechnique`, `EngineConfigGenerator`, `filter(Predicate, plans)` | port |
| `cudnn_frontend_get_plan.h` | 62 | `EngineConfigGenerator::cudnnGetPlan` definitions | port |
| `cudnn_frontend_find_plan.h` | 163 | `time_sorted_plan<>`, `cudnnFindPlan<>`, `cudnnFindPlanAndCache<>` | port (HIP events) |

Totals: v0.x-only headers excluding the umbrella, `utils` and `shim`: **9,593 lines**; including
`utils` and `shim`: **12,931 lines**. `Operation.h` alone is about 32% of the v0.x-only code.

The umbrella includes the v0.x headers at :106-127 but **not** `EngineConfigGenerator.h`,
`get_plan.h` or `find_plan.h`; samples include those explicitly. `cudnn_frontend_Rng.h` and
`cudnn_frontend_ReductionDesc.h` arrive transitively.

### 2.2 `_v8` aliases

| Aliases | Defined at (`upstream:include/`) |
|---|---|
| `ConvDesc`, `ConvDescBuilder`, `ReductionDesc`, `ReductionDescBuilder`, `EngineHeuristicsBuilder`, `EngineHeuristics`, `EngineBuilder`, `Engine`, `EngineConfig`, `EngineConfigBuilder`, `EngineFallbackList`, `EngineFallbackListBuilder`, `ResampleDesc`, `ResampleDescBuilder` | `cudnn_frontend.h:138-151` |
| `Tensor`, `TensorBuilder` | `cudnn_frontend_Tensor.h:580-581` |
| `Operation`, `OperationBuilder` | `cudnn_frontend_Operation.h:3113-3114` |
| `OperationGraph`, `OperationGraphBuilder` | `cudnn_frontend_OperationGraph.h:357-358` |
| `ExecutionPlan`, `ExecutionPlanBuilder` | `cudnn_frontend_ExecutionPlan.h:596-597` |
| `VariantPack`, `VariantPackBuilder` | `cudnn_frontend_VariantPack.h:199-200` |
| `MatMulDesc`, `MatMulDescBuilder` | `cudnn_frontend_MatMulDesc.h:196-197` |
| `PointWiseDescBuilder`, `PointWiseDesc`, `PointwiseDescBuilder`, `PointwiseDesc` | `cudnn_frontend_PointWiseDesc.h:407-410` |
| `ConvDesc`, `ConvDescBuilder` (duplicate of the umbrella's) | `cudnn_frontend_ConvDesc.h:422-423` |
| `ReductionDesc`, `ReductionDescBuilder` (duplicate of the umbrella's) | `cudnn_frontend_ReductionDesc.h:242-243` |
| `RngDesc`, `RngDescBuilder` | `cudnn_frontend_Rng.h:337-338` |
| `EngineHeuristicsBuilde` (sic, kept for source compatibility), `EngineHeuristics` (duplicate) | `cudnn_frontend_Heuristics.h:485-486` |
| `ExecutionPlanCache = ExecutionPlanCache_v2` | `cudnn_frontend_ExecutionPlanCache.h:181` |
| `EngineConfigList = std::vector<ManagedOpaqueDescriptor>` | `cudnn_frontend_EngineConfig.h:245` |
| `executionPlans_t`, `Predicate`, `GeneratorSource` | `cudnn_frontend_EngineConfigGenerator.h:30-32` |
| `feature_vector_t = std::vector<int64_t>` | `cudnn_frontend_utils.h:203` |

### 2.3 Base infrastructure

| Symbol | Location (`upstream:include/`) | Notes |
|---|---|---|
| `OpaqueBackendPointer` | `cudnn_backend_base.h` | ctor creates the descriptor, dtor destroys it; move-only; `get_backend_descriptor()`, `get_status()`, `is_good()` |
| `ManagedOpaqueDescriptor`, `make_shared_backend_pointer(type)` | `cudnn_backend_base.h` | `std::shared_ptr<OpaqueBackendPointer>` |
| `BackendDescriptor` | `cudnn_backend_base.h` | abstract base of every `*_v8`: `describe()`, `get_raw_desc()`, `get_status()`, `set_status()`, `set_error()`, `get_error()`, `get_desc()`, public `initialize_managed_backend_pointer(type)` (v9 graph nodes call it) |
| `cudnnException : std::runtime_error` | `cudnn_frontend_utils.h:205` | `what()`, `getCudnnStatus()` |
| `set_error_and_throw_exception` | `cudnn_frontend_utils.h:237` | appends `cudnnGetLastErrorString`; throws unless `NV_CUDNN_DISABLE_EXCEPTION` |
| `AllowAll` | `cudnn_frontend_utils.h:223` | always `false` (keep everything) |
| `get_pointwise_mode_port_count` | `cudnn_frontend_utils.h:796` | decides which pointwise slots are set |
| `detail::create_descriptor` / `destroy_descriptor` / `set_attribute` / `get_attribute` / `finalize` / `execute` | `cudnn_frontend_shim.h:531-581` | the only path from builders to the C API |
| Macros | `CUDNN_FRONTEND_UNUSED` (`utils.h:199`), `NV_CUDNN_DISABLE_EXCEPTION`, `CUDNN_FRONTEND_SKIP_JSON_LIB`, `NV_FE_CALL_TO_BACKEND` (`shim.h:171-180`), `NV_FE_CALL_TO_CUDA` / `NV_FE_CALL_TO_CU` (`shim.h:182-194`), `CUDNN_FE_LOG*` (`Logging.h:125-164`), `NV_CUDNN_FRONTEND_DISABLE_LOGGING` | `CUDNN_FE_LOG*` already exist, `#ifndef`-guarded, in the v9 shim |
| FE enums | `cudnn_frontend_utils.h:334-794` | `DescriptorType_t`, `NormMode_t`, `RngDistribution_t` become shim-owned (RFC §4.8) |

### 2.4 Classes and public methods

Builders return `XxxBuilder_v8&` from setters and `Xxx_v8&&` from `build()`; built objects are
move-only. Exceptions: `PointWiseDescBuilder_v8::build()` returns by value
(`cudnn_frontend_PointWiseDesc.h:210-211`), and `ExecutionPlan_v8` is **copyable**
(`cudnn_frontend_ExecutionPlan.h:156-158`). Caption directory: `upstream:include/`.

| Class (line) / builder (line) | Getters | Builder setters |
|---|---|---|
| `Tensor_v8` (`Tensor.h:54`) / `TensorBuilder_v8` (:195) | `getPackedElementCount`, `getDimCount`, `getDim`, `getStride`, `getDimArray`, `getStrideArray`, `getDataType`, `getId`, `getAlignment`, `isVirtualTensor`, `getDimensionCount` | `setDataType` ×2, `setDim`, `setStride`, `setStrides` (deprecated), `setId`, `setAlignment`, `setVirtual`, `setByValue`, `setConstValue<T>` (:254), `setVectorCountAndDimension`, `setReorderType` ×2, `setRaggedOffset`, `cloneFrom`, `build` (:317) |
| `ConvDesc_v8` (`ConvDesc.h:67`) / builder (:179) | `getComputePrecision`, `getComputeType`, `getSpatialDimCount`, `getPadding`, `getSpatialStride`, `getDilation`, `getMathMode`, `getDimensionCount`, `getStride` | `setComputeType` ×2, `setPrePadding`, `setPostPadding`, `setDilation`, `setSpatialStride`, `setSpatialDimCount`, `setMathMode` ×2; deprecated `setNDims`, `setDataType`, `setComputePrecision`, `setStrides` |
| `PointWiseDesc_v8` (`PointWiseDesc.h:59`) / builder (:113) | `getPortCount`, `getPointWiseMode` | `setComputeType` ×2, `setClipping`, `setMode` ×3 (the `cudnnNanPropagation_t` overload sets NaN propagation), `setMathPrecision`, `setReluLowerClip`, `setReluUpperClip`, `setReluLowerClipSlope`, `setEluAlpha`, `setSoftplusBeta`, `setSwishBeta`, `setAxis` |
| `MatMulDesc_v8` (`MatMulDesc.h:50`) / builder (:85) | — | `setComputeType` ×2, `setMathPrecision`, `setPaddingValue` |
| `ReductionDesc_v8` (`ReductionDesc.h:52`) / builder (:91) | — | `setComputeType` ×2, `setReductionOp` ×2, `setMathPrecision`, `setIsDeterministic` |
| `ResampleDesc_v8` (`Resample.h:50`) / builder (:172) | `getComputeType`, `getSpatialDimCount`, `getNanOpt`, `getMode`, `getPaddingMode`, `getSpatialStride`, `getPrePadding`, `getPostPadding`, `getWindowDim` | `setComputeType` ×2, `setNanPropagation`, `setPostPadding` / `setPrePadding` / `setSpatialStride` / `setSpatialDim` (fraction and `int64_t*` overloads), `setResampleMode` ×2, `setPaddingMode` ×2 |
| `RngDesc_v8` (`Rng.h:50`) / builder (:134) | `getNormalDistMean`, `getNormalDistStdDev`, `getUniformDistMax`, `getUniformDistMin` (**upstream bug**: returns the std-dev, :99-100), `getBernoulliDistProbability`, `getDistribution` | `setRngDistribution` ×2, `setNormalDistParams`, `setNormalDistMean`, `setNormalDistStdDev`, `setUniformDistParams`, `setUniformDistMax`, `setUniformDistMin`, `setBernoulliDistProbability` |
| `Operation_v8` (`Operation.h:105`) / builder (:258) | `getOutputTensor` (deprecated), `getTag`, `getFeatureVector` | §3 (ctors :2979, :3003; `build` :3011) |
| `OperationGraph_v8` (`OperationGraph.h:51`) / builder (:140) | `getEngineCount` (:74), `getOpCount`, `getTag`, `getFeatureVector` (first op's vector, :109-115), `getOps`; `setFeatureVector` | `setHandle`, `setOperationGraph` ×2, `addOperation`, `setIsDynamicShapeEnabled`, `setIsOverrideShapeEnabled`; tag = concatenation of `op.getTag() + '_'` (:165) |
| `EngineHeuristics_v8` (`Heuristics.h:50`) / builder (:150) | `getEngineConfig(count)`, `getEngineConfigCount` | `setOperationGraph` ×2, `setDeviceProperties`, `setHeurMode`, `setSMCount` |
| `get_heuristics_list` (`Heuristics.h`) | — | overloads :306 (`std::array<cudnnBackendHeurMode_t, N>`), :384 (`std::vector<std::string>`, substring match on `heuristics_instant` / `heuristics_mode_a` / `heuristics_fallback` / `heuristics_mode_b`, :396-411), :446 (`std::vector<HeurMode_t>`), :469 (`std::array<std::string, N>`) |
| `Engine_v8` (`Engine.h:49`) / builder (:210) | `getTag`, `getSupportedKnobs`, `getFinalizedKnobs`, `knobs_set`; nested `Knob` with `getMaxValue` (**upstream bug**: returns `minValue`, :95-96) | `setOperationGraph` ×2, `setGlobalEngineIdx` |
| `EngineConfig_v8` (`EngineConfig.h:48`) / builder (:101) | `getTag`; ctor eagerly creates `CUDNN_KNOB_TYPE_COUNTS` knob-choice descriptors (:73-76) | `setEngine` |
| `EngineFallbackList_v8` (`EngineFallbackList.h:78`) / builder (:121) | `getFallbackList` | `setOperationGraph`, `setOperation` ×2; `build` = heuristics in FALLBACK mode |
| `filter`, `hasNumericalNote<>`, `hasBehaviorNote<>` (`Filters.h:31, :43, :71`) | — | `filter` moves a config to `to` when the predicate returns **false** |
| `ExecutionPlan_v8` (`ExecutionPlan.h:52`) / builder (:333) | `getWorkspaceSize`, `getTag`, `setExecutionTime`, `getExecutionTime`, `getAllNumericNotes`, `getNumericNotes`, `getBehaviorNotes`, `getAllBehaviorNotes`, `getJsonRepresentation` (:123) | `setHandle`, `setEngineConfig` ×3, `setKernelCache`, `loadFromJson` (:469). No `execute()`: callers use `cudnnBackendExecute` directly |
| `VariantPack_v8` (`VariantPack.h:49`) / builder (:80) | — | `setDataPointers` ×2, `setUids` ×2, `setWorkspacePointer` |
| Errata (`Errata.h`, JSON builds only) | — | `load_from_config` (reads `CUDNN_ERRATA_JSON_FILE`), `check_shape`, `check_rule` :120, `check_errata`; a rule blocks tag prefix `<Op>_eng<N>_` within a `CUDNN_VERSION` range (:131-135) |
| `ExecutionPlanCache_v1` / `_v2` (`ExecutionPlanCache.h`) | — | keyed by conv feature vectors; `_v2` adds a saturation tracker |
| `EngineConfigGenerator` (`EngineConfigGenerator.h`) | — | `generate_engine_config`, `cudnnGetPlan` ×2, `cudnnFindPlan<Sampling>` ×2, `cudnnFindPlanAndCache<Sampling>`; `CudnnFindSamplingTechnique` |
| `time_sorted_plan<Sampling>` (`find_plan.h`) | — | one warm-up, then 1, 3 or 100 timed runs; HIP events in the shim |

### 2.5 Class → backend descriptor → attributes

Every `build()` does Create(type), one SetAttribute per attribute, Finalize. "Get" marks
`cudnnBackendGetAttribute`.

| Class | Backend descriptor | Attributes (attribute type) |
|---|---|---|
| `Tensor_v8` | `TENSOR` | `DATA_TYPE` (DATA_TYPE), `DIMENSIONS`, `STRIDES` (INT64×n), `UNIQUE_ID`, `BYTE_ALIGNMENT` (INT64), `IS_VIRTUAL`, `IS_BY_VALUE` (BOOLEAN), `VECTOR_COUNT`, `VECTORIZED_DIMENSION` (INT64), `RAGGED_OFFSET_DESC` (BACKEND_DESCRIPTOR), `REORDERING_MODE` (TENSOR_REORDERING_MODE), `CONSTANT_VALUE` (VOID_PTR) |
| `ConvDesc_v8` | `CONVOLUTION` | `COMP_TYPE` (DATA_TYPE), `CONV_MODE` (CONVOLUTION_MODE), `SPATIAL_DIMS` (INT64), `PRE_PADDINGS`, `POST_PADDINGS`, `DILATIONS`, `FILTER_STRIDES` (INT64×n) |
| `PointWiseDesc_v8` | `POINTWISE` | `MODE` (POINTWISE_MODE), `MATH_PREC` (DATA_TYPE), `NAN_PROPAGATION` (NAN_PROPOGATION), ReLU/ELU/softplus/swish parameters (DOUBLE), `AXIS` (INT64); mode-dependent |
| `MatMulDesc_v8` | `MATMUL` | `COMP_TYPE` (DATA_TYPE), `PADDING_VALUE` (DOUBLE) |
| `ReductionDesc_v8` | `REDUCTION` | `COMP_TYPE` (DATA_TYPE), `OPERATOR` (REDUCTION_OPERATOR_TYPE), `IS_DETERMINISTIC` (BOOLEAN) |
| `ResampleDesc_v8` | `RESAMPLE` | `MODE` (RESAMPLE_MODE), `COMP_TYPE` (DATA_TYPE), `NAN_PROPAGATION`, `PADDING_MODE` (PADDING_MODE), `SPATIAL_DIMS` (INT64), `WINDOW_DIMS`, `PRE_PADDINGS`, `POST_PADDINGS`, `STRIDES` (FRACTION×n) |
| `RngDesc_v8` | `RNG` | `DISTRIBUTION` (RNG_DISTRIBUTION), normal / uniform / Bernoulli parameters (DOUBLE) |
| `Operation_v8` | `OPERATION_*`, chosen by mode | §3 |
| `OperationGraph_v8` | `OPERATIONGRAPH` | `OPS` (BACKEND_DESCRIPTOR×n), `HANDLE` (HANDLE), `IS_DYNAMIC_SHAPE_ENABLED`, `IS_OVERRIDE_SHAPE_ENABLED` (BOOLEAN); Get `ENGINE_GLOBAL_COUNT` (INT64) |
| `EngineHeuristics_v8` | `ENGINEHEUR`; results are pre-created `ENGINECFG`s | `OPERATION_GRAPH` (BACKEND_DESCRIPTOR), `MODE` (HEUR_MODE), `DEVICEPROP` (BACKEND_DESCRIPTOR), `SM_COUNT_TARGET` (INT32); Get `RESULTS` (BACKEND_DESCRIPTOR×n; n = 0 queries the count) |
| `Engine_v8` | `ENGINE`, `KNOB_INFO` ×`CUDNN_KNOB_TYPE_COUNTS` | `OPERATION_GRAPH`, `GLOBAL_INDEX` (INT64); Get `KNOB_INFO`; Get `KNOB_INFO_TYPE` (KNOB_TYPE), `MAXIMUM_VALUE`, `MINIMUM_VALUE`, `STRIDE` (INT64) |
| `EngineConfig_v8` | `ENGINECFG`, `KNOB_CHOICE` ×COUNTS | `KNOB_CHOICE_KNOB_TYPE` (KNOB_TYPE), `KNOB_CHOICE_KNOB_VALUE` (INT64), `ENGINECFG_ENGINE`, `ENGINECFG_KNOB_CHOICES` (BACKEND_DESCRIPTOR) |
| `ExecutionPlan_v8` | `EXECUTION_PLAN`; temporaries `ENGINE`, `KNOB_CHOICE` | `ENGINE_CONFIG` (set and get), `KERNEL_CACHE`, `HANDLE` (`loadFromJson` only), `JSON_REPRESENTATION` (CHAR, set and get), Get `WORKSPACE_SIZE`; Get on config `ENGINECFG_ENGINE`, `ENGINECFG_KNOB_CHOICES`; Get on engine `GLOBAL_INDEX`, `NUMERICAL_NOTE`, `BEHAVIOR_NOTE`; Get on knob `KNOB_TYPE`, `KNOB_VALUE` |
| `VariantPack_v8` | `VARIANT_PACK` | `DATA_POINTERS` (VOID_PTR×n), `UNIQUE_IDS` (INT64×n), `WORKSPACE` (VOID_PTR) |
| Filters | temporary `ENGINE` | Get `ENGINECFG_ENGINE`, `ENGINE_NUMERICAL_NOTE` / `ENGINE_BEHAVIOR_NOTE` (`Filters.h:47-90`) |
| Errata | temporary `TENSOR` | Get the conv op's X/W, X/DW or DX/W tensors, then `TENSOR_DIMENSIONS` |
| find/get plan, cache | — | `cudnnBackendExecute`, `cudnnGetStream`, CUDA events |

---

## 3. Operation setters and attributes per op type

Caption for this section: `upstream:include/cudnn_frontend_Operation.h`.

### 3.1 Operation modes

`OperationBuilder_v8` accepts **16** modes (constructor :2979-2998; build dispatch :3072-3107):
convolution forward / backward-filter / backward-data, pointwise, matmul, reduction, gen-stats,
BN-finalize-statistics, BN-backward-weights, resample forward / backward, norm forward / backward, RNG,
reshape, paged-cache-load. Every other `DescriptorType_t` (concat, signal, block-scale
(de)quantize, MoE grouped matmul, transpose, slice) fails validation with
`CUDNN_STATUS_BAD_PARAM` "Unsupported cudnn backend descriptor type" (:3046-3051).

### 3.2 Common and op-specific setters

| Setter | Line | Notes |
|---|---|---|
| `setxDesc` (tensor and `ManagedOpaqueDescriptor` overloads) | :2415, :2409 | |
| `setyDesc` | :2448 | |
| `setwDesc` | :2456 | convolution only |
| `setbDesc`, `settDesc` | :2424, :2436 | pointwise only |
| `setdyDesc`, `setdxDesc`, `setdwDesc` | :2477, :2485, :2494 | |
| `setAlpha`, `setAlpha2`, `setBeta` (float and double) | :2920-2955 | |
| `setComputeType`, `setMathPrecision` | :2963, :2969 | |
| `setcDesc(ConvDesc_v8)` | :2544 | wrong op kind → `BAD_PARAM` |
| `setpwDesc(PointWiseDesc_v8)` | :2849 | wrong op kind → `BAD_PARAM`; classifies the mode as math / identity / act-fwd / act-bwd (:2858+) |
| `setmatmulDesc`, `setreductionDesc` | :2827, :2838 | wrong op kind → `BAD_PARAM` |
| `setResampleDesc`, `setRngDesc` | :2502, :2513 | wrong op kind → `BAD_PARAM` |

About 60 public setters share one builder holding about 44 descriptor slots; each `build_*` maps a
subset to op-specific attribute names (for example `setSavedMeanAndInvVar` feeds BN-finalize, norm
forward, norm backward and BN-backward-weights).

### 3.3 Per-op build table

| Op (tag) | `build_*` | Setters used | Backend attributes | hipDNN lowering (RFC §4.5): `hipdnn_frontend` attribute object + node + packer |
|---|---|---|---|---|
| Conv forward (`ConvFwd`) | :1964 | `setxDesc`, `setwDesc`, `setyDesc`, `setcDesc`, `setAlpha`, `setBeta` | `CONVOLUTION_FORWARD_X`, `_W`, `_Y`, `_CONV_DESC`, `_ALPHA`, `_BETA` (float, or double for a double conv compute type) (:1970-2037) | `ConvFpropAttributes` + conv-fprop node and `ConvolutionFpropPacker`; conv parameters folded into the attributes; alpha ≠ 1 / beta ≠ 0 → `NOT_SUPPORTED` |
| Conv backward-data (`ConvBwdData`) | :589 | `setwDesc`, one of `setdyDesc`/`setyDesc`, one of `setdxDesc`/`setxDesc`, `setcDesc`, alpha, beta | `CONVOLUTION_BWD_DATA_DX`, `_W`, `_DY`, `_CONV_DESC`, `_ALPHA`, `_BETA` (:596-668) | `ConvDgradAttributes` + `ConvolutionDgradPacker`; as above |
| Conv backward-filter (`ConvBwdFilter`) | :970 | `setxDesc`, one of `setdyDesc`/`setyDesc`, one of `setdwDesc`/`setwDesc`, `setcDesc`, alpha, beta | `CONVOLUTION_BWD_FILTER_X`, `_DW`, `_DY`, `_CONV_DESC`, `_ALPHA`, `_BETA` (:976-1048) | `ConvWgradAttributes` + `ConvolutionWgradPacker`; as above |
| Pointwise (JSON name of the mode, :449) | :444 | `setpwDesc`, `setxDesc`; forward: `setyDesc`; backward activation: `setdyDesc` + `setdxDesc`; `setbDesc` (≥ 3 ports), `settDesc` (4 ports); `setAlpha`, `setAlpha2` | `PW_DESCRIPTOR`, `XDESC`, `YDESC` or `DYDESC` + `DXDESC`, `ALPHA1`, `ALPHA2`, `BDESC`, `TDESC` (:455-567) | `PointwiseAttributes` + `PointwisePacker`; slots X/B/T/Y → IN_0/IN_1/IN_2/OUT_0, backward DY/X/DX → IN_0/IN_1/OUT_0; default-valued parameters left unset (the packer elides unset optionals); alpha1/alpha2 ≠ 1 → `NOT_SUPPORTED` |
| Matmul (`Matmul`) | :346 | `setaMatDesc`, `setbMatDesc`, `setcMatDesc`, `setmatmulDesc`, optional `setm/n/kOverrideDesc` | `MATMUL_ADESC`, `BDESC`, `CDESC`, `GEMM_M/N/K_OVERRIDE_DESC`, `MATMUL_DESC` (:350-426) | `MatmulAttributes` + `MatmulPacker` with the comp type; padding value and M/N/K overrides → `NOT_SUPPORTED` |
| Reduction (`Reduction`) | :298 | `setreductionDesc`, `setxDesc`, `setyDesc` | `REDUCTION_DESC`, `XDESC`, `YDESC` (:301-326) | `ReductionAttributes` + `ReductionPacker` (operator, comp type, determinism) |
| GenStats (`GenStats`) | :891 | `setxDesc`, `setSumDesc`, `setSqSumDesc`, `setGenStatsMode`, `setComputeType` | `GENSTATS_XDESC`, `SUMDESC`, `SQSUMDESC`, `MODE`, `MATH_PREC` (:896-951) | `NOT_SUPPORTED` at finalize |
| BN finalize (`BNFinalize`) | :688 | `setBNFinalizeMode`, `setComputeType`, `setSumDesc`, `setSqSumDesc`, `setScaleAndBias`, `setEqScaleAndBias`, `setPrev/NextRunningMeanAndVar`, `setSavedMeanAndInvVar`, `setEpsilonTensor`, `setExpDecayFactorTensor`, `setAccumCountTensor` | `BN_FINALIZE_*` (:705-876); optional attributes set only when non-null | `NOT_SUPPORTED` at finalize |
| BN backward weights (`Dgrad_Drelu_BN_Bwd`) | :1831 | `setComputeType`, `setxDesc`, `setSavedMeanAndInvVar`, `setScale`, `setdyDesc`, `setDScaleAndDBias`, `setEqScalesAndBias` | `BN_BWD_WEIGHTS_*` (:1836-1951) | `NOT_SUPPORTED` at finalize |
| Norm forward (`Norm_Fwd`) | :1068 | `setNormalizationMode`, `setNormFwdPhase`, `setxDesc`, `setScaleAndBias`, `setSavedMeanAndInvVar`, `setEpsilonTensor`, `setExpDecayFactorTensor`, `setPrev/NextRunningMeanAndVar`, `setyDesc`, `addPeerStatTensor` / `setPeerStatTensor` | `NORM_FWD_*` incl. `PEER_STAT_DESCS` (:1094-1238) | BATCH → `BatchnormAttributes` (training) / `BatchnormInferenceAttributes` (mean + inverse variance); LAYER → `LayernormAttributes` (the hipDNN node infers the normalized-dimension count); RMS → `RMSNormAttributes`; `EXP_AVG_FACTOR_DESC` → `MOMENTUM`; compute type synthesized as FLOAT; other modes and peer stats → `NOT_SUPPORTED` |
| Norm backward (`Norm_Bwd`) | :1255 | `setNormalizationMode`, `setxDesc`, `setSavedMeanAndInvVar`, `setdyDesc`, `setScale`, `setdxDesc`, `setDScaleAndDBias`, `addPeerStatTensor`, `setEpsilonTensor` | `NORM_BWD_*` (:1273-1365) | BATCH / LAYER / RMS → `BatchnormBackwardAttributes` / `LayernormBackwardAttributes` / `RMSNormBackwardAttributes`; compute type FLOAT; other modes and peer stats → `NOT_SUPPORTED` |
| Resample forward (`Resample_fwd`) | :1381 | `setxDesc`, `setyDesc`, `setResampleDesc`, `setAlpha`/`setBeta` (sent as double), `setidxDesc` | `RESAMPLE_FWD_XDESC`, `YDESC`, `ALPHA`, `BETA`, `DESC`, `IDXDESC` (:1385-1451) | `ResampleFwdAttributes` + `ResampleFwdPacker`; fractions → integers; NEAREST/BILINEAR → `NOT_SUPPORTED`; alpha ≠ 1 / beta ≠ 0 → `NOT_SUPPORTED` |
| Resample backward (`Resample_bwd`) | :1470 | `setdxDesc`, `setdyDesc`, `setResampleDesc`, alpha, beta, `setidxDesc`, optional `setxDesc`/`setyDesc` | `RESAMPLE_BWD_DXDESC`, `XDESC`, `YDESC`, `DYDESC`, `ALPHA`, `BETA`, `DESC`, `IDXDESC` (:1477-1578) | `ResampleBwdAttributes` + `ResampleBwdPacker`; as forward; `XDESC` / `YDESC` set → `NOT_SUPPORTED` |
| RNG (`Rng`) | :1605 | `setyDesc`, `setRngDesc`, `setSeed` or `setSeedDesc`, `setOffsetDesc` | `RNG_YDESC`, `RNG_SEED`, `RNG_DESC`, `RNG_OFFSET_DESC` (:1612-1675) | `NOT_SUPPORTED` at finalize |
| Reshape (`Reshape`) | :1764 | `setxDesc`, `setyDesc`, `setReshapeMode` | `RESHAPE_XDESC`, `YDESC`, `RESHAPE_MODE` (:1771-1805) | `NOT_SUPPORTED` at finalize |
| Paged cache load (no tag set) | :1700 | `setcontainerDesc`, `setpageTableDesc`, `setsequenceDesc`, `setyDesc` | `PAGED_CACHE_LOAD_CONTAINER_DESC`, `PAGE_TABLE_DESC`, `SEQUENCE_DESC`, `YDESC` (:1731-1752) | `NOT_SUPPORTED` at finalize |

### 3.4 Feature vector

`extract_feature_vector` (:2057-2120) runs only for the three convolution builds (:683, :1063, :2052).
It records the op type; x/w/y dimensions; padding, dilation and stride padded to 3-D; x/w/y/compute
types and the conv mode; x/w/y strides; and alpha and beta as bit patterns.
`OperationGraph_v8::getFeatureVector()` returns the first op's vector, which keys
`ExecutionPlanCache`. The shim keeps the algorithm; the enum-derived entries take the shim's own
values, which only matters within one process.

---

## 4. Consumer surfaces

Sources, each pinned to one commit (line numbers below refer to these exact files):

| Consumer | Commit | Files |
|---|---|---|
| PyTorch v2.14.1 | `5c4886908584029761b579af026dcfb627c84070` | <https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cudnn/Conv_v8.cpp>, <https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/ConvUtils.h>, <https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cudnn/MHA.cpp>, <https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/CMakeLists.txt>, <https://github.com/pytorch/pytorch/tree/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/quantized/cudnn> |
| OpenXLA | `df27398e46b6e6116094e51c190cb7c2355002a8` | <https://github.com/openxla/xla/blob/df27398e46b6e6116094e51c190cb7c2355002a8/xla/stream_executor/cuda/cuda_dnn.cc> |
| PaddlePaddle | `81b1dc6e10dac0b7d0daf37d8dceb7167779a827` | <https://github.com/PaddlePaddle/Paddle/blob/81b1dc6e10dac0b7d0daf37d8dceb7167779a827/paddle/phi/kernels/gpudnn/conv_cudnn_frontend.h> |
| ONNX Runtime | `648f708a73450c2f484f56c83ef9b6621e3c2b6b` | <https://github.com/microsoft/onnxruntime/blob/648f708a73450c2f484f56c83ef9b6621e3c2b6b/onnxruntime/core/providers/cuda/nn/conv.cc> |
| TransformerEngine | `33f3b3561943fcd9dd2f544f67dc2dfd62efdc9e` | <https://github.com/NVIDIA/TransformerEngine/blob/33f3b3561943fcd9dd2f544f67dc2dfd62efdc9e/transformer_engine/common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu> |

### 4.1 PyTorch: status

| Question | Answer | Evidence (under `aten/src/ATen/`) |
|---|---|---|
| Does convolution still use v0.x builders? | Yes: forward, backward-data, backward-filter and conv+add+bias+ReLU | `native/cudnn/Conv_v8.cpp` L12, L22-23 (includes), L445-576 (graph builders) |
| Compiled by default? | Yes, whenever `AT_CUDNN_ENABLED()` | `CMakeLists.txt` L162 (`native/cudnn/*.cpp`), L167 (`native/quantized/cudnn/*.cpp`) |
| Runtime default? | v8 path on; `TORCH_CUDNN_V8_API_DISABLED=1` falls back to `Conv_v7.cpp` | `native/ConvUtils.h` L96-105 (flag read at L97) |
| Other knobs | `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT`; `TORCH_CUDNN_V8_API_DEBUG`; `TORCH_CUDNN_USE_HEURISTIC_MODE_B` | `native/cudnn/Conv_v8.cpp` L263; `native/ConvUtils.h` L98; `native/ConvUtils.h` L91 |
| Other `cudnn_frontend.h` users | `native/cudnn/MHA.cpp` uses the v9 graph API (RFC 0012 scope) | `native/cudnn/MHA.cpp` L133 (include), L156 (`namespace fe = cudnn_frontend`) |
| Quantized ops | `quantized/cudnn/{Conv,Linear,BinaryOps}.cpp` and `utils.h` use v0.x builders; `Pooling.cpp` is legacy v7 API only | §4.3 |

Under ROCm, PyTorch's own build uses MIOpen sources, so the shim targets hipified builds of these CUDA
files (RFC 0021 Phase 10).

### 4.2 `Conv_v8.cpp` surface

| Symbol | Lines | Notes |
|---|---|---|
| `TensorBuilder` `.setDim .setStrides .setId .setAlignment .setDataType .setVirtual .build` | 63-108 | UIDs are chars (`'x'`, `'y'`, `'w'`, ...). Alignment is the data pointer's power-of-two alignment capped at 32 (`getAlignment`, L51-61): 32 for freshly allocated tensors, possibly less than 16 for offset views, which then need plugin API ≥ 1.3.0 (§1.3) |
| `ConvDescBuilder` `.setDataType .setMathMode .setNDims .setStrides .setPrePadding .setPostPadding .setDilation .build` | 119-138 | float compute for fp16/bf16 (L126-128) |
| `PointWiseDescBuilder` `.setMode(ADD, RELU_FWD) .setMathPrecision` | 492-503 | |
| `OperationBuilder(cudnnBackendDescriptorType_t)` `.setxDesc .setyDesc .setwDesc .setbDesc .setcDesc .setpwDesc .setAlpha .setAlpha2 .build`; `getOutputTensor` | 455-468, 508-568 | modes conv fwd / bwd-data / bwd-filter, pointwise; fused path is 4 ops (L569-570); bias viewed as `{1, C, 1, 1[, 1]}` (L1615-1616) |
| `OperationGraphBuilder` `.setHandle .setOperationGraph(n, ops) .build`; `getTag` | 469-474, 571-574 | tag feeds errata |
| `EngineHeuristicsBuilder` `.setOperationGraph .setHeurMode .build`; `getEngineConfigCount`, `getEngineConfig` | 597-605 | modes INSTANT, B |
| `EngineFallbackListBuilder` `.setOperationGraph .setOperation .build`; `getFallbackList` | 619-623 | |
| `filter`, `hasNumericalNote<NONDETERMINISTIC / DOWN_CONVERT_INPUTS / TENSOR_CORE>` | 140-168 | |
| `GeneratorSource`, `EngineConfigGenerator`, `generate_engine_config`, `cudnnGetPlan` | 634-641, 762-1002 | benchmark mode; `cudnnGetPlan` at L774 |
| `time_sorted_plan<CUDNN_FIND_SAMPLE_ONCE>` | 857-859, 913-915 | |
| `ExecutionPlanBuilder` `.setHandle .setEngineConfig(cfg, tag) .build`; `ExecutionPlan` stored by value in an LRU (L288-340); `getWorkspaceSize`, `get_raw_desc`, `getTag`, `getJsonRepresentation` | 1067-1104 | JSON parsed by the errata lambda (L742) |
| `VariantPackBuilder` `.setWorkspacePointer .setDataPointers .setUids .build` | 410-442 | built on every call |
| `load_from_config`, `check_errata` | 661-760 | §4.5 |
| `cudnnBackendExecute(handle, plan.get_raw_desc(), vp.get_raw_desc())` | 416, 441 | the only direct C call |
| `cudnnException` | 1019-1111 | |

### 4.3 Quantized cuDNN surface (`aten/src/ATen/native/quantized/cudnn/`)

| File | v0.x usage |
|---|---|
| `utils.h` | `TensorBuilder` (L224-264), `PointWiseDescBuilder` MUL / ADD / RELU_FWD (L268-291), `filterEngineConfigs` with `hasNumericalNote` + `filter` (L294-310) |
| `Conv.cpp` | `ConvDescBuilder` (L28-39), int32 compute (L101); int8 conv → fp32 virtual → MUL(bias) → ADD → [ReLU] → MUL(requant) → int8 (L143-222); `getOutputTensor` as `shared_ptr<OpaqueBackendPointer>` (L187); `std::optional<Operation>::emplace` (L152-163); heuristics INSTANT + fallback (L225-234); iterates the **unfiltered** config list (L238-241); `ExecutionPlan` cache (L52); execute (L131) |
| `Linear.cpp` | `MatMulDescBuilder().setMathPrecision(...)` (L29-33), int32 compute (L66); matmul + the same requant chain (L177-261); heuristics + fallback (L264-273); unfiltered loop (L277-280) |
| `BinaryOps.cpp` | MUL → ADD → [ReLU] → MUL, all non-virtual (L170-217); fallback on pointwise (L224-226); caches `ManagedOpaqueDescriptor` (`plan.get_desc()`, L65, L241) and executes via `get_backend_descriptor()` (L159); workspace forced to 0 (L138) |
| `Pooling.cpp` | legacy v7 C API only (out of scope) |

### 4.4 De-duplicated minimum surface (PyTorch)

| Area | Required |
|---|---|
| Headers | `cudnn_frontend.h`, `cudnn_frontend_find_plan.h`, `cudnn_frontend_get_plan.h`; `nlohmann::json` reachable |
| Descriptor builders | `TensorBuilder` (`setDim`, `setStrides`, `setId`, `setAlignment`, `setDataType`, `setVirtual`); `ConvDescBuilder` (`setDataType`, `setMathMode`, `setNDims`, `setStrides`, `setPrePadding`, `setPostPadding`, `setDilation`); `PointWiseDescBuilder` (`setMode`, `setMathPrecision`; ADD, MUL, RELU_FWD); `MatMulDescBuilder` (`setMathPrecision`) |
| Operations | `OperationBuilder(cudnnBackendDescriptorType_t)` with `setxDesc` (tensor or `shared_ptr<OpaqueBackendPointer>`), `setyDesc`, `setwDesc`, `setbDesc`, `setcDesc`, `setpwDesc`, `setaMatDesc`, `setbMatDesc`, `setcMatDesc`, `setmatmulDesc`, `setAlpha(float)`, `setAlpha2(float)`; `getOutputTensor`, `getTag`; move-constructible. Kinds: conv fwd / bwd-data / bwd-filter, pointwise, matmul |
| Graph and engines | `OperationGraphBuilder` (`setHandle`, `setOperationGraph(n, ops)`); `EngineHeuristicsBuilder`; `EngineFallbackListBuilder`; `EngineConfigList`, `ManagedOpaqueDescriptor`, `OpaqueBackendPointer` |
| Plans | `ExecutionPlanBuilder` (`setHandle`, `setEngineConfig(desc, tag)`); `ExecutionPlan` copyable, `getWorkspaceSize`, `get_raw_desc`, `get_desc`, `getTag`, `getJsonRepresentation`; `VariantPackBuilder` |
| Utilities | `filter`, `hasNumericalNote<>` (3 notes), `GeneratorSource`, `EngineConfigGenerator`, `generate_engine_config`, `cudnnGetPlan`, `executionPlans_t`, `Predicate`, `time_sorted_plan<CUDNN_FIND_SAMPLE_ONCE>`, `load_from_config`, `check_errata`, `cudnnException` |
| C surface | `cudnnBackendExecute`; `cudnnBackendDescriptor_t`, `cudnnBackendDescriptorType_t`, `cudnnBackendHeurMode_t` (INSTANT, B), `cudnnBackendNumericalNote_t`, `cudnnPointwiseMode_t` (ADD, MUL, RELU_FWD), `cudnnDataType_t` (FLOAT, HALF, BFLOAT16, INT8, INT32, DOUBLE), `CUDNN_CROSS_CORRELATION`, `CUDNN_MAJOR` / `CUDNN_MINOR` / `CUDNN_VERSION`, `cudnnStatus_t` |
| Not needed | `ExecutionPlanCache`, `get_heuristics_list`, `cudnnFindPlan`, `EngineBuilder` / `EngineConfigBuilder` / knobs, reduction / resample / norm, plan JSON load |

### 4.5 PyTorch's hard-coded errata

`plan_errata_exception` (`Conv_v8.cpp` L661-760) blocks plans by tag prefix `<Op>_eng<N>_` and
`CUDNN_VERSION` range (upstream `cudnn_frontend_Errata.h:120-135`; a `-1` end is open-ended). Rules
name **cuDNN** engine numbers:

| Rule | Op, engine | `CUDNN_VERSION` range | Extra condition | Live at the shim's 92200? |
|---|---|---|---|---|
| `rule_id` 188288 | ConvFwd eng5 | [92301, 92400) | — | no |
| `rule_id` `sm120_*` | ConvFwd eng58; ConvBwdData eng63; ConvBwdFilter eng63 | [92300, ∞) | `CUBLAS_VERSION < 130601` and plan JSON `engine.smVersion == 1200` | no |
| `rule_id` 163539 (3-D) | ConvFwd eng23 | [90800, 91500) | no `CUDNN_ERRATA_JSON_FILE` and `x.dim() > 4` | no |
| `rule_id` `ConvBwdData` (3-D) | ConvBwdData eng23 | [8000, ∞) | no `CUDNN_ERRATA_JSON_FILE` and `x.dim() > 4` | **yes** |

Only the 3-D `ConvBwdData eng23` rule is live. Shim indices are dense per graph over the enumerated
engines, sorted by signed hipDNN engine ID (RFC §4.6), so `eng23` names a hipDNN engine only when a
graph has ≥ 24 enumerated engines (none does today), and then an arbitrary one — a documented
divergence. The sm_120 lambda reads `plan_json["engine"]` as an object (L742), which is why the shim's
plan JSON always carries an `"engine"` object (RFC §4.7).

### 4.6 Other consumers

| Consumer | API generation | v0.x surface beyond PyTorch (line numbers at the pinned commit) |
|---|---|---|
| OpenXLA / TensorFlow (`cuda_dnn.cc`; TF vendors it under `third_party/xla/`) | Mixed: conv, fused conv, matmul fusion, norm on v0.x; attention, block-scaled matmul, serialized graphs on v9 (`graph.deserialize`, L6876) | `TensorBuilder.cloneFrom` (L2870); `ConvDescBuilder.setComputeType` / `setSpatialDimCount` (L2946-2949, L3293-3296, L3615-3618); pointwise MIN / MAX / ELU / RELU / DIV (L3021-3031), RELU6 via `setReluUpperClip` (L3700), leaky ReLU via `setReluLowerClipSlope` (L3715), GELU / TANH / SIGMOID (L3860-3866); reduction (L3435); `NORM_FORWARD` / `NORM_BACKWARD` with `setNormalizationMode` / `setNormFwdPhase` (L5832-5858); `get_heuristics_list<N>` with string modes (L3917-3921, L5396-5410); `EngineBuilder.setGlobalEngineIdx` (L3965); `Engine::getSupportedKnobs` / `setChoice` keyed by the raw knob-type value (L3981-3984); `EngineConfigBuilder` (L3989); raw `cudnnBackendCreateDescriptor` / `GetAttribute` / `DestroyDescriptor` on `EXECUTION_PLAN_ENGINE_CONFIG`, `ENGINECFG_ENGINE`, `ENGINE_GLOBAL_INDEX`, `ENGINECFG_KNOB_CHOICES`, `KNOB_CHOICE_KNOB_TYPE/VALUE` to persist autotune results (L5081-5199) |
| PaddlePaddle (`conv_cudnn_frontend.h`) | v0.x | conv + pointwise via `TensorBuilder` / `OperationBuilder` / `OperationGraphBuilder` (L101, L172, L182, L492-496); `hasNumericalNote` (L40, L199-211); `get_heuristics_list<2>` (L222); `EngineConfigGenerator` (L235); `cudnnFindPlan<CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>` (L262-263); `cudnnGetPlan` (L272); `getTag` / `getExecutionTime` (L277-278); own plan cache keyed by the feature vector, with stability tracked per plan tag (L384-403) |
| ONNX Runtime CUDA EP (`nn/conv.cc`) | v9 graph API only (L117-245, L416-430) | none (RFC 0012 scope) |
| TransformerEngine (fused attention) | v9 graph API only (L9, L26-29, L93, L593) | none (RFC 0012 scope) |

XLA persists `(GLOBAL_INDEX, knob choices)` keyed by the raw knob-type value (L5199) and replays it
(L3982); Paddle tracks plan stability by tag. This is why the shim's engine index is stable across
hipDNN re-rankings and its enum values are append-only (RFC §4.6, §4.8). The index is not stable across
changes to the set of enumerated engines (provider added, removed or upgraded, different GPU).

---

## 5. Legacy sample catalogue

### 5.1 Build and framework

- Catch2 v3 (`upstream:samples/legacy_samples/test_list.cpp:24`); one executable `legacy_samples`
  built from seven TUs: `test_list.cpp`, `conv_sample.cpp`, `fusion_sample.cpp`, `fp8_sample.cpp`,
  `norm_samples.cpp`, `helpers.cpp`, `fp16_emu.cpp`. `fp16_dev.cu` is present but not built and has
  no callers.
- **41** `TEST_CASE`s, all in `test_list.cpp` (verified: `rg -c "^\s*TEST_CASE\("` = 41).
- Every sample executes with `cudnnBackendExecute(handle, plan.get_raw_desc(), vp.get_raw_desc())`
  directly.

CUDA-only dependencies (all under `upstream:samples/legacy_samples/`):

| Dependency | Where | Harness treatment (RFC 0021 Phase 9) |
|---|---|---|
| `cuda_runtime.h` | `conv_sample.h:30`, `fusion_sample.h:30`, `fp8_sample.h:30`, `utils/helpers.h:29` | existing overlay |
| `driver_types.h`, `cuda_fp16.h`, `cuda_runtime_api.h` | `fp16_emu.h:26-30` | legacy-only overlay adds `driver_types.h` |
| `cudaMalloc` / `cudaMemcpy` / `cudaFree` / `cudaDeviceSynchronize` | `utils/helpers.h:244-306`, `test_list.cpp` | overlay |
| `cudaGetDeviceProperties`, `prop.major` arch gating | `helpers.cpp:28`; catch blocks (`fusion_sample.cpp:331`, `fp8_sample.cpp:169`, `norm_samples.cpp:249`) | struct-compatible `cudaDeviceProp` reporting 8.0 |
| `cudaDeviceReset` (via `FatalError`) | `utils/error_util.h` | overlay |
| `CudnnHandleDeleter`, `create_cudnn_handle` | `utils/helpers.h:70-86` | substituted helpers (ODR clash with the v9 shim's `cudnn.h`) |
| `throw_if` in a reopened `namespace cudnn_frontend` | `utils/error_util.h:54,62` | substituted |
| CPU references | `cpu_references.h` | kept |

### 5.2 Test cases

Caption: `upstream:samples/legacy_samples/test_list.cpp`. Driver functions are in `conv_sample.cpp`
(C), `fusion_sample.cpp` (F), `fp8_sample.cpp` (P) or `norm_samples.cpp` (N).

Expected-outcome tiers (RFC 0021 Phase 9): **RUN_CLEAN** must pass; **RUN** must not crash, outcome
reported; **XFAIL_RUN** must still fail on its anchor. Seeds below are adjusted after the first real
run.

| # | Test case (line) | Gate | Pattern | Data type | Driver and key APIs | Expected shim outcome |
|---:|---|---|---|---|---|---|
| 1 | "Tensor creation comparison" :33 | — | raw backend tensor vs `TensorBuilder` | FLOAT | raw `cudnnBackendCreateDescriptor` / `SetAttribute` / `Finalize` / `DestroyDescriptor` (:68-86), `TensorBuilder` | RUN_CLEAN: no engine involved |
| 2 | "Use global(index) for execution" :92 | — | conv backward-filter | FLOAT + CPU ref | C `run_from_global_index`: `EngineBuilder.setGlobalEngineIdx(0)`, knobs, `EngineConfigBuilder`, plan, variant pack | XFAIL_RUN: tensors use alignment 4 (`conv_sample.cpp:179-227`); no provider reports plugin API ≥ 1.3.0 |
| 3 | "Use heuristics for execution" :165 | — | conv forward | FLOAT + CPU ref | C `run_from_heuristics`: static `ExecutionPlanCache`, `EngineConfigGenerator`, `cudnnFindPlanAndCache<MEDIAN_OF_THREE>` | XFAIL_RUN: alignment 4 |
| 4 | "Use DNN based heuristics for execution" :241 | — | conv forward | FLOAT | C `run_from_heuristics` (mode argument ignored) | XFAIL_RUN: alignment 4 |
| 5 | "Use fallback for execution" :317 | — | conv backward-data | FLOAT + CPU ref | C `run_with_external_config`: `get_heuristics_list<2>({"heuristics_instant","heuristics_fallback"})`, plan from a raw `ManagedOpaqueDescriptor` | XFAIL_RUN: alignment 4 |
| 6 | "ConvBiasAct sample" :393 | — | conv → add(z) → add(bias) → ReLU | FLOAT | C `run_conv_add_bias_activation`: 4 `OperationBuilder`s, `getEngineCount`, `get_heuristics_list<2>` | XFAIL_RUN: alignment 4 (`conv_sample.cpp:90-178`); the 4-op fusion also has no engine |
| 7 | "Use cudnnFindPlan for execution" :447 | — | conv forward | FLOAT + CPU ref | C `run_from_cudnn_find`: `cudnnFindPlan<MEDIAN_OF_THREE>` with predicate | XFAIL_RUN: alignment 4 |
| 8 | "ConvBiasAct sample with cudnnFindPlan" :522 | — | conv → add → add → ReLU | HALF | C `run_conv_add_bias_activation_with_cudnn_find` | XFAIL_RUN: alignment 4; 4-op fusion |
| 9 | "Use cudnnGetPlan for execution" :576 | — | conv forward | FLOAT + CPU ref | C `run_from_cudnn_get`: `cudnnGetPlan`, max-workspace scan | XFAIL_RUN: alignment 4 |
| 10 | "ConvScaleBiasAddAct sample" :651 | — | conv → mul → add → add → leaky ReLU | HALF | F `run_conv_scale_bias_add_leaky_relu` | RUN |
| 11 | "ConvScaleBiasAddAct sample_float" :711 | — | same | FLOAT | same | RUN |
| 12 | "ConvBiasScaleAct sample" :771 | — | conv → add → mul → ReLU | HALF | F `run_conv_bias_scale_relu` | RUN |
| 13 | "ConvBiasScaleActSerialization sample" :826 | — | same + plan serialization | HALF | F `run_serialization_conv_bias_scale_relu`: `getJsonRepresentation`, `loadFromJson` | RUN: the JSON carries `payload` only when the engine's plan serialization succeeds; without it `loadFromJson` is `NOT_SUPPORTED` |
| 14 | "ConvScaleBiasActGenIndexSelection sample" :881 | — | conv → mul → add → ReLU → GEN_INDEX → CMP_LT / CMP_GT → LOGICAL_OR → BINARY_SELECT | HALF, INT32, BOOLEAN | F `run_conv_scale_bias_relu_gen_index_selection`: `setAxis`, `settDesc` | RUN |
| 15 | "ConvScaleBiasAct_int8 sample" :954 | — | conv (int32 compute) → mul → add → ReLU | INT8 I/O | F `run_conv_scale_bias_relu_int8` | XFAIL_RUN: no int8 conv engine (provider gap) |
| 16 | "PoolScaleBiasAct_int8 sample" :1009 | — | resample fwd (avgpool) → mul → add → ReLU | INT8 I/O, FLOAT compute | F `run_pool_scale_bias_relu_int8`: `ResampleDescBuilder` | XFAIL_RUN: int8 (provider gap) |
| 17 | "MatmulBiasAct sample" :1086 | — | matmul → add(bias) → GELU_APPROX_TANH_FWD | HALF | F `run_matmul_bias_gelu` | RUN |
| 18 | "MatmulBiasAct sample_float" :1130 | — | same | FLOAT | same | RUN |
| 19 | "MatmulDGeluDBias sample" :1174 | — | matmul → GELU_APPROX_TANH_BWD → reduction ADD | HALF | F `run_matmul_dgelu_dbias` | RUN |
| 20 | "ConvDrelu sample" :1219 | — | conv → RELU_BWD | HALF | F `run_conv_drelu` | XFAIL_RUN: alignment 4 (`fusion_sample.cpp:1898-1931`) |
| 21 | "DgradDrelu sample" :1285 | — | conv backward-data → RELU_BWD | HALF | F `run_dgrad_drelu` | XFAIL_RUN: alignment 4 (`fusion_sample.cpp:2062-2095`) |
| 22 | "ConvColReduction sample" :1352 | — | conv → reduction ADD | HALF | F `run_conv_reduction` | RUN |
| 23 | "Use errata to block global(index) for execution" :1405 | — | conv backward-filter, engine index 20 | FLOAT | C `block_using_errata`: `setGlobalEngineIdx(20)`, `check_errata` ×3 (tag, tag + shape, wildcard) | XFAIL_RUN: NVIDIA engine 20 has no shim counterpart (dense index, N ≪ 20); alignment 4 |
| 24 | "DP4A execution with cudnnFindPlan" :1456 | — | conv forward, INT8x4 vectorized | INT8, INT32 compute | C `run_dp4a`: `setVectorCountAndDimension` | XFAIL_RUN: `VECTOR_COUNT` ≠ 1 and `INT8x4` → `NOT_SUPPORTED` |
| 25 | "IMMA execution with manual autotuning" :1514 | — | conv forward, INT8x32 | INT8, INT32 compute | C `run_imma`: `setReorderType(INT8x32)`, `filter` ×2, `cudnnReorderFilterAndBiasInt8x32`, `check_errata` | XFAIL_RUN: reordering and reorder helper → `NOT_SUPPORTED` |
| 26 | "Use Plan cache for rerunning the same convolution" :1572 | — | conv forward, twice | FLOAT | C `run_from_heuristics` ×2; second run hits `ExecutionPlanCache` | XFAIL_RUN: alignment 4 (see note) |
| 27 | "Scale Bias Conv BNGenstats" :1669 | — | mul → add → leaky ReLU → conv → GEN_STATS | HALF, FLOAT stats | F `run_bn_conv_gen_stat` | XFAIL_RUN: GEN_STATS → `NOT_SUPPORTED` |
| 28 | "Dual Scale Bias Act Relu" :1714 | — | (mul → add) ×2 → add → ReLU | HALF | F `run_dsbar` | RUN |
| 29 | "Dual Scale Bias Act Relu with CPU Reference" :1744 | — | same | FLOAT | same | RUN |
| 30 | "Scale Bias Conv BNGenstats with CPU Reference" :1818 | — | as #27 | HALF, FLOAT stats | same; arch-gated (`is_ampere_arch` / `is_hopper_arch`) | XFAIL_RUN: GEN_STATS |
| 31 | "BN Finalize" :1953 | — | BN_FINALIZE_STATISTICS (training) | FLOAT stats, DOUBLE, INT64 by-value scalars | F `run_bn_finalize` | XFAIL_RUN: BN finalize → `NOT_SUPPORTED` |
| 32 | "Tensor cloning" :2027 | — | `cloneFrom` + data type / virtual changes | FLOAT → HALF | `TensorBuilder().cloneFrom(...)` (2 sections) | RUN_CLEAN: no engine involved |
| 33 | "Max pooling idx tensor dump" :2154 | ≥ 8600 | resample fwd maxpool with index tensor | HALF, INT8 index | F `run_maxpool_with_idx` | RUN |
| 34 | "Backward pooling" :2234 | ≥ 8600 | resample bwd avgpool and maxpool (2 sections) | HALF, INT8 index | F `run_backward_avgpool`, `run_backward_maxpool` | RUN |
| 35 | "Conv two global scales" :2351 | ≥ 8300 | conv → mul → mul | HALF, FLOAT compute | F `run_conv_two_global_scales` | RUN |
| 36 | "Conv Scale" :2425 | ≥ 8600 | conv → mul(scale) | FP8_E4M3 | P `run_fp8_conv_scale`: deprecated `ConvDescBuilder` setters, `setAlpha`/`setBeta` | XFAIL_RUN: FP8 |
| 37 | "Conv Descale Descale Amax Scale sample" :2482 | ≥ 8600 | conv → mul ×3 + reduction AMAX | FP8_E4M3 | P `run_fp8_conv_descale_descale_amax_scale` | XFAIL_RUN: FP8 |
| 38 | "Scale transpose convert amax sample" :2549 | ≥ 8600 | mul → IDENTITY (transpose/convert) + AMAX | FP16 → FP8_E4M3 | P `run_tranpose_scale_convert_fp16_fp8_amax` | XFAIL_RUN: FP8 |
| 39 | "Dgrad Descale Descale Amax Scale sample" :2596 | ≥ 8700 | conv backward-data → mul ×3 + AMAX | FP8_E4M3 | P `run_fp8_dgrad_descale_descale_amax_scale` | XFAIL_RUN: FP8 |
| 40 | "BN BWD Weights sample" :2669 | ≥ 90800 | pointwise + RELU_BWD → conv backward-data → BN_BWD_WEIGHTS | HALF, FLOAT stats | F `run_bn_bwd_weight`: `setEqScalesAndBias` | XFAIL_RUN: BN backward weights → `NOT_SUPPORTED` |
| 41 | "Batch normalization" :2744 | ≥ 8800 | NORM_FORWARD / NORM_BACKWARD batch norm with 2 peer-stat tensors (2 sections) | HALF, FLOAT stats | N `run_batch_norm_forward`, `execute_batch_norm_forward`, `run_batch_norm_backward`; caches copyable plans by feature vector | XFAIL_RUN: peer statistics → `NOT_SUPPORTED` |

Seed tally: 2 RUN_CLEAN, 14 RUN, 25 XFAIL_RUN.

Note on #23 and #26: plan-cache and errata cases would be natural RUN_CLEAN candidates, but both build
their graph with `create_common_descriptors` (alignment 4), and #23 also needs NVIDIA engine 20; they
are seeded XFAIL_RUN. A RUN_CLEAN errata or plan-cache check would need a 16-byte-aligned
variant, which the upstream corpus does not have.

Filter predicates the samples use: `isNonDeterministic`, `isReducedPrecisionReduction`,
`isDownConvertingInputs` (`hasNumericalNote<>`, `conv_sample.cpp:30-47`); `allowAll`
(`conv_sample.cpp:50`); `isRuntimeCompilation` (`hasBehaviorNote<>`, `fusion_sample.cpp:35-37`).

### 5.3 Coverage

| Area | Exercised |
|---|---|
| Ops | conv fwd / bwd-data / bwd-filter; pointwise (ADD, MUL, SUB, RELU_FWD/BWD, GELU / GELU_APPROX_TANH FWD/BWD, GEN_INDEX, CMP_LT/GT, LOGICAL_OR, BINARY_SELECT, IDENTITY); matmul; reduction (ADD, AMAX); GEN_STATS; BN_FINALIZE; BN_BWD_WEIGHTS; NORM_FWD/BWD (batch); RESAMPLE_FWD/BWD |
| Not exercised | RNG, RESHAPE, PAGED_CACHE_LOAD |
| Infrastructure | `EngineBuilder` and knobs, `EngineConfigBuilder`, heuristics, fallback list, `get_heuristics_list`, `filter`, notes, `EngineConfigGenerator` with get/find plan, `ExecutionPlanCache`, errata, plan JSON round trip, INT8x32 reorder, `cloneFrom`, by-value tensors, peer stats |
| Data types | FLOAT, HALF, INT8 (incl. x4 / x32), INT32, INT64, DOUBLE, BOOLEAN, FP8_E4M3 |

---

## 6. Version gates in the v0.x headers

The shim claims `CUDNN_VERSION` **92200**
(`projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/cudnn_runtime_version.h:27-30`), so
every `>=` gate at or below 92200 compiles in. The few `<` gates and the gates above 92200 compile
out; the right-hand column states the effect.

Caption directory: `upstream:include/`.

| Gate | Controls | Location | At 92200 |
|---|---|---|---|
| ≥ 8300 && < 8600 | heuristics mode B → INSTANT retry | `cudnn_frontend_Heuristics.h:420,427` | compiled out |
| ≥ 8600 | resample backward op | `cudnn_frontend_Operation.h:1471` | in |
| ≥ 8700 | matmul M/N/K override | `cudnn_frontend_Operation.h:379` | in |
| ≥ 8700 | resample backward X/Y | `cudnn_frontend_Operation.h:1488` | in |
| ≥ 8700 | RNG op | `cudnn_frontend_Operation.h:1606` | in |
| ≥ 8700 | reshape op | `cudnn_frontend_Operation.h:1765` | in |
| ≥ 8700 | `RngDesc` | `cudnn_frontend_Rng.h:57,148,215` | in |
| ≥ 8800 | RNG seed tensor, offset descriptor | `cudnn_frontend_Operation.h:1622,1665` | in |
| ≥ 8900 | tensor ragged offset | `cudnn_frontend_Tensor.h:479` | in |
| ≥ 8900 | matmul padding value | `cudnn_frontend_MatMulDesc.h:152` | in |
| ≥ 8905 | heuristics SM count | `cudnn_frontend_Heuristics.h:252` | in |
| ≥ 90400 | operation-graph dynamic shape | `cudnn_frontend_OperationGraph.h:292` | in; `true` → `NOT_SUPPORTED` at shim finalize |
| ≥ 90400 | plan kernel cache | `cudnn_frontend_ExecutionPlan.h:411` | in |
| < 90500 | paged-cache-load → `NOT_SUPPORTED` in the builder | `cudnn_frontend_Operation.h:1701` | compiled out: the builder builds the op; the shim rejects it at finalize |
| < 90800 | operation graph requires a handle | `cudnn_frontend_OperationGraph.h:241` | compiled out: the builder accepts a missing handle, but the shim's graph finalize returns `BAD_PARAM` (§1.3) |
| ≥ 90800 | heuristics device properties | `cudnn_frontend_Heuristics.h:236` | in |
| ≥ 91100 | reduction determinism | `cudnn_frontend_ReductionDesc.h:189` | in |
| ≥ 92100 | operation-graph override shape | `cudnn_frontend_OperationGraph.h:313` | in; `true` → `NOT_SUPPORTED` at shim finalize |
| ≥ 92200 | tensor constant value | `cudnn_frontend_Tensor.h:528` | in |
| ≥ 92200 | reshape-mode setter and attribute | `cudnn_frontend_Operation.h:1795,2594` | in |

`cudnn_frontend_utils.h` adds 66 enum-mapping gates (:262-2504) on 8600, 8700, 8800, 8900, 8906,
90100, 90500, 90700, 90900, 91100, 91400, 91500 and 92200, five of them `≥ 92200 && < 99900` (:1704,
:1712, :1720, :2173, :2178). All compile in except one dead `#elif CUDNN_VERSION >= 8700` (:2038),
whose `≥ 8800` branch (:2024) is taken.

`cudnn_frontend_shim.h` gates, for the subset the shim ports:

| Gate | Controls | Location | At 92200 |
|---|---|---|---|
| ≥ 90000 | `cudnnGetLastErrorString` | `:646` | in |
| ≥ 90500 | CUDA-graph populate / update | `:609`, `:626` | in, but dropped from the port |
| ≥ 92200 | causal conv1d wrapper | `:430` | in |
| ≥ 92300 && < 99900 | workspace-size query | `:588` | compiled out |

Legacy-sample gates (`upstream:samples/legacy_samples/test_list.cpp`) are ≥ 8300, 8600, 8700, 8800
and 90800 around test cases (§5.2), plus in-body `≥ 8301` gates at :1781 and :1880; all compile in.
PyTorch's errata ranges are evaluated against the same 92200 (§4.5).

### Runtime version checks

Upstream also branches at run time on `detail::get_backend_version()`. In the shim it returns the
claimed `CUDNN_VERSION`, 92200
(`projects/hipdnn/frontend/include/hipdnn_compatibility/cudnn/cudnn_frontend_utils.h:159-162`), as does
`cudnnGetVersion()`.

| Check | Location (`upstream:include/`) | At 92200 |
|---|---|---|
| `== 8500` (cuDNN 8.5 norm shape restriction in `validate_norm_op`) | `cudnn_frontend_Operation.h:2152` | false |
| `< 91100` (reject `IS_DETERMINISTIC` on old backends) | `cudnn_frontend_ReductionDesc.h:193` | false |
| `>= 8800` / `>= 8700` (F16x16 reordering) | `cudnn_frontend_utils.h:2026-2029` | not compiled: inside `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING`, which the shim ignores; the static branch (:2036) applies |
| `MINIMUM_VERSION > get_backend_version()` (`NV_CUDNN_FE_DYNAMIC_CHECK_*`) | `cudnn_frontend_shim.h:154`, `:164` | not compiled: dynamic loading only |
| `>= 90000` (fetch the last error string) | `cudnn_frontend_shim.h:645` | true |
| major = version / 1000, minor = (version / 100) % 10; cuDNN ≤ 8.2 table when major ≥ 8 and minor ≤ 2 | `cudnn_frontend_EngineFallbackList.h:32-37` | decodes 92200 as major 92, minor 2, so `get_fallback_engine_list` returns its hard-coded cuDNN 8.0-8.2 engine numbers (for example `iota(50)` for conv forward, :43-44). **Meaningless under the shim.** The function is `[[maybe_unused]] static` and is called by nothing in upstream `include/`, the legacy samples, PyTorch or XLA; `EngineFallbackListBuilder_v8::build()` uses FALLBACK heuristics instead (`cudnn_frontend_EngineFallbackList.h:160-165`) |
