// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Every public enum in hipBLASLt's headers, plus HIP's hipDataType and hipBLAS's common enums.
// Bind each value with one reg(e, "<C enum>", "<Python name>", <C value>) call: the
// hipblaslt-python-enum-sync pre-commit hook reads this file as text and compares it with the
// headers. Values left out on purpose are in ALLOWED_MISSING in src/hipblaslt/_coverage.py.
#include "init.hpp"

#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <map>
#include <string>
#include <tuple>

namespace nb = nanobind;

namespace
{
    // C enum name -> C value name -> (Python enum name, Python value name, integer value).
    using Registry
        = std::map<std::string,
                   std::map<std::string, std::tuple<std::string, std::string, long long>>>;

    Registry& registry()
    {
        static Registry r;
        return r;
    }

    template <typename E>
    void reg(nb::enum_<E>& e, const char* c_enum, const char* py_name, E value, const char* c_name)
    {
        e.value(py_name, value);
        registry()[c_enum][c_name]
            = {nb::cast<std::string>(e.attr("__name__")), py_name, static_cast<long long>(value)};
    }
} // namespace

// The C value's own spelling goes into the registry, so the coverage test can compare it with
// the header by name.
#define reg(e, c_enum, py_name, value) reg(e, c_enum, py_name, value, #value)

void init_enums(nb::module_& m)
{
    {
        nb::enum_<hipDataType> e(m, "DataType", nb::is_arithmetic());
        reg(e, "hipDataType", "R_32F", HIP_R_32F);
        reg(e, "hipDataType", "R_64F", HIP_R_64F);
        reg(e, "hipDataType", "R_16F", HIP_R_16F);
        reg(e, "hipDataType", "R_8I", HIP_R_8I);
        reg(e, "hipDataType", "R_32I", HIP_R_32I);
        reg(e, "hipDataType", "R_16BF", HIP_R_16BF);
        reg(e, "hipDataType", "R_4I", HIP_R_4I);
        reg(e, "hipDataType", "R_8F_E4M3", HIP_R_8F_E4M3);
        reg(e, "hipDataType", "R_8F_E5M2", HIP_R_8F_E5M2);
        reg(e, "hipDataType", "R_6F_E2M3", HIP_R_6F_E2M3);
        reg(e, "hipDataType", "R_6F_E3M2", HIP_R_6F_E3M2);
        reg(e, "hipDataType", "R_4F_E2M1", HIP_R_4F_E2M1);
        reg(e, "hipDataType", "R_8F_E4M3_FNUZ", HIP_R_8F_E4M3_FNUZ);
        reg(e, "hipDataType", "R_8F_E5M2_FNUZ", HIP_R_8F_E5M2_FNUZ);
    }
    {
        nb::enum_<hipblasComputeType_t> e(m, "ComputeType", nb::is_arithmetic());
        reg(e, "hipblasComputeType_t", "COMPUTE_16F", HIPBLAS_COMPUTE_16F);
        reg(e, "hipblasComputeType_t", "COMPUTE_32F", HIPBLAS_COMPUTE_32F);
        reg(e, "hipblasComputeType_t", "COMPUTE_32F_FAST_16F", HIPBLAS_COMPUTE_32F_FAST_16F);
        reg(e, "hipblasComputeType_t", "COMPUTE_32F_FAST_16BF", HIPBLAS_COMPUTE_32F_FAST_16BF);
        reg(e, "hipblasComputeType_t", "COMPUTE_32F_FAST_TF32", HIPBLAS_COMPUTE_32F_FAST_TF32);
        reg(e, "hipblasComputeType_t", "COMPUTE_64F", HIPBLAS_COMPUTE_64F);
        reg(e, "hipblasComputeType_t", "COMPUTE_32I", HIPBLAS_COMPUTE_32I);
    }
    {
        nb::enum_<hipblasOperation_t> e(m, "Operation", nb::is_arithmetic());
        reg(e, "hipblasOperation_t", "N", HIPBLAS_OP_N);
        reg(e, "hipblasOperation_t", "T", HIPBLAS_OP_T);
        reg(e, "hipblasOperation_t", "C", HIPBLAS_OP_C);
    }
    {
        nb::enum_<hipblasStatus_t> e(m, "Status", nb::is_arithmetic());
        reg(e, "hipblasStatus_t", "SUCCESS", HIPBLAS_STATUS_SUCCESS);
        reg(e, "hipblasStatus_t", "NOT_INITIALIZED", HIPBLAS_STATUS_NOT_INITIALIZED);
        reg(e, "hipblasStatus_t", "ALLOC_FAILED", HIPBLAS_STATUS_ALLOC_FAILED);
        reg(e, "hipblasStatus_t", "INVALID_VALUE", HIPBLAS_STATUS_INVALID_VALUE);
        reg(e, "hipblasStatus_t", "MAPPING_ERROR", HIPBLAS_STATUS_MAPPING_ERROR);
        reg(e, "hipblasStatus_t", "EXECUTION_FAILED", HIPBLAS_STATUS_EXECUTION_FAILED);
        reg(e, "hipblasStatus_t", "INTERNAL_ERROR", HIPBLAS_STATUS_INTERNAL_ERROR);
        reg(e, "hipblasStatus_t", "NOT_SUPPORTED", HIPBLAS_STATUS_NOT_SUPPORTED);
        reg(e, "hipblasStatus_t", "ARCH_MISMATCH", HIPBLAS_STATUS_ARCH_MISMATCH);
        reg(e, "hipblasStatus_t", "HANDLE_IS_NULLPTR", HIPBLAS_STATUS_HANDLE_IS_NULLPTR);
        reg(e, "hipblasStatus_t", "INVALID_ENUM", HIPBLAS_STATUS_INVALID_ENUM);
        reg(e, "hipblasStatus_t", "UNKNOWN", HIPBLAS_STATUS_UNKNOWN);
    }
    {
        nb::enum_<hipblasLtEpilogue_t> e(m, "Epilogue", nb::is_arithmetic());
        reg(e, "hipblasLtEpilogue_t", "DEFAULT", HIPBLASLT_EPILOGUE_DEFAULT);
        reg(e, "hipblasLtEpilogue_t", "RELU", HIPBLASLT_EPILOGUE_RELU);
        reg(e, "hipblasLtEpilogue_t", "BIAS", HIPBLASLT_EPILOGUE_BIAS);
        reg(e, "hipblasLtEpilogue_t", "RELU_BIAS", HIPBLASLT_EPILOGUE_RELU_BIAS);
        reg(e, "hipblasLtEpilogue_t", "GELU", HIPBLASLT_EPILOGUE_GELU);
        reg(e, "hipblasLtEpilogue_t", "GELU_BIAS", HIPBLASLT_EPILOGUE_GELU_BIAS);
        reg(e, "hipblasLtEpilogue_t", "RELU_AUX", HIPBLASLT_EPILOGUE_RELU_AUX);
        reg(e, "hipblasLtEpilogue_t", "RELU_AUX_BIAS", HIPBLASLT_EPILOGUE_RELU_AUX_BIAS);
        reg(e, "hipblasLtEpilogue_t", "DRELU", HIPBLASLT_EPILOGUE_DRELU);
        reg(e, "hipblasLtEpilogue_t", "DRELU_BGRAD", HIPBLASLT_EPILOGUE_DRELU_BGRAD);
        reg(e, "hipblasLtEpilogue_t", "GELU_AUX", HIPBLASLT_EPILOGUE_GELU_AUX);
        reg(e, "hipblasLtEpilogue_t", "GELU_AUX_BIAS", HIPBLASLT_EPILOGUE_GELU_AUX_BIAS);
        reg(e, "hipblasLtEpilogue_t", "DGELU", HIPBLASLT_EPILOGUE_DGELU);
        reg(e, "hipblasLtEpilogue_t", "DGELU_BGRAD", HIPBLASLT_EPILOGUE_DGELU_BGRAD);
        reg(e, "hipblasLtEpilogue_t", "BGRADA", HIPBLASLT_EPILOGUE_BGRADA);
        reg(e, "hipblasLtEpilogue_t", "BGRADB", HIPBLASLT_EPILOGUE_BGRADB);
        reg(e, "hipblasLtEpilogue_t", "SIGMOID", HIPBLASLT_EPILOGUE_SIGMOID);
        reg(e, "hipblasLtEpilogue_t", "SWISH_EXT", HIPBLASLT_EPILOGUE_SWISH_EXT);
        reg(e, "hipblasLtEpilogue_t", "SWISH_BIAS_EXT", HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT);
        reg(e, "hipblasLtEpilogue_t", "CLAMP_EXT", HIPBLASLT_EPILOGUE_CLAMP_EXT);
        reg(e, "hipblasLtEpilogue_t", "CLAMP_BIAS_EXT", HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT);
        reg(e, "hipblasLtEpilogue_t", "CLAMP_AUX_EXT", HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT);
        reg(e, "hipblasLtEpilogue_t", "CLAMP_AUX_BIAS_EXT", HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT);
    }
    {
        nb::enum_<hipblasLtMatmulDescAttributes_t> e(m, "MatmulDescAttr", nb::is_arithmetic());
        reg(e, "hipblasLtMatmulDescAttributes_t", "TRANSA", HIPBLASLT_MATMUL_DESC_TRANSA);
        reg(e, "hipblasLtMatmulDescAttributes_t", "TRANSB", HIPBLASLT_MATMUL_DESC_TRANSB);
        reg(e, "hipblasLtMatmulDescAttributes_t", "EPILOGUE", HIPBLASLT_MATMUL_DESC_EPILOGUE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "BIAS_POINTER",
            HIPBLASLT_MATMUL_DESC_BIAS_POINTER);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "BIAS_DATA_TYPE",
            HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "A_SCALE_POINTER",
            HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "B_SCALE_POINTER",
            HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "C_SCALE_POINTER",
            HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "D_SCALE_POINTER",
            HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "EPILOGUE_AUX_SCALE_POINTER",
            HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "EPILOGUE_AUX_POINTER",
            HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "EPILOGUE_AUX_LD",
            HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "EPILOGUE_AUX_BATCH_STRIDE",
            HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "POINTER_MODE",
            HIPBLASLT_MATMUL_DESC_POINTER_MODE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "AMAX_D_POINTER",
            HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "EPILOGUE_AUX_DATA_TYPE",
            HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "BIAS_BATCH_STRIDE",
            HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "A_SCALE_MODE",
            HIPBLASLT_MATMUL_DESC_A_SCALE_MODE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "B_SCALE_MODE",
            HIPBLASLT_MATMUL_DESC_B_SCALE_MODE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "SM_COUNT_TARGET",
            HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "ALPHA_BATCH_STRIDE",
            HIPBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "BETA_BATCH_STRIDE",
            HIPBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "COMPUTE_INPUT_TYPE_A_EXT",
            HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "COMPUTE_INPUT_TYPE_B_EXT",
            HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "EPILOGUE_ACT_ARG0_EXT",
            HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "EPILOGUE_ACT_ARG1_EXT",
            HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "STREAMK_TILE_SCHEDULING_EXT",
            HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT);
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "UNIFORM_SUMMATION_ORDER_EXT",
            HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT);
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "FUSED_EPILOGUE",
            HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE);
#endif
        reg(e,
            "hipblasLtMatmulDescAttributes_t",
            "A_INT4_ENCODING_EXT",
            HIPBLASLT_MATMUL_DESC_A_INT4_ENCODING_EXT);
    }
    {
        nb::enum_<hipblasLtMatrixLayoutAttribute_t> e(m, "MatrixLayoutAttr", nb::is_arithmetic());
        reg(e,
            "hipblasLtMatrixLayoutAttribute_t",
            "BATCH_COUNT",
            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT);
        reg(e,
            "hipblasLtMatrixLayoutAttribute_t",
            "STRIDED_BATCH_OFFSET",
            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET);
        reg(e, "hipblasLtMatrixLayoutAttribute_t", "TYPE", HIPBLASLT_MATRIX_LAYOUT_TYPE);
        reg(e, "hipblasLtMatrixLayoutAttribute_t", "ORDER", HIPBLASLT_MATRIX_LAYOUT_ORDER);
        reg(e, "hipblasLtMatrixLayoutAttribute_t", "ROWS", HIPBLASLT_MATRIX_LAYOUT_ROWS);
        reg(e, "hipblasLtMatrixLayoutAttribute_t", "COLS", HIPBLASLT_MATRIX_LAYOUT_COLS);
        reg(e, "hipblasLtMatrixLayoutAttribute_t", "LD", HIPBLASLT_MATRIX_LAYOUT_LD);
        reg(e,
            "hipblasLtMatrixLayoutAttribute_t",
            "BATCH_MODE",
            HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE);
        reg(e, "hipblasLtMatrixLayoutAttribute_t", "OFFSET", HIPBLASLT_MATRIX_LAYOUT_OFFSET);
        reg(e,
            "hipblasLtMatrixLayoutAttribute_t",
            "GROUPED_MATRIX_LAYOUT_ROWS_ARRAY",
            HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY);
        reg(e,
            "hipblasLtMatrixLayoutAttribute_t",
            "GROUPED_MATRIX_LAYOUT_COLS_ARRAY",
            HIPBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY);
        reg(e,
            "hipblasLtMatrixLayoutAttribute_t",
            "GROUPED_MATRIX_LAYOUT_LD_ARRAY",
            HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY);
        reg(e,
            "hipblasLtMatrixLayoutAttribute_t",
            "GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH",
            HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH);
        reg(e,
            "hipblasLtMatrixLayoutAttribute_t",
            "GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH",
            HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH);
    }
    {
        nb::enum_<hipblasLtMatmulPreferenceAttributes_t> e(
            m, "PreferenceAttr", nb::is_arithmetic());
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "SEARCH_MODE",
            HIPBLASLT_MATMUL_PREF_SEARCH_MODE);
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "MAX_WORKSPACE_BYTES",
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES);
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "SM_COUNT_TARGET",
            HIPBLASLT_MATMUL_PREF_SM_COUNT_TARGET);
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "GROUPED_DESC_D_AVERAGE_ROWS",
            HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS);
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "GROUPED_DESC_D_AVERAGE_COLS",
            HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS);
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "GROUPED_AVERAGE_REDUCTION_DIM",
            HIPBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM);
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "GROUPED_DESC_D_MAX_ROWS_EXT",
            HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_ROWS_EXT);
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "GROUPED_DESC_D_MAX_COLS_EXT",
            HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_COLS_EXT);
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "GROUPED_DESC_D_MAX_TOTAL_ROWS_EXT",
            HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_ROWS_EXT);
        reg(e,
            "hipblasLtMatmulPreferenceAttributes_t",
            "GROUPED_DESC_D_MAX_TOTAL_COLS_EXT",
            HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_COLS_EXT);
    }
    {
        nb::enum_<hipblasLtMatmulMatrixScale_t> e(m, "ScaleMode", nb::is_arithmetic());
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "SCALAR_32F",
            HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC32_UE8M0",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "OUTER_VEC_32F",
            HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "BLK32_UE8M0_32_8_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC32_16BF_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC64_16BF_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16BF_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC128_16BF_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16BF_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC32_16BF_ZP_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_ZP_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC64_16BF_ZP_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16BF_ZP_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC128_16BF_ZP_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16BF_ZP_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC32_16F_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC64_16F_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16F_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC128_16F_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16F_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC32_16F_ZP_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_ZP_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC64_16F_ZP_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16F_ZP_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "VEC128_16F_ZP_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16F_ZP_EXT);
        reg(e,
            "hipblasLtMatmulMatrixScale_t",
            "BLK32_UE8M0_128_4_EXT",
            HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT);
    }
    {
        nb::enum_<hipblasLtPointerMode_t> e(m, "PointerMode", nb::is_arithmetic());
        reg(e, "hipblasLtPointerMode_t", "HOST", HIPBLASLT_POINTER_MODE_HOST);
        reg(e, "hipblasLtPointerMode_t", "DEVICE", HIPBLASLT_POINTER_MODE_DEVICE);
        reg(e,
            "hipblasLtPointerMode_t",
            "ALPHA_DEVICE_VECTOR_BETA_HOST",
            HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST);
        reg(e,
            "hipblasLtPointerMode_t",
            "ALPHA_DEVICE_SCALAR_BETA_HOST_EXT",
            HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_SCALAR_BETA_HOST_EXT);
    }
    {
        nb::enum_<hipblasLtOrder_t> e(m, "Order", nb::is_arithmetic());
        reg(e, "hipblasLtOrder_t", "COL", HIPBLASLT_ORDER_COL);
        reg(e, "hipblasLtOrder_t", "ROW", HIPBLASLT_ORDER_ROW);
        reg(e, "hipblasLtOrder_t", "COL16_4R32", HIPBLASLT_ORDER_COL16_4R32);
        reg(e, "hipblasLtOrder_t", "COL16_4R16", HIPBLASLT_ORDER_COL16_4R16);
        reg(e, "hipblasLtOrder_t", "COL16_4R8", HIPBLASLT_ORDER_COL16_4R8);
        reg(e, "hipblasLtOrder_t", "COL16_4R4", HIPBLASLT_ORDER_COL16_4R4);
        reg(e, "hipblasLtOrder_t", "COL16_4R2", HIPBLASLT_ORDER_COL16_4R2);
        reg(e, "hipblasLtOrder_t", "COL256_R128_EXT", HIPBLASLT_ORDER_COL256_R128_EXT);
    }
    {
        nb::enum_<hipblasLtBatchMode_t> e(m, "BatchMode", nb::is_arithmetic());
        reg(e, "hipblasLtBatchMode_t", "STRIDED", HIPBLASLT_BATCH_MODE_STRIDED);
        reg(e, "hipblasLtBatchMode_t", "POINTER_ARRAY", HIPBLASLT_BATCH_MODE_POINTER_ARRAY);
        reg(e, "hipblasLtBatchMode_t", "GROUPED", HIPBLASLT_BATCH_MODE_GROUPED);
    }
    {
        nb::enum_<hipblasLtIntegerWidth_t> e(m, "IntegerWidth", nb::is_arithmetic());
        reg(e, "hipblasLtIntegerWidth_t", "INTEGER_WIDTH_32", HIPBLASLT_INTEGER_WIDTH_32);
        reg(e, "hipblasLtIntegerWidth_t", "INTEGER_WIDTH_64", HIPBLASLT_INTEGER_WIDTH_64);
    }
    {
        nb::enum_<hipblasLtInt4Encoding_t> e(m, "Int4Encoding", nb::is_arithmetic());
        reg(e, "hipblasLtInt4Encoding_t", "SIGNED_EXT", HIPBLASLT_INT4_ENCODING_SIGNED_EXT);
        reg(e,
            "hipblasLtInt4Encoding_t",
            "UNSIGNED_BIAS8_EXT",
            HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT);
    }
    {
        nb::enum_<hipblasLtStreamKTileSchedulingMode_t> e(
            m, "StreamKTileSchedulingMode", nb::is_arithmetic());
        reg(e,
            "hipblasLtStreamKTileSchedulingMode_t",
            "OFF",
            HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF);
        reg(e, "hipblasLtStreamKTileSchedulingMode_t", "ON", HIPBLASLT_STREAMK_TILE_SCHEDULING_ON);
        reg(e,
            "hipblasLtStreamKTileSchedulingMode_t",
            "AUTO",
            HIPBLASLT_STREAMK_TILE_SCHEDULING_AUTO);
    }
    {
        nb::enum_<hipblasLtMatrixTransformDescAttributes_t> e(
            m, "MatrixTransformDescAttr", nb::is_arithmetic());
        reg(e,
            "hipblasLtMatrixTransformDescAttributes_t",
            "SCALE_TYPE",
            HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE);
        reg(e,
            "hipblasLtMatrixTransformDescAttributes_t",
            "POINTER_MODE",
            HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE);
        reg(e,
            "hipblasLtMatrixTransformDescAttributes_t",
            "TRANSA",
            HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA);
        reg(e,
            "hipblasLtMatrixTransformDescAttributes_t",
            "TRANSB",
            HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB);
    }
    {
        nb::enum_<hipblaslt_ext::GemmType> e(m, "GemmType", nb::is_arithmetic());
        reg(e, "GemmType", "GEMM", hipblaslt_ext::GemmType::HIPBLASLT_GEMM);
        reg(e, "GemmType", "GROUPED_GEMM", hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM);
    }
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
    {
        nb::enum_<hipblasLtFuseableEpilogue_t> e(m, "FuseableEpilogue", nb::is_arithmetic());
        reg(e, "hipblasLtFuseableEpilogue_t", "A2A_PREFIX", HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX);
    }
#endif
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
    {
        nb::enum_<hipblasLtFusedEpilogueAttribute_t> e(m, "FusedEpilogueAttr", nb::is_arithmetic());
        reg(e,
            "hipblasLtFusedEpilogueAttribute_t",
            "A2A_PREFIX_SDMA_QUEUES",
            HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES);
        reg(e,
            "hipblasLtFusedEpilogueAttribute_t",
            "A2A_PREFIX_RECV_PTRS",
            HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS);
        reg(e,
            "hipblasLtFusedEpilogueAttribute_t",
            "A2A_PREFIX_EXTENT",
            HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT);
        reg(e,
            "hipblasLtFusedEpilogueAttribute_t",
            "A2A_PREFIX_COMPLETION_MODE",
            HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_COMPLETION_MODE);
        reg(e,
            "hipblasLtFusedEpilogueAttribute_t",
            "COMM_CHANNEL",
            HIPBLASLT_FUSED_EPILOGUE_COMM_CHANNEL);
    }
#endif
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
    {
        nb::enum_<hipblasLtA2ACompletionMode_t> e(m, "A2ACompletionMode", nb::is_arithmetic());
        reg(e,
            "hipblasLtA2ACompletionMode_t",
            "IN_KERNEL_FULL",
            HIPBLASLT_A2A_COMPLETION_IN_KERNEL_FULL);
    }
#endif

    m.def(
        "_enum_registry",
        []() { return registry(); },
        "{C enum: {C value: (Python enum, Python value, integer)}} for every bound value.");
}
