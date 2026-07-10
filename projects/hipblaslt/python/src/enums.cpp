// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-version.h>
#include <map>
#include <string>
#include "init.hpp"

namespace nb = nanobind;

// Registry: enum name -> {member name -> int value}. Populated as we bind.
static std::map<std::string, std::map<std::string, int>>& registry()
{
    static std::map<std::string, std::map<std::string, int>> r;
    return r;
}

template <typename E>
static void reg(nb::enum_<E>& e, const char* enum_name, const char* member, E value)
{
    e.value(member, value);
    registry()[enum_name][member] = static_cast<int>(value);
}

void init_enums(nb::module_& m)
{
    {
        nb::enum_<hipDataType> e(m, "DataType", nb::is_arithmetic());
        reg(e, "DataType", "R_16F", HIP_R_16F);
        reg(e, "DataType", "R_32F", HIP_R_32F);
        reg(e, "DataType", "R_64F", HIP_R_64F);
        reg(e, "DataType", "R_16BF", HIP_R_16BF);
        reg(e, "DataType", "R_8I", HIP_R_8I);
        reg(e, "DataType", "R_32I", HIP_R_32I);
        reg(e, "DataType", "R_8F_E4M3", HIP_R_8F_E4M3);
        reg(e, "DataType", "R_8F_E5M2", HIP_R_8F_E5M2);
        reg(e, "DataType", "R_8F_E4M3_FNUZ", HIP_R_8F_E4M3_FNUZ);
        reg(e, "DataType", "R_8F_E5M2_FNUZ", HIP_R_8F_E5M2_FNUZ);
    }
    {
        nb::enum_<hipblasComputeType_t> e(m, "ComputeType", nb::is_arithmetic());
        reg(e, "ComputeType", "COMPUTE_32F", HIPBLAS_COMPUTE_32F);
        reg(e, "ComputeType", "COMPUTE_32F_FAST_16F", HIPBLAS_COMPUTE_32F_FAST_16F);
        reg(e, "ComputeType", "COMPUTE_32F_FAST_16BF", HIPBLAS_COMPUTE_32F_FAST_16BF);
        reg(e, "ComputeType", "COMPUTE_64F", HIPBLAS_COMPUTE_64F);
        reg(e, "ComputeType", "COMPUTE_32I", HIPBLAS_COMPUTE_32I);
    }
    {
        nb::enum_<hipblasLtEpilogue_t> e(m, "Epilogue", nb::is_arithmetic());
        reg(e, "Epilogue", "DEFAULT", HIPBLASLT_EPILOGUE_DEFAULT);
        reg(e, "Epilogue", "RELU", HIPBLASLT_EPILOGUE_RELU);
        reg(e, "Epilogue", "BIAS", HIPBLASLT_EPILOGUE_BIAS);
        reg(e, "Epilogue", "RELU_BIAS", HIPBLASLT_EPILOGUE_RELU_BIAS);
        reg(e, "Epilogue", "GELU", HIPBLASLT_EPILOGUE_GELU);
        reg(e, "Epilogue", "GELU_BIAS", HIPBLASLT_EPILOGUE_GELU_BIAS);
        reg(e, "Epilogue", "SIGMOID", HIPBLASLT_EPILOGUE_SIGMOID_EXT);
        // NOTE: the coverage harness (Task 18) enumerates the header to catch any
        // member omitted here; extend this list when that test flags a gap.
    }
    {
        nb::enum_<hipblasLtMatmulDescAttributes_t> e(m, "MatmulDescAttr", nb::is_arithmetic());
        reg(e, "MatmulDescAttr", "TRANSA", HIPBLASLT_MATMUL_DESC_TRANSA);
        reg(e, "MatmulDescAttr", "TRANSB", HIPBLASLT_MATMUL_DESC_TRANSB);
        reg(e, "MatmulDescAttr", "EPILOGUE", HIPBLASLT_MATMUL_DESC_EPILOGUE);
        reg(e, "MatmulDescAttr", "BIAS_POINTER", HIPBLASLT_MATMUL_DESC_BIAS_POINTER);
        reg(e, "MatmulDescAttr", "A_SCALE_POINTER", HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER);
        reg(e, "MatmulDescAttr", "B_SCALE_POINTER", HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER);
        reg(e, "MatmulDescAttr", "D_SCALE_POINTER", HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER);
        reg(e, "MatmulDescAttr", "A_SCALE_MODE", HIPBLASLT_MATMUL_DESC_A_SCALE_MODE);
        reg(e, "MatmulDescAttr", "B_SCALE_MODE", HIPBLASLT_MATMUL_DESC_B_SCALE_MODE);
    }
    {
        nb::enum_<hipblasLtMatrixLayoutAttribute_t> e(m, "MatrixLayoutAttr", nb::is_arithmetic());
        reg(e, "MatrixLayoutAttr", "BATCH_COUNT", HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT);
        reg(e, "MatrixLayoutAttr", "STRIDED_BATCH_OFFSET", HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET);
    }
    {
        nb::enum_<hipblasLtMatmulPreferenceAttributes_t> e(m, "PreferenceAttr", nb::is_arithmetic());
        reg(e, "PreferenceAttr", "MAX_WORKSPACE_BYTES", HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES);
    }
    {
        nb::enum_<hipblasLtMatmulMatrixScale_t> e(m, "ScaleMode", nb::is_arithmetic());
        reg(e, "ScaleMode", "SCALAR_32F", HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
        reg(e, "ScaleMode", "VEC32_UE8M0", HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
        reg(e, "ScaleMode", "OUTER_VEC_32F", HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F);
// BLK32_UE8M0_32_8_EXT (value 1001) was added after hipBLASLt 1.2.x; guard by version.
#if HIPBLASLT_VERSION_MAJOR > 1 || (HIPBLASLT_VERSION_MAJOR == 1 && HIPBLASLT_VERSION_MINOR > 2)
        reg(e, "ScaleMode", "BLK32_UE8M0_32_8_EXT", HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT);
#endif
    }

    m.def("enum_members", [](const std::string& name) -> std::map<std::string, int> {
        auto it = registry().find(name);
        if(it == registry().end())
            return {};
        return it->second;
    }, "Return {member_name: int_value} for a bound enum.");
}
