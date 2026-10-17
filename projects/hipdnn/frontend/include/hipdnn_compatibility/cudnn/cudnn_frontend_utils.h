// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend_utils.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file cudnn_frontend_utils.h
 * @brief cuDNN FE enums, exceptions, enum converters and string helpers for the hipDNN shim.
 */

#pragma once

#if defined(HIPDNN_FRONTEND_SKIP_JSON_LIB) && !defined(CUDNN_FRONTEND_SKIP_JSON_LIB)
#define CUDNN_FRONTEND_SKIP_JSON_LIB
#endif

#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_compatibility/cudnn/cudnn_backend_base.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_Logging.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_shim.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_version.h>
#include <hipdnn_compatibility/cudnn/detail/backend/type_mapping.h>
#include <hipdnn_compatibility/cudnn/detail/frontend_types.h>

#if !defined(CUDNN_FRONTEND_SKIP_JSON_LIB) && !defined(HIPDNN_FRONTEND_SKIP_JSON_LIB)
#include <nlohmann/json.hpp>

using json = nlohmann::json;
#endif

// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define CUDNN_FRONTEND_UNUSED(X) ((void)X)

namespace hipdnn_frontend::compatibility::cudnn_frontend
{
static_assert(CUDNN_FRONTEND_VERSION == 12400,
              "cuDNN FE version pin changed: re-diff this file against upstream "
              "include/cudnn_frontend_utils.h at the new tag, then update this assert.");

// NOLINTBEGIN(readability-identifier-naming,modernize-use-override,readability-make-member-function-const,misc-const-correctness,modernize-return-braced-init-list):
// ported verbatim; upstream's spelling and shapes are the source-compatible surface.

/// Detailed feature_vector. Generally the Tensor and Operation properties
using feature_vector_t = std::vector<int64_t>;

class cudnnException : public std::runtime_error
{
public:
    cudnnException(const char* message, cudnnStatus_t status) noexcept
        : std::runtime_error(message)
    {
        error_status = status;
    }
    virtual const char* what() const noexcept
    {
        return std::runtime_error::what();
    }
    cudnnStatus_t getCudnnStatus()
    {
        return error_status;
    }

    cudnnStatus_t error_status;
};

inline bool AllowAll(cudnnBackendDescriptor_t engine_config)
{
    (void)engine_config;
    return false;
}

inline std::string to_string(const cudnnStatus_t status)
{
    return detail::get_error_string(status);
}

#ifndef NV_CUDNN_DISABLE_EXCEPTION
[[noreturn]]
#endif
static inline void set_error_and_throw_exception(const BackendDescriptor* desc,
                                                 cudnnStatus_t status,
                                                 const char* message)
{

    std::string padded_message = std::string(message) + detail::get_last_error_string_();
    if(desc != nullptr)
    {
        desc->set_status(status);
        desc->set_error(padded_message.c_str());
    }
#ifndef NV_CUDNN_DISABLE_EXCEPTION
    throw cudnnException(std::string(std::string(padded_message) + std::string(" cudnn_status: ")
                                     + to_string(status))
                             .c_str(),
                         status);
#endif
}

inline std::string to_string(cudnnBackendBehaviorNote_t note)
{
    switch(note)
    {
    case CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION:
        return std::string("CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION");
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER:
        return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER");
    case CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER:
        return std::string("CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER");
    case CUDNN_BEHAVIOR_NOTE_TYPE_COUNT:
        return std::string("CUDNN_BEHAVIOR_NOTE_TYPE_COUNT");
        // If none of the above cases hit, its definitely strict nan prop and should raise an error.
#if(CUDNN_VERSION >= 90500)
    case CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API:
        return std::string("CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API");
#endif
#if(CUDNN_VERSION >= 91500)
    case CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY:
        return std::string("CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY");
#endif
#ifndef NO_DEFAULT_IN_SWITCH
    default:
        return std::string("UNKNOWN_BEHAVIOR_NOTE");
#endif
    }
    return std::string("INVALID_BEHAVIOR_NOTE");
}

inline std::string to_string(cudnnBackendNumericalNote_t note)
{
    switch(note)
    {
    case CUDNN_NUMERICAL_NOTE_TENSOR_CORE:
        return std::string("CUDNN_NUMERICAL_NOTE_TENSOR_CORE");
    case CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS:
        return std::string("CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS");
    case CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION:
        return std::string("CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION");
    case CUDNN_NUMERICAL_NOTE_FFT:
        return std::string("CUDNN_NUMERICAL_NOTE_FFT");
    case CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC:
        return std::string("CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC");
    case CUDNN_NUMERICAL_NOTE_WINOGRAD:
        return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD");
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4:
        return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4");
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6:
        return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6");
    case CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13:
        return std::string("CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13");
    case CUDNN_NUMERICAL_NOTE_TYPE_COUNT:
        return std::string("CUDNN_NUMERICAL_NOTE_TYPE_COUNT");

        // If none of the above cases hit, its definitely strict nan prop and should raise an error.
#if(CUDNN_VERSION >= 90100)
    case CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP:
        return std::string("CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP");
#endif
#ifndef NO_DEFAULT_IN_SWITCH
    default:
        return std::string("UNKNOWN_NUMERICAL_NOTE");
#endif
    }
    return std::string("INVALID_NUMERICAL_NOTE");
}

#if(CUDNN_VERSION >= 8700)
inline std::string to_string(cudnnRngDistribution_t distribution)
{
    switch(distribution)
    {
    case CUDNN_RNG_DISTRIBUTION_BERNOULLI:
        return std::string("CUDNN_RNG_DISTRIBUTION_BERNOULLI");
    case CUDNN_RNG_DISTRIBUTION_UNIFORM:
        return std::string("CUDNN_RNG_DISTRIBUTION_UNIFORM");
    case CUDNN_RNG_DISTRIBUTION_NORMAL:
        return std::string("CUDNN_RNG_DISTRIBUTION_NORMAL");
#ifndef NO_DEFAULT_IN_SWITCH
    default:
        return std::string("UNKNOWN_CUDNN_DISTRIBUTION");
#endif
    }
    return std::string("");
}
#endif

inline int64_t get_pointwise_mode_port_count(const PointwiseMode_t& mode)
{
    switch(mode)
    {
    case PointwiseMode_t::NOT_SET:
        return 0;

    case PointwiseMode_t::ADD:
    case PointwiseMode_t::MUL:
    case PointwiseMode_t::DIV:
    case PointwiseMode_t::ADD_SQUARE:
    case PointwiseMode_t::SUB:
    case PointwiseMode_t::CMP_EQ:
    case PointwiseMode_t::CMP_NEQ:
    case PointwiseMode_t::CMP_GT:
    case PointwiseMode_t::CMP_GE:
    case PointwiseMode_t::CMP_LT:
    case PointwiseMode_t::CMP_LE:
    case PointwiseMode_t::LOGICAL_AND:
    case PointwiseMode_t::LOGICAL_OR:
    case PointwiseMode_t::MIN:
    case PointwiseMode_t::MAX:
    case PointwiseMode_t::MOD:
    case PointwiseMode_t::RELU_BWD:
    case PointwiseMode_t::TANH_BWD:
    case PointwiseMode_t::SIGMOID_BWD:
    case PointwiseMode_t::ELU_BWD:
    case PointwiseMode_t::GELU_BWD:
    case PointwiseMode_t::SOFTPLUS_BWD:
    case PointwiseMode_t::SWISH_BWD:
    case PointwiseMode_t::GELU_APPROX_TANH_BWD:
    case PointwiseMode_t::POW:
        return 3;

    case PointwiseMode_t::SQRT:
    case PointwiseMode_t::RELU_FWD:
    case PointwiseMode_t::TANH_FWD:
    case PointwiseMode_t::SIGMOID_FWD:
    case PointwiseMode_t::ELU_FWD:
    case PointwiseMode_t::GELU_FWD:
    case PointwiseMode_t::SOFTPLUS_FWD:
    case PointwiseMode_t::SWISH_FWD:
    case PointwiseMode_t::EXP:
    case PointwiseMode_t::LOG:
    case PointwiseMode_t::NEG:
    case PointwiseMode_t::ABS:
    case PointwiseMode_t::CEIL:
    case PointwiseMode_t::FLOOR:
    case PointwiseMode_t::COS:
    case PointwiseMode_t::TAN:
    case PointwiseMode_t::SIN:
    case PointwiseMode_t::RSQRT:
    case PointwiseMode_t::LOGICAL_NOT:
    case PointwiseMode_t::GEN_INDEX:
    case PointwiseMode_t::ERF:
    case PointwiseMode_t::GELU_APPROX_TANH_FWD:
    case PointwiseMode_t::IDENTITY:
    case PointwiseMode_t::RECIPROCAL:
        return 2;

    case PointwiseMode_t::BINARY_SELECT:
        return 4;
    default:
        break;
    }
    return -1;
}

inline std::ostream& operator<<(std::ostream& os, const DescriptorType_t& mode)
{
    switch(mode)
    {
    case DescriptorType_t::POINTWISE_DESCRIPTOR:
        os << "POINTWISE_DESCRIPTOR";
        break;
    case DescriptorType_t::CONVOLUTION_DESCRIPTOR:
        os << "CONVOLUTION_DESCRIPTOR";
        break;
    case DescriptorType_t::ENGINE_DESCRIPTOR:
        os << "ENGINE_DESCRIPTOR";
        break;
    case DescriptorType_t::ENGINECFG_DESCRIPTOR:
        os << "ENGINECFG_DESCRIPTOR";
        break;
    case DescriptorType_t::ENGINEHEUR_DESCRIPTOR:
        os << "ENGINEHEUR_DESCRIPTOR";
        break;
    case DescriptorType_t::EXECUTION_PLAN_DESCRIPTOR:
        os << "EXECUTION_PLAN_DESCRIPTOR";
        break;
    case DescriptorType_t::INTERMEDIATE_INFO_DESCRIPTOR:
        os << "INTERMEDIATE_INFO_DESCRIPTOR";
        break;
    case DescriptorType_t::KNOB_CHOICE_DESCRIPTOR:
        os << "KNOB_CHOICE_DESCRIPTOR";
        break;
    case DescriptorType_t::KNOB_INFO_DESCRIPTOR:
        os << "KNOB_INFO_DESCRIPTOR";
        break;
    case DescriptorType_t::LAYOUT_INFO_DESCRIPTOR:
        os << "LAYOUT_INFO_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
        os << "OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
        os << "OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
        os << "OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR:
        os << "OPERATION_POINTWISE_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR:
        os << "OPERATION_GEN_STATS_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATIONGRAPH_DESCRIPTOR:
        os << "OPERATIONGRAPH_DESCRIPTOR";
        break;
    case DescriptorType_t::VARIANT_PACK_DESCRIPTOR:
        os << "VARIANT_PACK_DESCRIPTOR";
        break;
    case DescriptorType_t::TENSOR_DESCRIPTOR:
        os << "TENSOR_DESCRIPTOR";
        break;
    case DescriptorType_t::MATMUL_DESCRIPTOR:
        os << "MATMUL_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR:
        os << "OPERATION_MATMUL_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR:
        os << "OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR";
        break;
    case DescriptorType_t::REDUCTION_DESCRIPTOR:
        os << "REDUCTION_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR:
        os << "OPERATION_REDUCTION_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR:
        os << "OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR";
        break;
    case DescriptorType_t::RESAMPLE_DESCRIPTOR:
        os << "RESAMPLE_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR:
        os << "OPERATION_RESAMPLE_FWD_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR:
        os << "OPERATION_RESAMPLE_BWD_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR:
        os << "OPERATION_CONCAT_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR:
        os << "OPERATION_SIGNAL_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR:
        os << "OPERATION_NORM_FORWARD_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR:
        os << "OPERATION_NORM_BACKWARD_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR:
        os << "OPERATION_RESHAPE_DESCRIPTOR";
        break;
    case DescriptorType_t::RNG_DESCRIPTOR:
        os << "RNG_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_RNG_DESCRIPTOR:
        os << "OPERATION_RNG_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR:
        os << "OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR:
        os << "OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR:
        os << "OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_CONCATENATE_DESCRIPTOR:
        os << "OPERATION_CONCATENATE_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR:
        os << "OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR:
        os << "OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_TRANSPOSE_DESCRIPTOR:
        os << "OPERATION_TRANSPOSE_DESCRIPTOR";
        break;
    case DescriptorType_t::OPERATION_SLICE_DESCRIPTOR:
        os << "OPERATION_SLICE_DESCRIPTOR";
        break;
    case DescriptorType_t::NOT_SET:
        os << "NOT_SET";
        break;
    default:
        break;
    }
    return os;
}

// NOLINTEND(readability-identifier-naming,modernize-use-override,readability-make-member-function-const,misc-const-correctness,modernize-return-braced-init-list)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

// NOLINTBEGIN(readability-identifier-naming): upstream's converter names are the
// source-compatible surface.

// Upstream's converters are hand-written switch tables; the shim routes them through
// backend::toCudnn/toFe so there is one mapping, keeping upstream's fallback values.
template <typename CudnnEnum, typename FeEnum>
cudnnStatus_t convertToCudnnOrInvalid(const FeEnum mode, CudnnEnum& cudnn_mode)
{
    if(auto value = backend::toCudnn(mode))
    {
        cudnn_mode = *value;
        return CUDNN_STATUS_SUCCESS;
    }
    return CUDNN_STATUS_INVALID_VALUE;
}

inline cudnnStatus_t convert_to_cudnn_type(const DataType_t mode, cudnnDataType_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline cudnnStatus_t convert_to_cudnn_type(const ReductionMode_t mode,
                                           cudnnReduceTensorOp_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline cudnnStatus_t convert_to_cudnn_type(const PointwiseMode_t mode,
                                           cudnnPointwiseMode_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline cudnnStatus_t convert_to_cudnn_type(const NumericalNote_t mode,
                                           cudnnBackendNumericalNote_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline cudnnStatus_t convert_to_cudnn_type(const BehaviorNote_t mode,
                                           cudnnBackendBehaviorNote_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline BehaviorNote_t convert_from_cudnn_type(const cudnnBackendBehaviorNote_t cudnn_mode)
{
    return backend::toFe(cudnn_mode).value_or(BehaviorNote_t::NOT_SET);
}

inline NumericalNote_t convert_from_cudnn_type(const cudnnBackendNumericalNote_t cudnn_mode)
{
    return backend::toFe(cudnn_mode).value_or(NumericalNote_t::NOT_SET);
}

inline cudnnStatus_t convert_to_cudnn_type(const DescriptorType_t mode,
                                           cudnnBackendDescriptorType_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline cudnnStatus_t convert_to_cudnn_type(const ResampleMode_t mode,
                                           cudnnResampleMode_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline cudnnStatus_t convert_to_cudnn_type(const PaddingMode_t mode, cudnnPaddingMode_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline cudnnStatus_t convert_to_cudnn_type(const NormMode_t mode,
                                           cudnnBackendNormMode_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline cudnnStatus_t convert_to_cudnn_type(const NormFwdPhase_t mode,
                                           cudnnBackendNormFwdPhase_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline cudnnStatus_t convert_to_cudnn_type(const ReshapeMode_t mode,
                                           cudnnBackendReshapeMode_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline void convert_from_cudnn_type(const cudnnPaddingMode_t cudnn_mode, PaddingMode_t& mode)
{
    mode = backend::toFe(cudnn_mode).value_or(PaddingMode_t::NOT_SET);
}

inline ConvolutionMode_t convert_from_cudnn_type(const cudnnConvolutionMode_t cudnn_mode)
{
    return backend::toFe(cudnn_mode).value_or(ConvolutionMode_t::NOT_SET);
}

inline cudnnConvolutionMode_t convert_to_cudnn_type(const ConvolutionMode_t cudnn_mode)
{
    return backend::toCudnn(cudnn_mode).value_or(CUDNN_CROSS_CORRELATION);
}

// To be deprecated. Only exists as setResampleMode(cudnnResampleMode_t) requires it.
inline void convert_from_cudnn_type(const cudnnResampleMode_t cudnn_mode, ResampleMode_t& mode)
{
    mode = backend::toFe(cudnn_mode).value_or(ResampleMode_t::NOT_SET);
}

inline void convert_from_cudnn_type(const cudnnBackendNormMode_t cudnn_mode, NormMode_t& mode)
{
    mode = backend::toFe(cudnn_mode).value_or(NormMode_t::NOT_SET);
}

inline void convert_from_cudnn_type(const cudnnBackendNormFwdPhase_t cudnn_mode,
                                    NormFwdPhase_t& mode)
{
    mode = backend::toFe(cudnn_mode).value_or(NormFwdPhase_t::NOT_SET);
}

inline void convert_from_cudnn_type(const cudnnBackendReshapeMode_t cudnn_mode, ReshapeMode_t& mode)
{
    mode = backend::toFe(cudnn_mode).value_or(ReshapeMode_t::NOT_SET);
}

inline cudnnStatus_t convert_to_cudnn_type(const TensorReordering_t mode,
                                           cudnnBackendTensorReordering_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline void convert_from_cudnn_type(const cudnnBackendTensorReordering_t cudnn_mode,
                                    TensorReordering_t& mode)
{
    mode = backend::toFe(cudnn_mode).value_or(TensorReordering_t::NONE);
}

inline DescriptorType_t convert_from_cudnn_type(const cudnnBackendDescriptorType_t cudnn_mode)
{
    return backend::toFe(cudnn_mode).value_or(DescriptorType_t::NOT_SET);
}

inline PointwiseMode_t convert_from_cudnn_type(const cudnnPointwiseMode_t cudnn_mode)
{
    return backend::toFe(cudnn_mode).value_or(PointwiseMode_t::NOT_SET);
}

inline DataType_t convert_from_cudnn_type(const cudnnDataType_t cudnn_mode)
{
    return backend::toFe(cudnn_mode).value_or(DataType_t::NOT_SET);
}

inline ReductionMode_t convert_from_cudnn_type(const cudnnReduceTensorOp_t cudnn_mode)
{
    return backend::toFe(cudnn_mode).value_or(ReductionMode_t::NOT_SET);
}

inline cudnnStatus_t convert_to_cudnn_type(const RngDistribution_t mode,
                                           cudnnRngDistribution_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline RngDistribution_t convert_from_cudnn_type(const cudnnRngDistribution_t cudnn_mode)
{
    return backend::toFe(cudnn_mode).value_or(RngDistribution_t::NOT_SET);
}

inline cudnnStatus_t convert_to_cudnn_type(const MoeGroupedMatmulMode_t mode,
                                           cudnnMoeGroupedMatmulMode_t& cudnn_mode)
{
    return convertToCudnnOrInvalid(mode, cudnn_mode);
}

inline MoeGroupedMatmulMode_t convert_from_cudnn_type(const cudnnMoeGroupedMatmulMode_t cudnn_mode)
{
    return backend::toFe(cudnn_mode).value_or(MoeGroupedMatmulMode_t::NOT_SET);
}

// NOLINTEND(readability-identifier-naming)

// Upstream serializes these enums with NLOHMANN_JSON_SERIALIZE_ENUM. Most are aliases of
// hipDNN's enums, which ADL never looks up in this namespace, so the shim renders the same
// strings from its own table instead. nullptr stands for JSON null; a value missing from
// upstream's table falls back to its first entry, as nlohmann does.
inline const char* enumName(PointwiseMode_t value) noexcept
{
    switch(value)
    {
    case PointwiseMode_t::ADD:
        return "ADD";
    case PointwiseMode_t::MUL:
        return "MUL";
    case PointwiseMode_t::SQRT:
        return "SQRT";
    case PointwiseMode_t::MAX:
        return "MAX";
    case PointwiseMode_t::MIN:
        return "MIN";
    case PointwiseMode_t::RELU_FWD:
        return "RELU_FWD";
    case PointwiseMode_t::TANH_FWD:
        return "TANH_FWD";
    case PointwiseMode_t::SIGMOID_FWD:
        return "SIGMOID_FWD";
    case PointwiseMode_t::ELU_FWD:
        return "ELU_FWD";
    case PointwiseMode_t::GELU_FWD:
        return "GELU_FWD";
    case PointwiseMode_t::SOFTPLUS_FWD:
        return "SOFTPLUS_FWD";
    case PointwiseMode_t::SWISH_FWD:
        return "SWISH_FWD";
    case PointwiseMode_t::RELU_BWD:
        return "RELU_BWD";
    case PointwiseMode_t::TANH_BWD:
        return "TANH_BWD";
    case PointwiseMode_t::SIGMOID_BWD:
        return "SIGMOID_BWD";
    case PointwiseMode_t::ELU_BWD:
        return "ELU_BWD";
    case PointwiseMode_t::GELU_BWD:
        return "GELU_BWD";
    case PointwiseMode_t::SOFTPLUS_BWD:
        return "SOFTPLUS_BWD";
    case PointwiseMode_t::SWISH_BWD:
        return "SWISH_BWD";
    case PointwiseMode_t::ERF:
        return "ERF";
    case PointwiseMode_t::IDENTITY:
        return "IDENTITY";
    case PointwiseMode_t::GELU_APPROX_TANH_BWD:
        return "GELU_APPROX_TANH_BWD";
    case PointwiseMode_t::GELU_APPROX_TANH_FWD:
        return "GELU_APPROX_TANH_FWD";
    case PointwiseMode_t::GEN_INDEX:
        return "GEN_INDEX";
    case PointwiseMode_t::BINARY_SELECT:
        return "BINARY_SELECT";
    case PointwiseMode_t::EXP:
        return "EXP";
    case PointwiseMode_t::LOG:
        return "LOG";
    case PointwiseMode_t::NEG:
        return "NEG";
    case PointwiseMode_t::MOD:
        return "MOD";
    case PointwiseMode_t::POW:
        return "POW";
    case PointwiseMode_t::ABS:
        return "ABS";
    case PointwiseMode_t::CEIL:
        return "CEIL";
    case PointwiseMode_t::COS:
        return "COS";
    case PointwiseMode_t::FLOOR:
        return "FLOOR";
    case PointwiseMode_t::RSQRT:
        return "RSQRT";
    case PointwiseMode_t::SIN:
        return "SIN";
    case PointwiseMode_t::LOGICAL_NOT:
        return "LOGICAL_NOT";
    case PointwiseMode_t::TAN:
        return "TAN";
    case PointwiseMode_t::SUB:
        return "SUB";
    case PointwiseMode_t::ADD_SQUARE:
        return "ADD_SQUARE";
    case PointwiseMode_t::DIV:
        return "DIV";
    case PointwiseMode_t::CMP_EQ:
        return "CMP_EQ";
    case PointwiseMode_t::CMP_NEQ:
        return "CMP_NEQ";
    case PointwiseMode_t::CMP_GT:
        return "CMP_GT";
    case PointwiseMode_t::CMP_GE:
        return "CMP_GE";
    case PointwiseMode_t::CMP_LT:
        return "CMP_LT";
    case PointwiseMode_t::CMP_LE:
        return "CMP_LE";
    case PointwiseMode_t::LOGICAL_AND:
        return "LOGICAL_AND";
    case PointwiseMode_t::LOGICAL_OR:
        return "LOGICAL_OR";
    case PointwiseMode_t::RECIPROCAL:
        return "RECIPROCAL";
    case PointwiseMode_t::NOT_SET:
    default:
        return nullptr;
    }
}

inline const char* enumName(DataType_t value) noexcept
{
    switch(value)
    {
    case DataType_t::FLOAT:
        return "FLOAT";
    case DataType_t::DOUBLE:
        return "DOUBLE";
    case DataType_t::HALF:
        return "HALF";
    case DataType_t::INT8:
        return "INT8";
    case DataType_t::INT32:
        return "INT32";
    case DataType_t::INT8x4:
        return "INT8x4";
    case DataType_t::UINT8:
        return "UINT8";
    case DataType_t::UINT8x4:
        return "UINT8x4";
    case DataType_t::INT8x32:
        return "INT8x32";
    case DataType_t::BFLOAT16:
        return "BFLOAT16";
    case DataType_t::INT64:
        return "INT64";
    case DataType_t::BOOLEAN:
        return "BOOLEAN";
    case DataType_t::FP8_E4M3:
        return "FP8_E4M3";
    case DataType_t::FP8_E5M2:
        return "FP8_E5M2";
    case DataType_t::FAST_FLOAT_FOR_FP8:
        return "FAST_FLOAT_FOR_FP8";
    case DataType_t::FP8_E8M0:
        return "FP8_E8M0";
    case DataType_t::FP4_E2M1:
        return "FP4_E2M1";
    case DataType_t::INT4:
        return "INT4";
    case DataType_t::COMPLEX_FP32:
        return "COMPLEX_FP32";
    case DataType_t::COMPLEX_FP64:
        return "COMPLEX_FP64";
    case DataType_t::NOT_SET:
    default:
        return nullptr;
    }
}

inline const char* enumName(ConvolutionMode_t value) noexcept
{
    switch(value)
    {
    case ConvolutionMode_t::CROSS_CORRELATION:
        return "CROSS_CORRELATION";
    case ConvolutionMode_t::CONVOLUTION:
    default:
        return "CONVOLUTION";
    }
}

inline const char* enumName(ReductionMode_t value) noexcept
{
    switch(value)
    {
    case ReductionMode_t::ADD:
        return "ADD";
    case ReductionMode_t::MUL:
        return "MUL";
    case ReductionMode_t::MIN:
        return "MIN";
    case ReductionMode_t::MAX:
        return "MAX";
    case ReductionMode_t::AMAX:
        return "AMAX";
    case ReductionMode_t::AVG:
        return "AVG";
    case ReductionMode_t::NORM1:
        return "NORM1";
    case ReductionMode_t::NORM2:
        return "NORM2";
    case ReductionMode_t::MUL_NO_ZEROS:
        return "MUL_NO_ZEROS";
    case ReductionMode_t::NOT_SET:
    default:
        return nullptr;
    }
}

inline const char* enumName(ResampleMode_t value) noexcept
{
    switch(value)
    {
    case ResampleMode_t::AVGPOOL_EXCLUDE_PADDING:
        return "AVGPOOL_EXCLUDE_PADDING";
    case ResampleMode_t::AVGPOOL_INCLUDE_PADDING:
        return "AVGPOOL_INCLUDE_PADDING";
    case ResampleMode_t::BILINEAR:
        return "BILINEAR";
    case ResampleMode_t::NEAREST:
        return "NEAREST";
    case ResampleMode_t::MAXPOOL:
        return "MAXPOOL";
    case ResampleMode_t::NOT_SET:
    default:
        return nullptr;
    }
}

inline const char* enumName(PaddingMode_t value) noexcept
{
    switch(value)
    {
    case PaddingMode_t::EDGE_VAL_PAD:
        return "EDGE_VAL_PAD";
    case PaddingMode_t::NEG_INF_PAD:
        return "NEG_INF_PAD";
    case PaddingMode_t::ZERO_PAD:
        return "ZERO_PAD";
    case PaddingMode_t::NOT_SET:
    default:
        return nullptr;
    }
}

inline const char* enumName(NormFwdPhase_t value) noexcept
{
    switch(value)
    {
    case NormFwdPhase_t::INFERENCE:
        return "INFERENCE";
    case NormFwdPhase_t::TRAINING:
        return "TRAINING";
    case NormFwdPhase_t::NOT_SET:
    default:
        return nullptr;
    }
}

inline const char* enumName(NormMode_t value) noexcept
{
    switch(value)
    {
    case NormMode_t::LAYER_NORM:
        return "LAYER_NORM";
    case NormMode_t::INSTANCE_NORM:
        return "INSTANCE_NORM";
    case NormMode_t::BATCH_NORM:
        return "BATCH_NORM";
    case NormMode_t::GROUP_NORM:
        return "GROUP_NORM";
    case NormMode_t::RMS_NORM:
        return "RMS_NORM";
    case NormMode_t::ADA_LAYER_NORM:
        return "ADA_LAYER_NORM";
    case NormMode_t::NOT_SET:
    default:
        return nullptr;
    }
}

inline const char* enumName(RngDistribution_t value) noexcept
{
    switch(value)
    {
    case RngDistribution_t::BERNOULLI:
        return "BERNOULLI";
    case RngDistribution_t::UNIFORM:
        return "UNIFORM";
    case RngDistribution_t::NORMAL:
        return "NORMAL";
    case RngDistribution_t::NOT_SET:
    default:
        return nullptr;
    }
}

inline const char* enumName(TensorReordering_t value) noexcept
{
    switch(value)
    {
    case TensorReordering_t::INT8x32:
        return "INT8x32";
    case TensorReordering_t::F16x16:
        return "F16x16";
    case TensorReordering_t::F8_128x4:
        return "F8_128x4";
    case TensorReordering_t::NONE:
    default:
        return "NONE";
    }
}

inline const char* enumName(ReshapeMode_t value) noexcept
{
    switch(value)
    {
    case ReshapeMode_t::VIEW_ONLY:
        return "VIEW_ONLY";
    case ReshapeMode_t::LOGICAL:
        return "LOGICAL";
    case ReshapeMode_t::NOT_SET:
    default:
        return nullptr;
    }
}

inline const char* enumName(HeurMode_t value) noexcept
{
    switch(value)
    {
    case HeurMode_t::B:
        return "B";
    case HeurMode_t::FALLBACK:
        return "FALLBACK";
    case HeurMode_t::OPENSOURCE:
        return "OPENSOURCE";
    case HeurMode_t::A:
    default:
        return "A";
    }
}

inline const char* enumName(BehaviorNote_t value) noexcept
{
    switch(value)
    {
    case BehaviorNote_t::RUNTIME_COMPILATION:
        return "RUNTIME_COMPILATION";
    case BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER:
        return "REQUIRES_FILTER_INT8x32_REORDER";
    case BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER:
        return "REQUIRES_BIAS_INT8x32_REORDER";
    case BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API:
        return "SUPPORTS_CUDA_GRAPH_NATIVE_API";
    case BehaviorNote_t::CUBLASLT_DEPENDENCY:
        return "CUBLASLT_DEPENDENCY";
    case BehaviorNote_t::NOT_SET:
    default:
        return "NOT_SET";
    }
}

inline const char* enumName(NumericalNote_t value) noexcept
{
    switch(value)
    {
    case NumericalNote_t::TENSOR_CORE:
        return "TENSOR_CORE";
    case NumericalNote_t::DOWN_CONVERT_INPUTS:
        return "DOWN_CONVERT_INPUTS";
    case NumericalNote_t::REDUCED_PRECISION_REDUCTION:
        return "REDUCED_PRECISION_REDUCTION";
    case NumericalNote_t::FFT:
        return "FFT";
    case NumericalNote_t::NONDETERMINISTIC:
        return "NONDETERMINISTIC";
    case NumericalNote_t::WINOGRAD:
        return "WINOGRAD";
    case NumericalNote_t::WINOGRAD_TILE_4x4:
        return "WINOGRAD_TILE_4x4";
    case NumericalNote_t::WINOGRAD_TILE_6x6:
        return "WINOGRAD_TILE_6x6";
    case NumericalNote_t::WINOGRAD_TILE_13x13:
        return "WINOGRAD_TILE_13x13";
    case NumericalNote_t::STRICT_NAN_PROP:
        return "STRICT_NAN_PROP";
    case NumericalNote_t::NOT_SET:
    default:
        return "NOT_SET";
    }
}

inline const char* enumName(BuildPlanPolicy_t value) noexcept
{
    switch(value)
    {
    case BuildPlanPolicy_t::ALL:
        return "ALL";
    case BuildPlanPolicy_t::HEURISTICS_CHOICE:
    default:
        return "HEURISTICS_CHOICE";
    }
}

inline const char* enumName(DiagonalAlignment_t value) noexcept
{
    switch(value)
    {
    case DiagonalAlignment_t::BOTTOM_RIGHT:
        return "BOTTOM_RIGHT";
    case DiagonalAlignment_t::TOP_LEFT:
    default:
        return "TOP_LEFT";
    }
}

inline const char* enumName(AttentionImplementation_t value) noexcept
{
    switch(value)
    {
    case AttentionImplementation_t::COMPOSITE:
        return "COMPOSITE";
    case AttentionImplementation_t::UNIFIED:
        return "UNIFIED";
    case AttentionImplementation_t::AUTO:
    default:
        return "AUTO";
    }
}

inline const char* enumName(MoeGroupedMatmulMode_t value) noexcept
{
    switch(value)
    {
    case MoeGroupedMatmulMode_t::NONE:
        return "NONE";
    case MoeGroupedMatmulMode_t::GATHER:
        return "GATHER";
    case MoeGroupedMatmulMode_t::SCATTER:
        return "SCATTER";
    case MoeGroupedMatmulMode_t::NOT_SET:
    default:
        return nullptr;
    }
}

// The exact text upstream's nlohmann::json{value}.dump() produces.
template <typename Enum, typename = decltype(enumName(std::declval<Enum>()))>
std::string enumJson(const Enum value)
{
    const char* name = enumName(value);
    return name == nullptr ? std::string("[null]") : "[\"" + std::string(name) + "\"]";
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
