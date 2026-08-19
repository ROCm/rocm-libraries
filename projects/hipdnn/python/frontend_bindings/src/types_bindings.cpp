// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <HipdnnBackendPluginLoadingMode.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace hipdnn_frontend;

void typesBindings(nb::module_& m)
{
    // Bind DataType enum
    nb::enum_<DataType>(m, "DataType")
        .value("NOT_SET", DataType::NOT_SET)
        .value("FLOAT", DataType::FLOAT)
        .value("HALF", DataType::HALF)
        .value("BFLOAT16", DataType::BFLOAT16)
        .value("DOUBLE", DataType::DOUBLE)
        .value("UINT8", DataType::UINT8)
        .value("INT32", DataType::INT32)
        .value("INT8", DataType::INT8)
        .value("FP8_E4M3", DataType::FP8_E4M3)
        .value("FP8_E5M2", DataType::FP8_E5M2)
        .value("FP8_E8M0", DataType::FP8_E8M0)
        .value("FP4_E2M1", DataType::FP4_E2M1)
        .value("INT4", DataType::INT4)
        .value("FP6_E2M3", DataType::FP6_E2M3)
        .value("FP6_E3M2", DataType::FP6_E3M2)
        .value("INT64", DataType::INT64)
        .value("BOOLEAN", DataType::BOOLEAN)
        .value("FP8_E4M3_FNUZ", DataType::FP8_E4M3_FNUZ)
        .value("FP8_E5M2_FNUZ", DataType::FP8_E5M2_FNUZ);

    // Bind ConvolutionMode enum
    nb::enum_<ConvolutionMode>(m, "ConvolutionMode")
        .value("NOT_SET", ConvolutionMode::NOT_SET)
        .value("CROSS_CORRELATION", ConvolutionMode::CROSS_CORRELATION)
        .value("CONVOLUTION", ConvolutionMode::CONVOLUTION);

    // Bind PointwiseMode enum
    nb::enum_<PointwiseMode>(m, "PointwiseMode")
        .value("NOT_SET", PointwiseMode::NOT_SET)
        .value("ABS", PointwiseMode::ABS)
        .value("ADD", PointwiseMode::ADD)
        .value("ADD_SQUARE", PointwiseMode::ADD_SQUARE)
        .value("BINARY_SELECT", PointwiseMode::BINARY_SELECT)
        .value("CEIL", PointwiseMode::CEIL)
        .value("CMP_EQ", PointwiseMode::CMP_EQ)
        .value("CMP_GE", PointwiseMode::CMP_GE)
        .value("CMP_GT", PointwiseMode::CMP_GT)
        .value("CMP_LE", PointwiseMode::CMP_LE)
        .value("CMP_LT", PointwiseMode::CMP_LT)
        .value("CMP_NEQ", PointwiseMode::CMP_NEQ)
        .value("DIV", PointwiseMode::DIV)
        .value("ELU_BWD", PointwiseMode::ELU_BWD)
        .value("ELU_FWD", PointwiseMode::ELU_FWD)
        .value("ERF", PointwiseMode::ERF)
        .value("EXP", PointwiseMode::EXP)
        .value("FLOOR", PointwiseMode::FLOOR)
        .value("GELU_APPROX_TANH_BWD", PointwiseMode::GELU_APPROX_TANH_BWD)
        .value("GELU_APPROX_TANH_FWD", PointwiseMode::GELU_APPROX_TANH_FWD)
        .value("GELU_BWD", PointwiseMode::GELU_BWD)
        .value("GELU_FWD", PointwiseMode::GELU_FWD)
        .value("GEN_INDEX", PointwiseMode::GEN_INDEX)
        .value("IDENTITY", PointwiseMode::IDENTITY)
        .value("LOG", PointwiseMode::LOG)
        .value("LOGICAL_AND", PointwiseMode::LOGICAL_AND)
        .value("LOGICAL_NOT", PointwiseMode::LOGICAL_NOT)
        .value("LOGICAL_OR", PointwiseMode::LOGICAL_OR)
        .value("MAX", PointwiseMode::MAX)
        .value("MIN", PointwiseMode::MIN)
        .value("MUL", PointwiseMode::MUL)
        .value("NEG", PointwiseMode::NEG)
        .value("RECIPROCAL", PointwiseMode::RECIPROCAL)
        .value("RELU_BWD", PointwiseMode::RELU_BWD)
        .value("RELU_FWD", PointwiseMode::RELU_FWD)
        .value("RSQRT", PointwiseMode::RSQRT)
        .value("SIGMOID_BWD", PointwiseMode::SIGMOID_BWD)
        .value("SIGMOID_FWD", PointwiseMode::SIGMOID_FWD)
        .value("SIN", PointwiseMode::SIN)
        .value("SOFTPLUS_BWD", PointwiseMode::SOFTPLUS_BWD)
        .value("SOFTPLUS_FWD", PointwiseMode::SOFTPLUS_FWD)
        .value("SQRT", PointwiseMode::SQRT)
        .value("SUB", PointwiseMode::SUB)
        .value("SWISH_BWD", PointwiseMode::SWISH_BWD)
        .value("SWISH_FWD", PointwiseMode::SWISH_FWD)
        .value("TAN", PointwiseMode::TAN)
        .value("TANH_BWD", PointwiseMode::TANH_BWD)
        .value("TANH_FWD", PointwiseMode::TANH_FWD);

    nb::enum_<ReductionMode>(m, "ReductionMode")
        .value("NOT_SET", ReductionMode::NOT_SET)
        .value("ADD", ReductionMode::ADD)
        .value("MUL", ReductionMode::MUL)
        .value("MIN", ReductionMode::MIN)
        .value("MAX", ReductionMode::MAX)
        .value("AMAX", ReductionMode::AMAX)
        .value("AVG", ReductionMode::AVG)
        .value("NORM1", ReductionMode::NORM1)
        .value("NORM2", ReductionMode::NORM2)
        .value("MUL_NO_ZEROS", ReductionMode::MUL_NO_ZEROS);

    nb::enum_<ResampleMode>(m, "ResampleMode")
        .value("NOT_SET", ResampleMode::NOT_SET)
        .value("MAXPOOL", ResampleMode::MAXPOOL)
        .value("AVGPOOL_EXCLUDE_PADDING", ResampleMode::AVGPOOL_EXCLUDE_PADDING)
        .value("AVGPOOL_INCLUDE_PADDING", ResampleMode::AVGPOOL_INCLUDE_PADDING)
        .value("BILINEAR", ResampleMode::BILINEAR)
        .value("NEAREST", ResampleMode::NEAREST);

    nb::enum_<PaddingMode>(m, "PaddingMode")
        .value("NOT_SET", PaddingMode::NOT_SET)
        .value("NEG_INF_PAD", PaddingMode::NEG_INF_PAD)
        .value("ZERO_PAD", PaddingMode::ZERO_PAD)
        .value("EDGE_VAL_PAD", PaddingMode::EDGE_VAL_PAD);

    nb::enum_<DiagonalAlignment>(m, "DiagonalAlignment")
        .value("TOP_LEFT", DiagonalAlignment::TOP_LEFT)
        .value("BOTTOM_RIGHT", DiagonalAlignment::BOTTOM_RIGHT);

    nb::enum_<AttentionImplementation>(m, "AttentionImplementation")
        .value("AUTO", AttentionImplementation::AUTO)
        .value("COMPOSITE", AttentionImplementation::COMPOSITE)
        .value("UNIFIED", AttentionImplementation::UNIFIED);

    nb::enum_<MoeGroupedMatmulMode>(m, "MoeGroupedMatmulMode")
        .value("NOT_SET", MoeGroupedMatmulMode::NOT_SET)
        .value("NONE", MoeGroupedMatmulMode::NONE)
        .value("GATHER", MoeGroupedMatmulMode::GATHER)
        .value("SCATTER", MoeGroupedMatmulMode::SCATTER);

    nb::enum_<NormFwdPhase>(m, "NormFwdPhase")
        .value("NOT_SET", NormFwdPhase::NOT_SET)
        .value("INFERENCE", NormFwdPhase::INFERENCE)
        .value("TRAINING", NormFwdPhase::TRAINING);

    // Bind HeuristicMode enum
    nb::enum_<HeuristicMode>(m, "HeuristicMode").value("FALLBACK", HeuristicMode::FALLBACK);

    // Bind BehaviorNote enum
    nb::enum_<BehaviorNote>(m, "BehaviorNote")
        .value("RUNTIME_COMPILATION", BehaviorNote::RUNTIME_COMPILATION)
        .value("REQUIRES_LAYOUT_TRANSFORM", BehaviorNote::REQUIRES_LAYOUT_TRANSFORM)
        .value("SUPPORTS_GRAPH_CAPTURE", BehaviorNote::SUPPORTS_GRAPH_CAPTURE)
        .value("EXTERNAL_LIBRARY_DEPENDENCY", BehaviorNote::EXTERNAL_LIBRARY_DEPENDENCY)
        .value("SUPPORTS_EXECUTION_PLAN_SERIALIZATION",
               BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION);

    // Bind BuildPlanPolicy enum
    nb::enum_<BuildPlanPolicy>(m, "BuildPlanPolicy")
        .value("HEURISTICS_CHOICE", BuildPlanPolicy::HEURISTICS_CHOICE)
        .value("ALL", BuildPlanPolicy::ALL);

    // Bind KnobValueType enum
    nb::enum_<KnobValueType>(m, "KnobValueType")
        .value("NOT_SET", KnobValueType::NOT_SET)
        .value("INT64", KnobValueType::INT64)
        .value("FLOAT64", KnobValueType::FLOAT64)
        .value("STRING", KnobValueType::STRING);

    // Bind ErrorCode enum
    nb::enum_<ErrorCode>(m, "ErrorCode")
        .value("OK", ErrorCode::OK)
        .value("INVALID_VALUE", ErrorCode::INVALID_VALUE)
        .value("HIPDNN_BACKEND_ERROR", ErrorCode::HIPDNN_BACKEND_ERROR)
        .value("ATTRIBUTE_NOT_SET", ErrorCode::ATTRIBUTE_NOT_SET)
        .value("GRAPH_NOT_SUPPORTED", ErrorCode::GRAPH_NOT_SUPPORTED)
        .value("SHAPE_DEDUCTION_FAILED", ErrorCode::SHAPE_DEDUCTION_FAILED)
        .value("INVALID_TENSOR_NAME", ErrorCode::INVALID_TENSOR_NAME)
        .value("INVALID_VARIANT_PACK", ErrorCode::INVALID_VARIANT_PACK)
        .value("GRAPH_EXECUTION_PLAN_CREATION_FAILED",
               ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED)
        .value("GRAPH_EXECUTION_FAILED", ErrorCode::GRAPH_EXECUTION_FAILED)
        .value("HEURISTIC_QUERY_FAILED", ErrorCode::HEURISTIC_QUERY_FAILED)
        .value("UNSUPPORTED_GRAPH_FORMAT", ErrorCode::UNSUPPORTED_GRAPH_FORMAT)
        .value("CUDA_API_FAILED", ErrorCode::CUDA_API_FAILED)
        .value("CUDNN_BACKEND_API_FAILED", ErrorCode::CUDNN_BACKEND_API_FAILED)
        .value("INVALID_CUDA_DEVICE", ErrorCode::INVALID_CUDA_DEVICE)
        .value("HANDLE_ERROR", ErrorCode::HANDLE_ERROR)
        .value("NVRTC_COMPILATION_FAILED", ErrorCode::NVRTC_COMPILATION_FAILED);

    // Bind Error struct
    nb::class_<Error>(m, "Error")
        .def(nb::init<>())
        .def(nb::init<ErrorCode, std::string>())
        .def_rw("code", &Error::code)
        .def_rw("err_msg", &Error::err_msg)
        .def("get_message", &Error::get_message)
        .def("get_code", &Error::get_code)
        .def("is_good", &Error::is_good)
        .def("is_bad", &Error::is_bad);

    // Bind PluginLoadingMode enum
    nb::enum_<hipdnnPluginLoadingMode_ext_t>(m, "PluginLoadingMode")
        .value("ADDITIVE", HIPDNN_PLUGIN_LOADING_ADDITIVE)
        .value("ABSOLUTE", HIPDNN_PLUGIN_LOADING_ABSOLUTE);
}
