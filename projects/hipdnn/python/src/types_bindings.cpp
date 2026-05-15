// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <HipdnnBackendPluginLoadingMode.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace hipdnn_frontend;

void types_bindings(nb::module_& m)
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

    // Bind HeuristicMode enum
    nb::enum_<HeuristicMode>(m, "HeuristicMode").value("FALLBACK", HeuristicMode::FALLBACK);

    // Bind ErrorCode enum
    nb::enum_<ErrorCode>(m, "ErrorCode")
        .value("OK", ErrorCode::OK)
        .value("INVALID_VALUE", ErrorCode::INVALID_VALUE)
        .value("HIPDNN_BACKEND_ERROR", ErrorCode::HIPDNN_BACKEND_ERROR)
        .value("ATTRIBUTE_NOT_SET", ErrorCode::ATTRIBUTE_NOT_SET);

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
