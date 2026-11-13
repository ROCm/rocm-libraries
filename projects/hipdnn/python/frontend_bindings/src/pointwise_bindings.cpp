// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace hipdnn_frontend;

namespace
{

using TensorPtr = std::shared_ptr<graph::TensorAttributes>;

graph::PointwiseAttributes
    pointwiseAttributes(PointwiseMode mode, DataType computeDataType, const std::string& name)
{
    graph::PointwiseAttributes attributes;
    attributes.set_mode(mode).set_compute_data_type(computeDataType).set_name(name);
    return attributes;
}

graph::PointwiseAttributes reluAttributes(PointwiseMode mode,
                                          std::optional<float> negativeSlope,
                                          std::optional<float> lowerClip,
                                          std::optional<float> upperClip,
                                          DataType computeDataType,
                                          const std::string& name)
{
    auto attributes = pointwiseAttributes(mode, computeDataType, name);
    if(negativeSlope)
    {
        attributes.set_relu_lower_clip_slope(*negativeSlope);
    }
    if(lowerClip)
    {
        attributes.set_relu_lower_clip(*lowerClip);
    }
    if(upperClip)
    {
        attributes.set_relu_upper_clip(*upperClip);
    }
    return attributes;
}

graph::PointwiseAttributes swishAttributes(PointwiseMode mode,
                                           std::optional<float> swishBeta,
                                           DataType computeDataType,
                                           const std::string& name)
{
    auto attributes = pointwiseAttributes(mode, computeDataType, name);
    if(swishBeta)
    {
        attributes.set_swish_beta(*swishBeta);
    }
    return attributes;
}

template <PointwiseMode Mode>
TensorPtr pointwiseUnary(graph::Graph& self,
                         TensorPtr input,
                         DataType computeDataType,
                         const std::string& name)
{
    return self.pointwise(std::move(input), pointwiseAttributes(Mode, computeDataType, name));
}

template <PointwiseMode Mode>
TensorPtr pointwiseBinary(graph::Graph& self,
                          TensorPtr in0,
                          TensorPtr in1,
                          DataType computeDataType,
                          const std::string& name)
{
    return self.pointwise(
        std::move(in0), std::move(in1), pointwiseAttributes(Mode, computeDataType, name));
}

template <PointwiseMode Mode>
TensorPtr pointwiseTernary(graph::Graph& self,
                           TensorPtr in0,
                           TensorPtr in1,
                           TensorPtr in2,
                           DataType computeDataType,
                           const std::string& name)
{
    return self.pointwise(std::move(in0),
                          std::move(in1),
                          std::move(in2),
                          pointwiseAttributes(Mode, computeDataType, name));
}

template <PointwiseMode Mode>
void defUnary(nb::class_<graph::Graph>& cls, const char* method, const char* doc)
{
    cls.def(method,
            &pointwiseUnary<Mode>,
            nb::arg("input"),
            nb::arg("compute_data_type") = DataType::NOT_SET,
            nb::arg("name") = "",
            doc);
}

// Argument names are per-method because cuDNN's differ (a/b, input/bias, loss/input, ...).
template <PointwiseMode Mode>
void defBinary(nb::class_<graph::Graph>& cls,
               const char* method,
               const char* arg0,
               const char* arg1,
               const char* doc)
{
    cls.def(method,
            &pointwiseBinary<Mode>,
            nb::arg(arg0),
            nb::arg(arg1),
            nb::arg("compute_data_type") = DataType::NOT_SET,
            nb::arg("name") = "",
            doc);
}

} // namespace

// Named pointwise methods with cuDNN's Python names, parameter names, order and defaults.
// cuDNN's mod, pow and cos are absent: hipDNN has no backend mode for them.
void pointwiseBindings(nb::class_<graph::Graph>& cls)
{
    defBinary<PointwiseMode::ADD>(cls, "add", "a", "b", "Elementwise a + b.");
    defBinary<PointwiseMode::ADD>(cls, "bias", "input", "bias", "Add bias to input.");
    defBinary<PointwiseMode::MUL>(cls, "mul", "a", "b", "Elementwise a * b.");
    defBinary<PointwiseMode::MUL>(cls, "scale", "input", "scale", "Multiply input by scale.");
    defUnary<PointwiseMode::SQRT>(cls, "sqrt", "Elementwise square root.");
    defBinary<PointwiseMode::MAX>(cls, "max", "input0", "input1", "Elementwise maximum.");
    defBinary<PointwiseMode::MIN>(cls, "min", "input0", "input1", "Elementwise minimum.");

    cls.def(
        "gen_index",
        [](graph::Graph& self,
           TensorPtr input,
           int64_t axis,
           DataType computeDataType,
           const std::string& name) {
            auto attributes = pointwiseAttributes(PointwiseMode::GEN_INDEX, computeDataType, name);
            attributes.set_axis(axis);
            return self.pointwise(std::move(input), std::move(attributes));
        },
        nb::arg("input"),
        nb::arg("axis"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "Index of each element of input along axis.");

    cls.def(
        "relu",
        [](graph::Graph& self,
           TensorPtr input,
           std::optional<float> negativeSlope,
           std::optional<float> lowerClip,
           std::optional<float> upperClip,
           DataType computeDataType,
           const std::string& name) {
            return self.pointwise(std::move(input),
                                  reluAttributes(PointwiseMode::RELU_FWD,
                                                 negativeSlope,
                                                 lowerClip,
                                                 upperClip,
                                                 computeDataType,
                                                 name));
        },
        nb::arg("input"),
        nb::arg("negative_slope") = nb::none(),
        nb::arg("lower_clip") = nb::none(),
        nb::arg("upper_clip") = nb::none(),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "ReLU forward; negative_slope, lower_clip and upper_clip are optional.");
    cls.def(
        "leaky_relu",
        [](graph::Graph& self,
           TensorPtr input,
           float negativeSlope,
           DataType computeDataType,
           const std::string& name) {
            return self.pointwise(
                std::move(input),
                reluAttributes(
                    PointwiseMode::RELU_FWD, negativeSlope, {}, {}, computeDataType, name));
        },
        nb::arg("input"),
        nb::arg("negative_slope"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "Leaky ReLU forward (ReLU with a lower-clip slope).");
    defUnary<PointwiseMode::TANH_FWD>(cls, "tanh", "Tanh forward.");
    defUnary<PointwiseMode::ELU_FWD>(cls, "elu", "ELU forward.");
    defUnary<PointwiseMode::GELU_FWD>(cls, "gelu", "GELU forward.");
    defUnary<PointwiseMode::SIGMOID_FWD>(cls, "sigmoid", "Sigmoid forward.");
    cls.def(
        "swish",
        [](graph::Graph& self,
           TensorPtr input,
           std::optional<float> swishBeta,
           DataType computeDataType,
           const std::string& name) {
            return self.pointwise(
                std::move(input),
                swishAttributes(PointwiseMode::SWISH_FWD, swishBeta, computeDataType, name));
        },
        nb::arg("input"),
        nb::arg("swish_beta") = nb::none(),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "Swish forward; swish_beta is optional.");
    defUnary<PointwiseMode::SOFTPLUS_FWD>(cls, "softplus", "Softplus forward.");
    defUnary<PointwiseMode::GELU_APPROX_TANH_FWD>(
        cls, "gelu_approx_tanh", "GELU forward, tanh approximation.");

    cls.def(
        "relu_backward",
        [](graph::Graph& self,
           TensorPtr loss,
           TensorPtr input,
           std::optional<float> negativeSlope,
           std::optional<float> lowerClip,
           std::optional<float> upperClip,
           DataType computeDataType,
           const std::string& name) {
            return self.pointwise(std::move(loss),
                                  std::move(input),
                                  reluAttributes(PointwiseMode::RELU_BWD,
                                                 negativeSlope,
                                                 lowerClip,
                                                 upperClip,
                                                 computeDataType,
                                                 name));
        },
        nb::arg("loss"),
        nb::arg("input"),
        nb::arg("negative_slope") = nb::none(),
        nb::arg("lower_clip") = nb::none(),
        nb::arg("upper_clip") = nb::none(),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "ReLU backward; negative_slope, lower_clip and upper_clip are optional.");
    cls.def(
        "leaky_relu_backward",
        [](graph::Graph& self,
           TensorPtr loss,
           TensorPtr input,
           float negativeSlope,
           DataType computeDataType,
           const std::string& name) {
            return self.pointwise(
                std::move(loss),
                std::move(input),
                reluAttributes(
                    PointwiseMode::RELU_BWD, negativeSlope, {}, {}, computeDataType, name));
        },
        nb::arg("loss"),
        nb::arg("input"),
        nb::arg("negative_slope"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("name") = "",
        "Leaky ReLU backward (ReLU backward with a lower-clip slope).");
    defBinary<PointwiseMode::TANH_BWD>(cls, "tanh_backward", "loss", "input", "Tanh backward.");
    defBinary<PointwiseMode::SIGMOID_BWD>(
        cls, "sigmoid_backward", "loss", "input", "Sigmoid backward.");
    defBinary<PointwiseMode::ELU_BWD>(cls, "elu_backward", "loss", "input", "ELU backward.");
    defBinary<PointwiseMode::GELU_BWD>(cls, "gelu_backward", "loss", "input", "GELU backward.");
    defBinary<PointwiseMode::SOFTPLUS_BWD>(
        cls, "softplus_backward", "loss", "input", "Softplus backward.");
    // cuDNN places compute_data_type before swish_beta here, unlike swish.
    cls.def(
        "swish_backward",
        [](graph::Graph& self,
           TensorPtr loss,
           TensorPtr input,
           DataType computeDataType,
           std::optional<float> swishBeta,
           const std::string& name) {
            return self.pointwise(
                std::move(loss),
                std::move(input),
                swishAttributes(PointwiseMode::SWISH_BWD, swishBeta, computeDataType, name));
        },
        nb::arg("loss"),
        nb::arg("input"),
        nb::arg("compute_data_type") = DataType::NOT_SET,
        nb::arg("swish_beta") = nb::none(),
        nb::arg("name") = "",
        "Swish backward; swish_beta is optional.");
    defBinary<PointwiseMode::GELU_APPROX_TANH_BWD>(
        cls, "gelu_approx_tanh_backward", "loss", "input", "GELU backward, tanh approximation.");

    defUnary<PointwiseMode::ERF>(cls, "erf", "Elementwise error function.");
    defUnary<PointwiseMode::IDENTITY>(cls, "identity", "Elementwise identity.");
    defUnary<PointwiseMode::EXP>(cls, "exp", "Elementwise exponential.");
    defUnary<PointwiseMode::LOG>(cls, "log", "Elementwise natural logarithm.");
    defUnary<PointwiseMode::NEG>(cls, "neg", "Elementwise negation.");
    defUnary<PointwiseMode::ABS>(cls, "abs", "Elementwise absolute value.");
    defUnary<PointwiseMode::CEIL>(cls, "ceil", "Elementwise ceiling.");
    defUnary<PointwiseMode::FLOOR>(cls, "floor", "Elementwise floor.");
    defUnary<PointwiseMode::RSQRT>(cls, "rsqrt", "Elementwise reciprocal square root.");
    defUnary<PointwiseMode::RECIPROCAL>(cls, "reciprocal", "Elementwise reciprocal.");
    defUnary<PointwiseMode::SIN>(cls, "sin", "Elementwise sine.");
    defUnary<PointwiseMode::TAN>(cls, "tan", "Elementwise tangent.");
    defUnary<PointwiseMode::LOGICAL_NOT>(cls, "logical_not", "Elementwise logical NOT.");
    defBinary<PointwiseMode::LOGICAL_AND>(cls, "logical_and", "a", "b", "Elementwise logical AND.");
    defBinary<PointwiseMode::LOGICAL_OR>(cls, "logical_or", "a", "b", "Elementwise logical OR.");
    defBinary<PointwiseMode::SUB>(cls, "sub", "a", "b", "Elementwise a - b.");
    defBinary<PointwiseMode::DIV>(cls, "div", "a", "b", "Elementwise a / b.");
    defBinary<PointwiseMode::ADD_SQUARE>(cls, "add_square", "a", "b", "Elementwise a + b * b.");
    defBinary<PointwiseMode::CMP_EQ>(
        cls, "cmp_eq", "input", "comparison", "Elementwise input == comparison.");
    defBinary<PointwiseMode::CMP_NEQ>(
        cls, "cmp_neq", "input", "comparison", "Elementwise input != comparison.");
    defBinary<PointwiseMode::CMP_GT>(
        cls, "cmp_gt", "input", "comparison", "Elementwise input > comparison.");
    defBinary<PointwiseMode::CMP_GE>(
        cls, "cmp_ge", "input", "comparison", "Elementwise input >= comparison.");
    defBinary<PointwiseMode::CMP_LT>(
        cls, "cmp_lt", "input", "comparison", "Elementwise input < comparison.");
    defBinary<PointwiseMode::CMP_LE>(
        cls, "cmp_le", "input", "comparison", "Elementwise input <= comparison.");

    cls.def("binary_select",
            &pointwiseTernary<PointwiseMode::BINARY_SELECT>,
            nb::arg("input0"),
            nb::arg("input1"),
            nb::arg("mask"),
            nb::arg("compute_data_type") = DataType::NOT_SET,
            nb::arg("name") = "",
            "Select input0 where mask is true, else input1.");
}
