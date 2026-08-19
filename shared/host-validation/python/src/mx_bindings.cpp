// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/mx.hpp>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace roc::host_validation::python_bindings {
void registerMxBindings(nb::module_& module) {
    nb::class_<MxBoundedDataParameters>(module, "MxBoundedDataParameters")
        .def(nb::init<double, double>(), "lower"_a = -1.0, "upper"_a = 1.0)
        .def_rw("lower", &MxBoundedDataParameters::lower)
        .def_rw("upper", &MxBoundedDataParameters::upper);

    nb::class_<MxAlternatingSignDataParameters>(module, "MxAlternatingSignDataParameters")
        .def(nb::init<double>(), "maximum_magnitude"_a = 1.0)
        .def_rw("maximum_magnitude", &MxAlternatingSignDataParameters::maximumMagnitude);

    nb::class_<MxNormalDataParameters>(module, "MxNormalDataParameters")
        .def(nb::init<double, double>(), "mean"_a = 0.0, "standard_deviation"_a = 1.0)
        .def_rw("mean", &MxNormalDataParameters::mean)
        .def_rw("standard_deviation", &MxNormalDataParameters::standardDeviation);

    nb::class_<MxUniformIntegerDataParameters>(module, "MxUniformIntegerDataParameters")
        .def(nb::init<int, int>(), "lower"_a, "upper"_a)
        .def_rw("lower", &MxUniformIntegerDataParameters::lower)
        .def_rw("upper", &MxUniformIntegerDataParameters::upper);

    nb::class_<MxDataRecipe>(module, "MxDataRecipe")
        .def_static("bounded", &MxDataRecipe::bounded, "parameters"_a = MxBoundedDataParameters{})
        .def_static("bounded_alternating_sign", &MxDataRecipe::boundedAlternatingSign,
                    "parameters"_a = MxAlternatingSignDataParameters{})
        .def_static("unbounded", &MxDataRecipe::unbounded)
        .def_static("identity", &MxDataRecipe::identity)
        .def_static("constant", &MxDataRecipe::constant, "value"_a)
        .def_static("sequential", &MxDataRecipe::sequential)
        .def_static("row_index", &MxDataRecipe::rowIndex)
        .def_static("column_index", &MxDataRecipe::columnIndex)
        .def_static("checkerboard", &MxDataRecipe::checkerboard)
        .def_static("scaled_diagonal", &MxDataRecipe::scaledDiagonal)
        .def_static("type_maximum", &MxDataRecipe::typeMaximum)
        .def_static("type_denormal_minimum", &MxDataRecipe::typeDenormalMinimum)
        .def_static("type_denormal_maximum", &MxDataRecipe::typeDenormalMaximum)
        .def_static("type_nan", &MxDataRecipe::typeNaN)
        .def_static("type_infinity", &MxDataRecipe::typeInfinity)
        .def_static("trigonometric", &MxDataRecipe::trigonometric)
        .def_static("normal", &MxDataRecipe::normal, "parameters"_a = MxNormalDataParameters{})
        .def_static("uniform_integer", &MxDataRecipe::uniformInteger, "parameters"_a);

    nb::enum_<MxScaleGenerationMode>(module, "MxScaleGenerationMode")
        .value("Derived", MxScaleGenerationMode::Derived)
        .value("Minimum", MxScaleGenerationMode::Minimum)
        .value("One", MxScaleGenerationMode::One)
        .value("Two", MxScaleGenerationMode::Two)
        .value("Maximum", MxScaleGenerationMode::Maximum)
        .value("NaN", MxScaleGenerationMode::NaN);

    nb::class_<MxGenerationProblem>(module, "MxGenerationProblem")
        .def(nb::init<>())
        .def_rw("data_type", &MxGenerationProblem::dataType)
        .def_rw("scale_type", &MxGenerationProblem::scaleType)
        .def_rw("shape", &MxGenerationProblem::shape)
        .def_rw("leading_dimension", &MxGenerationProblem::leadingDimension)
        .def_rw("block_axis", &MxGenerationProblem::blockAxis)
        .def_rw("block_size", &MxGenerationProblem::blockSize)
        .def_rw("data", &MxGenerationProblem::data)
        .def_rw("scale", &MxGenerationProblem::scale)
        .def_rw("seed", &MxGenerationProblem::seed);

    nb::class_<MxGenerationResult>(module, "MxGenerationResult")
        .def_ro("data", &MxGenerationResult::data)
        .def_ro("scales", &MxGenerationResult::scales)
        .def_ro("scale_indices", &MxGenerationResult::scaleIndices)
        .def_ro("reference", &MxGenerationResult::reference);

    module.def("generate_mx",
               static_cast<MxGenerationResult (*)(const MxGenerationProblem&)>(&generateMx),
               "problem"_a);
}
}  // namespace roc::host_validation::python_bindings
