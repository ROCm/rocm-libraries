// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/mx.hpp>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace roc::host_validation::python_bindings {
void registerMxBindings(nb::module_& module) {
    nb::enum_<MxGenerationMode>(module, "MxGenerationMode")
        .value("Bounded", MxGenerationMode::Bounded)
        .value("BoundedAlternatingSign", MxGenerationMode::BoundedAlternatingSign)
        .value("Unbounded", MxGenerationMode::Unbounded)
        .value("Identity", MxGenerationMode::Identity)
        .value("Ones", MxGenerationMode::Ones)
        .value("Zeros", MxGenerationMode::Zeros)
        .value("Sequential", MxGenerationMode::Sequential)
        .value("RowIndex", MxGenerationMode::RowIndex)
        .value("ColumnIndex", MxGenerationMode::ColumnIndex)
        .value("Checkerboard", MxGenerationMode::Checkerboard)
        .value("ScaledDiagonal", MxGenerationMode::ScaledDiagonal)
        .value("Twos", MxGenerationMode::Twos)
        .value("NegativeOnes", MxGenerationMode::NegativeOnes)
        .value("Maximum", MxGenerationMode::Maximum)
        .value("DenormalMinimum", MxGenerationMode::DenormalMinimum)
        .value("DenormalMaximum", MxGenerationMode::DenormalMaximum)
        .value("NaN", MxGenerationMode::NaN)
        .value("Infinity", MxGenerationMode::Infinity)
        .value("Trigonometric", MxGenerationMode::Trigonometric)
        .value("Normal", MxGenerationMode::Normal)
        .value("UniformInteger", MxGenerationMode::UniformInteger);

    nb::class_<MxGenerationRecipe>(module, "MxGenerationRecipe")
        .def(nb::init<>())
        .def_rw("mode", &MxGenerationRecipe::mode)
        .def_rw("parameter0", &MxGenerationRecipe::parameter0)
        .def_rw("parameter1", &MxGenerationRecipe::parameter1);

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

    module.def("generate_mx", &generateMx, "problem"_a);
}
}  // namespace roc::host_validation::python_bindings
