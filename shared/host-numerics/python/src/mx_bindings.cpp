// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/stl/optional.h>

#include <roc/host_numerics/mx.hpp>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace roc::host_numerics::python_bindings {
void registerMxBindings(nb::module_& module) {
    nb::enum_<MxDataQuantization>(module, "MxDataQuantization")
        .value("Nearest", MxDataQuantization::Nearest)
        .value("PreserveRange", MxDataQuantization::PreserveRange)
        .value("PreserveGeneratedEncoding", MxDataQuantization::PreserveGeneratedEncoding);

    nb::class_<MxRepresentedValueRange>(module, "MxRepresentedValueRange")
        .def(nb::init<double, double>(), "lower"_a = -1.0, "upper"_a = 1.0)
        .def_rw("lower", &MxRepresentedValueRange::lower)
        .def_rw("upper", &MxRepresentedValueRange::upper);

    nb::class_<MxDataGeneration>(module, "MxDataGeneration")
        .def_static("quantize", &MxDataGeneration::quantize, "recipe"_a)
        .def_static("preserve_range", &MxDataGeneration::preserveRange, "recipe"_a,
                    "represented_value_range"_a)
        .def_static("preserve_generated_encoding", &MxDataGeneration::preserveGeneratedEncoding,
                    "recipe"_a)
        .def_prop_ro("recipe", &MxDataGeneration::recipe, nb::rv_policy::reference_internal)
        .def_prop_ro("quantization", &MxDataGeneration::quantization)
        .def_prop_ro("represented_value_range", &MxDataGeneration::representedValueRange)
        .def("with_seed", &MxDataGeneration::withSeed, "seed"_a);

    nb::enum_<MxScaleGenerationMode>(module, "MxScaleGenerationMode")
        .value("Derived", MxScaleGenerationMode::Derived)
        .value("RandomFinite", MxScaleGenerationMode::RandomFinite)
        .value("Minimum", MxScaleGenerationMode::Minimum)
        .value("One", MxScaleGenerationMode::One)
        .value("Two", MxScaleGenerationMode::Two)
        .value("Maximum", MxScaleGenerationMode::Maximum)
        .value("NaN", MxScaleGenerationMode::NaN);

    nb::class_<MxGenerationProblem>(module, "MxGenerationProblem")
        .def(nb::init<Shape, MxDataGeneration>(), "shape"_a, "data"_a)
        .def_rw("data_type", &MxGenerationProblem::dataType)
        .def_rw("scale_type", &MxGenerationProblem::scaleType)
        .def_rw("shape", &MxGenerationProblem::shape)
        .def_rw("leading_dimension", &MxGenerationProblem::leadingDimension)
        .def_rw("block_axis", &MxGenerationProblem::blockAxis)
        .def_rw("block_size", &MxGenerationProblem::blockSize)
        .def_rw("data", &MxGenerationProblem::data)
        .def_rw("scale", &MxGenerationProblem::scale);

    nb::class_<MxGenerationResult>(module, "MxGenerationResult")
        .def_ro("data", &MxGenerationResult::data)
        .def_ro("scales", &MxGenerationResult::scales)
        .def_ro("scale_indices", &MxGenerationResult::scaleIndices)
        .def_ro("reference", &MxGenerationResult::reference);

    module.def("generate_mx",
               static_cast<MxGenerationResult (*)(const MxGenerationProblem&)>(&generateMx),
               "problem"_a);
}
}  // namespace roc::host_numerics::python_bindings
