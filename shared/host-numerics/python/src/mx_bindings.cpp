// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

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

    nb::class_<MxGenerationOptions>(module, "MxGenerationOptions")
        .def(nb::init<>())
        .def_rw("data_type", &MxGenerationOptions::dataType)
        .def_rw("scale_type", &MxGenerationOptions::scaleType)
        .def_rw("leading_dimension", &MxGenerationOptions::leadingDimension)
        .def_rw("block_axis", &MxGenerationOptions::blockAxis)
        .def_rw("block_size", &MxGenerationOptions::blockSize)
        .def_rw("scale", &MxGenerationOptions::scale);

    nb::class_<MxTensor>(module, "MxTensor")
        .def_ro("data", &MxTensor::data)
        .def_ro("scales", &MxTensor::scales)
        .def_ro("scale_indices", &MxTensor::scaleIndices)
        .def_ro("reference", &MxTensor::reference);

    module.def(
        "generate_mx",
        [](std::vector<size_t> shape, MxDataGeneration generation,
           const MxGenerationOptions& options) {
            return generateMx(Shape(std::move(shape)), std::move(generation), options);
        },
        "shape"_a, "generation"_a, "options"_a = MxGenerationOptions{});
    module.def("generate_mx", &generateMx, "shape"_a, "generation"_a,
               "options"_a = MxGenerationOptions{});
}
}  // namespace roc::host_numerics::python_bindings
