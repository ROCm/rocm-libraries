// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>

#include <roc/hostnumerics/shape.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerIndexRangeBindings(nb::module_& module) {
    nb::class_<LastDimensionFastest>(module, "LastDimensionFastest").def(nb::init<>());
    nb::class_<IndexRange>(module, "IndexRange")
        .def(nb::init<size_t, size_t, size_t>(), "first"_a, "stride"_a, "max_elements"_a)
        .def_ro("first", &IndexRange::first)
        .def_ro("stride", &IndexRange::stride)
        .def_ro("max_elements", &IndexRange::maxElements)
        .def_static("all", &IndexRange::all);
    module.def("prime_index_range", &primeIndexRange, "logical_elements"_a, "allocated_elements"_a,
               "requested_elements"_a);
}
}  // namespace roc::hostnumerics::python_bindings
