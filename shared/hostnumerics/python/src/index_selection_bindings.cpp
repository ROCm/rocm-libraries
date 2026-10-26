// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/index_selection.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerIndexSelectionBindings(nb::module_& module) {
    nb::class_<LastDimensionFastest>(module, "LastDimensionFastest").def(nb::init<>());
    nb::class_<IndexRange>(module, "IndexRange")
        .def(nb::init<size_t, size_t, size_t>(), "first"_a, "stride"_a, "max_elements"_a)
        .def_ro("first", &IndexRange::first)
        .def_ro("stride", &IndexRange::stride)
        .def_ro("max_elements", &IndexRange::maxElements)
        .def_static("all", &IndexRange::all);
    module.def("prime_index_range", &primeIndexRange, "logical_elements"_a, "allocated_elements"_a,
               "requested_elements"_a);
    nb::class_<IndexSelection>(module, "IndexSelection", R"doc(
Define an ordered set of coordinates to visit when paired with a Shape.

Own a flat IndexRange or sorted unique indices, plus an axis order listed from
slowest to fastest. Pass a complete axis permutation or LastDimensionFastest().
Shape checks the order's rank and index bounds, counts the selected elements,
and supplies coordinates. Explicit empty orders are valid only for rank zero.
Invalid logical arguments raise RuntimeError; values outside size_t raise TypeError.
)doc")
        .def(nb::init<IndexSelection::FlatIndices, std::vector<size_t>>(), "indices"_a, "order"_a)
        .def(nb::init<IndexSelection::FlatIndices, LastDimensionFastest>(), "indices"_a, "order"_a);
}
}  // namespace roc::hostnumerics::python_bindings
