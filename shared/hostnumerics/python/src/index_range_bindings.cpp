// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/shape.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerIndexRangeBindings(nb::module_& module) {
    nb::class_<LastDimensionFastest>(module, "LastDimensionFastest").def(nb::init<>());
    nb::class_<IndexRange>(module, "IndexRange", R"doc(
Flat indices first + ordinal * stride, with at most max_elements entries.
All three constructor arguments are required, and stride must be nonzero.
count(end) and indices(end) use an exclusive end bound: entries at or beyond
end are omitted, and a starting index at or beyond end yields an empty range.
)doc")
        .def(nb::init<size_t, size_t, size_t>(), "first"_a, "stride"_a, "max_elements"_a)
        .def_ro("first", &IndexRange::first)
        .def_ro("stride", &IndexRange::stride)
        .def_ro("max_elements", &IndexRange::maxElements)
        .def("count", &IndexRange::count, "end"_a,
             "Count entries below end, clipped to max_elements, without enumerating them.")
        .def("indices", &IndexRange::indices, "end"_a,
             "Return an owning list of entries below end, clipped to max_elements.")
        .def_static("all", &IndexRange::all,
                    "A range starting at zero with stride one and no maximum-count restriction.");
    module.def("prime_index_range", &primeIndexRange, "logical_elements"_a, "allocated_elements"_a,
               "requested_elements"_a, R"doc(
Choose a prime stride from allocated_elements / requested_elements.
The resulting count is approximate. Zero requested elements, or a request
at least as large as logical_elements, selects the full logical range.
)doc");
}
}  // namespace roc::hostnumerics::python_bindings
