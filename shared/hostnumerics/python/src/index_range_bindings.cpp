// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/index_range.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerIndexRangeBindings(nb::module_& module) {
    nb::class_<IndexRange>(module, "IndexRange", R"doc(
Flat indices start + ordinal * stride, all less than the exclusive end.
All three constructor arguments are required, and stride must be nonzero.
The argument order is (start, end, stride). A start at or beyond end yields
an empty range. count() and indices() use the bound stored in the range.
)doc")
        .def(nb::init<size_t, size_t, size_t>(), "start"_a, "end"_a, "stride"_a)
        .def_ro("start", &IndexRange::start)
        .def_ro("end", &IndexRange::end)
        .def_ro("stride", &IndexRange::stride)
        .def("count", &IndexRange::count, "Count entries without enumerating them.")
        .def("indices", &IndexRange::indices, "Return an owning list of the range's entries.")
        .def_static("all", &IndexRange::all, "end"_a,
                    "A range from zero to the exclusive end, with stride one.");
    module.def("prime_index_range", &primeIndexRange, "logical_elements"_a, "allocated_elements"_a,
               "requested_elements"_a, R"doc(
Choose a prime stride from allocated_elements / requested_elements.
The resulting count is approximate. Zero requested elements, or a request
at least as large as logical_elements, selects the full logical range.
)doc");
}
}  // namespace roc::hostnumerics::python_bindings
