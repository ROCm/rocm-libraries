// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/index_range.hpp>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerIndexRangeBindings(nb::module_& module) {
    nb::class_<IndexRange>(module, "IndexRange", R"doc(
Flat indices start + ordinal * stride, all less than the exclusive stop, as in
range(start, stop, stride). All three constructor arguments are required, and
stride must be nonzero. A start at or beyond stop yields an empty range.
Ranges compare and hash by value, support len() and iterate over their indices.
As for Python's range, len() fails with OverflowError above sys.maxsize;
count() returns the exact number of entries for any range.
)doc")
        .def(nb::init<size_t, size_t, size_t>(), "start"_a, "stop"_a, "stride"_a)
        .def_prop_ro("start", &IndexRange::start)
        .def_prop_ro("stop", &IndexRange::stop)
        .def_prop_ro("stride", &IndexRange::stride)
        .def("count", &IndexRange::count, "Count entries without enumerating them.")
        .def("indices", &IndexRange::indices, "Return an owning list of the range's entries.")
        .def_static("all", &IndexRange::all, "stop"_a,
                    "A range from zero to the exclusive stop, with stride one.")
        .def("__len__", &IndexRange::count)
        .def(
            "__iter__",
            [](const IndexRange& range) {
                return nb::make_iterator(nb::type<IndexRange>(), "IndexRangeIterator",
                                         range.begin(), range.end());
            },
            nb::keep_alive<0, 1>())
        .def("__repr__",
             [](const IndexRange& range) {
                 return "IndexRange(start=" + std::to_string(range.start()) +
                        ", stop=" + std::to_string(range.stop()) +
                        ", stride=" + std::to_string(range.stride()) + ")";
             })
        .def("__hash__",
             [](const IndexRange& range) {
                 const nb::tuple value =
                     nb::make_tuple(range.start(), range.stop(), range.stride());
                 const Py_hash_t result = PyObject_Hash(value.ptr());
                 if (result == -1) throw nb::python_error();
                 return result;
             })
        .def(
            "__eq__", [](const IndexRange& left, const IndexRange& right) { return left == right; },
            nb::is_operator())
        .def(
            "__ne__", [](const IndexRange& left, const IndexRange& right) { return left != right; },
            nb::is_operator());
    module.def(
        "prime_index_range",
        [](size_t logicalElements, size_t requestedElements) {
            return primeIndexRange(
                {.logicalElements = logicalElements, .requestedElements = requestedElements});
        },
        nb::kw_only(), "logical_elements"_a, "requested_elements"_a,
        R"doc(
Choose a prime stride from logical_elements / requested_elements.
The resulting count is approximate. Zero requested elements selects no index;
a request at least as large as logical_elements selects the full logical range.
Both arguments are keyword-only because the two counts are easy to transpose.
)doc");
}
}  // namespace roc::hostnumerics::python_bindings
