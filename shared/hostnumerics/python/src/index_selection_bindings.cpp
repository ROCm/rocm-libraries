// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/index_selection.hpp>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerIndexSelectionBindings(nb::module_& module) {
    nb::enum_<IndexSelectionKind>(module, "IndexSelectionKind")
        .value("All", IndexSelectionKind::All)
        .value("Strided", IndexSelectionKind::Strided)
        .value("Explicit", IndexSelectionKind::Explicit);

    nb::class_<IndexSelection>(module, "IndexSelection", R"doc(
Select logical indices independently of tensor storage.

All selects every index. Strided selects first, first + stride, ... below the
logical element count, limited by max_elements; stride must be nonzero.
Explicit indices are sorted and deduplicated, then checked against the logical
element count by indices and selected_count. Prime stride uses the smallest
prime at least allocated_elements / requested_elements, so its sample count is
approximate. Zero requests or requests at least logical_elements select all.

Factories with index_order require a permutation of the tensor's axes, listed
from slowest to fastest. Factories named last_dimension_fastest choose ascending
axes. index_order(rank) returns the chosen order and checks the rank; an explicit
empty order is valid only for rank zero. Selections own their orders and indices.
Invalid logical arguments raise RuntimeError; values that cannot be converted to
size_t raise TypeError.
)doc")
        .def_static(
            "all", [](const std::vector<size_t>& order) { return IndexSelection::all(order); },
            "index_order"_a)
        .def_static(
            "strided",
            [](size_t first, size_t stride, size_t maxElements, const std::vector<size_t>& order) {
                return IndexSelection::strided(first, stride, maxElements, order);
            },
            "first"_a, "stride"_a, "max_elements"_a, "index_order"_a)
        .def_static(
            "explicit_indices",
            [](std::vector<size_t> indices, const std::vector<size_t>& order) {
                return IndexSelection::explicitIndices(std::move(indices), order);
            },
            "indices"_a, "index_order"_a)
        .def_static(
            "prime_stride",
            [](size_t logical, size_t allocated, size_t requested,
               const std::vector<size_t>& order) {
                return IndexSelection::primeStride(logical, allocated, requested, order);
            },
            "logical_elements"_a, "allocated_elements"_a, "requested_elements"_a, "index_order"_a)
        .def_prop_ro("kind", &IndexSelection::kind)
        .def_prop_ro("selects_all", &IndexSelection::selectsAll)
        .def_prop_ro("first", &IndexSelection::first)
        .def_prop_ro("stride", &IndexSelection::stride)
        .def_prop_ro("max_elements", &IndexSelection::maxElements)
        .def("index_order", &IndexSelection::indexOrder, "rank"_a)
        .def_static("all_last_dimension_fastest", &IndexSelection::allLastDimensionFastest)
        .def_static("strided_last_dimension_fastest", &IndexSelection::stridedLastDimensionFastest,
                    "first"_a, "stride"_a, "max_elements"_a)
        .def_static("explicit_indices_last_dimension_fastest",
                    &IndexSelection::explicitIndicesLastDimensionFastest, "indices"_a)
        .def_static("prime_stride_last_dimension_fastest",
                    &IndexSelection::primeStrideLastDimensionFastest, "logical_elements"_a,
                    "allocated_elements"_a, "requested_elements"_a)
        .def("indices", &IndexSelection::indices, "logical_elements"_a)
        .def("selected_count", &IndexSelection::selectedCount, "logical_elements"_a);
}
}  // namespace roc::hostnumerics::python_bindings
