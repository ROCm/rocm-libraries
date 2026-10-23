// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/output_selection.hpp>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerOutputSelectionBindings(nb::module_& module) {
    nb::enum_<OutputSelectionKind>(module, "OutputSelectionKind")
        .value("All", OutputSelectionKind::All)
        .value("Strided", OutputSelectionKind::Strided)
        .value("Explicit", OutputSelectionKind::Explicit);

    nb::class_<OutputSelection>(module, "OutputSelection", R"doc(
Select logical output indices independently of tensor storage.

All selects every index. Strided selects first, first + stride, ... below the
logical element count, limited by max_elements; stride must be nonzero.
Explicit indices are sorted and deduplicated, then checked against the logical
element count by indices and selected_count. Prime stride uses the smallest
prime at least allocated_elements / requested_elements, so its sample count is
approximate. Zero requests or requests at least logical_elements select all.

Every factory requires index_order; strided also requires max_elements.
Selections copy index_order and return a copy from the property. Axes run from
slowest to fastest, as in Shape and Layout. An empty order resolves to ascending
axes (C order) when a consumer supplies the rank. The consumer validates a
nonempty order as a permutation of that rank's axes. Invalid logical arguments
raise RuntimeError; values that cannot be converted to size_t raise TypeError.
)doc")
        .def_static(
            "all", [](const std::vector<size_t>& order) { return OutputSelection::all(order); },
            "index_order"_a)
        .def_static(
            "strided",
            [](size_t first, size_t stride, size_t maxElements, const std::vector<size_t>& order) {
                return OutputSelection::strided(first, stride, maxElements, order);
            },
            "first"_a, "stride"_a, "max_elements"_a, "index_order"_a)
        .def_static(
            "explicit_indices",
            [](std::vector<size_t> indices, const std::vector<size_t>& order) {
                return OutputSelection::explicitIndices(std::move(indices), order);
            },
            "indices"_a, "index_order"_a)
        .def_static(
            "prime_stride",
            [](size_t logical, size_t allocated, size_t requested,
               const std::vector<size_t>& order) {
                return OutputSelection::primeStride(logical, allocated, requested, order);
            },
            "logical_elements"_a, "allocated_elements"_a, "requested_elements"_a, "index_order"_a)
        .def_prop_ro("kind", &OutputSelection::kind)
        .def_prop_ro("selects_all", &OutputSelection::selectsAll)
        .def_prop_ro("first", &OutputSelection::first)
        .def_prop_ro("stride", &OutputSelection::stride)
        .def_prop_ro("max_elements", &OutputSelection::maxElements)
        .def_prop_ro("index_order",
                     [](const OutputSelection& selection) {
                         const auto order = selection.indexOrder();
                         return std::vector<size_t>(order.begin(), order.end());
                     })
        .def("indices", &OutputSelection::indices, "logical_elements"_a)
        .def("selected_count", &OutputSelection::selectedCount, "logical_elements"_a);
}
}  // namespace roc::hostnumerics::python_bindings
