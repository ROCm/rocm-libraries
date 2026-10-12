// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/shape.hpp>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerShapeBindings(nb::module_& module) {
    nb::enum_<IndexOrder>(module, "IndexOrder",
                          "Logical enumeration order, independent of physical strides.")
        .value("FirstDimensionFastest", IndexOrder::FirstDimensionFastest)
        .value("LastDimensionFastest", IndexOrder::LastDimensionFastest);

    nb::class_<Shape>(module, "Shape",
                      "Owned nonnegative extents. A rank-zero shape describes one scalar.")
        .def(nb::init<>())
        .def(nb::init<std::vector<size_t>>(), "dimensions"_a)
        .def_prop_ro("rank", &Shape::rank)
        .def_prop_ro(
            "dimensions",
            [](const Shape& shape) {
                return std::vector<size_t>(shape.dimensions().begin(), shape.dimensions().end());
            },
            "Return a copy of the extents.")
        .def("extent", &Shape::extent, "dimension"_a)
        .def("__getitem__", &Shape::extent, "dimension"_a)
        .def_prop_ro("element_count", [](const Shape& shape) { return shape.elementCount(); })
        .def("element_count_in_range",
             static_cast<size_t (Shape::*)(size_t, size_t) const>(&Shape::elementCount),
             "first_dimension"_a, "one_past_last_dimension"_a,
             "Product over a half-open dimension range; an empty product is one.")
        .def("element_count_excluding", &Shape::elementCountExcluding, "dimension"_a)
        .def(
            "linear_index",
            [](const Shape& shape, const std::vector<size_t>& indices, IndexOrder order) {
                return shape.linearIndex(indices, order);
            },
            "indices"_a, "order"_a)
        .def("coordinates",
             static_cast<std::vector<size_t> (Shape::*)(size_t, IndexOrder) const>(
                 &Shape::coordinates),
             "linear_index"_a, "order"_a)
        .def(
            "__eq__", [](const Shape& left, const Shape& right) { return left == right; },
            nb::is_operator())
        .def(
            "__ne__", [](const Shape& left, const Shape& right) { return left != right; },
            nb::is_operator());
}
}  // namespace roc::hostnumerics::python_bindings
