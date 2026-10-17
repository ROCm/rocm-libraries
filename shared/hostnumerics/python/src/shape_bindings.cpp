// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/shape.hpp>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace {
nb::tuple shapeTuple(const Shape& shape) {
    return nb::tuple(
        nb::cast(std::vector<size_t>(shape.dimensions().begin(), shape.dimensions().end())));
}
}  // namespace

namespace roc::hostnumerics::python_bindings {
void registerShapeBindings(nb::module_& module) {
    nb::class_<Shape>(module, "Shape",
                      "Immutable Python tuple-like sequence of nonnegative extents. "
                      "A rank-zero shape describes one scalar. "
                      "The product of nonzero extents must fit size_t.")
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
        .def("__len__", &Shape::rank)
        .def("__iter__", [](const Shape& shape) { return shapeTuple(shape).attr("__iter__")(); })
        .def(
            "__getitem__",
            [](const Shape& shape, nb::handle key) -> nb::object {
                nb::object result = shapeTuple(shape).attr("__getitem__")(key);
                if (nb::isinstance<nb::slice>(key))
                    return nb::cast(Shape(nb::cast<std::vector<size_t>>(result)));
                return result;
            },
            "key"_a.none())
        .def(
            "__contains__",
            [](const Shape& shape, nb::handle value) {
                return shapeTuple(shape).attr("__contains__")(value);
            },
            "value"_a.none())
        .def(
            "count",
            [](const Shape& shape, nb::handle value) {
                return shapeTuple(shape).attr("count")(value);
            },
            "value"_a.none())
        .def(
            "index",
            [](const Shape& shape, nb::handle value, nb::args bounds) {
                return shapeTuple(shape).attr("index")(value, *bounds);
            },
            "value"_a.none(), "bounds"_a)
        .def("__hash__",
             [](const Shape& shape) {
                 const Py_hash_t result = PyObject_Hash(shapeTuple(shape).ptr());
                 if (result == -1) throw nb::python_error();
                 return result;
             })
        .def("__repr__",
             [](const Shape& shape) {
                 return "Shape(" + nb::cast<std::string>(nb::repr(shapeTuple(shape))) + ")";
             })
        .def_prop_ro("element_count", [](const Shape& shape) { return shape.elementCount(); })
        .def("element_count_in_range",
             static_cast<size_t (Shape::*)(size_t, size_t) const>(&Shape::elementCount),
             "first_dimension"_a, "one_past_last_dimension"_a,
             "Product over a half-open dimension range; an empty product is one.")
        .def(
            "project",
            [](const Shape& shape, const std::vector<size_t>& axes) { return shape.project(axes); },
            "axes"_a, "Select distinct dimensions in the given order.")
        .def(
            "linear_index",
            [](const Shape& shape, const std::vector<size_t>& indices,
               const std::vector<size_t>& order) { return shape.linearIndex(indices, order); },
            "indices"_a, "order"_a = std::vector<size_t>{},
            "Order lists axis numbers from outermost to innermost (slowest to fastest); "
            "empty selects row-major, with the last axis fastest.")
        .def(
            "coordinates",
            [](const Shape& shape, size_t index, const std::vector<size_t>& order) {
                return shape.coordinates(index, order);
            },
            "linear_index"_a, "order"_a = std::vector<size_t>{},
            "Invert linear_index using the same outermost-to-innermost axis order.")
        .def(
            "__eq__", [](const Shape& left, const Shape& right) { return left == right; },
            nb::is_operator())
        .def(
            "__ne__", [](const Shape& left, const Shape& right) { return left != right; },
            nb::is_operator());
}
}  // namespace roc::hostnumerics::python_bindings
