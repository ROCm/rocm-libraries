// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/layout.hpp>
#include <roc/hostnumerics/permutation.hpp>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerLayoutBindings(nb::module_& module) {
    nb::class_<Layout>(module, "Layout",
                       R"doc(
Maps coordinates to element offsets using a Shape, one stride per axis, and a
starting offset.

Strides and offsets count elements, not bytes. For Shape([2, 3]), C order uses
strides [3, 1] and Fortran order uses [1, 2]. A negative stride reverses an
axis and a zero stride repeats elements, so offsets can be negative and
distinct coordinates can share one. Layout does not know the datatype or the
extent of the underlying storage.

Examples:
    >>> layout = Layout.contiguous_last_dimension_fastest(Shape([2, 3]))
    >>> layout.strides
    [3, 1]
    >>> layout.element_offset([1, 2])
    5
    >>> reversed_rows = Layout(Shape([2, 3]), [-3, 1], 3)
    >>> reversed_rows.element_offset([1, 2])
    2

Layout() is a scalar with zero offset. Orders are Permutations. Invalid ranks,
coordinates, order ranks, and arithmetic overflow raise RuntimeError. Negative
axes or coordinates, and other values that do not fit size_t or ptrdiff_t,
raise TypeError. Layouts are immutable and compare and hash by shape, strides,
and offset.
Contiguous factories require each extent and running stride product, including
the final element count, to fit ptrdiff_t. Even empty shapes can overflow before
the traversal reaches a zero extent.
)doc")
        .def(nb::init<>())
        .def(nb::init<Shape, std::vector<ptrdiff_t>, ptrdiff_t>(), "shape"_a, "strides"_a,
             "offset"_a)
        .def_static(
            "contiguous",
            [](const Shape& shape, const Permutation& order) {
                return Layout::contiguous(shape, order);
            },
            "shape"_a, "order"_a,
            "Contiguous layout with zero offset. Order is a Permutation of the axes, from "
            "slowest to fastest, as in Shape.linear_index.")
        .def_static("contiguous_last_dimension_fastest", &Layout::contiguousLastDimensionFastest,
                    "shape"_a, "Contiguous C order, last dimension fastest.")
        .def_static("contiguous_first_dimension_fastest", &Layout::contiguousFirstDimensionFastest,
                    "shape"_a, "Contiguous Fortran order, first dimension fastest.")
        .def_prop_ro("shape", &Layout::shape, nb::rv_policy::copy)
        .def_prop_ro(
            "strides",
            [](const Layout& layout) {
                const auto strides = layout.strides();
                return std::vector<ptrdiff_t>(strides.begin(), strides.end());
            },
            "Return a copy of the strides.")
        .def("stride", &Layout::stride, "dimension"_a,
             "Return the signed stride at a nonnegative axis less than rank.")
        .def_prop_ro("offset", &Layout::offset)
        .def(
            "element_bounds",
            [](const Layout& layout) {
                const auto result = layout.getElementBounds();
                if (!result) throw std::runtime_error(result.error().message());
                return nb::make_tuple(result->first, result->second);
            },
            "Return inclusive minimum and maximum reachable element offsets. Empty shapes "
            "return (0, -1), and scalars return (offset, offset). Raises RuntimeError if "
            "an intermediate product or sum does not fit ptrdiff_t.")
        .def(
            "element_offset",
            [](const Layout& layout, const std::vector<size_t>& indices) {
                const auto result = layout.elementOffset(indices);
                if (!result) throw std::runtime_error(result.error().message());
                return *result;
            },
            "indices"_a,
            "Return offset + sum(indices[d] * strides[d]). Raises RuntimeError if any "
            "coordinate is out of range or any partial result does not fit ptrdiff_t.")
        .def("__repr__",
             [](const Layout& layout) {
                 const auto strides = layout.strides();
                 const nb::tuple strideValues =
                     nb::tuple(nb::cast(std::vector<ptrdiff_t>(strides.begin(), strides.end())));
                 return "Layout(" +
                        nb::cast<std::string>(
                            nb::repr(nb::cast(layout.shape(), nb::rv_policy::copy))) +
                        ", " + nb::cast<std::string>(nb::repr(strideValues)) + ", " +
                        std::to_string(layout.offset()) + ")";
             })
        .def("__hash__",
             [](const Layout& layout) {
                 const auto dimensions = layout.shape().dimensions();
                 const auto strides = layout.strides();
                 const nb::tuple value = nb::make_tuple(
                     nb::tuple(nb::cast(std::vector<size_t>(dimensions.begin(), dimensions.end()))),
                     nb::tuple(nb::cast(std::vector<ptrdiff_t>(strides.begin(), strides.end()))),
                     layout.offset());
                 const Py_hash_t result = PyObject_Hash(value.ptr());
                 if (result == -1) throw nb::python_error();
                 return result;
             })
        .def(
            "__eq__", [](const Layout& left, const Layout& right) { return left == right; },
            nb::is_operator())
        .def(
            "__ne__", [](const Layout& left, const Layout& right) { return left != right; },
            nb::is_operator());
}
}  // namespace roc::hostnumerics::python_bindings
