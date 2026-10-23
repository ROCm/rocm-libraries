// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/layout.hpp>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace roc::hostnumerics::python_bindings {
void registerLayoutBindings(nb::module_& module) {
    nb::class_<Layout>(module, "Layout",
                       R"doc(
Owned Shape, signed element strides, and starting element offset.

Strides and offsets count elements rather than bytes. For Shape([2, 3]),
contiguous C order uses strides [3, 1], and Fortran order uses [1, 2].
Negative strides describe reversal; zero strides describe repeated addresses.
Layout describes indexing independently of datatype and allocation bounds.

Coordinates and axes must be nonnegative, following Shape's convention.
Signed strides and offsets can produce negative element offsets. Invalid
ranks, bounds, and arithmetic overflow raise RuntimeError; values that cannot
be converted to size_t or ptrdiff_t raise TypeError. Layouts are immutable and
compare and hash by their shape, strides, and offset.
)doc")
        .def(nb::init<>())
        .def(nb::init<Shape, std::vector<ptrdiff_t>, ptrdiff_t>(), "shape"_a, "strides"_a,
             "offset"_a = 0)
        .def_static(
            "contiguous", [](const Shape& shape) { return Layout::contiguous(shape); }, "shape"_a,
            "Contiguous C order, with the last dimension fastest.")
        .def_static(
            "contiguous",
            [](const Shape& shape, const std::vector<size_t>& order) {
                return Layout::contiguous(shape, order);
            },
            "shape"_a, "order"_a,
            "Order lists each axis once, from slowest to fastest, as in Shape.linear_index.")
        .def_static("contiguous_last_dimension_fastest", &Layout::contiguousLastDimensionFastest,
                    "shape"_a, "Contiguous C order, with the last dimension fastest.")
        .def_static("contiguous_first_dimension_fastest", &Layout::contiguousFirstDimensionFastest,
                    "shape"_a, "Contiguous Fortran order, with the first dimension fastest.")
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
            "element_offset",
            [](const Layout& layout, const std::vector<size_t>& indices) {
                const auto result = layout.elementOffset(indices);
                if (!result) throw std::runtime_error(result.error().message());
                return *result;
            },
            "indices"_a,
            "Return offset + sum(indices[d] * strides[d]), in elements. All coordinates "
            "are validated before checking each intermediate multiplication and addition.")
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
