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
                       "Owned shape, signed element strides, and starting element offset.")
        .def(nb::init<>())
        .def(nb::init<Shape, std::vector<ptrdiff_t>, ptrdiff_t>(), "shape"_a, "strides"_a,
             "offset"_a = 0)
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
        .def("stride", &Layout::stride, "dimension"_a)
        .def_prop_ro("offset", &Layout::offset)
        .def(
            "element_offset",
            [](const Layout& layout, const std::vector<size_t>& indices) {
                return layout.elementOffset(indices);
            },
            "indices"_a, "Checked offset + sum(indices[d] * strides[d]), in elements.")
        .def(
            "__eq__", [](const Layout& left, const Layout& right) { return left == right; },
            nb::is_operator())
        .def(
            "__ne__", [](const Layout& left, const Layout& right) { return left != right; },
            nb::is_operator());
}
}  // namespace roc::hostnumerics::python_bindings
