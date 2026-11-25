// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/permutation.hpp>
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

template <typename Indices>
std::vector<std::vector<size_t>> selectedCoordinates(const Shape& shape, const Indices& indices,
                                                     const Permutation& order) {
    std::vector<std::vector<size_t>> result;
    shape.forEachCoordinate(indices, order, [&](size_t, std::span<const size_t> coordinates) {
        result.emplace_back(coordinates.begin(), coordinates.end());
    });
    return result;
}
}  // namespace

namespace roc::hostnumerics::python_bindings {
void registerShapeBindings(nb::module_& module) {
    nb::class_<Shape>(module, "Shape", R"doc(
Immutable sequence of nonnegative extents, independent of storage.

Integer indexing returns an extent at a nonnegative axis less than rank, using
the same bounds checking as extent. Select dimensions with from_dimensions.
Length, iteration, membership, count, index, and hashing follow tuple semantics.
Shapes compare equal to other Shapes with the same extents; they do not compare
equal to tuples or lists. Mutation, concatenation, repetition, slicing, and
ordering comparisons are not supported.

Examples:
    >>> shape = Shape([2, 3, 5])
    >>> shape[2]
    5
    >>> shape.from_dimensions([1, 2])
    Shape((3, 5))
    >>> shape.from_dimensions([2, 1, 0])
    Shape((5, 3, 2))
    >>> tuple(shape)
    (2, 3, 5)
    >>> len(shape)
    3

Shape() has rank zero and represents one scalar. Truthiness follows rank, not
element_count: bool(Shape()) is False, while bool(Shape([0])) is True. Use
element_count to check whether a shape has elements. A zero extent describes a
shape with no elements, so both index conversions reject it. Otherwise the product of
extents must fit size_t. If SIZE_MAX denotes the maximum size_t value,
[SIZE_MAX, 2] fails construction, while [SIZE_MAX, 2, 0] has zero elements.
Selecting dimensions can fail if the resulting nonempty shape is too
large, even when the original shape is empty.

Axes and coordinates must be nonnegative, including integer indexing, extent,
and from_dimensions. Index-conversion orders are Permutations; a plain list
raises TypeError. Invalid logical arguments raise RuntimeError. Membership,
count, and index retain their native Python exceptions. Values that cannot be
converted to size_t raise TypeError.
)doc")
        .def(nb::init<>())
        .def(nb::init<std::vector<size_t>>(), "dimensions"_a)
        .def_prop_ro("rank", &Shape::rank)
        .def_prop_ro(
            "dimensions",
            [](const Shape& shape) {
                return std::vector<size_t>(shape.dimensions().begin(), shape.dimensions().end());
            },
            "Return a copy of the extents.")
        .def("extent", &Shape::extent, "dimension"_a,
             "Return the extent at a nonnegative axis less than rank.")
        .def("__len__", &Shape::rank)
        .def("__iter__", [](const Shape& shape) { return shapeTuple(shape).attr("__iter__")(); })
        .def("__getitem__", &Shape::extent, "dimension"_a,
             "Return the extent at a nonnegative axis less than rank.")
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
        .def_prop_ro(
            "element_count", [](const Shape& shape) { return shape.elementCount(); },
            "Product of all extents; one for rank zero and zero if any extent is zero.")
        .def(
            "from_dimensions",
            [](const Shape& shape, const std::vector<size_t>& axes) {
                return shape.fromDimensions(axes);
            },
            "axes"_a,
            "Select axes in the supplied order, allowing repetition. The resulting "
            "shape must have a representable element count; an empty selection is a scalar.")
        .def(
            "linear_index_last_dimension_fastest",
            [](const Shape& shape, const std::vector<size_t>& indices) {
                return shape.linearIndexLastDimensionFastest(indices);
            },
            "indices"_a, "Convert coordinates to a logical index using ascending axes.")
        .def(
            "linear_index",
            [](const Shape& shape, const std::vector<size_t>& indices, const Permutation& order) {
                return shape.linearIndex(indices, order);
            },
            "indices"_a, "order"_a,
            R"doc(
Convert one coordinate per axis to a logical index, independent of storage strides.

Coordinates always use the original axis order. For Shape([2, 3, 5]), axes 0,
1, and 2 have extents 2, 3, and 5. Order is a Permutation of all axes, listed
from slowest to fastest (outermost to innermost loop); its last entry changes
fastest. Use linear_index_last_dimension_fastest for ascending axes.

For order Permutation([1, 2, 0]), traversal is equivalent to these three nested
loops:

    index = 0
    for i1 in range(3):
        for i2 in range(5):
            for i0 in range(2):
                visit(index, i0, i1, i2)
                index += 1

The formula is (i1 * 5 + i2) * 2 + i0. The logical strides in original axis
order are [1, 10, 2], and coordinates [1, 2, 3] map to index 27.

Other orders for the same shape and coordinates:
    [0, 1, 2]:           strides [15, 5, 1], index 28
    [0, 2, 1]:           strides [15, 1, 3], index 26
    [2, 1, 0]:           strides [1, 2, 6], index 23

Every coordinate must be less than its extent. A shape with a zero extent has
no valid coordinates. Shape().linear_index([], Permutation([])) returns 0 for its one
scalar.
)doc")
        .def(
            "coordinates_last_dimension_fastest",
            [](const Shape& shape, size_t index) {
                return shape.coordinatesLastDimensionFastest(index);
            },
            "linear_index"_a, "Convert a logical index to coordinates using ascending axes.")
        .def(
            "coordinates",
            [](const Shape& shape, size_t index, const Permutation& order) {
                return shape.coordinates(index, order);
            },
            "linear_index"_a, "order"_a,
            R"doc(
Invert linear_index using the same slowest-to-fastest axis order.

The logical index must lie in [0, element_count). The result always uses the
original axis order: Shape([2, 3, 5]).coordinates(27, Permutation([1, 2, 0]))
returns [1, 2, 3].
See linear_index for the nested-loop interpretation of order. A shape with a
zero extent has no valid index; Shape().coordinates(0, Permutation([])) returns []
for its scalar.
)doc")
        .def("selected_coordinates", &selectedCoordinates<IndexRange>, "range"_a, "order"_a,
             R"doc(
Return owning coordinate lists for the range, whose stop must not exceed
element_count.
Order is a slowest-to-fastest Permutation of the axes, checked even for empty
work. Each coordinate list uses the original axis order.
)doc")
        .def("selected_coordinates", &selectedCoordinates<std::vector<size_t>>, "indices"_a,
             "order"_a, R"doc(
Return owning coordinate lists for flat indices in the order given, including
repeats. Every index must be less than element_count. Bounds and the order's
rank are checked before traversal, including empty work. Each coordinate list uses the original axis order.
)doc")
        .def(
            "__eq__", [](const Shape& left, const Shape& right) { return left == right; },
            nb::is_operator())
        .def(
            "__ne__", [](const Shape& left, const Shape& right) { return left != right; },
            nb::is_operator());
}
}  // namespace roc::hostnumerics::python_bindings
