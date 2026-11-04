// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <roc/hostnumerics/permutation.hpp>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace {
nb::tuple permutationTuple(const Permutation& permutation) {
    return nb::tuple(nb::cast(std::vector<size_t>(permutation.begin(), permutation.end())));
}
}  // namespace

namespace roc::hostnumerics::python_bindings {
void registerPermutationBindings(nb::module_& module) {
    nb::class_<Permutation>(module, "Permutation", R"doc(
An ordering of [0, n) in which every value appears exactly once.

Axis orders are Permutations: entry i names the axis at position i, from
slowest to fastest. For Shape([2, 3, 5]), Permutation([1, 2, 0]) traverses axis
1 outermost and axis 0 innermost. Shape and Layout accept only Permutations as
orders, never plain lists, so a list of indices cannot be mistaken for an order.

Construction raises RuntimeError unless every value in [0, n) appears exactly
once. Permutations are immutable, compare and hash by value, and support len(),
indexing and iteration.

Examples:
    >>> Permutation([2, 0, 1]).apply(["a", "b", "c"])
    ['c', 'a', 'b']
    >>> Permutation([2, 0, 1]).inverse()
    Permutation([1, 2, 0])
    >>> Permutation.reverse(3)
    Permutation([2, 1, 0])
)doc")
        .def(nb::init<std::vector<size_t>>(), "entries"_a)
        .def_static("identity", &Permutation::identity, "size"_a,
                    "(0, 1, ..., size - 1): as an axis order, the last axis is fastest.")
        .def_static("reverse", &Permutation::reverse, "size"_a,
                    "(size - 1, ..., 1, 0): as an axis order, the first axis is fastest.")
        .def_prop_ro(
            "entries",
            [](const Permutation& permutation) {
                return std::vector<size_t>(permutation.begin(), permutation.end());
            },
            "Return a copy of the entries.")
        .def("inverse", &Permutation::inverse,
             "The permutation q with q[p[i]] == i, which undoes this one.")
        .def(
            "apply",
            [](const Permutation& permutation, const std::vector<nb::object>& values) {
                return permutation.apply(values);
            },
            "values"_a, "Return a list with result[i] == values[self[i]].")
        .def("__len__", &Permutation::size)
        .def("__getitem__",
             [](const Permutation& permutation, size_t position) {
                 if (position >= permutation.size())
                     throw nb::index_error(("Permutation position " + std::to_string(position) +
                                            " is outside [0, " +
                                            std::to_string(permutation.size()) + ").")
                                               .c_str());
                 return permutation[position];
             })
        .def("__iter__",
             [](const Permutation& permutation) {
                 return permutationTuple(permutation).attr("__iter__")();
             })
        .def("__hash__",
             [](const Permutation& permutation) {
                 const Py_hash_t result = PyObject_Hash(permutationTuple(permutation).ptr());
                 if (result == -1) throw nb::python_error();
                 return result;
             })
        .def("__repr__",
             [](const Permutation& permutation) {
                 return "Permutation(" +
                        nb::cast<std::string>(nb::repr(nb::list(permutationTuple(permutation)))) +
                        ")";
             })
        .def(
            "__eq__",
            [](const Permutation& left, const Permutation& right) { return left == right; },
            nb::is_operator())
        .def(
            "__ne__",
            [](const Permutation& left, const Permutation& right) { return left != right; },
            nb::is_operator());
}
}  // namespace roc::hostnumerics::python_bindings
