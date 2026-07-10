// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include "descriptors.hpp"
#include "init.hpp"

namespace nb = nanobind;
using namespace hipblaslt_py;

void init_descriptors(nb::module_& m)
{
    nb::class_<Handle>(m, "Handle")
        .def(nb::init<>())
        .def("close", &Handle::close)
        .def_prop_ro("ptr", &Handle::ptr)
        .def("__enter__", [](Handle& self) -> Handle& { return self; },
             nb::rv_policy::reference_internal)
        .def("__exit__", [](Handle& self, nb::object, nb::object, nb::object) {
            self.close();
            return false;
        });

    nb::class_<MatrixLayout>(m, "MatrixLayout")
        .def(nb::init<hipDataType, uint64_t, uint64_t, int64_t>(),
             nb::arg("dtype"), nb::arg("rows"), nb::arg("cols"), nb::arg("ld"))
        .def("set_attribute", &MatrixLayout::set_attribute,
             nb::arg("attr"), nb::arg("value"))
        .def_prop_ro("ptr", &MatrixLayout::ptr);

    nb::class_<MatmulDesc>(m, "MatmulDesc")
        .def(nb::init<hipblasComputeType_t, hipDataType>(),
             nb::arg("compute_type"), nb::arg("scale_type"))
        .def("set_attribute_int", &MatmulDesc::set_attribute_int,
             nb::arg("attr"), nb::arg("value"))
        .def("set_attribute_ptr", &MatmulDesc::set_attribute_ptr,
             nb::arg("attr"), nb::arg("ptr"))
        .def("get_attribute_int", &MatmulDesc::get_attribute_int,
             nb::arg("attr"))
        .def_prop_ro("ptr", &MatmulDesc::ptr);

    nb::class_<Preference>(m, "Preference")
        .def(nb::init<>())
        .def("set_max_workspace", &Preference::set_max_workspace,
             nb::arg("nbytes"))
        .def_prop_ro("ptr", &Preference::ptr);
}
