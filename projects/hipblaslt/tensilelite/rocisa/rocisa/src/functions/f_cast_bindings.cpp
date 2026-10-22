// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "functions/f_cast.hpp"
#include "instruction/common.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

void cast_func(nb::module_ m)
{
    m.def("VSaturateCastInt",
          &rocisa::VSaturateCastInt,
          "Saturate cast int",
          nb::arg("SumIdxVgpr"),
          nb::arg("tmpVgprIdx"),
          nb::arg("tmpSgprIdx"),
          nb::arg("lowerBound"),
          nb::arg("upperBound"),
          nb::arg("type")    = rocisa::SaturateCastType::NORMAL,
          nb::arg("initGpr") = true);
}
