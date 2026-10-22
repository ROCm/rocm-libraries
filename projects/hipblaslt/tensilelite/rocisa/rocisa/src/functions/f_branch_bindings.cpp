// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "functions/f_branch.hpp"
#include "base.hpp"
#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;

void branch_func(nb::module_ m)
{
    m.def("BranchIfZero",
          &rocisa::BranchIfZero,
          "Branch if the given SGPR is zero",
          nb::arg("sgprName"),
          nb::arg("computeDataType"),
          nb::arg("tmpSgprIdx"),
          nb::arg("laneSC"),
          nb::arg("label"),
          nb::arg("waveFrontSize"));
    m.def("BranchIfNotZero",
          &rocisa::BranchIfNotZero,
          "Branch if the given SGPR is not zero",
          nb::arg("sgprName"),
          nb::arg("computeDataType"),
          nb::arg("label"));
}
