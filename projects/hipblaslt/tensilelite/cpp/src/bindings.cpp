// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include "emit_mfma.hpp"

namespace nb = nanobind;

NB_MODULE(_tl_emit, m)
{
    m.def("emitMfmaInstruction",
          &tl_emit::emitMfmaInstruction,
          nb::arg("mxInstType"),
          nb::arg("miK"),
          nb::arg("sourceSwap"),
          nb::arg("miArchVgpr"),
          nb::arg("vgprAStart"),
          nb::arg("opASize"),
          nb::arg("vgprBStart"),
          nb::arg("opBSize"),
          nb::arg("vgprCStart"),
          nb::arg("opCSize"),
          nb::arg("cIsAccvgpr"),
          nb::arg("vgprDStart"),
          nb::arg("opDSize"),
          nb::arg("dIsAccvgpr"),
          nb::arg("scaleAVgpr")   = -1,
          nb::arg("scaleBVgpr")   = -1,
          nb::arg("scaleAsel")    = -1,
          nb::arg("scaleBsel")    = -1,
          nb::arg("tmpScaleVgpr") = -1,
          nb::arg("comment")      = "");
}
