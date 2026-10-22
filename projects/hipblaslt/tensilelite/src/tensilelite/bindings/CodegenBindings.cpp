// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/SkinnyGEMM.hpp>
#include <tensilelite/codegen/Target.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

using namespace rocm::tensilelite::codegen;

NB_MODULE(tensilelite_codegen, m) {
  m.doc() = "Kernels emitted with the TensileLite C++ codegen library.";

  const SkinnyGEMMConfig defaults;
  nb::class_<SkinnyGEMMConfig>(
      m, "SkinnyGEMMConfig",
      "Compile-time knobs of the skinny MXFP8 GEMM, named after the "
      "constants of the MLIR reference: bm (BM), tn (TN), lanesK (LANES_K, one "
      "of 4, 8, 16 and 64), waves (WAVES, so a block has 64 * waves lanes), "
      "prefetch (PF, 0 means no ring), upIters (UP_ITERS, 0 rings the scales, "
      "else the up-front vector size), partialSteps (PARTIAL_STEPS, 0, 1 or "
      "2), group (GROUP), numXcds (NUM_XCDS) and grid3d (GRID_3D, false "
      "selects a persistent 1-D launch).")
      .def(
          "__init__",
          [](SkinnyGEMMConfig *self, int bm, int tn, int lanesK, int waves,
             int prefetch, int upIters, int partialSteps, int group,
             int numXcds, bool grid3d) {
            new (self) SkinnyGEMMConfig{bm,       tn,      lanesK,       waves,
                                        prefetch, upIters, partialSteps, group,
                                        numXcds,  grid3d};
          },
          nb::kw_only(), "bm"_a = defaults.bm, "tn"_a = defaults.tn,
          "lanesK"_a = defaults.lanesK, "waves"_a = defaults.waves,
          "prefetch"_a = defaults.prefetch, "upIters"_a = defaults.upIters,
          "partialSteps"_a = defaults.partialSteps, "group"_a = defaults.group,
          "numXcds"_a = defaults.numXcds, "grid3d"_a = defaults.grid3d)
      .def_rw("bm", &SkinnyGEMMConfig::bm)
      .def_rw("tn", &SkinnyGEMMConfig::tn)
      .def_rw("lanesK", &SkinnyGEMMConfig::lanesK)
      .def_rw("waves", &SkinnyGEMMConfig::waves)
      .def_rw("prefetch", &SkinnyGEMMConfig::prefetch)
      .def_rw("upIters", &SkinnyGEMMConfig::upIters)
      .def_rw("partialSteps", &SkinnyGEMMConfig::partialSteps)
      .def_rw("group", &SkinnyGEMMConfig::group)
      .def_rw("numXcds", &SkinnyGEMMConfig::numXcds)
      .def_rw("grid3d", &SkinnyGEMMConfig::grid3d);

  m.def("skinnyGEMMKernelName", &skinnyGEMMKernelName, "config"_a,
        "Returns the symbol of the GEMM kernel for config, in the format of "
        "the reference. Raises ValueError for invalid knobs.");

  // The arch string is resolved here: Target keeps per-thread rocisa state, and
  // each call builds its kernel on the calling thread.
  m.def(
      "emitSkinnyGEMM",
      [](const SkinnyGEMMConfig &config, const std::string &arch) {
        return emitSkinnyGEMM(Target::get(arch), config);
      },
      "config"_a, "arch"_a = "gfx950",
      "Returns the assembly text of the GEMM kernel for config, for the 3-D "
      "launch with ring scales and no partial K steps (grid3d, group = 1, "
      "numXcds = 1, upIters = 0, partialSteps = 0). Raises ValueError for "
      "invalid knobs, an unsupported arch, the features that are not emitted "
      "yet (the message names the knob) and a configuration that exceeds the "
      "register budget.");

  m.def(
      "emitSplitKReduce",
      [](const std::string &arch) {
        return emitSplitKReduce(Target::get(arch));
      },
      "arch"_a = "gfx950",
      "Returns the assembly text of the splitk_reduce kernel: D[i, j] is the "
      "bf16 truncation of the f32 sum over the splits of ws[z * m * n + j * m "
      "+ i]. Block (256, 1, 1), grid (ceil(m / 256), n, 1). Raises ValueError "
      "for an unsupported arch.");
}
