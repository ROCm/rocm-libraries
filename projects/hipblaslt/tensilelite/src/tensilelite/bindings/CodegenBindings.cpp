// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/SkinnyGEMM.hpp>
#include <tensilelite/Codegen/Target.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;

// The version is an int because the JIT module already registers the
// CodeObjectVersion enum.
static KernelAbi makeAbi(int32_t codeObjectVersion, int32_t maxPreloadDwords) {
  if (maxPreloadDwords < 0)
    throw std::invalid_argument("maxPreloadDwords must be non-negative, got " +
                                std::to_string(maxPreloadDwords));
  KernelAbi abi;
  abi.maxPreloadDwords = maxPreloadDwords;
  switch (codeObjectVersion) {
  case 4:
    abi.codeObjectVersion = CodeObjectVersion::v4;
    return abi;
  case 5:
    abi.codeObjectVersion = CodeObjectVersion::v5;
    return abi;
  case 6:
    abi.codeObjectVersion = CodeObjectVersion::v6;
    return abi;
  default:
    throw std::invalid_argument("codeObjectVersion must be 4, 5 or 6, got " +
                                std::to_string(codeObjectVersion));
  }
}

NB_MODULE(tensilelite_codegen, m) {
  m.doc() = "Kernels emitted with the TensileLite C++ codegen library.";

  const SkinnyGEMMConfig defaults;
  nb::class_<SkinnyGEMMConfig>(
      m, "SkinnyGEMMConfig",
      "Compile-time knobs of the skinny MXFP8 GEMM: bm (BM), tn (TN), "
      "lanesK (LANES_K, one of 4, 8, 16 and 64), waves (WAVES, so a block has "
      "64 * waves lanes), "
      "prefetch (PF, 0 means no ring), upIters (UP_ITERS, 0 rings the scales, "
      "else the up-front vector size), partialSteps (PARTIAL_STEPS, 0 or "
      "1), group (GROUP), numXcds (NUM_XCDS), grid3d (GRID_3D, false "
      "selects a persistent 1-D launch) and useBeta (USE_BETA, true prefetches "
      "C and adds beta * C), mfma (true selects the scaled-MFMA compute "
      "path, whose knobs have the meanings documented in SkinnyGEMM.hpp) and "
      "lds (true stages the operands of the MFMA path through LDS).")
      .def(
          "__init__",
          [](SkinnyGEMMConfig *self, int32_t bm, int32_t tn, int32_t lanesK,
             int32_t waves, int32_t prefetch, int32_t upIters,
             int32_t partialSteps, int32_t group, int32_t numXcds, bool grid3d,
             bool useBeta, bool mfma, bool lds) {
            new (self) SkinnyGEMMConfig{bm,       tn,      lanesK,       waves,
                                        prefetch, upIters, partialSteps, group,
                                        numXcds,  grid3d,  useBeta,      mfma,
                                        lds};
          },
          nb::kw_only(), "bm"_a = defaults.bm, "tn"_a = defaults.tn,
          "lanesK"_a = defaults.lanesK, "waves"_a = defaults.waves,
          "prefetch"_a = defaults.prefetch, "upIters"_a = defaults.upIters,
          "partialSteps"_a = defaults.partialSteps, "group"_a = defaults.group,
          "numXcds"_a = defaults.numXcds, "grid3d"_a = defaults.grid3d,
          "useBeta"_a = defaults.useBeta, "mfma"_a = defaults.mfma,
          "lds"_a = defaults.lds)
      .def_rw("bm", &SkinnyGEMMConfig::bm)
      .def_rw("tn", &SkinnyGEMMConfig::tn)
      .def_rw("lanesK", &SkinnyGEMMConfig::lanesK)
      .def_rw("waves", &SkinnyGEMMConfig::waves)
      .def_rw("prefetch", &SkinnyGEMMConfig::prefetch)
      .def_rw("upIters", &SkinnyGEMMConfig::upIters)
      .def_rw("partialSteps", &SkinnyGEMMConfig::partialSteps)
      .def_rw("group", &SkinnyGEMMConfig::group)
      .def_rw("numXcds", &SkinnyGEMMConfig::numXcds)
      .def_rw("grid3d", &SkinnyGEMMConfig::grid3d)
      .def_rw("useBeta", &SkinnyGEMMConfig::useBeta)
      .def_rw("mfma", &SkinnyGEMMConfig::mfma)
      .def_rw("lds", &SkinnyGEMMConfig::lds);

  m.def("skinnyGEMMKernelName", &skinnyGEMMKernelName, "config"_a,
        "Returns the symbol of the GEMM kernel for config, in the format "
        "gemm_BM<v>_TN<v>_.... Raises ValueError for invalid knobs.");

  // The arch string is resolved here: Target keeps per-thread rocisa state, and
  // each call builds its kernel on the calling thread.
  m.def(
      "emitSkinnyGEMM",
      [](const SkinnyGEMMConfig &config, const std::string &arch,
         int32_t codeObjectVersion, int32_t maxPreloadDwords) {
        return emitSkinnyGEMM(Target::get(arch), config,
                              makeAbi(codeObjectVersion, maxPreloadDwords));
      },
      "config"_a, "arch"_a = "gfx950", "codeObjectVersion"_a = 6,
      "maxPreloadDwords"_a = 0,
      "Returns the assembly text of the GEMM kernel for config, for the 3-D "
      "launch (grid3d, group = 1, numXcds = 1) with any upIters and "
      "partialSteps. Raises ValueError for invalid knobs, an unsupported arch, "
      "the features that are not emitted yet (grid3d = false, group, numXcds; "
      "the message names the knob) and a configuration that exceeds the "
      "register budget. codeObjectVersion (4, 5 or 6) selects the directives "
      "and the metadata, and raises ValueError for another value; the kernargs "
      "are the same at every version. maxPreloadDwords (default 0, no "
      "preloading) preloads the leading kernarg pointers into SGPRs, at most "
      "that many dwords and 14 on the supported targets, and raises ValueError "
      "if negative or if the arch does not support kernarg preloading.");

  m.def(
      "emitSplitKReduce",
      [](const std::string &arch, int32_t codeObjectVersion,
         int32_t maxPreloadDwords) {
        return emitSplitKReduce(Target::get(arch),
                                makeAbi(codeObjectVersion, maxPreloadDwords));
      },
      "arch"_a = "gfx950", "codeObjectVersion"_a = 6, "maxPreloadDwords"_a = 0,
      "Returns the assembly text of the splitk_reduce kernel: D[i, j] is the "
      "bf16 truncation of fma(beta, C, alpha * sum), with C at the byte offset "
      "j * ldC + 2 * i (ldD and ldC are byte strides) and sum the "
      "f32 sum over the splits of ws[z * m * n + j * m + i]; C is read only "
      "when beta is not zero. Block (256, 1, 1), grid (ceil(m / 256), n, 1). "
      "codeObjectVersion (4, 5 or 6) selects the directives and the metadata. "
      "maxPreloadDwords (default 0) preloads the kernel arguments into SGPRs. "
      "Raises ValueError for an unsupported arch, code object version or "
      "maxPreloadDwords.");
}
