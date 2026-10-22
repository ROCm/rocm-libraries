// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/SkinnyGEMM.hpp>

#include <stdexcept>
#include <string>

namespace rocm::tensilelite::codegen {

namespace {

void require(bool condition, const char *knob, int value, const char *rule) {
  if (condition)
    return;
  throw std::invalid_argument(std::string(knob) + " must be " + rule +
                              ", got " + std::to_string(value));
}

void validate(const SkinnyGEMMConfig &config) {
  require(config.lanesK == 4 || config.lanesK == 8 || config.lanesK == 16 ||
              config.lanesK == 64,
          "lanesK", config.lanesK, "4, 8, 16 or 64");
  require(config.bm >= 1, "bm", config.bm, "at least 1");
  require(config.tn >= 1, "tn", config.tn, "at least 1");
  // A block of 64 * waves lanes cannot exceed 1024 lanes.
  require(config.waves >= 1 && config.waves <= 16, "waves", config.waves,
          "between 1 and 16");
  require(config.prefetch >= 0, "prefetch", config.prefetch, "at least 0");
  require(config.upIters >= 0, "upIters", config.upIters, "at least 0");
  require(config.partialSteps >= 0 && config.partialSteps <= 2, "partialSteps",
          config.partialSteps, "0, 1 or 2");
  require(config.group >= 1, "group", config.group, "at least 1");
  require(config.numXcds >= 1, "numXcds", config.numXcds, "at least 1");
}

} // namespace

std::string skinnyGEMMKernelName(const SkinnyGEMMConfig &config) {
  validate(config);
  return "gemm_BM" + std::to_string(config.bm) + "_TN" +
         std::to_string(config.tn) + "_LANES_K" +
         std::to_string(config.lanesK) + "_WAVES" +
         std::to_string(config.waves) + "_UP_ITERS" +
         std::to_string(config.upIters) + "_PARTIAL_STEPS" +
         std::to_string(config.partialSteps) + "_PF" +
         std::to_string(config.prefetch) + "_GROUP" +
         std::to_string(config.group) + "_NUM_XCDS" +
         std::to_string(config.numXcds) + "_GRID_3D" +
         std::to_string(config.grid3d ? 1 : 0);
}

// The emitter lands in later milestones, so the target is unused until then.
std::string emitSkinnyGEMM(const Target &, const SkinnyGEMMConfig &config) {
  validate(config);
  throw std::invalid_argument("not implemented yet");
}

} // namespace rocm::tensilelite::codegen
