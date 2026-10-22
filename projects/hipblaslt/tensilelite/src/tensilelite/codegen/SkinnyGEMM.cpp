// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/Kernel.hpp>
#include <tensilelite/codegen/SkinnyGEMM.hpp>

#include "SkinnyGEMMEmitter.hpp"

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

namespace {

// The reference's features that are not emitted yet.
void requireImplemented(const SkinnyGEMMConfig &config) {
  if (config.upIters > 0)
    throw std::invalid_argument(
        "upIters > 0 (up-front scales) is not implemented yet");
  if (config.partialSteps != 0)
    throw std::invalid_argument(
        "partialSteps != 0 (partial K steps) is not implemented yet");
  if (!config.grid3d)
    throw std::invalid_argument(
        "grid3d = false (persistent launch) is not implemented yet");
  if (config.group != 1)
    throw std::invalid_argument("grid3d requires group = 1");
  if (config.numXcds != 1)
    throw std::invalid_argument("grid3d requires numXcds = 1");
}

// The kernarg layout of the reference: the pointers, the i32 scalars, then
// the hidden group sizes and block counts.
detail::SkinnyArguments declareArguments(KernelSignature &signature) {
  detail::SkinnyArguments arguments;
  arguments.d = signature.addPointer("D");
  arguments.a = signature.addPointer("A");
  arguments.b = signature.addPointer("B");
  arguments.scalesA = signature.addPointer("scalesA");
  arguments.scalesB = signature.addPointer("scalesB");
  arguments.ws = signature.addPointer("ws");
  arguments.ldD = signature.addScalar("ldD", 4);
  arguments.ldA = signature.addScalar("ldA", 4);
  arguments.ldB = signature.addScalar("ldB", 4);
  arguments.m = signature.addScalar("m", 4);
  arguments.n = signature.addScalar("n", 4);
  arguments.k = signature.addScalar("k", 4);
  arguments.spk = signature.addScalar("spk", 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeX, 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeY, 4);
  signature.addHidden(KernelArgumentKind::hiddenGroupSizeZ, 4);
  arguments.blocksX =
      signature.addHidden(KernelArgumentKind::hiddenBlockCountX, 4);
  arguments.blocksY =
      signature.addHidden(KernelArgumentKind::hiddenBlockCountY, 4);
  arguments.blocksZ =
      signature.addHidden(KernelArgumentKind::hiddenBlockCountZ, 4);
  return arguments;
}

} // namespace

std::string emitSkinnyGEMM(const Target &target,
                           const SkinnyGEMMConfig &config) {
  validate(config);
  requireImplemented(config);
  Kernel kernel(target, skinnyGEMMKernelName(config), 64 * config.waves);
  // The workgroup ids are the tile (tx, ty) and the split kz.
  kernel.enableWorkgroupIds(true, true);
  detail::SkinnyArguments arguments = declareArguments(kernel.signature());
  detail::SkinnyGEMMEmitter(kernel, config, arguments).emit();
  try {
    kernel.builder().allocator().checkLimits();
  } catch (const std::runtime_error &error) {
    throw std::invalid_argument(std::string("config exceeds the register "
                                            "budget: ") +
                                error.what());
  }
  return kernel.toAssembly();
}

} // namespace rocm::tensilelite::codegen
