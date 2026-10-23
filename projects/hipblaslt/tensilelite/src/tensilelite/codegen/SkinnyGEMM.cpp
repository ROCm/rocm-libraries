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
  detail::SkinnyDims dims(config);
  if (dims.ldsBytes > target.maxLdsSize())
    throw std::invalid_argument(
        "config exceeds the LDS budget: " + std::to_string(dims.ldsBytes) +
        " bytes of " + std::to_string(target.maxLdsSize()));
  Kernel kernel(target, skinnyGEMMKernelName(config), 64 * config.waves);
  if (dims.scalesShared)
    kernel.setLdsSize(dims.ldsBytes);
  // The workgroup ids are the tile (tx, ty) and the split kz; the persistent
  // launch only has the block id x.
  if (config.grid3d)
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
