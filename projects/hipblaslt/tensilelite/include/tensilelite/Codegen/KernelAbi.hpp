// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_KERNELABI_HPP
#define ROCM_TENSILELITE_CODEGEN_KERNELABI_HPP

#include <tensilelite/API/Compiler.hpp>

#include <cstdint>

namespace rocm::tensilelite::codegen {

/// The choices that fix how a kernel receives its arguments, made once when the
/// Kernel is constructed.
struct KernelAbi {
  CodeObjectVersion codeObjectVersion = CodeObjectVersion::v6;
  /// Upper bound on kernarg dwords preloaded into SGPRs; 0 disables preloading.
  int32_t maxPreloadDwords = 0;
};

/// Construction-time options of a Kernel.
struct KernelOptions {
  KernelAbi abi;
  /// Enable the workgroup ids y and z, which the hardware places in the system
  /// SGPRs after workgroup id x.
  bool workgroupIdY = false;
  bool workgroupIdZ = false;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_KERNELABI_HPP
