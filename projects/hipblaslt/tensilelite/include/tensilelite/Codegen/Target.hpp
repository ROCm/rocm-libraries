// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_CODEGEN_TARGET_HPP
#define ROCM_TENSILELITE_CODEGEN_TARGET_HPP

#include "helper.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace rocm::tensilelite::codegen {

/// Description of a GPU that code is generated for.
///
/// A Target is a small copyable value. Target::get is the only way to obtain
/// one, which guarantees rocisa has been initialized for its ISA.
class Target {
public:
  /// Returns the target for arch. Only gfx950 is supported.
  ///
  /// The first call for an ISA initializes rocisa's ISA capabilities, which
  /// runs the assembler about 130 times; it happens once per process and is
  /// thread-safe. Throws std::invalid_argument for unsupported architectures.
  static Target get(std::string_view arch);

  /// Points the calling thread's rocisa state at this target. rocisa keeps the
  /// selected ISA per thread, so every thread that builds or prints
  /// instructions must call this first.
  void selectForCurrentThread() const;

  /// ISA version as rocisa identifies it, such as {9, 5, 0}.
  IsaVersion isaVersion() const { return isa; }
  /// Number of lanes in a wavefront.
  int32_t wavefrontSize() const { return wavefront; }
  /// Architecture name, such as gfx950.
  std::string_view archName() const { return arch; }
  /// Full ISA name for .amdgcn_target and the assembler, such as
  /// amdgcn-amd-amdhsa--gfx950 (no feature suffixes).
  std::string_view isaName() const { return isaFullName; }
  /// Maximum number of arch VGPRs a kernel may use.
  int32_t maxVgpr() const { return maxVgprs; }
  /// Maximum number of SGPRs a kernel may use.
  int32_t maxSgpr() const { return maxSgprs; }
  /// Maximum number of AGPRs a kernel may use: the unified register file
  /// minus the arch VGPRs, as the Python kernel writer computes it.
  int32_t maxAgpr() const { return maxAgprs; }
  /// Size of the unified register file that a kernel's arch VGPRs, rounded up
  /// to the accum offset, and its AGPRs share.
  int32_t maxUnifiedVgpr() const { return maxUnifiedVgprs; }
  /// Largest LDS allocation per workgroup in bytes (rocisa's DeviceLDS).
  int32_t maxLdsSize() const { return maxLdsBytes; }
  /// Largest vmcnt an s_waitcnt can encode.
  int32_t maxVmcnt() const { return maxVmcnts; }
  /// Largest lgkmcnt an s_waitcnt can encode.
  int32_t maxLgkmcnt() const { return maxLgkmcnts; }
  /// Number of XCDs; the dispatcher assigns workgroup L (x fastest over the
  /// grid) to XCD L mod xcdCount().
  int32_t xcdCount() const { return xcds; }

private:
  Target(IsaVersion isa, int32_t wavefront, int32_t xcds, std::string_view arch,
         std::string_view isaFullName);

  IsaVersion isa;
  int32_t wavefront;
  int32_t xcds;
  std::string arch;
  std::string isaFullName;
  int32_t maxVgprs = 0;
  int32_t maxSgprs = 0;
  int32_t maxAgprs = 0;
  int32_t maxUnifiedVgprs = 0;
  int32_t maxLdsBytes = 0;
  int32_t maxVmcnts = 0;
  int32_t maxLgkmcnts = 0;
};

} // namespace rocm::tensilelite::codegen

#endif // ROCM_TENSILELITE_CODEGEN_TARGET_HPP
