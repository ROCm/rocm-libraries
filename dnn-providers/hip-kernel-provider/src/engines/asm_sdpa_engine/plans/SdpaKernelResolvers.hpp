// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "SdpaModuleCache.hpp"
#include "device/PerDeviceResolver.hpp"

#include <optional>
#include <string>

namespace asm_sdpa_engine
{

// =============================================================================
// Per-device kernel resolution
// =============================================================================
//
// A plan selects its kernels for an architecture at build time, but the modules
// those kernels live in belong to one device each. These resolvers keep what the
// plan needs to load the same kernels on any device of that architecture, so
// execute() can launch on the device the executing handle's stream names rather
// than the one the plan was built under. See device::PerDeviceResolver for the
// admission rule.

/// The forward module one device launches.
using SdpaFwdKernels = hip_kernel_provider::device::PerDeviceResolver<CachedModule>;

/// One backward stage's kernel: where it lives in the kpack archive and its entry point.
struct SdpaKernelRef
{
    std::string tocKey;
    std::string knlName;
};

/// The backward modules one device launches. `post` is empty on the A16 path, which has
/// no DQ_CONVERT stage.
struct SdpaBwdModules
{
    CachedModule odo;
    CachedModule dqdkdv;
    std::optional<CachedModule> post;
};

using SdpaBwdKernels = hip_kernel_provider::device::PerDeviceResolver<SdpaBwdModules>;

/// Whether a PCI chip id names an MI308-class gfx942 board, which has its own forward
/// binaries. Exposed for testing.
bool isMi308ChipId(int chipId);

/// The kpack toc key of a forward kernel. On gfx942 it depends on the board as well as
/// the architecture, so it is derived per device. Exposed for testing.
std::string fwdKernelTocKey(std::string coName, const std::string& archId, bool isMi308);

/// Loads the forward kernel on each device a plan executes on.
///
/// The CSV row (and with it ts_qo and the kernel name) is chosen by architecture, but
/// on gfx942 the binary is not: MI300- and MI308-class boards share gfx942 and load
/// different code objects. The toc key is therefore re-derived for every device rather
/// than carried over from the build device.
class SdpaFwdKernelResolver : public SdpaFwdKernels
{
public:
    SdpaFwdKernelResolver(SdpaModuleCache& cache,
                          const std::string& arch,
                          std::string coName,
                          std::string knlName);

protected:
    CachedModule resolveForDevice(int deviceOrdinal,
                                  const std::string& reportedArch) const override;

    /// Whether @p deviceOrdinal is an MI308-class board.
    virtual bool queryIsMi308(int deviceOrdinal) const;

    /// Loads @p tocKey on @p deviceOrdinal through the module cache.
    virtual CachedModule loadModule(const std::string& tocKey, int deviceOrdinal) const;

private:
    SdpaModuleCache* _cache;
    std::string _coName;
    std::string _knlName;
};

/// Loads the backward stages on each device a plan executes on. Their toc keys depend
/// on the architecture only, so every device loads the same entries.
class SdpaBwdKernelResolver : public SdpaBwdKernels
{
public:
    SdpaBwdKernelResolver(SdpaModuleCache& cache,
                          const std::string& arch,
                          SdpaKernelRef odo,
                          SdpaKernelRef dqdkdv,
                          std::optional<SdpaKernelRef> post);

protected:
    SdpaBwdModules resolveForDevice(int deviceOrdinal,
                                    const std::string& reportedArch) const override;

    /// Loads one stage on @p deviceOrdinal through the module cache.
    virtual CachedModule loadModule(const SdpaKernelRef& kernel, int deviceOrdinal) const;

private:
    SdpaModuleCache* _cache;
    SdpaKernelRef _odo;
    SdpaKernelRef _dqdkdv;
    std::optional<SdpaKernelRef> _post;
};

} // namespace asm_sdpa_engine
