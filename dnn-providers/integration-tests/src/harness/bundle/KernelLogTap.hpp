// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_data_sdk/logging/CallbackTypes.h>

namespace hipdnn_integration_tests::bundle
{

/// The process-wide logging facilities the tap drives: the user log callbacks and the global
/// log level.
///
/// A seam, like the other collaborators of the harness: the real one calls the frontend and
/// needs a backend (FrontendKernelLogControl.cpp, which the unit-test binary does not
/// compile), and a unit test fills in a fake that records the order of the calls.
class IKernelLogControl
{
public:
    IKernelLogControl() = default;
    virtual ~IKernelLogControl() = default;

    IKernelLogControl(const IKernelLogControl&) = delete;
    IKernelLogControl& operator=(const IKernelLogControl&) = delete;
    IKernelLogControl(IKernelLogControl&&) = delete;
    IKernelLogControl& operator=(IKernelLogControl&&) = delete;

    /// Registers `callback` as a synchronous user log callback at INFO, keyed by
    /// (`callback`, `handle`). False when the registration failed.
    virtual bool addUserCallback(hipdnnUserLogCallback_t callback,
                                 hipdnnUserLogCallbackHandle_t handle)
        = 0;

    /// Removes the callback `addUserCallback` registered. When it returns the callback is
    /// not running and will not be called again.
    virtual void removeUserCallback(hipdnnUserLogCallback_t callback,
                                    hipdnnUserLogCallbackHandle_t handle)
        = 0;

    /// The global log level, or nullopt when it cannot be read.
    virtual std::optional<hipdnnSeverity_t> globalLevel() = 0;

    /// Sets the global log level, which is also what pushes it into the loaded plugins. False
    /// when it could not be set.
    virtual bool setGlobalLevel(hipdnnSeverity_t level) = 0;
};

/// Collects the kernels the engines name in their `kernel: <engine> <kernel>` INFO lines
/// between `begin()` and `finish()`, for the outcome record of one case.
///
/// Plugins log only at the level pushed into them, and that level is `off` unless
/// HIPDNN_LOG_LEVEL is set; only the global level setter pushes one. So `begin()` registers
/// the callback first and raises the level to INFO second: registering first keeps the
/// backend from creating a stderr console sink for a level that was off until then.
/// `finish()` restores the level `begin()` found, exactly, and unregisters. A level at INFO
/// already is left alone.
///
/// One case at a time: `begin()` on a tap that is active discards the collection of the
/// earlier case, so lines never leak from one case into the next. The callback may run on
/// whichever thread logs, so the collection is guarded.
class KernelLogTap
{
public:
    /// A null `control` makes the tap inert: nothing is registered and no kernel is reported.
    explicit KernelLogTap(std::shared_ptr<IKernelLogControl> control);
    ~KernelLogTap();

    KernelLogTap(const KernelLogTap&) = delete;
    KernelLogTap& operator=(const KernelLogTap&) = delete;
    KernelLogTap(KernelLogTap&&) = delete;
    KernelLogTap& operator=(KernelLogTap&&) = delete;

    /// Starts collecting for one case.
    void begin();

    /// Stops collecting and returns the distinct kernel names, sorted. Empty when the tap was
    /// not active.
    std::vector<std::string> finish();

    /// The kernel name of one formatted log line, or nullopt when the line is not a kernel
    /// line. The line is the pattern-formatted one the callback receives
    /// ("[time] [tid N] [info] [component] kernel: <engine> <kernel>"), so the marker is
    /// matched after the closing bracket of the prefix, not anywhere: other log lines say
    /// "... kernel: ..." too (the ASM_SDPA stage lines).
    static std::optional<std::string> kernelNameOf(std::string_view line);

private:
    static void
        onLog(hipdnnUserLogCallbackHandle_t handle, hipdnnSeverity_t severity, const char* message);

    void collect(std::string_view line);

    /// Restores the level and removes the callback when the tap is active. Does not touch the
    /// collection.
    void stop() noexcept;

    std::shared_ptr<IKernelLogControl> _control;
    bool _active = false;
    /// The level to put back, set only when `begin()` raised the level.
    std::optional<hipdnnSeverity_t> _levelToRestore;
    std::mutex _mutex;
    std::set<std::string> _kernels;
};

} // namespace hipdnn_integration_tests::bundle
