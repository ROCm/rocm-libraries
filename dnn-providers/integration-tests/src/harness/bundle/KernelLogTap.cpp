// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/KernelLogTap.hpp"

#include <utility>

namespace hipdnn_integration_tests::bundle
{

namespace
{

constexpr std::string_view K_KERNEL_MARKER = "kernel: ";
// The pattern-formatted line ends its prefix with the component in brackets, so a kernel line
// is the marker right after "] ".
constexpr std::string_view K_PREFIXED_KERNEL_MARKER = "] kernel: ";
constexpr std::string_view K_BLANKS = " \t\r\n";

} // namespace

KernelLogTap::KernelLogTap(std::shared_ptr<IKernelLogControl> control)
    : _control(std::move(control))
{
}

KernelLogTap::~KernelLogTap()
{
    // A case that left without a `finished` record must not leave the callback behind: it
    // points at this object.
    stop();
}

void KernelLogTap::begin()
{
    stop();
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _kernels.clear();
    }
    if(_control == nullptr)
    {
        return;
    }

    const auto previousLevel = _control->globalLevel();
    // Before the level goes up: the first registration initializes the backend's logger, which
    // opens a stderr console sink when the level is already above off.
    if(!_control->addUserCallback(&KernelLogTap::onLog, this))
    {
        return;
    }
    _active = true;

    // A level that cannot be read cannot be put back, so it is left alone.
    if(previousLevel.has_value() && *previousLevel != HIPDNN_SEV_INFO
       && _control->setGlobalLevel(HIPDNN_SEV_INFO))
    {
        _levelToRestore = previousLevel;
    }
}

std::vector<std::string> KernelLogTap::finish()
{
    const bool wasActive = _active;
    stop();
    if(!wasActive)
    {
        return {};
    }

    const std::lock_guard<std::mutex> lock(_mutex);
    return {_kernels.begin(), _kernels.end()};
}

void KernelLogTap::stop() noexcept
{
    if(!_active)
    {
        return;
    }
    _active = false;

    if(_levelToRestore.has_value())
    {
        static_cast<void>(_control->setGlobalLevel(*_levelToRestore));
        _levelToRestore.reset();
    }
    _control->removeUserCallback(&KernelLogTap::onLog, this);
}

std::optional<std::string> KernelLogTap::kernelNameOf(std::string_view line)
{
    std::string_view payload;
    if(line.substr(0, K_KERNEL_MARKER.size()) == K_KERNEL_MARKER)
    {
        payload = line.substr(K_KERNEL_MARKER.size());
    }
    else
    {
        const auto marker = line.find(K_PREFIXED_KERNEL_MARKER);
        if(marker == std::string_view::npos)
        {
            return std::nullopt;
        }
        payload = line.substr(marker + K_PREFIXED_KERNEL_MARKER.size());
    }

    // "<engine> <kernel>": the engine is the first word, the kernel is the rest.
    const auto end = payload.find_last_not_of(K_BLANKS);
    if(end == std::string_view::npos)
    {
        return std::nullopt;
    }
    payload = payload.substr(0, end + 1);
    const auto engineEnd = payload.find(' ');
    if(engineEnd == std::string_view::npos)
    {
        return std::nullopt;
    }
    const auto nameStart = payload.find_first_not_of(' ', engineEnd);
    if(nameStart == std::string_view::npos)
    {
        return std::nullopt;
    }
    return std::string(payload.substr(nameStart));
}

void KernelLogTap::onLog(hipdnnUserLogCallbackHandle_t handle,
                         [[maybe_unused]] hipdnnSeverity_t severity,
                         const char* message)
{
    if(handle == nullptr || message == nullptr)
    {
        return;
    }
    static_cast<KernelLogTap*>(handle)->collect(message);
}

void KernelLogTap::collect(std::string_view line)
{
    auto name = kernelNameOf(line);
    if(!name.has_value())
    {
        return;
    }
    const std::lock_guard<std::mutex> lock(_mutex);
    _kernels.insert(std::move(*name));
}

} // namespace hipdnn_integration_tests::bundle
