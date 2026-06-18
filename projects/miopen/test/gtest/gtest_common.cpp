/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "gtest_common.hpp"

std::ostream& operator<<(std::ostream& os, const DevDescription& dd)
{
    return os << dd.name << "(" << dd.cu_cnt << ", " << dd.wavefront_size << ")";
}

MockTargetProperties::MockTargetProperties(const TargetProperties& target_properties,
                                           const DevDescription& dev_description,
                                           bool disable_xnack)
    : TargetProperties{target_properties}, name{dev_description.name}, xnack_disabled{disable_xnack}
{
}

const std::string& MockTargetProperties::Name() const { return name; }

bool MockTargetProperties::isXnackEnabled() const
{
    return xnack_disabled ? false : TargetProperties::isXnackEnabled();
}

MockHandle::MockHandle(const DevDescription& dev_description, bool disable_xnack)
    : dev_descr{dev_description},
      target_properties{Handle::GetTargetProperties(), dev_description, disable_xnack}
{
}

const miopen::TargetProperties& MockHandle::GetTargetProperties() const
{
    return target_properties;
}

std::size_t MockHandle::GetMaxComputeUnits() const { return dev_descr.cu_cnt; }
std::size_t MockHandle::GetWavefrontWidth() const { return dev_descr.wavefront_size; }

std::size_t MockHandle::GetMaxMemoryAllocSize() const
{
    return std::numeric_limits<std::size_t>::max();
}

bool MockHandle::CooperativeLaunchSupported() const { return false; }

std::string_view GetBaseDeviceName(std::string_view dev_name)
{
    const auto suffix_pos = dev_name.find(':');
    return suffix_pos == std::string_view::npos ? dev_name : dev_name.substr(0, suffix_pos);
}

Gpu GetGpuType(const std::string& dev_name)
{
    const auto arch = std::string{GetBaseDeviceName(dev_name)};

    if(arch == "gfx900")
        return Gpu::gfx900;
    if(arch == "gfx906")
        return Gpu::gfx906;
    if(arch == "gfx908")
        return Gpu::gfx908;
    if(arch == "gfx90a")
        return Gpu::gfx90A;
    if(arch == "gfx942")
        return Gpu::gfx94X;
    if(miopen::StartsWith(arch, "gfx95"))
        return Gpu::gfx950;
    if(miopen::StartsWith(arch, "gfx103"))
        return Gpu::gfx103X;
    if(miopen::StartsWith(arch, "gfx110"))
        return Gpu::gfx110X;
    if(miopen::StartsWith(arch, "gfx115"))
        return Gpu::gfx115X;
    if(miopen::StartsWith(arch, "gfx120"))
        return Gpu::gfx120X;
    if(miopen::StartsWith(arch, "gfx125"))
        return Gpu::gfx125X;
    throw std::runtime_error("unknown_gpu");
}

Gpu GetDevGpuType()
{
    static const auto dev = GetGpuType(get_handle().GetDeviceName());
    return dev;
}

const std::multimap<Gpu, DevDescription>& GetAllKnownDevices()
{
    static_assert(Gpu::gfx125X == Gpu::gfxLast);

    // https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
    static const std::multimap<Gpu, DevDescription> known_devs = {
        // clang-format off
        {Gpu::gfx900,  {"gfx900",  64, 64}},
        {Gpu::gfx906,  {"gfx906",  60, 64}},
        {Gpu::gfx906,  {"gfx906",  64, 64}},
        {Gpu::gfx908,  {"gfx908",  120, 64}},
        {Gpu::gfx90A,  {"gfx90a",  104, 64}},
        {Gpu::gfx90A,  {"gfx90a",  110, 64}},
        {Gpu::gfx94X,  {"gfx942",  228, 64}},
        {Gpu::gfx94X,  {"gfx942",  304, 64}},
        {Gpu::gfx950,  {"gfx950",  256, 64}},
        {Gpu::gfx103X, {"gfx1030", 30, 32}},
        {Gpu::gfx103X, {"gfx1030", 36, 32}},
        {Gpu::gfx103X, {"gfx1030", 40, 32}},
        {Gpu::gfx103X, {"gfx1031", 18, 32}},
        {Gpu::gfx103X, {"gfx1031", 20, 32}},
        {Gpu::gfx103X, {"gfx1032", 14, 32}},
        {Gpu::gfx103X, {"gfx1032", 16, 32}},
        {Gpu::gfx110X, {"gfx1100", 35, 32}},
        {Gpu::gfx110X, {"gfx1100", 40, 32}},
        {Gpu::gfx110X, {"gfx1100", 42, 32}},
        {Gpu::gfx110X, {"gfx1100", 48, 32}},
        {Gpu::gfx110X, {"gfx1101", 24, 32}},
        {Gpu::gfx110X, {"gfx1101", 27, 32}},
        {Gpu::gfx110X, {"gfx1101", 30, 32}},
        {Gpu::gfx110X, {"gfx1102", 16, 32}},
        {Gpu::gfx110X, {"gfx1103", 6, 32}},
        {Gpu::gfx115X, {"gfx1150", 24, 32}},
        {Gpu::gfx115X, {"gfx1151", 40, 32}},
        {Gpu::gfx115X, {"gfx1152", 8, 32}},
        {Gpu::gfx115X, {"gfx1153", 2, 32}},
        {Gpu::gfx120X, {"gfx1201", 32, 32}},
        {Gpu::gfx120X, {"gfx1201", 28, 32}},
        {Gpu::gfx125X, {"gfx1250", 32, 32}} // TODO: update when public
        // clang-format on
    };
    return known_devs;
}

bool IsTestSupportedByDevice(Gpu supported_devs)
{
    if((supported_devs & GetDevGpuType()) != Gpu::None)
    {
        return true;
    }
    return false;
}
