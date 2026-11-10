// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "engines/kernel_ingestor_engine/packs/ConvDepthwiseMfmaProblem.hpp"

#include <limits>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_mfma
{

namespace
{

using conv_depthwise_valu::DepthwiseLayout;
using conv_depthwise_valu::DepthwiseValuProblem;

constexpr int64_t INT_LIMIT = std::numeric_limits<int>::max();
/// LDS a block may claim (64 KiB).
constexpr int64_t LDS_BYTES = 65536;
/// Bytes of one 16-bit element.
constexpr int64_t ELEMENT_BYTES = 2;

int64_t divUp(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

bool isPowerOfTwoUpTo(int64_t value, int64_t limit)
{
    return value >= 1 && value <= limit && (value & (value - 1)) == 0;
}

} // namespace

int64_t mfmaLdsBytes(const MfmaConfig& config)
{
    // Channels-first loads each wave's rows straight into its fragments.
    if(config.layout != "NHWC")
    {
        return 0;
    }
    const int64_t positions = config.atom / 8;
    const int64_t passes = (config.kw + config.stride + positions - 1) / positions;
    const int64_t columns = config.fold * (config.stride * (32 / config.fold) + positions * passes);
    // Two staged input rows and two output rows, 16 bytes per (column, wave).
    return 2 * (columns + 32) * config.waves * 16;
}

bool mfmaRunsOn(int64_t atom, std::string_view gcnArchName, bool bf16, std::string_view layout)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(gcnArchName);
    if(atom == 32)
    {
        // gfx94x runs K = 32 as the 2:4-sparse MFMA: the channel-diagonal taps (NHWC) compress,
        // the channel-pair core's dense input operand (NCHW) does not.
        return arch == "gfx950"
               || (layout == "NHWC" && (arch == "gfx940" || arch == "gfx941" || arch == "gfx942"));
    }
    if(atom == 16)
    {
        if(arch == "gfx908")
        {
            return !bf16;
        }
        return arch == "gfx90a" || arch == "gfx940" || arch == "gfx941" || arch == "gfx942"
               || arch == "gfx950";
    }
    return false;
}

std::optional<MfmaLaunch> mfmaLaunchFor(const DepthwiseValuProblem& problem,
                                        const MfmaConfig& config)
{
    const bool channelsLast = config.layout == "NHWC";
    if((!channelsLast && config.layout != "NCHW")
       || problem.layout
              != (channelsLast ? DepthwiseLayout::CHANNELS_LAST : DepthwiseLayout::CHANNELS_FIRST)
       || problem.ndims != 2 || problem.gatherOnly)
    {
        return std::nullopt;
    }
    const auto& a = problem.args;
    // The run convolution's stride on both axes (backward data runs its unit-stride forward
    // form; a strided one is gatherOnly), unit dilation.
    if((config.stride != 1 && config.stride != 2) || a.sh != config.stride || a.sw != config.stride
       || a.dh != 1 || a.dw != 1)
    {
        return std::nullopt;
    }
    if((config.atom != 16 && config.atom != 32) || config.kh < 1 || config.kw < 1
       || config.kh != a.kh || config.kw != a.kw)
    {
        return std::nullopt;
    }
    // Channels-last: a wave owns 8 channels of the block's 8 x waves. Channels-first: a wave owns
    // a channel pair, any count (a missing second channel reads and writes nothing).
    if(!isPowerOfTwoUpTo(config.waves, 16) || (channelsLast && a.c % (8 * config.waves) != 0))
    {
        return std::nullopt;
    }
    // Images sharing the MFMA columns: 32 (channels-last) or 128 (channels-first) of them.
    if(config.depth < 1 || config.depth > 4 || !isPowerOfTwoUpTo(config.fold, channelsLast ? 4 : 8)
       || config.rows < 1 || config.rows > INT_LIMIT)
    {
        return std::nullopt;
    }
    if(mfmaLdsBytes(config) > LDS_BYTES)
    {
        return std::nullopt;
    }
    // A buffer resource spans the block's folded images; its 32-bit offsets must reach them all
    // with the out-of-range offset 2^31 past every one.
    const int64_t inBytes = int64_t{a.hi} * a.wi * a.c * ELEMENT_BYTES;
    const int64_t outBytes = int64_t{a.ho} * a.wo * a.c * ELEMENT_BYTES;
    if(config.fold * inBytes > INT_LIMIT || config.fold * outBytes > INT_LIMIT)
    {
        return std::nullopt;
    }
    MfmaLaunch launch;
    launch.flip = problem.flipFilter;
    launch.block = 64 * config.waves;
    const int64_t chunks = divUp(a.ho, config.rows);
    launch.grid
        = divUp(a.n, config.fold) * chunks
          * (channelsLast ? divUp(a.wo, 32 / config.fold) * (a.c / (8 * config.waves))
                          : divUp(a.wo, 128 / config.fold) * divUp(divUp(a.c, 2), config.waves));
    if(launch.grid < 1 || launch.grid > INT_LIMIT)
    {
        return std::nullopt;
    }
    auto& args = launch.args;
    args.n = a.n;
    args.c = a.c;
    args.hi = a.hi;
    args.wi = a.wi;
    args.ho = a.ho;
    args.wo = a.wo;
    args.ph = a.ph;
    args.pw = a.pw;
    args.rows = static_cast<int>(config.rows);
    args.chunks = static_cast<int>(chunks);
    return launch;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_depthwise_mfma

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
