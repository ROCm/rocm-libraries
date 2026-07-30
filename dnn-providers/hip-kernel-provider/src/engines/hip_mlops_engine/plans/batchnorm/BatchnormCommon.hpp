// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cmath>
#include <cstddef>

namespace hip_kernel_provider
{
namespace batchnorm
{

enum class Direction : int
{
    FORWARD_TRAINING = 0,
    FORWARD_INFERENCE = 1,
    BACKWARD = 2
};

struct KernelConfig
{
    int variant = -1;
    size_t vectorsize = 1;
    size_t xlocalsize = 1;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;
    size_t nelements = 1;
};

// Compute workgroup size configuration given a problem (NHWC) and a vectorsize
// It supports only 2D workgroups
inline void getLocalConfigNHWC(size_t c,
                               size_t h,
                               size_t w,
                               bool isFp32,
                               size_t minWorkgroups,
                               size_t vectorsize,
                               size_t& xlocalsize,
                               size_t& ylocalsize)
{
    // Compute workgroup size
    unsigned int xlocalsizeLimit = 64;
    if(vectorsize > 1)
    {
        xlocalsizeLimit = isFp32 ? 16 : 32;
    }

    // shared memory size per workgroup is fixed
    unsigned int maxLocalsize = 1024 / vectorsize;

    // default local config in case the while loop is not entered
    xlocalsize = std::min(
        size_t{1} << static_cast<size_t>(std::ceil(std::log2(std::max(c / vectorsize, size_t{1})))),
        static_cast<size_t>(xlocalsizeLimit));
    ylocalsize = maxLocalsize / xlocalsize;

    size_t nworkgroups = 0;
    // decrease maxLocalsize until the number of workgroups is greater than 80%
    // of the available CUs
    while(nworkgroups < minWorkgroups && maxLocalsize >= xlocalsizeLimit && maxLocalsize > 64)
    {
        // xlocalsize must be power of 2 as reductions in the kernels rely on it, here c is rounded
        // up to next power of 2.
        xlocalsize = std::min(size_t{1} << static_cast<size_t>(
                                  std::ceil(std::log2(std::max(c / vectorsize, size_t{1})))),
                              static_cast<size_t>(xlocalsizeLimit));
        ylocalsize = maxLocalsize / xlocalsize;
        nworkgroups = ((c / vectorsize + xlocalsize - 1) / xlocalsize)
                      * ((h * w + ylocalsize - 1) / ylocalsize);
        maxLocalsize >>= 1;
    }
}

// Provide workgroup sizes for spatial multiple configuration.
// It returns the preferred spatial multiple configuration, which is used without tuning.
// If tuning is enabled, this configuration is also added to the group of instances.
inline void getSpatialMultipleConfig(size_t c,
                                     size_t h,
                                     size_t w,
                                     bool isLayoutNHWC,
                                     bool isFp32,
                                     size_t minWorkgroups,
                                     size_t vectorsize,
                                     size_t& xlocalsize,
                                     size_t& ylocalsize)
{
    // Initialize to safe defaults at the start of the function
    xlocalsize = 1;
    ylocalsize = 1;

    const size_t inCstride = h * w;

    if(isLayoutNHWC)
    {
        if(c % vectorsize != 0)
        {
            // xlocalsize and ylocalsize already initialized to 1
            return;
        }
        getLocalConfigNHWC(c, h, w, isFp32, minWorkgroups, vectorsize, xlocalsize, ylocalsize);
    }
    else
    {
        if(inCstride % vectorsize != 0)
        {
            // xlocalsize and ylocalsize already initialized to 1
            return;
        }
        // xlocalsize stays at 1
        ylocalsize = 1024;
        if(ylocalsize > inCstride / vectorsize)
        {
            // No need to use workgroups larger than the HW dimension
            ylocalsize = std::max(
                size_t{64},
                size_t{1} << static_cast<size_t>(std::ceil(std::log2(inCstride / vectorsize))));
        }
    }
}

// Check if spatial multiple implementation can be used for a given problem
// and workgroup configuration.
inline bool isSpatialMultipleApplicable(size_t n,
                                        size_t c,
                                        size_t h,
                                        size_t w,
                                        bool isLayoutNHWC,
                                        bool isFp32,
                                        size_t vectorsize,
                                        unsigned int stashValues,
                                        size_t ylocalsize,
                                        size_t zlocalsize,
                                        size_t nelements)
{
    const auto inCstride = static_cast<unsigned int>(h * w);

    if(isLayoutNHWC)
    {
        // check if the provided vectorsize can be used
        if(c % vectorsize != 0)
        {
            return false;
        }

        stashValues *= (isFp32 ? 1 : 2);
        const unsigned int lastYlocalsize = inCstride % ylocalsize == 0
                                                ? static_cast<unsigned int>(ylocalsize)
                                                : inCstride % ylocalsize;

        const unsigned int lastZocalsize
            = n % (zlocalsize * nelements) == 0
                  ? static_cast<unsigned int>(zlocalsize * nelements)
                  : n % static_cast<unsigned int>(zlocalsize * nelements);

        // FP32:
        //  - last block must have enough space to stash intermediate results in HW dimension
        //  - if last block doesn't fit, intermediate results are stored in N dimension which must
        //    be large enough
        // Mix precision:
        //  - last block must have enough space to stash intermediate results in HW dimension
        //  - if last block doesn't fit, intermediate results are stored in N dimension which must
        //    be large enough
        //  - if C is not multiple of 2, intermediate results are stored in N dimension splitting
        //    float values in group of 2 bytes. N must be large enough
        if((!isFp32 && (c % 2 != 0 && lastZocalsize < stashValues))
           || ((lastYlocalsize < stashValues) && (lastZocalsize < stashValues)))
        {
            return false;
        }
    }
    else
    {
        // check if the provided vectorsize can be used
        if(inCstride % vectorsize != 0)
        {
            return false;
        }

        const unsigned int lastYlocalsize = inCstride % ylocalsize == 0
                                                ? static_cast<unsigned int>(ylocalsize)
                                                : inCstride % ylocalsize;

        const unsigned int lastZocalsize
            = n % (zlocalsize * nelements) == 0
                  ? static_cast<unsigned int>(zlocalsize * nelements)
                  : n % static_cast<unsigned int>(zlocalsize * nelements);
        // Restrictions:
        //  - last block must have enough space to stash intermediate results in HW dimension
        //  - if last block doesn't fit, intermediate results are stored in N dimension which must
        //    be large enough
        stashValues *= (isFp32 ? 1 : 2);
        if(lastYlocalsize < stashValues && lastZocalsize < stashValues)
        {
            return false;
        }
    }
    return true;
}

inline bool useMultiple(
    size_t n, size_t h, size_t w, bool isFp16OrBfp16Mix, bool isLayoutNHWC, Direction direction)
{
    const auto inCstride = static_cast<unsigned int>(h * w);
    const auto inNhw = static_cast<unsigned int>(n) * inCstride;
    const auto thrInNhw = static_cast<unsigned int>(32 * 1024 * 1024);

    if(!isLayoutNHWC && direction == Direction::BACKWARD)
    {
        if((inNhw < thrInNhw && inCstride > 1024) || (inNhw < thrInNhw && inCstride > 512)
           || inCstride <= 512)
        {
            return false;
        }
    }

    if(!isLayoutNHWC && direction == Direction::FORWARD_TRAINING)
    {
        const bool condition1 = (n < 3) || (inCstride <= 512)
                                || (inNhw < 33554432 && inCstride > 1024)
                                || (n >= 256 && inCstride > 60 && isFp16OrBfp16Mix)
                                || (isFp16OrBfp16Mix && inCstride > 512);

        const bool condition2 = (n <= 768) || (inCstride <= 150);

        if(condition1 && condition2)
        {
            return false;
        }
    }

    return true;
}

// Provide the stash method to use for spatial multiple implementation
inline int getStashMethod(bool isLayoutNHWC,
                          bool isFp32,
                          unsigned int stashValues,
                          size_t c,
                          size_t n,
                          size_t inCstride,
                          size_t ylocalsize,
                          size_t zlocalsize,
                          size_t nelements)
{
    // See `batchnorm_functions.hpp` for stash implementation of different methods
    int stashMethod = 0;
    stashValues *= (isFp32 ? 1 : 2);
    const unsigned int lastYlocalsize = (inCstride) % ylocalsize == 0
                                            ? static_cast<unsigned int>(ylocalsize)
                                            : static_cast<unsigned int>((inCstride) % ylocalsize);
    const unsigned int lastZocalsize = n % (zlocalsize * nelements) == 0
                                           ? static_cast<unsigned int>(zlocalsize * nelements)
                                           : n % static_cast<unsigned int>(zlocalsize * nelements);
    if(lastYlocalsize < stashValues && lastZocalsize >= stashValues)
    {
        stashMethod = 1;
    }
    if(isLayoutNHWC && !isFp32 && (c % 2 != 0) && (lastZocalsize >= stashValues))
    {
        stashMethod = 2;
    }
    return stashMethod;
}

inline void defaultConfigSpatialSingle(size_t n,
                                       size_t h,
                                       size_t w,
                                       bool isFp16Mix,
                                       bool isBfp16Mix,
                                       bool isLayoutNHWC,
                                       Direction direction,
                                       KernelConfig& config)
{
    const auto inCstride = static_cast<unsigned int>(h * w);
    const auto inNhw = static_cast<unsigned int>(n * inCstride);

    // NCHW supports also variants 0 and 3 which can be much faster than
    // variant 1 but have more restrictions. Here we decide if we use variant
    // 0, 1, 3
    // In case variant 0 or 3 are selected, we add also variant 1 for tuning.
    // Almost always variant 0 and 3 will be faster than variant 1 but
    // we add the latter for tuning to be sure and because it is cheap to run.
    // NOTE: Currently we don't have the tuning infrastructure in place, so we
    // are only selecting one variant to run based on heuristics.
    if(!isLayoutNHWC)
    {
        if(direction == Direction::BACKWARD)
        {
            if((inCstride < 200) && (inCstride > 60) && isFp16Mix)
            {
                config.variant = 1;
                config.vectorsize = 1;
                return;
            }

            // N*H*W < 32M and H*W > 1024
            // use batchnorm variant#1 implementation which parallelize
            // work groups over channels and loop through NHW.
            if((inNhw < (32 * 1024 * 1024) && inCstride > 1024))
            {
                config.variant = 1;
                config.vectorsize = 1;
                return;
            }

            // N*H*W < 32M and H*W > 512
            // use batchnorm variant#1 or variant#3 implementation which
            // parallelize work groups over channels and loop through N.
            if(inNhw < (32 * 1024 * 1024) && inCstride > 512)
            {
                if(n >= 32)
                {
                    config.variant = 1;
                    config.vectorsize = 1;
                    return;
                }

                config.variant = 3;
                config.vectorsize = 1;
                return;
            }

            // H*W < 512
            // use batchnorm variant#0 or variant#3 implementation
            // based on batch size and H*W
            if(inCstride <= 512)
            {
                if((n > 64) && (inCstride > 160))
                {
                    config.variant = 3;
                    config.vectorsize = 1;
                    return;
                }

                config.variant = 0;
                config.vectorsize = 1;
                return;
            }
        }
        else
        {
            if(inCstride > 512 && inCstride <= 1024 && n < 32)
            {
                config.variant = 3;
                config.vectorsize = 1;
                return;
            }

            if((inNhw < 33554432 && inCstride > 1024)
               || ((n >= 256) && (inCstride > 60) && (isFp16Mix || isBfp16Mix))
               || ((inCstride > 512) && (isFp16Mix || isBfp16Mix)))
            {
                config.variant = 1;
                config.vectorsize = 1;
                return;
            }

            config.variant = 0;
            config.vectorsize = 1;
            return;
        }
        config.variant = 1;
        config.vectorsize = 1;
    }
    else
    {
        config.variant = 1;
        config.vectorsize = 1;
    }
}

// Add spatial multiple instances for given problem.
// The first instance added is based on heuristics and is the default one if spatial
// multiple is the default method.
// Additional instances are added:
//  - for NCHW all supported vector sizes smaller than the default one
//    (the default is the largest applicable)
//  - for NHWC an hybrid approach is used, xlocalsize and vectorsize are set using heuristics,
//    while ylocalsize, zlocalsize and nelements are added to the tuning with some
//    additional restrictions based on heuristics to keep the number of instances low
inline void defaultConfigSpatialMultiple(size_t n,
                                         size_t c,
                                         size_t h,
                                         size_t w,
                                         bool isLayoutNHWC,
                                         bool isFp32,
                                         size_t minWorkgroups,
                                         unsigned int stashValues,
                                         KernelConfig& config)
{
    size_t xlocalsizeDefault = 1;
    size_t ylocalsizeDefault = 1;
    const size_t zlocalsizeDefault = 1;
    size_t vectorsizeDefault = 4;
    const size_t nelementsDefault = n;

    if(isLayoutNHWC)
    {
        // First add the default instance, which should work well for a large range of problems
        {
            getSpatialMultipleConfig(c,
                                     h,
                                     w,
                                     isLayoutNHWC,
                                     isFp32,
                                     minWorkgroups,
                                     vectorsizeDefault,
                                     xlocalsizeDefault,
                                     ylocalsizeDefault);

            if(isSpatialMultipleApplicable(n,
                                           c,
                                           h,
                                           w,
                                           isLayoutNHWC,
                                           isFp32,
                                           vectorsizeDefault,
                                           stashValues,
                                           ylocalsizeDefault,
                                           zlocalsizeDefault,
                                           nelementsDefault))
            {
                config.variant = 2;
                config.vectorsize = vectorsizeDefault;
                config.xlocalsize = xlocalsizeDefault;
                config.ylocalsize = ylocalsizeDefault;
                config.zlocalsize = zlocalsizeDefault;
                config.nelements = nelementsDefault;
            }
            else
            {
                if(vectorsizeDefault > 1)
                {
                    vectorsizeDefault = 1;
                    getSpatialMultipleConfig(c,
                                             h,
                                             w,
                                             isLayoutNHWC,
                                             isFp32,
                                             minWorkgroups,
                                             vectorsizeDefault,
                                             xlocalsizeDefault,
                                             ylocalsizeDefault);

                    if(isSpatialMultipleApplicable(n,
                                                   c,
                                                   h,
                                                   w,
                                                   isLayoutNHWC,
                                                   isFp32,
                                                   vectorsizeDefault,
                                                   stashValues,
                                                   ylocalsizeDefault,
                                                   zlocalsizeDefault,
                                                   nelementsDefault))
                    {
                        config.variant = 2;
                        config.vectorsize = vectorsizeDefault;
                        config.xlocalsize = xlocalsizeDefault;
                        config.ylocalsize = ylocalsizeDefault;
                        config.zlocalsize = zlocalsizeDefault;
                        config.nelements = nelementsDefault;
                    }
                }
            }
        }

        // NOTE: We can add more instances for tuning here but we don't have
        // the tuning infrastructure in place yet, so we are adding only one
        // instance.
        return;
    }

    // For NCHW we add all the supported vector sizes smaller than the default (if they are
    // applicable)
    while(vectorsizeDefault > 0)
    {
        getSpatialMultipleConfig(c,
                                 h,
                                 w,
                                 isLayoutNHWC,
                                 isFp32,
                                 minWorkgroups,
                                 vectorsizeDefault,
                                 xlocalsizeDefault,
                                 ylocalsizeDefault);

        if(isSpatialMultipleApplicable(n,
                                       c,
                                       h,
                                       w,
                                       isLayoutNHWC,
                                       isFp32,
                                       vectorsizeDefault,
                                       stashValues,
                                       ylocalsizeDefault,
                                       zlocalsizeDefault,
                                       nelementsDefault))
        {
            config.variant = 2;
            config.vectorsize = vectorsizeDefault;
            config.xlocalsize = xlocalsizeDefault;
            config.ylocalsize = ylocalsizeDefault;
            config.zlocalsize = zlocalsizeDefault;
            config.nelements = nelementsDefault;
        }
        vectorsizeDefault >>= 1;
    }
}

} // namespace batchnorm

} // namespace hip_kernel_provider
