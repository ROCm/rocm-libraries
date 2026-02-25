// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cmath>
#include <cstddef>

namespace hip_kernel_plugin
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
inline void GetLocalConfigNHWC(size_t c,
                               size_t h,
                               size_t w,
                               bool isFp32,
                               size_t min_workgroups,
                               size_t vectorsize,
                               size_t& xlocalsize,
                               size_t& ylocalsize)
{
    // Compute workgroup size
    unsigned int xlocalsize_limit = vectorsize > 1 ? (isFp32 ? 16 : 32) : 64;
    // shared memory size per workgroup is fixed
    unsigned int max_localsize = 1024 / vectorsize;

    size_t nworkgroups = 0;
    // decrease max_localsize until the number of workgroups is greater than 80%
    // of the available CUs
    while(nworkgroups < min_workgroups && max_localsize >= xlocalsize_limit && max_localsize > 64)
    {
        // xlocalsize must be power of 2 as reductions in the kernels rely on it, here c is rounded
        // up to next power of 2.
        xlocalsize
            = std::min(size_t{1} << static_cast<size_t>(std::ceil(std::log2(c / vectorsize))),
                       static_cast<size_t>(xlocalsize_limit));
        ylocalsize = max_localsize / xlocalsize;
        nworkgroups = ((c / vectorsize + xlocalsize - 1) / xlocalsize)
                      * ((h * w + ylocalsize - 1) / ylocalsize);
        max_localsize >>= 1;
    }
}

// Provide workgroup sizes for spatial multiple configuration.
// It returns the preferred spatial multiple configuration, which is used without tuning.
// If tuning is enabled, this configuration is also added to the group of instances.
inline void GetSpatialMultipleConfig(size_t c,
                                     size_t h,
                                     size_t w,
                                     bool isLayoutNHWC,
                                     bool isFp32,
                                     size_t min_workgroups,
                                     size_t vectorsize,
                                     size_t& xlocalsize,
                                     size_t& ylocalsize)
{
    // Initialize to safe defaults at the start of the function
    xlocalsize = 1;
    ylocalsize = 1;

    size_t in_cstride = h * w;

    if(isLayoutNHWC)
    {
        if(c % vectorsize != 0)
        {
            // xlocalsize and ylocalsize already initialized to 1
            return;
        }
        GetLocalConfigNHWC(c, h, w, isFp32, min_workgroups, vectorsize, xlocalsize, ylocalsize);
    }
    else
    {
        if(in_cstride % vectorsize != 0)
        {
            // xlocalsize and ylocalsize already initialized to 1
            return;
        }
        // xlocalsize stays at 1
        ylocalsize = 1024;
        if(ylocalsize > in_cstride / vectorsize)
        {
            // No need to use workgroups larger than the HW dimension
            ylocalsize = std::max(
                size_t{64},
                size_t{1} << static_cast<size_t>(std::ceil(std::log2(in_cstride / vectorsize))));
        }
    }
}

// Check if spatial multiple implementation can be used for a given problem
// and workgroup configuration.
inline bool IsSpatialMultipleApplicable(size_t n,
                                        size_t c,
                                        size_t h,
                                        size_t w,
                                        bool isLayoutNHWC,
                                        bool isFp32,
                                        size_t vectorsize,
                                        unsigned int stash_values,
                                        size_t ylocalsize,
                                        size_t zlocalsize,
                                        size_t nelements)
{
    unsigned int in_cstride = static_cast<unsigned int>(h * w);

    if(isLayoutNHWC)
    {
        // check if the provided vectorsize can be used
        if(c % vectorsize != 0)
        {
            return false;
        }

        stash_values *= (isFp32 ? 1 : 2);
        unsigned int last_ylocalsize = in_cstride % ylocalsize == 0
                                           ? static_cast<unsigned int>(ylocalsize)
                                           : in_cstride % ylocalsize;

        unsigned int last_zlocalsize = n % (zlocalsize * nelements) == 0
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
        if((!isFp32 && (c % 2 != 0 && last_zlocalsize < stash_values))
           || ((last_ylocalsize < stash_values) && (last_zlocalsize < stash_values)))
        {
            return false;
        }
    }
    else
    {
        // check if the provided vectorsize can be used
        if(in_cstride % vectorsize != 0)
        {
            return false;
        }

        unsigned int last_ylocalsize = in_cstride % ylocalsize == 0
                                           ? static_cast<unsigned int>(ylocalsize)
                                           : in_cstride % ylocalsize;

        unsigned int last_zlocalsize = n % (zlocalsize * nelements) == 0
                                           ? static_cast<unsigned int>(zlocalsize * nelements)
                                           : n % static_cast<unsigned int>(zlocalsize * nelements);
        // Restrictions:
        //  - last block must have enough space to stash intermediate results in HW dimension
        //  - if last block doesn't fit, intermediate results are stored in N dimension which must
        //    be large enough
        stash_values *= (isFp32 ? 1 : 2);
        if(last_ylocalsize < stash_values && last_zlocalsize < stash_values)
        {
            return false;
        }
    }
    return true;
}

inline bool UseMultiple(
    size_t n, size_t h, size_t w, bool isFp16OrBfp16Mix, bool isLayoutNHWC, Direction direction)
{
    unsigned int in_cstride = static_cast<unsigned int>(h * w);
    unsigned int in_nhw = static_cast<unsigned int>(n) * in_cstride;

    // Check heuristics (used to choose between spatial single and multiple for performance)
    if(!isLayoutNHWC && direction == Direction::BACKWARD
       && (!((in_nhw >= static_cast<unsigned int>(32 * 1024 * 1024) || in_cstride <= 1024)
             && (in_nhw >= static_cast<unsigned int>(32 * 1024 * 1024) || in_cstride <= 512)
             && in_cstride > 512)))
    {
        return false;
    }

    if(!isLayoutNHWC && direction == Direction::FORWARD_TRAINING
       && (!((n >= 3 && in_cstride > 512 && (in_nhw >= 33554432 || in_cstride <= 1024)
              && ((n < 256) || (in_cstride <= 60) || !isFp16OrBfp16Mix)
              && (!isFp16OrBfp16Mix || in_cstride <= 512))
             || ((n > 768) && (in_cstride > 150)))))
    {
        return false;
    }

    return true;
}

// Provide the stash method to use for spatial multiple implementation
inline int GetStashMethod(bool isLayoutNHWC,
                          bool isFp32,
                          unsigned int stash_values,
                          size_t c,
                          size_t n,
                          size_t in_cstride,
                          size_t ylocalsize,
                          size_t zlocalsize,
                          size_t nelements)
{
    // See `batchnorm_functions.hpp` for stash implementation of different methods
    int stash_method = 0;
    stash_values *= (isFp32 ? 1 : 2);
    unsigned int last_ylocalsize = (in_cstride) % ylocalsize == 0
                                       ? static_cast<unsigned int>(ylocalsize)
                                       : static_cast<unsigned int>((in_cstride) % ylocalsize);
    unsigned int last_zlocalsize = n % (zlocalsize * nelements) == 0
                                       ? static_cast<unsigned int>(zlocalsize * nelements)
                                       : n % static_cast<unsigned int>(zlocalsize * nelements);
    if(last_ylocalsize < stash_values && last_zlocalsize >= stash_values)
    {
        stash_method = 1;
    }
    if(isLayoutNHWC && !isFp32 && (c % 2 != 0) && (last_zlocalsize >= stash_values))
    {
        stash_method = 2;
    }
    return stash_method;
}

inline void DefaultConfigSpatialSingle(size_t n,
                                       size_t h,
                                       size_t w,
                                       bool isFp16Mix,
                                       bool isBfp16Mix,
                                       bool isLayoutNHWC,
                                       Direction direction,
                                       KernelConfig& config)
{
    unsigned int in_cstride = static_cast<unsigned int>(h * w);
    unsigned int in_nhw = static_cast<unsigned int>(n * in_cstride);

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
            if((in_cstride < 200) && (in_cstride > 60) && isFp16Mix)
            {
                config.variant = 1;
                config.vectorsize = 1;
                return;
            }

            // N*H*W < 32M and H*W > 1024
            // use batchnorm variant#1 implementation which parallelize
            // work groups over channels and loop through NHW.
            if((in_nhw < (32 * 1024 * 1024) && in_cstride > 1024))
            {
                config.variant = 1;
                config.vectorsize = 1;
                return;
            }
            // N*H*W < 32M and H*W > 512
            // use batchnorm variant#1 or variant#3 implementation which
            // parallelize work groups over channels and loop through N.
            else if(in_nhw < (32 * 1024 * 1024) && in_cstride > 512)
            {
                if(n >= 32)
                {
                    config.variant = 1;
                    config.vectorsize = 1;
                    return;
                }
                else
                {
                    config.variant = 3;
                    config.vectorsize = 1;
                    return;
                }
            }
            // H*W < 512
            // use batchnorm variant#0 or variant#3 implementation
            // based on batch size and H*W
            else if(in_cstride <= 512)
            {
                if((n > 64) && (in_cstride > 160))
                {
                    config.variant = 3;
                    config.vectorsize = 1;
                    return;
                }
                else
                {
                    config.variant = 0;
                    config.vectorsize = 1;
                    return;
                }
            }
        }
        else
        {
            // clang-format off
            if(in_cstride > 512 && in_cstride <= 1024 && n < 32)
            {
                config.variant = 3;
                config.vectorsize = 1;
                return;
            }

            if( (in_nhw < 33554432 && in_cstride > 1024) ||
            ((n >= 256) && (in_cstride > 60) && (isFp16Mix || isBfp16Mix)) ||
            ((in_cstride > 512) && (isFp16Mix || isBfp16Mix)))
            {
                config.variant = 1;
                config.vectorsize = 1;
                return;
            }
            else if(in_cstride <= 512)
            {
                config.variant = 0;
                config.vectorsize = 1;
                return;
            }
            // clang-format on
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
inline void DefaultConfigSpatialMultiple(size_t n,
                                         size_t c,
                                         size_t h,
                                         size_t w,
                                         bool isLayoutNHWC,
                                         bool isFp32,
                                         size_t min_workgroups,
                                         unsigned int stash_values,
                                         KernelConfig& config)
{
    size_t xlocalsize_default = 0;
    size_t ylocalsize_default = 0;
    size_t vectorsize_default = 4;
    size_t zlocalsize_default = 1;
    size_t nelements_default = n;

    if(isLayoutNHWC)
    {
        // First add the default instance, which should work well for a large range of problems
        {
            GetSpatialMultipleConfig(c,
                                     h,
                                     w,
                                     isLayoutNHWC,
                                     isFp32,
                                     min_workgroups,
                                     vectorsize_default,
                                     xlocalsize_default,
                                     ylocalsize_default);

            if(IsSpatialMultipleApplicable(n,
                                           c,
                                           h,
                                           w,
                                           isLayoutNHWC,
                                           isFp32,
                                           vectorsize_default,
                                           stash_values,
                                           ylocalsize_default,
                                           zlocalsize_default,
                                           nelements_default))
            {
                config.variant = 2;
                config.vectorsize = vectorsize_default;
                config.xlocalsize = xlocalsize_default;
                config.ylocalsize = ylocalsize_default;
                config.zlocalsize = zlocalsize_default;
                config.nelements = nelements_default;
            }
            else
            {
                if(vectorsize_default > 1)
                {
                    vectorsize_default = 1;
                    GetSpatialMultipleConfig(c,
                                             h,
                                             w,
                                             isLayoutNHWC,
                                             isFp32,
                                             min_workgroups,
                                             vectorsize_default,
                                             xlocalsize_default,
                                             ylocalsize_default);

                    if(IsSpatialMultipleApplicable(n,
                                                   c,
                                                   h,
                                                   w,
                                                   isLayoutNHWC,
                                                   isFp32,
                                                   vectorsize_default,
                                                   stash_values,
                                                   ylocalsize_default,
                                                   zlocalsize_default,
                                                   nelements_default))
                    {
                        config.variant = 2;
                        config.vectorsize = vectorsize_default;
                        config.xlocalsize = xlocalsize_default;
                        config.ylocalsize = ylocalsize_default;
                        config.zlocalsize = zlocalsize_default;
                        config.nelements = nelements_default;
                    }
                }
            }
        }

        // NOTE: We can add more instances for tuning here but we don't have
        // the tuning infrastructure in place yet, so we are adding only one
        // instance.
        return;
    }
    else
    {
        // For NCHW we add all the supported vector sizes smaller than the default (if they are
        // applicable)
        while(vectorsize_default > 0)
        {
            GetSpatialMultipleConfig(c,
                                     h,
                                     w,
                                     isLayoutNHWC,
                                     isFp32,
                                     min_workgroups,
                                     vectorsize_default,
                                     xlocalsize_default,
                                     ylocalsize_default);

            if(IsSpatialMultipleApplicable(n,
                                           c,
                                           h,
                                           w,
                                           isLayoutNHWC,
                                           isFp32,
                                           vectorsize_default,
                                           stash_values,
                                           ylocalsize_default,
                                           zlocalsize_default,
                                           nelements_default))
            {
                config.variant = 2;
                config.vectorsize = vectorsize_default;
                config.xlocalsize = xlocalsize_default;
                config.ylocalsize = ylocalsize_default;
                config.zlocalsize = zlocalsize_default;
                config.nelements = nelements_default;
            }
            vectorsize_default >>= 1;
        }
    }
}

} // namespace batchnorm

} // namespace hip_kernel_plugin
