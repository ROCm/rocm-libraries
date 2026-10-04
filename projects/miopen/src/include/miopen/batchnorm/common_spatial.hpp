/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/batchnorm/problem_description.hpp>
#include <limits>

namespace miopen {

namespace solver {

namespace batchnorm {

// Maximum vector width supported by the batch-norm spatial kernels.
// Matches the DEFINE_VECTOR_MAPPING entries in src/kernels/vector_types.hpp and
// the static_assert allowlists in src/kernels/miopen_math.hpp.
// Increment this constant (and add the corresponding DEFINE_VECTOR_MAPPING) when
// adding a new vector width.
inline constexpr size_t kMaxSupportedVectorSize = 8;

// Compute workgroup size configuration given a problem (NHWC) and a vectorsize
// It supports only 2D workgroups
inline void GetLocalConfigNHWC(const miopen::batchnorm::ProblemDescription& problem,
                               size_t vectorsize,
                               size_t& xlocalsize,
                               size_t& ylocalsize)
{
    bool bfp32parm =
        problem.GetXDesc().GetType() == miopenHalf || problem.GetXDesc().GetType() == miopenBFloat16
            ? false
            : true;

    size_t n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    // Compute workgroup size
    unsigned int xlocalsize_limit = vectorsize > 1 ? (bfp32parm ? 16 : 32) : 64;
    // shared memory size per workgroup is fixed
    unsigned int max_localsize = 1024 / vectorsize;

    size_t nworkgroups = 0;
    // decrease max_localsize until the number of workgroups is greater than 80%
    // of the available CUs
    while(nworkgroups < problem.GetMinWorkgroups() && max_localsize >= xlocalsize_limit &&
          max_localsize > 64)
    {
        // xlocalsize must be power of 2 as reductions in the kernels rely on it, here c is rounded
        // up to next power of 2.
        xlocalsize  = std::min(size_t{1 << int(std::ceil(std::log2(c / vectorsize)))},
                              size_t{xlocalsize_limit});
        ylocalsize  = max_localsize / xlocalsize;
        nworkgroups = ((c / vectorsize + xlocalsize - 1) / xlocalsize) *
                      ((h * w + ylocalsize - 1) / ylocalsize);
        max_localsize >>= 1;
    }
}

// Provide workgroup sizes for spatial multiple configuration.
// It returns the preferred spatial multiple configuration, which is used without tuning.
// If tuning is enabled, this configuration is also added to the group of instances.
inline void GetSpatialMultipleConfig(const miopen::batchnorm::ProblemDescription& problem,
                                     size_t vectorsize,
                                     size_t& xlocalsize,
                                     size_t& ylocalsize)
{
    // Initialize to safe defaults at the start of the function
    xlocalsize = 1;
    ylocalsize = 1;

    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    if(problem.IsLayoutNHWC())
    {
        if(c % vectorsize != 0)
        {
            // xlocalsize and ylocalsize already initialized to 1
            return;
        }
        GetLocalConfigNHWC(problem, vectorsize, xlocalsize, ylocalsize);
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
            ylocalsize = std::max(size_t{64},
                                  size_t{1 << int(std::ceil(std::log2(in_cstride / vectorsize)))});
        }
    }
}

// Return true if spatial multiple is the preferred method to be used.
// The function is based on heuristics and it returns always true for NHWC.
inline bool UseMultiple(const miopen::batchnorm::ProblemDescription& problem)
{
    size_t n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    bool bfpmixparm = (problem.GetXDesc().GetType() == miopenHalf ||
                       problem.GetXDesc().GetType() == miopenBFloat16) &&
                              problem.GetBnScale().GetType() == miopenFloat
                          ? true
                          : false;

    unsigned int in_cstride = h * w;
    unsigned int in_nhw     = n * in_cstride;
    // Check heuristics (used to choose between spatial single and multiple for performance)
    if(!problem.IsLayoutNHWC() &&
       problem.GetDirection() == miopen::batchnorm::Direction::Backward &&
       (!((in_nhw >= static_cast<size_t>(32 * 1024 * 1024) || in_cstride <= 1024) &&
          (in_nhw >= static_cast<size_t>(32 * 1024 * 1024) || in_cstride <= 512) &&
          in_cstride > 512)))
    {
        return false;
    }

    if(!problem.IsLayoutNHWC() &&
       problem.GetDirection() == miopen::batchnorm::Direction::ForwardTraining &&
       (!((n >= 3 && in_cstride > 512 && (in_nhw >= 33554432 || in_cstride <= 1024) &&
           ((n < 256) || (in_cstride <= 60) || !bfpmixparm) &&
           (!bfpmixparm || in_cstride <= 512)) ||
          ((n > 768) && (in_cstride > 150)))))
    {
        return false;
    }

    return true;
}

// Provide the stash method to use for spatial multiple implementation
inline int GetStashMethod(bool IsLayoutNHWC,
                          miopenDataType_t problem_type,
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
    stash_values *= (problem_type == miopenFloat ? 1 : 2);
    unsigned int last_ylocalsize =
        (in_cstride) % ylocalsize == 0 ? ylocalsize : (in_cstride) % ylocalsize;
    unsigned int last_zlocalsize =
        n % (zlocalsize * nelements) == 0 ? (zlocalsize * nelements) : n % (zlocalsize * nelements);
    if(last_ylocalsize < stash_values && last_zlocalsize >= static_cast<size_t>(stash_values))
    {
        stash_method = 1;
    }
    if(IsLayoutNHWC && !(problem_type == miopenFloat) && (c % 2 != 0) &&
       (last_zlocalsize >= stash_values))
    {
        stash_method = 2;
    }
    return stash_method;
}

// Legacy spatial single variants (0, 1, 3)
// Variant<variant>-<vectorsize>
inline std::string GetKernelIdFromVariant(int variant, size_t vectorsize)
{
    std::stringstream stream;
    stream << "Variant" << variant << "-" << vectorsize;
    return stream.str();
}

// Spatial multiple and buffered spatial single (variant 4)
// Variant<variant>-<vectorsize>-<xlocalsize>-<ylocalsize>-<zlocalsize>-<nelements>
inline std::string GetKernelIdFromVariant(int variant,
                                          size_t vectorsize,
                                          size_t xlocalsize,
                                          size_t ylocalsize,
                                          size_t zlocalsize,
                                          size_t nelements)
{
    std::stringstream stream;
    stream << "Variant" << variant << "-" << vectorsize << "-" << xlocalsize << "-" << ylocalsize
           << "-" << zlocalsize << "-" << nelements;
    return stream.str();
}

// Return tuning parameters from kernel_id string
// Variant 2 and specialized variants 4–6 and 8 store explicit dimensions.
inline void GetVariantFromKernelId(const std::string& kernel_id,
                                   int& variant,
                                   size_t& vectorsize,
                                   size_t& xlocalsize,
                                   size_t& ylocalsize,
                                   size_t& zlocalsize,
                                   size_t& nelements)
{
    std::stringstream iss(&kernel_id[7]);
    std::string segment;
    std::vector<std::string> seglist;

    while(std::getline(iss, segment, '-'))
    {
        seglist.push_back(segment);
    }
    variant    = std::stoi(seglist[0]);
    vectorsize = std::stoi(seglist[1]);
    if(variant != 2 && variant != 4 && variant != 5 && variant != 6 && variant != 8)
    {
        // For variant 0, 1, 3 (spatial single), kernel_id only contains
        // variant and vectorsize. The workgroup sizes (xlocalsize, ylocalsize,
        // zlocalsize, nelements) are not stored in kernel_id and must be
        // computed by the caller based on problem-size heuristics.
        return;
    }
    if((variant == 4 || variant == 5 || variant == 6 || variant == 8) && seglist.size() != 6)
    {
        xlocalsize = ylocalsize = zlocalsize = nelements = 0;
        return;
    }
    xlocalsize = std::stoi(seglist[2]);
    ylocalsize = std::stoi(seglist[3]);
    zlocalsize = std::stoi(seglist[4]);
    nelements  = std::stoi(seglist[5]);
}

// Variant 4 retains each thread's FP16/BF16 input in packed registers. Keep the same
// 256-byte bound as the kernel, including vector padding in the last slot.
inline bool IsSpatialBufferedApplicable(const miopen::batchnorm::ProblemDescription& problem,
                                        size_t vectorsize,
                                        size_t blocksize)
{
    const auto input_type = problem.GetXDesc().GetType();
    if(problem.GetDirection() != miopen::batchnorm::Direction::ForwardTraining ||
       problem.GetMode() != miopenBNSpatial || !problem.Is2D() || !problem.IsLayoutNCHW() ||
       (input_type != miopenHalf && input_type != miopenBFloat16) || !problem.IsScaleFp32() ||
       problem.GetBnBias().GetType() != miopenFloat ||
       problem.GetYDesc().GetType() != input_type || !problem.GetXDesc().IsPacked() ||
       !problem.GetYDesc().IsPacked() || !problem.GetBnScale().IsPacked() ||
       !problem.GetBnBias().IsPacked() ||
       problem.GetXDesc().GetLengths() != problem.GetYDesc().GetLengths() ||
       (vectorsize != 4 && vectorsize != 8) ||
       (blocksize != 256 && blocksize != 512 && blocksize != 1024))
        return false;

    size_t n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());
    if(n == 0 || n > 64 || c == 0 || h == 0 || h > 4800 || w == 0 || w > 4800 / h)
        return false;
    const size_t hw  = h * w;
    const size_t nhw = n * hw;
    const size_t slots = (nhw / vectorsize + blocksize - 1) / blocksize;
    return hw % vectorsize == 0 && slots * vectorsize * GetTypeSize(input_type) <= 256 &&
           c <= std::numeric_limits<unsigned int>::max() / nhw;
}

inline void AddSpatialBufferedConfigs(const miopen::batchnorm::ProblemDescription& problem,
                                      std::vector<std::string>& valid_kernels)
{
    // Device/wave restrictions and conservative untuned selection are handled by
    // the forward solver; append candidates without changing legacy tuning order.
    for(const size_t blocksize : {1024, 512, 256})
    {
        for(const size_t vectorsize : {4, 8})
        {
            if(IsSpatialBufferedApplicable(problem, vectorsize, blocksize))
                valid_kernels.push_back(
                    GetKernelIdFromVariant(4, vectorsize, blocksize, 1, 1, 1));
        }
    }
}

inline bool IsSpatialStreamingApplicable(const miopen::batchnorm::ProblemDescription& problem,
                                         size_t vectorsize,
                                         size_t blocksize)
{
    if(problem.GetDirection() != miopen::batchnorm::Direction::ForwardTraining ||
       problem.GetMode() != miopenBNSpatial || !problem.Is2D() || !problem.IsLayoutNCHW() ||
       !problem.IsBFp16() || !problem.IsScaleFp32() ||
       problem.GetBnBias().GetType() != miopenFloat ||
       problem.GetYDesc().GetType() != miopenBFloat16 || !problem.GetXDesc().IsPacked() ||
       !problem.GetYDesc().IsPacked() || !problem.GetBnScale().IsPacked() ||
       !problem.GetBnBias().IsPacked() ||
       problem.GetXDesc().GetLengths() != problem.GetYDesc().GetLengths() ||
       (vectorsize != 4 && vectorsize != 8) ||
       (blocksize != 256 && blocksize != 512 && blocksize != 1024))
        return false;
    const auto& lengths = problem.GetXDesc().GetLengths();
    const size_t n = lengths[0], c = lengths[1], h = lengths[2], w = lengths[3];
    if(n == 0 || n > 64 || c == 0 || h == 0 || h > 8192 || w == 0 || w > 8192 / h)
        return false;
    const size_t hw = h * w, nhw = n * hw;
    return hw % vectorsize == 0 && c <= std::numeric_limits<unsigned int>::max() / nhw;
}

inline void AddSpatialStreamingConfigs(const miopen::batchnorm::ProblemDescription& problem,
                                       std::vector<std::string>& valid_kernels)
{
    if(problem.GetXDesc().GetLengths()[1] < 128)
        return;
    for(const size_t blocksize : {1024, 512, 256})
        for(const size_t vectorsize : {4, 8})
            if(IsSpatialStreamingApplicable(problem, vectorsize, blocksize))
                valid_kernels.push_back(
                    GetKernelIdFromVariant(6, vectorsize, blocksize, 1, 1, 1));
}

// Add spatial single instances for given problem
inline void DefaultConfigSpatialSingle(const miopen::batchnorm::ProblemDescription& problem,
                                       std::vector<std::string>& valid_kernels)
{
    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nhw     = n * in_cstride;

    bool bfpmixparm =
        problem.GetXDesc().GetType() == miopenHalf && problem.GetBnScale().GetType() == miopenFloat
            ? true
            : false;

    bool bbfpmixparam = problem.GetXDesc().GetType() == miopenBFloat16 &&
                                problem.GetBnScale().GetType() == miopenFloat
                            ? true
                            : false;

    // NCHW supports also variants 0 and 3 which can be much faster than
    // variant 1 but have more restrictions. Here we decide if we use variant
    // 0, 1, 3
    // In case variant 0 or 3 are selected, we add also variant 1 for tuning.
    // Almost always variant 0 and 3 will be faster than variant 1 but
    // we add the latter for tuning to be sure and because it is cheap
    if(!problem.IsLayoutNHWC())
    {
        if(problem.GetDirection() == miopen::batchnorm::Direction::Backward)
        {
            if((in_cstride < 200) && (in_cstride > 60) && bfpmixparm)
            {
                valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                return;
            }

            // N*H*W < 32M and H*W > 1024
            // use batchnorm variant#1 implementation which parallelize
            // work groups over channels and loop through NHW.
            if((in_nhw < (32 * 1024 * 1024) && in_cstride > 1024))
            {
                valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                return;
            }
            // N*H*W < 32M and H*W > 512
            // use batchnorm variant#1 or variant#3 implementation which
            // parallelize work groups over channels and loop through N.
            else if(in_nhw < (32 * 1024 * 1024) && in_cstride > 512)
            {
                if(n >= 32)
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }
                else
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(3, 1));
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }
            }
            // H*W < 512  use batchnorm variant#0 or variant#3 implementation
            // based on batch size and H*W
            else if(in_cstride <= 512)
            {
                if((n > 64) && (in_cstride > 160))
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(3, 1));
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }
                else
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(0, 1));
                    valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                    return;
                }
            }
        }
        else
        {
            // clang-format off
            if(in_cstride > 512 && in_cstride <= 1024 && n < 32)
            {
                valid_kernels.push_back(GetKernelIdFromVariant(3, 1));
                valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                return;
            }

            if( (in_nhw < 33554432 && in_cstride > 1024) ||
            ((n >= 256) && (in_cstride > 60) && (bfpmixparm || bbfpmixparam)) ||
            ((in_cstride > 512) && (bfpmixparm || bbfpmixparam)))
            {
                valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                if(in_cstride <= 512)
                {
                    valid_kernels.push_back(GetKernelIdFromVariant(0, 1));
                }
                return;
            }
            else if(in_cstride <= 512)
            {
                valid_kernels.push_back(GetKernelIdFromVariant(0, 1));
                valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
                return;
            }
            // clang-format on
        }
        valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
    }
    else
    {
        valid_kernels.push_back(GetKernelIdFromVariant(1, 1));
    }
}

// Check if spatial multiple implementation can be used for a given problem
// and workgroup configuration.
inline bool IsSpatialMultipleApplicable(const miopen::batchnorm::ProblemDescription& problem,
                                        size_t vectorsize,
                                        unsigned int stash_values,
                                        size_t ylocalsize,
                                        size_t zlocalsize,
                                        size_t nelements)
{
    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    if(problem.IsLayoutNHWC())
    {
        // check if the provided vectorsize can be used
        if(c % vectorsize != 0)
        {
            return false;
        }

        bool bfp32parm = problem.GetXDesc().GetType() == miopenHalf ||
                                 problem.GetXDesc().GetType() == miopenBFloat16
                             ? false
                             : true;

        stash_values *= (bfp32parm ? 1 : 2);
        unsigned int last_ylocalsize =
            in_cstride % ylocalsize == 0 ? ylocalsize : in_cstride % ylocalsize;

        unsigned int last_zlocalsize = n % (zlocalsize * nelements) == 0
                                           ? (zlocalsize * nelements)
                                           : n % (zlocalsize * nelements);

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
        if((!bfp32parm && (c % 2 != 0 && last_zlocalsize < static_cast<size_t>(stash_values))) ||
           ((last_ylocalsize < stash_values) &&
            (last_zlocalsize < static_cast<size_t>(stash_values))))
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

        unsigned int last_ylocalsize =
            in_cstride % ylocalsize == 0 ? ylocalsize : in_cstride % ylocalsize;

        unsigned int last_zlocalsize = n % (zlocalsize * nelements) == 0
                                           ? (zlocalsize * nelements)
                                           : n % (zlocalsize * nelements);
        // Restrictions:
        //  - last block must have enough space to stash intermediate results in HW dimension
        //  - if last block doesn't fit, intermediate results are stored in N dimension which must
        //    be large enough
        stash_values *= (problem.GetXDesc().GetType() == miopenFloat ? 1 : 2);
        if(last_ylocalsize < stash_values && last_zlocalsize < static_cast<size_t>(stash_values))
        {
            return false;
        }
    }
    return true;
}

// Set vectorsize and xlocalsize for NHWC (heuristics based approach)
inline void GetHeuristicsConfigTuningNHWC(const miopen::batchnorm::ProblemDescription& problem,
                                          size_t& vectorsize,
                                          size_t& xlocalsize)
{
    size_t n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(problem.GetXDesc().GetLengths());
    size_t in_cstride    = h * w;

    // if c is not a power of 2, set vectorsize and xlocalsize pair to have modulo equal
    // to zero or the highest possible in order to minimize the number of inactive threads
    size_t c_next_pow2 = size_t{1 << int(std::ceil(std::log2(c)))};
    if(c != c_next_pow2)
    {
        size_t max_modulo = 0;
        for(size_t vs = 8; vs > 1; vs >>= 1)
        {
            for(size_t xl = 64; xl > 8; xl >>= 1)
            {
                size_t xl_pow2 = std::min(size_t{1 << int(std::ceil(std::log2(c / vs)))}, xl);
                size_t modulo  = c % (xl_pow2 * vs);
                if(modulo == 0)
                {
                    vectorsize = vs;
                    xlocalsize = xl_pow2;
                    break;
                }
                else
                {
                    if(modulo > max_modulo)
                    {
                        vectorsize = vs;
                        xlocalsize = xl_pow2;
                        max_modulo = modulo;
                    }
                }
            }
        }
        return;
    }

    // In case c is power of 2, the previous method is suboptimal, so we set vectorsize and
    // localsize based on fine-grained heuristics
    if(problem.GetDirection() == miopen::batchnorm::Direction::ForwardTraining)
    {
        if(c <= 64)
        {
            vectorsize = 2;
            xlocalsize = 32;
        }
        else if(c == 128)
        {
            vectorsize = 2;
            xlocalsize = (in_cstride >= 4096) ? 64 : 32;
        }
        else if(c == 256)
        {
            vectorsize = (in_cstride >= 1024) ? 8 : 2;
            xlocalsize = (in_cstride >= 1024) ? 32 : 64;
        }
        else if(c == 512)
        {
            vectorsize = (in_cstride >= 256) ? 8 : 2;
            xlocalsize = (in_cstride >= 256) ? 32 : 64;
        }
        else if(c == 1024)
        {
            vectorsize = (n > 64) ? 8 : (in_cstride <= 64) ? 2 : 8;
            xlocalsize = 32;
        }
        else // c > 1024
        {
            vectorsize = (n > 64) ? 8 : (in_cstride <= 64) ? 4 : 8;
            xlocalsize = (in_cstride >= 256) ? 64 : 32;
        }
    }
    else
    {
        if(c <= 64)
        {
            vectorsize = 2;
            xlocalsize = 32;
        }
        else if(c == 128)
        {
            vectorsize = 2;
            xlocalsize = (in_cstride >= 64) ? 64 : 32;
        }
        else if(c == 256)
        {
            vectorsize = (n < 64) ? ((in_cstride > 4096) ? 8 : 2) : ((in_cstride >= 1024) ? 8 : 2);
            xlocalsize =
                (n < 64) ? ((in_cstride <= 4096) ? 64 : 32) : ((in_cstride < 1024) ? 64 : 32);
        }
        else if(c == 512)
        {
            vectorsize = (n < 64) ? ((in_cstride >= 4096) ? 8 : 2) : ((in_cstride >= 256) ? 8 : 2);
            xlocalsize =
                (n < 64) ? ((in_cstride >= 4096) ? 32 : 64) : ((in_cstride > 256) ? 32 : 64);
        }
        else if(c == 1024)
        {
            vectorsize = (n < 64) ? ((in_cstride <= 1024) ? 2 : 8) : ((in_cstride <= 256) ? 4 : 8);
            xlocalsize =
                (n < 64) ? ((in_cstride <= 1024) ? 64 : 32) : ((in_cstride <= 256) ? 64 : 32);
        }
        else // c > 1024
        {
            vectorsize = (in_cstride <= 64) ? 4 : 8;
            xlocalsize = 64;
        }
    }
    xlocalsize = std::min(size_t{1 << int(std::ceil(std::log2(c / vectorsize)))}, xlocalsize);
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
inline void DefaultConfigSpatialMultiple(const miopen::batchnorm::ProblemDescription& problem,
                                         unsigned int stash_values,
                                         std::vector<std::string>& valid_kernels)
{
    int n, c, h, w;
    std::tie(n, c, h, w)    = tien<4>(problem.GetXDesc().GetLengths());
    unsigned int in_cstride = h * w;

    size_t xlocalsize_default = 0;
    size_t ylocalsize_default = 0;
    size_t vectorsize_default = 4;
    size_t zlocalsize_default = 1;
    size_t nelements_default  = n;

    // Tuning instances: add the full parameter space
    if(problem.IsLayoutNHWC())
    {
        // First add the default instance, which should work well for a large range of problems
        {
            GetSpatialMultipleConfig(
                problem, vectorsize_default, xlocalsize_default, ylocalsize_default);
            if(IsSpatialMultipleApplicable(problem,
                                           vectorsize_default,
                                           stash_values,
                                           ylocalsize_default,
                                           zlocalsize_default,
                                           nelements_default))
            {
                valid_kernels.push_back(GetKernelIdFromVariant(2,
                                                               vectorsize_default,
                                                               xlocalsize_default,
                                                               ylocalsize_default,
                                                               zlocalsize_default,
                                                               nelements_default));
            }
            else
            {
                if(vectorsize_default > 1)
                {
                    vectorsize_default = 1;
                    GetSpatialMultipleConfig(
                        problem, vectorsize_default, xlocalsize_default, ylocalsize_default);

                    if(IsSpatialMultipleApplicable(problem,
                                                   1,
                                                   stash_values,
                                                   ylocalsize_default,
                                                   zlocalsize_default,
                                                   nelements_default))
                    {
                        valid_kernels.push_back(GetKernelIdFromVariant(2,
                                                                       vectorsize_default,
                                                                       xlocalsize_default,
                                                                       ylocalsize_default,
                                                                       zlocalsize_default,
                                                                       nelements_default));
                    }
                }
            }
        }

        // This is a case where variant 1 will probably work better than variant 2, so
        // we don't add other instances.
        if(c <= 4)
        {
            return;
        }

        // Add other instances to be added to tuning
        // xlocalsize and vectorsize are set using heuristics
        size_t vectorsize = 1;
        size_t xlocalsize = 64;
        {
            size_t reference_dimension = problem.IsLayoutNHWC() ? c : in_cstride;
            if(problem.IsLayoutNHWC())
            {
                GetHeuristicsConfigTuningNHWC(problem, vectorsize, xlocalsize);
            }
            while(reference_dimension % vectorsize != 0)
            {
                vectorsize >>= 1;
            }
            if(vectorsize == 1)
            {
                xlocalsize =
                    std::min(size_t{1 << int(std::ceil(std::log2(c / vectorsize)))}, size_t{64});
            }
        }

        // Given xlocalsize and vectorsize, add instances with different
        // ylocalsize, zlocalsize, nelements

        // We consider max_localsize = 1024 for vector size 1,2,4 and 512 for vectorsize 8.
        // Additionally, max_localsize = 1024 / vectorsize is added when vectorization is used.
        std::vector<size_t> max_localsize_vector = {1024 / (1 << (vectorsize / 8))};
        if(vectorsize > 1)
        {
            max_localsize_vector.push_back(1024 / vectorsize);
        }
        // Default case is zlocalsize 1, but with batch sizes >= 10, zlocalsize 2
        // can be beneficial
        std::vector<size_t> zlocalsize_vector = {1};
        if(n >= 10)
        {
            zlocalsize_vector.push_back(2);
        }
        for(const size_t& max_localsize : max_localsize_vector)
        {
            for(const size_t& zlocalsize : zlocalsize_vector)
            {
                // restrictions on ylocalsize are based on heuristics to decrease the amount
                // of instances removing the least used cases
                size_t ylocalsize = max_localsize / xlocalsize / zlocalsize;
                if(problem.GetDirection() == miopen::batchnorm::Direction::ForwardTraining)
                {
                    if(ylocalsize < 8 || ylocalsize > 32)
                    {
                        continue;
                    }
                }
                else
                {
                    if(in_cstride > 16384)
                    {
                        if(ylocalsize < 8 || ylocalsize > 32)
                        {
                            continue;
                        }
                    }
                    else
                    {
                        if(ylocalsize > 16)
                        {
                            continue;
                        }
                    }
                }

                // Use multiple zblocks if batch size is large enough.
                // nelements = 32 is an optimal value for the current implementation when
                // the batch size is large enough.
                std::vector<size_t> nelements_vector = {n / zlocalsize};
                if(n / zlocalsize > 64)
                {
                    nelements_vector.push_back(32);
                }
                for(const size_t& nelements : nelements_vector)
                {
                    // Restriction of the current implementation
                    if(n % nelements != 0)
                    {
                        continue;
                    }

                    // Restriction based on the number of CUs
                    size_t xgridsize =
                        xlocalsize * ((c / vectorsize + xlocalsize - 1) / xlocalsize);
                    size_t ygridsize = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
                    size_t zgridsize = zlocalsize * ((n / nelements + zlocalsize - 1) / zlocalsize);
                    size_t nWG       = (xgridsize / xlocalsize) * (ygridsize / ylocalsize) *
                                 (zgridsize / zlocalsize);
                    if(in_cstride > 64 && nWG < problem.GetMinWorkgroups())
                    {
                        continue;
                    }

                    // Avoid inserting the default spatial multiple instance twice
                    if(vectorsize == vectorsize_default && xlocalsize == xlocalsize_default &&
                       ylocalsize == ylocalsize_default && zlocalsize == zlocalsize_default &&
                       nelements == nelements_default)
                    {
                        continue;
                    }

                    // Check if the instance is applicable and add it
                    if(IsSpatialMultipleApplicable(
                           problem, vectorsize, stash_values, ylocalsize, zlocalsize, nelements))
                    {
                        valid_kernels.push_back(GetKernelIdFromVariant(
                            2, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements));
                    }
                }
            }
        }
    }
    else
    {
        // For NCHW we add all the supported vector sizes smaller than the default (if they are
        // applicable)
        while(vectorsize_default > 0)
        {
            GetSpatialMultipleConfig(
                problem, vectorsize_default, xlocalsize_default, ylocalsize_default);

            if(IsSpatialMultipleApplicable(problem,
                                           vectorsize_default,
                                           stash_values,
                                           ylocalsize_default,
                                           zlocalsize_default,
                                           nelements_default))
            {
                valid_kernels.push_back(GetKernelIdFromVariant(2,
                                                               vectorsize_default,
                                                               xlocalsize_default,
                                                               ylocalsize_default,
                                                               zlocalsize_default,
                                                               nelements_default));
            }
            vectorsize_default >>= 1;
        }

        // Keep the existing default first. Small mixed-precision batches can benefit
        // from shorter per-thread batch loops and more independent workgroups.
        // Exact divisors avoid a partial per-thread loop; the final z workgroup may
        // still contain inactive threads.
        if(problem.GetDirection() == miopen::batchnorm::Direction::ForwardTraining &&
           problem.IsBFp16() && problem.IsScaleFp32() && problem.Is2D() &&
           problem.IsLayoutNCHW() && problem.GetXDesc().IsPacked() &&
           problem.GetYDesc().IsPacked() && n >= 8 && n <= 64)
        {
            // gfx125 wave32 also supports 16-byte BF16 transactions. Keep vector8
            // behind the forward solver's device check and retain the old default.
            constexpr size_t vectorsize_vector8 = 8;
            size_t xlocalsize_vector8, ylocalsize_vector8;
            GetSpatialMultipleConfig(
                problem, vectorsize_vector8, xlocalsize_vector8, ylocalsize_vector8);
            if(IsSpatialMultipleApplicable(problem,
                                           vectorsize_vector8,
                                           stash_values,
                                           ylocalsize_vector8,
                                           zlocalsize_default,
                                           nelements_default))
            {
                valid_kernels.push_back(GetKernelIdFromVariant(2,
                                                               vectorsize_vector8,
                                                               xlocalsize_vector8,
                                                               ylocalsize_vector8,
                                                               zlocalsize_default,
                                                               nelements_default));
            }

            std::vector<size_t> batch_elements;
            for(size_t target : {size_t{32}, size_t{16}, size_t{8}})
            {
                size_t nelements = std::min(target, static_cast<size_t>(n / 2));
                while(n % nelements != 0)
                    --nelements;
                if(nelements > 1 &&
                   std::find(batch_elements.begin(), batch_elements.end(), nelements) ==
                       batch_elements.end())
                {
                    batch_elements.push_back(nelements);
                }
            }

            for(size_t vectorsize : {vectorsize_vector8, size_t{4}, size_t{2}, size_t{1}})
            {
                if(in_cstride % vectorsize != 0)
                    continue;

                size_t xlocalsize, ylocalsize_max;
                GetSpatialMultipleConfig(problem, vectorsize, xlocalsize, ylocalsize_max);
                for(size_t ylocalsize : {size_t{256}, size_t{512}})
                {
                    if(ylocalsize > ylocalsize_max)
                        continue;
                    for(size_t zlocalsize : {size_t{1}, size_t{2}})
                    {
                        for(size_t nelements : batch_elements)
                        {
                            // Limit split-batch tuning to 32 additional instances. For
                            // z=2, use only the smaller workgroup and shorter loops.
                            if(zlocalsize == 2 && (ylocalsize != 256 || nelements > 16))
                                continue;
                            if(!IsSpatialMultipleApplicable(problem,
                                                            vectorsize,
                                                            stash_values,
                                                            ylocalsize,
                                                            zlocalsize,
                                                            nelements))
                                continue;

                            const size_t ytile = ylocalsize * vectorsize;
                            const size_t ytail = (in_cstride - 1) % ytile + 1;
                            const size_t ztile = zlocalsize * nelements;
                            const size_t ztail = (n - 1) % ztile + 1;
                            const int stash_method = GetStashMethod(false,
                                                                    problem.GetXDesc().GetType(),
                                                                    stash_values,
                                                                    c,
                                                                    n,
                                                                    in_cstride,
                                                                    ylocalsize,
                                                                    zlocalsize,
                                                                    nelements);
                            // FP32 stash values occupy two BF16 slots each. Check
                            // the actual vectorized spatial tile and allow for
                            // worst-case alignment padding with odd spatial sizes.
                            const size_t stash_slots = 2 * stash_values;
                            const size_t padding     = in_cstride % 2;
                            if((stash_method == 0 && ytail < stash_slots + padding) ||
                               (stash_method == 1 &&
                                (ztail < stash_slots || padding != 0)))
                                continue;

                            valid_kernels.push_back(GetKernelIdFromVariant(
                                2, vectorsize, xlocalsize, ylocalsize, zlocalsize, nelements));
                        }
                    }
                }
            }
        }
    }
}

} // namespace batchnorm

} // namespace solver

} // namespace miopen
