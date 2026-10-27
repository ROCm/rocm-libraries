// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
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

class ProblemDescription
{
public:
    ProblemDescription(size_t n,
                       size_t c,
                       size_t h,
                       size_t w,
                       bool isLayoutNHWC,
                       bool useFp16Mix,
                       bool useBfp16Mix,
                       Direction direction,
                       size_t minWorkgroups)
        : _n(n)
        , _c(c)
        , _h(h)
        , _w(w)
        , _isLayoutNHWC(isLayoutNHWC)
        , _useFp16Mix(useFp16Mix)
        , _useBfp16Mix(useBfp16Mix)
        , _direction(direction)
        , _minWorkgroups(minWorkgroups)
    {
    }

    size_t n() const
    {
        return _n;
    }
    size_t c() const
    {
        return _c;
    }
    size_t h() const
    {
        return _h;
    }
    size_t w() const
    {
        return _w;
    }
    bool isLayoutNHWC() const
    {
        return _isLayoutNHWC;
    }
    bool useFp16Mix() const
    {
        return _useFp16Mix;
    }
    bool useBfp16Mix() const
    {
        return _useBfp16Mix;
    }
    bool useFp32() const
    {
        return !_useFp16Mix && !_useBfp16Mix;
    }
    Direction direction() const
    {
        return _direction;
    }
    size_t minWorkgroups() const
    {
        return _minWorkgroups;
    }

    unsigned int inCstride() const
    {
        return static_cast<unsigned int>(_h * _w);
    }
    unsigned int inNhw() const
    {
        return static_cast<unsigned int>(_n * _h * _w);
    }
    unsigned int inChw() const
    {
        return static_cast<unsigned int>(_c * _h * _w);
    }
    unsigned int inNchw() const
    {
        return static_cast<unsigned int>(_n * _c * _h * _w);
    }

private:
    size_t _n;
    size_t _c;
    size_t _h;
    size_t _w;
    bool _isLayoutNHWC;
    bool _useFp16Mix;
    bool _useBfp16Mix;
    Direction _direction;
    size_t _minWorkgroups;
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

// ---------------------------------------------------------------------------------------------
// Declared tensor alignment
//
// The backend rejects, at execute, any device pointer whose address is not a multiple of the
// tensor's declared alignment (TensorAttributes::alignment(), bytes, default 16). A declared
// alignment A therefore guarantees only `address % A == 0`, so it covers an N-byte access exactly
// when N divides A: declared 16 covers 16, 8, 4, 2 and 1; declared 8 covers 8 but not 16;
// declared 24 covers 8 but not 16. A kernel access of `width` elements of `elementBytes` bytes
// needs `width * elementBytes` covered.
// ---------------------------------------------------------------------------------------------

// TensorAttributes::alignment() default: what every graph declared before plugin API 1.3.0.
constexpr size_t DEFAULT_TENSOR_ALIGNMENT_BYTES = 16;

// Element sizes the batchnorm kernels are instantiated with.
constexpr size_t FP32_BYTES = 4;
constexpr size_t HALF_BYTES = 2; // fp16 and bf16

// Widest vector, in elements, that any batchnorm kernel loads or stores in one access.
constexpr size_t MAX_VECTOR_SIZE = 4;

// Fixed-width accesses of the one-workgroup-per-channel kernels (variant 1). They do not depend
// on the selected vector size, and only the NCHW layout has them.
//  - Forward training reads x as `fp_type4` when hw >= FWD_TRAIN_SINGLE_WIDE_READ_MIN_HW
//    (kernels/batchnorm/BatchNormFwdTrainSpatial.cpp:264-296: the reinterpret_cast at 275 and 293).
//    Its writes are scalar.
//  - Backward reads x and dy BWD_SINGLE_WIDE_READ_SIZE elements at a time
//    (kernels/batchnorm/BatchNormBwdSpatial.cpp:356,505-507,527-529) and, in the final pass, x and
//    dy BWD_SINGLE_WIDE_WRITE_SIZE elements at a time (:357,571-574), writing dx the same width
//    (:612). They are taken as always executed, whatever the shape.
constexpr size_t FWD_TRAIN_SINGLE_WIDE_READ_SIZE = 4;
constexpr unsigned int FWD_TRAIN_SINGLE_WIDE_READ_MIN_HW = 4096;
constexpr size_t BWD_SINGLE_WIDE_READ_SIZE = 4;
constexpr size_t BWD_SINGLE_WIDE_WRITE_SIZE = 2;

// Declared alignment, in bytes, of the tensors a spatial (training or backward) kernel accesses,
// grouped by how the kernel touches them. Each value is the alignment every tensor of the group
// is guaranteed to have (0 guarantees nothing). The default is what every graph declared before
// plugin API 1.3.0.
struct SpatialAlignment
{
    // Activations the kernel reads: x (and dy in backward).
    size_t input = DEFAULT_TENSOR_ALIGNMENT_BYTES;
    // Activation the kernel writes: y (dx in backward). In the multi-workgroup path this buffer
    // also holds the stash of per-workgroup partial sums.
    size_t output = DEFAULT_TENSOR_ALIGNMENT_BYTES;
    // Per-channel fp32 tensors: scale, bias, mean, inv_variance, running statistics, dscale, dbias.
    size_t channel = DEFAULT_TENSOR_ALIGNMENT_BYTES;
};

// Alignment, in bytes, that a kernel configuration needs from each tensor group.
struct SpatialAlignmentNeed
{
    size_t input;
    size_t output;
    size_t channel;
};

// True when a declared alignment of `declaredBytes` guarantees `needBytes`-byte alignment.
inline bool alignmentCovers(size_t declaredBytes, size_t needBytes)
{
    return declaredBytes != 0 && needBytes != 0 && declaredBytes % needBytes == 0;
}

// Compute workgroup size configuration given a problem (NHWC) and a vectorsize
// It supports only 2D workgroups
inline void getLocalConfigNHWC(const ProblemDescription& problem,
                               size_t vectorsize,
                               size_t& xlocalsize,
                               size_t& ylocalsize)
{
    // Compute workgroup size
    unsigned int xlocalsizeLimit = 64;
    if(vectorsize > 1)
    {
        xlocalsizeLimit = problem.useFp32() ? 16 : 32;
    }

    // shared memory size per workgroup is fixed
    unsigned int maxLocalsize = 1024 / vectorsize;

    // default local config in case the while loop is not entered
    xlocalsize = std::min(size_t{1} << static_cast<size_t>(
                              std::ceil(std::log2(std::max(problem.c() / vectorsize, size_t{1})))),
                          static_cast<size_t>(xlocalsizeLimit));
    ylocalsize = maxLocalsize / xlocalsize;

    size_t nworkgroups = 0;
    // decrease maxLocalsize until the number of workgroups is greater than 80%
    // of the available CUs
    while(nworkgroups < problem.minWorkgroups() && maxLocalsize >= xlocalsizeLimit
          && maxLocalsize > 64)
    {
        // xlocalsize must be power of 2 as reductions in the kernels rely on it, here c is rounded
        // up to next power of 2.
        xlocalsize = std::min(size_t{1} << static_cast<size_t>(std::ceil(
                                  std::log2(std::max(problem.c() / vectorsize, size_t{1})))),
                              static_cast<size_t>(xlocalsizeLimit));
        ylocalsize = maxLocalsize / xlocalsize;
        nworkgroups = ((problem.c() / vectorsize + xlocalsize - 1) / xlocalsize)
                      * ((problem.inCstride() + ylocalsize - 1) / ylocalsize);
        maxLocalsize >>= 1;
    }
}

// Provide workgroup sizes for spatial multiple configuration.
// It returns the preferred spatial multiple configuration, which is used without tuning.
// If tuning is enabled, this configuration is also added to the group of instances.
inline void getSpatialMultipleConfig(const ProblemDescription& problem,
                                     size_t vectorsize,
                                     size_t& xlocalsize,
                                     size_t& ylocalsize)
{
    // Initialize to safe defaults at the start of the function
    xlocalsize = 1;
    ylocalsize = 1;

    const size_t inCstride = problem.inCstride();

    if(problem.isLayoutNHWC())
    {
        if(problem.c() % vectorsize != 0)
        {
            // xlocalsize and ylocalsize already initialized to 1
            return;
        }
        getLocalConfigNHWC(problem, vectorsize, xlocalsize, ylocalsize);
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
inline bool isSpatialMultipleApplicable(const ProblemDescription& problem,
                                        size_t vectorsize,
                                        unsigned int stashValues,
                                        size_t ylocalsize,
                                        size_t zlocalsize,
                                        size_t nelements)
{
    const auto inCstride = problem.inCstride();

    if(problem.isLayoutNHWC())
    {
        // check if the provided vectorsize can be used
        if(problem.c() % vectorsize != 0)
        {
            return false;
        }

        stashValues *= (problem.useFp32() ? 1 : 2);
        const unsigned int lastYlocalsize = inCstride % ylocalsize == 0
                                                ? static_cast<unsigned int>(ylocalsize)
                                                : inCstride % ylocalsize;

        const unsigned int lastZocalsize
            = problem.n() % (zlocalsize * nelements) == 0
                  ? static_cast<unsigned int>(zlocalsize * nelements)
                  : problem.n() % static_cast<unsigned int>(zlocalsize * nelements);

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
        if((!problem.useFp32() && (problem.c() % 2 != 0 && lastZocalsize < stashValues))
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
            = problem.n() % (zlocalsize * nelements) == 0
                  ? static_cast<unsigned int>(zlocalsize * nelements)
                  : problem.n() % static_cast<unsigned int>(zlocalsize * nelements);
        // Restrictions:
        //  - last block must have enough space to stash intermediate results in HW dimension
        //  - if last block doesn't fit, intermediate results are stored in N dimension which must
        //    be large enough
        stashValues *= (problem.useFp32() ? 1 : 2);
        if(lastYlocalsize < stashValues && lastZocalsize < stashValues)
        {
            return false;
        }
    }
    return true;
}

inline bool useMultiple(const ProblemDescription& problem)
{
    const auto inCstride = problem.inCstride();
    const auto inNhw = problem.inNhw();
    const auto thrInNhw = static_cast<unsigned int>(32 * 1024 * 1024);

    if(!problem.isLayoutNHWC() && problem.direction() == Direction::BACKWARD)
    {
        if((inNhw < thrInNhw && inCstride > 1024) || (inNhw < thrInNhw && inCstride > 512)
           || inCstride <= 512)
        {
            return false;
        }
    }

    if(!problem.isLayoutNHWC() && problem.direction() == Direction::FORWARD_TRAINING)
    {
        const bool condition1
            = (problem.n() < 3) || (inCstride <= 512) || (inNhw < 33554432 && inCstride > 1024)
              || (problem.n() >= 256 && inCstride > 60
                  && (problem.useFp16Mix() || problem.useBfp16Mix()))
              || ((problem.useFp16Mix() || problem.useBfp16Mix()) && inCstride > 512);

        const bool condition2 = (problem.n() <= 768) || (inCstride <= 150);

        if(condition1 && condition2)
        {
            return false;
        }
    }

    return true;
}

// Provide the stash method to use for spatial multiple implementation
inline int getStashMethod(const ProblemDescription& problem,
                          unsigned int stashValues,
                          size_t ylocalsize,
                          size_t zlocalsize,
                          size_t nelements)
{
    // See `batchnorm_functions.hpp` for stash implementation of different methods
    int stashMethod = 0;
    stashValues *= (problem.useFp32() ? 1 : 2);
    const unsigned int lastYlocalsize = problem.inCstride() % ylocalsize == 0
                                            ? static_cast<unsigned int>(ylocalsize)
                                            : problem.inCstride() % ylocalsize;
    const unsigned int lastZocalsize
        = problem.n() % (zlocalsize * nelements) == 0
              ? static_cast<unsigned int>(zlocalsize * nelements)
              : problem.n() % static_cast<unsigned int>(zlocalsize * nelements);
    if(lastYlocalsize < stashValues && lastZocalsize >= stashValues)
    {
        stashMethod = 1;
    }
    if(problem.isLayoutNHWC() && !problem.useFp32() && (problem.c() % 2 != 0)
       && (lastZocalsize >= stashValues))
    {
        stashMethod = 2;
    }
    return stashMethod;
}

inline void defaultConfigSpatialSingle(const ProblemDescription& problem, KernelConfig& config)
{
    const auto n = problem.n();
    const auto inCstride = problem.inCstride();
    const auto inNhw = problem.inNhw();

    // NCHW supports also variants 0 and 3 which can be much faster than
    // variant 1 but have more restrictions. Here we decide if we use variant
    // 0, 1, 3
    // In case variant 0 or 3 are selected, we add also variant 1 for tuning.
    // Almost always variant 0 and 3 will be faster than variant 1 but
    // we add the latter for tuning to be sure and because it is cheap to run.
    // NOTE: Currently we don't have the tuning infrastructure in place, so we
    // are only selecting one variant to run based on heuristics.
    if(!problem.isLayoutNHWC())
    {
        if(problem.direction() == Direction::BACKWARD)
        {
            if((inCstride < 200) && (inCstride > 60) && problem.useFp16Mix())
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
               || ((n >= 256) && (inCstride > 60)
                   && (problem.useFp16Mix() || problem.useBfp16Mix()))
               || ((inCstride > 512) && (problem.useFp16Mix() || problem.useBfp16Mix())))
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

// Size in bytes of one x/y/dy/dx element of the problem.
inline size_t activationElementBytes(const ProblemDescription& problem)
{
    return problem.useFp32() ? FP32_BYTES : HALF_BYTES;
}

// Alignment the single-workgroup-per-channel kernels (variants 0, 1, 3) need. Every access is a
// scalar one except the fixed-width ones of variant 1 in NCHW, see FWD_TRAIN_SINGLE_WIDE_READ_SIZE
// and BWD_SINGLE_WIDE_READ_SIZE. Per-channel values are read and written one float at a time
// (BatchNormFwdTrainSpatial.cpp:258-259,1060-1073, BatchNormBwdSpatial.cpp:420-422,556-557).
inline SpatialAlignmentNeed spatialSingleAlignmentNeed(const ProblemDescription& problem,
                                                       int variant)
{
    size_t readSize = 1;
    size_t writeSize = 1;
    if(variant == 1 && !problem.isLayoutNHWC())
    {
        if(problem.direction() == Direction::BACKWARD)
        {
            readSize = BWD_SINGLE_WIDE_READ_SIZE;
            writeSize = BWD_SINGLE_WIDE_WRITE_SIZE;
        }
        else if(problem.direction() == Direction::FORWARD_TRAINING
                && problem.inCstride() >= FWD_TRAIN_SINGLE_WIDE_READ_MIN_HW)
        {
            readSize = FWD_TRAIN_SINGLE_WIDE_READ_SIZE;
        }
    }

    const size_t activationBytes = activationElementBytes(problem);
    return {readSize * activationBytes, writeSize * activationBytes, FP32_BYTES};
}

// Alignment the spatial multiple kernels (variant 2) need at a vector size.
//  - Activations are accessed as vectors of `vectorsize` elements in both layouts
//    (BatchNormFwdTrainSpatial.cpp:735,747,923; BatchNormBwdSpatial.cpp:1034,1190-1192,1452-1466).
//  - Per-channel fp32 tensors are accessed as a vector of `vectorsize` floats in NHWC and as a
//    single float in NCHW (BatchNormFwdTrainSpatial.cpp:679-681,1148-1159;
//    BatchNormBwdSpatial.cpp:1162-1167,1338-1339,1421-1429).
//  - The output buffer doubles as the stash of per-workgroup partial sums, which is accessed as
//    per-channel values (BatchnormFunctions.hpp:232,263), so it needs the per-channel width too.
//    This holds for fp16 and bf16 outputs as well: the stash is fp32 data in a 2-byte buffer.
inline SpatialAlignmentNeed spatialMultipleAlignmentNeed(const ProblemDescription& problem,
                                                         size_t vectorsize)
{
    const size_t activationBytes = vectorsize * activationElementBytes(problem);
    const size_t channelBytes = (problem.isLayoutNHWC() ? vectorsize : size_t{1}) * FP32_BYTES;
    return {activationBytes, std::max(activationBytes, channelBytes), channelBytes};
}

// True when every tensor group's declared alignment covers what the configuration needs.
inline bool fitsDeclaredAlignment(const SpatialAlignmentNeed& need,
                                  const SpatialAlignment& alignment)
{
    return alignmentCovers(alignment.input, need.input)
           && alignmentCovers(alignment.output, need.output)
           && alignmentCovers(alignment.channel, need.channel);
}

// Sets `config` to the spatial multiple instance of the given vector size when the declared
// alignment covers its accesses and the instance is applicable to the problem. Returns whether
// it did.
inline bool trySpatialMultipleConfig(const ProblemDescription& problem,
                                     size_t vectorsize,
                                     unsigned int stashValues,
                                     const SpatialAlignment& alignment,
                                     KernelConfig& config)
{
    if(!fitsDeclaredAlignment(spatialMultipleAlignmentNeed(problem, vectorsize), alignment))
    {
        return false;
    }

    size_t xlocalsize = 1;
    size_t ylocalsize = 1;
    const size_t zlocalsize = 1;
    const size_t nelements = problem.n();
    getSpatialMultipleConfig(problem, vectorsize, xlocalsize, ylocalsize);

    if(!isSpatialMultipleApplicable(
           problem, vectorsize, stashValues, ylocalsize, zlocalsize, nelements))
    {
        return false;
    }

    config.variant = 2;
    config.vectorsize = vectorsize;
    config.xlocalsize = xlocalsize;
    config.ylocalsize = ylocalsize;
    config.zlocalsize = zlocalsize;
    config.nelements = nelements;
    return true;
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
// A vector size whose accesses the declared alignment does not cover is skipped, which leaves
// the narrower sizes: NHWC falls back from MAX_VECTOR_SIZE to 1, NCHW tries each of 4, 2 and 1.
// `config.variant` stays -1 when no instance qualifies.
inline void defaultConfigSpatialMultiple(const ProblemDescription& problem,
                                         unsigned int stashValues,
                                         KernelConfig& config,
                                         const SpatialAlignment& alignment = SpatialAlignment{})
{
    if(problem.isLayoutNHWC())
    {
        // First add the default instance, which should work well for a large range of problems
        if(!trySpatialMultipleConfig(problem, MAX_VECTOR_SIZE, stashValues, alignment, config))
        {
            trySpatialMultipleConfig(problem, 1, stashValues, alignment, config);
        }

        // NOTE: We can add more instances for tuning here but we don't have
        // the tuning infrastructure in place yet, so we are adding only one
        // instance.
        return;
    }

    // For NCHW we add all the supported vector sizes smaller than the default (if they are
    // applicable)
    for(size_t vectorsize = MAX_VECTOR_SIZE; vectorsize > 0; vectorsize >>= 1)
    {
        trySpatialMultipleConfig(problem, vectorsize, stashValues, alignment, config);
    }
}

// Selects the kernel configuration of a training or backward problem under the declared tensor
// alignment. Spatial multiple is preferred when the heuristic says so, with the vector size
// folded down to what the alignment covers. Otherwise, or when no spatial multiple instance
// qualifies, the spatial single configuration is used. When that one reads wider than the
// alignment covers (the fixed-width reads of variant 1 in NCHW), the only variant left is spatial
// multiple at a narrower vector size, so it is tried even where the heuristic would not pick it.
// Returns false, with `config.variant` -1, when no configuration fits: the problem cannot run at
// that alignment.
//
// The result does not depend on the problem's minWorkgroups. That only moves the NHWC workgroup
// sizes, and NHWC always ends in a scalar spatial single configuration that needs no more than
// scalar element alignment. NCHW never reads it.
inline bool selectSpatialConfig(const ProblemDescription& problem,
                                unsigned int stashValues,
                                const SpatialAlignment& alignment,
                                KernelConfig& config)
{
    config = KernelConfig{};

    // Define default configuration based on heuristics and
    // add all other valid configurations for the given problem
    if(useMultiple(problem))
    {
        defaultConfigSpatialMultiple(problem, stashValues, config, alignment);
    }

    if(config.variant == -1)
    {
        // Spatial multiple is not preferred or found no valid configuration: get a default
        // spatial single configuration as fallback
        defaultConfigSpatialSingle(problem, config);
        if(!fitsDeclaredAlignment(spatialSingleAlignmentNeed(problem, config.variant), alignment))
        {
            config = KernelConfig{};
            defaultConfigSpatialMultiple(problem, stashValues, config, alignment);
        }
    }

    return config.variant != -1;
}

} // namespace batchnorm

} // namespace hip_kernel_provider
