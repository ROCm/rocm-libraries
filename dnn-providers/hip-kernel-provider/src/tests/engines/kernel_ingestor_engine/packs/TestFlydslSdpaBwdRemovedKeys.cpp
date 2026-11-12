// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"

/**
 * @file TestFlydslSdpaBwdRemovedKeys.cpp
 * @brief Keys no query row sees (outside every row's window, past a batch's valid key length)
 *        hold NaN or infinity in K, in V or in both on the device; no gradient may take the
 *        value. Each gradient must be finite and match the reference, and dK and dV must be
 *        exactly zero at those keys. A failure names the gradient: dQ is the dq object's, dK
 *        and dV the dkdv object's.
 *
 * The shipped hipkernel:flydsl_sdpa_bwd_split and hipkernel:flydsl_sdpa_bwd_merged engines run
 * the cases through the harness of FlydslSdpaBwdDeviceHarness.hpp; the control case has no
 * poison and shows the graph itself is served.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;

enum class Removal
{
    /// Bottom-right window (left 48, right 0) over more keys than queries: keys 0..208 are in
    /// no row's window, a straddled tile among them.
    WINDOW,
    /// Padded batches with the second batch's valid key length 301: its keys past it. A window
    /// beside padding_mask with bottom-right alignment is declined (MW-13), so the two are apart.
    TAIL,
    /// Top-left causal with 100 queries over 384 keys: 100 is no tile multiple, so the last
    /// query tile holds padded rows, and keys 100..383 are seen by no row.
    TOP_LEFT_SHORT_QUERY,
    /// The TAIL lengths under top-left causal alignment.
    TAIL_TOP_LEFT
};

BwdCase
    removedKeysCase(DataType dtype, Removal removal, BwdPoison poison, float value, int64_t headDim)
{
    BwdCase c;
    c.dtype = dtype;
    c.batch = 2;
    c.heads = 4;
    c.kvHeads = 4;
    c.seqQ = 128;
    c.seqKv = 384;
    c.headDim = headDim;
    if(removal == Removal::WINDOW)
    {
        c.mask = BwdMask::BOTTOM_RIGHT;
        c.windowLeft = 48;
        c.windowRight = 0;
    }
    if(removal == Removal::TOP_LEFT_SHORT_QUERY)
    {
        c.mask = BwdMask::TOP_LEFT;
        c.seqQ = 100;
    }
    if(removal == Removal::TAIL || removal == Removal::TAIL_TOP_LEFT)
    {
        if(removal == Removal::TAIL_TOP_LEFT)
        {
            c.mask = BwdMask::TOP_LEFT;
        }
        c.lengthsQ = {128, 100};
        c.lengthsKv = {384, 301};
    }
    c.poison = poison;
    c.poisonValue = value;
    return c;
}

void runRemovedKeys(std::string_view engine,
                    DataType dtype,
                    Removal removal,
                    BwdPoison poison,
                    const std::vector<int64_t>& headDims)
{
    for(const int64_t headDim : headDims)
    {
        for(const float value :
            {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
        {
            SCOPED_TRACE("head dim " + std::to_string(headDim) + ", poison "
                         + std::to_string(value));
            expectCase(engine, removedKeysCase(dtype, removal, poison, value, headDim));
        }
    }
}

class TestGpuFlydslSdpaBwdSplitRemovedKeys : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdSplitRemovedKeys()
        : FlydslSdpaBwdDevice(SPLIT_OPS)
    {
    }
};

class TestGpuFlydslSdpaBwdMergedRemovedKeys : public FlydslSdpaBwdDevice
{
protected:
    TestGpuFlydslSdpaBwdMergedRemovedKeys()
        : FlydslSdpaBwdDevice(MERGED_OPS)
    {
    }
};

// One suite per dtype: the datatype token ends the suite name.
using TestGpuFlydslSdpaBwdSplitRemovedKeysBfp16 = TestGpuFlydslSdpaBwdSplitRemovedKeys;
using TestGpuFlydslSdpaBwdSplitRemovedKeysFp16 = TestGpuFlydslSdpaBwdSplitRemovedKeys;
using TestGpuFlydslSdpaBwdSplitRemovedKeysFp32 = TestGpuFlydslSdpaBwdSplitRemovedKeys;
using TestGpuFlydslSdpaBwdMergedRemovedKeysBfp16 = TestGpuFlydslSdpaBwdMergedRemovedKeys;
using TestGpuFlydslSdpaBwdMergedRemovedKeysFp16 = TestGpuFlydslSdpaBwdMergedRemovedKeys;

#define FLYDSL_SDPA_BWD_REMOVED_KEYS_CASES(SUITE, ENGINE, DTYPE, HEAD_DIMS)                     \
    TEST_F(SUITE, ControlWithoutPoison)                                                         \
    {                                                                                           \
        for(const auto removal : {Removal::WINDOW,                                              \
                                  Removal::TAIL,                                                \
                                  Removal::TOP_LEFT_SHORT_QUERY,                                \
                                  Removal::TAIL_TOP_LEFT})                                      \
        {                                                                                       \
            expectCase(ENGINE,                                                                  \
                       removedKeysCase(DTYPE, removal, BwdPoison::NONE, 0.0F, (HEAD_DIMS)[0])); \
        }                                                                                       \
    }                                                                                           \
    TEST_F(SUITE, WindowPoisonK)                                                                \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::WINDOW, BwdPoison::K, HEAD_DIMS);                \
    }                                                                                           \
    TEST_F(SUITE, WindowPoisonV)                                                                \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::WINDOW, BwdPoison::V, HEAD_DIMS);                \
    }                                                                                           \
    TEST_F(SUITE, WindowPoisonKAndV)                                                            \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::WINDOW, BwdPoison::K_AND_V, HEAD_DIMS);          \
    }                                                                                           \
    TEST_F(SUITE, TailPoisonK)                                                                  \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::TAIL, BwdPoison::K, HEAD_DIMS);                  \
    }                                                                                           \
    TEST_F(SUITE, TailPoisonV)                                                                  \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::TAIL, BwdPoison::V, HEAD_DIMS);                  \
    }                                                                                           \
    TEST_F(SUITE, TailPoisonKAndV)                                                              \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::TAIL, BwdPoison::K_AND_V, HEAD_DIMS);            \
    }                                                                                           \
    TEST_F(SUITE, TopLeftShortQueryPoisonK)                                                     \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::TOP_LEFT_SHORT_QUERY, BwdPoison::K, HEAD_DIMS);  \
    }                                                                                           \
    TEST_F(SUITE, TopLeftShortQueryPoisonV)                                                     \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::TOP_LEFT_SHORT_QUERY, BwdPoison::V, HEAD_DIMS);  \
    }                                                                                           \
    TEST_F(SUITE, TopLeftShortQueryPoisonKAndV)                                                 \
    {                                                                                           \
        runRemovedKeys(                                                                         \
            ENGINE, DTYPE, Removal::TOP_LEFT_SHORT_QUERY, BwdPoison::K_AND_V, HEAD_DIMS);       \
    }                                                                                           \
    TEST_F(SUITE, TailTopLeftPoisonK)                                                           \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::TAIL_TOP_LEFT, BwdPoison::K, HEAD_DIMS);         \
    }                                                                                           \
    TEST_F(SUITE, TailTopLeftPoisonV)                                                           \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::TAIL_TOP_LEFT, BwdPoison::V, HEAD_DIMS);         \
    }                                                                                           \
    TEST_F(SUITE, TailTopLeftPoisonKAndV)                                                       \
    {                                                                                           \
        runRemovedKeys(ENGINE, DTYPE, Removal::TAIL_TOP_LEFT, BwdPoison::K_AND_V, HEAD_DIMS);   \
    }

const std::vector<int64_t> HEAD_DIMS_16_BIT{64, 128};
const std::vector<int64_t> HEAD_DIMS_FLOAT{64};

FLYDSL_SDPA_BWD_REMOVED_KEYS_CASES(TestGpuFlydslSdpaBwdSplitRemovedKeysBfp16,
                                   SPLIT_ENGINE,
                                   DataType::BFLOAT16,
                                   HEAD_DIMS_16_BIT)
FLYDSL_SDPA_BWD_REMOVED_KEYS_CASES(TestGpuFlydslSdpaBwdSplitRemovedKeysFp16,
                                   SPLIT_ENGINE,
                                   DataType::HALF,
                                   HEAD_DIMS_16_BIT)
FLYDSL_SDPA_BWD_REMOVED_KEYS_CASES(TestGpuFlydslSdpaBwdSplitRemovedKeysFp32,
                                   SPLIT_ENGINE,
                                   DataType::FLOAT,
                                   HEAD_DIMS_FLOAT)
FLYDSL_SDPA_BWD_REMOVED_KEYS_CASES(TestGpuFlydslSdpaBwdMergedRemovedKeysBfp16,
                                   MERGED_ENGINE,
                                   DataType::BFLOAT16,
                                   HEAD_DIMS_16_BIT)
FLYDSL_SDPA_BWD_REMOVED_KEYS_CASES(TestGpuFlydslSdpaBwdMergedRemovedKeysFp16,
                                   MERGED_ENGINE,
                                   DataType::HALF,
                                   HEAD_DIMS_16_BIT)

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
