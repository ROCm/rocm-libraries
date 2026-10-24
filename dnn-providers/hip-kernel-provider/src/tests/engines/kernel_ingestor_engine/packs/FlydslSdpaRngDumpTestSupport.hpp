// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaDropout.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaPhiloxOracle.hpp"

/**
 * @file FlydslSdpaRngDumpTestSupport.hpp
 * @brief The checks every family's device harness runs on an RNG dump (FlydslSdpaRngDump.hpp):
 *        the dump is the D1 oracle's mask bit for bit, it passes hipDNN's keep-rate check, and
 *        O is hipDNN's CPU reference run with the dump as its explicit mask.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// A dump's extents: FLOAT [B, Hq, Sq, Skv], packed row-major.
struct RngDumpExtents
{
    int64_t batch = 0;
    int64_t heads = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;

    size_t elements() const
    {
        return static_cast<size_t>(batch * heads * seqLenQ * seqLenKv);
    }

    size_t index(int64_t b, int64_t h, int64_t i, int64_t j) const
    {
        return static_cast<size_t>(((b * heads + h) * seqLenQ + i) * seqLenKv + j);
    }
};

/// A launch's RNG dump on the device, NaN until written (an element the writer skips then fails
/// the 0/1 check), and its host copy; nothing is allocated when the graph names no dump.
class DeviceRngDump
{
public:
    DeviceRngDump(const RngDumpExtents& extents, bool wanted)
    {
        if(!wanted)
        {
            return;
        }
        _host.assign(extents.elements(), std::numeric_limits<float>::quiet_NaN());
        EXPECT_EQ(hipSuccess, hipMalloc(&_device, bytes()));
        EXPECT_EQ(hipSuccess, hipMemcpy(_device, _host.data(), bytes(), hipMemcpyHostToDevice));
    }

    ~DeviceRngDump()
    {
        static_cast<void>(hipFree(_device));
    }

    DeviceRngDump(const DeviceRngDump&) = delete;
    DeviceRngDump& operator=(const DeviceRngDump&) = delete;
    DeviceRngDump(DeviceRngDump&&) = delete;
    DeviceRngDump& operator=(DeviceRngDump&&) = delete;

    void* device() const
    {
        return _device;
    }

    /// Copies the dump back from the device; returns the host copy.
    const std::vector<float>& readBack()
    {
        if(_device != nullptr)
        {
            EXPECT_EQ(hipSuccess, hipMemcpy(_host.data(), _device, bytes(), hipMemcpyDeviceToHost));
        }
        return _host;
    }

    const std::vector<float>& host() const
    {
        return _host;
    }

private:
    size_t bytes() const
    {
        return _host.size() * sizeof(float);
    }

    std::vector<float> _host;
    void* _device = nullptr;
};

/// The raw keep bit at every element -- masked keys, padded rows and keys past a batch's
/// length included -- equal to the D1 oracle's (FlydslSdpaPhiloxOracle.hpp), as exactly 1.0 or
/// 0.0; and hipDNN's harness's keep-rate check over the whole buffer (RngDumpHandoff.hpp
/// checkRngDump: within 6 sigma + 1/n of 1 - p).
inline void expectRngDumpIsTheOracleMask(const std::vector<float>& dump,
                                         const RngDumpExtents& extents,
                                         float p,
                                         uint64_t seed,
                                         uint64_t offset)
{
    ASSERT_EQ(dump.size(), extents.elements());
    const auto threshold = flydsl_sdpa::dropoutThreshold(p);
    size_t wrong = 0;
    size_t kept = 0;
    std::string first;
    for(int64_t b = 0; b < extents.batch; ++b)
    {
        for(int64_t h = 0; h < extents.heads; ++h)
        {
            for(int64_t i = 0; i < extents.seqLenQ; ++i)
            {
                for(int64_t j = 0; j < extents.seqLenKv; ++j)
                {
                    const float value = dump[extents.index(b, h, i, j)];
                    const bool keeps = philox_oracle::keeps(
                        seed,
                        offset,
                        philox_oracle::MaskCoordinates{
                            b, h, i, j, extents.heads, extents.seqLenQ, extents.seqLenKv},
                        threshold);
                    kept += value == 1.0F ? 1U : 0U;
                    if(!(value == (keeps ? 1.0F : 0.0F)) && wrong++ == 0)
                    {
                        first = "b=" + std::to_string(b) + " h=" + std::to_string(h)
                                + " i=" + std::to_string(i) + " j=" + std::to_string(j) + ": dump "
                                + std::to_string(value) + ", oracle " + (keeps ? "keeps" : "drops");
                    }
                }
            }
        }
    }
    EXPECT_EQ(wrong, 0U) << wrong << " of " << dump.size() << " elements differ, first " << first;
    const auto n = static_cast<double>(dump.size());
    const double keep = 1.0 - static_cast<double>(p);
    const double rate = static_cast<double>(kept) / n;
    const double bound = 6.0 * std::sqrt(keep * (1.0 - keep) / n) + 1.0 / n;
    EXPECT_LE(std::abs(rate - keep), bound) << "keep rate " << rate << ", expected " << keep;
}

/// What the CPU reference needs beyond Q, K, V and the dump.
struct RngDumpReferenceProblem
{
    RngDumpExtents extents;
    int64_t kvHeads = 0;
    int64_t headDim = 0;
    int64_t headDimV = 0;
    float scale = 1.0F;
    float p = 0.0F;
    /// The graph's bounds in the reference's terms (cuDNN's, SEM-001), -1 for none.
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeft = true;
    /// Per-batch lengths under padding; empty for a dense graph.
    std::vector<int32_t> seqLensQ;
    std::vector<int32_t> seqLensKv;
};

/**
 * @brief O against hipDNN's CPU reference (CpuFpReferenceSdpa::forward) run with the dump as
 *        its explicit mask and 1 / (1 - p) as its scale -- how hipDNN's references verify a
 *        Philox engine (their isSdpaPhiloxDropoutWithDump) -- within @p tolerance.
 *
 * @p qAt, @p kAt, @p vAt and @p oAt read element (b, h, s, e) of the engine's tensors as the
 * harness holds them.
 */
template <typename QAt, typename KAt, typename VAt, typename OAt>
void expectOutputIsTheReferenceUnderTheDump(const RngDumpReferenceProblem& problem,
                                            const std::vector<float>& dump,
                                            QAt qAt,
                                            KAt kAt,
                                            VAt vAt,
                                            OAt oAt,
                                            double tolerance)
{
    using hipdnn_data_sdk::utilities::Tensor;
    const auto& e = problem.extents;
    Tensor<float> q({e.batch, e.heads, e.seqLenQ, problem.headDim});
    Tensor<float> k({e.batch, problem.kvHeads, e.seqLenKv, problem.headDim});
    Tensor<float> v({e.batch, problem.kvHeads, e.seqLenKv, problem.headDimV});
    Tensor<float> o({e.batch, e.heads, e.seqLenQ, problem.headDimV});
    Tensor<float> mask({e.batch, e.heads, e.seqLenQ, e.seqLenKv});
    for(int64_t b = 0; b < e.batch; ++b)
    {
        for(int64_t h = 0; h < e.heads; ++h)
        {
            for(int64_t s = 0; s < e.seqLenQ; ++s)
            {
                for(int64_t d = 0; d < problem.headDim; ++d)
                {
                    q.setHostValue(static_cast<float>(qAt(b, h, s, d)), b, h, s, d);
                }
                for(int64_t j = 0; j < e.seqLenKv; ++j)
                {
                    mask.setHostValue(dump[e.index(b, h, s, j)], b, h, s, j);
                }
            }
        }
        for(int64_t h = 0; h < problem.kvHeads; ++h)
        {
            for(int64_t j = 0; j < e.seqLenKv; ++j)
            {
                for(int64_t d = 0; d < problem.headDim; ++d)
                {
                    k.setHostValue(static_cast<float>(kAt(b, h, j, d)), b, h, j, d);
                }
                for(int64_t d = 0; d < problem.headDimV; ++d)
                {
                    v.setHostValue(static_cast<float>(vAt(b, h, j, d)), b, h, j, d);
                }
            }
        }
    }
    std::optional<Tensor<int32_t>> lengthsQ;
    std::optional<Tensor<int32_t>> lengthsKv;
    hipdnn_test_sdk::utilities::SdpaFwdOptions<float> options;
    options.setScale(problem.scale)
        .setBand(problem.leftBound, problem.rightBound, problem.topLeft)
        .setDropout(&mask, flydsl_sdpa::dropoutScale(problem.p));
    if(!problem.seqLensQ.empty())
    {
        lengthsQ.emplace(std::vector<int64_t>{e.batch, 1, 1, 1});
        lengthsKv.emplace(std::vector<int64_t>{e.batch, 1, 1, 1});
        for(int64_t b = 0; b < e.batch; ++b)
        {
            lengthsQ->setHostValue(problem.seqLensQ[static_cast<size_t>(b)], b, 0, 0, 0);
            lengthsKv->setHostValue(problem.seqLensKv[static_cast<size_t>(b)], b, 0, 0, 0);
        }
        options.setSeqLens(&*lengthsQ, &*lengthsKv);
    }
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(q, k, v, o, options);

    double worst = 0.0;
    std::string where;
    for(int64_t b = 0; b < e.batch; ++b)
    {
        for(int64_t h = 0; h < e.heads; ++h)
        {
            for(int64_t s = 0; s < e.seqLenQ; ++s)
            {
                for(int64_t d = 0; d < problem.headDimV; ++d)
                {
                    const auto expected = static_cast<double>(o.getHostValue(b, h, s, d));
                    const double actual = oAt(b, h, s, d);
                    const double error = std::isfinite(actual) ? std::abs(actual - expected)
                                                               : std::numeric_limits<double>::max();
                    if(error > worst)
                    {
                        worst = error;
                        where = "b=" + std::to_string(b) + " h=" + std::to_string(h)
                                + " s=" + std::to_string(s) + " d=" + std::to_string(d) + ": got "
                                + std::to_string(actual) + ", reference "
                                + std::to_string(expected);
                    }
                }
            }
        }
    }
    EXPECT_LE(worst, tolerance) << where;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
