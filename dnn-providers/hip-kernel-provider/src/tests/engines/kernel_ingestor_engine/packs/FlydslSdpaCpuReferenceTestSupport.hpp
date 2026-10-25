// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>

/**
 * @file FlydslSdpaCpuReferenceTestSupport.hpp
 * @brief An engine's O and LSE against hipDNN's CPU reference (CpuFpReferenceSdpa::forward),
 *        run on the same inputs the device read: the check every family's device harness
 *        runs on an attention sink, which the reference takes as FLOAT [1, Hq, 1, 1] (cuDNN's
 *        dtype; SEM-022), with a bias, a band and padding as the graph has them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// What the CPU reference needs beyond the operands' values.
struct SdpaCpuReferenceProblem
{
    int64_t batch = 0;
    int64_t heads = 0;
    int64_t kvHeads = 0;
    int64_t valueHeads = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t headDim = 0;
    int64_t headDimV = 0;
    float scale = 1.0F;
    /// The graph's bounds in the reference's terms (cuDNN's, SEM-001), -1 for none.
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeft = true;
    /// Per-batch lengths under padding; empty for a dense graph.
    std::vector<int32_t> seqLensQ;
    std::vector<int32_t> seqLensKv;
    /// The sink's logit per query head, as the device read it (widened exactly to f32 when it
    /// was in the query's dtype); empty for no sink.
    std::vector<float> sink;
    bool withBias = false;
    /// Whether the engine wrote the LSE.
    bool withLse = false;
};

/**
 * @brief O within @p tolerance, and the LSE within @p lseTolerance, of hipDNN's CPU reference.
 *
 * @p qAt, @p kAt, @p vAt and @p oAt read element (b, h, s, e) of the engine's tensors as the
 * harness holds them, @p biasAt the bias at (b, h, s, j) broadcast to [B, Hq, Sq, Skv], and
 * @p lseAt the LSE at (b, h, s). An LSE of -inf must be exactly -inf.
 */
template <typename QAt, typename KAt, typename VAt, typename BiasAt, typename OAt, typename LseAt>
void expectSdpaIsTheCpuReference(const SdpaCpuReferenceProblem& problem,
                                 QAt qAt,
                                 KAt kAt,
                                 VAt vAt,
                                 BiasAt biasAt,
                                 OAt oAt,
                                 LseAt lseAt,
                                 double tolerance,
                                 double lseTolerance)
{
    using hipdnn_data_sdk::utilities::Tensor;
    const auto& p = problem;
    Tensor<float> q({p.batch, p.heads, p.seqLenQ, p.headDim});
    Tensor<float> k({p.batch, p.kvHeads, p.seqLenKv, p.headDim});
    Tensor<float> v({p.batch, p.valueHeads, p.seqLenKv, p.headDimV});
    Tensor<float> o({p.batch, p.heads, p.seqLenQ, p.headDimV});
    Tensor<float> lse({p.batch, p.heads, p.seqLenQ, 1});
    std::optional<Tensor<float>> bias;
    if(p.withBias)
    {
        bias.emplace(std::vector<int64_t>{p.batch, p.heads, p.seqLenQ, p.seqLenKv});
    }
    for(int64_t b = 0; b < p.batch; ++b)
    {
        for(int64_t h = 0; h < p.heads; ++h)
        {
            for(int64_t s = 0; s < p.seqLenQ; ++s)
            {
                for(int64_t e = 0; e < p.headDim; ++e)
                {
                    q.setHostValue(static_cast<float>(qAt(b, h, s, e)), b, h, s, e);
                }
                for(int64_t j = 0; bias.has_value() && j < p.seqLenKv; ++j)
                {
                    bias->setHostValue(static_cast<float>(biasAt(b, h, s, j)), b, h, s, j);
                }
            }
        }
        for(int64_t h = 0; h < p.kvHeads; ++h)
        {
            for(int64_t j = 0; j < p.seqLenKv; ++j)
            {
                for(int64_t e = 0; e < p.headDim; ++e)
                {
                    k.setHostValue(static_cast<float>(kAt(b, h, j, e)), b, h, j, e);
                }
            }
        }
        for(int64_t h = 0; h < p.valueHeads; ++h)
        {
            for(int64_t j = 0; j < p.seqLenKv; ++j)
            {
                for(int64_t e = 0; e < p.headDimV; ++e)
                {
                    v.setHostValue(static_cast<float>(vAt(b, h, j, e)), b, h, j, e);
                }
            }
        }
    }
    std::optional<Tensor<float>> sink;
    if(!p.sink.empty())
    {
        sink.emplace(std::vector<int64_t>{1, p.heads, 1, 1});
        for(int64_t h = 0; h < p.heads; ++h)
        {
            sink->setHostValue(p.sink[static_cast<size_t>(h)], 0, h, 0, 0);
        }
    }
    std::optional<Tensor<int32_t>> lengthsQ;
    std::optional<Tensor<int32_t>> lengthsKv;
    hipdnn_test_sdk::utilities::SdpaFwdOptions<float> options;
    options.setScale(p.scale).setBand(p.leftBound, p.rightBound, p.topLeft).setLse(&lse);
    if(sink.has_value())
    {
        options.setSink(&*sink);
    }
    if(bias.has_value())
    {
        options.setMask(&*bias);
    }
    if(!p.seqLensQ.empty())
    {
        lengthsQ.emplace(std::vector<int64_t>{p.batch, 1, 1, 1});
        lengthsKv.emplace(std::vector<int64_t>{p.batch, 1, 1, 1});
        for(int64_t b = 0; b < p.batch; ++b)
        {
            lengthsQ->setHostValue(p.seqLensQ[static_cast<size_t>(b)], b, 0, 0, 0);
            lengthsKv->setHostValue(p.seqLensKv[static_cast<size_t>(b)], b, 0, 0, 0);
        }
        options.setSeqLens(&*lengthsQ, &*lengthsKv);
    }
    hipdnn_test_sdk::utilities::CpuFpReferenceSdpa::forward(q, k, v, o, options);

    double worst = 0.0;
    std::string where;
    double worstLse = 0.0;
    std::string whereLse;
    for(int64_t b = 0; b < p.batch; ++b)
    {
        for(int64_t h = 0; h < p.heads; ++h)
        {
            for(int64_t s = 0; s < p.seqLenQ; ++s)
            {
                const auto at = "b=" + std::to_string(b) + " h=" + std::to_string(h)
                                + " s=" + std::to_string(s);
                for(int64_t e = 0; e < p.headDimV; ++e)
                {
                    const auto expected = static_cast<double>(o.getHostValue(b, h, s, e));
                    const double actual = oAt(b, h, s, e);
                    const double error = std::isfinite(actual) ? std::abs(actual - expected)
                                                               : std::numeric_limits<double>::max();
                    if(error > worst)
                    {
                        worst = error;
                        where = at + " d=" + std::to_string(e) + ": got " + std::to_string(actual)
                                + ", reference " + std::to_string(expected);
                    }
                }
                if(!p.withLse)
                {
                    continue;
                }
                const auto expected = static_cast<double>(lse.getHostValue(b, h, s, 0));
                const double actual = lseAt(b, h, s);
                double error = std::numeric_limits<double>::max();
                if(std::isinf(expected) && expected < 0.0)
                {
                    error = std::isinf(actual) && actual < 0.0 ? 0.0 : error;
                }
                else if(std::isfinite(actual))
                {
                    error = std::abs(actual - expected);
                }
                if(error > worstLse)
                {
                    worstLse = error;
                    whereLse = at + ": got " + std::to_string(actual) + ", reference "
                               + std::to_string(expected);
                }
            }
        }
    }
    EXPECT_LE(worst, tolerance) << "O against CpuFpReferenceSdpa, " << where;
    EXPECT_LE(worstLse, lseTolerance) << "LSE against CpuFpReferenceSdpa, " << whereLse;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
