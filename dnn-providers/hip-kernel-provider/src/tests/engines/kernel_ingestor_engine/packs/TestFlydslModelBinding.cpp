// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose, as in the TestFlydsl*Packs suites: an ungated
// source list then yields an empty translation unit rather than a suite reading a shard
// nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestFlydslModelBinding.cpp
 * @brief Census of the trained models the FlyDSL engines bind for the shard's
 *        architecture: each bound ranker (L2) must survive the loader's pre-flight, and
 *        each bound engine predictor (L1) must record the selector revision this build
 *        computes.
 *
 * A model stamped with another revision is refused by GenericEngine (the estimate reports
 * PredictionStatus::UNAVAILABLE) with only a log line, and a ranker the loader refuses
 * leaves the kernels in priority order; every other census and generator check stays
 * green. This suite reads the revision the way makeEngine() does and fails instead, as
 * the rocKE ModelBinding suites do (ModelBindingTestSupport.hpp).
 *
 * One suite covers every FlyDSL engine, because FlyDSL ops ship for different
 * architectures: each case checks the engines this shard holds and fails if it holds
 * none. An engine binding no model for this architecture is held to loading none, so the
 * check stays live the day one is installed. Registered with hkp_register_census_tests()
 * in src/tests/CMakeLists.txt over every architecture a FlyDSL op ships for.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

constexpr const char* METRIC = "tflops";

/// Every FlyDSL engine, whether or not it ships models today.
constexpr std::array<std::string_view, 10> FLYDSL_ENGINES{
    "hipkernel:flydsl_rmsnorm",
    "hipkernel:flydsl_rmsnorm_bwd",
    "hipkernel:flydsl_sdpa_aiw",
    "hipkernel:flydsl_sdpa_cdna",
    "hipkernel:flydsl_sdpa_cdna_decode",
    "hipkernel:flydsl_sdpa_cdna_dualwave",
    "hipkernel:flydsl_sdpa_rdna",
    "hipkernel:flydsl_sdpa_rdna_decode",
    "hipkernel:flydsl_sdpa_bwd_merged",
    "hipkernel:flydsl_sdpa_bwd_split",
};

/// The FlyDSL engines this shard holds.
std::vector<const hipdnn_plugin_sdk::ingestor::DescriptorSet*> loadedFlydslSets()
{
    std::vector<const hipdnn_plugin_sdk::ingestor::DescriptorSet*> sets;
    for(const auto engine : FLYDSL_ENGINES)
    {
        if(const auto* set = findLoadedSet(engine))
        {
            sets.push_back(set);
        }
    }
    return sets;
}

/// The architecture the shard was packed for, which is also the key a UED binds its models
/// under: the census reads one arch's shard, and every FlyDSL pack in it names that arch.
/// Read across all the engines, because the split engine's set holds no pack of its own
/// (its kernels are its slot ops').
std::string shardArch(const std::vector<const hipdnn_plugin_sdk::ingestor::DescriptorSet*>& sets)
{
    std::set<std::string> arches;
    for(const auto* set : sets)
    {
        for(const auto& pack : set->packs)
        {
            arches.insert(pack.arch.begin(), pack.arch.end());
        }
    }
    EXPECT_EQ(arches.size(), 1U) << "the FlyDSL packs of this shard do not name exactly one "
                                    "architecture";
    return arches.size() == 1 ? *arches.begin() : std::string();
}

TEST(TestFlydslModelBinding, EveryBoundL2KernelRankerLoads)
{
    const auto sets = loadedFlydslSets();
    ASSERT_FALSE(sets.empty()) << "this shard holds no FlyDSL engine";
    const auto arch = shardArch(sets);
    ASSERT_FALSE(arch.empty());
    for(const auto* set : sets)
    {
        SCOPED_TRACE(set->engine.name + " on " + arch);
        if(boundIds(set->engine.sortKernelCatalog, arch).empty())
        {
            // No ranker for this arch: the kernels rank by priority, and none may load.
            EXPECT_EQ(findModel(set->heuristicsByMetric, METRIC, arch), nullptr)
                << "the UED binds no " << arch << " sort_kernel_catalog ranker, yet one loaded";
            continue;
        }
        expectBoundRankerLoads(*set, METRIC, arch);
    }
}

TEST(TestFlydslModelBinding, EveryBoundL1EnginePredictorLoadsAndMatchesThisBuild)
{
    const auto sets = loadedFlydslSets();
    ASSERT_FALSE(sets.empty()) << "this shard holds no FlyDSL engine";
    const auto arch = shardArch(sets);
    ASSERT_FALSE(arch.empty());
    for(const auto* set : sets)
    {
        SCOPED_TRACE(set->engine.name + " on " + arch);
        expectBoundPredictorMatchesThisBuild(*set, METRIC, arch);
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
