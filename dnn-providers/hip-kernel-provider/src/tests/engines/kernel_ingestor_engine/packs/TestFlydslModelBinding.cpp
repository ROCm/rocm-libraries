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

#include "tests/engines/kernel_ingestor_engine/packs/FlydslRmsNormBwdTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/FlydslSdpaBwdTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestFlydslModelBinding.cpp
 * @brief Census of the trained models the FlyDSL engines bind for the shard's
 *        architecture. Each bound ranker (L2) must survive the loader's pre-flight, build
 *        its scorer the way the runtime builds it (the model file hashes to the digest its
 *        UHD records and its weights parse), and rank the kernels the packs ship for a
 *        graph its engine serves. Each bound engine predictor (L1) must record the selector
 *        revision this build computes, build its scorer the same way, and give a finite
 *        AVAILABLE prediction for a graph its engine serves.
 *
 * A model stamped with another revision is refused by GenericEngine (the estimate reports
 * PredictionStatus::UNAVAILABLE) with only a log line. A ranker the loader refuses, and a
 * model whose bytes do not hash to the digest its UHD records or whose weights do not parse
 * (a foreign or corrupted model.bin loads, then is refused when its scorer is built), leave
 * the kernels in priority order or the engine without an estimate; every other census and
 * generator check stays green. This suite reads the revision the way makeEngine() does,
 * builds each scorer and asks it once, and fails instead, as the rocKE ModelBinding suites
 * do for the checks they share (ModelBindingTestSupport.hpp). The graphs it asks about are
 * built host-only with the engines' own test graph helpers (probeGraphs()), so no device
 * is needed; a bound model whose engine serves none of them fails the case.
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

/// A graph with each shape family the FlyDSL engines serve, to ask a bound model about: the
/// first one an engine's graph match accepts on the shard's architecture. Built with the
/// graph helpers of the engines' own suites; nothing runs.
ProbeGraphs probeGraphs()
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    ProbeGraphs probes;
    const auto sdpa = [&probes](const std::string& name, const SdpaGraphSpec& spec) {
        probes.push_back(
            {"SDPA forward " + name, [spec] { return buildFlydslSdpaRdnaGraph(spec); }});
    };
    const auto withShape = [](data_objects::DataType dtype,
                              int64_t seqQ,
                              int64_t seqKv,
                              int64_t headDim,
                              SdpaMask mask = SdpaMask::NONE) {
        SdpaGraphSpec spec;
        spec.dataType = dtype;
        spec.seqLenQ = seqQ;
        spec.seqLenKv = seqKv;
        spec.headDim = headDim;
        spec.mask = mask;
        return spec;
    };
    sdpa("bf16 prefill", withShape(data_objects::DataType::BFLOAT16, 128, 128, 64));
    sdpa("fp16 prefill", withShape(data_objects::DataType::HALF, 128, 128, 64));
    sdpa("bf16 head dim 128", withShape(data_objects::DataType::BFLOAT16, 512, 512, 128));
    sdpa("bf16 causal",
         withShape(data_objects::DataType::BFLOAT16, 512, 512, 128, SdpaMask::TOP_LEFT_CAUSAL));
    sdpa("bf16 decode", withShape(data_objects::DataType::BFLOAT16, 1, 1024, 128));
    sdpa("fp16 decode", withShape(data_objects::DataType::HALF, 1, 1024, 64));

    probes.push_back({"SDPA backward bf16", [] {
                          SdpaBwdSpec spec;
                          spec.dtype = data_objects::DataType::BFLOAT16;
                          return buildSdpaBwdGraph(spec);
                      }});
    probes.push_back({"SDPA backward fp16", [] { return buildSdpaBwdGraph(SdpaBwdSpec{}); }});
    // Rank-4 operands, which the frontend lowers an RMSNorm to, and the rank-2 rows the
    // engines' own suites build. A model answers only for the ranks its features cover.
    probes.push_back({"RMSNorm forward bf16 rank 4", [] {
                          RmsNormGraphExtras extras;
                          extras.epsilonDims = {1, 1, 1, 1};
                          return buildFlydslRmsNormGraph(data_objects::DataType::BFLOAT16,
                                                         {8, 4, 8, 128},
                                                         FLYDSL_TEST_EPSILON,
                                                         false,
                                                         std::vector<int64_t>{1, 4, 8, 128},
                                                         std::nullopt,
                                                         std::nullopt,
                                                         false,
                                                         false,
                                                         data_objects::NormFwdPhase::NOT_SET,
                                                         extras);
                      }});
    probes.push_back({"RMSNorm backward bf16 rank 4", [] {
                          RmsNormBwdGraphSpec spec;
                          spec.xDims = {8, 4, 8, 128};
                          spec.scaleDims = std::vector<int64_t>{1, 4, 8, 128};
                          return buildFlydslRmsNormBwdGraph(spec);
                      }});
    probes.push_back({"RMSNorm forward bf16 rank 2", [] { return buildFlydslRmsNormGraph(); }});
    probes.push_back({"RMSNorm backward bf16 rank 2", [] { return buildFlydslRmsNormBwdGraph(); }});
    return probes;
}

TEST(TestFlydslModelBinding, EveryBoundL2KernelRankerLoads)
{
    const auto sets = loadedFlydslSets();
    ASSERT_FALSE(sets.empty()) << "this shard holds no FlyDSL engine";
    const auto arch = shardArch(sets);
    ASSERT_FALSE(arch.empty());
    const auto probes = probeGraphs();
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
        expectBoundRankerLoads(*set, METRIC, arch, probes);
    }
}

TEST(TestFlydslModelBinding, EveryBoundL1EnginePredictorLoadsAndMatchesThisBuild)
{
    const auto sets = loadedFlydslSets();
    ASSERT_FALSE(sets.empty()) << "this shard holds no FlyDSL engine";
    const auto arch = shardArch(sets);
    ASSERT_FALSE(arch.empty());
    const auto probes = probeGraphs();
    for(const auto* set : sets)
    {
        SCOPED_TRACE(set->engine.name + " on " + arch);
        expectBoundPredictorMatchesThisBuild(*set, METRIC, arch, probes);
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
