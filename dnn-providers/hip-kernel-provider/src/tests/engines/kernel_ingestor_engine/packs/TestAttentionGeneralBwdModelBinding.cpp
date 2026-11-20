// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ModelBindingTestSupport.hpp"

/**
 * @file TestAttentionGeneralBwdModelBinding.cpp
 * @brief Census of the trained main-slot ranker hipkernel:AttentionGeneralBwd binds in its
 *        UED for one architecture: it must survive the loader's pre-flight.
 *
 * The failure is silent at run time. The loader prunes a ranker it refuses (a trained_against
 * id that does not match the UED, a KMD or a UMD; a model file the packer lost) into
 * `unavailableHeuristicArches` with an error line, and the slot then ranks in declared order.
 * That order does not put the engine's own pick first, and `default` (the native score) is not
 * consulted for an arch whose own entry was refused, so the engine looks healthy while its
 * ranker is gone and no other test notices. This suite reads the loaded main slot the way the
 * runtime does and fails the build instead.
 *
 * A bound model the packed tree lacks (its `model.bin` never pulled) is judged as
 * ModelBindingTestSupport.hpp describes.
 *
 * Census binary only: every case needs the packed shard and HIPDNN_TEST_EXPECTED_ARCH, which
 * the hkp_register_census_tests() entry in src/tests/CMakeLists.txt supplies. A shard holds one
 * architecture's tree, so a model bound for the other architecture is refused there by design
 * (its provenance names UMDs only the other shard carries) and is not checked.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace ingestor = hipdnn_plugin_sdk::ingestor;

constexpr const char* ROLE = "main-slot sort_kernel_catalog";

class TestAttentionGeneralBwdModelBinding : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _arch = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_EXPECTED_ARCH");
        ASSERT_FALSE(_arch.empty()) << "HIPDNN_TEST_EXPECTED_ARCH is unset, so this census has "
                                       "no architecture to check against";
        ASSERT_TRUE(_arch == "gfx942" || _arch == "gfx950")
            << "the engine ships gfx942 and gfx950, not " << _arch;

        const auto& sets = discoverDescriptorSets();
        const auto match = std::find_if(sets.begin(), sets.end(), [](const auto& set) {
            return set.engine.name == ATTENTION_GENERAL_BWD_ENGINE_NAME;
        });
        ASSERT_NE(match, sets.end())
            << "descriptor set '" << ATTENTION_GENERAL_BWD_ENGINE_NAME
            << "' did not load: its descriptors were not installed in this shard";
        _set = &*match;

        const auto slot = std::find_if(
            _set->slots.begin(), _set->slots.end(), [](const auto& s) { return s.name == "main"; });
        ASSERT_NE(slot, _set->slots.end()) << "the loaded engine has no `main` slot";
        _slot = &*slot;
        const auto bound = std::find_if(_set->engine.slots.begin(),
                                        _set->engine.slots.end(),
                                        [](const auto& s) { return s.name == "main"; });
        ASSERT_NE(bound, _set->engine.slots.end()) << "the UED declares no `main` slot";
        _bound = &*bound;
    }

    std::string _arch;
    const ingestor::DescriptorSet* _set = nullptr;
    const ingestor::SlotDescriptorSet* _slot = nullptr;
    const ingestor::SlotDescriptor* _bound = nullptr;
};

} // namespace

TEST_F(TestAttentionGeneralBwdModelBinding, TheNativeScoreIsBoundAsTheMainSlotsDefault)
{
    // The native score ranks the engine's own pick first. It is the ranker of any
    // architecture without its own entry, so it must load wherever the engine does.
    const auto ids = boundIds(_bound->sortKernelCatalog, "default");
    ASSERT_EQ(ids.size(), 1u) << "the UED binds no `default` main-slot ranker";
    ASSERT_TRUE(_slot->heuristic.has_value()) << "the `default` main-slot ranker did not load";
    EXPECT_EQ(hipdnn_flatbuffers_sdk::utilities::formatUuid(_slot->heuristic->id), ids.front());
    EXPECT_FALSE(_slot->heuristic->nativeSymbol.empty())
        << "the `default` main-slot ranker is not the native score";
    EXPECT_TRUE(_slot->heuristic->score.metric.empty()) << "a slot ranker is metric-less";
}

TEST_F(TestAttentionGeneralBwdModelBinding, TheBoundMainSlotRankerLoads)
{
    const auto ids = boundIds(_bound->sortKernelCatalog, _arch);
    ASSERT_FALSE(ids.empty()) << "the UED binds no " << _arch
                              << " main-slot ranker: the trained model is not installed, and "
                                 "this arch ranks with the native score alone";
    ASSERT_EQ(ids.size(), 1u) << "the UED binds more than one " << _arch << " main-slot ranker";
    if(!packedTreeObligesLoad(ids, ROLE))
    {
        return;
    }

    // The loader prunes a ranker it refuses into unavailableHeuristicArches; a pruned one is
    // absent from the map, so presence is the load check, and the refusal list must not name
    // this architecture (it may name the other one: see the file comment).
    const auto refused = _slot->unavailableHeuristicArches.find("");
    EXPECT_TRUE(refused == _slot->unavailableHeuristicArches.end()
                || refused->second.count(_arch) == 0u)
        << "the loader refused the " << _arch
        << " main-slot ranker (see its error line); the slot ranks in declared order";
    const auto* model = findModel(_slot->heuristicsByMetric, "", _arch);
    ASSERT_NE(model, nullptr)
        << "no main-slot ranker for " << _arch
        << " survived the loader's pre-flight; the slot ranks in declared order, and the "
           "engine's own pick is no longer first";
    EXPECT_EQ(hipdnn_flatbuffers_sdk::utilities::formatUuid(model->id), ids.front())
        << "the loaded main-slot ranker is not the one the UED binds";
    EXPECT_TRUE(model->score.metric.empty()) << "a slot ranker is metric-less";
    ASSERT_TRUE(model->trainedAgainst.has_value()) << "the ranker records no provenance";
    EXPECT_EQ(model->trainedAgainst->slot, "main")
        << "the ranker was trained for another slot than the one it is bound to";
    EXPECT_EQ(model->trainedAgainst->kmd.id, _slot->schema.id)
        << "the ranker was trained against another main-slot KMD than the shipped one";
    expectModelArtifactPresent(*model, ROLE);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
