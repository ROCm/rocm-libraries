// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/EngineOrdering.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionGeneralBwdTestSupport.hpp"

/**
 * @file TestAttentionGeneralBwdPacks.cpp
 * @brief A census of what hipkernel:AttentionGeneralBwd loads for one architecture, read
 *        through discoverDescriptorSets() from the packed shard the census entry names.
 *
 * The engine is slotted: a `prep` slot of single-launch kernels and a `main` slot whose
 * kernels each declare the main, dQ and dK/dV convert launches. The shard must hold
 * exactly the catalog the C++ plan selects from (AttentionGeneralBwdCatalog.generated.hpp)
 * for its architecture, each kernel carrying the KMD record the kernel matcher encodes,
 * each launch the symbol the plan names and the argument list the handler marshals.
 *
 * Census binary only: every case needs the shard and HIPDNN_TEST_EXPECTED_ARCH, which the
 * hkp_register_census_tests() entry in src/tests/CMakeLists.txt supplies.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace general_bwd_test;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

constexpr std::size_t AOT_BUDGET_PER_PACK = 1000;

std::vector<ingestor::KernelArgument> signatureOf(const generated::AbiParam* abi, std::size_t size)
{
    std::vector<ingestor::KernelArgument> signature;
    uint32_t offset = 0;
    for(std::size_t i = 0; i < size; ++i)
    {
        const bool pointer = abi[i].kind == 'Q';
        const uint32_t bytes = pointer || abi[i].kind == 'q' ? 8U : 4U;
        offset = (offset + bytes - 1) / bytes * bytes;
        signature.push_back(ingestor::KernelArgument{
            pointer ? "global_buffer" : "by_value", bytes, offset, std::string(abi[i].name)});
        offset += bytes;
    }
    return signature;
}

template <std::size_t N>
std::vector<ingestor::KernelArgument> signatureOf(const std::array<generated::AbiParam, N>& abi)
{
    return signatureOf(abi.data(), N);
}

class TestAttentionGeneralBwdPacks : public ::testing::Test
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
            << "' did not load: its descriptors were not installed in this shard, or its "
               "native symbols did not register and every descriptor naming one was dropped";
        _set = &*match;

        for(const auto& row : generated::CATALOG)
        {
            if(row.arch == _arch)
            {
                _rows.push_back(&row);
            }
        }
        ASSERT_FALSE(_rows.empty());
    }

    const ingestor::SlotDescriptorSet* slot(std::string_view name) const
    {
        for(const auto& candidate : _set->slots)
        {
            if(candidate.name == name)
            {
                return &candidate;
            }
        }
        return nullptr;
    }

    std::vector<const ingestor::KernelDescriptor*> kernelsOf(std::string_view slotName) const
    {
        std::vector<const ingestor::KernelDescriptor*> kernels;
        if(const auto* found = slot(slotName))
        {
            for(const auto& pack : found->packs)
            {
                for(const auto& kernel : pack.kernels)
                {
                    kernels.push_back(&kernel);
                }
            }
        }
        return kernels;
    }

    std::string _arch;
    const ingestor::DescriptorSet* _set = nullptr;
    std::vector<const generated::CatalogRow*> _rows;
};

} // namespace

TEST_F(TestAttentionGeneralBwdPacks, DeclaresThePrepAndMainSlotsAndTheSharedWorkspace)
{
    ASSERT_EQ(_set->engine.slots.size(), 2u);
    EXPECT_EQ(_set->engine.slots[0].name, "prep");
    EXPECT_EQ(_set->engine.slots[1].name, "main");
    EXPECT_EQ(_set->engine.sharedWorkspaceToken, "ws.bytes");
    EXPECT_EQ(_set->engine.graphMatchNativeSymbol, GRAPH_MATCHER_SYMBOL);
    EXPECT_EQ(_set->engine.sdkVersion, hipdnn_data_sdk::utilities::Version{std::string("1.4.0")});
}

TEST_F(TestAttentionGeneralBwdPacks, LoadsExactlyThePacksOfTheRequestedArch)
{
    std::set<std::string> prep;
    std::set<std::string> main;
    for(const auto* name : {"prep", "main"})
    {
        const auto* found = slot(name);
        ASSERT_NE(found, nullptr) << name;
        for(const auto& pack : found->packs)
        {
            EXPECT_EQ(pack.slot, name);
            EXPECT_TRUE(ingestor::archSupports(pack.arch, _arch))
                << "pack '" << pack.name << "' does not cover " << _arch;
            EXPECT_LE(pack.kernels.size(), AOT_BUDGET_PER_PACK) << pack.name;
            (std::string_view(name) == "prep" ? prep : main).insert(pack.name);
        }
    }
    EXPECT_EQ(prep, std::set<std::string>{"hipkernel:attention_general_bwd_prep_" + _arch});
    std::set<std::string> expectedMain{"hipkernel:attention_general_bwd_main_" + _arch};
    if(_arch == "gfx950")
    {
        expectedMain.insert("hipkernel:attention_general_bwd_main_splitdq_gfx950");
    }
    EXPECT_EQ(main, expectedMain);
}

TEST_F(TestAttentionGeneralBwdPacks, EveryMainKernelDeclaresTheFourMainLaunches)
{
    const std::set<std::string> launches{"main", "dq", "convert_dk", "convert_dv"};
    for(const auto* kernel : kernelsOf("main"))
    {
        std::set<std::string> names;
        for(const auto& [name, source] : kernel->sources)
        {
            names.insert(name);
            EXPECT_EQ(source.kind, ingestor::KernelSourceKind::KPACK)
                << kernel->name << " " << name;
        }
        EXPECT_EQ(names, launches) << kernel->name;
    }
    for(const auto* kernel : kernelsOf("prep"))
    {
        EXPECT_TRUE(kernel->sources.empty()) << kernel->name;
        EXPECT_EQ(kernel->source.kind, ingestor::KernelSourceKind::KPACK) << kernel->name;
    }
}

TEST_F(TestAttentionGeneralBwdPacks, EveryKernelIsOneCatalogRowAndEveryRowShips)
{
    std::map<const generated::CatalogRow*, std::size_t> mainHits;
    for(const auto* kernel : kernelsOf("main"))
    {
        std::vector<const generated::CatalogRow*> rows;
        for(const auto* row : _rows)
        {
            if(mainKernelOf(*row).metadata == kernel->metadata)
            {
                rows.push_back(row);
            }
        }
        ASSERT_EQ(rows.size(), 1u) << kernel->name << " is not exactly one catalog row";
        ++mainHits[rows.front()];
    }
    EXPECT_EQ(mainHits.size(), _rows.size()) << "a catalog row ships no main kernel";

    std::set<ingestor::MetadataValues> prepRecords;
    for(const auto* row : _rows)
    {
        prepRecords.insert(prepKernelOf(*row).metadata);
    }
    std::set<ingestor::MetadataValues> shippedPrep;
    for(const auto* kernel : kernelsOf("prep"))
    {
        EXPECT_EQ(prepRecords.count(kernel->metadata), 1u) << kernel->name;
        EXPECT_TRUE(shippedPrep.insert(kernel->metadata).second)
            << kernel->name << " repeats another prep kernel's record";
    }
    EXPECT_EQ(shippedPrep, prepRecords) << "a prep kernel the catalog launches does not ship";
}

TEST_F(TestAttentionGeneralBwdPacks, EveryLaunchIsTheSymbolThePlanNames)
{
    for(const auto* kernel : kernelsOf("main"))
    {
        const generated::CatalogRow* match = nullptr;
        for(const auto* row : _rows)
        {
            if(mainKernelOf(*row).metadata == kernel->metadata)
            {
                match = row;
            }
        }
        ASSERT_NE(match, nullptr) << kernel->name;
        const auto& sources = kernel->sources;
        EXPECT_EQ(sources.at("main").symbol, match->mainKernel) << kernel->name;
        EXPECT_EQ(sources.at("dq").symbol,
                  match->dq == generated::DqMode::SPLIT ? match->dqKernel : match->convertKernel)
            << kernel->name;
        EXPECT_EQ(sources.at("convert_dk").symbol, match->convertKernel) << kernel->name;
        EXPECT_EQ(sources.at("convert_dv").symbol, match->convertKernel) << kernel->name;
    }
    for(const auto* kernel : kernelsOf("prep"))
    {
        bool known = false;
        for(const auto* row : _rows)
        {
            known = known
                    || (prepKernelOf(*row).metadata == kernel->metadata
                        && kernel->source.symbol == row->prepKernel);
        }
        EXPECT_TRUE(known) << kernel->name << " packs symbol " << kernel->source.symbol;
    }
}

TEST_F(TestAttentionGeneralBwdPacks, EveryLaunchRecordsTheAbiTheHandlerMarshals)
{
    const auto prep = signatureOf(generated::ABI_PREP);
    const auto main = signatureOf(generated::ABI_MAIN);
    const auto convert = signatureOf(generated::ABI_CONVERT);
    const auto dq = signatureOf(generated::ABI_DQ);
    for(const auto* kernel : kernelsOf("prep"))
    {
        EXPECT_NO_THROW(requireSignatureMatch(
            kernel->source.signature, prep, kernel->source.symbol, kernel->name));
    }
    for(const auto* kernel : kernelsOf("main"))
    {
        const bool split
            = kernel->metadata.at("dq_mode") == ingestor::MetadataValue{std::string("split")};
        for(const auto& [name, source] : kernel->sources)
        {
            const auto& postMain = name == "dq" && split ? dq : convert;
            const auto& expected = name == "main" ? main : postMain;
            EXPECT_NO_THROW(
                requireSignatureMatch(source.signature, expected, source.symbol, kernel->name))
                << name;
        }
    }
}

// Where both engines serve a graph (gfx942), ASM SDPA is faster, and this engine adds no
// Config rule: it relies on the static order ranking ASM first, with no engine-prediction
// model that could score it above ASM.
TEST_F(TestAttentionGeneralBwdPacks, AsmSdpaRanksAheadOfThisEngine)
{
    using namespace hipdnn_data_sdk::utilities;
    const int64_t generalBwd = engineNameToId(ATTENTION_GENERAL_BWD_ENGINE_NAME);
    std::vector<int64_t> ids{generalBwd, ASM_SDPA_ENGINE_ID};
    sortEngineIds(ids);
    EXPECT_EQ(ids, (std::vector<int64_t>{ASM_SDPA_ENGINE_ID, generalBwd}));

    EXPECT_TRUE(_set->engine.predictEngine.empty());
    EXPECT_TRUE(_set->enginePredictionsByMetric.empty());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
