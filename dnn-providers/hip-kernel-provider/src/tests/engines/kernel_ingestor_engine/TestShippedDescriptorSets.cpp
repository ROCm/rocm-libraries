// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "TestDescriptorRoot.hpp"

/// @file TestShippedDescriptorSets.cpp
/// @brief RFC 0019 §12 descriptor-set validation over each staged descriptor tree (unit, and
///        production when this build packs one).
///
/// The loader drops invalid sets with only a warning, so this is where a bad shipped
/// descriptor fails the build. It reads the staged trees, since mis-staging is invisible in
/// source.
namespace hipdnn_plugin_sdk::ingestor
{
namespace
{

namespace detail
{
/// Which MetadataValue alternatives are lists (a trait; this target is not C++20).
template <typename T>
struct IsList : std::false_type
{
};
template <typename T>
struct IsList<std::vector<T>> : std::true_type
{
};
} // namespace detail

/// One staged descriptor tree this suite validates.
struct StagedTree
{
    /// Instantiation suffix.
    const char* name;
    /// The build target that stages the tree, named in the diagnostics.
    const char* packTarget;
    /// False when this build packs no such tree; every case then skips.
    bool stagedByThisBuild;
    /// Where the tree is. Called only when stagedByThisBuild.
    std::filesystem::path (*root)();
};

std::filesystem::path unitRoot()
{
    // The unit root this binary stages, holding every unit pack.
    return hip_kernel_provider::testing::descriptorSetRoot(HIPKERNELPROVIDER_TEST_SET_UNIT_RELDIR);
}

#ifdef HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR
constexpr bool PRODUCT_TREE_STAGED = true;

std::filesystem::path productRoot()
{
    // The arch-neutral production root, as the provider loads it.
    return hip_kernel_provider::testing::descriptorSetRoot(
        HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR);
}
#else
constexpr bool PRODUCT_TREE_STAGED = false;

std::filesystem::path productRoot()
{
    return {};
}
#endif

/// A kernel's completed metadata tuple: every schema field (not just knobs, e.g. `dtype`),
/// defaults applied, in schema order.
std::string metadataTuple(const MetadataSchema& schema, const MetadataValues& metadata)
{
    std::ostringstream tuple;
    for(const auto& field : schema.fields)
    {
        const auto& knob = field.name;
        tuple << knob << "=";
        if(const auto found = metadata.find(knob); found != metadata.end())
        {
            std::visit(
                [&tuple](const auto& held) {
                    using Held = std::decay_t<decltype(held)>;
                    if constexpr(detail::IsList<Held>::value)
                    {
                        // Element-wise, so kernels differing inside a list still differ.
                        for(const auto& element : held)
                        {
                            tuple << element << ",";
                        }
                    }
                    else
                    {
                        tuple << held;
                    }
                },
                found->second);
        }
        else if(field.defaultValue.has_value())
        {
            // An omitted optional field resolves to its default.
            std::visit(
                [&tuple](const auto& held) {
                    using Held = std::decay_t<decltype(held)>;
                    if constexpr(detail::IsList<Held>::value)
                    {
                        for(const auto& element : held)
                        {
                            tuple << element << ",";
                        }
                    }
                    else
                    {
                        tuple << held;
                    }
                },
                *field.defaultValue);
        }
        else
        {
            tuple << "<unset>";
        }
        tuple << ";";
    }
    return tuple.str();
}

/// Parameterized over the staged trees; each tree is loaded once and shared across cases.
class TestShippedDescriptorSets : public ::testing::TestWithParam<StagedTree>
{
protected:
    void SetUp() override
    {
        const auto& tree = GetParam();
        if(!tree.stagedByThisBuild)
        {
            GTEST_SKIP() << "this build packs no " << tree.name
                         << " descriptor tree: " << tree.packTarget
                         << " is not wired, because no production descriptor declares an "
                            "architecture in GPU_TARGETS (or the production root is empty)";
        }
    }

    static std::filesystem::path root()
    {
        return GetParam().root();
    }

    static const std::vector<DescriptorSet>& sets()
    {
        static std::map<std::filesystem::path, std::vector<DescriptorSet>> s_loaded;
        const auto treeRoot = root();
        auto found = s_loaded.find(treeRoot);
        if(found == s_loaded.end())
        {
            found
                = s_loaded.emplace(treeRoot, resolveDescriptorSets(loadDescriptorCatalog(treeRoot)))
                      .first;
        }
        return found->second;
    }
};

TEST_P(TestShippedDescriptorSets, TheStagedTreeContainsDescriptorsAtAll)
{
    // Guards every other case: an empty tree (e.g. after a reconfigure wipes staging) makes
    // them pass vacuously.
    ASSERT_TRUE(std::filesystem::exists(root()))
        << "no staged descriptor tree at " << root() << ". Build " << GetParam().packTarget
        << " first.";
    EXPECT_FALSE(sets().empty()) << "the staged tree at " << root() << " parsed to zero engines";
}

TEST_P(TestShippedDescriptorSets, EveryEngineResolvesItsMetadataSchema)
{
    // RFC 0019 §4: without its KMD schema an engine cannot type-check its knobs.
    for(const auto& set : sets())
    {
        EXPECT_FALSE(set.schema.fields.empty())
            << "engine '" << set.engine.name << "' resolved no metadata schema";
    }
}

TEST_P(TestShippedDescriptorSets, EveryAdvertisedKnobIsDeclaredInTheSchema)
{
    // A knob with no schema field has no type or default, so a query for it silently
    // returns nothing.
    for(const auto& set : sets())
    {
        std::set<std::string> declared;
        for(const auto& field : set.schema.fields)
        {
            declared.insert(field.name);
        }

        for(const auto& knob : set.engine.knobs)
        {
            EXPECT_TRUE(declared.count(knob) != 0)
                << "engine '" << set.engine.name << "' advertises knob '" << knob
                << "' that its metadata schema does not declare";
        }
    }
}

TEST_P(TestShippedDescriptorSets, EveryHeuristicReferenceResolves)
{
    // RFC 0019 §3.1: a dangling reference silently degrades to declared order.
    for(const auto& set : sets())
    {
        if(set.engine.sortKernelCatalog.empty())
        {
            continue; // shipping no model is a legitimate choice; §5 step 7 covers it
        }

        EXPECT_TRUE(set.heuristic.has_value() || !set.heuristicsByMetric.empty())
            << "engine '" << set.engine.name << "' names a heuristic that did not resolve";
    }
}

TEST_P(TestShippedDescriptorSets, EveryHeuristicDeclaresSomethingToScoreWith)
{
    // A UHD needs a native symbol or an on-disk artifact.
    for(const auto& set : sets())
    {
        std::vector<const HeuristicDescriptor*> all;
        if(set.heuristic.has_value())
        {
            all.push_back(&*set.heuristic);
        }
        for(const auto& [metric, byArch] : set.heuristicsByMetric)
        {
            for(const auto& [arch, heuristic] : byArch)
            {
                all.push_back(&heuristic);
            }
        }

        for(const auto* heuristic : all)
        {
            // What counts as scorable is adapter-specific; there is no common payload field.
            switch(heuristic->adapter)
            {
            case UhdAdapter::STATIC_ORDER:
                // Declared fields only, all optional; nothing to require.
                break;

            case UhdAdapter::NATIVE:
                EXPECT_FALSE(heuristic->nativeSymbol.empty())
                    << "engine '" << set.engine.name << "' ships heuristic '" << heuristic->name
                    << "' with an empty native symbol";
                // Resolved at registration, not on disk.
                break;

            case UhdAdapter::TREE_DATA:
            case UhdAdapter::TABLE:
            case UhdAdapter::CUSTOM_LIBRARY:
            {
                ASSERT_FALSE(heuristic->modelArtifactPath.empty())
                    << "engine '" << set.engine.name << "' ships heuristic '" << heuristic->name
                    << "' with an empty model artifact path";

                const auto artifact = heuristic->baseDir / heuristic->modelArtifactPath;
                EXPECT_TRUE(std::filesystem::exists(artifact))
                    << "engine '" << set.engine.name << "' ships heuristic '" << heuristic->name
                    << "' whose artifact is missing: " << artifact;
                break;
            }

            // -Wswitch-default; every enum member is handled above.
            default:
                ADD_FAILURE() << "engine '" << set.engine.name << "' ships heuristic '"
                              << heuristic->name << "' naming an adapter this test does not know";
                break;
            }
        }
    }
}

TEST_P(TestShippedDescriptorSets, NoTwoKernelsOfAPackShareAMetadataTuple)
{
    // The completed tuple is the catalog key, so duplicates make the choice an accident of
    // catalog order. Scoped per pack: different packs answer different graphs.
    for(const auto& set : sets())
    {
        if(set.schema.fields.empty())
        {
            continue; // reported by EveryEngineResolvesItsMetadataSchema
        }

        for(const auto& pack : set.packs)
        {
            std::map<std::string, std::string> seen; // tuple -> first kernel that claimed it
            for(const auto& kernel : pack.kernels)
            {
                const auto tuple = metadataTuple(set.schema, kernel.metadata);
                const auto [entry, inserted] = seen.emplace(tuple, kernel.name);
                EXPECT_TRUE(inserted) << "engine '" << set.engine.name << "' pack '" << pack.name
                                      << "': kernels '" << entry->second << "' and '" << kernel.name
                                      << "' share metadata tuple " << tuple;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestShippedDescriptorSets,
    ::testing::Values(StagedTree{"Unit", "hkp_packaging_unit", true, &unitRoot},
                      StagedTree{
                          "Product", "hkp_packaging_product", PRODUCT_TREE_STAGED, &productRoot}),
    [](const ::testing::TestParamInfo<StagedTree>& info) { return std::string(info.param.name); });

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor
