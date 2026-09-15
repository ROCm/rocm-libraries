// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <set>
#include <sstream>
#include <type_traits>
#include <variant>
#include <string>
#include <vector>

/// @file TestShippedDescriptorSets.cpp
/// @brief RFC 0019 §12's descriptor-set validation, run over the descriptors this build ships.
///
/// The loader already enforces these rules -- and enforces them by *dropping* the offending set,
/// which is the right behavior at runtime for third-party data and the wrong place to find out
/// about our own. A malformed shipped descriptor produces a warning in a log nobody reads and an
/// engine that silently declines every graph; every functional test still passes, because they
/// assert on what the surviving engines do. So the check has to run somewhere that fails a build.
///
/// It reads the staged tree rather than the source tree: staging is what packaging produces and
/// what the provider loads at runtime, and a descriptor correct in source but mis-staged is
/// exactly the failure a source-tree check cannot see.
namespace hipdnn_plugin_sdk::ingestor
{
namespace
{

namespace detail
{
/// Which MetadataValue alternatives are lists. Spelled as a trait rather than detected from
/// streamability, which is a C++20 idiom this target does not compile with.
template <typename T>
struct IsList : std::false_type
{
};
template <typename T>
struct IsList<std::vector<T>> : std::true_type
{
};
} // namespace detail

std::filesystem::path shippedDescriptorRoot()
{
    // HIPDNN_TEST_DESCRIPTOR_DIR already ends in HIPDNN_DESCRIPTOR_SUBDIR -- it is the staged
    // root the provider itself loads from, not the plugin directory above it.
    return {HIPDNN_TEST_DESCRIPTOR_DIR};
}

/// A kernel's completed metadata tuple: every schema field, defaults applied, in schema order.
///
/// Every field, not just the UED's advertised knobs. The knobs are the subset a caller may set;
/// the tuple that identifies a kernel is the whole schema. Keying on knobs alone reports the
/// shipped f32 and f16 kernels as duplicates, when what distinguishes them is `dtype` -- a
/// schema field the engine does not expose as a knob.
///
/// Order comes from the schema rather than from the kernel's own map so two kernels are compared
/// on the same axes in the same sequence -- map iteration order would make the comparison depend
/// on field names.
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
                        // A list-valued knob, rendered element-wise so two kernels differing
                        // only inside the list still produce different tuples.
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
            // A kernel that omits an optional field takes the default, so that is the value it
            // is actually resolved by.
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

/// Holder for the shared load. Not a fixture: every case below is a TEST, and the sets are
/// read-only once parsed.
class ShippedSets
{
public:
    /// Loaded once: the sets are immutable and parsing the tree per case buys nothing.
    static const std::vector<DescriptorSet>& sets()
    {
        static const std::vector<DescriptorSet> s_loaded
            = resolveDescriptorSets(loadDescriptorCatalog(shippedDescriptorRoot()));
        return s_loaded;
    }
};

TEST(TestShippedDescriptorSets, TheStagedTreeContainsDescriptorsAtAll)
{
    // Guards every other case here. An empty tree makes them all vacuously pass, and an empty
    // tree is a real and recurring state: reconfiguring the build wipes the staged content, and
    // engines then decline everything with no error anywhere.
    ASSERT_TRUE(std::filesystem::exists(shippedDescriptorRoot()))
        << "no staged descriptor tree at " << shippedDescriptorRoot()
        << ". Build hkp_descriptor_staging first.";
    EXPECT_FALSE(ShippedSets::sets().empty())
        << "the staged tree parsed to zero engines";
}

TEST(TestShippedDescriptorSets, EveryEngineResolvesItsMetadataSchema)
{
    // RFC 0019 §4: a UED names its KMD by id. An unresolvable one leaves the schema empty, and
    // an engine with no schema cannot type-check the knobs it advertises.
    for(const auto& set : ShippedSets::sets())
    {
        EXPECT_FALSE(set.schema.fields.empty())
            << "engine '" << set.engine.name << "' resolved no metadata schema";
    }
}

TEST(TestShippedDescriptorSets, EveryAdvertisedKnobIsDeclaredInTheSchema)
{
    // The knob names in a UED are what a caller queries and what autotune sweeps. A knob with
    // no schema field has no type and no default, so it reads back as absent -- the query
    // succeeds and returns nothing, rather than failing.
    for(const auto& set : ShippedSets::sets())
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

TEST(TestShippedDescriptorSets, EveryHeuristicReferenceResolves)
{
    // RFC 0019 §3.1: the engine owns the UHD that ranks it. A dangling reference degrades to
    // declared order, which is a legal ranking -- so the engine keeps working and simply stops
    // using the model it shipped. Nothing downstream can tell those two apart.
    for(const auto& set : ShippedSets::sets())
    {
        if(!set.engine.heuristicId.has_value())
        {
            continue; // shipping no model is a legitimate choice; §5 step 7 covers it
        }

        EXPECT_TRUE(set.heuristic.has_value() || !set.heuristicsByArch.empty())
            << "engine '" << set.engine.name << "' names a heuristic that did not resolve";
    }
}

TEST(TestShippedDescriptorSets, EveryHeuristicDeclaresSomethingToScoreWith)
{
    // A UHD is either a native symbol or an artifact on disk. Neither being present is a
    // descriptor that parses, resolves, and scores nothing.
    for(const auto& set : ShippedSets::sets())
    {
        std::vector<const HeuristicDescriptor*> all;
        if(set.heuristic.has_value())
        {
            all.push_back(&*set.heuristic);
        }
        for(const auto& [arch, heuristic] : set.heuristicsByArch)
        {
            all.push_back(&heuristic);
        }

        for(const auto* heuristic : all)
        {
            // What "something to score with" means is adapter-specific, because the
            // descriptor IS the UHD: there is no single `payload` field that every kind
            // fills in. Checking the wrong one per adapter would pass vacuously.
            switch(heuristic->adapter)
            {
            case UhdAdapter::STATIC_ORDER:
                // Ranks by declared fields alone -- no symbol, no artifact. The fields are
                // optional (empty means priority then id), so there is nothing to require.
                break;

            case UhdAdapter::NATIVE:
                EXPECT_FALSE(heuristic->nativeSymbol.empty())
                    << "engine '" << set.engine.name << "' ships heuristic '" << heuristic->name
                    << "' with an empty native symbol";
                // The symbol is resolved at registration, not on disk.
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

            // -Wswitch-default. The enum is closed and every member is handled above.
            default:
                ADD_FAILURE() << "engine '" << set.engine.name << "' ships heuristic '"
                              << heuristic->name << "' naming an adapter this test does not know";
                break;
            }
        }
    }
}

TEST(TestShippedDescriptorSets, NoTwoKernelsOfAPackShareAMetadataTuple)
{
    // The completed metadata tuple is the catalog key -- it is how a plan resolves a kernel
    // once matchers have chosen the pack. Two kernels answering to the same tuple make the
    // choice between them an accident of catalog order, and a UHD ranking them cannot express
    // a preference it has no feature to express.
    //
    // Scoped to the pack, which is the collision domain: two packs of one engine are selected
    // by different matchers, so the same tuple in each is answering a different graph.
    for(const auto& set : ShippedSets::sets())
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
                EXPECT_TRUE(inserted)
                    << "engine '" << set.engine.name << "' pack '" << pack.name << "': kernels '"
                    << entry->second << "' and '" << kernel.name << "' share metadata tuple "
                    << tuple;
            }
        }
    }
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor
