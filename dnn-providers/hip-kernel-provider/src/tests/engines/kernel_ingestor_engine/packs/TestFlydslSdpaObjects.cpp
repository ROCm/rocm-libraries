// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRdnaTestGraphs.hpp"

/**
 * @file TestFlydslSdpaObjects.cpp
 * @brief Every FlyDSL SDPA object this device's shard carries, launched by name.
 *
 * Each SDPA object is its own compiled kernel: dtype, head dim (or its bucket), V's head
 * dim, the causal variant, the bias and its dtype, the sink, the tile, kv_align and
 * any_diagonal are baked into it. The dispatch suites launch the object a case's metadata
 * search (RDNA) or the kernel matcher (CDNA) picks for a handful of shapes, so most of the
 * shipped objects -- the top-ranked default for bf16 d128 non-causal prefill among them --
 * were never run, and an object whose bytes are wrong passes every check short of a launch.
 *
 * So each staged object gets a graph built from its own metadata -- a head dim it serves
 * (its own, or for a bucketed object its bound and a width inside it), V's head dim, a
 * causal mask at a bottom-right diagonal with fewer queries than keys (or, for an object
 * built for offset 0 only, a square top-left one), its bias in the dtype it reads, its
 * sink, a key length its kv_align admits and one that leaves a ragged last tile otherwise,
 * a decode row count its tile holds and a key length long enough to split -- and is
 * launched by name through its own engine's dispatch handler, against the dispatch suites'
 * double-precision reference (LSE included). The object's own kernel matcher has to admit
 * the graph first: an object that admits no graph built from its metadata would never be
 * selected, which a launch by name would otherwise hide.
 *
 * Two tiers. `OnePerForm` runs one object per (engine, form) -- a form being what changes
 * the code an object runs beyond its head dim -- and is the pre-merge case. `Full/...Every`
 * runs all of them, bucketed objects at two widths; it is the nightly one.
 *
 * `TestFlydslSdpaObjectGraphs` is the host half: for every object every loaded shard
 * carries, on every architecture, the graph this suite would run is one the object's
 * matchers admit. It needs no device, so the derivation for CDNA and gfx12 objects is
 * checked on any machine.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

// Restated rather than shared with the packs (as TestFlydslSdpaCdnaEngine.cpp does), so a
// rename on either side fails here.
constexpr PackSymbols FLYDSL_SDPA_CDNA_GENERIC{"hipkernel:flydsl_sdpa_cdna",
                                               "hipkernel.flydsl_sdpa_cdna.graph_match",
                                               "",
                                               "hipkernel.flydsl_sdpa_cdna.kernel_match",
                                               "hipkernel.flydsl_sdpa_cdna.score",
                                               "hipkernel.flydsl_sdpa_cdna.dispatch",
                                               "flydsl_sdpa_cdna.q.uid",
                                               "flydsl_sdpa_cdna.k.uid",
                                               "flydsl_sdpa_cdna.o.uid"};
constexpr PackSymbols FLYDSL_SDPA_CDNA_DUALWAVE_ENGINE{
    "hipkernel:flydsl_sdpa_cdna_dualwave",
    "hipkernel.flydsl_sdpa_cdna_dualwave.graph_match",
    "",
    "hipkernel.flydsl_sdpa_cdna_dualwave.kernel_match",
    "hipkernel.flydsl_sdpa_cdna_dualwave.score",
    "hipkernel.flydsl_sdpa_cdna_dualwave.dispatch",
    "flydsl_sdpa_cdna.q.uid",
    "flydsl_sdpa_cdna.k.uid",
    "flydsl_sdpa_cdna.o.uid"};
constexpr PackSymbols FLYDSL_SDPA_CDNA_DECODE_ENGINE{
    "hipkernel:flydsl_sdpa_cdna_decode",
    "hipkernel.flydsl_sdpa_cdna_decode.graph_match",
    "",
    "hipkernel.flydsl_sdpa_cdna_decode.kernel_match",
    "hipkernel.flydsl_sdpa_cdna_decode.score",
    "hipkernel.flydsl_sdpa_cdna_decode.dispatch",
    "flydsl_sdpa_cdna.q.uid",
    "flydsl_sdpa_cdna.k.uid",
    "flydsl_sdpa_cdna.o.uid"};

/// The six FlyDSL SDPA engines, each a kernel family of its own.
enum class ObjectKind
{
    RDNA_PREFILL,
    RDNA_DECODE,
    AIW,
    CDNA,
    CDNA_DUALWAVE,
    CDNA_DECODE,
};

struct ObjectEngine
{
    ObjectKind kind;
    /// The op as CMake ships it (HIPKERNELPROVIDER_FLYDSL_ARCHS_<op>).
    const char* op;
    /// The engine whose kernel matcher, score and dispatch the objects use.
    const PackSymbols* engine;
    /// The engine whose graph matcher binds the graph: the RDNA engines register the
    /// shared matcher under their own symbol; the CDNA ones share the generic engine's.
    const PackSymbols* graphEngine;
};

constexpr std::array<ObjectEngine, 6> OBJECT_ENGINES{{
    {ObjectKind::RDNA_PREFILL, "sdpa_rdna", &FLYDSL_SDPA_RDNA, &FLYDSL_SDPA_RDNA},
    {ObjectKind::RDNA_DECODE,
     "sdpa_rdna_decode",
     &FLYDSL_SDPA_RDNA_DECODE,
     &FLYDSL_SDPA_RDNA_DECODE},
    {ObjectKind::AIW, "sdpa_aiw", &FLYDSL_SDPA_AIW, &FLYDSL_SDPA_AIW},
    {ObjectKind::CDNA, "sdpa_cdna", &FLYDSL_SDPA_CDNA_GENERIC, &FLYDSL_SDPA_CDNA_GENERIC},
    {ObjectKind::CDNA_DUALWAVE,
     "sdpa_cdna_dualwave",
     &FLYDSL_SDPA_CDNA_DUALWAVE_ENGINE,
     &FLYDSL_SDPA_CDNA_GENERIC},
    {ObjectKind::CDNA_DECODE,
     "sdpa_cdna_decode",
     &FLYDSL_SDPA_CDNA_DECODE_ENGINE,
     &FLYDSL_SDPA_CDNA_GENERIC},
}};

const ObjectEngine& objectEngine(ObjectKind kind)
{
    return *std::find_if(OBJECT_ENGINES.begin(), OBJECT_ENGINES.end(), [kind](const auto& entry) {
        return entry.kind == kind;
    });
}

bool isCdna(ObjectKind kind)
{
    return kind == ObjectKind::CDNA || kind == ObjectKind::CDNA_DUALWAVE
           || kind == ObjectKind::CDNA_DECODE;
}

/// The RDNA harness's name for an RDNA engine.
RdnaObjectEngine rdnaObjectEngine(ObjectKind kind)
{
    if(kind == ObjectKind::AIW)
    {
        return RdnaObjectEngine::AIW;
    }
    return kind == ObjectKind::RDNA_DECODE ? RdnaObjectEngine::DECODE : RdnaObjectEngine::PREFILL;
}

/// Which of the staged objects a case launches.
enum class ObjectSelection
{
    ONE_PER_FORM,
    EVERY
};

/// A metadata field of type @p T, or nullptr when absent or of another type.
template <typename T>
const T* metadataValue(const KernelDefinition& kernel, const char* field)
{
    const auto it = kernel.metadata.find(std::string(field));
    return it == kernel.metadata.end() ? nullptr : std::get_if<T>(&it->second);
}

/// A metadata integer, or @p absent when the object's KMD has no such field (each CDNA
/// family leaves out the fields its kernel fixes).
int64_t intField(const KernelDefinition& kernel, const char* field, int64_t absent = 0)
{
    const auto* value = metadataValue<int64_t>(kernel, field);
    return value == nullptr ? absent : *value;
}

std::string stringField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = metadataValue<std::string>(kernel, field);
    return value == nullptr ? std::string() : *value;
}

/// The metadata fields a graph is derived from. The RDNA engines spell the generic head dim
/// 0 and bound it by head_dim_max; the CDNA ones bucket by qk_mask.
struct ObjectTraits
{
    std::string dtype;
    int64_t headDim = 0;
    int64_t headDimMax = 0;
    int64_t headDimV = 0;
    bool causal = false;
    bool bias = false;
    /// The bias in the query's dtype (RDNA `bias_q`; every CDNA bias) rather than f32.
    bool biasInQueryType = false;
    bool sink = false;
    /// The object drops attention probabilities (aiw `has_dropout`).
    bool dropout = false;
    bool qkMask = false;
    int64_t kvAlign = 0;
    bool anyDiagonal = true;
    int64_t hdPair = 0;
    int64_t dvSplit = 1;
};

ObjectTraits traitsOf(ObjectKind kind, const KernelDefinition& kernel)
{
    ObjectTraits traits;
    traits.dtype = stringField(kernel, "dtype");
    traits.headDim = intField(kernel, "head_dim");
    traits.headDimMax = intField(kernel, "head_dim_max", traits.headDim);
    traits.headDimV = intField(kernel, "head_dim_v");
    traits.causal = intField(kernel, "causal") == 1;
    traits.bias = intField(kernel, "has_bias") == 1;
    traits.sink = intField(kernel, "has_sink") == 1;
    traits.dropout = intField(kernel, "has_dropout") == 1;
    traits.qkMask = intField(kernel, "qk_mask") == 1;
    traits.kvAlign = intField(kernel, "kv_align");
    traits.anyDiagonal = intField(kernel, "any_diagonal", 1) == 1;
    traits.hdPair = intField(kernel, "hd_pair");
    traits.dvSplit = intField(kernel, "dv_split", 1);
    if(kind == ObjectKind::AIW)
    {
        traits.biasInQueryType = intField(kernel, "bias_f32") == 0;
    }
    else if(isCdna(kind))
    {
        // The strided bias objects (vendored-kernel modification 31) are the CDNA ones that
        // read an f32 bias; they run one, the others a bias in the query's dtype.
        traits.biasInQueryType = intField(kernel, "bias_kstrided") == 0;
    }
    else
    {
        traits.biasInQueryType = intField(kernel, "bias_q") == 1;
    }
    return traits;
}

/// Whether the object serves a range of head dims rather than one: an RDNA or aiw generic
/// tier (head_dim 0), or a CDNA qk_mask bucket.
bool isBucketed(ObjectKind kind, const ObjectTraits& traits)
{
    return isCdna(kind) ? traits.qkMask : traits.headDim == 0;
}

/// What changes the code an object runs beyond its head dim: one object of each is the
/// pre-merge tier.
using ObjectForm
    = std::tuple<std::string, bool, bool, bool, bool, bool, bool, int64_t, bool, int64_t, int64_t>;

ObjectForm formOf(ObjectKind kind, const KernelDefinition& kernel)
{
    const auto traits = traitsOf(kind, kernel);
    const bool mla = traits.headDimV != 0 && traits.headDimV != traits.headDim;
    return {traits.dtype,
            traits.causal,
            traits.bias,
            traits.biasInQueryType,
            traits.sink,
            isBucketed(kind, traits),
            mla,
            traits.kvAlign,
            traits.anyDiagonal,
            traits.hdPair,
            traits.dvSplit};
}

/// A head dim inside a bucket of @p bound: about five eighths of it, on the multiple of 8
/// every bucketed object needs; or, for an object whose V has a width of its own
/// (@p fixedValueHeadDim, 0 when V follows Q/K), at least 64 past it, since V is never
/// wider than Q/K (576 for a 768 bucket over V 512, the MLA shape).
int64_t widthInside(int64_t bound, int64_t fixedValueHeadDim)
{
    const int64_t width = std::max<int64_t>(bound * 5 / 8 / 8 * 8, fixedValueHeadDim + 64);
    return std::max<int64_t>(8, std::min(width, bound));
}

/// One shape an object runs, as alternatives: the first one the object's matchers admit is
/// the one launched. Alternatives differ only in mask geometry (a bottom-right diagonal
/// with fewer queries than keys first, a square top-left one for an object built for
/// diagonal offset 0 only).
using ShapeAlternatives = std::vector<SdpaGraphSpec>;

/// The shapes @p kernel runs: its head dim, or for a bucketed object its bound and (under
/// EVERY) a width inside it.
std::vector<ShapeAlternatives>
    objectShapes(ObjectKind kind, const KernelDefinition& kernel, ObjectSelection selection)
{
    const auto traits = traitsOf(kind, kernel);
    const bool bucketed = isBucketed(kind, traits);
    const int64_t bound = isCdna(kind) ? traits.headDim : traits.headDimMax;
    // V's head dim: the object's own when it differs from Q/K's (MLA). A bucketed object
    // whose V is as wide as its bucket follows Q/K's runtime width instead.
    const bool vFollows = traits.headDimV == 0 || traits.headDimV == traits.headDim;
    const int64_t inside = widthInside(bound, vFollows ? 0 : traits.headDimV);
    std::vector<int64_t> widths{bucketed ? bound : traits.headDim};
    if(bucketed && selection == ObjectSelection::EVERY && inside != bound)
    {
        widths.push_back(inside);
    }

    std::vector<ShapeAlternatives> shapes;
    for(const auto width : widths)
    {
        SdpaGraphSpec spec;
        spec.dataType = traits.dtype == "f16" ? DataType::HALF : DataType::BFLOAT16;
        if(traits.dtype == "f32")
        {
            // The CDNA generic objects' exact f32 (vendored-kernel modification 35).
            spec.dataType = DataType::FLOAT;
        }
        spec.headDim = width;
        spec.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(width)));
        spec.stats = SdpaStats::LSE;

        if(!vFollows)
        {
            spec.valueHeadDim = traits.headDimV;
        }

        spec.withSink = traits.sink;
        spec.withBias = traits.bias;
        if(traits.dropout)
        {
            setDropout(spec, 0.2F);
        }
        if(traits.bias)
        {
            spec.biasDataType = traits.biasInQueryType ? spec.dataType : DataType::FLOAT;
            if(kind == ObjectKind::AIW && !traits.biasInQueryType)
            {
                // aiw loads an f32 mask 8 x f32 at a time at that type's natural alignment.
                spec.alignment[FLYDSL_SDPA_RDNA_BIAS_UID] = 32;
            }
        }

        // Sizes: grouped query heads, a ragged last query tile, and a key length that leaves
        // a ragged last KV tile unless the object is built for an aligned one. A bias's rows
        // stay 16-byte aligned (a multiple of 8 keys).
        switch(kind)
        {
        case ObjectKind::RDNA_PREFILL:
        case ObjectKind::AIW:
        default:
            spec.batch = 2;
            spec.heads = 4;
            spec.kvHeads = 2;
            spec.seqLenQ = 150;
            spec.seqLenKv = 200;
            break;
        case ObjectKind::RDNA_DECODE:
            // A few query positions of a GQA group: 20 rows (two 16-row tiles) for a
            // row-tiled object, the 16 one tile holds for a bias object; and enough keys
            // that the launch splits them.
            spec.batch = 1;
            spec.heads = 8;
            spec.kvHeads = 2;
            spec.seqLenQ = traits.bias ? 4 : 5;
            spec.seqLenKv = 2085;
            break;
        case ObjectKind::CDNA:
            spec.batch = 2;
            spec.heads = 4;
            spec.kvHeads = 2;
            spec.seqLenQ = 200;
            spec.seqLenKv = traits.bias ? 328 : 300;
            if(traits.kvAlign > 0)
            {
                spec.seqLenKv = 5 * traits.kvAlign;
            }
            break;
        case ObjectKind::CDNA_DUALWAVE:
            spec.batch = 1;
            spec.heads = 4;
            spec.kvHeads = 2;
            spec.seqLenQ = 300;
            spec.seqLenKv = traits.bias ? 328 : 333;
            break;
        case ObjectKind::CDNA_DECODE:
            spec.batch = 1;
            spec.heads = 8;
            spec.kvHeads = 2;
            spec.seqLenQ = 4;
            spec.seqLenKv = 3000;
            break;
        }

        ShapeAlternatives alternatives;
        if(traits.causal)
        {
            spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
            alternatives.push_back(spec);
            // Offset 0: square, top-left.
            spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
            spec.seqLenQ = spec.seqLenKv;
            alternatives.push_back(spec);
        }
        else
        {
            alternatives.push_back(spec);
        }
        shapes.push_back(std::move(alternatives));
    }
    return shapes;
}

/// The device a host check stands in for when it matches an object of another arch.
DeviceProperties devicePropertiesFor(const std::string& arch)
{
    DeviceProperties properties;
    properties.gcnArchName = arch;
    properties.warpSize = arch.rfind("gfx9", 0) == 0 ? 64 : 32;
    properties.multiProcessorCount = 64;
    return properties;
}

/// Whether @p kernel's engine binds the graph @p spec describes and the object admits it.
bool objectAdmits(const ObjectEngine& engine,
                  const SdpaGraphSpec& spec,
                  const KernelDefinition& kernel,
                  const DeviceProperties& properties)
{
    const GraphFixture fixture(buildFlydslSdpaRdnaGraph(spec), properties);
    const auto bound = matchesGraph(*engine.graphEngine, fixture.context());
    return bound.has_value() && matchesKernel(*engine.engine, fixture.context(), kernel, *bound);
}

/// The first of @p alternatives the object admits, if any.
std::optional<SdpaGraphSpec> admittedShape(const ObjectEngine& engine,
                                           const ShapeAlternatives& alternatives,
                                           const KernelDefinition& kernel,
                                           const DeviceProperties& properties)
{
    for(const auto& spec : alternatives)
    {
        if(objectAdmits(engine, spec, kernel, properties))
        {
            return spec;
        }
    }
    return std::nullopt;
}

std::string describe(const SdpaGraphSpec& spec)
{
    std::string mask = "none";
    if(spec.mask == SdpaMask::BOTTOM_RIGHT_CAUSAL)
    {
        mask = "bottom-right causal";
    }
    else if(spec.mask == SdpaMask::TOP_LEFT_CAUSAL)
    {
        mask = "top-left causal";
    }
    return "B=" + std::to_string(spec.batch) + " Hq=" + std::to_string(spec.heads)
           + " Hk=" + std::to_string(spec.kvHeads) + " Sq=" + std::to_string(spec.seqLenQ)
           + " Skv=" + std::to_string(spec.seqLenKv) + " D=" + std::to_string(spec.headDim)
           + " Dv=" + std::to_string(spec.valueHeadDim.value_or(spec.headDim)) + " mask=" + mask
           + (spec.withBias ? " bias" : "") + (spec.withSink ? " sink" : "")
           + (spec.dropoutProbability.value_or(0.0F) > 0.0F ? " dropout" : "");
}

/// @p kernels in name order, or one per form when @p selection asks for that: the first of
/// each in name order, so the choice is the same on every run.
std::vector<KernelDefinition>
    select(ObjectKind kind, std::vector<KernelDefinition> kernels, ObjectSelection selection)
{
    std::sort(kernels.begin(), kernels.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });
    if(selection == ObjectSelection::EVERY)
    {
        return kernels;
    }
    std::vector<KernelDefinition> chosen;
    std::vector<ObjectForm> seen;
    for(auto& kernel : kernels)
    {
        auto form = formOf(kind, kernel);
        if(std::find(seen.begin(), seen.end(), form) == seen.end())
        {
            seen.push_back(std::move(form));
            chosen.push_back(std::move(kernel));
        }
    }
    return chosen;
}

/// Launches each selected staged object of @p kind on each of its shapes.
void expectStagedObjectsMatch(ObjectKind kind, ObjectSelection selection)
{
    const auto& engine = objectEngine(kind);
    const auto staged = stagedKernels(engine.engine->engineName);
    if(staged.empty())
    {
        reportUnstaged(engine.op,
                       std::string("this build staged no ") + engine.op
                           + " objects for this device");
        return;
    }

    const auto device = packedArchDeviceProperties();
    const auto kernels = select(kind, staged, selection);
    int launches = 0;
    for(const auto& kernel : kernels)
    {
        for(const auto& alternatives : objectShapes(kind, kernel, selection))
        {
            const auto spec = admittedShape(engine, alternatives, kernel, device);
            if(!spec.has_value())
            {
                ADD_FAILURE() << kernel.name << " admits no graph built from its metadata ("
                              << describe(alternatives.front()) << ")";
                continue;
            }
            SCOPED_TRACE(kernel.name + ": " + describe(*spec));
            if(isCdna(kind))
            {
                expectCdnaObjectMatchesReference(*spec, kernel);
            }
            else
            {
                expectRdnaObjectMatchesReference(*spec, rdnaObjectEngine(kind), kernel);
            }
            ++launches;
        }
    }
    std::cout << "[ objects  ] " << engine.op << " on " << device.gcnArchName << ": "
              << kernels.size() << " of " << staged.size() << " staged objects, " << launches
              << " launches\n";
    ::testing::Test::RecordProperty("objects", static_cast<int>(kernels.size()));
    ::testing::Test::RecordProperty("staged", static_cast<int>(staged.size()));
}

// ---------------------------------------------------------------------------
// host: the graph schema the engines claim
// ---------------------------------------------------------------------------

// Every FlyDSL SDPA engine's UED claims the graph schema its matchers were last reviewed
// against: each field the schema added up to it is gated where it is read, or by
// usesUnreadSchemaFields (FlydslSdpaShared.hpp; COVERAGE.md "Graph schema"). The claim is the
// newest version this SDK defines, so a ragged graph reaches the matchers; an SDK that adds a
// version fails the first check until its fields are reviewed and both values move together
// (gen_descriptors.py SDPA_SDK_VERSION).
TEST(TestFlydslSdpaGraphSchema, EveryEngineClaimsTheReviewedSchema)
{
    const hipdnn_data_sdk::utilities::Version reviewed{std::string("1.4.0")};
    EXPECT_EQ(
        hipdnn_data_sdk::utilities::Version{
            std::string(hipdnn_plugin_sdk::K_MAX_SUPPORTED_API_VERSION)}
            .str(),
        reviewed.str())
        << "the SDK defines a graph schema the FlyDSL SDPA matchers were not reviewed against";
    // A packed graph with every 1.x feature: overridable shapes aside, which the matchers
    // decline themselves, it is within the claim.
    const auto& packedFloor
        = hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(false, true, true, true, true);
    for(const auto op : {"sdpa_rdna",
                         "sdpa_rdna_decode",
                         "sdpa_aiw",
                         "sdpa_cdna",
                         "sdpa_cdna_dualwave",
                         "sdpa_cdna_decode"})
    {
        const auto* staged = findLoadedSet(std::string("hipkernel:flydsl_") + op);
        if(staged == nullptr)
        {
            EXPECT_TRUE(flydslShippedArches(op).empty())
                << "this build ships " << op << " but loaded no objects for it";
            continue;
        }
        EXPECT_EQ(staged->engine.sdkVersion.str(), reviewed.str()) << op;
        EXPECT_FALSE(staged->engine.sdkVersion < packedFloor) << op;
    }
}

// ---------------------------------------------------------------------------
// host: every loaded object, on every architecture, admits the graph built for it
// ---------------------------------------------------------------------------

class TestFlydslSdpaObjectGraphs : public ::testing::TestWithParam<ObjectKind>
{
};

TEST_P(TestFlydslSdpaObjectGraphs, EveryLoadedObjectAdmitsTheGraphsBuiltFromItsMetadata)
{
    const auto& engine = objectEngine(GetParam());
    const auto* staged = findLoadedSet(engine.engine->engineName);
    if(staged == nullptr)
    {
        if(flydslShipsAnywhere(engine.op))
        {
            ADD_FAILURE() << "this build ships " << engine.op << " but loaded no objects for it";
            return;
        }
        GTEST_SKIP() << "this build ships no " << engine.op << " objects";
    }

    std::map<std::string, std::vector<KernelDefinition>> objectsPerArch;
    for(const auto& pack : staged->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const KernelDefinition definition{kernel.id,
                                              pack.id,
                                              pack.dispatchId,
                                              kernel.source,
                                              kernel.metadata,
                                              kernel.priority,
                                              kernel.arch.empty() ? pack.arch : kernel.arch,
                                              kernel.originDirectory,
                                              kernel.name,
                                              kernel.treeRoot,
                                              kernel.sources};
            for(const auto& arch : definition.arch)
            {
                objectsPerArch[arch].push_back(definition);
                for(const auto& alternatives :
                    objectShapes(GetParam(), definition, ObjectSelection::EVERY))
                {
                    EXPECT_TRUE(
                        admittedShape(engine, alternatives, definition, devicePropertiesFor(arch))
                            .has_value())
                        << definition.name << " on " << arch
                        << " admits no graph built from its metadata ("
                        << describe(alternatives.front()) << ")";
                }
            }
        }
    }
    for(const auto& [arch, kernels] : objectsPerArch)
    {
        std::cout << "[ objects  ] " << engine.op << " " << arch << ": " << kernels.size()
                  << " objects, "
                  << select(GetParam(), kernels, ObjectSelection::ONE_PER_FORM).size()
                  << " forms\n";
    }
}

std::string kindName(const ::testing::TestParamInfo<ObjectKind>& info)
{
    std::string name = objectEngine(info.param).op;
    name.erase(std::remove(name.begin(), name.end(), '_'), name.end());
    return name;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestFlydslSdpaObjectGraphs,
                         ::testing::Values(ObjectKind::RDNA_PREFILL,
                                           ObjectKind::RDNA_DECODE,
                                           ObjectKind::AIW,
                                           ObjectKind::CDNA,
                                           ObjectKind::CDNA_DUALWAVE,
                                           ObjectKind::CDNA_DECODE),
                         kindName);

// ---------------------------------------------------------------------------
// device: each staged object, launched by name
// ---------------------------------------------------------------------------

class TestGpuFlydslSdpaObjects : public ::testing::TestWithParam<ObjectSelection>
{
};

TEST_P(TestGpuFlydslSdpaObjects, RdnaPrefillObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectKind::RDNA_PREFILL, GetParam());
}

TEST_P(TestGpuFlydslSdpaObjects, RdnaDecodeObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectKind::RDNA_DECODE, GetParam());
}

TEST_P(TestGpuFlydslSdpaObjects, AiwObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectKind::AIW, GetParam());
}

TEST_P(TestGpuFlydslSdpaObjects, CdnaObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectKind::CDNA, GetParam());
}

TEST_P(TestGpuFlydslSdpaObjects, CdnaDualwaveObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectKind::CDNA_DUALWAVE, GetParam());
}

TEST_P(TestGpuFlydslSdpaObjects, CdnaDecodeObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectKind::CDNA_DECODE, GetParam());
}

std::string selectionName(const ::testing::TestParamInfo<ObjectSelection>& info)
{
    return info.param == ObjectSelection::EVERY ? "Every" : "OnePerForm";
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuFlydslSdpaObjects,
                         ::testing::Values(ObjectSelection::ONE_PER_FORM),
                         selectionName);

INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuFlydslSdpaObjects,
                         ::testing::Values(ObjectSelection::EVERY),
                         selectionName);

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
