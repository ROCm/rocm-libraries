// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdDeviceHarness.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdVerdictTable.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdObjectsPerObject.cpp
 * @brief Every object of the split SDPA backward engine, launched on its own.
 *
 * The split engine runs three kernels per graph, one object from each of the slots prep, dkdv
 * and dq (and, for a graph whose dBias is broadcast, the dBias reduce kernel its dq object
 * names). Which object a slot runs is the engine's ranking, so the dispatch suites reach the
 * few objects the default picks name and the rest only through knobs; an object whose bytes
 * are wrong passes every check short of a launch.
 *
 * So each staged object gets a graph built from its own metadata -- its dtype, its head dim
 * (the tile of an exact object; the lowest dim above its floor, and under EVERY its tile, for a
 * padded one), the causal class as a bottom-right mask over fewer queries than keys, the
 * bias and, where the dq object stores it, dBias, the baked dropout, and a grouped query for a
 * packed dkdv object -- and is launched on its own: the object is pinned in its slot through the
 * knob settings the engine enumerates for the graph (the selection hipdnn_bench --sweep uses; the
 * slot's knob tuples are unique per graph), the other two slots keep the engine's default
 * for the graph, and the plan is checked to have built the pinned object before dQ, dK, dV and
 * dBias are compared with the harness's double-precision reference. Each object runs once per
 * shape; there is no cross product of slots.
 *
 * prep's sink twin is the one object left out: the engine's kernel matcher never admits it, since
 * it takes two arguments no graph supplies.
 *
 * The dBias reduce companions are the dq objects that name a reduce kernel, launched on a bias
 * whose gradient is summed over batch and heads. A multi-launch object (gfx11) is checked to
 * carry the companion it was enrolled with.
 *
 * Two tiers. `OnePerForm` runs one object per (slot, form) -- a form being what changes the
 * code an object runs beyond its head dim -- and is the pre-merge case. `Full/...Every` runs all
 * of them, padded objects at two widths; it is the nightly one.
 *
 * `TestFlydslSdpaBwdObjectGraphs` is the host half: for every object every loaded shard carries,
 * on every architecture it lists, the graph this suite would run is one the engine's graph
 * matcher binds and the object's kernel matcher admits. It needs no device, so the derivation
 * for gfx950 objects is checked on a gfx11 machine and the reverse.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device;

constexpr const char* SPLIT_ENGINE_NAME = "hipkernel:flydsl_sdpa_bwd_split";
constexpr const char* REDUCE_LAUNCH_NAME = "dbias_reduce";

/// The groups of objects launched on their own. The dBias reduce companions are dq objects,
/// launched on a graph whose dBias needs them.
enum class ObjectSlot
{
    PREP,
    DKDV,
    DQ,
    DBIAS_REDUCE
};

struct SlotTraits
{
    ObjectSlot slot;
    /// The engine slot whose catalog the objects are in.
    FlydslSdpaBwdSplitSlot engineSlot;
    /// The op as CMake ships it (HIPKERNELPROVIDER_FLYDSL_ARCHS_<op>).
    const char* op;
    /// The test name's stem.
    const char* label;
};

constexpr std::array<SlotTraits, 4> SLOTS{{
    {ObjectSlot::PREP, FlydslSdpaBwdSplitSlot::PREP, "sdpa_bwd_prep", "prep"},
    {ObjectSlot::DKDV, FlydslSdpaBwdSplitSlot::DKDV, "sdpa_bwd_dkdv", "dkdv"},
    {ObjectSlot::DQ, FlydslSdpaBwdSplitSlot::DQ, "sdpa_bwd_dq", "dq"},
    {ObjectSlot::DBIAS_REDUCE, FlydslSdpaBwdSplitSlot::DQ, "sdpa_bwd_dq", "dbias_reduce"},
}};

const SlotTraits& traitsOfSlot(ObjectSlot slot)
{
    return *std::find_if(
        SLOTS.begin(), SLOTS.end(), [slot](const auto& entry) { return entry.slot == slot; });
}

/// Which of the staged objects a case launches.
enum class ObjectSelection
{
    ONE_PER_FORM,
    EVERY
};

/// Whether @p kernel is one of @p slot's objects: every object of its engine slot, except that
/// the reduce companions are the dq objects that name a reduce kernel, and that prep's sink twin
/// is left out. The twin takes two arguments no SDPA backward graph supplies, so the engine's
/// kernel matcher never admits it (FlydslSdpaBwdSplitNative.cpp), and no graph reaches it.
bool isObjectOf(ObjectSlot slot, const KernelDefinition& kernel)
{
    if(slot == ObjectSlot::PREP)
    {
        return intField(kernel, "sink") != 1;
    }
    return slot != ObjectSlot::DBIAS_REDUCE || !stringField(kernel, "reduce_symbol").empty();
}

DataType dataTypeOf(const std::string& dtype)
{
    if(dtype == "f16")
    {
        return DataType::HALF;
    }
    return dtype == "bf16" ? DataType::BFLOAT16 : DataType::FLOAT;
}

// The head-dim tiles the engine resolves a `tail` floor against, restated rather than shared
// with the engine (FlydslSdpaBwdSplitNative.cpp) so that a change on either side fails here:
// the 16-bit ladder, and the fp32 one, which has no 384.
constexpr std::array<int64_t, 10> LADDER{32, 64, 96, 128, 160, 192, 224, 256, 384, 512};
constexpr std::array<int64_t, 9> FP32_LADDER{32, 64, 96, 128, 160, 192, 224, 256, 512};

/// The floor of an object: `g1` is 0, `tail` the rung of its dtype's ladder below its tile,
/// anything else the number, as an integer or as text.
int64_t floorOf(const KernelDefinition& kernel)
{
    if(const auto* number = metadataValue<int64_t>(kernel, "floor"))
    {
        return *number;
    }
    const auto text = stringField(kernel, "floor");
    if(text == "g1")
    {
        return 0;
    }
    if(text == "tail")
    {
        const auto tile = intField(kernel, "head_dim");
        int64_t below = 0;
        const auto scan = [&](const auto& ladder) {
            for(const int64_t rung : ladder)
            {
                below = rung < tile ? rung : below;
            }
        };
        if(stringField(kernel, "dtype") == "fp32")
        {
            scan(FP32_LADDER);
        }
        else
        {
            scan(LADDER);
        }
        return below;
    }
    return std::stoll(text);
}

/// The graphs an object runs, as cases: its tile, or for a padded object the lowest head dim
/// above its floor and under EVERY also its tile.
std::vector<BwdCase>
    objectCases(ObjectSlot slot, const KernelDefinition& kernel, ObjectSelection selection)
{
    const auto tile = intField(kernel, "head_dim");
    std::vector<int64_t> widths{tile};
    if(intField(kernel, "padded") == 1)
    {
        widths = {floorOf(kernel) + 1};
        if(selection == ObjectSelection::EVERY && widths.front() != tile)
        {
            widths.push_back(tile);
        }
    }

    std::vector<BwdCase> cases;
    for(const auto width : widths)
    {
        BwdCase c;
        c.dtype = dataTypeOf(stringField(kernel, "dtype"));
        c.headDim = width;
        c.batch = 2;
        c.heads = 4;
        c.kvHeads = 4;
        // Fewer queries than keys, neither a whole tile; the keys a multiple of 8 so a bias
        // row is a whole vector.
        c.seqQ = 150;
        c.seqKv = 200;
        if(slot == ObjectSlot::DKDV)
        {
            // A packed object serves a grouped query; an unpacked one serves one too.
            c.heads = intField(kernel, "gqa_pack") == 1 ? 8 : 4;
            c.kvHeads = 2;
        }
        if(slot != ObjectSlot::PREP)
        {
            // prep has no class: it takes whatever graph the other two run.
            c.mask = intField(kernel, "causal") == 1 ? BwdMask::BOTTOM_RIGHT : BwdMask::NONE;
            c.dropout = intField(kernel, "dropout") == 1 ? 0.2F : 0.0F;
            if(intField(kernel, "has_bias") == 1)
            {
                c.bias = true;
                // A dq object built without the dB store serves a bias without a gradient.
                c.dbias = slot != ObjectSlot::DQ || intField(kernel, "store_db") != 0;
            }
        }
        if(slot == ObjectSlot::DBIAS_REDUCE)
        {
            c.biasOverBatch = true;
            c.biasOverHead = true;
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

/// Whether the split engine's graph matcher binds @p c's graph and @p kernel, an object of
/// @p slot, admits it on @p properties.
bool objectAdmits(ObjectSlot slot,
                  const BwdCase& c,
                  const KernelDefinition& kernel,
                  const DeviceProperties& properties)
{
    const GraphFixture fixture(buildSdpaBwdGraph(specOf(c)), properties);
    const auto symbols = splitSymbols(traitsOfSlot(slot).engineSlot);
    const auto bound = matchesGraph(symbols, fixture.context());
    return bound.has_value() && matchesKernel(symbols, fixture.context(), kernel, *bound);
}

/// The bias part of describe(): none, a bias, or a bias with its gradient.
const char* biasLabel(const BwdCase& c)
{
    if(!c.bias)
    {
        return "";
    }
    return c.dbias ? " bias+dbias" : " bias";
}

std::string describe(const BwdCase& c)
{
    return "B=" + std::to_string(c.batch) + " Hq=" + std::to_string(c.heads)
           + " Hk=" + std::to_string(c.kvHeads) + " Sq=" + std::to_string(c.seqQ)
           + " Skv=" + std::to_string(c.seqKv) + " D=" + std::to_string(c.headDim)
           + (c.mask == BwdMask::BOTTOM_RIGHT ? " bottom-right causal" : "") + biasLabel(c)
           + (c.biasOverBatch && c.biasOverHead ? " broadcast" : "")
           + (c.dropout > 0.0F ? " dropout" : "");
}

/// The fields whose values change the code an object runs beyond its head dim.
const std::vector<const char*>& formFields()
{
    static const std::vector<const char*> s_fields{"dtype",
                                                   "causal",
                                                   "has_bias",
                                                   "dropout",
                                                   "padded",
                                                   "hints",
                                                   "dform",
                                                   "gqa_pack",
                                                   "store_db",
                                                   "bias_stage",
                                                   "sink"};
    return s_fields;
}

/// What an object's form is: its values of formFields(), and for a multi-launch object the
/// reduce companion it carries.
std::string formOf(const KernelDefinition& kernel)
{
    std::string form;
    for(const char* field : formFields())
    {
        const auto it = kernel.metadata.find(field);
        form += std::string(field) + "=";
        if(it != kernel.metadata.end())
        {
            if(const auto* number = std::get_if<int64_t>(&it->second))
            {
                form += std::to_string(*number);
            }
            else if(const auto* text = std::get_if<std::string>(&it->second))
            {
                form += *text;
            }
        }
        form += ";";
    }
    form += "reduce=" + stringField(kernel, "reduce_symbol");
    const auto companion = kernel.sources.find(REDUCE_LAUNCH_NAME);
    form += companion == kernel.sources.end() ? "" : ":" + companion->second.sha256;
    return form;
}

/// @p kernels in name order, or one per form when @p selection asks for that: the first of
/// each in name order, so the choice is the same on every run.
std::vector<KernelDefinition> select(std::vector<KernelDefinition> kernels,
                                     ObjectSelection selection)
{
    std::sort(kernels.begin(), kernels.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });
    if(selection == ObjectSelection::EVERY)
    {
        return kernels;
    }
    std::vector<KernelDefinition> chosen;
    std::set<std::string> seen;
    for(auto& kernel : kernels)
    {
        if(seen.insert(formOf(kernel)).second)
        {
            chosen.push_back(std::move(kernel));
        }
    }
    return chosen;
}

/// The objects of @p slot among the engine slot's @p kernels.
std::vector<KernelDefinition> objectsOf(ObjectSlot slot,
                                        const std::vector<KernelDefinition>& kernels)
{
    std::vector<KernelDefinition> objects;
    std::copy_if(kernels.begin(),
                 kernels.end(),
                 std::back_inserter(objects),
                 [slot](const auto& kernel) { return isObjectOf(slot, kernel); });
    return objects;
}

/// The engine slot @p name of the loaded engine, or nullptr.
const SlotDescriptor* engineSlotOf(const DescriptorSet& set, const std::string& name)
{
    const auto found = std::find_if(set.engine.slots.begin(),
                                    set.engine.slots.end(),
                                    [&](const auto& slot) { return slot.name == name; });
    return found == set.engine.slots.end() ? nullptr : &*found;
}

/// What selects @p kernel in its slot: the slot's integer-valued enrolled knobs it carries.
ObjectPin pinOf(ObjectSlot slot, const DescriptorSet& set, const KernelDefinition& kernel)
{
    const std::string engineSlot = slotName(traitsOfSlot(slot).engineSlot);
    ObjectPin pin;
    pin.slot = engineSlot;
    pin.kernelName = kernel.name;
    pin.kernelId = toString(kernel.kernelId);
    if(const auto* declared = engineSlotOf(set, engineSlot))
    {
        for(const auto& knob : declared->knobs)
        {
            if(const auto* value = metadataValue<int64_t>(kernel, knob.c_str()))
            {
                std::string name = engineSlot;
                name.append(".").append(knob);
                pin.narrowing.emplace_back(std::move(name), *value);
            }
        }
    }
    if(slot == ObjectSlot::DBIAS_REDUCE)
    {
        const auto companion = kernel.sources.find(REDUCE_LAUNCH_NAME);
        pin.reduceDigest = companion == kernel.sources.end() ? "" : companion->second.sha256;
    }
    return pin;
}

/// Launches each selected staged object of @p slot on each of its shapes.
void expectStagedObjectsMatch(ObjectSlot slot, ObjectSelection selection)
{
    const auto& traits = traitsOfSlot(slot);
    const std::string engineSlot = slotName(traits.engineSlot);
    const auto slotKernels = stagedSlotKernels(SPLIT_ENGINE_NAME, engineSlot);
    const auto staged = objectsOf(slot, slotKernels);
    if(staged.empty())
    {
        reportUnstaged(traits.op,
                       std::string("this build staged no ") + traits.label
                           + " objects for this device");
        return;
    }

    const auto& set = loadedSet(SPLIT_ENGINE_NAME);
    const auto device = packedArchDeviceProperties();
    const auto kernels = select(staged, selection);
    int launches = 0;
    for(const auto& kernel : kernels)
    {
        for(const auto& c : objectCases(slot, kernel, selection))
        {
            SCOPED_TRACE(kernel.name + ": " + describe(c));
            if(!objectAdmits(slot, c, kernel, device))
            {
                ADD_FAILURE() << kernel.name << " admits no graph built from its metadata";
                continue;
            }
            const auto pin = pinOf(slot, set, kernel);
            expectCase(SPLIT_ENGINE, c, &pin);
            ++launches;
        }
    }
    std::cout << "[ objects  ] " << traits.label << " on " << device.gcnArchName << ": "
              << kernels.size() << " of " << staged.size() << " staged objects, " << launches
              << " launches\n";
    ::testing::Test::RecordProperty("objects", static_cast<int>(kernels.size()));
    ::testing::Test::RecordProperty("staged", static_cast<int>(staged.size()));
    ::testing::Test::RecordProperty("launches", launches);
}

// ---------------------------------------------------------------------------
// host: every loaded object, on every architecture, admits the graph built for it
// ---------------------------------------------------------------------------

class TestFlydslSdpaBwdObjectGraphs : public ::testing::TestWithParam<ObjectSlot>
{
};

TEST_P(TestFlydslSdpaBwdObjectGraphs, EveryLoadedObjectAdmitsTheGraphsBuiltFromItsMetadata)
{
    const auto& traits = traitsOfSlot(GetParam());
    const std::string engineSlot = slotName(traits.engineSlot);
    const auto* loaded = findLoadedSet(SPLIT_ENGINE_NAME);
    if(loaded == nullptr)
    {
        if(flydslShipsAnywhere(traits.op))
        {
            ADD_FAILURE() << "this build ships " << traits.op << " but loaded no objects for it";
            return;
        }
        GTEST_SKIP() << "this build ships no " << traits.op << " objects";
    }
    const auto slot
        = std::find_if(loaded->slots.begin(), loaded->slots.end(), [&](const auto& candidate) {
              return candidate.name == engineSlot;
          });
    ASSERT_NE(slot, loaded->slots.end()) << "the loaded engine has no " << engineSlot << " slot";

    std::vector<KernelDefinition> kernels;
    for(const auto& pack : slot->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            kernels.push_back(definitionOf(pack, kernel));
        }
    }
    const auto objects = objectsOf(GetParam(), kernels);
    if(objects.empty())
    {
        ADD_FAILURE() << "this build ships " << traits.op << " but loaded no " << traits.label
                      << " objects";
        return;
    }

    std::map<std::string, std::vector<KernelDefinition>> objectsPerArch;
    for(const auto& definition : objects)
    {
        EXPECT_FALSE(definition.arch.empty()) << definition.name << " lists no architecture";
        for(const auto& arch : definition.arch)
        {
            objectsPerArch[arch].push_back(definition);
            for(const auto& c : objectCases(GetParam(), definition, ObjectSelection::EVERY))
            {
                EXPECT_TRUE(objectAdmits(GetParam(), c, definition, devicePropertiesFor(arch)))
                    << definition.name << " on " << arch
                    << " admits no graph built from its metadata (" << describe(c) << ")";
            }
        }
    }
    for(const auto& [arch, archObjects] : objectsPerArch)
    {
        const auto forms = select(archObjects, ObjectSelection::ONE_PER_FORM);
        size_t quickLaunches = 0;
        for(const auto& form : forms)
        {
            quickLaunches += objectCases(GetParam(), form, ObjectSelection::ONE_PER_FORM).size();
        }
        size_t fullLaunches = 0;
        for(const auto& object : archObjects)
        {
            fullLaunches += objectCases(GetParam(), object, ObjectSelection::EVERY).size();
        }
        std::cout << "[ objects  ] " << traits.label << " " << arch << ": " << archObjects.size()
                  << " objects, " << forms.size() << " forms, " << quickLaunches
                  << " launches one per form, " << fullLaunches << " launches every object\n";
    }
}

std::string slotParamName(const ::testing::TestParamInfo<ObjectSlot>& info)
{
    std::string name = traitsOfSlot(info.param).label;
    name.erase(std::remove(name.begin(), name.end(), '_'), name.end());
    return name;
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslSdpaBwdObjectGraphs,
    ::testing::Values(ObjectSlot::PREP, ObjectSlot::DKDV, ObjectSlot::DQ, ObjectSlot::DBIAS_REDUCE),
    slotParamName);

// ---------------------------------------------------------------------------
// device: each staged object, launched on its own
// ---------------------------------------------------------------------------

class TestGpuFlydslSdpaBwdObjects : public ::testing::TestWithParam<ObjectSelection>
{
};

TEST_P(TestGpuFlydslSdpaBwdObjects, PrepObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectSlot::PREP, GetParam());
}

TEST_P(TestGpuFlydslSdpaBwdObjects, DkdvObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectSlot::DKDV, GetParam());
}

TEST_P(TestGpuFlydslSdpaBwdObjects, DqObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectSlot::DQ, GetParam());
}

TEST_P(TestGpuFlydslSdpaBwdObjects, DbiasReduceObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    expectStagedObjectsMatch(ObjectSlot::DBIAS_REDUCE, GetParam());
}

std::string selectionName(const ::testing::TestParamInfo<ObjectSelection>& info)
{
    return info.param == ObjectSelection::EVERY ? "Every" : "OnePerForm";
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuFlydslSdpaBwdObjects,
                         ::testing::Values(ObjectSelection::ONE_PER_FORM),
                         selectionName);

INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuFlydslSdpaBwdObjects,
                         ::testing::Values(ObjectSelection::EVERY),
                         selectionName);

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
