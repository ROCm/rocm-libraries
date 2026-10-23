// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestFlydslSdpaBwdEngine.cpp
 * @brief The flyDSL merged SDPA backward pack across its native seams, over a synthetic
 *        descriptor set: no staged shard, no device.
 *
 * The census over what the pack ships is TestFlydslSdpaBwdPacks.cpp. Here a graph is built
 * from the shared matcher's test graphs, run through graph_match, and offered to rows
 * written in the test; a launch is read through describeFlydslSdpaBwdMergedLaunchForTesting,
 * which computes the arguments the dispatch handler's launch() passes without touching a
 * device. Every decline the engine adds over the shared matcher sits beside an accepting
 * control that differs in one field; the matcher's own declines are TestFlydslSdpaBwdMatcher.
 *
 * Kernarg dumps are decoded against the kernel's parameter list (`_PARAMS["merged"]` and
 * `_PARAMS["dbias_reduce"]` in `_sdpa_bwd_cdna_build.py`), and every case states the whole
 * list: a value that moves, or an argument that slides a position, fails it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

using data_objects::DataType;
using flydsl_sdpa_bwd::SdpaBwdProblem;

const PackSymbols MERGED{"hipkernel:flydsl_sdpa_bwd_merged",
                         "hipkernel.flydsl_sdpa_bwd_merged.graph_match",
                         "",
                         "hipkernel.flydsl_sdpa_bwd_merged.kernel_match",
                         "hipkernel.flydsl_sdpa_bwd_merged.score",
                         "hipkernel.flydsl_sdpa_bwd_merged.dispatch",
                         "",
                         "",
                         ""};

const char* const REDUCE_SYMBOL = "fmha_bwd_dbias_reduce_gfx950_kernel_1";

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

/// The dimensions of a graph; the default is the shared matcher's baseline.
struct Shape
{
    int64_t batch = 2;
    int64_t heads = 8;
    int64_t kvHeads = 8;
    int64_t seqQ = 16;
    int64_t seqKv = 16;
    int64_t headDimQk = 64;
    int64_t headDimV = 64;
};

SdpaBwdSpec specOf(const Shape& shape, DataType dtype = DataType::HALF)
{
    SdpaBwdSpec spec;
    spec.dtype = dtype;
    spec.batch = shape.batch;
    spec.heads = shape.heads;
    spec.kvHeads = shape.kvHeads;
    spec.seqQ = shape.seqQ;
    spec.seqKv = shape.seqKv;
    spec.headDimQk = shape.headDimQk;
    spec.headDimV = shape.headDimV;
    return spec;
}

SdpaBwdSpec baseSpec()
{
    return SdpaBwdSpec{};
}

SdpaBwdSpec withCausal(const SdpaBwdSpec& spec)
{
    return spec.withEdit([](SdpaBwdGraph& g) { g.attributes.causalMask = true; });
}

/// A bias of the graph's full dims and, with @p withDbias, its gradient.
SdpaBwdSpec withBias(const SdpaBwdSpec& spec, bool withDbias, std::vector<int64_t> dims = {})
{
    return spec.withEdit([=](SdpaBwdGraph& g) {
        addBias(g, dims);
        if(withDbias)
        {
            addDbias(g);
        }
    });
}

SdpaBwdSpec withDropout(const SdpaBwdSpec& spec,
                        float probability = 0.2F,
                        DataType seedType = DataType::INT64,
                        bool passByValue = false)
{
    return spec.withEdit(
        [=](SdpaBwdGraph& g) { addDropout(g, probability, seedType, passByValue); });
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

/// A merged object's record, as the descriptor generator writes it.
struct RowSpec
{
    std::string dtype = "f16";
    int64_t tile = 64;
    int64_t causal = 0;
    int64_t hasBias = 0;
    int64_t dropout = 0;
    int64_t padded = 0;
    int64_t storeDb = -1;
    int64_t gqaPack = -1;
    int64_t blockThreads = 256;
    int64_t blockKv = 64;
    int64_t blockM = 64;
    std::string reduceSymbol;
    int64_t priority = 100;
};

KernelDefinition makeRow(const RowSpec& row)
{
    KernelDefinition kernel;
    kernel.name = "sdpa_bwd_merged_" + row.dtype + "_d" + std::to_string(row.tile);
    kernel.priority = row.priority;
    kernel.metadata = {{"dtype", row.dtype},
                       {"head_dim", row.tile},
                       {"head_dim_v", row.tile},
                       {"causal", row.causal},
                       {"has_bias", row.hasBias},
                       {"dropout", row.dropout},
                       {"padded", row.padded},
                       {"hints", int64_t{1}},
                       {"floor", std::string("tail")},
                       {"dform", std::string("bk")},
                       {"role_order", std::string("concat")},
                       {"store_db", row.storeDb},
                       {"gqa_pack", row.gqaPack},
                       {"block_threads", row.blockThreads},
                       {"block_kv", row.blockKv},
                       {"block_m", row.blockM},
                       {"reduce_symbol", row.reduceSymbol}};
    return kernel;
}

RowSpec plainRow(int64_t tile = 64)
{
    RowSpec row;
    row.tile = tile;
    return row;
}

RowSpec causalRow(int64_t tile = 64)
{
    RowSpec row = plainRow(tile);
    row.causal = 1;
    return row;
}

RowSpec biasRow(int64_t tile = 64)
{
    RowSpec row = plainRow(tile);
    row.hasBias = 1;
    row.reduceSymbol = REDUCE_SYMBOL;
    return row;
}

RowSpec dropoutRow(int64_t tile = 64)
{
    RowSpec row = plainRow(tile);
    row.dropout = 1;
    return row;
}

/// The merged table's coverage: exact objects at tiles 32 to 192 and one padded object at 32.
std::vector<RowSpec> catalog()
{
    std::vector<RowSpec> rows;
    for(const int64_t tile : {32, 64, 96, 128, 160, 192})
    {
        rows.push_back(plainRow(tile));
    }
    RowSpec padded = plainRow(32);
    padded.padded = 1;
    rows.push_back(padded);
    return rows;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

std::optional<BoundTokens> graphMatch(const SdpaBwdSpec& spec)
{
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    return matchesGraph(MERGED, fixture.context());
}

/// Whether @p row serves the graph of @p spec; the graph itself must be served.
bool serves(const SdpaBwdSpec& spec, const RowSpec& row)
{
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(MERGED, fixture.context());
    if(!bound.has_value())
    {
        throw std::logic_error("test graph is declined by graph_match");
    }
    return matchesKernel(MERGED, fixture.context(), makeRow(row), *bound);
}

bool anyServes(const SdpaBwdSpec& spec, const std::vector<RowSpec>& rows)
{
    if(!graphMatch(spec).has_value())
    {
        return false;
    }
    for(const auto& row : rows)
    {
        if(serves(spec, row))
        {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Launches
// ---------------------------------------------------------------------------

/// A fake device address per tensor uid, and a workspace base; only ever compared.
uint64_t ptrOf(int64_t uid)
{
    return static_cast<uint64_t>(uid) * 0x100000;
}

constexpr uint64_t WORKSPACE = 0x70000000;

void* asPointer(uint64_t address)
{
    return reinterpret_cast<void*>(static_cast<uintptr_t>(address));
}

/// Every uid the test graphs use, and the host address of a runtime scale when one is given.
std::vector<hipdnnPluginDeviceBuffer_t> buffersFor(float* hostScale = nullptr)
{
    std::vector<hipdnnPluginDeviceBuffer_t> buffers;
    for(int64_t uid = sdpa_bwd_uid::Q; uid <= sdpa_bwd_uid::DROPOUT_SCALE_INV; ++uid)
    {
        buffers.push_back({uid, asPointer(ptrOf(uid))});
    }
    if(hostScale != nullptr)
    {
        buffers[static_cast<size_t>(sdpa_bwd_uid::SCALE - 1)].ptr = hostScale;
    }
    return buffers;
}

FlydslSdpaBwdMergedLaunchDump dumpOf(const SdpaBwdSpec& spec,
                                     const RowSpec& row,
                                     uint64_t workspace = 0,
                                     float* hostScale = nullptr)
{
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    return describeFlydslSdpaBwdMergedLaunchForTesting(
        fixture.context(), makeRow(row), buffersFor(hostScale), asPointer(workspace));
}

void expectBadParam(const std::function<void()>& action)
{
    try
    {
        action();
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM) << e.what();
        return;
    }
    FAIL() << "expected a BAD_PARAM exception";
}

// ---------------------------------------------------------------------------
// Kernarg images
// ---------------------------------------------------------------------------

/// One kernel parameter: its name and kind -- p pointer, i int32, f float32, l int64.
struct Slot
{
    std::string name;
    char kind;
};

size_t slotBytes(char kind)
{
    return kind == 'p' || kind == 'l' ? 8 : 4;
}

/// `_PARAMS["merged"]`, in order.
const std::vector<Slot>& mergedSlots()
{
    static const std::vector<Slot> s_slots = [] {
        std::vector<Slot> slots;
        const auto add = [&slots](char kind, std::initializer_list<const char*> names) {
            for(const char* name : names)
            {
                slots.push_back({name, kind});
            }
        };
        add('p', {"Q", "K", "V", "B", "O", "DO", "DK", "DV", "DQ", "DB", "LSE"});
        add('i', {"batch_size"});
        add('p', {"seqinfo_q0", "seqinfo_q1", "seqinfo_k0", "seqinfo_k1"});
        add('i',
            {"varlen_bits",
             "num_seqlens",
             "max_seqlen_q",
             "max_seqlen_k",
             "window_left",
             "window_right"});
        add('p', {"philox_seed_ptr", "philox_offset1"});
        add('l', {"philox_offset2"});
        add('i', {"idropout_p"});
        add('f', {"dropout_scale"});
        add('i', {"num_head_q", "num_head_k", "hdim_qk", "hdim_vo"});
        add('f', {"sm_scale"});
        for(const char* tensor : {"q", "k", "v", "o", "do", "dk", "dv", "dq", "b", "db"})
        {
            const std::string stem = std::string("stride_") + tensor;
            const bool rowAxis = std::string(tensor) == "b" || std::string(tensor) == "db";
            slots.push_back({stem + "_batch", 'l'});
            slots.push_back({stem + "_head", 'l'});
            slots.push_back({stem + (rowAxis ? "_seq_q" : "_seq"), 'l'});
        }
        add('i', {"rt_flags"});
        return slots;
    }();
    return s_slots;
}

/// `_PARAMS["dbias_reduce"]`, in order.
const std::vector<Slot>& reduceSlots()
{
    static const std::vector<Slot> s_slots = {{"DB", 'p'},
                                              {"DBias", 'p'},
                                              {"num_batch", 'i'},
                                              {"num_head", 'i'},
                                              {"seqlen_q", 'i'},
                                              {"seqlen_k", 'i'},
                                              {"reduce_batch", 'i'},
                                              {"reduce_head", 'i'},
                                              {"stride_ws_batch", 'l'},
                                              {"stride_ws_head", 'l'},
                                              {"stride_ws_seq", 'l'},
                                              {"stride_out_batch", 'l'},
                                              {"stride_out_head", 'l'},
                                              {"stride_out_seq", 'l'}};
    return s_slots;
}

/// The raw bits of a value as its kernel parameter holds it.
uint64_t bits(float value)
{
    uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}

uint64_t bits(int32_t value)
{
    return static_cast<uint32_t>(value);
}

uint64_t bits(int64_t value)
{
    return static_cast<uint64_t>(value);
}

uint64_t bits(uint64_t value)
{
    return value;
}

/// The parameter values a kernarg image holds, by name, read at each parameter's natural offset.
std::map<std::string, uint64_t> decode(const std::vector<unsigned char>& image,
                                       const std::vector<Slot>& slots)
{
    std::map<std::string, uint64_t> values;
    size_t offset = 0;
    for(const auto& slot : slots)
    {
        const size_t size = slotBytes(slot.kind);
        offset = (offset + size - 1) / size * size;
        uint64_t raw = 0;
        if(offset + size <= image.size())
        {
            std::memcpy(&raw, image.data() + offset, size);
        }
        values[slot.name] = raw;
        offset += size;
    }
    return values;
}

size_t imageBytes(const std::vector<Slot>& slots)
{
    size_t offset = 0;
    for(const auto& slot : slots)
    {
        const size_t size = slotBytes(slot.kind);
        offset = (offset + size - 1) / size * size + size;
    }
    return offset;
}

/// Every parameter of one launch: starts from the baseline graph's dense launch and is edited
/// into the case under test.
class Golden
{
public:
    explicit Golden(const Shape& shape = {}, int64_t tile = 64)
    {
        const auto ceil8 = [](int64_t value) { return (value + 7) / 8 * 8; };
        // The matcher records 0 on a unit axis.
        const auto stride
            = [](int64_t extent, int64_t value) { return extent > 1 ? value : int64_t{0}; };
        const auto set3
            = [&](const std::string& tensor, int64_t heads, int64_t seq, int64_t headDim) {
                  this->set("stride_" + tensor + "_batch",
                            stride(shape.batch, heads * seq * ceil8(headDim)));
                  this->set("stride_" + tensor + "_head", stride(heads, seq * ceil8(headDim)));
                  // A unit sequence axis keeps its row pitch.
                  this->set("stride_" + tensor + "_seq", ceil8(headDim));
              };
        for(const auto& slot : mergedSlots())
        {
            _values[slot.name] = 0;
        }
        using namespace sdpa_bwd_uid;
        set("Q", ptrOf(Q));
        set("K", ptrOf(K));
        set("V", ptrOf(V));
        // A graph without bias passes O for B, DB and every seqinfo, seed and offset pointer.
        set("B", ptrOf(O));
        set("O", ptrOf(O));
        set("DO", ptrOf(DO));
        set("DK", ptrOf(DK));
        set("DV", ptrOf(DV));
        set("DQ", ptrOf(DQ));
        set("DB", ptrOf(O));
        set("LSE", ptrOf(STATS));
        set("batch_size", static_cast<int32_t>(shape.batch));
        for(const char* name : {"seqinfo_q0", "seqinfo_q1", "seqinfo_k0", "seqinfo_k1"})
        {
            set(name, ptrOf(O));
        }
        set("max_seqlen_q", static_cast<int32_t>(shape.seqQ));
        set("max_seqlen_k", static_cast<int32_t>(shape.seqKv));
        set("philox_seed_ptr", ptrOf(O));
        set("philox_offset1", ptrOf(O));
        set("dropout_scale", 1.0F);
        set("num_head_q", static_cast<int32_t>(shape.heads));
        set("num_head_k", static_cast<int32_t>(shape.kvHeads));
        set("hdim_qk", static_cast<int32_t>(shape.headDimQk));
        set("hdim_vo", static_cast<int32_t>(shape.headDimV));
        set("sm_scale", 1.0F);
        set3("q", shape.heads, shape.seqQ, shape.headDimQk);
        set3("k", shape.kvHeads, shape.seqKv, shape.headDimQk);
        set3("v", shape.kvHeads, shape.seqKv, shape.headDimV);
        set3("o", shape.heads, shape.seqQ, shape.headDimV);
        set3("do", shape.heads, shape.seqQ, shape.headDimV);
        set3("dk", shape.kvHeads, shape.seqKv, shape.headDimQk);
        set3("dv", shape.kvHeads, shape.seqKv, shape.headDimV);
        set3("dq", shape.heads, shape.seqQ, shape.headDimQk);
        const bool padded = shape.headDimQk != tile || shape.headDimV != tile;
        set("rt_flags", static_cast<int32_t>(padded ? 1 : 0));
    }

    template <typename T>
    Golden& set(const std::string& name, T value)
    {
        _values.at(name) = bits(value);
        return *this;
    }

    /// The (batch, head, row) strides of the bias, or of dB with @p tensor "db".
    Golden& strides(const std::string& tensor, int64_t batch, int64_t head, int64_t row)
    {
        set("stride_" + tensor + "_batch", batch);
        set("stride_" + tensor + "_head", head);
        set("stride_" + tensor + "_seq_q", row);
        return *this;
    }

    const std::map<std::string, uint64_t>& values() const
    {
        return _values;
    }

private:
    std::map<std::string, uint64_t> _values;
};

void expectKernarg(const std::vector<unsigned char>& image, const Golden& golden)
{
    ASSERT_EQ(image.size(), 452U);
    ASSERT_EQ(imageBytes(mergedSlots()), 452U);
    const auto actual = decode(image, mergedSlots());
    for(const auto& slot : mergedSlots())
    {
        EXPECT_EQ(actual.at(slot.name), golden.values().at(slot.name)) << slot.name;
    }
}

/// The reduce kernel's parameters, as a launch hands them over.
struct ReduceGolden
{
    uint64_t workspace;
    uint64_t dBias;
    int32_t numBatch;
    int32_t numHead;
    int32_t seqLenQ;
    int32_t seqLenK;
    int32_t reduceBatch;
    int32_t reduceHead;
    std::array<int64_t, 3> workspaceStrides;
    std::array<int64_t, 3> outStrides;
};

void expectReduceKernarg(const std::vector<unsigned char>& image, const ReduceGolden& golden)
{
    ASSERT_EQ(image.size(), 88U);
    ASSERT_EQ(imageBytes(reduceSlots()), 88U);
    const auto actual = decode(image, reduceSlots());
    EXPECT_EQ(actual.at("DB"), golden.workspace);
    EXPECT_EQ(actual.at("DBias"), golden.dBias);
    EXPECT_EQ(actual.at("num_batch"), bits(golden.numBatch));
    EXPECT_EQ(actual.at("num_head"), bits(golden.numHead));
    EXPECT_EQ(actual.at("seqlen_q"), bits(golden.seqLenQ));
    EXPECT_EQ(actual.at("seqlen_k"), bits(golden.seqLenK));
    EXPECT_EQ(actual.at("reduce_batch"), bits(golden.reduceBatch));
    EXPECT_EQ(actual.at("reduce_head"), bits(golden.reduceHead));
    EXPECT_EQ(actual.at("stride_ws_batch"), bits(golden.workspaceStrides[0]));
    EXPECT_EQ(actual.at("stride_ws_head"), bits(golden.workspaceStrides[1]));
    EXPECT_EQ(actual.at("stride_ws_seq"), bits(golden.workspaceStrides[2]));
    EXPECT_EQ(actual.at("stride_out_batch"), bits(golden.outStrides[0]));
    EXPECT_EQ(actual.at("stride_out_head"), bits(golden.outStrides[1]));
    EXPECT_EQ(actual.at("stride_out_seq"), bits(golden.outStrides[2]));
}

size_t workspaceBytesOf(const SdpaBwdSpec& spec, const RowSpec& row)
{
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(MERGED, fixture.context());
    if(!bound.has_value())
    {
        throw std::logic_error("test graph is declined by graph_match");
    }
    return dispatchHandler(MERGED).workspaceBytes(fixture.context(), *bound, makeRow(row));
}

// ---------------------------------------------------------------------------
// graph_match
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMergedGraph, RegistersItsFourSymbols)
{
    EXPECT_NE(graphMatcher(MERGED), nullptr);
    EXPECT_NE(kernelMatcher(MERGED), nullptr);
    EXPECT_NE(scorer(MERGED), nullptr);
    EXPECT_NE(&dispatchHandler(MERGED), nullptr);
}

TEST(TestFlydslSdpaBwdMergedGraph, ServesThePlainGraphTheSharedMatcherAccepts)
{
    EXPECT_TRUE(std::holds_alternative<SdpaBwdProblem>(matchSdpaBwdSpec(baseSpec())));
    EXPECT_TRUE(graphMatch(baseSpec()).has_value());
}

TEST(TestFlydslSdpaBwdMergedGraph, DeclinesAGraphTheSharedMatcherDeclines)
{
    // Every tensor in fp32: DT-2, a [data_type] decline.
    const auto spec = specOf({}, DataType::FLOAT);
    EXPECT_TRUE(std::holds_alternative<flydsl_sdpa_bwd::Decline>(matchSdpaBwdSpec(spec)));
    EXPECT_FALSE(graphMatch(spec).has_value());
}

TEST(TestFlydslSdpaBwdMergedGraph, DeclinesAForwardGraphWithoutAnalysingIt)
{
    const auto spec
        = baseSpec().withEdit([](SdpaBwdGraph& g) { g.nodeKind = SdpaBwdNodeKind::FORWARD; });
    EXPECT_FALSE(graphMatch(spec).has_value());
}

TEST(TestFlydslSdpaBwdMergedGraph, EN1_DeclinesAHeadSizeNoMergedObjectCovers)
{
    const auto atDim = [](int64_t qk, int64_t v) {
        Shape shape;
        shape.headDimQk = qk;
        shape.headDimV = v;
        return specOf(shape);
    };
    // The shared matcher serves both; the merged table (tiles 32 to 192) covers neither.
    for(const int64_t headDim : {100, 224})
    {
        ASSERT_TRUE(
            std::holds_alternative<SdpaBwdProblem>(matchSdpaBwdSpec(atDim(headDim, headDim))));
        EXPECT_FALSE(graphMatch(atDim(headDim, headDim)).has_value()) << headDim;
    }
    EXPECT_FALSE(graphMatch(atDim(64, 40)).has_value());
    EXPECT_TRUE(graphMatch(atDim(96, 96)).has_value());
    EXPECT_TRUE(graphMatch(atDim(24, 24)).has_value());
}

TEST(TestFlydslSdpaBwdMergedGraph, EN2_DeclinesPaddedVarlenThatTheMatcherAccepts)
{
    const auto padded = baseSpec().withEdit([](SdpaBwdGraph& g) { addSeqLens(g); });
    // PV-1: the shared matcher serves it; the merged engine does not.
    ASSERT_TRUE(std::holds_alternative<SdpaBwdProblem>(matchSdpaBwdSpec(padded)));
    EXPECT_FALSE(graphMatch(padded).has_value());
    EXPECT_TRUE(graphMatch(baseSpec()).has_value());
}

TEST(TestFlydslSdpaBwdMergedGraph, EN2_DeclinesPaddedVarlenWithAMask)
{
    const auto padded = withCausal(baseSpec()).withEdit([](SdpaBwdGraph& g) { addSeqLens(g); });
    ASSERT_TRUE(std::holds_alternative<SdpaBwdProblem>(matchSdpaBwdSpec(padded)));
    EXPECT_FALSE(graphMatch(padded).has_value());
    EXPECT_TRUE(graphMatch(withCausal(baseSpec())).has_value());
}

// ---------------------------------------------------------------------------
// kernel_match
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMergedKernel, ServesAPlainGraphWithAPlainObject)
{
    EXPECT_TRUE(serves(baseSpec(), plainRow()));
}

TEST(TestFlydslSdpaBwdMergedKernel, ServesTheMaskedGraphsWithTheCausalObjectOnly)
{
    const auto bottomRight
        = baseSpec().withEdit([](SdpaBwdGraph& g) { g.attributes.causalMaskBottomRight = true; });
    const auto window = baseSpec().withEdit([](SdpaBwdGraph& g) { g.attributes.leftBound = 4; });
    for(const auto& spec : {withCausal(baseSpec()), bottomRight, window})
    {
        EXPECT_TRUE(serves(spec, causalRow()));
        EXPECT_FALSE(serves(spec, plainRow()));
    }
    EXPECT_FALSE(serves(baseSpec(), causalRow()));
}

TEST(TestFlydslSdpaBwdMergedKernel, ServesABiasGraphWithABiasObjectOnly)
{
    const auto spec = withBias(baseSpec(), false);
    EXPECT_TRUE(serves(spec, biasRow()));
    EXPECT_FALSE(serves(spec, plainRow()));
    EXPECT_FALSE(serves(baseSpec(), biasRow()));
}

TEST(TestFlydslSdpaBwdMergedKernel, ServesADtypeOnlyWithItsObject)
{
    RowSpec bf16 = plainRow();
    bf16.dtype = "bf16";
    EXPECT_TRUE(serves(baseSpec(), plainRow()));
    EXPECT_FALSE(serves(baseSpec(), bf16));
    EXPECT_TRUE(serves(specOf({}, DataType::BFLOAT16), bf16));
    EXPECT_FALSE(serves(specOf({}, DataType::BFLOAT16), plainRow()));
}

TEST(TestFlydslSdpaBwdMergedKernel, ServesDropoutOnlyWithADropoutObject)
{
    const auto spec = withDropout(baseSpec());
    EXPECT_TRUE(serves(spec, dropoutRow()));
    EXPECT_FALSE(serves(spec, plainRow()));
    EXPECT_FALSE(serves(baseSpec(), dropoutRow()));
}

TEST(TestFlydslSdpaBwdMergedKernel, EN1_ServesAnExactObjectOnlyAtItsTile)
{
    Shape wide;
    wide.headDimQk = 96;
    wide.headDimV = 96;
    EXPECT_FALSE(serves(specOf(wide), plainRow(64)));
    EXPECT_TRUE(serves(specOf(wide), plainRow(96)));
}

TEST(TestFlydslSdpaBwdMergedKernel, EN1_DoesNotServeAValueDimThatDiffersFromTheKeyDim)
{
    Shape mla;
    mla.headDimQk = 64;
    mla.headDimV = 32;
    EXPECT_FALSE(anyServes(specOf(mla), catalog()));
    Shape equal;
    EXPECT_TRUE(serves(specOf(equal), plainRow(64)));
}

TEST(TestFlydslSdpaBwdMergedKernel, EN1_ServesAPaddedObjectAtOrBelowItsTile)
{
    RowSpec padded = plainRow(32);
    padded.padded = 1;
    const auto servesDims = [&](int64_t qk, int64_t v) {
        Shape shape;
        shape.headDimQk = qk;
        shape.headDimV = v;
        return serves(specOf(shape), padded);
    };
    EXPECT_TRUE(servesDims(24, 24));
    EXPECT_TRUE(servesDims(32, 32));
    EXPECT_TRUE(servesDims(32, 16));
    EXPECT_TRUE(servesDims(8, 8));
}

TEST(TestFlydslSdpaBwdMergedKernel, EN1_ServesTheSmallestTilePaddedBelowIt)
{
    RowSpec padded = plainRow(32);
    padded.padded = 1;
    Shape small;
    small.headDimQk = 24;
    small.headDimV = 24;
    EXPECT_TRUE(serves(specOf(small), padded));
    EXPECT_FALSE(serves(specOf(small), plainRow(32)));
}

TEST(TestFlydslSdpaBwdMergedKernel, EN1_ServesNoHeadDimTheCatalogDoesNotCover)
{
    const auto rows = catalog();
    for(const int64_t headDim : {24, 32, 64, 96, 128, 160, 192})
    {
        Shape shape;
        shape.headDimQk = headDim;
        shape.headDimV = headDim;
        EXPECT_TRUE(anyServes(specOf(shape), rows)) << "D " << headDim;
    }
    // D = 100 sits between tiles and D = 224 above the merged table: no row serves either,
    // and the graph beside them (D = 96) is served.
    for(const int64_t headDim : {100, 224, 256})
    {
        Shape shape;
        shape.headDimQk = headDim;
        shape.headDimV = headDim;
        EXPECT_FALSE(anyServes(specOf(shape), rows)) << "D " << headDim;
    }
    Shape unequal;
    unequal.headDimQk = 64;
    unequal.headDimV = 32;
    EXPECT_FALSE(anyServes(specOf(unequal), rows));
}

TEST(TestFlydslSdpaBwdMergedKernel, EN3_ServesAGqaPackedObjectOnlyAGroupedQueryGraph)
{
    RowSpec packed = plainRow();
    packed.gqaPack = 1;
    Shape grouped;
    grouped.kvHeads = 2;
    EXPECT_TRUE(serves(specOf(grouped), packed));
    EXPECT_FALSE(serves(baseSpec(), packed));
    // A grouped graph is served unpacked too.
    EXPECT_TRUE(serves(specOf(grouped), plainRow()));
}

TEST(TestFlydslSdpaBwdMergedKernel, EN3_ServesAMultiQueryGraphWithAPackedObject)
{
    RowSpec packed = plainRow();
    packed.gqaPack = 1;
    Shape multiQuery;
    multiQuery.kvHeads = 1;
    EXPECT_TRUE(serves(specOf(multiQuery), packed));
}

TEST(TestFlydslSdpaBwdMergedKernel, BI8_DeclinesABiasWhoseRowStrideIsNotAMultipleOfEight)
{
    Shape ragged;
    ragged.seqKv = 12;
    auto noStore = biasRow();
    noStore.storeDb = 0;
    // Every bias object stages the bias, whatever else it is built for.
    EXPECT_FALSE(serves(withBias(specOf(ragged), false), biasRow()));
    EXPECT_FALSE(serves(withBias(specOf(ragged), false), noStore));
    EXPECT_TRUE(serves(withBias(baseSpec(), false), biasRow()));
    EXPECT_TRUE(serves(withBias(baseSpec(), false), noStore));
}

TEST(TestFlydslSdpaBwdMergedKernel, BI8_DeclinesABiasWhoseHeadStrideIsNotAMultipleOfEight)
{
    const auto padded = baseSpec().withEdit([](SdpaBwdGraph& g) {
        auto& bias = addBias(g);
        // A row stride of 16 with head planes 4 elements past their rows.
        bias.strides = {g.heads * (g.seqQ * 16 + 4), g.seqQ * 16 + 4, 16, 1};
    });
    EXPECT_FALSE(serves(padded, biasRow()));
    EXPECT_TRUE(serves(withBias(baseSpec(), false), biasRow()));
}

TEST(TestFlydslSdpaBwdMergedKernel, BI8_ServesABroadcastBiasWhoseBroadcastAxesHaveNoStride)
{
    const auto spec = withBias(baseSpec(), false, {1, 1, 16, 16});
    EXPECT_TRUE(serves(spec, biasRow()));
}

TEST(TestFlydslSdpaBwdMergedKernel, BI8_AppliesToABiasObjectNotAPlainOne)
{
    // The same 12-wide bias rows: only a bias object stages them.
    Shape ragged;
    ragged.seqKv = 12;
    EXPECT_FALSE(serves(withBias(specOf(ragged), false), biasRow()));
    EXPECT_TRUE(serves(specOf(ragged), plainRow()));
}

TEST(TestFlydslSdpaBwdMergedKernel, ServesAGraphWithDbiasOnlyWithAnObjectThatStoresIt)
{
    auto noStore = biasRow();
    noStore.storeDb = 0;
    const auto withGradient = withBias(baseSpec(), true);
    const auto withoutGradient = withBias(baseSpec(), false);
    EXPECT_TRUE(serves(withGradient, biasRow()));
    EXPECT_FALSE(serves(withGradient, noStore));
    EXPECT_TRUE(serves(withoutGradient, noStore));
}

TEST(TestFlydslSdpaBwdMergedKernel, ServesABroadcastDbiasGraphWithABiasObject)
{
    for(const auto& dims : {std::vector<int64_t>{1, 8, 16, 16},
                            std::vector<int64_t>{2, 1, 16, 16},
                            std::vector<int64_t>{1, 1, 16, 16}})
    {
        EXPECT_TRUE(serves(withBias(baseSpec(), true, dims), biasRow()));
    }
}

TEST(TestFlydslSdpaBwdMergedKernel, DeclinesARecordMissingAField)
{
    const auto spec = baseSpec();
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    const auto bound = matchesGraph(MERGED, fixture.context());
    ASSERT_TRUE(bound.has_value());
    auto kernel = makeRow(plainRow());
    EXPECT_TRUE(matchesKernel(MERGED, fixture.context(), kernel, *bound));
    kernel.metadata.erase("block_m");
    EXPECT_FALSE(matchesKernel(MERGED, fixture.context(), kernel, *bound));
}

TEST(TestFlydslSdpaBwdMergedKernel, DeclinesALaunchGridPastInt32)
{
    // One block per tile of one row: B * (Hk * Sk + Hq * Sq) blocks.
    RowSpec tiny = plainRow();
    tiny.blockKv = 1;
    tiny.blockM = 1;
    Shape fits;
    fits.batch = int64_t{1} << 20;
    Shape overflows;
    overflows.batch = int64_t{1} << 26;
    EXPECT_TRUE(serves(specOf(fits), tiny));
    EXPECT_FALSE(serves(specOf(overflows), tiny));
    expectBadParam([&] { static_cast<void>(dumpOf(specOf(overflows), tiny)); });
}

TEST(TestFlydslSdpaBwdMergedKernel, DeclinesADbiasReducePastTheGridRowLimit)
{
    // A head-broadcast dBias reduces one plane per batch: one grid row each, at most 65535.
    const auto planes = [](int64_t batch) {
        Shape shape;
        shape.batch = batch;
        shape.heads = 2;
        shape.kvHeads = 2;
        return withBias(specOf(shape), true, {batch, 1, 16, 16});
    };
    EXPECT_TRUE(serves(planes(65535), biasRow()));
    EXPECT_FALSE(serves(planes(65536), biasRow()));
    expectBadParam([&] { static_cast<void>(dumpOf(planes(65536), biasRow(), WORKSPACE)); });
}

TEST(TestFlydslSdpaBwdMergedKernel, ScoresAnObjectByItsPriority)
{
    const GraphFixture fixture(buildSdpaBwdGraph(baseSpec()));
    const auto bound = matchesGraph(MERGED, fixture.context());
    ASSERT_TRUE(bound.has_value());
    RowSpec first = plainRow();
    first.priority = 7;
    RowSpec second = plainRow();
    second.priority = 4000;
    EXPECT_EQ(scoreKernel(MERGED, fixture.context(), makeRow(first), *bound), 7.0);
    EXPECT_EQ(scoreKernel(MERGED, fixture.context(), makeRow(second), *bound), 4000.0);
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMergedWorkspace, IsEmptyWithoutDropoutOrABroadcastDbias)
{
    EXPECT_EQ(workspaceBytesOf(baseSpec(), plainRow()), 0U);
    EXPECT_EQ(workspaceBytesOf(withCausal(baseSpec()), causalRow()), 0U);
    EXPECT_EQ(workspaceBytesOf(withBias(baseSpec(), false), biasRow()), 0U);
    // A dBias that is not broadcast is written in place.
    EXPECT_EQ(workspaceBytesOf(withBias(baseSpec(), true), biasRow()), 0U);
}

TEST(TestFlydslSdpaBwdMergedWorkspace, HoldsTheSeedAndOffsetUnlessTheyAre64BitDeviceTensors)
{
    EXPECT_EQ(workspaceBytesOf(withDropout(baseSpec(), 0.2F, DataType::INT64, false), dropoutRow()),
              0U);
    EXPECT_EQ(workspaceBytesOf(withDropout(baseSpec(), 0.2F, DataType::INT32, false), dropoutRow()),
              16U);
    EXPECT_EQ(workspaceBytesOf(withDropout(baseSpec(), 0.2F, DataType::INT64, true), dropoutRow()),
              16U);
    EXPECT_EQ(workspaceBytesOf(withDropout(baseSpec(), 0.2F, DataType::INT32, true), dropoutRow()),
              16U);
}

TEST(TestFlydslSdpaBwdMergedWorkspace, HoldsAFullShapeDbiasWhereTheBiasIsBroadcast)
{
    // (B, Hq, Sq, Sk) elements of 2 bytes, whichever axes the bias broadcasts.
    constexpr size_t FULL_BYTES = 2 * 8 * 16 * 16 * 2;
    for(const auto& dims : {std::vector<int64_t>{1, 8, 16, 16},
                            std::vector<int64_t>{2, 1, 16, 16},
                            std::vector<int64_t>{1, 1, 16, 16}})
    {
        EXPECT_EQ(workspaceBytesOf(withBias(baseSpec(), true, dims), biasRow()), FULL_BYTES);
    }
    // A broadcast bias without a gradient needs none.
    EXPECT_EQ(workspaceBytesOf(withBias(baseSpec(), false, {1, 8, 16, 16}), biasRow()), 0U);
}

TEST(TestFlydslSdpaBwdMergedWorkspace, PlacesTheDbiasBufferOnThe256ByteBoundaryAfterTheSeed)
{
    constexpr size_t FULL_BYTES = 2 * 8 * 16 * 16 * 2;
    const auto spec
        = withDropout(withBias(baseSpec(), true, {1, 8, 16, 16}), 0.2F, DataType::INT32, false);
    auto row = biasRow();
    row.dropout = 1;
    EXPECT_EQ(workspaceBytesOf(spec, row), 256 + FULL_BYTES);
}

TEST(TestFlydslSdpaBwdMergedWorkspace, AgreesWithTheOffsetsALaunchUses)
{
    constexpr size_t FULL_BYTES = 2 * 8 * 16 * 16 * 2;
    struct Case
    {
        SdpaBwdSpec spec;
        RowSpec row;
        bool seed;
        bool dbias;
    };
    auto dropoutBias = biasRow();
    dropoutBias.dropout = 1;
    const std::vector<Case> cases{
        {baseSpec(), plainRow(), false, false},
        {withDropout(baseSpec(), 0.2F, DataType::INT64, false), dropoutRow(), false, false},
        {withDropout(baseSpec(), 0.2F, DataType::INT32, false), dropoutRow(), true, false},
        {withDropout(baseSpec(), 0.2F, DataType::INT64, true), dropoutRow(), true, false},
        {withBias(baseSpec(), true, {1, 8, 16, 16}), biasRow(), false, true},
        {withDropout(withBias(baseSpec(), true, {2, 1, 16, 16}), 0.2F, DataType::INT32, false),
         dropoutBias,
         true,
         true}};
    for(const auto& test : cases)
    {
        const auto bytes = workspaceBytesOf(test.spec, test.row);
        const auto dump = dumpOf(test.spec, test.row, WORKSPACE);
        const auto values = decode(dump.kernarg, mergedSlots());
        EXPECT_EQ(dump.workspaceBytes, bytes);
        EXPECT_EQ(dump.seedUploaded, test.seed);
        if(test.seed)
        {
            EXPECT_EQ(values.at("philox_seed_ptr"), WORKSPACE);
            EXPECT_EQ(values.at("philox_offset1"), WORKSPACE + 8);
        }
        const size_t dbiasOffset = test.seed ? 256 : 0;
        if(test.dbias)
        {
            EXPECT_EQ(values.at("DB"), WORKSPACE + dbiasOffset);
            // The region ends exactly where the workspace does.
            EXPECT_EQ(dbiasOffset + FULL_BYTES, bytes);
        }
        else
        {
            EXPECT_EQ(bytes, test.seed ? 16U : 0U);
        }
    }
}

TEST(TestFlydslSdpaBwdMergedWorkspace, IsRequiredByALaunchThatNeedsOne)
{
    const auto spec = withDropout(baseSpec(), 0.2F, DataType::INT32, false);
    expectBadParam([&] { static_cast<void>(dumpOf(spec, dropoutRow(), 0)); });
    EXPECT_NO_THROW(static_cast<void>(dumpOf(spec, dropoutRow(), WORKSPACE)));
}

// ---------------------------------------------------------------------------
// Kernarg dumps: one whole parameter list per signature class
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMergedKernarg, Dense)
{
    const auto dump = dumpOf(baseSpec(), plainRow());
    expectKernarg(dump.kernarg, Golden());
    // One block per (batch, key tile, key head) and (batch, query tile, query head): 2 * (8 + 8).
    EXPECT_EQ(dump.grid, (std::array<unsigned int, 3>{32, 1, 1}));
    EXPECT_EQ(dump.block, 256U);
    EXPECT_FALSE(dump.reduces);
    EXPECT_FALSE(dump.seedUploaded);
}

TEST(TestFlydslSdpaBwdMergedKernarg, DenseOfAnotherShapeAndATileWithAPaddedHead)
{
    Shape shape;
    shape.batch = 3;
    shape.heads = 4;
    shape.kvHeads = 2;
    shape.seqQ = 100;
    shape.seqKv = 72;
    shape.headDimQk = 40;
    shape.headDimV = 24;
    RowSpec padded = plainRow(64);
    padded.padded = 1;
    padded.blockKv = 32;
    padded.blockM = 16;
    padded.blockThreads = 512;
    const auto dump = dumpOf(specOf(shape), padded);
    // The padded head sets rt_flags bit 0.
    expectKernarg(dump.kernarg, Golden(shape, 64));
    // 3 * (2 * ceil(72 / 32) + 4 * ceil(100 / 16)) = 3 * (6 + 28).
    EXPECT_EQ(dump.grid[0], 102U);
    EXPECT_EQ(dump.block, 512U);
}

TEST(TestFlydslSdpaBwdMergedKernarg, StatsInTheTokenMajorLayoutSetTheLseLayoutBit)
{
    const auto spec = baseSpec().withEdit(
        [](SdpaBwdGraph& g) { g.tensor("stats").strides = {g.seqQ * g.heads, 1, g.heads, 1}; });
    const auto dump = dumpOf(spec, plainRow());
    expectKernarg(dump.kernarg, Golden().set("varlen_bits", int32_t{1} << 16));
}

TEST(TestFlydslSdpaBwdMergedKernarg, Dropout)
{
    const auto spec = withDropout(baseSpec(), 0.2F, DataType::INT64, false);
    const auto dump = dumpOf(spec, dropoutRow());
    expectKernarg(dump.kernarg,
                  Golden()
                      .set("philox_seed_ptr", ptrOf(sdpa_bwd_uid::SEED))
                      .set("philox_offset1", ptrOf(sdpa_bwd_uid::OFFSET))
                      .set("idropout_p", int32_t{-1288490175})
                      .set("dropout_scale", 1.25F)
                      .set("rt_flags", int32_t{2}));
    EXPECT_FALSE(dump.seedUploaded);
}

TEST(TestFlydslSdpaBwdMergedKernarg, DropoutWithAPaddedHeadSetsBothRuntimeFlags)
{
    Shape shape;
    shape.headDimQk = 40;
    shape.headDimV = 40;
    RowSpec row = dropoutRow(64);
    row.padded = 1;
    const auto dump = dumpOf(withDropout(specOf(shape), 0.2F), row);
    EXPECT_EQ(decode(dump.kernarg, mergedSlots()).at("rt_flags"), 3U);
}

TEST(TestFlydslSdpaBwdMergedKernarg, ScaleValue)
{
    const auto spec
        = baseSpec().withEdit([](SdpaBwdGraph& g) { g.attributes.attnScaleValue = 0.125F; });
    const auto dump = dumpOf(spec, plainRow());
    expectKernarg(dump.kernarg, Golden().set("sm_scale", 0.125F));
}

TEST(TestFlydslSdpaBwdMergedKernarg, ScaleTensorBakedAndSuppliedAtExecute)
{
    const auto baked = baseSpec().withEdit([](SdpaBwdGraph& g) { addScaleTensor(g, 0.5F); });
    expectKernarg(dumpOf(baked, plainRow()).kernarg, Golden().set("sm_scale", 0.5F));

    const auto runtime
        = baseSpec().withEdit([](SdpaBwdGraph& g) { addScaleTensor(g, std::nullopt); });
    float hostScale = -0.75F;
    expectKernarg(dumpOf(runtime, plainRow(), 0, &hostScale).kernarg,
                  Golden().set("sm_scale", -0.75F));
    // The value is read at each launch, so another one is another launch.
    hostScale = 3.0F;
    expectKernarg(dumpOf(runtime, plainRow(), 0, &hostScale).kernarg,
                  Golden().set("sm_scale", 3.0F));
}

TEST(TestFlydslSdpaBwdMergedKernarg, ARuntimeScaleIsCheckedAtLaunch)
{
    const auto runtime
        = baseSpec().withEdit([](SdpaBwdGraph& g) { addScaleTensor(g, std::nullopt); });
    for(const float good : {0.0F, -1.2F, 1048576.0F, -1048576.0F})
    {
        float scale = good;
        EXPECT_NO_THROW(static_cast<void>(dumpOf(runtime, plainRow(), 0, &scale))) << good;
    }
    for(const float bad : {1048577.0F,
                           -2097152.0F,
                           std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN()})
    {
        float scale = bad;
        expectBadParam([&] { static_cast<void>(dumpOf(runtime, plainRow(), 0, &scale)); });
    }
}

TEST(TestFlydslSdpaBwdMergedKernarg, MaskSentinelsForPlainCausalAtEitherAlignment)
{
    const auto topLeft = dumpOf(withCausal(baseSpec()), causalRow());
    expectKernarg(topLeft.kernarg,
                  Golden()
                      .set("window_left", int32_t{-2147483647})
                      .set("window_right", int32_t{-2147483647}));

    const auto bottomRight
        = baseSpec().withEdit([](SdpaBwdGraph& g) { g.attributes.causalMaskBottomRight = true; });
    expectKernarg(dumpOf(bottomRight, causalRow()).kernarg,
                  Golden()
                      .set("window_left", int32_t{-2147483646})
                      .set("window_right", int32_t{-2147483646}));

    // Rectangular, so the sentinel is not the literal offset the kernel would otherwise read.
    Shape rect;
    rect.seqQ = 12;
    const auto rectangular
        = specOf(rect).withEdit([](SdpaBwdGraph& g) { g.attributes.causalMaskBottomRight = true; });
    expectKernarg(dumpOf(rectangular, causalRow()).kernarg,
                  Golden(rect)
                      .set("window_left", int32_t{-2147483646})
                      .set("window_right", int32_t{-2147483646}));
}

/// A masked graph and the window literals the kernel is handed for it.
struct WindowCase
{
    const char* name;
    int64_t seqQ;
    int64_t seqKv;
    bool bottomRight;
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
    int32_t expectedLeft;
    int32_t expectedRight;
};

TEST(TestFlydslSdpaBwdMergedKernarg, WindowLiteralsInTheKernelsConvention)
{
    // cuDNN keeps key j for row i when i + offset - left < j <= i + offset + right; the
    // kernel keeps i - window_left <= j <= i + window_right, so window_left = left - 1 - offset
    // and window_right = right + offset, each clamped to the extent it can reach.
    const std::vector<WindowCase> cases{
        {"LeftBoundOnly", 16, 16, false, 4, std::nullopt, 3, 16},
        {"BottomRightWithABoundBelowAndTheDiagonalAbove", 12, 16, true, 3, 0, -2, 4},
        {"RightBoundPastTheKeysClampsToTheKeyLength", 16, 16, false, std::nullopt, 100, 16, 16},
        {"LeftBoundPastTheQueriesClampsToTheQueryLength", 16, 16, false, 1000, 2, 16, 2},
        {"BottomRightWithBothBounds", 16, 24, true, 5, 3, -4, 11},
        {"MoreQueriesThanKeys", 24, 16, true, 4, 2, 11, -6},
        {"AZeroLeftBoundWithARightBound", 16, 16, false, 0, 3, -1, 3}};
    for(const auto& test : cases)
    {
        SCOPED_TRACE(test.name);
        Shape shape;
        shape.seqQ = test.seqQ;
        shape.seqKv = test.seqKv;
        const auto spec = specOf(shape).withEdit([&test](SdpaBwdGraph& g) {
            g.attributes.leftBound = test.leftBound;
            g.attributes.rightBound = test.rightBound;
            if(test.bottomRight)
            {
                g.attributes.diagonalAlignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
            }
        });
        const auto dump = dumpOf(spec, causalRow());
        expectKernarg(dump.kernarg,
                      Golden(shape)
                          .set("window_left", test.expectedLeft)
                          .set("window_right", test.expectedRight));
    }
}

TEST(TestFlydslSdpaBwdMergedKernarg, ABiasWithoutDbiasIsReadAndNeverStored)
{
    const auto dump = dumpOf(withBias(baseSpec(), false), biasRow());
    // B is the bias with its strides; DB stays O over zero strides, which suppresses the stores.
    expectKernarg(dump.kernarg,
                  Golden().set("B", ptrOf(sdpa_bwd_uid::BIAS)).strides("b", 2048, 256, 16));
    EXPECT_FALSE(dump.reduces);
}

TEST(TestFlydslSdpaBwdMergedKernarg, ADbiasThatIsNotBroadcastIsWrittenInPlace)
{
    const auto dump = dumpOf(withBias(baseSpec(), true), biasRow());
    expectKernarg(dump.kernarg,
                  Golden()
                      .set("B", ptrOf(sdpa_bwd_uid::BIAS))
                      .set("DB", ptrOf(sdpa_bwd_uid::DBIAS))
                      .strides("b", 2048, 256, 16)
                      .strides("db", 2048, 256, 16));
    EXPECT_FALSE(dump.reduces);
    EXPECT_EQ(dump.workspaceBytes, 0U);
}

TEST(TestFlydslSdpaBwdMergedKernarg, ADbiasOverUnitAxesKeepsAStrideSoTheKernelStoresIt)
{
    // The kernel suppresses dB for an all-zero stride triple, and every axis here has one element.
    Shape unit;
    unit.batch = 1;
    unit.heads = 1;
    unit.kvHeads = 1;
    unit.seqQ = 1;
    unit.seqKv = 16;
    const auto dump = dumpOf(withBias(specOf(unit), true), biasRow());
    const auto values = decode(dump.kernarg, mergedSlots());
    EXPECT_EQ(values.at("DB"), ptrOf(sdpa_bwd_uid::DBIAS));
    EXPECT_EQ(values.at("stride_db_batch"), 0U);
    EXPECT_EQ(values.at("stride_db_head"), 0U);
    EXPECT_EQ(values.at("stride_db_seq_q"), 16U);
}

TEST(TestFlydslSdpaBwdMergedKernarg, ASingleRowQueryKeepsARowPitch)
{
    // Sq = 1: the matcher records 0 for the sequence stride of Q, O, dO and dQ; the kernels need
    // the row pitch to bound their padding rows.
    Shape shape;
    shape.seqQ = 1;
    expectKernarg(dumpOf(specOf(shape), plainRow()).kernarg, Golden(shape));
}

TEST(TestFlydslSdpaBwdMergedKernarg, ASingleRowKeyKeepsARowPitch)
{
    Shape shape;
    shape.seqKv = 1;
    expectKernarg(dumpOf(specOf(shape), plainRow()).kernarg, Golden(shape));
}

TEST(TestFlydslSdpaBwdMergedKernarg, ARowPitchFollowsEachOperandsOwnHeadDim)
{
    Shape shape;
    shape.seqQ = 1;
    shape.seqKv = 1;
    shape.headDimQk = 24;
    shape.headDimV = 16;
    RowSpec row = plainRow(32);
    row.padded = 1;
    const auto values = decode(dumpOf(specOf(shape), row).kernarg, mergedSlots());
    for(const char* tensor : {"q", "k", "dk", "dq"})
    {
        EXPECT_EQ(values.at(std::string("stride_") + tensor + "_seq"), 24U) << tensor;
    }
    for(const char* tensor : {"v", "o", "do", "dv"})
    {
        EXPECT_EQ(values.at(std::string("stride_") + tensor + "_seq"), 16U) << tensor;
    }
}

TEST(TestFlydslSdpaBwdMergedKernarg, ADbiasOverASingleRowQueryKeepsARowPitchWithBatchAndHeads)
{
    // Sq = 1 with B = 2: the batch and head strides are non-zero, the row stride is not.
    Shape shape;
    shape.seqQ = 1;
    const auto dump = dumpOf(withBias(specOf(shape), true), biasRow());
    const auto values = decode(dump.kernarg, mergedSlots());
    EXPECT_EQ(values.at("DB"), ptrOf(sdpa_bwd_uid::DBIAS));
    EXPECT_EQ(values.at("stride_db_batch"), 128U);
    EXPECT_EQ(values.at("stride_db_head"), 16U);
    EXPECT_EQ(values.at("stride_db_seq_q"), 16U);
}

// ---------------------------------------------------------------------------
// The dBias reduce launch
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMergedReduce, SumsOverABatchBroadcastBias)
{
    const auto dump = dumpOf(withBias(baseSpec(), true, {1, 8, 16, 16}), biasRow(), WORKSPACE);
    expectKernarg(dump.kernarg,
                  Golden()
                      .set("B", ptrOf(sdpa_bwd_uid::BIAS))
                      .set("DB", WORKSPACE)
                      .strides("b", 0, 256, 16)
                      .strides("db", 2048, 256, 16));
    ASSERT_TRUE(dump.reduces);
    EXPECT_EQ(dump.reduceSymbol, REDUCE_SYMBOL);
    expectReduceKernarg(
        dump.reduceKernarg,
        {WORKSPACE, ptrOf(sdpa_bwd_uid::DBIAS), 2, 8, 16, 16, 1, 0, {2048, 256, 16}, {0, 256, 16}});
    // ceil(16 * ceil(16 / 8) / 256) blocks per plane, one plane per kept (batch, head).
    EXPECT_EQ(dump.reduceGrid, (std::array<unsigned int, 3>{1, 8, 1}));
    EXPECT_EQ(dump.reduceBlock, 256U);
    EXPECT_EQ(dump.workspaceBytes, 8192U);
}

TEST(TestFlydslSdpaBwdMergedReduce, SumsOverAHeadBroadcastBias)
{
    const auto dump = dumpOf(withBias(baseSpec(), true, {2, 1, 16, 16}), biasRow(), WORKSPACE);
    ASSERT_TRUE(dump.reduces);
    expectReduceKernarg(
        dump.reduceKernarg,
        {WORKSPACE, ptrOf(sdpa_bwd_uid::DBIAS), 2, 8, 16, 16, 0, 1, {2048, 256, 16}, {256, 0, 16}});
    EXPECT_EQ(dump.reduceGrid, (std::array<unsigned int, 3>{1, 2, 1}));
}

TEST(TestFlydslSdpaBwdMergedReduce, SumsOverABiasBroadcastOverBothAxes)
{
    const auto dump = dumpOf(withBias(baseSpec(), true, {1, 1, 16, 16}), biasRow(), WORKSPACE);
    ASSERT_TRUE(dump.reduces);
    expectReduceKernarg(
        dump.reduceKernarg,
        {WORKSPACE, ptrOf(sdpa_bwd_uid::DBIAS), 2, 8, 16, 16, 1, 1, {2048, 256, 16}, {0, 0, 16}});
    EXPECT_EQ(dump.reduceGrid, (std::array<unsigned int, 3>{1, 1, 1}));
}

TEST(TestFlydslSdpaBwdMergedReduce, CoversEveryVectorOfEveryRowOfALongerPlane)
{
    Shape tall;
    tall.seqQ = 64;
    tall.seqKv = 100;
    // Rows of 104 elements, so every bias object can stage the bias.
    const auto spec = specOf(tall).withEdit([](SdpaBwdGraph& g) {
        addBias(g, {1, 8, 64, 100}).strides = {8 * 64 * 104, 64 * 104, 104, 1};
        addDbias(g);
    });
    ASSERT_TRUE(serves(spec, biasRow()));
    const auto dump = dumpOf(spec, biasRow(), WORKSPACE);
    ASSERT_TRUE(dump.reduces);
    // 64 rows of ceil(100 / 8) = 13 vectors, 256 to a block; the workspace rows are 100 wide.
    EXPECT_EQ(dump.reduceGrid, (std::array<unsigned int, 3>{4, 8, 1}));
    const auto values = decode(dump.reduceKernarg, reduceSlots());
    EXPECT_EQ(values.at("stride_ws_seq"), 100U);
    EXPECT_EQ(values.at("stride_ws_head"), 6400U);
    EXPECT_EQ(values.at("stride_ws_batch"), 51200U);
    EXPECT_EQ(dump.workspaceBytes, size_t{2 * 8 * 64 * 100 * 2});
}

TEST(TestFlydslSdpaBwdMergedReduce, ReadsTheReduceSymbolOnlyForABroadcastDbias)
{
    auto unnamed = biasRow();
    unnamed.reduceSymbol = "";
    // Not broadcast: the object may name no reduce kernel.
    EXPECT_NO_THROW(static_cast<void>(dumpOf(withBias(baseSpec(), true), unnamed)));
    EXPECT_NO_THROW(
        static_cast<void>(dumpOf(withBias(baseSpec(), false, {1, 8, 16, 16}), unnamed)));

    // Broadcast: the symbol is the record's.
    auto renamed = biasRow();
    renamed.reduceSymbol = "another_reduce_symbol";
    const auto spec = withBias(baseSpec(), true, {1, 8, 16, 16});
    EXPECT_EQ(dumpOf(spec, biasRow(), WORKSPACE).reduceSymbol, REDUCE_SYMBOL);
    EXPECT_EQ(dumpOf(spec, renamed, WORKSPACE).reduceSymbol, "another_reduce_symbol");
}

TEST(TestFlydslSdpaBwdMergedReduce, RefusesABroadcastDbiasOnAnObjectWithoutAReduceSymbol)
{
    auto unnamed = biasRow();
    unnamed.reduceSymbol = "";
    auto absent = biasRow();
    const auto spec = withBias(baseSpec(), true, {1, 8, 16, 16});
    expectBadParam([&] { static_cast<void>(dumpOf(spec, unnamed, WORKSPACE)); });

    auto absentRecord = makeRow(absent);
    absentRecord.metadata.erase("reduce_symbol");
    const GraphFixture fixture(buildSdpaBwdGraph(spec));
    expectBadParam([&] {
        static_cast<void>(describeFlydslSdpaBwdMergedLaunchForTesting(
            fixture.context(), absentRecord, buffersFor(), asPointer(WORKSPACE)));
    });
    // prepare() resolves the symbol before it loads any code, so the refusal needs no archive.
    const auto bound = matchesGraph(MERGED, fixture.context());
    ASSERT_TRUE(bound.has_value());
    expectBadParam([&] {
        static_cast<void>(
            dispatchHandler(MERGED).prepare(fixture.context(), *bound, makeRow(unnamed)));
    });
}

// ---------------------------------------------------------------------------
// Seed and offset
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBwdMergedSeed, A64BitDeviceTensorIsReadInPlace)
{
    const auto spec = withDropout(baseSpec(), 0.2F, DataType::INT64, false);
    const auto dump = dumpOf(spec, dropoutRow());
    const auto values = decode(dump.kernarg, mergedSlots());
    EXPECT_EQ(values.at("philox_seed_ptr"), ptrOf(sdpa_bwd_uid::SEED));
    EXPECT_EQ(values.at("philox_offset1"), ptrOf(sdpa_bwd_uid::OFFSET));
    EXPECT_EQ(values.at("philox_offset2"), 0U);
    EXPECT_FALSE(dump.seedUploaded);
}

TEST(TestFlydslSdpaBwdMergedSeed, A32BitDeviceTensorIsCopiedIntoTheLowHalfOfAZeroedWord)
{
    const auto spec = withDropout(baseSpec(), 0.2F, DataType::INT32, false);
    const auto dump = dumpOf(spec, dropoutRow(), WORKSPACE);
    const auto values = decode(dump.kernarg, mergedSlots());
    EXPECT_EQ(values.at("philox_seed_ptr"), WORKSPACE);
    EXPECT_EQ(values.at("philox_offset1"), WORKSPACE + 8);
    EXPECT_TRUE(dump.seedUploaded);
    EXPECT_TRUE(dump.seedCopiedFromDevice);
    EXPECT_TRUE(dump.seedWords.empty());
}

TEST(TestFlydslSdpaBwdMergedSeed, APassByValueSeedIsWrittenAsFourWords)
{
    const auto bakedWords = [](DataType type, SdpaBwdTensorValue seed, SdpaBwdTensorValue offset) {
        const auto spec = withDropout(baseSpec(), 0.2F, type, true).withEdit([=](SdpaBwdGraph& g) {
            g.tensor("seed").value = seed;
            g.tensor("offset").value = offset;
        });
        const auto dump = dumpOf(spec, dropoutRow(), WORKSPACE);
        const auto values = decode(dump.kernarg, mergedSlots());
        EXPECT_EQ(values.at("philox_seed_ptr"), WORKSPACE);
        EXPECT_EQ(values.at("philox_offset1"), WORKSPACE + 8);
        EXPECT_TRUE(dump.seedUploaded);
        EXPECT_FALSE(dump.seedCopiedFromDevice);
        return dump.seedWords;
    };
    using Words = std::vector<uint32_t>;
    EXPECT_EQ(bakedWords(DataType::INT32, int32_t{7}, int32_t{9}), (Words{7, 0, 9, 0}));
    // A negative int32 is zero-extended: its unsigned bit pattern, then a zero high word.
    EXPECT_EQ(bakedWords(DataType::INT32, int32_t{-3}, int32_t{-1}),
              (Words{0xFFFFFFFDU, 0, 0xFFFFFFFFU, 0}));
    EXPECT_EQ(bakedWords(DataType::INT64, int64_t{0x100000005LL}, int64_t{-2}),
              (Words{5, 1, 0xFFFFFFFEU, 0xFFFFFFFFU}));
}

// ---------------------------------------------------------------------------
// dropout_threshold: the C++ port against philox.py
// ---------------------------------------------------------------------------

struct ThresholdCase
{
    float probability;
    int32_t threshold;
    double scale;
};

// The Python values, computed from a FlyDSL environment with
//
//   cd dnn-providers/hip-kernel-provider/flydsl/kernels_src
//   python -c '
//   import struct
//   from kernels.attention_bwd_cdna.philox import dropout_threshold
//   f32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
//   for p in (1e-7, 1e-3, 0.1, 0.2, 0.25, 0.3, 0.4999, 0.5, 0.7, 0.9, 0.999, 0.99999994):
//       q = f32(p)
//       print(p, dropout_threshold(q), repr(f32(1.0 / (1.0 - q))))'
//
// A graph carries p as a float, so each is rounded to f32 first; the scale is 1 / (1 - p)
// in double, rounded once to f32.
const std::vector<ThresholdCase> THRESHOLDS{{1e-7F, -2147483218, 1.0000001192092896},
                                            {1e-3F, -2143188680, 1.001001000404358},
                                            {0.1F, -1717986911, 1.1111111640930176},
                                            {0.2F, -1288490175, 1.25},
                                            {0.25F, -1073741823, 1.3333333730697632},
                                            {0.3F, -858993407, 1.4285714626312256},
                                            {0.4999F, -429439, 1.9996001720428467},
                                            {0.5F, 0, 2.0},
                                            {0.7F, 858993407, 3.3333332538604736},
                                            {0.9F, 1717986815, 9.999998092651367},
                                            {0.999F, 2143188735, 1000.0128784179688},
                                            {0.99999994F, 2147483391, 16777216.0}};

TEST(TestFlydslSdpaBwdMergedDropout, ThresholdAndScaleEqualThePythonValues)
{
    for(const auto& test : THRESHOLDS)
    {
        const auto dump = dumpOf(withDropout(baseSpec(), test.probability), dropoutRow());
        const auto values = decode(dump.kernarg, mergedSlots());
        EXPECT_EQ(values.at("idropout_p"), bits(test.threshold)) << test.probability;
        EXPECT_EQ(values.at("dropout_scale"), bits(static_cast<float>(test.scale)))
            << test.probability;
    }
}

TEST(TestFlydslSdpaBwdMergedDropout, WithoutDropoutThresholdIsZeroAndScaleOne)
{
    const auto values = decode(dumpOf(baseSpec(), plainRow()).kernarg, mergedSlots());
    EXPECT_EQ(values.at("idropout_p"), 0U);
    EXPECT_EQ(values.at("dropout_scale"), bits(1.0F));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
