// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledLaunch.hpp>

#include "GraphPatternTestSupport.hpp"

/// @file TestCompiledLaunch.cpp
/// @brief A UDD's declarative single launch: compilation, evaluation against its value
///        rules, the per-engine reference check, and the saved form of a resolved launch.

namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_plugin_sdk::HipdnnPluginException;
using hipdnn_plugin_sdk::ingestor::testing::BINARY_POINTWISE_PATTERN;
using hipdnn_plugin_sdk::ingestor::testing::compilePatternText;
using hipdnn_plugin_sdk::ingestor::testing::PatternGraphBuilder;

constexpr const char* WHERE = "UDD in test.udd.json";
constexpr int64_t TWO_POW_53 = int64_t{1} << 53;
constexpr int64_t U32_MAX = std::numeric_limits<uint32_t>::max();

/// The smallest valid declarative UDD.
nlohmann::json minimalUdd()
{
    return nlohmann::json::parse(R"({"version": "1.0", "id": "00000000-0000-0000-0000-000000000001",
        "name": "test.launch", "grid": {"x": 1, "y": 1, "z": 1},
        "block": {"x": 64, "y": 1, "z": 1}, "args_signature": []})");
}

/// minimalUdd() with the JSON merge patch @p patch applied; a null removes a key.
nlohmann::json patchedUdd(const std::string& patch)
{
    auto udd = minimalUdd();
    udd.merge_patch(nlohmann::json::parse(patch));
    return udd;
}

std::shared_ptr<const CompiledLaunch> compile(const nlohmann::json& udd)
{
    return compileLaunch(udd, "test.launch", WHERE);
}

/// The message compileLaunch() refuses @p udd with, or "" when it compiles.
std::string compileError(const nlohmann::json& udd)
{
    try
    {
        compile(udd);
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        return error.what();
    }
    return {};
}

/// Publishes a tensor the way publishTensor() names its tokens.
void publish(BoundTokens& bound,
             const std::string& root,
             int64_t uid,
             const std::vector<int64_t>& dims,
             const std::vector<int64_t>& strides)
{
    bound[root] = uid;
    bound[tensorField(root, "uid")] = uid;
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        bound[tensorElement(root, "dims", axis)] = dims[axis];
        bound[tensorElement(root, "strides", axis)] = strides[axis];
    }
}

/// A forward convolution binding: x 2x3x8x8, w 4x3x3x3, y 2x4x6x6, all NCHW-packed.
BoundTokens convBinding()
{
    BoundTokens bound;
    publish(bound, "x", 11, {2, 3, 8, 8}, {192, 64, 8, 1});
    publish(bound, "w", 12, {4, 3, 3, 3}, {27, 9, 3, 1});
    publish(bound, "y", 13, {2, 4, 6, 6}, {144, 36, 6, 1});
    return bound;
}

KernelDefinition convKernel()
{
    KernelDefinition kernel;
    kernel.metadata = {{"block_size", int64_t{64}}, {"tile", std::vector<int64_t>{16, 8}}};
    return kernel;
}

DeviceProperties reportedDevice()
{
    DeviceProperties properties;
    properties.warpSize = 32;
    return properties;
}

/// Resolves @p launch for a throwaway graph on @p properties.
ResolvedLaunch resolveOn(const CompiledLaunch& launch,
                         const BoundTokens& bound,
                         const KernelDefinition& kernel,
                         const DeviceProperties& properties)
{
    const auto graph = PatternGraphBuilder().tensor(1).build();
    const auto base = graph.context();
    const MatchContext ctx{base.graph, 0, properties};
    return launch.resolve(ctx, bound, kernel);
}

size_t workspaceOn(const CompiledLaunch& launch,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel,
                   const DeviceProperties& properties)
{
    const auto graph = PatternGraphBuilder().tensor(1).build();
    const auto base = graph.context();
    const MatchContext ctx{base.graph, 0, properties};
    return launch.workspaceBytes(ctx, bound, kernel);
}

/// A conv-like launch over x, w and y that reads the binding, $kernel and $device.
const char* const CONV_UDD = R"({"version": "1.0", "id": "00000000-0000-0000-0000-000000000002",
    "name": "conv.declarative",
    "grid": {"x": {"ceil_div": [{"*": ["$y.dims[0]", "$y.dims[1]", "$y.dims[2]", "$y.dims[3]"]},
                                "$kernel.block_size"]},
             "y": 1, "z": "$kernel.tile[1]"},
    "block": {"x": "$kernel.block_size", "y": 1, "z": 1},
    "shared_mem_bytes": {"*": ["$kernel.tile[0]", 4]},
    "workspace_bytes": {"*": ["$x.dims[0]", "$device.warp_size", 8]},
    "args_signature": [
        {"name": "x", "kind": "pointer", "source": {"from": "tensor", "ref": "$x"}},
        {"name": "w", "kind": "pointer", "source": {"from": "tensor", "ref": "$w"}},
        {"name": "y", "kind": "pointer", "source": {"from": "tensor", "ref": "$y"}},
        {"name": "ws", "kind": "workspace"},
        {"name": "n", "kind": "scalar", "type": "i32", "source": {"from": "dim", "ref": "$x", "axis": 0}},
        {"name": "k", "kind": "scalar", "type": "i32", "source": {"from": "dim", "ref": "$w", "axis": 0}},
        {"name": "x_stride_c", "kind": "scalar", "type": "i64",
         "source": {"from": "stride", "ref": "$x", "axis": 1}},
        {"name": "out_h_last", "kind": "scalar", "type": "u32",
         "source": {"from": "expr", "expr": {"-": ["$y.dims[2]", 1]}}},
        {"name": "half_c", "kind": "scalar", "type": "f32",
         "source": {"from": "expr", "expr": {"/": ["$x.dims[1]", 2]}}}]})";

ResolvedLaunchArgument pointer(int64_t uid)
{
    ResolvedLaunchArgument argument;
    argument.kind = LaunchArgumentKind::POINTER;
    argument.tensorUid = uid;
    return argument;
}

template <typename T>
ResolvedLaunchArgument scalar(LaunchScalarType type, T value)
{
    ResolvedLaunchArgument argument;
    argument.kind = LaunchArgumentKind::SCALAR;
    argument.scalarType = type;
    std::memcpy(argument.bytes.data(), &value, sizeof(value));
    return argument;
}

ResolvedLaunch expectedConvLaunch()
{
    ResolvedLaunchArgument workspace;
    workspace.kind = LaunchArgumentKind::WORKSPACE;

    ResolvedLaunch launch;
    // 2 * 4 * 6 * 6 = 288 outputs over 64 threads.
    launch.grid = {5, 1, 8};
    launch.block = {64, 1, 1};
    launch.sharedMemBytes = 64;
    launch.workspaceBytes = size_t{2} * 32 * 8;
    launch.arguments = {pointer(11),
                        pointer(12),
                        pointer(13),
                        workspace,
                        scalar(LaunchScalarType::I32, int32_t{2}),
                        scalar(LaunchScalarType::I32, int32_t{4}),
                        scalar(LaunchScalarType::I64, int64_t{64}),
                        scalar(LaunchScalarType::U32, uint32_t{5}),
                        scalar(LaunchScalarType::F32, 1.5F)};
    return launch;
}

MetadataSchema schemaWith(std::vector<MetadataField> fields)
{
    MetadataSchema schema;
    schema.name = "test schema";
    schema.fields = std::move(fields);
    return schema;
}

} // namespace

TEST(TestCompiledLaunch, ResolvesAConvLaunchOverTheBindingKernelAndDevice)
{
    const auto launch = compile(nlohmann::json::parse(CONV_UDD));
    const auto resolved = resolveOn(*launch, convBinding(), convKernel(), reportedDevice());

    EXPECT_EQ(resolved, expectedConvLaunch());
    EXPECT_EQ(workspaceOn(*launch, convBinding(), convKernel(), reportedDevice()), 512U);
}

TEST(TestCompiledLaunch, KeepsTheDeclarativeBodyAndDefaultsTheOptionalSizesToZero)
{
    const auto launch = compile(minimalUdd());
    auto body = minimalUdd();
    body.erase("version");
    body.erase("id");
    body.erase("name");

    EXPECT_EQ(launch->source(), body);
    EXPECT_EQ(launch->name(), "test.launch");
    const auto resolved = resolveOn(*launch, {}, convKernel(), reportedDevice());
    EXPECT_EQ(resolved.sharedMemBytes, 0U);
    EXPECT_EQ(resolved.workspaceBytes, 0U);
}

TEST(TestCompiledLaunch, AnExtensionKeyDoesNotChangeTheDeclarativeBody)
{
    // The body is what descriptor equality and the selector revision compare, so an
    // annotation alone must not make two copies of one UDD differ.
    const auto plain = compile(patchedUdd(R"({"args_signature": [
        {"name": "n", "kind": "scalar", "type": "i32", "source": {"from": "expr", "expr": 1}}]})"));
    const auto annotated = compile(patchedUdd(R"({"x-note": "tuned", "_rev": 2, "provenance": {},
        "grid": {"x-why": 1},
        "args_signature": [{"name": "n", "kind": "scalar", "type": "i32", "_doc": "count",
            "source": {"from": "expr", "expr": 1, "x-origin": "manual"}}]})"));

    EXPECT_EQ(annotated->source(), plain->source());
}

TEST(TestCompiledLaunch, ReportsEveryReferenceIncludingTheSourceSugar)
{
    const auto launch = compile(nlohmann::json::parse(CONV_UDD));

    EXPECT_EQ(launch->references(),
              (std::vector<std::string>{"device.warp_size",
                                        "kernel.block_size",
                                        "kernel.tile[0]",
                                        "kernel.tile[1]",
                                        "w",
                                        "w.dims[0]",
                                        "x",
                                        "x.dims[0]",
                                        "x.dims[1]",
                                        "x.strides[1]",
                                        "y",
                                        "y.dims[0]",
                                        "y.dims[1]",
                                        "y.dims[2]",
                                        "y.dims[3]"}));
}

TEST(TestCompiledLaunch, DerivesTheKernargSignatureFromTheArgumentKinds)
{
    const auto launch = compile(nlohmann::json::parse(CONV_UDD));
    const KernelArgument buffer{"global_buffer", 8, 0, ""};

    EXPECT_EQ(launch->signature(),
              (std::vector<KernelArgument>{buffer,
                                           buffer,
                                           buffer,
                                           buffer,
                                           {"by_value", 4, 0, ""},
                                           {"by_value", 4, 0, ""},
                                           {"by_value", 8, 0, ""},
                                           {"by_value", 4, 0, ""},
                                           {"by_value", 4, 0, ""}}));
}

struct CompileRefusalCase
{
    const char* name;
    /// A JSON merge patch over minimalUdd().
    const char* patch;
    const char* expected;
};

class TestCompiledLaunchRefusal : public ::testing::TestWithParam<CompileRefusalCase>
{
};

/// Each case is a static error, so it is refused when the UDD is read, naming the UDD.
TEST_P(TestCompiledLaunchRefusal, RefusesTheUddAtCompile)
{
    const auto message = compileError(patchedUdd(GetParam().patch));

    EXPECT_NE(message.find(GetParam().expected), std::string::npos) << message;
    EXPECT_NE(message.find(WHERE), std::string::npos) << message;
}

INSTANTIATE_TEST_SUITE_P(
    Udds,
    TestCompiledLaunchRefusal,
    ::testing::Values(
        CompileRefusalCase{"MissingGrid", R"({"grid": null})", "missing required key 'grid'"},
        CompileRefusalCase{"MissingBlock", R"({"block": null})", "missing required key 'block'"},
        CompileRefusalCase{"MissingArgsSignature",
                           R"({"args_signature": null})",
                           "missing required key 'args_signature'"},
        CompileRefusalCase{
            "MissingGridAxis", R"({"grid": {"z": null}})", "missing required key 'z'"},
        CompileRefusalCase{
            "MissingBlockAxis", R"({"block": {"x": null}})", "missing required key 'x'"},
        CompileRefusalCase{"UnknownRootKey", R"({"grids": 1})", "unknown key 'grids'"},
        CompileRefusalCase{"UnknownAxisKey", R"({"block": {"w": 1}})", "unknown key 'w'"},
        CompileRefusalCase{"GridNotAnObject", R"({"grid": 4})", "must be a JSON object"},
        CompileRefusalCase{"BothArms", R"({"dispatch_symbol": "test.native"})", "dispatch_symbol"},
        CompileRefusalCase{"Launches", R"({"launches": []})", "'launches' in"},
        CompileRefusalCase{"Intermediates", R"({"intermediates": {}})", "is not supported yet"},
        CompileRefusalCase{"ArgsNotAnArray", R"({"args_signature": {}})", "must be an array"},
        CompileRefusalCase{"ExplicitOffset",
                           R"({"args_signature": [{"name": "x", "kind": "pointer", "offset": 0,
                               "source": {"from": "tensor", "ref": "$x"}}]})",
                           "is not supported yet"},
        CompileRefusalCase{"ScalarSource",
                           R"({"args_signature": [{"name": "a", "kind": "scalar", "type": "f32",
                               "source": {"from": "scalar", "ref": "$alpha"}}]})",
                           "source 'scalar' in argument 'a' of UDD in test.udd.json is not "
                           "supported yet"},
        CompileRefusalCase{"AttributeSource",
                           R"({"args_signature": [{"name": "a", "kind": "scalar", "type": "i32",
                               "source": {"from": "attribute", "ref": "$conv.pad"}}]})",
                           "source 'attribute' in argument 'a'"},
        CompileRefusalCase{"WorkspaceSource",
                           R"({"args_signature": [{"name": "a", "kind": "pointer",
                               "source": {"from": "workspace"}}]})",
                           "unknown source 'workspace'"},
        CompileRefusalCase{"UnknownArgumentKey",
                           R"({"args_signature": [{"name": "x", "kind": "pointer", "size": 8,
                               "source": {"from": "tensor", "ref": "$x"}}]})",
                           "unknown key 'size'"},
        CompileRefusalCase{"MissingArgumentName",
                           R"({"args_signature": [{"kind": "workspace"}]})",
                           "missing required key 'name' in args_signature[0]"},
        CompileRefusalCase{"EmptyArgumentName",
                           R"({"args_signature": [{"name": "", "kind": "workspace"}]})",
                           "must not be empty"},
        CompileRefusalCase{"DuplicateArgumentName",
                           R"({"args_signature": [{"name": "a", "kind": "workspace"},
                               {"name": "a", "kind": "pointer",
                                "source": {"from": "tensor", "ref": "$x"}}]})",
                           "argument name 'a' is listed twice"},
        CompileRefusalCase{"UnknownKind",
                           R"({"args_signature": [{"name": "a", "kind": "buffer"}]})",
                           "'kind' in args_signature[0]"},
        CompileRefusalCase{"UnsupportedScalarType",
                           R"({"args_signature": [{"name": "a", "kind": "scalar", "type": "f16",
                               "source": {"from": "expr", "expr": 1}}]})",
                           "'f16'"},
        CompileRefusalCase{"ScalarWithoutType",
                           R"({"args_signature": [{"name": "a", "kind": "scalar",
                               "source": {"from": "expr", "expr": 1}}]})",
                           "missing required key 'type'"},
        CompileRefusalCase{
            "ScalarWithoutSource",
            R"({"args_signature": [{"name": "a", "kind": "scalar", "type": "i32"}]})",
            "missing required key 'source'"},
        CompileRefusalCase{"PointerWithType",
                           R"({"args_signature": [{"name": "x", "kind": "pointer", "type": "i64",
                               "source": {"from": "tensor", "ref": "$x"}}]})",
                           "takes no 'type'"},
        CompileRefusalCase{"WorkspaceWithSource",
                           R"({"args_signature": [{"name": "ws", "kind": "workspace",
                               "source": {"from": "expr", "expr": 1}}]})",
                           "takes no 'source'"},
        CompileRefusalCase{"PointerFromDim",
                           R"({"args_signature": [{"name": "x", "kind": "pointer",
                               "source": {"from": "dim", "ref": "$x", "axis": 0}}]})",
                           "its source must be 'tensor'"},
        CompileRefusalCase{"ScalarFromTensor",
                           R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "i64",
                               "source": {"from": "tensor", "ref": "$x"}}]})",
                           "not 'tensor'"},
        CompileRefusalCase{"AxisOnATensorSource",
                           R"({"args_signature": [{"name": "x", "kind": "pointer",
                               "source": {"from": "tensor", "ref": "$x", "axis": 0}}]})",
                           "unknown key 'axis'"},
        CompileRefusalCase{"RefOnAnExprSource",
                           R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "i32",
                               "source": {"from": "expr", "expr": 1, "ref": "$x"}}]})",
                           "unknown key 'ref'"},
        CompileRefusalCase{"MissingAxis",
                           R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "i32",
                               "source": {"from": "dim", "ref": "$x"}}]})",
                           "missing required key 'axis'"},
        CompileRefusalCase{"NegativeAxis",
                           R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "i32",
                               "source": {"from": "stride", "ref": "$x", "axis": -1}}]})",
                           "must be a non-negative integer"},
        CompileRefusalCase{"FractionalAxis",
                           R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "i32",
                               "source": {"from": "dim", "ref": "$x", "axis": 1.5}}]})",
                           "must be a non-negative integer"},
        CompileRefusalCase{"RefWithoutSigil",
                           R"({"args_signature": [{"name": "x", "kind": "pointer",
                               "source": {"from": "tensor", "ref": "x"}}]})",
                           "must name a pattern variable"},
        CompileRefusalCase{"EscapedRef",
                           R"({"args_signature": [{"name": "x", "kind": "pointer",
                               "source": {"from": "tensor", "ref": "$$x"}}]})",
                           "must name a pattern variable"},
        CompileRefusalCase{"RefToATensorField",
                           R"({"args_signature": [{"name": "x", "kind": "pointer",
                               "source": {"from": "tensor", "ref": "$x.uid"}}]})",
                           "must name a pattern variable"},
        CompileRefusalCase{"StringLiteralFormula",
                           R"({"grid": {"x": "64"}})",
                           "grid.x of UDD in test.udd.json must be an expression object"},
        CompileRefusalCase{"BoolLiteralFormula", R"({"block": {"y": true}})", "block.y of"},
        CompileRefusalCase{"EscapedStringFormula", R"({"block": {"y": "$$n"}})", "block.y of"},
        CompileRefusalCase{"ExpressionTheLanguageRejects",
                           R"({"workspace_bytes": {"hipdnn.custom": [1]}})",
                           "workspace_bytes of UDD in test.udd.json does not compile"},
        CompileRefusalCase{"ZeroGridLiteral", R"({"grid": {"y": 0}})", "grid.y of"},
        CompileRefusalCase{"BlockLiteralAboveUint32",
                           R"({"block": {"x": 4294967296}})",
                           "block.x of UDD in test.udd.json is 4294967296"},
        CompileRefusalCase{
            "NegativeSharedMemLiteral", R"({"shared_mem_bytes": -1})", "shared_mem_bytes of"},
        CompileRefusalCase{
            "FractionalWorkspaceLiteral", R"({"workspace_bytes": 1.5})", "workspace_bytes of"},
        CompileRefusalCase{"WorkspaceLiteralBeyondUint64",
                           R"({"workspace_bytes": 18446744073709551615})",
                           "workspace_bytes of"},
        CompileRefusalCase{"I32LiteralOverflow",
                           R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "i32",
                               "source": {"from": "expr", "expr": 2147483648}}]})",
                           "argument 'n' of"},
        CompileRefusalCase{"U32LiteralNegative",
                           R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "u32",
                               "source": {"from": "expr", "expr": -1}}]})",
                           "argument 'n' of"},
        CompileRefusalCase{"F32LiteralOutsideTheFloatRange",
                           R"({"args_signature": [{"name": "a", "kind": "scalar", "type": "f32",
                               "source": {"from": "expr", "expr": 1e39}}]})",
                           "argument 'a' of"}),
    [](const ::testing::TestParamInfo<CompileRefusalCase>& info) { return info.param.name; });

TEST(TestCompiledLaunch, RefusesARootThatIsNotAnObject)
{
    EXPECT_NE(compileError(nlohmann::json::array()).find("must be a JSON object"),
              std::string::npos);
}

TEST(TestCompiledLaunch, AcceptsExtensionKeysAndLiteralsOnTheirBoundaries)
{
    // Every literal sits on the edge of its rule, so each must compile.
    EXPECT_EQ(compileError(patchedUdd(R"({"x-note": 1, "grid": {"x": 4294967295},
        "shared_mem_bytes": 4294967295, "workspace_bytes": 9007199254740992,
        "args_signature": [
            {"name": "a", "kind": "scalar", "type": "i32", "source": {"from": "expr", "expr": -2147483648}},
            {"name": "b", "kind": "scalar", "type": "f32", "source": {"from": "expr", "expr": 0.1}}]})")),
              "");
}

struct RangeCase
{
    const char* name;
    /// A JSON merge patch over minimalUdd() that reads `$v` into one field.
    const char* patch;
    /// The value bound to `v`; nullopt leaves it unbound, so the formula reads null.
    std::optional<MetadataValue> value;
};

class TestCompiledLaunchRange : public ::testing::TestWithParam<RangeCase>
{
};

/// A value that breaks its rule only once a graph is bound throws BAD_PARAM, which the
/// plan walks read as "this kernel does not apply", and names the UDD and the field.
TEST_P(TestCompiledLaunchRange, ThrowsBadParamNamingTheField)
{
    const auto launch = compile(patchedUdd(GetParam().patch));
    BoundTokens bound;
    if(GetParam().value.has_value())
    {
        bound["v"] = *GetParam().value;
    }

    try
    {
        resolveOn(*launch, bound, convKernel(), reportedDevice());
        FAIL() << "resolve accepted the value";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
        EXPECT_EQ(std::string(error.what()).rfind("test.launch: ", 0), 0U) << error.what();
        EXPECT_NE(std::string(error.what()).find(" evaluated to "), std::string::npos)
            << error.what();
    }
}

namespace
{
const char* const I32_ARG = R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "i32",
    "source": {"from": "expr", "expr": "$v"}}]})";
const char* const U32_ARG = R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "u32",
    "source": {"from": "expr", "expr": "$v"}}]})";
const char* const I64_ARG = R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "i64",
    "source": {"from": "expr", "expr": "$v"}}]})";
const char* const F32_ARG = R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "f32",
    "source": {"from": "expr", "expr": "$v"}}]})";
} // namespace

INSTANTIATE_TEST_SUITE_P(
    Values,
    TestCompiledLaunchRange,
    ::testing::Values(
        RangeCase{"GridZero", R"({"grid": {"x": "$v"}})", MetadataValue{int64_t{0}}},
        RangeCase{"GridAboveUint32", R"({"grid": {"x": "$v"}})", MetadataValue{U32_MAX + 1}},
        RangeCase{"GridNonIntegral", R"({"grid": {"z": "$v"}})", MetadataValue{2.5}},
        RangeCase{"GridFromAnAbsentToken", R"({"grid": {"x": "$v"}})", std::nullopt},
        RangeCase{"GridFromAString", R"({"grid": {"x": "$v"}})", MetadataValue{std::string("8")}},
        RangeCase{"GridFromABool", R"({"grid": {"x": "$v"}})", MetadataValue{true}},
        RangeCase{
            "GridFromAList", R"({"grid": {"x": "$v"}})", MetadataValue{std::vector<int64_t>{4}}},
        RangeCase{"BlockZero", R"({"block": {"y": "$v"}})", MetadataValue{int64_t{0}}},
        RangeCase{"SharedMemNegative", R"({"shared_mem_bytes": "$v"})", MetadataValue{int64_t{-1}}},
        RangeCase{
            "SharedMemAboveUint32", R"({"shared_mem_bytes": "$v"})", MetadataValue{U32_MAX + 1}},
        RangeCase{"WorkspaceAboveTwoPow53",
                  R"({"workspace_bytes": "$v"})",
                  MetadataValue{TWO_POW_53 + 1}},
        RangeCase{"WorkspaceNegative", R"({"workspace_bytes": "$v"})", MetadataValue{int64_t{-8}}},
        RangeCase{"I32AboveRange", I32_ARG, MetadataValue{int64_t{2147483648}}},
        RangeCase{"I32BelowRange", I32_ARG, MetadataValue{int64_t{-2147483649}}},
        RangeCase{"I32NonIntegral", I32_ARG, MetadataValue{0.5}},
        RangeCase{"U32Negative", U32_ARG, MetadataValue{int64_t{-1}}},
        RangeCase{"U32AboveRange", U32_ARG, MetadataValue{U32_MAX + 1}},
        RangeCase{"I64AboveTwoPow53", I64_ARG, MetadataValue{TWO_POW_53 + 1}},
        RangeCase{"I64BelowMinusTwoPow53", I64_ARG, MetadataValue{-TWO_POW_53 - 1}},
        RangeCase{"F32OutsideTheFloatRange", F32_ARG, MetadataValue{1e39}},
        RangeCase{"F32FromAnAbsentToken", F32_ARG, std::nullopt}),
    [](const ::testing::TestParamInfo<RangeCase>& info) { return info.param.name; });

TEST(TestCompiledLaunch, AcceptsEveryValueOnTheEdgeOfItsRule)
{
    const auto launch = compile(patchedUdd(R"({
        "grid": {"x": "$grid"}, "block": {"y": "$one"},
        "shared_mem_bytes": "$grid", "workspace_bytes": "$workspace",
        "args_signature": [
            {"name": "a", "kind": "scalar", "type": "i32", "source": {"from": "expr", "expr": "$i32"}},
            {"name": "b", "kind": "scalar", "type": "u32", "source": {"from": "expr", "expr": "$grid"}},
            {"name": "c", "kind": "scalar", "type": "i64", "source": {"from": "expr", "expr": "$i64"}},
            {"name": "d", "kind": "scalar", "type": "f32", "source": {"from": "expr", "expr": "$f32"}},
            {"name": "e", "kind": "scalar", "type": "i32", "source": {"from": "expr", "expr": "$whole"}}]})"));
    const BoundTokens bound{{"grid", U32_MAX},
                            {"one", int64_t{1}},
                            {"workspace", TWO_POW_53},
                            {"i32", int64_t{std::numeric_limits<int32_t>::min()}},
                            {"i64", -TWO_POW_53},
                            {"f32", 0.1},
                            // A whole double is an integer.
                            {"whole", 7.0}};

    const auto resolved = resolveOn(*launch, bound, convKernel(), reportedDevice());

    EXPECT_EQ(resolved.grid[0], std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(resolved.block[1], 1U);
    EXPECT_EQ(resolved.sharedMemBytes, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(resolved.workspaceBytes, static_cast<size_t>(TWO_POW_53));
    ASSERT_EQ(resolved.arguments.size(), 5U);
    EXPECT_EQ(resolved.arguments[0],
              scalar(LaunchScalarType::I32, std::numeric_limits<int32_t>::min()));
    EXPECT_EQ(resolved.arguments[1],
              scalar(LaunchScalarType::U32, std::numeric_limits<uint32_t>::max()));
    EXPECT_EQ(resolved.arguments[2], scalar(LaunchScalarType::I64, -TWO_POW_53));
    EXPECT_EQ(resolved.arguments[3], scalar(LaunchScalarType::F32, 0.1F));
    EXPECT_EQ(resolved.arguments[4], scalar(LaunchScalarType::I32, int32_t{7}));
}

TEST(TestCompiledLaunch, ADeviceFactTheDeviceDidNotReportThrowsBadParam)
{
    // An unreported warp size reads null, not 0, so the workspace has no value.
    const auto launch = compile(nlohmann::json::parse(CONV_UDD));
    try
    {
        workspaceOn(*launch, convBinding(), convKernel(), DeviceProperties{});
        FAIL() << "workspaceBytes accepted an unreported device fact";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
        EXPECT_NE(std::string(error.what()).find("workspace_bytes evaluated to null"),
                  std::string::npos)
            << error.what();
    }
}

TEST(TestCompiledLaunch, APointerToATensorWithoutABoundUidThrowsBadParam)
{
    // An optional operand the graph leaves out publishes no token at all.
    const auto launch = compile(patchedUdd(R"({"args_signature": [
        {"name": "bias", "kind": "pointer", "source": {"from": "tensor", "ref": "$bias"}}]})"));
    try
    {
        resolveOn(*launch, convBinding(), convKernel(), reportedDevice());
        FAIL() << "resolve accepted an unbound tensor";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
        EXPECT_NE(std::string(error.what()).find("'bias'"), std::string::npos) << error.what();
    }
}

TEST(TestCompiledLaunch, EveryReferenceToWhatThePatternPublishesResolves)
{
    const GraphMatch pattern{compilePatternText(BINARY_POINTWISE_PATTERN)};
    const auto schema = schemaWith({{"block_size", MetadataType::INT, std::nullopt}});
    const auto launch = compile(patchedUdd(R"({
        "grid": {"x": {"ceil_div": ["$output.dims[0]", "$kernel.block_size"]}},
        "block": {"x": "$kernel.block_size"},
        "workspace_bytes": {"*": ["$device.cu_count", "$graph.node_count"]},
        "args_signature": [
            {"name": "a", "kind": "pointer", "source": {"from": "tensor", "ref": "$input_a"}},
            {"name": "b", "kind": "pointer", "source": {"from": "tensor", "ref": "$input_b"}},
            {"name": "c", "kind": "pointer", "source": {"from": "tensor", "ref": "$output"}},
            {"name": "ws", "kind": "workspace"},
            {"name": "n", "kind": "scalar", "type": "i64",
             "source": {"from": "stride", "ref": "$input_a", "axis": 2}}]})"));

    EXPECT_EQ(launchReferenceError(*launch, pattern, schema), "");
}

TEST(TestCompiledLaunch, RefusesATensorSourceThePatternDoesNotBind)
{
    const GraphMatch pattern{compilePatternText(BINARY_POINTWISE_PATTERN)};
    const auto schema = schemaWith({});

    EXPECT_EQ(launchReferenceError(*compile(patchedUdd(R"({"args_signature": [
                  {"name": "w", "kind": "pointer", "source": {"from": "tensor", "ref": "$weights"}}]})")),
                                   pattern,
                                   schema),
              "binds argument 'w' to $weights, which is not a tensor the engine's graph_match "
              "binds");
    EXPECT_NE(launchReferenceError(*compile(patchedUdd(R"({"args_signature": [
                  {"name": "k", "kind": "scalar", "type": "i32",
                   "source": {"from": "dim", "ref": "$weights", "axis": 0}}]})")),
                                   pattern,
                                   schema)
                  .find("binds argument 'k' to $weights"),
              std::string::npos);
}

TEST(TestCompiledLaunch, RefusesAFormulaReferenceThePairDoesNotPublish)
{
    const GraphMatch pattern{compilePatternText(BINARY_POINTWISE_PATTERN)};
    const auto schema = schemaWith({{"block_size", MetadataType::INT, std::nullopt}});
    const auto errorFor = [&](const char* patch) {
        return launchReferenceError(*compile(patchedUdd(patch)), pattern, schema);
    };

    EXPECT_EQ(errorFor(R"({"block": {"x": "$kernel.tile_n"}})"),
              "reads $kernel.tile_n, which metadata schema 'test schema' does not declare");
    EXPECT_EQ(errorFor(R"({"grid": {"x": "$input_a.layout"}})"),
              "reads $input_a.layout, but tensor 'input_a' publishes no such field");
    EXPECT_EQ(errorFor(R"({"workspace_bytes": "$device.lds_sz"})"),
              "reads $device.lds_sz, which is not a device property");
    EXPECT_EQ(errorFor(R"({"args_signature": [{"name": "n", "kind": "scalar", "type": "i32",
                  "source": {"from": "expr", "expr": "$weights.dims[0]"}}]})"),
              "reads $weights.dims[0], which the engine's graph_match does not publish");
}

TEST(TestCompiledLaunch, ANativeGraphMatchChecksOnlyKernelAndDeviceReferences)
{
    // A native match publishes its tensors at runtime, so no tensor name is known at load.
    const GraphMatch native{NativeGraphMatch{"test.native"}};
    const auto schema = schemaWith({{"block_size", MetadataType::INT, std::nullopt}});
    const auto pointerTo = R"({"block": {"x": "$kernel.block_size"}, "args_signature": [
        {"name": "x", "kind": "pointer", "source": {"from": "tensor", "ref": "$anything"}},
        {"name": "n", "kind": "scalar", "type": "i32",
         "source": {"from": "dim", "ref": "$anything", "axis": 3}}]})";

    EXPECT_EQ(launchReferenceError(*compile(patchedUdd(pointerTo)), native, schema), "");
    EXPECT_EQ(launchReferenceError(
                  *compile(patchedUdd(R"({"block": {"x": "$kernel.tile"}})")), native, schema),
              "reads $kernel.tile, which metadata schema 'test schema' does not declare");
}

TEST(TestCompiledLaunch, AnEngineWithoutGraphMatchBindsNoTensor)
{
    const GraphMatch none{};
    const auto schema = schemaWith({{"block_size", MetadataType::INT, std::nullopt}});

    EXPECT_EQ(launchReferenceError(*compile(patchedUdd(R"({"block": {"x": "$kernel.block_size"},
                  "workspace_bytes": "$device.lds_size",
                  "args_signature": [{"name": "ws", "kind": "workspace"}]})")),
                                   none,
                                   schema),
              "");
    EXPECT_EQ(launchReferenceError(*compile(patchedUdd(R"({"args_signature": [
                  {"name": "x", "kind": "pointer", "source": {"from": "tensor", "ref": "$x"}}]})")),
                                   none,
                                   schema),
              "binds argument 'x' to $x, but the engine declares no graph_match and so binds no "
              "tensor");
    EXPECT_NE(launchReferenceError(
                  *compile(patchedUdd(R"({"grid": {"x": "$graph.node_count"}})")), none, schema)
                  .find("reads $graph.node_count, but the engine declares no graph_match"),
              std::string::npos);
}

TEST(TestCompiledLaunch, ASavedLaunchRestoresToTheSameLaunch)
{
    const auto launch = expectedConvLaunch();

    EXPECT_EQ(restoreResolvedLaunch(saveResolvedLaunch(launch)), launch);

    // Negative and extreme scalars keep their exact bits through the int64 payload.
    ResolvedLaunch edges;
    edges.arguments = {scalar(LaunchScalarType::I32, int32_t{-7}),
                       scalar(LaunchScalarType::U32, std::numeric_limits<uint32_t>::max()),
                       scalar(LaunchScalarType::I64, -TWO_POW_53),
                       scalar(LaunchScalarType::F32, -0.1F)};
    EXPECT_EQ(restoreResolvedLaunch(saveResolvedLaunch(edges)), edges);
}

struct MalformedPayloadCase
{
    const char* name;
    const char* key;
    /// The replacement value of `key`; nullopt removes it.
    std::optional<MetadataValue> value;
};

class TestCompiledLaunchMalformedPayload : public ::testing::TestWithParam<MalformedPayloadCase>
{
};

/// A saved plan is read off disk, so every field is checked before the launch uses it.
TEST_P(TestCompiledLaunchMalformedPayload, IsRefused)
{
    auto values = saveResolvedLaunch(expectedConvLaunch());
    if(GetParam().value.has_value())
    {
        values[GetParam().key] = *GetParam().value;
    }
    else
    {
        values.erase(GetParam().key);
    }

    try
    {
        restoreResolvedLaunch(values);
        FAIL() << "restore accepted the payload";
    }
    catch(const HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_NE(std::string(error.what()).find("saved declarative launch"), std::string::npos);
    }
}

namespace
{
/// The argument lists of expectedConvLaunch() with entry @p index of @p list replaced.
MetadataValue convList(const char* list, size_t index, int64_t value)
{
    auto values = saveResolvedLaunch(expectedConvLaunch());
    auto entries = std::get<std::vector<int64_t>>(values.at(list));
    entries[index] = value;
    return entries;
}
} // namespace

INSTANTIATE_TEST_SUITE_P(
    Payloads,
    TestCompiledLaunchMalformedPayload,
    ::testing::Values(
        MalformedPayloadCase{"MissingGrid", "grid", std::nullopt},
        MalformedPayloadCase{"MissingArgumentValues", "argument_values", std::nullopt},
        MalformedPayloadCase{"UnknownKey", "launches", MetadataValue{int64_t{1}}},
        MalformedPayloadCase{"GridNotAList", "grid", MetadataValue{int64_t{1}}},
        MalformedPayloadCase{"GridOfTwoAxes", "grid", MetadataValue{std::vector<int64_t>{1, 1}}},
        MalformedPayloadCase{
            "ZeroBlockAxis", "block", MetadataValue{std::vector<int64_t>{64, 0, 1}}},
        MalformedPayloadCase{"SharedMemNotAnInteger", "shared_mem_bytes", MetadataValue{1.0}},
        MalformedPayloadCase{
            "SharedMemAboveUint32", "shared_mem_bytes", MetadataValue{U32_MAX + 1}},
        MalformedPayloadCase{"NegativeWorkspace", "workspace_bytes", MetadataValue{int64_t{-1}}},
        MalformedPayloadCase{"ListsOfDifferentLengths",
                             "argument_scalar_types",
                             MetadataValue{std::vector<int64_t>{0}}},
        MalformedPayloadCase{"UnknownKind", "argument_kinds", convList("argument_kinds", 0, 3)},
        MalformedPayloadCase{
            "UnknownScalarType", "argument_scalar_types", convList("argument_scalar_types", 4, 4)},
        MalformedPayloadCase{"PointerWithAScalarType",
                             "argument_scalar_types",
                             convList("argument_scalar_types", 0, 2)},
        MalformedPayloadCase{
            "WorkspaceWithAValue", "argument_values", convList("argument_values", 3, 8)},
        MalformedPayloadCase{
            "I32OutOfRange", "argument_values", convList("argument_values", 4, int64_t{1} << 31)},
        MalformedPayloadCase{"U32Negative", "argument_values", convList("argument_values", 7, -1)},
        // 0x7fc00000 is a quiet NaN.
        MalformedPayloadCase{
            "F32NotFinite", "argument_values", convList("argument_values", 8, 0x7fc00000)}),
    [](const ::testing::TestParamInfo<MalformedPayloadCase>& info) { return info.param.name; });

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
