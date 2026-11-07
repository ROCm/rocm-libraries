// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// How the bundle harness grades a half or bfloat16 convolution-forward output: the selection in
// IntegrationBundleVerificationHarness::compareAgainst() between the dynamic tolerance and the
// per-op default, and where the dynamic tolerance gets its input statistics.
//
// A failing comparison prints the atol and rtol it graded with, so the tests read the tolerance
// the harness actually handed to the comparison from that report. A TOML override is applied by
// gradingForTensor(); the harness reads it through a lookup a test can replace, because the
// TestConfig singleton of this binary is initialized once and without a settings file, so the
// override tests below load a settings file of their own. gradingForTensor() itself is covered in
// TestTomlGuards.cpp.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesConv.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/TestSettings.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/input-init/FillRecipe.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::conv::AbsRelTolerance;
using hipdnn_test_sdk::utilities::conv::ConvFpropErrorScale;

namespace
{

// x [1, 4, 6, 6], w [8, 4, 3, 3], y [1, 8, 4, 4]: K = 4 * 3 * 3 = 36 products per output element.
constexpr uint64_t K_ACCUMULATION = 36;
constexpr size_t K_X_ELEMS = 144;
constexpr size_t K_W_ELEMS = 288;
constexpr size_t K_OUT_ELEMS = 128;

constexpr int64_t X_UID = 0;
constexpr int64_t W_UID = 1;
constexpr int64_t Y_UID = 2;
constexpr int64_t SCALE_UID = 3;
constexpr int64_t OUT_UID = 4;
constexpr int64_t MUL_OUT_UID = 5;
constexpr int64_t BIAS_UID = 6;
constexpr int64_t ADD_OUT_UID = 7;

enum class Topology
{
    PLAIN, ///< conv -> y
    VARIANT3, ///< conv -> mul(y, scale) -> add(bias) -> relu -> out, as the Variant3 bundles
};

enum class ScaleSource
{
    BAKED, ///< the scale is a scalar constant stored in the graph
    FILLED, ///< the scale is a tensor filled from a recipe
};

struct Spec
{
    bool isHalf = true;
    Topology topology = Topology::PLAIN;
    ScaleSource scaleSource = ScaleSource::BAKED;
    double bakedValue = 0.5;
    std::string convCompute = "float";
    bool inputBlobs = false;

    int64_t outUid() const
    {
        return topology == Topology::PLAIN ? Y_UID : OUT_UID;
    }
};

std::string tensorJson(int64_t uid,
                       const char* name,
                       const std::string& dims,
                       const std::string& strides,
                       const std::string& dtype,
                       bool isVirtual,
                       const std::string& extra = {})
{
    std::ostringstream out;
    out << R"({"name": ")" << name << R"(", "uid": )" << uid << R"(, "dims": )" << dims
        << R"(, "strides": )" << strides << R"(, "data_type": ")" << dtype << R"(", "virtual": )"
        << (isVirtual ? "true" : "false") << extra << "}";
    return out.str();
}

std::string pointwiseJson(
    const char* name, const char* operation, int64_t in0, std::optional<int64_t> in1, int64_t out)
{
    std::ostringstream o;
    o << R"({"compute_data_type": "float", "inputs": {"axis_tensor_uid": null, )"
      << R"("elu_alpha": null, "in_0_tensor_uid": )" << in0 << R"(, "in_1_tensor_uid": )"
      << (in1.has_value() ? std::to_string(*in1) : std::string("null"))
      << R"(, "in_2_tensor_uid": null, "operation": ")" << operation
      << R"(", "relu_lower_clip": null, "relu_lower_clip_slope": null, )"
      << R"("relu_upper_clip": null, "softplus_beta": null, "swish_beta": null}, "name": ")" << name
      << R"(", "outputs": {"out_0_tensor_uid": )" << out << R"(}, "type": "PointwiseAttributes"})";
    return o.str();
}

std::string graphJson(const Spec& spec)
{
    const std::string dtype = spec.isHalf ? "half" : "bfloat16";
    const std::string yDims = "[1, 8, 4, 4]";
    const std::string yStrides = "[128, 16, 4, 1]";
    const bool plain = spec.topology == Topology::PLAIN;

    std::ostringstream convNode;
    convNode << R"({"compute_data_type": ")" << spec.convCompute
             << R"(", "inputs": {"w_tensor_uid": 1, "x_tensor_uid": 0}, "name": "conv", )"
             << R"("outputs": {"y_tensor_uid": 2}, "parameters": {"conv_mode": )"
             << R"("CROSS_CORRELATION", "dilation": [1, 1], "post_padding": [0, 0], )"
             << R"("pre_padding": [0, 0], "stride": [1, 1]}, "type": "ConvolutionFwdAttributes"})";

    std::vector<std::string> nodes{convNode.str()};
    std::vector<std::string> tensors{
        tensorJson(X_UID, "x", "[1, 4, 6, 6]", "[144, 36, 6, 1]", dtype, false),
        tensorJson(W_UID, "w", "[8, 4, 3, 3]", "[36, 9, 3, 1]", dtype, false),
        tensorJson(Y_UID, "y", yDims, yStrides, dtype, !plain),
    };
    if(!plain)
    {
        std::string scaleExtra;
        if(spec.scaleSource == ScaleSource::BAKED)
        {
            scaleExtra = std::string(R"(, "value": )") + std::to_string(spec.bakedValue)
                         + R"(, "value_type": ")" + (spec.isHalf ? "Float16Value" : "BFloat16Value")
                         + "\"";
        }
        nodes.push_back(pointwiseJson("mul", "mul", Y_UID, SCALE_UID, MUL_OUT_UID));
        nodes.push_back(pointwiseJson("add", "add", MUL_OUT_UID, BIAS_UID, ADD_OUT_UID));
        nodes.push_back(pointwiseJson("relu", "relu_fwd", ADD_OUT_UID, std::nullopt, OUT_UID));
        tensors.push_back(tensorJson(
            SCALE_UID, "scale", "[1, 1, 1, 1]", "[1, 1, 1, 1]", dtype, false, scaleExtra));
        tensors.push_back(tensorJson(OUT_UID, "out", yDims, yStrides, dtype, false));
        tensors.push_back(tensorJson(MUL_OUT_UID, "mul_out", yDims, yStrides, dtype, true));
        tensors.push_back(
            tensorJson(BIAS_UID, "bias", "[1, 8, 1, 1]", "[8, 1, 1, 1]", dtype, false));
        tensors.push_back(tensorJson(ADD_OUT_UID, "add_out", yDims, yStrides, dtype, true));
    }

    const auto join = [](const std::vector<std::string>& parts) {
        std::string joined;
        for(const auto& part : parts)
        {
            joined += joined.empty() ? part : ", " + part;
        }
        return joined;
    };
    std::ostringstream graph;
    graph << R"({"nodes": [)" << join(nodes) << R"(], "tensors": [)" << join(tensors)
          << R"(], "io_data_type": ")" << dtype << R"(", "compute_data_type": "float", )"
          << R"("intermediate_data_type": "float", "name": ""})";
    return graph.str();
}

void writeConstant(void* destination, size_t count, bool isHalf, float value)
{
    if(isHalf)
    {
        std::fill_n(static_cast<half*>(destination), count, static_cast<half>(value));
    }
    else
    {
        std::fill_n(static_cast<bfloat16*>(destination), count, static_cast<bfloat16>(value));
    }
}

template <typename T>
void writeBlob(const std::filesystem::path& path, size_t count, float value)
{
    const std::vector<T> data(count, static_cast<T>(value));
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(T)));
}

/// What a comparison reported.
struct Graded
{
    bool failed = false;
    std::string messages;

    /// The atol and rtol of the failure report, which are the ones the tensor was graded with.
    std::optional<AbsRelTolerance> reportedTolerance() const
    {
        const std::regex pattern(R"(Tolerance: atol=([^ ]+) rtol=([^\s]+))");
        std::smatch match;
        if(!std::regex_search(messages, match, pattern))
        {
            return std::nullopt;
        }
        return AbsRelTolerance{std::stod(match[1].str()), std::stod(match[2].str())};
    }
};

class TestConvFwdToleranceGradingFixture : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;
    testing_support::HarnessMocks _mocks;
    int _bundleCount = 0;
    std::optional<IntegrationBundleVerificationHarness::GradingLookup> _gradingLookup;

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(claimScratchDirectory("convtol"));
        _tempDir = _scopedDir->path();
    }

    /// Grades with the overrides of this settings text, applied the way the default lookup applies
    /// the TestConfig ones.
    void useTomlSettings(const std::string& toml)
    {
        const auto path = _tempDir / "settings.toml";
        std::ofstream(path) << toml;
        const auto settings = std::make_shared<TestSettings>(path);
        _gradingLookup
            = [settings](
                  const std::string& testName, const std::string& label, float atol, float rtol) {
                  return gradingForTensor(testName,
                                          label,
                                          settings->findValidatorOverride(testName, label),
                                          settings->findToleranceOverride(testName),
                                          atol,
                                          rtol);
              };
    }

    std::shared_ptr<IntegrationTestBundle> loadBundle(const Spec& spec)
    {
        const std::string name = "conv_tol_" + std::to_string(_bundleCount++);
        const auto dir = _tempDir / name;
        std::filesystem::create_directories(dir);
        std::ofstream(dir / (name + ".json")) << graphJson(spec);
        fixtures::writeMetadata(dir, name);
        if(spec.inputBlobs)
        {
            // Inputs only, so the bundle has no golden output and the reference grades it.
            const auto base = (dir / name).string();
            if(spec.isHalf)
            {
                writeBlob<half>(base + ".tensor0.bin", K_X_ELEMS, 2.0f);
                writeBlob<half>(base + ".tensor1.bin", K_W_ELEMS, 0.5f);
            }
            else
            {
                writeBlob<bfloat16>(base + ".tensor0.bin", K_X_ELEMS, 2.0f);
                writeBlob<bfloat16>(base + ".tensor1.bin", K_W_ELEMS, 0.5f);
            }
        }
        return fixtures::loadWrittenBundle(dir, name);
    }

    /// Runs the bundle with an engine that writes `engineValue` to every output element and a
    /// reference that writes `referenceValue`. `prepare` may pin fill recipes on the harness.
    template <typename Prepare>
    Graded grade(const Spec& spec, float engineValue, float referenceValue, Prepare&& prepare)
    {
        using ::testing::_;
        const bool isHalf = spec.isHalf;
        const int64_t outUid = spec.outUid();
        ON_CALL(_mocks.engineRunner, execute(_, _, _))
            .WillByDefault(
                [=](GraphSession&, const std::optional<LoadedEngine>&, VariantPack& pack) {
                    writeConstant(pack.at(outUid), K_OUT_ELEMS, isHalf, engineValue);
                    return EngineOpResult::succeeded();
                });
        for(auto* reference : {&_mocks.cpuReference, &_mocks.gpuReference})
        {
            ON_CALL(*reference, execute(_, _, _))
                .WillByDefault([=](void*, size_t, const VariantPack& pack) {
                    writeConstant(pack.at(outUid), K_OUT_ELEMS, isHalf, referenceValue);
                });
        }

        IntegrationBundleVerificationHarness harness(
            _mocks.dependencies(testing_support::hostPolicy(VerificationMode::AUTO)));
        harness.setBundle(loadBundle(spec), "conv-tolerance-test-bundle");
        if(_gradingLookup.has_value())
        {
            harness.setGradingLookup(*_gradingLookup);
        }
        prepare(harness.inputFillRecipes());

        ::testing::TestPartResultArray results;
        testing_support::driveHarness(harness, &results);
        return Graded{testing_support::anyFailed(results), testing_support::allMessages(results)};
    }

    Graded grade(const Spec& spec, float engineValue, float referenceValue)
    {
        return grade(spec, engineValue, referenceValue, [](InputFillRecipes&) {});
    }

    static AbsRelTolerance dynamicTolerance(bool isHalf, const ConvFpropErrorScale& scale)
    {
        return isHalf
                   ? hipdnn_test_sdk::utilities::conv::calculateConvFpropOutputTolerance<half>(
                         scale)
                   : hipdnn_test_sdk::utilities::conv::calculateConvFpropOutputTolerance<bfloat16>(
                         scale);
    }

    /// The scale of a problem whose x and w are both uniform on [-1, 1], the default fill.
    static ConvFpropErrorScale defaultFillScale(double outputScale = 1.0)
    {
        ConvFpropErrorScale scale;
        scale.accumulationLength = K_ACCUMULATION;
        scale.inputMeanSquare = 1.0 / 3.0;
        scale.weightMeanSquare = 1.0 / 3.0;
        scale.outputScale = outputScale;
        return scale;
    }

    /// An engine value so far from the reference that every tolerance fails it: the report then
    /// names the tolerance that graded it.
    static constexpr float K_REFERENCE = 8.0f;
    static constexpr float K_FAR_OFF = 64.0f;

    static void expectReported(const Graded& graded, const AbsRelTolerance& expected)
    {
        ASSERT_TRUE(graded.failed) << graded.messages;
        const auto reported = graded.reportedTolerance();
        ASSERT_TRUE(reported.has_value()) << graded.messages;
        // The report prints a float with six digits.
        EXPECT_NEAR(reported->atol, expected.atol, 1e-4 * expected.atol);
        EXPECT_NEAR(reported->rtol, expected.rtol, 1e-4 * expected.rtol);
    }
};

} // namespace

// ── The dynamic tolerance replaces the fixed default ────────────────────────

TEST_F(TestConvFwdToleranceGradingFixture, HalfOutputIsGradedWithTheHalfDynamicTolerance)
{
    Spec spec;
    spec.isHalf = true;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), dynamicTolerance(true, defaultFillScale()));
}

TEST_F(TestConvFwdToleranceGradingFixture, Bfloat16OutputIsGradedWithTheBfloat16DynamicTolerance)
{
    Spec spec;
    spec.isHalf = false;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE),
                   dynamicTolerance(false, defaultFillScale()));
}

// Half at |ref| = 8: the band is about 0.039 (atol 0.023 + rtol 0.002 * 8), the fixed default's
// 0.9. Four half steps (0.03125) are inside the dynamic band, eight (0.0625) outside it and far
// inside the fixed one.
TEST_F(TestConvFwdToleranceGradingFixture, HalfOutputInsideTheDynamicBandPassesAndOutsideFails)
{
    Spec spec;
    spec.isHalf = true;
    EXPECT_FALSE(grade(spec, K_REFERENCE + 0.03125f, K_REFERENCE).failed);
    EXPECT_TRUE(grade(spec, K_REFERENCE + 0.0625f, K_REFERENCE).failed);
}

// bfloat16 at |ref| = 8: the band is about 0.125, two bfloat16 steps (0.0625 each); the fixed
// default's 0.9 would pass four.
TEST_F(TestConvFwdToleranceGradingFixture, Bfloat16OutputInsideTheDynamicBandPassesAndOutsideFails)
{
    Spec spec;
    spec.isHalf = false;
    EXPECT_FALSE(grade(spec, K_REFERENCE + 0.0625f, K_REFERENCE).failed);
    EXPECT_TRUE(grade(spec, K_REFERENCE + 0.25f, K_REFERENCE).failed);
}

// A graph the model does not describe keeps the per-op default for both atol and rtol.
TEST_F(TestConvFwdToleranceGradingFixture, ConvolutionComputingInHalfKeepsTheDefaultTolerance)
{
    Spec spec;
    spec.convCompute = "half";
    const auto bundle = loadBundle(spec);
    const auto fixedDefault = static_cast<double>(tolerance::defaultTolerance(
        bundle->graphWrapper(), hipdnn_flatbuffers_sdk::data_objects::DataType::HALF));
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE),
                   AbsRelTolerance{fixedDefault, fixedDefault});
}

// ── Where the input statistics come from ────────────────────────────────────

TEST_F(TestConvFwdToleranceGradingFixture, FillRecipesGiveTheInputStatistics)
{
    const Spec spec;
    const auto graded = grade(spec, K_FAR_OFF, K_REFERENCE, [](InputFillRecipes& recipes) {
        recipes.set(X_UID, FillRecipe::free(-2.0f, 2.0f));
        recipes.set(W_UID, FillRecipe::fixed(0.5f));
    });
    ConvFpropErrorScale scale;
    scale.accumulationLength = K_ACCUMULATION;
    scale.inputMeanSquare = 4.0 / 3.0; // (lo^2 + lo*hi + hi^2) / 3 for [-2, 2]
    scale.weightMeanSquare = 0.25;
    expectReported(graded, dynamicTolerance(true, scale));
}

// The element type of a blob input is the tensor's own, not the output's or half's: the same
// constants as bfloat16 blobs give the same moments.
TEST_F(TestConvFwdToleranceGradingFixture, GoldenBlobInputsAreScannedOnTheHost)
{
    for(const bool isHalf : {true, false})
    {
        Spec spec;
        spec.isHalf = isHalf;
        spec.inputBlobs = true;
        ConvFpropErrorScale scale;
        scale.accumulationLength = K_ACCUMULATION;
        scale.inputMeanSquare = 4.0; // x is 2.0 everywhere
        scale.weightMeanSquare = 0.25; // w is 0.5 everywhere
        expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), dynamicTolerance(isHalf, scale));
    }
}

// ── A fused multiply: baked constant or filled tensor ───────────────────────

// The shipped Variant3 half and bfloat16 cases bake the scale as 0.5 and have no recipe for it:
// the band narrows by that factor, not the default fill's 1.
TEST_F(TestConvFwdToleranceGradingFixture, BakedScaleBelowOneNarrowsTheTolerance)
{
    for(const bool isHalf : {true, false})
    {
        Spec spec;
        spec.isHalf = isHalf;
        spec.topology = Topology::VARIANT3;
        spec.bakedValue = 0.5;
        expectReported(grade(spec, K_FAR_OFF, K_REFERENCE),
                       dynamicTolerance(isHalf, defaultFillScale(0.5)));
    }
}

// A baked constant above 1 widens it: reading the default fill's 1 instead would fail correct
// output near zero.
TEST_F(TestConvFwdToleranceGradingFixture, BakedScaleAboveOneWidensTheTolerance)
{
    Spec spec;
    spec.topology = Topology::VARIANT3;
    spec.bakedValue = 4.0;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE),
                   dynamicTolerance(true, defaultFillScale(4.0)));
}

// A negative constant scales by its magnitude.
TEST_F(TestConvFwdToleranceGradingFixture, BakedNegativeScaleCountsByItsMagnitude)
{
    Spec spec;
    spec.topology = Topology::VARIANT3;
    spec.bakedValue = -2.0;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE),
                   dynamicTolerance(true, defaultFillScale(2.0)));
}

// A scale that is a tensor takes its largest magnitude from the recipe that fills it: [-3, 1]
// peaks at 3.
TEST_F(TestConvFwdToleranceGradingFixture, FilledScaleTakesItsLargestMagnitudeFromTheRecipe)
{
    Spec spec;
    spec.topology = Topology::VARIANT3;
    spec.scaleSource = ScaleSource::FILLED;
    const auto graded = grade(spec, K_FAR_OFF, K_REFERENCE, [](InputFillRecipes& recipes) {
        recipes.set(SCALE_UID, FillRecipe::free(-3.0f, 1.0f));
    });
    expectReported(graded, dynamicTolerance(true, defaultFillScale(3.0)));
}

// The bias add and the relu of the Variant3 graph leave the scale alone: a baked 1 behaves as
// the plain graph does.
TEST_F(TestConvFwdToleranceGradingFixture, BiasAddAndReluDoNotChangeTheTolerance)
{
    Spec spec;
    spec.topology = Topology::VARIANT3;
    spec.bakedValue = 1.0;
    const auto graded = grade(spec, K_FAR_OFF, K_REFERENCE, [](InputFillRecipes& recipes) {
        recipes.set(BIAS_UID, FillRecipe::fixed(0.001f));
    });
    expectReported(graded, dynamicTolerance(true, defaultFillScale()));
}

// ── A TOML override replaces the dynamic tolerance ──────────────────────────

// [[tolerance_overrides]] outranks the dynamic tolerance: the entry's atol and rtol grade the
// tensor, and an engine value outside the dynamic band but inside the entry's passes.
TEST_F(TestConvFwdToleranceGradingFixture, TomlToleranceOverrideReplacesTheDynamicTolerance)
{
    useTomlSettings(R"([meta]
version = 1

[[tolerance_overrides]]
filters = ["TestConvFwdToleranceGradingFixture.TomlToleranceOverride*"]
atol = 0.5
rtol = 0.25
)");
    for(const bool isHalf : {true, false})
    {
        Spec spec;
        spec.isHalf = isHalf;
        expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), AbsRelTolerance{0.5, 0.25});
        // 0.25 is two bfloat16 steps and eight half steps at 8: outside either dynamic band.
        EXPECT_FALSE(grade(spec, K_REFERENCE + 0.25f, K_REFERENCE).failed)
            << (isHalf ? "half" : "bfloat16");
    }
}

// The same entry applies where the harness falls back to the per-op default.
TEST_F(TestConvFwdToleranceGradingFixture, TomlToleranceOverrideAlsoReplacesTheFixedDefault)
{
    useTomlSettings(R"([meta]
version = 1

[[tolerance_overrides]]
filters = ["TestConvFwdToleranceGradingFixture.TomlToleranceOverrideAlso*"]
atol = 0.5
rtol = 0.25
)");
    Spec spec;
    spec.convCompute = "half";
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), AbsRelTolerance{0.5, 0.25});
}

// An entry that matches no test leaves the dynamic tolerance alone.
TEST_F(TestConvFwdToleranceGradingFixture, TomlOverrideOfAnotherSuiteKeepsTheDynamicTolerance)
{
    useTomlSettings(R"([meta]
version = 1

[[tolerance_overrides]]
filters = ["SomeOtherSuite.*"]
atol = 0.5
rtol = 0.25
)");
    const Spec spec;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), dynamicTolerance(true, defaultFillScale()));
}

// [[validator_overrides]] outranks [[tolerance_overrides]]: an rms entry grades the tensor by its
// threshold, so a value that the entry's allclose tolerance would pass fails.
TEST_F(TestConvFwdToleranceGradingFixture, TomlRmsValidatorOutranksTheToleranceOverride)
{
    useTomlSettings(R"([meta]
version = 1

[[tolerance_overrides]]
filters = ["TestConvFwdToleranceGradingFixture.TomlRmsValidator*"]
atol = 0.5
rtol = 0.25

[[validator_overrides]]
filters = ["TestConvFwdToleranceGradingFixture.TomlRmsValidator*"]
tensors = ["*"]
validator = "rms"
rms_threshold = 1e-4
)");
    const Spec spec;
    EXPECT_TRUE(grade(spec, K_REFERENCE + 0.25f, K_REFERENCE).failed);
}

// NOLINTEND(readability-identifier-naming)
