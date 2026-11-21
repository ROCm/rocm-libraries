// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// How the bundle harness grades a half, bfloat16 or float convolution output: the selection in
// IntegrationBundleVerificationHarness::compareAgainst() between the dynamic tolerance and the
// per-op default, where the dynamic tolerance gets its input statistics, and which engine table
// ([conv_fwd_tolerance], [conv_dgrad_tolerance], [conv_wrw_tolerance]) sets its coefficients.
//
// A failing comparison prints the atol and rtol it graded with, so the tests read the tolerance
// the harness actually handed to the comparison from that report. A TOML override is applied by
// gradingForTensor() to the settings of the harness policy. Production fills that from TestConfig,
// which this binary initializes once and without a settings file: most override tests below put a
// settings file of their own in the policy, and the engine-file tests build a TestConfig of their
// own from a file and take the policy's settings from productionPolicy(), as a run does.
// gradingForTensor() itself is covered in TestTomlGuards.cpp.

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
#include "harness/TestConfig.hpp"
#include "harness/TestSettings.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/ProductionPolicy.hpp"
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
    VARIANT2, ///< conv -> add(bias) -> relu -> out, as develop's Variant2 bundles
    TWO_BRANCHES, ///< conv -> y; mul(y, scale) -> one output; add(y, bias) -> another output
};

enum class ScaleSource
{
    BAKED, ///< the scale is a scalar constant stored in the graph
    FILLED, ///< the scale is a tensor filled from a recipe
};

enum class Direction
{
    FWD,
    DGRAD, ///< x is the output, from dy and w
    WRW, ///< w is the output, from x and dy
};

struct Spec
{
    bool isHalf = true;
    bool isFloat = false; ///< a float graph; isHalf then does not matter
    Topology topology = Topology::PLAIN;
    ScaleSource scaleSource = ScaleSource::BAKED;
    double bakedValue = 0.5;
    std::string mulLabel = "mul_out"; ///< the name of the multiply's output of TWO_BRANCHES
    std::string addLabel = "add_out"; ///< the name of the add's output of TWO_BRANCHES
    std::string convCompute = "float";
    std::string intermediateType = "float"; ///< the type of every virtual tensor
    std::string pointwiseCompute = "float"; ///< the compute type of every pointwise node
    bool inputBlobs = false;
    std::string inputType; ///< the type of x and w; empty: the graph's own
    /// A plain graph of the other directions (topology and scale are for forward only).
    Direction direction = Direction::FWD;

    std::vector<int64_t> outputUids() const
    {
        switch(direction)
        {
        case Direction::DGRAD:
            return {X_UID};
        case Direction::WRW:
            return {W_UID};
        default:
            break;
        }
        switch(topology)
        {
        case Topology::PLAIN:
            return {Y_UID};
        case Topology::TWO_BRANCHES:
            return {MUL_OUT_UID, ADD_OUT_UID};
        default:
            return {OUT_UID};
        }
    }

    /// The elements of each output.
    size_t outputElems() const
    {
        switch(direction)
        {
        case Direction::DGRAD:
            return K_X_ELEMS;
        case Direction::WRW:
            return K_W_ELEMS;
        default:
            return K_OUT_ELEMS;
        }
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

std::string pointwiseJson(const char* name,
                          const char* operation,
                          int64_t in0,
                          std::optional<int64_t> in1,
                          int64_t out,
                          const std::string& computeType)
{
    std::ostringstream o;
    o << R"({"compute_data_type": ")" << computeType << R"(", "inputs": {"axis_tensor_uid": null, )"
      << R"("elu_alpha": null, "in_0_tensor_uid": )" << in0 << R"(, "in_1_tensor_uid": )"
      << (in1.has_value() ? std::to_string(*in1) : std::string("null"))
      << R"(, "in_2_tensor_uid": null, "operation": ")" << operation
      << R"(", "relu_lower_clip": null, "relu_lower_clip_slope": null, )"
      << R"("relu_upper_clip": null, "softplus_beta": null, "swish_beta": null}, "name": ")" << name
      << R"(", "outputs": {"out_0_tensor_uid": )" << out << R"(}, "type": "PointwiseAttributes"})";
    return o.str();
}

/// A plain convolution backward-data or backward-weights graph on the tensors of the forward one:
/// x [1, 4, 6, 6], w [8, 4, 3, 3] and dy [1, 8, 4, 4] (the forward y). The output is x (dgrad) or
/// w (wrw), and the other two are the inputs.
std::string backwardGraphJson(const Spec& spec)
{
    std::string dtype = spec.isHalf ? "half" : "bfloat16";
    if(spec.isFloat)
    {
        dtype = "float";
    }
    const bool dgrad = spec.direction == Direction::DGRAD;
    std::ostringstream convNode;
    convNode << R"({"compute_data_type": "float", "inputs": )"
             << (dgrad ? R"({"dy_tensor_uid": 2, "w_tensor_uid": 1}, "name": "dgrad", )"
                         R"("outputs": {"dx_tensor_uid": 0})"
                       : R"({"dy_tensor_uid": 2, "x_tensor_uid": 0}, "name": "wrw", )"
                         R"("outputs": {"dw_tensor_uid": 1})")
             << R"(, "parameters": {"conv_mode": "CROSS_CORRELATION", "dilation": [1, 1], )"
             << R"("post_padding": [0, 0], "pre_padding": [0, 0], "stride": [1, 1]}, "type": ")"
             << (dgrad ? "ConvolutionBwdAttributes" : "ConvolutionWrwAttributes") << R"("})";
    const std::vector<std::string> tensors{
        tensorJson(X_UID, dgrad ? "dx" : "x", "[1, 4, 6, 6]", "[144, 36, 6, 1]", dtype, false),
        tensorJson(W_UID, dgrad ? "w" : "dw", "[8, 4, 3, 3]", "[36, 9, 3, 1]", dtype, false),
        tensorJson(Y_UID, "dy", "[1, 8, 4, 4]", "[128, 16, 4, 1]", dtype, false),
    };
    std::ostringstream graph;
    graph << R"({"nodes": [)" << convNode.str() << R"(], "tensors": [)" << tensors[0] << ", "
          << tensors[1] << ", " << tensors[2] << R"(], "io_data_type": ")" << dtype
          << R"(", "compute_data_type": "float", "intermediate_data_type": "float", )"
          << R"("name": ""})";
    return graph.str();
}

std::string graphJson(const Spec& spec)
{
    if(spec.direction != Direction::FWD)
    {
        return backwardGraphJson(spec);
    }
    std::string dtype = spec.isHalf ? "half" : "bfloat16";
    if(spec.isFloat)
    {
        dtype = "float";
    }
    const std::string inputDtype = spec.inputType.empty() ? dtype : spec.inputType;
    const std::string yDims = "[1, 8, 4, 4]";
    const std::string yStrides = "[128, 16, 4, 1]";
    const bool plain = spec.topology == Topology::PLAIN;
    const bool topology3 = spec.topology == Topology::VARIANT3;
    const bool twoBranches = spec.topology == Topology::TWO_BRANCHES;

    std::ostringstream convNode;
    convNode << R"({"compute_data_type": ")" << spec.convCompute
             << R"(", "inputs": {"w_tensor_uid": 1, "x_tensor_uid": 0}, "name": "conv", )"
             << R"("outputs": {"y_tensor_uid": 2}, "parameters": {"conv_mode": )"
             << R"("CROSS_CORRELATION", "dilation": [1, 1], "post_padding": [0, 0], )"
             << R"("pre_padding": [0, 0], "stride": [1, 1]}, "type": "ConvolutionFwdAttributes"})";

    std::vector<std::string> nodes{convNode.str()};
    std::vector<std::string> tensors{
        tensorJson(X_UID, "x", "[1, 4, 6, 6]", "[144, 36, 6, 1]", inputDtype, false),
        tensorJson(W_UID, "w", "[8, 4, 3, 3]", "[36, 9, 3, 1]", inputDtype, false),
        tensorJson(Y_UID, "y", yDims, yStrides, plain ? dtype : spec.intermediateType, !plain),
    };
    std::string scaleExtra;
    if(spec.scaleSource == ScaleSource::BAKED)
    {
        std::string valueType = "BFloat16Value";
        if(spec.isFloat)
        {
            valueType = "Float32Value";
        }
        else if(spec.isHalf)
        {
            valueType = "Float16Value";
        }
        scaleExtra = std::string(R"(, "value": )") + std::to_string(spec.bakedValue)
                     + R"(, "value_type": ")" + valueType + "\"";
    }
    if(topology3)
    {
        nodes.push_back(
            pointwiseJson("mul", "mul", Y_UID, SCALE_UID, MUL_OUT_UID, spec.pointwiseCompute));
        nodes.push_back(
            pointwiseJson("add", "add", MUL_OUT_UID, BIAS_UID, ADD_OUT_UID, spec.pointwiseCompute));
        nodes.push_back(pointwiseJson(
            "relu", "relu_fwd", ADD_OUT_UID, std::nullopt, OUT_UID, spec.pointwiseCompute));
        tensors.push_back(tensorJson(
            SCALE_UID, "scale", "[1, 1, 1, 1]", "[1, 1, 1, 1]", dtype, false, scaleExtra));
        tensors.push_back(
            tensorJson(MUL_OUT_UID, "mul_out", yDims, yStrides, spec.intermediateType, true));
    }
    else if(twoBranches)
    {
        nodes.push_back(
            pointwiseJson("mul", "mul", Y_UID, SCALE_UID, MUL_OUT_UID, spec.pointwiseCompute));
        nodes.push_back(
            pointwiseJson("add", "add", Y_UID, BIAS_UID, ADD_OUT_UID, spec.pointwiseCompute));
        tensors.push_back(tensorJson(
            SCALE_UID, "scale", "[1, 1, 1, 1]", "[1, 1, 1, 1]", dtype, false, scaleExtra));
        tensors.push_back(
            tensorJson(MUL_OUT_UID, spec.mulLabel.c_str(), yDims, yStrides, dtype, false));
        tensors.push_back(
            tensorJson(BIAS_UID, "bias", "[1, 8, 1, 1]", "[8, 1, 1, 1]", dtype, false));
        tensors.push_back(
            tensorJson(ADD_OUT_UID, spec.addLabel.c_str(), yDims, yStrides, dtype, false));
    }
    else if(!plain)
    {
        nodes.push_back(
            pointwiseJson("add", "add", Y_UID, BIAS_UID, ADD_OUT_UID, spec.pointwiseCompute));
        nodes.push_back(
            pointwiseJson("relu", "relu_fwd", ADD_OUT_UID, std::nullopt, OUT_UID, "float"));
    }
    if(!plain && !twoBranches)
    {
        tensors.push_back(tensorJson(OUT_UID, "out", yDims, yStrides, dtype, false));
        tensors.push_back(
            tensorJson(BIAS_UID, "bias", "[1, 8, 1, 1]", "[8, 1, 1, 1]", dtype, false));
        tensors.push_back(
            tensorJson(ADD_OUT_UID, "add_out", yDims, yStrides, spec.intermediateType, true));
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

void writeConstant(void* destination, size_t count, bool isHalf, bool isFloat, float value)
{
    if(isFloat)
    {
        std::fill_n(static_cast<float*>(destination), count, value);
    }
    else if(isHalf)
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

    /// The same, of the output with this uid, when the comparison reported several outputs.
    std::optional<AbsRelTolerance> reportedToleranceOf(int64_t uid) const
    {
        const std::regex pattern(
            R"(Tensor: [^\n]*\(UID )" + std::to_string(uid)
            + R"(, output\)\n[^\n]*\n  Tolerance: atol=([^ ]+) rtol=([^\s]+))");
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
    std::optional<TestSettings> _settings;
    std::unique_ptr<TestConfig> _engineConfig;
    const TestSettings* _policySettings = nullptr;
    /// The device the harness policy says it runs on (the raw gcnArchName).
    std::string _arch = "gfx942:sramecc+:xnack-";

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(claimScratchDirectory("convtol"));
        _tempDir = _scopedDir->path();
    }

    /// Grades with the overrides of this settings text, which the harness policy carries the way
    /// production carries the TestConfig ones.
    void useTomlSettings(const std::string& toml)
    {
        const auto path = _tempDir / "settings.toml";
        std::ofstream(path) << toml;
        _settings.emplace(path);
        _policySettings = &*_settings;
    }

    /// Grades with the settings of an engine file the way a run gets them: a TestConfig loads the
    /// file its options name, and productionPolicy() carries what it loaded into the policy.
    void useEngineFile(const std::string& toml)
    {
        const auto path = _tempDir / "engine.toml";
        std::ofstream(path) << toml;
        TestConfigOptions options;
        options.configPath = path;
        _engineConfig = std::make_unique<TestConfig>(std::move(options));
        _policySettings = productionPolicy(TensorPlacement::HOST, *_engineConfig).testSettings;
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
            if(spec.inputType == "float" || spec.isFloat)
            {
                writeBlob<float>(base + ".tensor0.bin", K_X_ELEMS, 2.0f);
                writeBlob<float>(base + ".tensor1.bin", K_W_ELEMS, 0.5f);
            }
            else if(spec.isHalf)
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
        const bool isFloat = spec.isFloat;
        const auto outUids = spec.outputUids();
        const auto outElems = spec.outputElems();
        testing_support::engineWrites(
            _mocks.engineRunner,
            [=](VariantPack& pack, float value) {
                for(const int64_t outUid : outUids)
                {
                    writeConstant(pack.at(outUid), outElems, isHalf, isFloat, value);
                }
            },
            engineValue);
        for(auto* reference : {&_mocks.cpuReference, &_mocks.gpuReference})
        {
            ON_CALL(*reference, execute(_, _, _))
                .WillByDefault([=](void*, size_t, const VariantPack& pack) {
                    for(const int64_t outUid : outUids)
                    {
                        writeConstant(pack.at(outUid), outElems, isHalf, isFloat, referenceValue);
                    }
                });
        }

        auto policy = testing_support::hostPolicy(VerificationMode::AUTO);
        policy.testSettings = _policySettings;
        policy.arch = _arch;
        IntegrationBundleVerificationHarness harness(_mocks.dependencies(std::move(policy)));
        harness.setBundle(loadBundle(spec), "conv-tolerance-test-bundle");
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

    // The negative check at a long reduction (see HalfOutputJustOutsideTheComputedBand...).
    static constexpr float K_LONG_X_BOUND = 16.0f;
    static constexpr float K_LONG_W_BOUND = 1.41421356f; // sqrt(2)
    static constexpr double K_LONG_REDUCTION = 18432.0; // A of the longest conv-fwd test-plan case

    /// E[v^2] of a draw uniform on [-bound, bound].
    static double uniformMeanSquare(float bound)
    {
        return static_cast<double>(bound) * static_cast<double>(bound) / 3.0;
    }

    static void widenFills(InputFillRecipes& recipes)
    {
        recipes.set(X_UID, FillRecipe::free(-K_LONG_X_BOUND, K_LONG_X_BOUND));
        recipes.set(W_UID, FillRecipe::free(-K_LONG_W_BOUND, K_LONG_W_BOUND));
    }

    static ConvFpropErrorScale longReductionScale()
    {
        ConvFpropErrorScale scale;
        scale.accumulationLength = K_ACCUMULATION;
        scale.inputMeanSquare = uniformMeanSquare(K_LONG_X_BOUND);
        scale.weightMeanSquare = uniformMeanSquare(K_LONG_W_BOUND);
        return scale;
    }

    /// The band of the model in tolerances.md, written out here and not taken from the library:
    /// 2 * eps * |ref| + (sqrt(K) * u32 + lambda * eps) * sigma + minNormal.
    static double documentedBand(bool isHalf, double sigma, double reference)
    {
        const double eps = std::ldexp(1.0, isHalf ? -10 : -7);
        const double lambda = isHalf ? 3.0 : 0.0;
        const double minNormal = std::ldexp(1.0, isHalf ? -14 : -126);
        const double u32 = std::ldexp(1.0, -24);
        const double atol
            = (std::sqrt(static_cast<double>(K_ACCUMULATION)) * u32 + lambda * eps) * sigma
              + minNormal;
        return atol + 2.0 * eps * std::abs(reference);
    }

    /// The value the output type stores for `value`.
    static double stored(bool isHalf, double value)
    {
        const auto asFloat = static_cast<float>(value);
        return isHalf ? static_cast<double>(static_cast<float>(static_cast<half>(asFloat)))
                      : static_cast<double>(static_cast<float>(static_cast<bfloat16>(asFloat)));
    }

    /// An engine value 0.95 of the computed band from the reference passes, one 1.05 of it away
    /// fails, on either side of the reference.
    void expectBandIsTheLimit(bool isHalf, double reference)
    {
        SCOPED_TRACE(::testing::Message()
                     << (isHalf ? "half" : "bfloat16") << ", reference " << reference);
        const auto scale = longReductionScale();
        const double sigma = std::sqrt(static_cast<double>(K_ACCUMULATION) * scale.inputMeanSquare
                                       * scale.weightMeanSquare);
        ASSERT_NEAR(sigma, std::sqrt(K_LONG_REDUCTION / 9.0), 1e-6 * sigma);

        const auto tolerance = dynamicTolerance(isHalf, scale);
        const double band = tolerance.atol + tolerance.rtol * std::abs(reference);
        ASSERT_NEAR(band, documentedBand(isHalf, sigma, reference), 1e-9 * band);
        ASSERT_EQ(stored(isHalf, reference), reference) << "pick a reference the type stores";

        Spec spec;
        spec.isHalf = isHalf;
        for(const double direction : {1.0, -1.0})
        {
            for(const bool outside : {false, true})
            {
                const double engine
                    = stored(isHalf, reference + direction * (outside ? 1.05 : 0.95) * band);
                const double distance = std::abs(engine - reference);
                ASSERT_TRUE(outside ? distance > band : distance < band)
                    << "rounding to the output type moved " << engine << " across the band "
                    << band;
                const auto graded = grade(
                    spec, static_cast<float>(engine), static_cast<float>(reference), widenFills);
                EXPECT_EQ(graded.failed, outside)
                    << "engine " << engine << " is " << distance / band << " bands from "
                    << reference << "\n"
                    << graded.messages;
            }
        }
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

// Half at |ref| = 8: the band is about 0.022 (atol 0.006 + rtol 0.002 * 8), the fixed default's
// 0.9. Two half steps (0.0156) are inside the dynamic band, four (0.03125) outside it and far
// inside the fixed one.
TEST_F(TestConvFwdToleranceGradingFixture, HalfOutputInsideTheDynamicBandPassesAndOutsideFails)
{
    Spec spec;
    spec.isHalf = true;
    EXPECT_FALSE(grade(spec, K_REFERENCE + 0.015625f, K_REFERENCE).failed);
    EXPECT_TRUE(grade(spec, K_REFERENCE + 0.03125f, K_REFERENCE).failed);
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

// ── The band is the limit: just outside fails, just inside passes ───────────

// The graph here has K = 36, so the fills are widened until sigma, the typical output magnitude,
// is that of A = 18,432 products (the longest reduction of the conv-fwd test plan) over inputs
// uniform on [-1, 1]: sigma^2 = K * E[x^2] * E[w^2] = 36 * 256/3 * 2/3 = 2048. The half atol is
// then about 0.53, five times the fixed 0.1 that the dynamic tolerance replaces.
//
// The engine value is the reference moved by 1.05 and by 0.95 times the band that
// calculateConvFpropOutputTolerance() computes, to the nearest value of the output type, so the
// test follows the formula instead of fixed steps. Rounding to the output type moves a value by
// up to half a step, so the test asserts that each value kept the side of the band it was picked
// for. The band is also compared with the model written out from tolerances.md: both ends of the
// test use the one function, so without that comparison a change that widens or narrows it
// moves the limit and the values together and nothing fails.
TEST_F(TestConvFwdToleranceGradingFixture, HalfOutputJustOutsideTheComputedBandFailsAtLongReduction)
{
    for(const double reference : {0.0, 10.0})
    {
        expectBandIsTheLimit(true, reference);
    }
}

// bfloat16's atol is tiny next to its rtol at this sigma (about 1.6e-5), so the reference 0 grades
// the summation-order term alone and the reference 10 the rtol. 10 sits at 1.25 times a power of
// two, where the band is two and a half steps: a whole number of steps could not be told on
// either side of it.
TEST_F(TestConvFwdToleranceGradingFixture,
       Bfloat16OutputJustOutsideTheComputedBandFailsAtLongReduction)
{
    for(const double reference : {0.0, 10.0})
    {
        expectBandIsTheLimit(false, reference);
    }
}

// A graph the model does not describe keeps the per-op default for both atol and rtol.
TEST_F(TestConvFwdToleranceGradingFixture, ConvolutionComputingInHalfKeepsTheDefaultTolerance)
{
    Spec spec;
    spec.convCompute = "half";
    const auto bundle = loadBundle(spec);
    const auto fixedDefault = tolerance::defaultTolerance(
        bundle->graphWrapper(), hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE),
                   AbsRelTolerance{static_cast<double>(fixedDefault.atol),
                                   static_cast<double>(fixedDefault.rtol)});
}

// The Variant3 graph with every intermediate stored in the output type: the convolution's
// result y, the multiply's result and the add's result are each rounded to it (K = 36, x and w
// uniform on [-1, 1] so sigma = 2, the scale 0.5, the bias uniform on [-1, 1]). The points are
// written out here from the model of tolerances.md, not taken from the library: eps * magnitude
// * slope with y (2 * 0.5: the multiply scales its error), the multiply's result (1) and the
// add's result (sqrt(1 + 1/3)), which the relu passes on as it is.
TEST_F(TestConvFwdToleranceGradingFixture,
       IntermediatesStoredInTheOutputTypeAreGradedWithTheirPoints)
{
    for(const bool isHalf : {true, false})
    {
        Spec spec;
        spec.isHalf = isHalf;
        spec.topology = Topology::VARIANT3;
        spec.intermediateType = isHalf ? "half" : "bfloat16";
        SCOPED_TRACE(spec.intermediateType);
        const double eps = std::ldexp(1.0, isHalf ? -10 : -7);
        auto scale = defaultFillScale(0.5);
        scale.roundingPoints = {{eps, 2.0, 0.5}, {eps, 1.0, 1.0}, {eps, std::sqrt(4.0 / 3.0), 1.0}};
        expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), dynamicTolerance(isHalf, scale));
    }
}

// The same graph with float tensors and pointwise nodes computing in the output type: every
// operand that is float is converted and rounded, as is every result stored in another type
// than the node computes in. The scale and the bias are stored in the output type and need no
// rounding; the graph output is rounded once, by the output type.
TEST_F(TestConvFwdToleranceGradingFixture,
       PointwiseNodesComputingInTheOutputTypeAreGradedWithTheirPoints)
{
    for(const bool isHalf : {true, false})
    {
        Spec spec;
        spec.isHalf = isHalf;
        spec.topology = Topology::VARIANT3;
        spec.pointwiseCompute = isHalf ? "half" : "bfloat16";
        SCOPED_TRACE(spec.pointwiseCompute);
        const double eps = std::ldexp(1.0, isHalf ? -10 : -7);
        const double afterAdd = std::sqrt(4.0 / 3.0);
        auto scale = defaultFillScale(0.5);
        scale.roundingPoints = {{eps, 2.0, 0.5}, // y, an operand of the multiply
                                {eps, 1.0, 1.0}, // the multiply's result
                                {eps, 1.0, 1.0}, // the multiply's result, an operand of the add
                                {eps, afterAdd, 1.0}, // the add's result
                                {eps, afterAdd, 1.0}}; // the add's result, an operand of the relu
        expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), dynamicTolerance(isHalf, scale));
    }
}

// develop's Variant2 graph: the convolution's result and the add's result are float tensors, the
// add computes in the output's type through its compute type, and the relu computes in float.
// The add converts the convolution's result (sigma = 2) to the type and rounds its own result
// (sqrt(sigma^2 + 1/3), the bias uniform on [-1, 1]): two points, which the CI failures of the
// bfloat16 cases at four rounding steps from the reference needed.
TEST_F(TestConvFwdToleranceGradingFixture, AddComputingInTheOutputTypeOverFloatTensorsHasTwoPoints)
{
    for(const bool isHalf : {true, false})
    {
        Spec spec;
        spec.isHalf = isHalf;
        spec.topology = Topology::VARIANT2;
        spec.pointwiseCompute = isHalf ? "half" : "bfloat16";
        SCOPED_TRACE(spec.pointwiseCompute);
        const double eps = std::ldexp(1.0, isHalf ? -10 : -7);
        auto scale = defaultFillScale();
        scale.roundingPoints = {{eps, 2.0, 1.0}, {eps, std::sqrt(4.0 + 1.0 / 3.0), 1.0}};
        expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), dynamicTolerance(isHalf, scale));
    }
}

// An intermediate of a type the model does not know keeps the per-op default for both atol and
// rtol.
TEST_F(TestConvFwdToleranceGradingFixture, IntermediateOfAnUnknownTypeKeepsTheDefaultTolerance)
{
    Spec spec;
    spec.topology = Topology::VARIANT3;
    spec.intermediateType = "int32";
    const auto bundle = loadBundle(spec);
    const auto fixedDefault = tolerance::defaultTolerance(
        bundle->graphWrapper(), hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE),
                   AbsRelTolerance{static_cast<double>(fixedDefault.atol),
                                   static_cast<double>(fixedDefault.rtol)});
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

// A blob input is read as the element type of the graph, half or bfloat16: the same constants give
// the same moments in either. Every tensor of this graph has the output's type, so the case does
// not tell a blob's own type from the output's; GoldenBlobInputsAreScannedAsTheirOwnType does.
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
        scale.inputMean = 2.0;
        scale.weightMean = 0.5;
        expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), dynamicTolerance(isHalf, scale));
    }
}

// A golden blob is scanned as the type it holds, not as the type of the graph's output: x and w
// are float blobs under a half graph, and a scan that read their bytes as half would see other
// numbers.
TEST_F(TestConvFwdToleranceGradingFixture, GoldenBlobInputsAreScannedAsTheirOwnType)
{
    Spec spec;
    spec.isHalf = true;
    spec.inputBlobs = true;
    spec.inputType = "float";
    ConvFpropErrorScale scale;
    scale.accumulationLength = K_ACCUMULATION;
    scale.inputMeanSquare = 4.0; // x is 2.0 everywhere
    scale.weightMeanSquare = 0.25; // w is 0.5 everywhere
    scale.inputMean = 2.0;
    scale.weightMean = 0.5;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), dynamicTolerance(true, scale));
}

// ── A fused multiply: baked constant or filled tensor ───────────────────────

// The conv-fwd test-plan Variant3 half and bfloat16 cases bake the scale as 0.5 and have no recipe for it:
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

// ── The tolerance lookup names an output by its label ───────────────────────

namespace
{

void expectReportedOf(const Graded& graded, int64_t uid, const AbsRelTolerance& expected)
{
    SCOPED_TRACE(::testing::Message() << "output uid " << uid);
    const auto reported = graded.reportedToleranceOf(uid);
    ASSERT_TRUE(reported.has_value()) << graded.messages;
    EXPECT_NEAR(reported->atol, expected.atol, 1e-4 * expected.atol);
    EXPECT_NEAR(reported->rtol, expected.rtol, 1e-4 * expected.rtol);
}

} // namespace

// conv -> y, then mul(y, 0.5) -> one output and add(y, bias) -> the other: the two outputs are
// 0.5 y and y + bias. The harness gets a label from the comparison and maps it back to the
// output's uid, so each output is graded with the scale of its own branch, 0.5 and 1.
TEST_F(TestConvFwdToleranceGradingFixture, OutputsWithDistinctLabelsAreGradedWithTheirOwnScale)
{
    Spec spec;
    spec.topology = Topology::TWO_BRANCHES;
    spec.bakedValue = 0.5;
    const auto graded = grade(spec, K_FAR_OFF, K_REFERENCE);
    ASSERT_TRUE(graded.failed) << graded.messages;
    const auto scaled = dynamicTolerance(true, defaultFillScale(0.5));
    const auto unscaled = dynamicTolerance(true, defaultFillScale(1.0));
    ASSERT_LT(scaled.atol, unscaled.atol) << "the two outputs must not share a tolerance";
    expectReportedOf(graded, MUL_OUT_UID, scaled);
    expectReportedOf(graded, ADD_OUT_UID, unscaled);
}

// Two outputs that carry one label cannot be told apart by the lookup, which has the label and
// no uid: neither is graded with a dynamic tolerance, both keep the per-op default.
TEST_F(TestConvFwdToleranceGradingFixture, OutputsSharingALabelKeepThePerOpDefault)
{
    Spec spec;
    spec.topology = Topology::TWO_BRANCHES;
    spec.bakedValue = 0.5;
    spec.mulLabel = "shared_out";
    spec.addLabel = "shared_out";
    const auto bundle = loadBundle(spec);
    const auto fixedDefault = tolerance::defaultTolerance(
        bundle->graphWrapper(), hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
    const AbsRelTolerance expected{static_cast<double>(fixedDefault.atol),
                                   static_cast<double>(fixedDefault.rtol)};
    ASSERT_GT(expected.atol, dynamicTolerance(true, defaultFillScale(1.0)).atol)
        << "the default must differ from the dynamic tolerance for the test to tell them apart";
    const auto graded = grade(spec, K_FAR_OFF, K_REFERENCE);
    ASSERT_TRUE(graded.failed) << graded.messages;
    expectReportedOf(graded, MUL_OUT_UID, expected);
    expectReportedOf(graded, ADD_OUT_UID, expected);
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
        // 0.25 is four bfloat16 steps and 32 half steps at 8: outside either dynamic band.
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

// ── Float outputs ───────────────────────────────────────────────────────────

namespace
{

AbsRelTolerance floatTolerance(const ConvFpropErrorScale& scale, std::optional<double> coefficient)
{
    hipdnn_test_sdk::utilities::conv::ConvToleranceCoefficients coefficients;
    coefficients.summationOrder = coefficient;
    return hipdnn_test_sdk::utilities::conv::calculateConvFpropOutputTolerance<float>(scale,
                                                                                      coefficients);
}

constexpr const char* K_ENGINE_COEFFICIENT_TOML = R"([meta]
version = 1

[conv_fwd_tolerance.float]
summation_order = 9.0
)";

} // namespace

// A float output is graded with the dynamic tolerance and, when the engine sets nothing, the base
// coefficient of the SDK (4).
TEST_F(TestConvFwdToleranceGradingFixture, FloatOutputIsGradedWithTheBaseCoefficient)
{
    Spec spec;
    spec.isFloat = true;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), floatTolerance(defaultFillScale(), 4.0));
}

// The engine's TOML coefficient replaces the base one.
TEST_F(TestConvFwdToleranceGradingFixture, FloatOutputIsGradedWithTheCoefficientOfTheEngine)
{
    useTomlSettings(K_ENGINE_COEFFICIENT_TOML);
    Spec spec;
    spec.isFloat = true;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), floatTolerance(defaultFillScale(), 9.0));
}

// An engine value 1.5 bands from the reference (at the base coefficient) fails with no setting
// and passes under the engine's wider coefficient; one inside the base band passes either way.
TEST_F(TestConvFwdToleranceGradingFixture, EngineCoefficientDecidesAFloatOutputBetweenTheBands)
{
    const auto base = floatTolerance(defaultFillScale(), 4.0);
    const auto wide = floatTolerance(defaultFillScale(), 9.0);
    const auto reference = static_cast<double>(K_REFERENCE);
    const double baseBand = base.atol + base.rtol * reference;
    const double wideBand = wide.atol + wide.rtol * reference;
    ASSERT_LT(1.6 * baseBand, wideBand);
    const auto between = static_cast<float>(reference + 1.5 * baseBand);
    const auto inside = static_cast<float>(reference + 0.5 * baseBand);
    ASSERT_GT(static_cast<double>(between) - reference, baseBand);
    ASSERT_LT(static_cast<double>(inside) - reference, baseBand);

    Spec spec;
    spec.isFloat = true;
    EXPECT_TRUE(grade(spec, between, K_REFERENCE).failed);
    EXPECT_FALSE(grade(spec, inside, K_REFERENCE).failed);

    useTomlSettings(K_ENGINE_COEFFICIENT_TOML);
    EXPECT_FALSE(grade(spec, between, K_REFERENCE).failed);
    EXPECT_FALSE(grade(spec, inside, K_REFERENCE).failed);
}

// An explicit [[tolerance_overrides]] entry outranks the dynamic tolerance whatever coefficient
// the engine sets.
TEST_F(TestConvFwdToleranceGradingFixture, TomlToleranceOverrideOutranksTheFloatCoefficient)
{
    useTomlSettings(std::string(K_ENGINE_COEFFICIENT_TOML) + R"(
[[tolerance_overrides]]
filters = ["TestConvFwdToleranceGradingFixture.TomlToleranceOverrideOutranksTheFloat*"]
atol = 0.5
rtol = 0.25
)");
    Spec spec;
    spec.isFloat = true;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), AbsRelTolerance{0.5, 0.25});
    EXPECT_FALSE(grade(spec, K_REFERENCE + 0.25f, K_REFERENCE).failed);
}

// The coefficient is read for a float output only: half and bfloat16 keep their own.
TEST_F(TestConvFwdToleranceGradingFixture, FloatCoefficientOfTheEngineLeavesHalfAndBfloat16Alone)
{
    useTomlSettings(K_ENGINE_COEFFICIENT_TOML);
    for(const bool isHalf : {true, false})
    {
        Spec spec;
        spec.isHalf = isHalf;
        expectReported(grade(spec, K_FAR_OFF, K_REFERENCE),
                       dynamicTolerance(isHalf, defaultFillScale()));
    }
}

// A float graph with a baked scale is graded with that scale: the constant is a Float32Value
// here, and its largest magnitude narrows the band as it does a half one's.
TEST_F(TestConvFwdToleranceGradingFixture, FloatGraphWithABakedScaleIsGradedWithTheScale)
{
    Spec spec;
    spec.isFloat = true;
    spec.topology = Topology::VARIANT3;
    spec.bakedValue = 0.5;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), floatTolerance(defaultFillScale(0.5), 4.0));
}

// A float graph whose pointwise nodes compute in half is graded with the rounding points of those
// nodes, as a half or bfloat16 graph is (K = 36, sigma = 2, scale 0.5, bias uniform on [-1, 1]).
// Every operand and result that is float is converted to half and rounded, and the model's rtol
// does not cover it: the points, written out here from the model of tolerances.md, are y (2 *
// 0.5: the multiply scales its error), the baked scale (slope: the size of y), the multiply's
// result, which the add reads, the bias, the add's result, which the relu reads, and the relu's.
TEST_F(TestConvFwdToleranceGradingFixture,
       FloatGraphWithNodesComputingInHalfIsGradedWithTheirPoints)
{
    Spec spec;
    spec.isFloat = true;
    spec.topology = Topology::VARIANT3;
    spec.bakedValue = 0.5;
    spec.pointwiseCompute = "half";
    const double eps = std::ldexp(1.0, -10);
    const double afterAdd = std::sqrt(4.0 / 3.0);
    auto scale = defaultFillScale(0.5);
    scale.roundingPoints = {{eps, 2.0, 0.5}, // y, an operand of the multiply
                            {eps, 0.5, 2.0}, // the scale
                            {eps, 1.0, 1.0}, // the multiply's result
                            {eps, 1.0, 1.0}, // the multiply's result, an operand of the add
                            {eps, std::sqrt(1.0 / 3.0), 1.0}, // the bias
                            {eps, afterAdd, 1.0}, // the add's result
                            {eps, afterAdd, 1.0}, // the add's result, an operand of the relu
                            {eps, afterAdd, 1.0}}; // the relu's result, the graph output
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), floatTolerance(scale, 4.0));
}

// The route of an engine's settings file into a run: a TestConfig loads the file, productionPolicy()
// carries what it loaded into the policy, and the harness grades with the coefficient the file
// sets. A run that lost the link would grade with the base coefficient.
TEST_F(TestConvFwdToleranceGradingFixture,
       EngineFileCoefficientReachesTheGradingThroughTheProductionPolicy)
{
    useEngineFile(R"([meta]
version = 1

[conv_fwd_tolerance.float]
summation_order = 7.0

[[tolerance_overrides]]
filters = ["SomeOtherSuite.*"]
atol = 0.5
rtol = 0.25
)");
    ASSERT_NE(_policySettings, nullptr);
    Spec spec;
    spec.isFloat = true;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), floatTolerance(defaultFillScale(), 7.0));
}

// The same route carries the file's [[tolerance_overrides]]: an entry that matches the test
// replaces the dynamic tolerance of the float output, whatever coefficient the file sets.
TEST_F(TestConvFwdToleranceGradingFixture,
       EngineFileToleranceOverrideReachesTheGradingThroughTheProductionPolicy)
{
    useEngineFile(R"([meta]
version = 1

[conv_fwd_tolerance.float]
summation_order = 7.0

[[tolerance_overrides]]
filters = ["TestConvFwdToleranceGradingFixture.EngineFileToleranceOverride*"]
atol = 0.5
rtol = 0.25
)");
    ASSERT_NE(_policySettings, nullptr);
    Spec spec;
    spec.isFloat = true;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), AbsRelTolerance{0.5, 0.25});
    EXPECT_FALSE(grade(spec, K_REFERENCE + 0.25f, K_REFERENCE).failed);
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

// ── Every direction reads its own engine table ──────────────────────────────
//
// A convolution output is graded with the coefficients of [conv_<direction>_tolerance.<type>] for
// the direction of its own convolution and its own type. The file below sets a different value in
// each of the direction's tables, so an output read from another direction's table, or another
// type's, is graded with a tolerance that is not the one expected.

namespace
{

using hipdnn_test_sdk::utilities::conv::ConvDgradErrorScale;
using hipdnn_test_sdk::utilities::conv::ConvToleranceCoefficients;
using hipdnn_test_sdk::utilities::conv::ConvWrwErrorScale;

constexpr const char* K_THREE_TABLES_TOML = R"([meta]
version = 1

[conv_fwd_tolerance.half]
lambda = 5.0

[conv_dgrad_tolerance.half]
lambda = 20.0

[conv_wrw_tolerance.half]
lambda = 40.0
lambda_per_quartic_root_k = 1.0

[conv_fwd_tolerance.float]
summation_order = 6.0

[conv_dgrad_tolerance.float]
summation_order = 9.0

[conv_wrw_tolerance.float]
summation_order = 12.0
)";

ConvToleranceCoefficients coefficientsOf(std::optional<double> summationOrder,
                                         std::optional<double> lambda = std::nullopt,
                                         std::optional<double> lambdaPerQuarticRootK = std::nullopt)
{
    ConvToleranceCoefficients coefficients;
    coefficients.summationOrder = summationOrder;
    coefficients.lambda = lambda;
    coefficients.lambdaPerQuarticRootK = lambdaPerQuarticRootK;
    return coefficients;
}

Spec backwardSpec(Direction direction, bool isHalf, bool isFloat = false)
{
    Spec spec;
    spec.direction = direction;
    spec.isHalf = isHalf;
    spec.isFloat = isFloat;
    return spec;
}

/// dx [1, 4, 6, 6] from dy [1, 8, 4, 4] and w [8, 4, 3, 3]: 8 * 3 * 3 products per element, the
/// operands uniform on [-1, 1].
ConvDgradErrorScale dgradScale()
{
    ConvDgradErrorScale scale;
    scale.accumulationLength = 72;
    scale.dyMeanSquare = 1.0 / 3.0;
    scale.weightMeanSquare = 1.0 / 3.0;
    return scale;
}

/// dw [8, 4, 3, 3] from x [1, 4, 6, 6] and dy [1, 8, 4, 4]: 1 * 4 * 4 products per element.
ConvWrwErrorScale wrwScale()
{
    ConvWrwErrorScale scale;
    scale.accumulationLength = 16;
    scale.inputMeanSquare = 1.0 / 3.0;
    scale.dyMeanSquare = 1.0 / 3.0;
    return scale;
}

template <typename OutputType>
AbsRelTolerance dgradTolerance(const ConvToleranceCoefficients& coefficients = {})
{
    return hipdnn_test_sdk::utilities::conv::calculateConvDgradOutputTolerance<OutputType>(
        dgradScale(), coefficients);
}

template <typename OutputType>
AbsRelTolerance wrwTolerance(const ConvToleranceCoefficients& coefficients = {})
{
    return hipdnn_test_sdk::utilities::conv::calculateConvWrwOutputTolerance<OutputType>(
        wrwScale(), coefficients);
}

template <typename OutputType>
AbsRelTolerance fwdTolerance(const ConvToleranceCoefficients& coefficients = {})
{
    ConvFpropErrorScale scale;
    scale.accumulationLength = K_ACCUMULATION;
    scale.inputMeanSquare = 1.0 / 3.0;
    scale.weightMeanSquare = 1.0 / 3.0;
    return hipdnn_test_sdk::utilities::conv::calculateConvFpropOutputTolerance<OutputType>(
        scale, coefficients);
}

/// Two tolerances that grade apart: the check that a test of "this table and not that one" can
/// tell the tables apart.
void expectApart(const AbsRelTolerance& a, const AbsRelTolerance& b)
{
    EXPECT_GT(std::abs(a.atol - b.atol) + std::abs(a.rtol - b.rtol), 1e-2 * (a.atol + a.rtol));
}

} // namespace

TEST_F(TestConvFwdToleranceGradingFixture, DgradOutputIsGradedWithTheDgradTableAndNoOther)
{
    useTomlSettings(K_THREE_TABLES_TOML);
    const auto own = dgradTolerance<half>(coefficientsOf(std::nullopt, 20.0));
    expectApart(own, dgradTolerance<half>(coefficientsOf(std::nullopt, 5.0))); // the forward table
    expectApart(own, dgradTolerance<half>(coefficientsOf(std::nullopt, 40.0, 1.0))); // wrw
    expectApart(own, dgradTolerance<half>());
    expectReported(grade(backwardSpec(Direction::DGRAD, true), K_FAR_OFF, K_REFERENCE), own);
}

TEST_F(TestConvFwdToleranceGradingFixture, WrwOutputIsGradedWithTheWrwTableAndNoOther)
{
    useTomlSettings(K_THREE_TABLES_TOML);
    // lambda 40 and 1 per K^(1/4), which is 2 at K = 16
    const auto own = wrwTolerance<half>(coefficientsOf(std::nullopt, 40.0, 1.0));
    expectApart(own, wrwTolerance<half>(coefficientsOf(std::nullopt, 5.0))); // the forward table
    expectApart(own, wrwTolerance<half>(coefficientsOf(std::nullopt, 20.0))); // dgrad
    expectApart(own, wrwTolerance<half>());
    expectReported(grade(backwardSpec(Direction::WRW, true), K_FAR_OFF, K_REFERENCE), own);
}

TEST_F(TestConvFwdToleranceGradingFixture, ForwardOutputIsGradedWithTheForwardTableAndNoOther)
{
    useTomlSettings(K_THREE_TABLES_TOML);
    const auto own = fwdTolerance<half>(coefficientsOf(std::nullopt, 5.0));
    expectApart(own, fwdTolerance<half>(coefficientsOf(std::nullopt, 20.0)));
    expectApart(own, fwdTolerance<half>(coefficientsOf(std::nullopt, 40.0, 1.0)));
    expectApart(own, fwdTolerance<half>());
    Spec spec;
    spec.isHalf = true;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), own);
}

// The summation order of a float output is read per direction too.
TEST_F(TestConvFwdToleranceGradingFixture, FloatOutputOfEachDirectionIsGradedWithItsOwnTable)
{
    useTomlSettings(K_THREE_TABLES_TOML);
    const auto dgrad = dgradTolerance<float>(coefficientsOf(9.0));
    const auto wrw = wrwTolerance<float>(coefficientsOf(12.0));
    const auto fwd = fwdTolerance<float>(coefficientsOf(6.0));
    expectApart(dgrad, dgradTolerance<float>(coefficientsOf(6.0)));
    expectApart(dgrad, dgradTolerance<float>(coefficientsOf(12.0)));
    expectApart(wrw, wrwTolerance<float>(coefficientsOf(6.0)));
    expectApart(wrw, wrwTolerance<float>(coefficientsOf(9.0)));
    expectApart(fwd, fwdTolerance<float>(coefficientsOf(9.0)));
    expectReported(grade(backwardSpec(Direction::DGRAD, false, true), K_FAR_OFF, K_REFERENCE),
                   dgrad);
    expectReported(grade(backwardSpec(Direction::WRW, false, true), K_FAR_OFF, K_REFERENCE), wrw);
    Spec spec;
    spec.isFloat = true;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), fwd);
}

// A table sets the coefficients of its own output type: bfloat16 outputs of the direction have no
// table in this file and keep the SDK's allowance, as does a direction with no table at all.
TEST_F(TestConvFwdToleranceGradingFixture,
       OutputOfATypeOrDirectionWithoutATableKeepsTheSdkAllowance)
{
    useTomlSettings(K_THREE_TABLES_TOML);
    expectReported(grade(backwardSpec(Direction::DGRAD, false), K_FAR_OFF, K_REFERENCE),
                   dgradTolerance<bfloat16>());
    expectReported(grade(backwardSpec(Direction::WRW, false), K_FAR_OFF, K_REFERENCE),
                   wrwTolerance<bfloat16>());

    useTomlSettings(R"([meta]
version = 1

[conv_dgrad_tolerance.half]
lambda = 20.0
)");
    expectReported(grade(backwardSpec(Direction::DGRAD, true), K_FAR_OFF, K_REFERENCE),
                   dgradTolerance<half>(coefficientsOf(std::nullopt, 20.0)));
    expectReported(grade(backwardSpec(Direction::WRW, true), K_FAR_OFF, K_REFERENCE),
                   wrwTolerance<half>());
    Spec spec;
    spec.isHalf = true;
    expectReported(grade(spec, K_FAR_OFF, K_REFERENCE), fwdTolerance<half>());
}

// An engine's lambda is in units of the root mean square of the output, which has the means of
// the inputs in it: dy and w uniform on [0, 2] have a mean of 1 and a mean square of 4/3, so the
// rms of a dx element is sqrt(72 * 16/9 + 72 * 71) = 72.4 where its zero-mean spread is 11.3.
TEST_F(TestConvFwdToleranceGradingFixture, EngineLambdaIsPricedOnTheMeansOfTheInputs)
{
    useTomlSettings(K_THREE_TABLES_TOML);
    ConvDgradErrorScale scale = dgradScale();
    scale.dyMeanSquare = 4.0 / 3.0;
    scale.weightMeanSquare = 4.0 / 3.0;
    scale.dyMean = 1.0;
    scale.weightMean = 1.0;
    const auto coefficients = coefficientsOf(std::nullopt, 20.0);
    const auto expected = hipdnn_test_sdk::utilities::conv::calculateConvDgradOutputTolerance<half>(
        scale, coefficients);
    auto withoutMeans = scale;
    withoutMeans.dyMean = 0.0;
    withoutMeans.weightMean = 0.0;
    ASSERT_GT(expected.atol,
              2.0
                  * hipdnn_test_sdk::utilities::conv::calculateConvDgradOutputTolerance<half>(
                        withoutMeans, coefficients)
                        .atol);

    const auto positive = [](InputFillRecipes& recipes) {
        recipes.set(Y_UID, FillRecipe::free(0.0f, 2.0f));
        recipes.set(W_UID, FillRecipe::free(0.0f, 2.0f));
    };
    expectReported(grade(backwardSpec(Direction::DGRAD, true), K_FAR_OFF, K_REFERENCE, positive),
                   expected);
}

// An arch table of an engine file grades the outputs of the devices it names, in place of the
// table of the output type, and no other device's.
namespace
{

constexpr const char* K_ARCH_TABLES_TOML = R"([meta]
version = 1

[conv_dgrad_tolerance.half]
lambda = 20.0

[conv_dgrad_tolerance.half.gfx90a]
lambda = 46.0

[conv_wrw_tolerance.half]
lambda = 13.0
lambda_per_quartic_root_k = 2.5

[conv_wrw_tolerance.half.gfx90a]
lambda = 35.0
lambda_per_quartic_root_k = 2.0
)";

} // namespace

TEST_F(TestConvFwdToleranceGradingFixture, ArchTableOfTheDeviceChangesTheBandOfEachDirection)
{
    useTomlSettings(K_ARCH_TABLES_TOML);
    _arch = "gfx90a:sramecc+:xnack-";
    const auto dgrad = dgradTolerance<half>(coefficientsOf(std::nullopt, 46.0));
    const auto wrw = wrwTolerance<half>(coefficientsOf(std::nullopt, 35.0, 2.0));
    expectApart(dgrad, dgradTolerance<half>(coefficientsOf(std::nullopt, 20.0)));
    expectApart(wrw, wrwTolerance<half>(coefficientsOf(std::nullopt, 13.0, 2.5)));
    expectReported(grade(backwardSpec(Direction::DGRAD, true), K_FAR_OFF, K_REFERENCE), dgrad);
    expectReported(grade(backwardSpec(Direction::WRW, true), K_FAR_OFF, K_REFERENCE), wrw);
}

TEST_F(TestConvFwdToleranceGradingFixture, ArchTableOfAnotherDeviceLeavesTheTypeTableInForce)
{
    useTomlSettings(K_ARCH_TABLES_TOML);
    _arch = "gfx942:sramecc+:xnack-";
    expectReported(grade(backwardSpec(Direction::DGRAD, true), K_FAR_OFF, K_REFERENCE),
                   dgradTolerance<half>(coefficientsOf(std::nullopt, 20.0)));
    expectReported(grade(backwardSpec(Direction::WRW, true), K_FAR_OFF, K_REFERENCE),
                   wrwTolerance<half>(coefficientsOf(std::nullopt, 13.0, 2.5)));
}

// The arch table replaces the type table whole: with no key of its own it is the SDK's values.
TEST_F(TestConvFwdToleranceGradingFixture, ArchTableDoesNotInheritTheTypeTableCoefficients)
{
    useTomlSettings(R"([meta]
version = 1

[conv_wrw_tolerance.half]
lambda = 13.0
lambda_per_quartic_root_k = 2.5

[conv_wrw_tolerance.half.gfx90a]
lambda = 35.0
)");
    _arch = "gfx90a:sramecc+:xnack-";
    const auto own = wrwTolerance<half>(coefficientsOf(std::nullopt, 35.0));
    expectApart(own, wrwTolerance<half>(coefficientsOf(std::nullopt, 35.0, 2.5)));
    expectReported(grade(backwardSpec(Direction::WRW, true), K_FAR_OFF, K_REFERENCE), own);
}

// An arch table of a type and direction touches no other: the half arch table leaves the float
// and bfloat16 outputs of the same direction on the SDK's values.
TEST_F(TestConvFwdToleranceGradingFixture, ArchTableOfOneTypeLeavesTheOtherTypesAlone)
{
    useTomlSettings(K_ARCH_TABLES_TOML);
    _arch = "gfx90a:sramecc+:xnack-";
    expectReported(grade(backwardSpec(Direction::DGRAD, false), K_FAR_OFF, K_REFERENCE),
                   dgradTolerance<bfloat16>());
    expectReported(grade(backwardSpec(Direction::DGRAD, false, true), K_FAR_OFF, K_REFERENCE),
                   dgradTolerance<float>());
}

// NOLINTEND(readability-identifier-naming)
