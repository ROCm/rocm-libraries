// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// cudnn_frontend_utils.h must come first: this TU proves it compiles standalone.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <initializer_list>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;
namespace backend = fe::detail::backend;
using fe::AttentionImplementation_t;
using fe::BehaviorNote_t;
using fe::BuildPlanPolicy_t;
using fe::ConvolutionMode_t;
using fe::DataType_t;
using fe::DescriptorType_t;
using fe::DiagonalAlignment_t;
using fe::HeurMode_t;
using fe::MoeGroupedMatmulMode_t;
using fe::NormFwdPhase_t;
using fe::NormMode_t;
using fe::NumericalNote_t;
using fe::PaddingMode_t;
using fe::PointwiseMode_t;
using fe::ReductionMode_t;
using fe::ResampleMode_t;
using fe::ReshapeMode_t;
using fe::RngDistribution_t;
using fe::TensorReordering_t;

std::optional<std::string> text(const char* name)
{
    return name == nullptr ? std::nullopt : std::optional<std::string>(name);
}

// Expected values are upstream's NLOHMANN_JSON_SERIALIZE_ENUM tables; nullptr is JSON null.
template <typename Enum>
void expectNames(std::initializer_list<std::pair<Enum, const char*>> rows)
{
    for(const auto& [value, expected] : rows)
    {
        SCOPED_TRACE(static_cast<int>(value));
        EXPECT_EQ(text(fe::detail::enumName(value)), text(expected));
    }
}

class DescribedDescriptor : public fe::BackendDescriptor
{
public:
    std::string describe() const override
    {
        return "described";
    }
};

class TestCudnnShimV8Utils : public hipdnn_shim_test::ShimV8MockBackendFixture
{
};

TEST_F(TestCudnnShimV8Utils, ConvertersFailClosedLikeUpstream)
{
    cudnnDataType_t out = CUDNN_DATA_DOUBLE;
    EXPECT_EQ(fe::detail::convert_to_cudnn_type(DataType_t::NOT_SET, out),
              CUDNN_STATUS_INVALID_VALUE);
    EXPECT_EQ(out, CUDNN_DATA_DOUBLE);
    EXPECT_EQ(fe::detail::convert_to_cudnn_type(DataType_t::FP6_E2M3, out),
              CUDNN_STATUS_INVALID_VALUE);
    EXPECT_EQ(out, CUDNN_DATA_DOUBLE);
    EXPECT_EQ(fe::detail::convert_to_cudnn_type(DataType_t::HALF, out), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(out, CUDNN_DATA_HALF);

    cudnnBackendBehaviorNote_t note = CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
    EXPECT_EQ(fe::detail::convert_to_cudnn_type(BehaviorNote_t::REQUIRES_LAYOUT_TRANSFORM, note),
              CUDNN_STATUS_INVALID_VALUE);
    EXPECT_EQ(note, CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION);

    EXPECT_EQ(fe::detail::convert_from_cudnn_type(CUDNN_DATA_INT8x4), DataType_t::INT8x4);
    // 20 is inside cudnnDataType_t's range but names no enumerator.
    EXPECT_EQ(fe::detail::convert_from_cudnn_type(static_cast<cudnnDataType_t>(20)),
              DataType_t::NOT_SET);

    // In-range values the C enums do not name: the out-param forms overwrite the
    // output with upstream's fallback.
    PaddingMode_t padding = PaddingMode_t::ZERO_PAD;
    fe::detail::convert_from_cudnn_type(static_cast<cudnnPaddingMode_t>(3), padding);
    EXPECT_EQ(padding, PaddingMode_t::NOT_SET);
    ResampleMode_t resample = ResampleMode_t::MAXPOOL;
    fe::detail::convert_from_cudnn_type(static_cast<cudnnResampleMode_t>(6), resample);
    EXPECT_EQ(resample, ResampleMode_t::NOT_SET);

    TensorReordering_t reordering = TensorReordering_t::NONE;
    fe::detail::convert_from_cudnn_type(CUDNN_TENSOR_REORDERING_F16x16, reordering);
    EXPECT_EQ(reordering, TensorReordering_t::F16x16);
}

TEST_F(TestCudnnShimV8Utils, ConvolutionModeNotSetMapsToCrossCorrelation)
{
    EXPECT_EQ(fe::detail::convert_to_cudnn_type(ConvolutionMode_t::NOT_SET),
              CUDNN_CROSS_CORRELATION);
    EXPECT_EQ(fe::detail::convert_to_cudnn_type(ConvolutionMode_t::CONVOLUTION), CUDNN_CONVOLUTION);
    EXPECT_EQ(fe::detail::convert_from_cudnn_type(CUDNN_CONVOLUTION),
              ConvolutionMode_t::CONVOLUTION);
}

TEST_F(TestCudnnShimV8Utils, ResampleAvgpoolAliasMapsToIncludePadding)
{
    ResampleMode_t mode = ResampleMode_t::NOT_SET;
    fe::detail::convert_from_cudnn_type(CUDNN_RESAMPLE_AVGPOOL, mode);
    EXPECT_EQ(mode, ResampleMode_t::AVGPOOL_INCLUDE_PADDING);
}

TEST_F(TestCudnnShimV8Utils, EnumNameMatchesUpstreamTables)
{
    expectNames<PointwiseMode_t>({
        {PointwiseMode_t::NOT_SET, nullptr},
        {PointwiseMode_t::ADD, "ADD"},
        {PointwiseMode_t::MUL, "MUL"},
        {PointwiseMode_t::SQRT, "SQRT"},
        {PointwiseMode_t::MAX, "MAX"},
        {PointwiseMode_t::MIN, "MIN"},
        {PointwiseMode_t::RELU_FWD, "RELU_FWD"},
        {PointwiseMode_t::TANH_FWD, "TANH_FWD"},
        {PointwiseMode_t::SIGMOID_FWD, "SIGMOID_FWD"},
        {PointwiseMode_t::ELU_FWD, "ELU_FWD"},
        {PointwiseMode_t::GELU_FWD, "GELU_FWD"},
        {PointwiseMode_t::SOFTPLUS_FWD, "SOFTPLUS_FWD"},
        {PointwiseMode_t::SWISH_FWD, "SWISH_FWD"},
        {PointwiseMode_t::RELU_BWD, "RELU_BWD"},
        {PointwiseMode_t::TANH_BWD, "TANH_BWD"},
        {PointwiseMode_t::SIGMOID_BWD, "SIGMOID_BWD"},
        {PointwiseMode_t::ELU_BWD, "ELU_BWD"},
        {PointwiseMode_t::GELU_BWD, "GELU_BWD"},
        {PointwiseMode_t::SOFTPLUS_BWD, "SOFTPLUS_BWD"},
        {PointwiseMode_t::SWISH_BWD, "SWISH_BWD"},
        {PointwiseMode_t::ERF, "ERF"},
        {PointwiseMode_t::IDENTITY, "IDENTITY"},
        {PointwiseMode_t::GELU_APPROX_TANH_BWD, "GELU_APPROX_TANH_BWD"},
        {PointwiseMode_t::GELU_APPROX_TANH_FWD, "GELU_APPROX_TANH_FWD"},
        {PointwiseMode_t::GEN_INDEX, "GEN_INDEX"},
        {PointwiseMode_t::BINARY_SELECT, "BINARY_SELECT"},
        {PointwiseMode_t::EXP, "EXP"},
        {PointwiseMode_t::LOG, "LOG"},
        {PointwiseMode_t::NEG, "NEG"},
        {PointwiseMode_t::MOD, "MOD"},
        {PointwiseMode_t::POW, "POW"},
        {PointwiseMode_t::ABS, "ABS"},
        {PointwiseMode_t::CEIL, "CEIL"},
        {PointwiseMode_t::COS, "COS"},
        {PointwiseMode_t::FLOOR, "FLOOR"},
        {PointwiseMode_t::RSQRT, "RSQRT"},
        {PointwiseMode_t::SIN, "SIN"},
        {PointwiseMode_t::LOGICAL_NOT, "LOGICAL_NOT"},
        {PointwiseMode_t::TAN, "TAN"},
        {PointwiseMode_t::SUB, "SUB"},
        {PointwiseMode_t::ADD_SQUARE, "ADD_SQUARE"},
        {PointwiseMode_t::DIV, "DIV"},
        {PointwiseMode_t::CMP_EQ, "CMP_EQ"},
        {PointwiseMode_t::CMP_NEQ, "CMP_NEQ"},
        {PointwiseMode_t::CMP_GT, "CMP_GT"},
        {PointwiseMode_t::CMP_GE, "CMP_GE"},
        {PointwiseMode_t::CMP_LT, "CMP_LT"},
        {PointwiseMode_t::CMP_LE, "CMP_LE"},
        {PointwiseMode_t::LOGICAL_AND, "LOGICAL_AND"},
        {PointwiseMode_t::LOGICAL_OR, "LOGICAL_OR"},
        {PointwiseMode_t::RECIPROCAL, "RECIPROCAL"},
        {PointwiseMode_t::COUNT, nullptr},
    });
    expectNames<DataType_t>({
        {DataType_t::NOT_SET, nullptr},
        {DataType_t::FLOAT, "FLOAT"},
        {DataType_t::DOUBLE, "DOUBLE"},
        {DataType_t::HALF, "HALF"},
        {DataType_t::INT8, "INT8"},
        {DataType_t::INT32, "INT32"},
        {DataType_t::INT8x4, "INT8x4"},
        {DataType_t::UINT8, "UINT8"},
        {DataType_t::UINT8x4, "UINT8x4"},
        {DataType_t::INT8x32, "INT8x32"},
        {DataType_t::BFLOAT16, "BFLOAT16"},
        {DataType_t::INT64, "INT64"},
        {DataType_t::BOOLEAN, "BOOLEAN"},
        {DataType_t::FP8_E4M3, "FP8_E4M3"},
        {DataType_t::FP8_E5M2, "FP8_E5M2"},
        {DataType_t::FAST_FLOAT_FOR_FP8, "FAST_FLOAT_FOR_FP8"},
        {DataType_t::FP8_E8M0, "FP8_E8M0"},
        {DataType_t::FP4_E2M1, "FP4_E2M1"},
        {DataType_t::INT4, "INT4"},
        {DataType_t::COMPLEX_FP32, "COMPLEX_FP32"},
        {DataType_t::COMPLEX_FP64, "COMPLEX_FP64"},
        {DataType_t::FP6_E2M3, nullptr},
        {DataType_t::FP6_E3M2, nullptr},
        {DataType_t::FP8_E4M3_FNUZ, nullptr},
        {DataType_t::FP8_E5M2_FNUZ, nullptr},
    });
    expectNames<ConvolutionMode_t>({
        {ConvolutionMode_t::CONVOLUTION, "CONVOLUTION"},
        {ConvolutionMode_t::CROSS_CORRELATION, "CROSS_CORRELATION"},
        {ConvolutionMode_t::NOT_SET, "CONVOLUTION"},
    });
    expectNames<ReductionMode_t>({
        {ReductionMode_t::NOT_SET, nullptr},
        {ReductionMode_t::ADD, "ADD"},
        {ReductionMode_t::MUL, "MUL"},
        {ReductionMode_t::MIN, "MIN"},
        {ReductionMode_t::MAX, "MAX"},
        {ReductionMode_t::AMAX, "AMAX"},
        {ReductionMode_t::AVG, "AVG"},
        {ReductionMode_t::NORM1, "NORM1"},
        {ReductionMode_t::NORM2, "NORM2"},
        {ReductionMode_t::MUL_NO_ZEROS, "MUL_NO_ZEROS"},
    });
    expectNames<ResampleMode_t>({
        {ResampleMode_t::NOT_SET, nullptr},
        {ResampleMode_t::AVGPOOL_EXCLUDE_PADDING, "AVGPOOL_EXCLUDE_PADDING"},
        {ResampleMode_t::AVGPOOL_INCLUDE_PADDING, "AVGPOOL_INCLUDE_PADDING"},
        {ResampleMode_t::BILINEAR, "BILINEAR"},
        {ResampleMode_t::NEAREST, "NEAREST"},
        {ResampleMode_t::MAXPOOL, "MAXPOOL"},
    });
    expectNames<PaddingMode_t>({
        {PaddingMode_t::NOT_SET, nullptr},
        {PaddingMode_t::EDGE_VAL_PAD, "EDGE_VAL_PAD"},
        {PaddingMode_t::NEG_INF_PAD, "NEG_INF_PAD"},
        {PaddingMode_t::ZERO_PAD, "ZERO_PAD"},
    });
    expectNames<NormFwdPhase_t>({
        {NormFwdPhase_t::NOT_SET, nullptr},
        {NormFwdPhase_t::INFERENCE, "INFERENCE"},
        {NormFwdPhase_t::TRAINING, "TRAINING"},
    });
    expectNames<NormMode_t>({
        {NormMode_t::NOT_SET, nullptr},
        {NormMode_t::LAYER_NORM, "LAYER_NORM"},
        {NormMode_t::INSTANCE_NORM, "INSTANCE_NORM"},
        {NormMode_t::BATCH_NORM, "BATCH_NORM"},
        {NormMode_t::GROUP_NORM, "GROUP_NORM"},
        {NormMode_t::RMS_NORM, "RMS_NORM"},
        {NormMode_t::ADA_LAYER_NORM, "ADA_LAYER_NORM"},
    });
    expectNames<RngDistribution_t>({
        {RngDistribution_t::NOT_SET, nullptr},
        {RngDistribution_t::BERNOULLI, "BERNOULLI"},
        {RngDistribution_t::UNIFORM, "UNIFORM"},
        {RngDistribution_t::NORMAL, "NORMAL"},
    });
    expectNames<TensorReordering_t>({
        {TensorReordering_t::NONE, "NONE"},
        {TensorReordering_t::INT8x32, "INT8x32"},
        {TensorReordering_t::F16x16, "F16x16"},
        {TensorReordering_t::F8_128x4, "F8_128x4"},
    });
    expectNames<ReshapeMode_t>({
        {ReshapeMode_t::NOT_SET, nullptr},
        {ReshapeMode_t::VIEW_ONLY, "VIEW_ONLY"},
        {ReshapeMode_t::LOGICAL, "LOGICAL"},
    });
    expectNames<HeurMode_t>({
        {HeurMode_t::A, "A"},
        {HeurMode_t::B, "B"},
        {HeurMode_t::FALLBACK, "FALLBACK"},
        {HeurMode_t::OPENSOURCE, "OPENSOURCE"},
    });
    expectNames<BehaviorNote_t>({
        {BehaviorNote_t::NOT_SET, "NOT_SET"},
        {BehaviorNote_t::RUNTIME_COMPILATION, "RUNTIME_COMPILATION"},
        {BehaviorNote_t::REQUIRES_FILTER_INT8x32_REORDER, "REQUIRES_FILTER_INT8x32_REORDER"},
        {BehaviorNote_t::REQUIRES_BIAS_INT8x32_REORDER, "REQUIRES_BIAS_INT8x32_REORDER"},
        {BehaviorNote_t::SUPPORTS_CUDA_GRAPH_NATIVE_API, "SUPPORTS_CUDA_GRAPH_NATIVE_API"},
        {BehaviorNote_t::CUBLASLT_DEPENDENCY, "CUBLASLT_DEPENDENCY"},
        {BehaviorNote_t::REQUIRES_LAYOUT_TRANSFORM, "NOT_SET"},
        {BehaviorNote_t::SUPPORTS_GRAPH_CAPTURE, "NOT_SET"},
        {BehaviorNote_t::EXTERNAL_LIBRARY_DEPENDENCY, "NOT_SET"},
        {BehaviorNote_t::SUPPORTS_EXECUTION_PLAN_SERIALIZATION, "NOT_SET"},
    });
    expectNames<NumericalNote_t>({
        {NumericalNote_t::NOT_SET, "NOT_SET"},
        {NumericalNote_t::TENSOR_CORE, "TENSOR_CORE"},
        {NumericalNote_t::DOWN_CONVERT_INPUTS, "DOWN_CONVERT_INPUTS"},
        {NumericalNote_t::REDUCED_PRECISION_REDUCTION, "REDUCED_PRECISION_REDUCTION"},
        {NumericalNote_t::FFT, "FFT"},
        {NumericalNote_t::NONDETERMINISTIC, "NONDETERMINISTIC"},
        {NumericalNote_t::WINOGRAD, "WINOGRAD"},
        {NumericalNote_t::WINOGRAD_TILE_4x4, "WINOGRAD_TILE_4x4"},
        {NumericalNote_t::WINOGRAD_TILE_6x6, "WINOGRAD_TILE_6x6"},
        {NumericalNote_t::WINOGRAD_TILE_13x13, "WINOGRAD_TILE_13x13"},
        {NumericalNote_t::STRICT_NAN_PROP, "STRICT_NAN_PROP"},
    });
    expectNames<BuildPlanPolicy_t>({
        {BuildPlanPolicy_t::HEURISTICS_CHOICE, "HEURISTICS_CHOICE"},
        {BuildPlanPolicy_t::ALL, "ALL"},
    });
    expectNames<DiagonalAlignment_t>({
        {DiagonalAlignment_t::TOP_LEFT, "TOP_LEFT"},
        {DiagonalAlignment_t::BOTTOM_RIGHT, "BOTTOM_RIGHT"},
    });
    expectNames<AttentionImplementation_t>({
        {AttentionImplementation_t::AUTO, "AUTO"},
        {AttentionImplementation_t::COMPOSITE, "COMPOSITE"},
        {AttentionImplementation_t::UNIFIED, "UNIFIED"},
    });
    expectNames<MoeGroupedMatmulMode_t>({
        {MoeGroupedMatmulMode_t::NOT_SET, nullptr},
        {MoeGroupedMatmulMode_t::NONE, "NONE"},
        {MoeGroupedMatmulMode_t::GATHER, "GATHER"},
        {MoeGroupedMatmulMode_t::SCATTER, "SCATTER"},
    });
}

TEST_F(TestCudnnShimV8Utils, EnumJsonMatchesNlohmannRendering)
{
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    EXPECT_EQ(fe::detail::enumJson(DataType_t::FLOAT), nlohmann::json{std::string("FLOAT")}.dump());
    EXPECT_EQ(fe::detail::enumJson(DataType_t::NOT_SET), nlohmann::json{nullptr}.dump());
    EXPECT_EQ(fe::detail::enumJson(BehaviorNote_t::NOT_SET),
              nlohmann::json{std::string("NOT_SET")}.dump());
#else
    GTEST_SKIP() << "nlohmann::json is not available under CUDNN_FRONTEND_SKIP_JSON_LIB";
#endif
}

TEST_F(TestCudnnShimV8Utils, ThrowHelperAppendsLastErrorAndStatus)
{
    cudnnBackendDescriptor_t raw = nullptr;
    ASSERT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &raw),
              CUDNN_STATUS_SUCCESS);
    // The first failure reveals the shim's message; the second leaves the same one pending.
    ASSERT_NE(cudnnBackendFinalize(raw), CUDNN_STATUS_SUCCESS);
    const std::string shimError = fe::detail::get_last_error_string_();
    ASSERT_FALSE(shimError.empty());
    ASSERT_NE(cudnnBackendFinalize(raw), CUDNN_STATUS_SUCCESS);

    DescribedDescriptor desc;
    std::optional<fe::cudnnException> caught;
    try
    {
        fe::set_error_and_throw_exception(&desc, CUDNN_STATUS_BAD_PARAM, "X");
    }
    catch(const fe::cudnnException& exception)
    {
        caught = exception;
    }

    ASSERT_TRUE(caught.has_value());
    EXPECT_EQ(caught->getCudnnStatus(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(std::string(caught->what()), "X" + shimError + " cudnn_status: mock status");
    EXPECT_EQ(desc.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(std::string(desc.get_error()), "X" + shimError);
    EXPECT_EQ(fe::detail::get_last_error_string_(), "");

    EXPECT_EQ(cudnnBackendDestroyDescriptor(raw), CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8Utils, VersionHelpersReportClaim)
{
    static_assert(fe::detail::get_compiled_version() == 92200);
    EXPECT_EQ(fe::detail::get_backend_version_string(), "9.22.0");
    EXPECT_EQ(fe::detail::convert_version_to_str(8907), "8.9.7");
}

TEST_F(TestCudnnShimV8Utils, NoteAndDistributionStringsMatchUpstream)
{
    EXPECT_EQ(fe::to_string(CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP),
              "CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP");
    EXPECT_EQ(fe::to_string(CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY),
              "CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY");
    EXPECT_EQ(fe::to_string(CUDNN_RNG_DISTRIBUTION_NORMAL), "CUDNN_RNG_DISTRIBUTION_NORMAL");
}

TEST_F(TestCudnnShimV8Utils, PointwisePortCountMatchesUpstream)
{
    EXPECT_EQ(fe::get_pointwise_mode_port_count(PointwiseMode_t::NOT_SET), 0);
    EXPECT_EQ(fe::get_pointwise_mode_port_count(PointwiseMode_t::ADD), 3);
    EXPECT_EQ(fe::get_pointwise_mode_port_count(PointwiseMode_t::RELU_FWD), 2);
    EXPECT_EQ(fe::get_pointwise_mode_port_count(PointwiseMode_t::BINARY_SELECT), 4);
    EXPECT_EQ(fe::get_pointwise_mode_port_count(PointwiseMode_t::COUNT), -1);
}

TEST_F(TestCudnnShimV8Utils, DescriptorTypeStreamsUpstreamName)
{
    std::ostringstream os;
    os << DescriptorType_t::TENSOR_DESCRIPTOR << ' ' << DescriptorType_t::NOT_SET;
    EXPECT_EQ(os.str(), "TENSOR_DESCRIPTOR NOT_SET");
}

TEST_F(TestCudnnShimV8Utils, DescriptorStreamsOnlyWhenLoggingEnabled)
{
    bool& enabled = fe::isLoggingEnabled();
    const bool saved = enabled;
    const DescribedDescriptor desc;

    enabled = true;
    std::ostringstream on;
    on << desc;
    enabled = false;
    std::ostringstream off;
    off << desc;
    enabled = saved;

    EXPECT_EQ(on.str(), "described");
    EXPECT_EQ(off.str(), "");
}

TEST_F(TestCudnnShimV8Utils, GetEnvironmentIsNullForUnsetOrEmpty)
{
    const char* const name = "HIPDNN_CUDNN_SHIM_V8_UTILS_PROBE";
    hipdnn_data_sdk::utilities::unsetEnv(name);
    EXPECT_EQ(fe::get_environment(name), nullptr);

    hipdnn_data_sdk::utilities::setEnv(name, "");
    EXPECT_EQ(fe::get_environment(name), nullptr);

    hipdnn_data_sdk::utilities::setEnv(name, "set");
    EXPECT_EQ(text(fe::get_environment(name)), text("set"));

    hipdnn_data_sdk::utilities::unsetEnv(name);
}

TEST_F(TestCudnnShimV8Utils, OpaquePointerOwnsShimDescriptor)
{
    std::weak_ptr<backend::Descriptor> object;
    {
        fe::ManagedOpaqueDescriptor pointer
            = fe::make_shared_backend_pointer(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        ASSERT_TRUE(pointer->is_good());
        cudnnBackendDescriptor_t raw = pointer->get_backend_descriptor();
        ASSERT_NE(raw, nullptr);
        // White-box: raw is only inspected while the owner keeps it alive.
        const backend::Handle* handle = backend::liveHandle(raw);
        ASSERT_NE(handle, nullptr);
        object = handle->object;

        const fe::ManagedOpaqueDescriptor copy = pointer;
        pointer.reset();
        EXPECT_FALSE(object.expired());
    }
    EXPECT_TRUE(object.expired());
}

} // namespace
