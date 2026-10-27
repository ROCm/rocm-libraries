// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Exhaustive by-name checks of the cuDNN C enum <-> shim FE enum tables. The C
// side is enumerated from cudnn_v8_enum_values.inc, so a new C enumerator without
// a mapping decision fails here; the FE lists are guarded by -Wswitch-enum.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>
#include <hipdnn_compatibility/cudnn/detail/type_mapping.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace
{

namespace shim = hipdnn_frontend::compatibility::cudnn_frontend::detail;
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// clang-format off
#define HIPDNN_TEST_FE_DATA_TYPE(X) \
    X(NOT_SET)                      \
    X(FLOAT)                        \
    X(DOUBLE)                       \
    X(HALF)                         \
    X(INT8)                         \
    X(INT32)                        \
    X(INT8x4)                       \
    X(UINT8)                        \
    X(UINT8x4)                      \
    X(INT8x32)                      \
    X(BFLOAT16)                     \
    X(INT64)                        \
    X(BOOLEAN)                      \
    X(FP8_E4M3)                     \
    X(FP8_E5M2)                     \
    X(FAST_FLOAT_FOR_FP8)           \
    X(FP8_E8M0)                     \
    X(FP4_E2M1)                     \
    X(INT4)                         \
    X(COMPLEX_FP32)                 \
    X(COMPLEX_FP64)                 \
    X(FP6_E2M3)                     \
    X(FP6_E3M2)                     \
    X(FP8_E4M3_FNUZ)                \
    X(FP8_E5M2_FNUZ)

#define HIPDNN_TEST_FE_POINTWISE_MODE(X) \
    X(NOT_SET)                           \
    X(ADD)                               \
    X(MUL)                               \
    X(SQRT)                              \
    X(MAX)                               \
    X(MIN)                               \
    X(RELU_FWD)                          \
    X(TANH_FWD)                          \
    X(SIGMOID_FWD)                       \
    X(ELU_FWD)                           \
    X(GELU_FWD)                          \
    X(SOFTPLUS_FWD)                      \
    X(SWISH_FWD)                         \
    X(RELU_BWD)                          \
    X(TANH_BWD)                          \
    X(SIGMOID_BWD)                       \
    X(ELU_BWD)                           \
    X(GELU_BWD)                          \
    X(SOFTPLUS_BWD)                      \
    X(SWISH_BWD)                         \
    X(ERF)                               \
    X(IDENTITY)                          \
    X(GELU_APPROX_TANH_BWD)              \
    X(GELU_APPROX_TANH_FWD)              \
    X(GEN_INDEX)                         \
    X(BINARY_SELECT)                     \
    X(EXP)                               \
    X(LOG)                               \
    X(NEG)                               \
    X(MOD)                               \
    X(POW)                               \
    X(ABS)                               \
    X(CEIL)                              \
    X(COS)                               \
    X(FLOOR)                             \
    X(RSQRT)                             \
    X(SIN)                               \
    X(LOGICAL_NOT)                       \
    X(TAN)                               \
    X(SUB)                               \
    X(ADD_SQUARE)                        \
    X(DIV)                               \
    X(CMP_EQ)                            \
    X(CMP_NEQ)                           \
    X(CMP_GT)                            \
    X(CMP_GE)                            \
    X(CMP_LT)                            \
    X(CMP_LE)                            \
    X(LOGICAL_AND)                       \
    X(LOGICAL_OR)                        \
    X(RECIPROCAL)                        \
    X(COUNT)

#define HIPDNN_TEST_FE_CONVOLUTION_MODE(X) \
    X(NOT_SET)                             \
    X(CONVOLUTION)                         \
    X(CROSS_CORRELATION)

#define HIPDNN_TEST_FE_REDUCTION_MODE(X) \
    X(NOT_SET)                           \
    X(ADD)                               \
    X(MUL)                               \
    X(MIN)                               \
    X(MAX)                               \
    X(AMAX)                              \
    X(AVG)                               \
    X(NORM1)                             \
    X(NORM2)                             \
    X(MUL_NO_ZEROS)

#define HIPDNN_TEST_FE_RESAMPLE_MODE(X) \
    X(NOT_SET)                          \
    X(AVGPOOL_EXCLUDE_PADDING)          \
    X(AVGPOOL_INCLUDE_PADDING)          \
    X(BILINEAR)                         \
    X(NEAREST)                          \
    X(MAXPOOL)

#define HIPDNN_TEST_FE_PADDING_MODE(X) \
    X(NOT_SET)                         \
    X(EDGE_VAL_PAD)                    \
    X(NEG_INF_PAD)                     \
    X(ZERO_PAD)

#define HIPDNN_TEST_FE_NORM_FWD_PHASE(X) \
    X(NOT_SET)                           \
    X(INFERENCE)                         \
    X(TRAINING)

#define HIPDNN_TEST_FE_NORM_MODE(X) \
    X(NOT_SET)                      \
    X(LAYER_NORM)                   \
    X(INSTANCE_NORM)                \
    X(BATCH_NORM)                   \
    X(GROUP_NORM)                   \
    X(RMS_NORM)                     \
    X(ADA_LAYER_NORM)

#define HIPDNN_TEST_FE_BEHAVIOR_NOTE(X)      \
    X(NOT_SET)                               \
    X(RUNTIME_COMPILATION)                   \
    X(REQUIRES_FILTER_INT8x32_REORDER)       \
    X(REQUIRES_BIAS_INT8x32_REORDER)         \
    X(SUPPORTS_CUDA_GRAPH_NATIVE_API)        \
    X(CUBLASLT_DEPENDENCY)                   \
    X(REQUIRES_LAYOUT_TRANSFORM)             \
    X(SUPPORTS_GRAPH_CAPTURE)                \
    X(EXTERNAL_LIBRARY_DEPENDENCY)           \
    X(SUPPORTS_EXECUTION_PLAN_SERIALIZATION)

#define HIPDNN_TEST_FE_NUMERICAL_NOTE(X) \
    X(NOT_SET)                           \
    X(TENSOR_CORE)                       \
    X(DOWN_CONVERT_INPUTS)               \
    X(REDUCED_PRECISION_REDUCTION)       \
    X(FFT)                               \
    X(NONDETERMINISTIC)                  \
    X(WINOGRAD)                          \
    X(WINOGRAD_TILE_4x4)                 \
    X(WINOGRAD_TILE_6x6)                 \
    X(WINOGRAD_TILE_13x13)               \
    X(STRICT_NAN_PROP)

#define HIPDNN_TEST_FE_TENSOR_REORDERING(X) \
    X(NONE)                                 \
    X(INT8x32)                              \
    X(F16x16)                               \
    X(F8_128x4)

#define HIPDNN_TEST_FE_RESHAPE_MODE(X) \
    X(NOT_SET)                         \
    X(VIEW_ONLY)                       \
    X(LOGICAL)

#define HIPDNN_TEST_FE_KNOB_TYPE(X) \
    X(NOT_SET)                      \
    X(SWIZZLE)                      \
    X(TILE_SIZE)                    \
    X(EDGE)                         \
    X(MULTIPLY)                     \
    X(SPLIT_K_BUF)                  \
    X(TILEK)                        \
    X(STAGES)                       \
    X(REDUCTION_MODE)               \
    X(SPLIT_K_SLC)                  \
    X(IDX_MODE)                     \
    X(SPECFILT)                     \
    X(KERNEL_CFG)                   \
    X(WORKSPACE)                    \
    X(TILE_CGA_M)                   \
    X(TILE_CGA_N)                   \
    X(BLOCK_SIZE)                   \
    X(OCCUPANCY)                    \
    X(ARRAY_SIZE_PER_THREAD)        \
    X(SPLIT_COLS)                   \
    X(TILE_ROWS)                    \
    X(TILE_COLS)                    \
    X(LOAD_SIZE)                    \
    X(CTA_COUNT)                    \
    X(STREAM_K)                     \
    X(SPLIT_P_SLC)                  \
    X(TILE_M)                       \
    X(TILE_N)                       \
    X(WARP_SPEC_CFG)

#define HIPDNN_TEST_FE_RNG_DISTRIBUTION(X) \
    X(NOT_SET)                             \
    X(BERNOULLI)                           \
    X(UNIFORM)                             \
    X(NORMAL)

#define HIPDNN_TEST_FE_DESCRIPTOR_TYPE(X)               \
    X(NOT_SET)                                          \
    X(POINTWISE_DESCRIPTOR)                             \
    X(CONVOLUTION_DESCRIPTOR)                           \
    X(ENGINE_DESCRIPTOR)                                \
    X(ENGINECFG_DESCRIPTOR)                             \
    X(ENGINEHEUR_DESCRIPTOR)                            \
    X(EXECUTION_PLAN_DESCRIPTOR)                        \
    X(INTERMEDIATE_INFO_DESCRIPTOR)                     \
    X(KNOB_CHOICE_DESCRIPTOR)                           \
    X(KNOB_INFO_DESCRIPTOR)                             \
    X(LAYOUT_INFO_DESCRIPTOR)                           \
    X(OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)         \
    X(OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) \
    X(OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)   \
    X(OPERATION_POINTWISE_DESCRIPTOR)                   \
    X(OPERATION_GEN_STATS_DESCRIPTOR)                   \
    X(OPERATIONGRAPH_DESCRIPTOR)                        \
    X(VARIANT_PACK_DESCRIPTOR)                          \
    X(TENSOR_DESCRIPTOR)                                \
    X(MATMUL_DESCRIPTOR)                                \
    X(OPERATION_MATMUL_DESCRIPTOR)                      \
    X(OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)      \
    X(REDUCTION_DESCRIPTOR)                             \
    X(OPERATION_REDUCTION_DESCRIPTOR)                   \
    X(OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)              \
    X(RESAMPLE_DESCRIPTOR)                              \
    X(OPERATION_RESAMPLE_FWD_DESCRIPTOR)                \
    X(OPERATION_RESAMPLE_BWD_DESCRIPTOR)                \
    X(OPERATION_CONCAT_DESCRIPTOR)                      \
    X(OPERATION_SIGNAL_DESCRIPTOR)                      \
    X(OPERATION_NORM_FORWARD_DESCRIPTOR)                \
    X(OPERATION_NORM_BACKWARD_DESCRIPTOR)               \
    X(OPERATION_RESHAPE_DESCRIPTOR)                     \
    X(RNG_DESCRIPTOR)                                   \
    X(OPERATION_RNG_DESCRIPTOR)                         \
    X(OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR)            \
    X(OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR)        \
    X(OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR)      \
    X(OPERATION_CONCATENATE_DESCRIPTOR)                 \
    X(OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR)          \
    X(OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR)      \
    X(OPERATION_TRANSPOSE_DESCRIPTOR)                   \
    X(OPERATION_SLICE_DESCRIPTOR)
// clang-format on

template <typename Fe>
struct FeEnumerator
{
    Fe value;
    std::string_view name;
};

template <typename Fe>
struct FeEnumerators;

#define HIPDNN_TEST_FE_ROW(NAME) FeEnumerator<Fe>{Fe::NAME, #NAME},
#define HIPDNN_TEST_FE_CASE(NAME) case Fe::NAME:
#define HIPDNN_TEST_FE_ENUM(FE_T, LIST)            \
    template <>                                    \
    struct FeEnumerators<fe::FE_T>                 \
    {                                              \
        using Fe = fe::FE_T;                       \
        static std::vector<FeEnumerator<Fe>> all() \
        {                                          \
            return {LIST(HIPDNN_TEST_FE_ROW)};     \
        }                                          \
        static constexpr bool listed(Fe value)     \
        {                                          \
            switch(value)                          \
            {                                      \
                LIST(HIPDNN_TEST_FE_CASE)          \
                return true;                       \
            default:                               \
                return false;                      \
            }                                      \
        }                                          \
    };

// A FE enumerator missing from its list above is a compile error here.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
HIPDNN_TEST_FE_ENUM(DataType_t, HIPDNN_TEST_FE_DATA_TYPE)
HIPDNN_TEST_FE_ENUM(PointwiseMode_t, HIPDNN_TEST_FE_POINTWISE_MODE)
HIPDNN_TEST_FE_ENUM(ConvolutionMode_t, HIPDNN_TEST_FE_CONVOLUTION_MODE)
HIPDNN_TEST_FE_ENUM(ReductionMode_t, HIPDNN_TEST_FE_REDUCTION_MODE)
HIPDNN_TEST_FE_ENUM(ResampleMode_t, HIPDNN_TEST_FE_RESAMPLE_MODE)
HIPDNN_TEST_FE_ENUM(PaddingMode_t, HIPDNN_TEST_FE_PADDING_MODE)
HIPDNN_TEST_FE_ENUM(NormFwdPhase_t, HIPDNN_TEST_FE_NORM_FWD_PHASE)
HIPDNN_TEST_FE_ENUM(NormMode_t, HIPDNN_TEST_FE_NORM_MODE)
HIPDNN_TEST_FE_ENUM(BehaviorNote_t, HIPDNN_TEST_FE_BEHAVIOR_NOTE)
HIPDNN_TEST_FE_ENUM(NumericalNote_t, HIPDNN_TEST_FE_NUMERICAL_NOTE)
HIPDNN_TEST_FE_ENUM(TensorReordering_t, HIPDNN_TEST_FE_TENSOR_REORDERING)
HIPDNN_TEST_FE_ENUM(ReshapeMode_t, HIPDNN_TEST_FE_RESHAPE_MODE)
HIPDNN_TEST_FE_ENUM(KnobType_t, HIPDNN_TEST_FE_KNOB_TYPE)
HIPDNN_TEST_FE_ENUM(RngDistribution_t, HIPDNN_TEST_FE_RNG_DISTRIBUTION)
HIPDNN_TEST_FE_ENUM(DescriptorType_t, HIPDNN_TEST_FE_DESCRIPTOR_TYPE)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

template <typename C>
struct Mapping
{
    static constexpr bool MAPPED = false;
};

// C_PREFIX + FE enumerator name = the cuDNN enumerator name.
#define HIPDNN_TEST_MAPPING(C_T, FE_T, PREFIX)               \
    template <>                                              \
    struct Mapping<C_T>                                      \
    {                                                        \
        static constexpr bool MAPPED = true;                 \
        using Fe = fe::FE_T;                                 \
        static constexpr std::string_view C_TYPE = #C_T;     \
        static constexpr std::string_view C_PREFIX = PREFIX; \
    };

HIPDNN_TEST_MAPPING(cudnnDataType_t, DataType_t, "CUDNN_DATA_")
HIPDNN_TEST_MAPPING(cudnnPointwiseMode_t, PointwiseMode_t, "CUDNN_POINTWISE_")
HIPDNN_TEST_MAPPING(cudnnConvolutionMode_t, ConvolutionMode_t, "CUDNN_")
HIPDNN_TEST_MAPPING(cudnnReduceTensorOp_t, ReductionMode_t, "CUDNN_REDUCE_TENSOR_")
HIPDNN_TEST_MAPPING(cudnnResampleMode_t, ResampleMode_t, "CUDNN_RESAMPLE_")
HIPDNN_TEST_MAPPING(cudnnPaddingMode_t, PaddingMode_t, "CUDNN_")
HIPDNN_TEST_MAPPING(cudnnBackendNormFwdPhase_t, NormFwdPhase_t, "CUDNN_NORM_FWD_")
HIPDNN_TEST_MAPPING(cudnnBackendNormMode_t, NormMode_t, "CUDNN_")
HIPDNN_TEST_MAPPING(cudnnBackendBehaviorNote_t, BehaviorNote_t, "CUDNN_BEHAVIOR_NOTE_")
HIPDNN_TEST_MAPPING(cudnnBackendNumericalNote_t, NumericalNote_t, "CUDNN_NUMERICAL_NOTE_")
HIPDNN_TEST_MAPPING(cudnnBackendTensorReordering_t, TensorReordering_t, "CUDNN_TENSOR_REORDERING_")
HIPDNN_TEST_MAPPING(cudnnBackendReshapeMode_t, ReshapeMode_t, "CUDNN_RESHAPE_")
HIPDNN_TEST_MAPPING(cudnnBackendKnobType_t, KnobType_t, "CUDNN_KNOB_TYPE_")
HIPDNN_TEST_MAPPING(cudnnRngDistribution_t, RngDistribution_t, "CUDNN_RNG_DISTRIBUTION_")
HIPDNN_TEST_MAPPING(cudnnBackendDescriptorType_t, DescriptorType_t, "CUDNN_BACKEND_")

using MappedTypes = std::tuple<cudnnDataType_t,
                               cudnnPointwiseMode_t,
                               cudnnConvolutionMode_t,
                               cudnnReduceTensorOp_t,
                               cudnnResampleMode_t,
                               cudnnPaddingMode_t,
                               cudnnBackendNormFwdPhase_t,
                               cudnnBackendNormMode_t,
                               cudnnBackendBehaviorNote_t,
                               cudnnBackendNumericalNote_t,
                               cudnnBackendTensorReordering_t,
                               cudnnBackendReshapeMode_t,
                               cudnnBackendKnobType_t,
                               cudnnRngDistribution_t,
                               cudnnBackendDescriptorType_t>;

template <typename T, typename List>
struct Contains;

template <typename T, typename... Ts>
struct Contains<T, std::tuple<Ts...>> : std::disjunction<std::is_same<T, Ts>...>
{
};

constexpr std::array<std::string_view, 5> UNMAPPED_C_TYPES = {"cudnnBackendHeurMode_t",
                                                              "cudnnBnFinalizeStatsMode_t",
                                                              "cudnnGenStatsMode_t",
                                                              "cudnnNanPropagation_t",
                                                              "cudnnTensorFormat_t"};

// FE enumerators whose counterpart is not the same-named cuDNN enumerator;
// nullopt means no counterpart. NOT_SET never maps and is not listed.
struct Exception
{
    std::string_view cType;
    std::string_view enumerator;
    std::optional<std::string_view> counterpart;
};

constexpr std::array<Exception, 10> FE_EXCEPTIONS = {
    Exception{"cudnnDataType_t", "FP6_E2M3", std::nullopt},
    Exception{"cudnnDataType_t", "FP6_E3M2", std::nullopt},
    Exception{"cudnnDataType_t", "FP8_E4M3_FNUZ", std::nullopt},
    Exception{"cudnnDataType_t", "FP8_E5M2_FNUZ", std::nullopt},
    Exception{"cudnnPointwiseMode_t", "COUNT", std::nullopt},
    Exception{"cudnnBackendBehaviorNote_t", "REQUIRES_LAYOUT_TRANSFORM", std::nullopt},
    Exception{"cudnnBackendBehaviorNote_t", "SUPPORTS_GRAPH_CAPTURE", std::nullopt},
    Exception{"cudnnBackendBehaviorNote_t", "EXTERNAL_LIBRARY_DEPENDENCY", std::nullopt},
    Exception{"cudnnBackendBehaviorNote_t", "SUPPORTS_EXECUTION_PLAN_SERIALIZATION", std::nullopt},
    // Upstream converts both CONCAT spellings to the one backend type.
    Exception{"cudnnBackendDescriptorType_t",
              "OPERATION_CONCATENATE_DESCRIPTOR",
              "CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR"},
};

template <std::size_t N>
const Exception* findException(const std::array<Exception, N>& exceptions,
                               std::string_view cType,
                               std::string_view enumerator)
{
    for(const Exception& exception : exceptions)
    {
        if(exception.cType == cType && exception.enumerator == enumerator)
        {
            return &exception;
        }
    }
    return nullptr;
}

struct CEnumerator
{
    std::string_view cType;
    std::string_view name;
    std::int64_t value;
};

const std::vector<CEnumerator>& cEnumerators()
{
    static const std::vector<CEnumerator> s_all = {
#define HIPDNN_CUDNN_SHIM_ENUM_VALUE(ENUM_T, NAME, VALUE) {#ENUM_T, #NAME, VALUE},
#define HIPDNN_CUDNN_SHIM_ENUM_COUNT(ENUM_T, NAME, VALUE) {#ENUM_T, #NAME, VALUE},
#define HIPDNN_CUDNN_SHIM_ENUM_ALIAS(ENUM_T, NAME, TARGET)
#include "data/cudnn_v8/cudnn_v8_enum_values.inc"
#undef HIPDNN_CUDNN_SHIM_ENUM_VALUE
#undef HIPDNN_CUDNN_SHIM_ENUM_COUNT
#undef HIPDNN_CUDNN_SHIM_ENUM_ALIAS
    };
    return s_all;
}

std::optional<std::string_view> cName(std::string_view cType, std::int64_t value)
{
    for(const CEnumerator& enumerator : cEnumerators())
    {
        if(enumerator.cType == cType && enumerator.value == value)
        {
            return enumerator.name;
        }
    }
    return std::nullopt;
}

template <typename Fe>
std::optional<std::string_view> feName(Fe value)
{
    for(const FeEnumerator<Fe>& enumerator : FeEnumerators<Fe>::all())
    {
        if(enumerator.value == value)
        {
            return enumerator.name;
        }
    }
    return std::nullopt;
}

class CSideChecker
{
public:
    std::set<std::string_view> mappedTypes;
    std::set<std::string_view> unmappedTypes;

    // A COUNT sentinel has no FE counterpart.
    template <typename C>
    void visit(C value, std::string_view cType, std::string_view name, bool isCount)
    {
        if constexpr(Mapping<C>::MAPPED)
        {
            static_assert(Contains<C, MappedTypes>::value, "add the type to MappedTypes");
            mappedTypes.insert(cType);
            check<C>(value, name, isCount);
        }
        else
        {
            unmappedTypes.insert(cType);
        }
    }

private:
    template <typename C>
    void check(C value, std::string_view name, bool isCount)
    {
        using Fe = typename Mapping<C>::Fe;
        const std::optional<Fe> result = shim::toFe(value);
        if(isCount)
        {
            EXPECT_FALSE(result.has_value()) << name;
            return;
        }
        ASSERT_EQ(name.substr(0, Mapping<C>::C_PREFIX.size()), Mapping<C>::C_PREFIX) << name;
        const std::string_view expected = name.substr(Mapping<C>::C_PREFIX.size());
        ASSERT_TRUE(result.has_value()) << name << " must map to " << expected;
        EXPECT_EQ(feName(*result), expected) << name;
        EXPECT_EQ(shim::toCudnn(*result), value) << name << " must round-trip";
    }
};

class FeSideChecker
{
public:
    std::set<std::string_view> usedExceptions;

    template <typename C>
    void check()
    {
        using Fe = typename Mapping<C>::Fe;
        for(const FeEnumerator<Fe>& enumerator : FeEnumerators<Fe>::all())
        {
            const std::optional<C> result = shim::toCudnn(enumerator.value);
            if(enumerator.name == "NOT_SET")
            {
                EXPECT_FALSE(result.has_value()) << Mapping<C>::C_TYPE;
                continue;
            }

            std::optional<std::string> expected;
            const Exception* exception
                = findException(FE_EXCEPTIONS, Mapping<C>::C_TYPE, enumerator.name);
            if(exception != nullptr)
            {
                usedExceptions.insert(enumerator.name);
                if(exception->counterpart)
                {
                    expected = std::string(*exception->counterpart);
                }
            }
            else
            {
                expected = std::string(Mapping<C>::C_PREFIX) + std::string(enumerator.name);
            }

            if(!expected)
            {
                EXPECT_FALSE(result.has_value())
                    << enumerator.name << " must have no cuDNN counterpart";
                continue;
            }
            ASSERT_TRUE(result.has_value()) << enumerator.name << " must map to " << *expected;
            EXPECT_EQ(cName(Mapping<C>::C_TYPE, static_cast<std::int64_t>(*result)), *expected);
            if(exception == nullptr)
            {
                EXPECT_EQ(shim::toFe(*result), enumerator.value)
                    << enumerator.name << " must round-trip";
            }
        }
    }
};

// Each table row maps both ways: its FE value to its cuDNN value, and its cuDNN
// value to an FE value that maps back to it.
template <typename Table>
constexpr bool rowsRoundTrip(const Table& table)
{
    for(const auto& row : table)
    {
        const auto fe = shim::toFe(row.cudnn);
        if(shim::toCudnn(row.fe) != row.cudnn || !fe || shim::toCudnn(*fe) != row.cudnn)
        {
            return false;
        }
    }
    return true;
}

// Rows whose FE value aliases another row's cuDNN value, so toFe returns the other FE value.
template <typename Table>
constexpr int feAliasRows(const Table& table)
{
    int aliases = 0;
    for(const auto& row : table)
    {
        aliases += shim::toFe(row.cudnn) == row.fe ? 0 : 1;
    }
    return aliases;
}

static_assert(rowsRoundTrip(shim::DATA_TYPE_MAP) && feAliasRows(shim::DATA_TYPE_MAP) == 0);
static_assert(rowsRoundTrip(shim::POINTWISE_MODE_MAP)
              && feAliasRows(shim::POINTWISE_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::CONVOLUTION_MODE_MAP)
              && feAliasRows(shim::CONVOLUTION_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::REDUCTION_MODE_MAP)
              && feAliasRows(shim::REDUCTION_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::RESAMPLE_MODE_MAP) && feAliasRows(shim::RESAMPLE_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::PADDING_MODE_MAP) && feAliasRows(shim::PADDING_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::NORM_FWD_PHASE_MAP)
              && feAliasRows(shim::NORM_FWD_PHASE_MAP) == 0);
static_assert(rowsRoundTrip(shim::NORM_MODE_MAP) && feAliasRows(shim::NORM_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::BEHAVIOR_NOTE_MAP) && feAliasRows(shim::BEHAVIOR_NOTE_MAP) == 0);
static_assert(rowsRoundTrip(shim::NUMERICAL_NOTE_MAP)
              && feAliasRows(shim::NUMERICAL_NOTE_MAP) == 0);
static_assert(rowsRoundTrip(shim::TENSOR_REORDERING_MAP)
              && feAliasRows(shim::TENSOR_REORDERING_MAP) == 0);
static_assert(rowsRoundTrip(shim::RESHAPE_MODE_MAP) && feAliasRows(shim::RESHAPE_MODE_MAP) == 0);
static_assert(rowsRoundTrip(shim::KNOB_TYPE_MAP) && feAliasRows(shim::KNOB_TYPE_MAP) == 0);
static_assert(rowsRoundTrip(shim::RNG_DISTRIBUTION_MAP)
              && feAliasRows(shim::RNG_DISTRIBUTION_MAP) == 0);
// OPERATION_CONCATENATE_DESCRIPTOR aliases OPERATION_CONCAT_DESCRIPTOR.
static_assert(rowsRoundTrip(shim::DESCRIPTOR_TYPE_MAP)
              && feAliasRows(shim::DESCRIPTOR_TYPE_MAP) == 1);
static_assert(shim::toFe(CUDNN_RESAMPLE_AVGPOOL) == fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING);
static_assert(!shim::toCudnn(fe::DataType_t::NOT_SET).has_value());

} // namespace

TEST(TestCudnnShimV8TypeMapping, CEnumeratorsMapToSameNamedFeEnumerators)
{
    CSideChecker checker;
#define HIPDNN_CUDNN_SHIM_ENUM_VALUE(ENUM_T, NAME, VALUE) \
    checker.visit(NAME, #ENUM_T, #NAME, false);
#define HIPDNN_CUDNN_SHIM_ENUM_COUNT(ENUM_T, NAME, VALUE) checker.visit(NAME, #ENUM_T, #NAME, true);
#define HIPDNN_CUDNN_SHIM_ENUM_ALIAS(ENUM_T, NAME, TARGET)
#include "data/cudnn_v8/cudnn_v8_enum_values.inc"
#undef HIPDNN_CUDNN_SHIM_ENUM_VALUE
#undef HIPDNN_CUDNN_SHIM_ENUM_COUNT
#undef HIPDNN_CUDNN_SHIM_ENUM_ALIAS

    EXPECT_EQ(checker.mappedTypes.size(), std::tuple_size_v<MappedTypes>);
    EXPECT_EQ(checker.unmappedTypes,
              std::set<std::string_view>(UNMAPPED_C_TYPES.begin(), UNMAPPED_C_TYPES.end()))
        << "every cuDNN C enum needs a mapping decision";
}

TEST(TestCudnnShimV8TypeMapping, FeEnumeratorsMapBackToSameNamedCEnumerators)
{
    FeSideChecker checker;
    std::apply([&](auto... tags) { (checker.check<decltype(tags)>(), ...); }, MappedTypes{});
    for(const Exception& exception : FE_EXCEPTIONS)
    {
        EXPECT_EQ(checker.usedExceptions.count(exception.enumerator), 1u)
            << "stale exception " << exception.enumerator;
    }
}
