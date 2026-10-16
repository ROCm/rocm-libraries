// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Exhaustive by-name checks of the cuDNN C enum <-> shim FE enum tables and the
// SET-time range check for enum-valued attributes. The C side is enumerated
// from cudnn_v8_enum_values.inc, so a new C enumerator without a mapping
// decision fails here; the FE lists are guarded by -Wswitch-enum.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>
#include <hipdnn_compatibility/cudnn/detail/backend/type_mapping.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

namespace
{

namespace backend = hipdnn_frontend::compatibility::cudnn_frontend::detail::backend;
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

#define HIPDNN_TEST_FE_HEUR_MODE(X) \
    X(A)                            \
    X(B)                            \
    X(FALLBACK)                     \
    X(OPENSOURCE)

#define HIPDNN_TEST_FE_RNG_DISTRIBUTION(X) \
    X(NOT_SET)                             \
    X(BERNOULLI)                           \
    X(UNIFORM)                             \
    X(NORMAL)

#define HIPDNN_TEST_FE_MOE_GROUPED_MATMUL_MODE(X) \
    X(NOT_SET)                                    \
    X(NONE)                                       \
    X(GATHER)                                     \
    X(SCATTER)

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
HIPDNN_TEST_FE_ENUM(HeurMode_t, HIPDNN_TEST_FE_HEUR_MODE)
HIPDNN_TEST_FE_ENUM(RngDistribution_t, HIPDNN_TEST_FE_RNG_DISTRIBUTION)
HIPDNN_TEST_FE_ENUM(MoeGroupedMatmulMode_t, HIPDNN_TEST_FE_MOE_GROUPED_MATMUL_MODE)
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
HIPDNN_TEST_MAPPING(cudnnBackendHeurMode_t, HeurMode_t, "CUDNN_HEUR_MODE_")
HIPDNN_TEST_MAPPING(cudnnRngDistribution_t, RngDistribution_t, "CUDNN_RNG_DISTRIBUTION_")
HIPDNN_TEST_MAPPING(cudnnMoeGroupedMatmulMode_t,
                    MoeGroupedMatmulMode_t,
                    "CUDNN_MOE_GROUPED_MATMUL_MODE_")
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
                               cudnnBackendHeurMode_t,
                               cudnnRngDistribution_t,
                               cudnnMoeGroupedMatmulMode_t,
                               cudnnBackendDescriptorType_t>;

template <typename T, typename List>
struct Contains;

template <typename T, typename... Ts>
struct Contains<T, std::tuple<Ts...>> : std::disjunction<std::is_same<T, Ts>...>
{
};

constexpr std::array<std::string_view, 6> UNMAPPED_C_TYPES = {"cudnnBackendAttributeName_t",
                                                              "cudnnBackendAttributeType_t",
                                                              "cudnnBnFinalizeStatsMode_t",
                                                              "cudnnGenStatsMode_t",
                                                              "cudnnNanPropagation_t",
                                                              "cudnnTensorFormat_t"};

constexpr std::array<std::string_view, 4> COUNT_SENTINELS = {"CUDNN_BEHAVIOR_NOTE_TYPE_COUNT",
                                                             "CUDNN_HEUR_MODES_COUNT",
                                                             "CUDNN_KNOB_TYPE_COUNTS",
                                                             "CUDNN_NUMERICAL_NOTE_TYPE_COUNT"};

// Enumerators whose counterpart is not the same-named enumerator; nullopt means
// no counterpart. NOT_SET and the *_COUNT sentinels never map and are not listed.
struct Exception
{
    std::string_view cType;
    std::string_view enumerator;
    std::optional<std::string_view> counterpart;
};

constexpr std::array<Exception, 2> C_EXCEPTIONS = {
    // cuDNN declares AVGPOOL as an alias of AVGPOOL_INCLUDE_PADDING.
    Exception{"cudnnResampleMode_t", "CUDNN_RESAMPLE_AVGPOOL", "AVGPOOL_INCLUDE_PADDING"},
    Exception{"cudnnBackendHeurMode_t", "CUDNN_HEUR_MODE_INSTANT", std::nullopt},
};

constexpr std::array<Exception, 11> FE_EXCEPTIONS = {
    Exception{"cudnnDataType_t", "FP6_E2M3", std::nullopt},
    Exception{"cudnnDataType_t", "FP6_E3M2", std::nullopt},
    Exception{"cudnnDataType_t", "FP8_E4M3_FNUZ", std::nullopt},
    Exception{"cudnnDataType_t", "FP8_E5M2_FNUZ", std::nullopt},
    Exception{"cudnnPointwiseMode_t", "COUNT", std::nullopt},
    Exception{"cudnnBackendBehaviorNote_t", "REQUIRES_LAYOUT_TRANSFORM", std::nullopt},
    Exception{"cudnnBackendBehaviorNote_t", "SUPPORTS_GRAPH_CAPTURE", std::nullopt},
    Exception{"cudnnBackendBehaviorNote_t", "EXTERNAL_LIBRARY_DEPENDENCY", std::nullopt},
    Exception{"cudnnBackendBehaviorNote_t", "SUPPORTS_EXECUTION_PLAN_SERIALIZATION", std::nullopt},
    // Open-source heuristics run in the frontend; there is no backend mode.
    Exception{"cudnnBackendHeurMode_t", "OPENSOURCE", std::nullopt},
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

bool isCountSentinel(std::string_view name)
{
    for(const std::string_view sentinel : COUNT_SENTINELS)
    {
        if(sentinel == name)
        {
            return true;
        }
    }
    return false;
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
#include "data/cudnn_v8/cudnn_v8_enum_values.inc"
#undef HIPDNN_CUDNN_SHIM_ENUM_VALUE
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
    std::set<std::string_view> usedExceptions;

    template <typename C>
    void visit(C value, std::string_view cType, std::string_view name)
    {
        if constexpr(Mapping<C>::MAPPED)
        {
            static_assert(Contains<C, MappedTypes>::value, "add the type to MappedTypes");
            mappedTypes.insert(cType);
            check<C>(value, name);
        }
        else
        {
            unmappedTypes.insert(cType);
        }
    }

private:
    template <typename C>
    void check(C value, std::string_view name)
    {
        using Fe = typename Mapping<C>::Fe;
        const std::optional<Fe> result = backend::toFe(value);
        if(isCountSentinel(name))
        {
            EXPECT_FALSE(result.has_value()) << name;
            return;
        }

        std::optional<std::string_view> expected;
        const Exception* exception = findException(C_EXCEPTIONS, Mapping<C>::C_TYPE, name);
        if(exception != nullptr)
        {
            usedExceptions.insert(name);
            expected = exception->counterpart;
        }
        else
        {
            ASSERT_EQ(name.substr(0, Mapping<C>::C_PREFIX.size()), Mapping<C>::C_PREFIX) << name;
            expected = name.substr(Mapping<C>::C_PREFIX.size());
        }

        if(!expected)
        {
            EXPECT_FALSE(result.has_value()) << name << " must have no FE counterpart";
            return;
        }
        ASSERT_TRUE(result.has_value()) << name << " must map to " << *expected;
        EXPECT_EQ(feName(*result), expected) << name;
        if(exception == nullptr)
        {
            EXPECT_EQ(backend::toCudnn(*result), value) << name << " must round-trip";
        }
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
            EXPECT_TRUE(FeEnumerators<Fe>::listed(enumerator.value));
            const std::optional<C> result = backend::toCudnn(enumerator.value);
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
                EXPECT_EQ(backend::toFe(*result), enumerator.value)
                    << enumerator.name << " must round-trip";
            }
        }
    }
};

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

class ComputingDescriptor final : public backend::Descriptor
{
public:
    using Descriptor::Descriptor;
    using Descriptor::storeComputed;

protected:
    void finalizeImpl() override {}
};

static_assert(backend::toFe(CUDNN_DATA_HALF) == fe::DataType_t::HALF);
static_assert(backend::toCudnn(fe::PointwiseMode_t::RELU_FWD) == CUDNN_POINTWISE_RELU_FWD);
static_assert(!backend::toCudnn(fe::DataType_t::NOT_SET).has_value());

} // namespace

TEST(TestCudnnShimV8TypeMapping, CEnumeratorsMapToSameNamedFeEnumerators)
{
    CSideChecker checker;
#define HIPDNN_CUDNN_SHIM_ENUM_VALUE(ENUM_T, NAME, VALUE) checker.visit(NAME, #ENUM_T, #NAME);
#include "data/cudnn_v8/cudnn_v8_enum_values.inc"
#undef HIPDNN_CUDNN_SHIM_ENUM_VALUE

    EXPECT_EQ(checker.mappedTypes.size(), std::tuple_size_v<MappedTypes>);
    EXPECT_EQ(checker.unmappedTypes,
              std::set<std::string_view>(UNMAPPED_C_TYPES.begin(), UNMAPPED_C_TYPES.end()))
        << "every cuDNN C enum needs a mapping decision";
    for(const Exception& exception : C_EXCEPTIONS)
    {
        EXPECT_EQ(checker.usedExceptions.count(exception.enumerator), 1u)
            << "stale exception " << exception.enumerator;
    }
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

TEST(TestCudnnShimV8TypeMapping, ValueCountMatchesDeclaredEnumerators)
{
    const std::map<cudnnBackendAttributeType_t, std::string_view> enumTypes = {
        {CUDNN_TYPE_BEHAVIOR_NOTE, "cudnnBackendBehaviorNote_t"},
        {CUDNN_TYPE_BN_FINALIZE_STATS_MODE, "cudnnBnFinalizeStatsMode_t"},
        {CUDNN_TYPE_CONVOLUTION_MODE, "cudnnConvolutionMode_t"},
        {CUDNN_TYPE_DATA_TYPE, "cudnnDataType_t"},
        {CUDNN_TYPE_GENSTATS_MODE, "cudnnGenStatsMode_t"},
        {CUDNN_TYPE_HEUR_MODE, "cudnnBackendHeurMode_t"},
        {CUDNN_TYPE_KNOB_TYPE, "cudnnBackendKnobType_t"},
        {CUDNN_TYPE_NAN_PROPOGATION, "cudnnNanPropagation_t"},
        {CUDNN_TYPE_NORM_FWD_PHASE, "cudnnBackendNormFwdPhase_t"},
        {CUDNN_TYPE_NORM_MODE, "cudnnBackendNormMode_t"},
        {CUDNN_TYPE_NUMERICAL_NOTE, "cudnnBackendNumericalNote_t"},
        {CUDNN_TYPE_PADDING_MODE, "cudnnPaddingMode_t"},
        {CUDNN_TYPE_POINTWISE_MODE, "cudnnPointwiseMode_t"},
        {CUDNN_TYPE_REDUCTION_OPERATOR_TYPE, "cudnnReduceTensorOp_t"},
        {CUDNN_TYPE_RESAMPLE_MODE, "cudnnResampleMode_t"},
        {CUDNN_TYPE_RESHAPE_MODE, "cudnnBackendReshapeMode_t"},
        {CUDNN_TYPE_RNG_DISTRIBUTION, "cudnnRngDistribution_t"},
        {CUDNN_TYPE_TENSOR_REORDERING_MODE, "cudnnBackendTensorReordering_t"},
    };
    std::size_t attributeTypes = 0;
    for(const CEnumerator& attributeType : cEnumerators())
    {
        if(attributeType.cType != "cudnnBackendAttributeType_t")
        {
            continue;
        }
        ++attributeTypes;
        const auto type = static_cast<cudnnBackendAttributeType_t>(attributeType.value);
        const auto found = enumTypes.find(type);
        if(found == enumTypes.end())
        {
            EXPECT_FALSE(backend::enumValueCount(type).has_value()) << attributeType.name;
            EXPECT_FALSE(backend::isEnumAttributeType(type)) << attributeType.name;
            continue;
        }
        std::int32_t declared = 0;
        for(const CEnumerator& enumerator : cEnumerators())
        {
            if(enumerator.cType == found->second && !isCountSentinel(enumerator.name))
            {
                ++declared;
            }
        }
        EXPECT_GT(declared, 0) << found->second;
        EXPECT_EQ(backend::enumValueCount(type), declared) << attributeType.name;
        EXPECT_TRUE(backend::isEnumAttributeType(type)) << attributeType.name;
    }
    EXPECT_EQ(attributeTypes, static_cast<std::size_t>(CUDNN_TYPE_VOID_PTR) + 1);
}

TEST(TestCudnnShimV8TypeMapping, OutOfRangeSetIsBadParam)
{
    struct Usage
    {
        cudnnBackendAttributeName_t name;
        cudnnBackendAttributeType_t type;
    };
    const std::vector<Usage> usages = {
#define HIPDNN_CUDNN_SHIM_ATTR_USAGE(ATTR, TYPE) {ATTR, TYPE},
#include "data/cudnn_v8/cudnn_v8_attribute_usage.inc"
#undef HIPDNN_CUDNN_SHIM_ATTR_USAGE
    };

    std::set<cudnnBackendAttributeType_t> exercised;
    for(const Usage& usage : usages)
    {
        const std::optional<std::int32_t> valueCount = backend::enumValueCount(usage.type);
        const std::optional<backend::AttributeInfo> info = backend::attributeInfo(usage.name);
        ASSERT_TRUE(info.has_value());
        if(!valueCount || !info->settable)
        {
            continue;
        }
        exercised.insert(usage.type);
        cudnnBackendDescriptor_t descriptor
            = backend::makeHandle(std::make_shared<backend::StorageOnlyDescriptor>(info->owner));

        for(const std::int32_t bad : {-1, *valueCount, std::numeric_limits<std::int32_t>::max()})
        {
            EXPECT_EQ(cudnnBackendSetAttribute(descriptor, usage.name, usage.type, 1, &bad),
                      CUDNN_STATUS_BAD_PARAM)
                << info->name << " = " << bad;
            EXPECT_NE(lastError().find(info->name), std::string::npos) << info->name;
        }
        for(const std::int32_t good : {0, *valueCount - 1})
        {
            EXPECT_EQ(cudnnBackendSetAttribute(descriptor, usage.name, usage.type, 1, &good),
                      CUDNN_STATUS_SUCCESS)
                << info->name << " = " << good;
        }
        EXPECT_EQ(cudnnBackendDestroyDescriptor(descriptor), CUDNN_STATUS_SUCCESS);
    }
    // BEHAVIOR_NOTE and NUMERICAL_NOTE attributes are GET-only (see below).
    EXPECT_EQ(exercised.size(), 16u);
}

TEST(TestCudnnShimV8TypeMapping, OutOfRangeComputedValueIsRejected)
{
    ComputingDescriptor engine(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    const std::array<cudnnBackendNumericalNote_t, 2> numerical
        = {CUDNN_NUMERICAL_NOTE_FFT, CUDNN_NUMERICAL_NOTE_TYPE_COUNT};
    try
    {
        engine.storeComputed(CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                             CUDNN_TYPE_NUMERICAL_NOTE,
                             static_cast<std::int64_t>(numerical.size()),
                             numerical.data());
        ADD_FAILURE() << "CUDNN_NUMERICAL_NOTE_TYPE_COUNT was stored";
    }
    catch(const backend::ShimError& error)
    {
        EXPECT_EQ(error.status(), CUDNN_STATUS_BAD_PARAM);
        EXPECT_NE(std::string(error.what()).find("CUDNN_ATTR_ENGINE_NUMERICAL_NOTE[1]"),
                  std::string::npos)
            << error.what();
    }

    const std::array<cudnnBackendBehaviorNote_t, 2> behavior
        = {CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY,
           CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API};
    EXPECT_NO_THROW(engine.storeComputed(CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                                         CUDNN_TYPE_BEHAVIOR_NOTE,
                                         static_cast<std::int64_t>(behavior.size()),
                                         behavior.data()));
}
