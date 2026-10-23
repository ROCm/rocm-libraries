// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile_grouped_conv_common.hpp"

#include <ck_tile/core.hpp>
#include <ck_tile/host/convolution_parameter.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/ops/gemm.hpp>
#include <ck_tile/ops/grouped_convolution.hpp>
#include <ck_tile/ops/epilogue/cshuffle_epilogue.hpp>
#include <ck_tile/ops/elementwise.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace miopen::solver::ck_tile_detail {
namespace {
using namespace ck_tile;
namespace layout = ck_tile::tensor_layout::convolution;

constexpr int64_t DescriptorLimit = std::numeric_limits<index_t>::max() - 128;

bool IsGfx1250()
{
    int device = 0;
    hipDeviceProp_t props{};
    if(hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&props, device) != hipSuccess)
        return false;
    const std::string arch = props.gcnArchName;
    return arch.substr(0, arch.find(':')) == "gfx1250";
}

// All descriptor products, rounded tile lengths and clear/convert sizes must be
// representable before constructing CK's int32 descriptors (and their products).
bool PackedTensor(const std::array<int64_t, 6>& lengths,
                  const std::array<int64_t, 6>& strides,
                  int rank,
                  bool weight)
{
    int64_t product   = 1;
    const auto append = [&](int dim) {
        const int64_t length = lengths[dim];
        if(length < 1 || length > DescriptorLimit || strides[dim] != product ||
           product > DescriptorLimit / length)
            return false;
        product *= length;
        return true;
    };
    if(!append(2))
        return false;
    if(!weight && !append(0))
        return false;
    for(int dim = rank + 2; dim >= 3; --dim)
        if(!append(dim))
            return false;
    return append(1) && (!weight || append(0));
}

bool ValidArguments(const Arguments& a)
{
    if((a.spatial_dims != 2 && a.spatial_dims != 3) || a.split_k < 1 ||
       a.split_k > DescriptorLimit / 64)
        return false;
    if(!PackedTensor(a.input_lengths, a.input_strides, a.spatial_dims, false) ||
       !PackedTensor(a.weight_lengths, a.weight_strides, a.spatial_dims, true) ||
       !PackedTensor(a.output_lengths, a.output_strides, a.spatial_dims, false))
        return false;
    if(a.input_lengths[0] != a.weight_lengths[0] || a.input_lengths[0] != a.output_lengths[0] ||
       a.input_lengths[1] != a.output_lengths[1] || a.input_lengths[2] != a.weight_lengths[2] ||
       a.weight_lengths[1] != a.output_lengths[2])
        return false;
    int64_t padded_input    = a.input_lengths[0] * a.input_lengths[1] * a.input_lengths[2];
    int64_t backward_groups = 1;
    for(int i = 0; i < a.spatial_dims; ++i)
    {
        if(a.strides[i] < 1 || a.strides[i] > DescriptorLimit || a.dilations[i] < 1 ||
           a.dilations[i] > DescriptorLimit || a.left_pads[i] < 0 ||
           a.left_pads[i] > DescriptorLimit || a.right_pads[i] < 0 ||
           a.right_pads[i] > DescriptorLimit)
            return false;
        const int64_t effective = (a.weight_lengths[i + 3] - 1) * a.dilations[i] + 1;
        const int64_t padded    = a.input_lengths[i + 3] + a.left_pads[i] + a.right_pads[i];
        if(effective > DescriptorLimit || padded > DescriptorLimit || padded < effective ||
           (padded - effective) / a.strides[i] + 1 != a.output_lengths[i + 3])
            return false;
        if(padded_input > DescriptorLimit / padded)
            return false;
        padded_input *= padded;
        if(a.direction == Direction::BackwardData)
        {
            const int64_t tilde = a.strides[i] / std::gcd(a.strides[i], a.dilations[i]);
            // The BWD constructor enumerates these slices before its support
            // predicate checks its fixed 128-group descriptor capacity.
            if(backward_groups > 128 / tilde)
                return false;
            backward_groups *= tilde;
        }
    }
    return true;
}

size_t TensorElements(const std::array<int64_t, 6>& lengths, int rank)
{
    size_t result = 1;
    for(int i = 0; i < rank + 3; ++i)
        result *= static_cast<size_t>(lengths[i]);
    return result;
}

conv::ConvParam MakeConvParam(const Arguments& a)
{
    const auto spatial = [&](const auto& values, int offset) {
        return std::vector<long_index_t>(values.begin() + offset,
                                         values.begin() + offset + a.spatial_dims);
    };
    conv::ConvParam p(static_cast<long_index_t>(a.spatial_dims),
                      a.input_lengths[0],
                      a.input_lengths[1],
                      a.weight_lengths[1],
                      a.input_lengths[2],
                      spatial(a.weight_lengths, 3),
                      spatial(a.input_lengths, 3),
                      spatial(a.strides, 0),
                      spatial(a.dilations, 0),
                      spatial(a.left_pads, 0),
                      spatial(a.right_pads, 0));
    const auto output = spatial(a.output_lengths, 3);
    if(p.output_spatial_lengths_ != output)
        throw std::invalid_argument("CK Tile convolution output geometry mismatch");
    p.output_spatial_lengths_ = output;
    return p;
}

template <Direction D, typename Traits>
struct DirectionTraits;

template <typename Traits>
struct DirectionTraits<Direction::Forward, Traits>
{
    using HostArgs       = GroupedConvFwdHostArgs<>;
    using AsLayout       = typename Traits::AsLayoutFwd;
    using BsLayout       = typename Traits::BsLayoutFwd;
    using CLayout        = typename Traits::CLayoutFwd;
    using ImplicitTraits = typename Traits::template GroupedConvImplicitGemmTraitsFwd<1>;
    template <typename Partitioner, typename Pipeline, typename Epilogue>
    using Kernel = GroupedConvolutionForwardKernel<Traits, Partitioner, Pipeline, Epilogue>;
    static HostArgs MakeHostArgs(const Arguments& a, const conv::ConvParam& p)
    {
        return {p, a.input, a.weight, {}, const_cast<void*>(a.output), a.split_k};
    }
};

template <typename Traits>
struct DirectionTraits<Direction::BackwardData, Traits>
{
    using HostArgs       = GroupedConvBwdDataHostArgs;
    using AsLayout       = typename Traits::AsLayoutBwdData;
    using BsLayout       = typename Traits::BsLayoutBwdData;
    using CLayout        = typename Traits::CLayoutBwdData;
    using ImplicitTraits = typename Traits::template GroupedConvImplicitGemmTraitsBwdData<1>;
    template <typename Partitioner, typename Pipeline, typename Epilogue>
    using Kernel = GroupedConvolutionBackwardDataKernel<Traits, Partitioner, Pipeline, Epilogue>;
    static HostArgs MakeHostArgs(const Arguments& a, const conv::ConvParam& p)
    {
        return {p, const_cast<void*>(a.input), a.weight, {}, a.output, a.split_k};
    }
};

template <typename Traits>
struct DirectionTraits<Direction::BackwardWeight, Traits>
{
    using HostArgs       = GroupedConvBwdWeightHostArgs;
    using AsLayout       = typename Traits::AsLayoutBwdWeight;
    using BsLayout       = typename Traits::BsLayoutBwdWeight;
    using CLayout        = typename Traits::CLayoutBwdWeight;
    using ImplicitTraits = typename Traits::template GroupedConvImplicitGemmTraitsBwdWeight<1>;
    template <typename Partitioner, typename Pipeline, typename Epilogue>
    using Kernel = GroupedConvolutionBackwardWeightKernel<Traits, Partitioner, Pipeline, Epilogue>;
    static HostArgs MakeHostArgs(const Arguments& a, const conv::ConvParam& p)
    {
        return {p, a.input, const_cast<void*>(a.weight), {}, a.output, a.split_k};
    }
};

template <Pipeline P>
struct PipelineSelector;

template <>
struct PipelineSelector<Pipeline::CompV3>
{
    template <typename Problem>
    using Type = GemmPipelineAgBgCrCompV3<Problem, GroupedConvUniversalPipelineAgBgCrPolicy>;
};

template <>
struct PipelineSelector<Pipeline::BasicV1>
{
    template <typename Problem>
    using Type = GemmPipelineAGmemBGmemCRegV1<Problem, GroupedConvUniversalPipelineAgBgCrPolicy>;
};

template <>
struct PipelineSelector<Pipeline::Memory>
{
    template <typename Problem>
    using Type = GemmPipelineAgBgCrMem<Problem, GroupedConvUniversalPipelineAgBgCrPolicy>;
};

// Native WMMA, fully padded, intrawave and single LDS, using the grouped-conv
// policy and CShuffle recipe shared by the CK examples and generator.
template <Direction D, int Rank, typename T, size_t ConfigIndex, bool FloatBwdScratch = false>
struct Engine
{
    static constexpr auto Config = KernelConfigs[ConfigIndex];
    static constexpr bool TwoStage =
        !std::is_same_v<T, float> && (D == Direction::BackwardWeight || FloatBwdScratch);
    static constexpr index_t WarpK = std::is_same_v<T, float> ? 4 : 32;
    using Result                   = std::conditional_t<TwoStage, float, T>;
    using Shape                    = TileGemmShape<sequence<Config.m, Config.n, Config.k>,
                                                   sequence<Config.wave_m, Config.wave_n, 1>,
                                                   sequence<16, 16, WarpK>>;
    using Traits                   = GroupedConvTraits<Rank,
                                                       ConvolutionSpecialization::Default,
                                                       std::conditional_t<Rank == 2, layout::NHWGC, layout::NDHWGC>,
                                                       std::conditional_t<Rank == 2, layout::GKYXC, layout::GKZYXC>,
                                                       tuple<>,
                                                       std::conditional_t<Rank == 2, layout::NHWGK, layout::NDHWGK>,
                                                       Config.vector_a,
                                                       Config.vector_b,
                                                       Config.vector_c,
                                                       1,
                                                       false,
                                                       false>;
    using Dir                      = DirectionTraits<D, Traits>;
    using Partitioner =
        GemmSpatiallyLocalTilePartitioner<Shape,
                                          Traits::FixedGemmParams::TilePartitionerGroupNum,
                                          Traits::FixedGemmParams::TilePartitionerM01>;
    using UniversalTraits = TileGemmUniversalTraits<true,
                                                    true,
                                                    true,
                                                    false,
                                                    typename Dir::AsLayout,
                                                    typename Dir::BsLayout,
                                                    typename Dir::CLayout,
                                                    false,
                                                    false,
                                                    false,
                                                    1>;
    using Problem         = UniversalGemmPipelineProblem<T,
                                                         T,
                                                         float,
                                                         Shape,
                                                         UniversalTraits,
                                                         GemmPipelineScheduler::Intrawave,
                                                         element_wise::PassThrough,
                                                         element_wise::PassThrough,
                                                         T,
                                                         T,
                                                         true,
                                                         Config.vector_a,
                                                         Config.vector_b>;
    using Pipeline        = typename PipelineSelector<Config.pipeline>::template Type<Problem>;
    using Epilogue =
        CShuffleEpilogue<CShuffleEpilogueProblem<T,
                                                 T,
                                                 tuple<>,
                                                 float,
                                                 Result,
                                                 typename Traits::ImplicitGemmDsLayout,
                                                 typename Traits::FixedGemmParams::ELayout,
                                                 element_wise::PassThrough,
                                                 Config.m,
                                                 Config.n,
                                                 Config.wave_m,
                                                 Config.wave_n,
                                                 16,
                                                 16,
                                                 WarpK,
                                                 false,
                                                 1,
                                                 true,
                                                 Config.vector_c,
                                                 1,
                                                 false>>;
    using Kernel = typename Dir::template Kernel<Partitioner, Pipeline, Epilogue>;

    // Scalar conversion also covers odd channel/filter counts. Keep the existing
    // ElementWise kernel/policy, adjusting the shape's vector width, not the math.
    struct ConvertShape : ElementWiseShape<sequence<8>, sequence<2048>, sequence<64>, float>
    {
        static constexpr index_t kVectorM = 1;
        static constexpr index_t kRepeatM = kBlockM / (kWarpPerBlockM * kThreadPerWarpM);
    };
    using ConvertProblem =
        ElementWisePipelineProblem<float, float, T, ConvertShape, element_wise::UnaryConvert>;
    using ConvertKernel = ElementWiseKernel<ConvertProblem, ElementWiseDefaultPolicy>;

    static size_t ResultElements(const Arguments& a)
    {
        if constexpr(D == Direction::BackwardData)
            return TensorElements(a.input_lengths, Rank);
        else
            return TensorElements(a.weight_lengths, Rank);
    }

    static void* ResultPointer(const Arguments& a)
    {
        if constexpr(D == Direction::BackwardData)
            return const_cast<void*>(a.input);
        else
            return const_cast<void*>(a.weight);
    }

    static auto MakeArgs(const Arguments& a)
    {
        auto host = Dir::MakeHostArgs(a, MakeConvParam(a));
        if constexpr(TwoStage)
        {
            if constexpr(D == Direction::BackwardData)
                host.in_ptr = a.workspace;
            else
                host.wei_ptr = a.workspace;
        }
        return Kernel::MakeKernelArgs(host);
    }

    template <typename KArgs>
    static bool Matches(const Arguments& a, const KArgs& k)
    {
        if(k.k_batch != a.split_k)
            return false;
        for(int i = 0; i < Rank + 3; ++i)
            if(k.in_g_n_c_wis_lengths[i] != a.input_lengths[i] ||
               k.wei_g_k_c_xs_lengths[i] != a.weight_lengths[i] ||
               k.out_g_n_k_wos_lengths[i] != a.output_lengths[i])
                return false;
        for(int i = 0; i < Rank; ++i)
            if(k.conv_filter_strides[i] != a.strides[i] ||
               k.conv_filter_dilations[i] != a.dilations[i] ||
               k.input_left_pads[i] != a.left_pads[i] || k.input_right_pads[i] != a.right_pads[i])
                return false;
        return true;
    }

    static bool Supported(const Arguments& a)
    {
        const auto k = MakeArgs(a);
        if(!Matches(a, k) || !Kernel::IsSupportedArgument(k))
            return false;
        if constexpr(TwoStage)
            return ConvertKernel::IsSupportedArgument(
                make_tuple(static_cast<index_t>(ResultElements(a))));
        return true;
    }

    static size_t WorkspaceSize(const Arguments& a)
    {
        if constexpr(TwoStage)
        {
            const size_t elements = ResultElements(a);
            if(elements > std::numeric_limits<size_t>::max() / sizeof(float))
                throw std::overflow_error("CK Tile convolution workspace size overflow");
            return elements * sizeof(float);
        }
        return 0;
    }

    static float Enqueue(const Arguments& a, hipStream_t stream)
    {
        if(a.input == nullptr || a.weight == nullptr || a.output == nullptr)
            throw std::invalid_argument("CK Tile convolution requires tensor pointers");
        const auto disjoint =
            [](const void* first, size_t first_bytes, const void* second, size_t second_bytes) {
                const uintptr_t first_address  = reinterpret_cast<uintptr_t>(first);
                const uintptr_t second_address = reinterpret_cast<uintptr_t>(second);
                // Subtractions avoid wrapping either interval's end address.
                if(first_bytes > std::numeric_limits<uintptr_t>::max() - first_address ||
                   second_bytes > std::numeric_limits<uintptr_t>::max() - second_address)
                    return false;
                return first_address <= second_address
                           ? second_address - first_address >= first_bytes
                           : first_address - second_address >= second_bytes;
            };
        const size_t input_bytes  = TensorElements(a.input_lengths, Rank) * sizeof(T);
        const size_t weight_bytes = TensorElements(a.weight_lengths, Rank) * sizeof(T);
        const size_t output_bytes = TensorElements(a.output_lengths, Rank) * sizeof(T);
        bool tensors_disjoint;
        if constexpr(D == Direction::Forward)
            tensors_disjoint = disjoint(a.output, output_bytes, a.input, input_bytes) &&
                               disjoint(a.output, output_bytes, a.weight, weight_bytes);
        else if constexpr(D == Direction::BackwardData)
            tensors_disjoint = disjoint(a.input, input_bytes, a.output, output_bytes) &&
                               disjoint(a.input, input_bytes, a.weight, weight_bytes);
        else
            tensors_disjoint = disjoint(a.weight, weight_bytes, a.input, input_bytes) &&
                               disjoint(a.weight, weight_bytes, a.output, output_bytes);
        if(!tensors_disjoint)
            throw std::invalid_argument("CK Tile convolution result overlaps input storage");
        if constexpr(TwoStage)
        {
            if(a.workspace == nullptr ||
               reinterpret_cast<uintptr_t>(a.workspace) % alignof(float) != 0)
                throw std::invalid_argument("CK Tile convolution requires aligned FP32 workspace");
            const size_t scratch_bytes = WorkspaceSize(a);
            if(!disjoint(a.workspace, scratch_bytes, a.input, input_bytes) ||
               !disjoint(a.workspace, scratch_bytes, a.output, output_bytes) ||
               !disjoint(a.workspace, scratch_bytes, a.weight, weight_bytes))
                throw std::invalid_argument(
                    "CK Tile convolution workspace overlaps tensor storage");
        }
        auto k = MakeArgs(a);
        if(!Matches(a, k) || !Kernel::IsSupportedArgument(k))
            throw std::invalid_argument("Unsupported CK Tile convolution arguments");
        const stream_config config{stream, false};
        const auto preprocess = [&]() {
            if constexpr(D == Direction::BackwardData)
                hip_check_error(
                    hipMemsetAsync(k.in_ptr, 0, ResultElements(a) * sizeof(Result), stream));
            else if constexpr(D == Direction::BackwardWeight)
                if(k.k_batch > 1)
                    hip_check_error(
                        hipMemsetAsync(k.wei_ptr,
                                       0,
                                       TensorElements(a.weight_lengths, Rank) * sizeof(Result),
                                       stream));
        };
        // Each selected pipeline dispatches its own hot loop/tail on device.
        // Enqueue the convolution exactly once, then the scalar scratch conversion.
        const auto launch = [&]() {
            const auto conv =
                make_kernel<1>(Kernel{}, Kernel::GridSize(k), Kernel::BlockSize(), 0, k);
            if constexpr(TwoStage)
            {
                const index_t count = static_cast<index_t>(ResultElements(a));
                const auto lens     = make_tuple(count);
                if(!ConvertKernel::IsSupportedArgument(lens))
                    throw std::invalid_argument("Unsupported CK Tile conversion arguments");
                const auto convert =
                    make_kernel<1>(ConvertKernel{},
                                   dim3(integer_divide_ceil(count, ConvertShape::kBlockM)),
                                   dim3(ConvertKernel::BlockSize()),
                                   0,
                                   lens,
                                   make_tuple(index_t{1}),
                                   make_tuple(index_t{1}),
                                   make_tuple(static_cast<float*>(a.workspace)),
                                   static_cast<T*>(ResultPointer(a)));
                return launch_kernel_time_mask(config, preprocess, conv, convert);
            }
            else
                return launch_kernel_time_mask(config, preprocess, conv);
        };
        return launch();
    }
};

enum class Operation
{
    Support,
    Workspace,
    Run
};

struct Result
{
    bool supported   = false;
    size_t workspace = 0;
    float time       = 0;
};

template <Direction D, int Rank, typename T, size_t ConfigIndex, bool FloatBwdScratch = false>
Result Invoke(const Arguments& a, Operation op, hipStream_t stream)
{
    using E          = Engine<D, Rank, T, ConfigIndex, FloatBwdScratch>;
    constexpr auto c = KernelConfigs[ConfigIndex];
    // Split-K rounding uses this recipe's K tile, not the legacy tile's 64.
    // Validate it before MakeKernelArgs forms signed int32 descriptors.
    int64_t gemm_k = D == Direction::BackwardWeight ? a.input_lengths[1]
                     : D == Direction::Forward      ? a.input_lengths[2]
                                                    : a.weight_lengths[1];
    for(int i = 0; i < Rank; ++i)
        gemm_k *=
            D == Direction::BackwardWeight ? a.output_lengths[i + 3] : a.weight_lengths[i + 3];
    const int64_t grain = static_cast<int64_t>(a.split_k) * c.k;
    if(grain > DescriptorLimit || ((gemm_k + grain - 1) / grain) * c.k > DescriptorLimit)
    {
        if(op == Operation::Run)
            throw std::invalid_argument("CK Tile convolution exceeds split descriptor limits");
        return {};
    }
    // Tile buffer offsets are signed int32 bytes even when element descriptors
    // fit int32; FP32 scratch results must obey that same byte limit.
    const auto max_elements = static_cast<size_t>(DescriptorLimit) / sizeof(T);
    const auto max_result_elements =
        static_cast<size_t>(DescriptorLimit) / sizeof(typename E::Result);
    const bool byte_extent_valid =
        TensorElements(a.input_lengths, Rank) <=
            (D == Direction::BackwardData ? max_result_elements : max_elements) &&
        TensorElements(a.output_lengths, Rank) <= max_elements &&
        TensorElements(a.weight_lengths, Rank) <=
            (D == Direction::BackwardWeight ? max_result_elements : max_elements);
    if(!byte_extent_valid)
    {
        if(op == Operation::Run)
            throw std::invalid_argument("CK Tile convolution exceeds buffer byte-offset limits");
        return {};
    }
    if(op == Operation::Run)
        return {true, 0, E::Enqueue(a, stream)};
    if(!E::Supported(a))
        return {};
    return {true, op == Operation::Workspace ? E::WorkspaceSize(a) : 0, 0};
}

template <Direction D, int Rank, typename T, size_t ConfigIndex = 0>
Result DispatchConfig(const Arguments& a, Operation op, hipStream_t stream)
{
    if(a.config_index == static_cast<int>(ConfigIndex))
    {
        constexpr auto c = KernelConfigs[ConfigIndex];
        // Native FP32 WMMA cannot consume vector8. Do not instantiate those types.
        if constexpr(!std::is_same_v<T, float> ||
                     (c.vector_a <= 4 && c.vector_b <= 4 && c.vector_c <= 4))
        {
            // Accumulate low-precision BWD split partials in FP32 and round once.
            // Support, workspace queries and execution all select the same type.
            if constexpr(D == Direction::BackwardData && !std::is_same_v<T, float>)
            {
                // Preserve the scalar low-precision split-K support contract.
                if(a.split_k > 1)
                {
                    if constexpr(c.vector_c % 2 != 0)
                        return {};
                    else
                        return Invoke<D, Rank, T, ConfigIndex, true>(a, op, stream);
                }
            }
            return Invoke<D, Rank, T, ConfigIndex>(a, op, stream);
        }
        else
            return {};
    }
    if constexpr(ConfigIndex + 1 < KernelConfigs.size())
        return DispatchConfig<D, Rank, T, ConfigIndex + 1>(a, op, stream);
    return {};
}

template <Direction D, int Rank>
Result DispatchType(const Arguments& a, Operation op, hipStream_t stream)
{
    switch(a.data_type)
    {
    case DataType::Half: return DispatchConfig<D, Rank, fp16_t>(a, op, stream);
    case DataType::BFloat16: return DispatchConfig<D, Rank, bf16_t>(a, op, stream);
    case DataType::Float: return DispatchConfig<D, Rank, float>(a, op, stream);
    }
    return {};
}

template <int Rank>
Result DispatchDirection(const Arguments& a, Operation op, hipStream_t stream)
{
    switch(a.direction)
    {
    case Direction::Forward: return DispatchType<Direction::Forward, Rank>(a, op, stream);
    case Direction::BackwardData: return DispatchType<Direction::BackwardData, Rank>(a, op, stream);
    case Direction::BackwardWeight:
        return DispatchType<Direction::BackwardWeight, Rank>(a, op, stream);
    }
    return {};
}

Result Dispatch(const Arguments& a, Operation op, hipStream_t stream = nullptr)
{
    if(!ValidArguments(a) || !IsGfx1250())
    {
        if(op == Operation::Run)
            throw std::invalid_argument(
                "CK Tile convolution requires supported packed gfx1250 arguments");
        return {};
    }
    return a.spatial_dims == 2 ? DispatchDirection<2>(a, op, stream)
                               : DispatchDirection<3>(a, op, stream);
}
} // namespace

bool IsSupported(const Arguments& a) { return Dispatch(a, Operation::Support).supported; }

size_t GetWorkspaceSize(const Arguments& a) { return Dispatch(a, Operation::Workspace).workspace; }

float Run(const Arguments& a, hipStream_t stream)
{
    const auto result = Dispatch(a, Operation::Run, stream);
    if(!result.supported)
        throw std::invalid_argument("Invalid CK Tile convolution direction or data type");
    return result.time;
}

std::string GetKernelId(Direction direction, DataType data_type, int spatial_dims, int config_index)
{
    if((spatial_dims != 2 && spatial_dims != 3) || config_index < 0 ||
       static_cast<size_t>(config_index) >= KernelConfigs.size())
        return {};
    const auto& c = KernelConfigs[config_index];
    if(data_type == DataType::Float && (c.vector_a > 4 || c.vector_b > 4 || c.vector_c > 4))
        return {};
    const char* dir  = nullptr;
    const char* type = nullptr;
    switch(direction)
    {
    case Direction::Forward: dir = "Fwd"; break;
    case Direction::BackwardData: dir = "BwdData"; break;
    case Direction::BackwardWeight: dir = "BwdWeight"; break;
    }
    switch(data_type)
    {
    case DataType::Half: type = "fp16"; break;
    case DataType::BFloat16: type = "bf16"; break;
    case DataType::Float: type = "fp32"; break;
    }
    if(dir == nullptr || type == nullptr)
        return {};
    const auto prefix =
        std::string("CKTileGroupedConv") + dir + "<" + std::to_string(spatial_dims) + ", " + type;
    if(config_index == 0)
        return prefix + ", CompV3, " + std::to_string(c.m) + "x" + std::to_string(c.n) + "x" +
               std::to_string(c.k) + ", vec" + std::to_string(c.vector_a) + ">";
    const char* pipeline = c.pipeline == Pipeline::CompV3    ? "CompV3"
                           : c.pipeline == Pipeline::BasicV1 ? "BasicV1"
                                                             : "Memory";
    // Recipe v2 records variable and fixed policies, including scratch accumulation.
    // Low-precision BWD selects float scratch only for split-K greater than one.
    return prefix + ", recipe2, " + pipeline + ", " + std::to_string(c.m) + "x" +
           std::to_string(c.n) + "x" + std::to_string(c.k) + ", waves" + std::to_string(c.wave_m) +
           "x" + std::to_string(c.wave_n) + "x1, warp16x16x" +
           (data_type == DataType::Float ? "4" : "32") + ", vec" + std::to_string(c.vector_a) +
           "x" + std::to_string(c.vector_b) + "x" + std::to_string(c.vector_c) +
           ", Intrawave, padMNK, spatiallyLocal8x4, merge1, Default, waveGroups1, DSB0, "
           "CShuffle-passThrough-fixedVec-blockedN1-transpose0, accumFP32, " +
           (direction == Direction::BackwardWeight && data_type != DataType::Float
                ? "scratchFP32-scalarCast>"
            : direction == Direction::BackwardData && data_type != DataType::Float
                ? "scratchFP32-ifSplitGT1-scalarCast>"
                : "scratchNone>");
}
} // namespace miopen::solver::ck_tile_detail
