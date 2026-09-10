// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule_executor.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_v128_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_v128_schedule.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"

namespace tdm_v128_test::lifetime {

enum class Operand
{
    Q,
    K,
    V,
};

struct Destination
{
    Operand operand;
    int tile;
    int stage;

    bool operator==(const Destination& other) const
    {
        return operand == other.operand && tile == other.tile && stage == other.stage;
    }
};

struct Location
{
    int iteration = -1;
    bool qk       = false;
    int stage     = -1;
    int wmma      = -1;
};

enum class ObligationKind
{
    CompilerConsumerCompletion,
    DestinationLayout,
    PhysicalRegisterLease,
    PhysicalTensorIssueCount,
    HardwareWaitAndBarrierLowering,
};

struct Obligation
{
    ObligationKind kind;
    Location consumer;
    Destination destination;
    std::string requirement;
    int access        = -1;
    int maximum_dscnt = -1;
};

struct Read
{
    Destination destination;
    int access;
    int lds_bank;
    Location producer;
    int issue_event;
    int first_consumer_event = -1;
    int last_consumer_event  = -1;
    int release_event        = -1;
    bool completed           = false;
    Location first_consumer{};
    Location last_consumer{};
};

struct Result
{
    std::vector<std::string> errors;
    std::vector<Obligation> isa_obligations;
    std::vector<Read> reads;
    int wmma_count               = 0;
    int compiler_fence_count     = 0;
    int tensor_issue_count       = 0;
    int unused_next_k_reads      = 0;
    int backedges                = 0;
    int pending_ds_at_exit       = 0;
    int pending_tensors_at_exit  = 0;
    bool exact_operand_consumers = false;

    // This is conditional logical validity, never a kernel/ISA acceptance decision.
    bool LogicalContractHolds() const { return errors.empty(); }

    bool HasError(const std::string& text) const
    {
        return std::any_of(errors.begin(), errors.end(), [&](const auto& error) {
            return error.find(text) != std::string::npos;
        });
    }

    int CountObligations(ObligationKind kind) const
    {
        return std::count_if(isa_obligations.begin(), isa_obligations.end(), [&](const auto& item) {
            return item.kind == kind;
        });
    }
};

template <typename Geometry>
struct Options
{
    int iterations = 1;
    std::array<int, Geometry::kQkStages> qk_tails{};
    std::array<int, Geometry::kPvStages> pv_tails{};
    int k_tensor_wait    = 0;
    int v_tensor_wait    = 0;
    int v_reuse_ds_limit = Geometry::kKSuLoadCount;
    int k_reuse_ds_limit = Geometry::kVStageLoadCount;

    // These describe required control-flow edges, not extra emitted instructions.
    bool empty_work_returns_before_prologue   = true;
    bool require_compiler_consumer_completion = true;
    bool q_reuse_barrier                      = true;
    bool v_reuse_barrier                      = true;
    bool k_reuse_barrier                      = true;
    bool epilogue_ds_drain                    = true;
    bool epilogue_tensor_drain                = true;
};

template <typename Geometry, typename Tuning>
Options<Geometry> OptionsFromTuning(int iterations)
{
    static_assert(Tuning::QkTailDsCounts::size() == Geometry::kQkStages);
    static_assert(Tuning::PvTailDsCounts::size() == Geometry::kPvStages);
    Options<Geometry> options;
    options.iterations            = iterations;
    options.k_tensor_wait         = Tuning::kKWait;
    options.v_tensor_wait         = Tuning::kVWait;
    options.epilogue_tensor_drain = Tuning::kTailDrain;
    ck_tile::static_for<0, Geometry::kQkStages, 1>{}(
        [&](auto stage) { options.qk_tails[stage] = Tuning::QkTailDsCounts::at(stage); });
    ck_tile::static_for<0, Geometry::kPvStages, 1>{}(
        [&](auto stage) { options.pv_tails[stage] = Tuning::PvTailDsCounts::at(stage); });
    return options;
}

namespace detail {

template <int Loads, int Wmmas, int Elements>
struct OperandConsumers
{
    std::array<std::array<bool, Loads>, Wmmas> uses{};
    std::array<int, Elements> owner{};
    std::array<int, Loads> first{};
    std::array<int, Loads> last{};
    bool valid = true;
};

// Follow the load emitter's SFC/vector scatter and the WMMA emitter's B slice.
// Logical thread-buffer offsets are independent of lane/partition coordinates.
template <typename Window, typename Distribution, typename BlockGemm, bool IsQk>
CK_TILE_HOST_DEVICE constexpr auto MakeOperandConsumers()
{
    using namespace ck_tile;
    using Base                = typename Window::Base;
    using Traits              = typename Base::Traits;
    using SFC                 = typename Traits::SFC_Ys;
    using Warp                = typename BlockGemm::WarpGemm;
    constexpr int loads       = Window::NumAccessPerCoord;
    constexpr int wmmas       = IsQk ? 4 * BlockGemm::KIterPerWarp : 16;
    constexpr auto descriptor = Distribution{}.get_ys_to_d_descriptor();
    constexpr int elements    = descriptor.get_element_space_size();
    static_assert(Traits::PackedSize == 1 && sizeof(typename Traits::vector_t) == 16);
    static_assert(BlockGemm::MIterPerWarp == 2);
    static_assert(IsQk ? BlockGemm::NIterPerWarp == 2
                       : BlockGemm::NIterPerWarp == 8 && BlockGemm::KIterPerWarp == 1);
    OperandConsumers<loads, wmmas, elements> result;
    for(auto& owner : result.owner)
        owner = -1;
    for(auto& first : result.first)
        first = -1;
    for(auto& last : result.last)
        last = -1;
    static_for<0, loads, 1>{}([&](auto access) {
        constexpr auto input = SFC::get_index(access);
        static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
            constexpr auto source = generate_tuple(
                [&](auto dim) { return input[dim] + (dim == Traits::VectorDimY ? j : 0); },
                number<Base::NDimY>{});
            constexpr auto destination = [&] {
                if constexpr(IsQk)
                    return source;
                else
                    return DefaultTranspose<typename Base::DataType>::group_func(source);
            }();
            constexpr int offset = descriptor.calculate_offset(destination);
            static_assert(offset >= 0 && offset < elements);
            if(result.owner[offset] != -1)
                result.valid = false;
            result.owner[offset] = access;
        });
    });
    for(const auto owner : result.owner)
        if(owner == -1)
            result.valid = false;

    constexpr auto warp_lengths =
        to_sequence(typename Warp::BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    static_for<0, wmmas, 1>{}([&](auto ordinal) {
        constexpr int wmma = decltype(ordinal)::value;
        constexpr int n =
            IsQk ? wmma / (2 * BlockGemm::KIterPerWarp) : 2 * (wmma % 4) + (wmma / 4) % 2;
        constexpr int k = IsQk ? (wmma / 2) % BlockGemm::KIterPerWarp : 0;
        static_ford<remove_cvref_t<decltype(warp_lengths)>>{}([&](auto inner) {
            constexpr auto index = [&] {
                if constexpr(IsQk)
                    return merge_sequences(sequence<n, k>{}, inner);
                else
                    return merge_sequences(sequence<k, n>{}, inner);
            }();
            constexpr int offset = descriptor.calculate_offset(index);
            static_assert(offset >= 0 && offset < elements);
            const int access = result.owner[offset];
            if(access < 0 || access >= loads)
            {
                result.valid = false;
                return;
            }
            result.uses[wmma][access] = true;
            if(result.first[access] == -1)
                result.first[access] = wmma;
            result.last[access] = wmma;
        });
    });
    for(int access = 0; access < loads; ++access)
        if(result.first[access] < 0 || result.last[access] < result.first[access])
            result.valid = false;
    return result;
}

template <typename Geometry>
struct NativeBf16Operands
{
    using Data     = ck_tile::bf16_t;
    using Warps    = ck_tile::sequence<4, 1, 1>;
    using WarpTile = ck_tile::sequence<16, 16, 32>;
    struct Problem
    {
        using QDataType    = Data;
        using KDataType    = Data;
        using VDataType    = Data;
        using PDataType    = Data;
        using SaccDataType = float;
        using OaccDataType = float;
        using BlockFmhaShape =
            ck_tile::TileFmhaShape<ck_tile::sequence<128, 128, 32, 128, 32, Geometry::kHeadDimQK>,
                                   Warps,
                                   WarpTile,
                                   Warps,
                                   WarpTile,
                                   true>;
        static constexpr int kBlockSize = 128;
    };
    using Policy =
        ck_tile::BlockFmhaPipelineQRKSVSTdmV128Policy<Geometry,
                                                      ck_tile::FmhaTdmV128DefaultTuning<Geometry>,
                                                      ck_tile::FmhaTdmV128ScheduleFor<Geometry>>;
    // Host compilation otherwise selects MFMA. Device-pass assertions below bind
    // these native encodings and windows to the real gfx125 policy selection.
    using QkWarp =
        ck_tile::WarpGemmWmma_f32_16x16x32_bf16_bf16<true, ck_tile::WGAttrNumAccessEnum::Default>;
    using PvWarp =
        ck_tile::WarpGemmWmma_f32_16x16x32_bf16_bf16<true, ck_tile::WGAttrNumAccessEnum::Double>;
    template <int N, int K, typename Warp, ck_tile::GemmLoopOrder Order>
    using Gemm = ck_tile::BlockGemmARegBRegCRegV2<
        ck_tile::BlockGemmProblem<
            Data,
            Data,
            float,
            128,
            ck_tile::TileGemmShape<ck_tile::sequence<128, N, K>, Warps, WarpTile>>,
        ck_tile::BlockGemmARegBRegCRegV2CustomPolicy<Data, Data, float, Warps, Warp, Order>>;
    using Qk                = Gemm<32, Geometry::kHeadDimQK, QkWarp, ck_tile::GemmLoopOrder::MNK>;
    using Pv                = Gemm<128, 32, PvWarp, ck_tile::GemmLoopOrder::KMN>;
    using KEncoding         = ck_tile::remove_cvref_t<decltype(Qk::MakeBBlockDistributionEncode())>;
    using VEncoding         = ck_tile::remove_cvref_t<decltype(Pv::MakeBBlockDistributionEncode())>;
    using KDistribution     = decltype(ck_tile::make_static_tile_distribution(KEncoding{}));
    using VDistribution     = decltype(ck_tile::make_static_tile_distribution(VEncoding{}));
    using VReadDistribution = decltype(Policy::template MakeVRegTileDistribution<Problem>());
    using KDescriptor       = decltype(Policy::template MakeKLdsBlockDescriptor<Problem>());
    using VDescriptor       = decltype(Policy::template MakeVLdsBlockDescriptor<Problem>());
    using KView             = decltype(ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        static_cast<Data*>(nullptr), KDescriptor{}));
    using VView             = decltype(ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        static_cast<Data*>(nullptr), VDescriptor{}));
    using KWindow           = decltype(ck_tile::make_tile_window(
        std::declval<KView>(),
        ck_tile::make_tuple(ck_tile::number<32>{}, ck_tile::number<Geometry::kHeadDimQK>{}),
        ck_tile::multi_index<2>{},
        KDistribution{}));
    using VWindow           = decltype(ck_tile::make_tile_window(
        std::declval<VView>(),
        ck_tile::make_tuple(ck_tile::number<32>{}, ck_tile::number<128>{}),
        ck_tile::multi_index<2>{},
        VReadDistribution{}));
    static constexpr auto k = MakeOperandConsumers<KWindow, KDistribution, Qk, true>();
    static constexpr auto v = MakeOperandConsumers<VWindow, VDistribution, Pv, false>();
    static_assert(k.valid && v.valid);
    static_assert(KWindow::NumAccessPerCoord == Geometry::kKSuLoadCount);
    static_assert(VWindow::NumAccessPerCoord == Geometry::kVStageLoadCount);

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    using ActualQk =
        ck_tile::remove_cvref_t<decltype(Policy::template GetQKBlockGemmSu<Problem>())>;
    using ActualPv = ck_tile::remove_cvref_t<decltype(Policy::template GetPVBlockGemm<Problem>())>;
    using ActualKDistribution = decltype(Policy::template MakeKSuRegTileDistribution<Problem>());
    static_assert(std::is_same_v<Qk, ActualQk> && std::is_same_v<Pv, ActualPv>);
    static_assert(std::is_same_v<KDistribution, ActualKDistribution>);
    using TransposedEncoding =
        typename ck_tile::OutputTileDistributionTraits<typename VReadDistribution::DstrEncode,
                                                       Data>::TransposedDstrEncode;
    static_assert(std::is_same_v<VEncoding, TransposedEncoding>);
#endif
};

template <typename Geometry>
constexpr bool HasExactConsumers = std::is_same_v<Geometry, ck_tile::LegacyD192Geometry> ||
                                   std::is_same_v<Geometry, ck_tile::FmhaTdmD128Geometry>;

// Every ordinal denotes an actual load-window access, not a physical VGPR number.
class Machine
{
    struct Tensor
    {
        Operand operand;
        int tile;
        int bank;
        bool completed = false;
    };

    Result& result_;
    std::vector<Tensor> tensors_;
    std::vector<std::size_t> ds_queue_;
    std::vector<std::size_t> tensor_queue_;
    std::array<int, 4> bank_owner_{{-1, -1, -1, -1}};
    int event_ = 0;

    public:
    explicit Machine(Result& result) : result_(result) {}

    void Error(const char* text) { result_.errors.emplace_back(text); }

    static int Bank(Operand operand, int tile)
    {
        return operand == Operand::Q ? 0 : (operand == Operand::V ? 2 : 0) + tile % 2;
    }

    void IssueTensor(Operand operand, int tile)
    {
        ++event_;
        const int bank = Bank(operand, tile);
        for(const auto index : ds_queue_)
            if(result_.reads[index].lds_bank == bank)
                Error("LDS overwrite before DS reader completion");
        for(const auto index : tensor_queue_)
            if(tensors_[index].bank == bank)
                Error("LDS overwrite before previous TDM writer completion");
        bank_owner_[bank] = static_cast<int>(tensors_.size());
        tensor_queue_.push_back(tensors_.size());
        tensors_.push_back({operand, tile, bank});
        ++result_.tensor_issue_count;
    }

    void WaitTensor(int keep)
    {
        ++event_;
        if(keep < 0)
        {
            Error("negative tensor wait");
            return;
        }
        while(tensor_queue_.size() > static_cast<std::size_t>(keep))
        {
            tensors_[tensor_queue_.front()].completed = true;
            tensor_queue_.erase(tensor_queue_.begin());
        }
    }

    void WaitDs(int keep)
    {
        ++event_;
        if(keep < 0)
        {
            Error("negative DS wait");
            return;
        }
        while(ds_queue_.size() > static_cast<std::size_t>(keep))
        {
            result_.reads[ds_queue_.front()].completed = true;
            ds_queue_.erase(ds_queue_.begin());
        }
    }

    void IssueRead(Destination destination, int access, Location producer)
    {
        ++event_;
        const int bank  = Bank(destination.operand, destination.tile);
        const int owner = bank_owner_[bank];
        if(owner < 0 || tensors_[owner].operand != destination.operand ||
           tensors_[owner].tile != destination.tile)
            Error("DS read has no matching TDM producer or reads an overwritten LDS generation");
        else if(!tensors_[owner].completed)
            Error("DS read before TDM producer completion");
        for(const auto& read : result_.reads)
            if(read.destination == destination && read.access == access)
                Error("duplicate logical destination access");
        ds_queue_.push_back(result_.reads.size());
        result_.reads.push_back({destination, access, bank, producer, event_});
    }

    void Consume(Destination destination,
                 int count,
                 Location consumer,
                 bool require_completion,
                 int selected_access = -1)
    {
        ++event_;
        std::vector<int> coverage(count);
        int pending_position = -1;
        for(auto& read : result_.reads)
        {
            if(!(read.destination == destination))
                continue;
            if(read.access < 0 || read.access >= count)
                Error("destination ordinal out of range");
            else
                ++coverage[read.access];
            if(selected_access >= 0 && read.access != selected_access)
                continue;
            if(read.release_event != -1)
                Error("consumer after logical destination release");
            if(read.first_consumer_event == -1)
            {
                read.first_consumer_event = event_;
                read.first_consumer       = consumer;
            }
            read.last_consumer_event = event_;
            read.last_consumer       = consumer;
        }
        for(const auto visits : coverage)
            if(visits != 1)
                Error("consumer missing unique producer coverage");
        for(std::size_t i = 0; i < ds_queue_.size(); ++i)
            if(result_.reads[ds_queue_[i]].destination == destination &&
               (selected_access < 0 || result_.reads[ds_queue_[i]].access == selected_access))
                pending_position = static_cast<int>(i);
        if(pending_position < 0)
            return;
        if(!require_completion)
        {
            Error("consumer before DS completion; compiler completion contract missing");
            return;
        }

        result_.isa_obligations.push_back(
            {ObligationKind::CompilerConsumerCompletion,
             consumer,
             destination,
             "Prove an emitted hardware wait completes the required destination access before use. "
             "The model requires completion here; it does not insert or observe a wait.",
             selected_access,
             static_cast<int>(ds_queue_.size()) - pending_position - 1});
        // Conditional transfer: continue only under the recorded completion requirement.
        // Retire the required FIFO prefix, preserving every unrelated younger operation.
        for(int i = 0; i <= pending_position; ++i)
            result_.reads[ds_queue_[i]].completed = true;
        ds_queue_.erase(ds_queue_.begin(), ds_queue_.begin() + pending_position + 1);
    }

    int PendingDsCount() const { return static_cast<int>(ds_queue_.size()); }

    void FinishConsumption(Destination destination)
    {
        ++event_;
        for(auto& read : result_.reads)
            if(read.destination == destination)
                read.last_consumer_event = event_;
    }

    void Release(Destination destination)
    {
        ++event_;
        for(auto& read : result_.reads)
        {
            if(!(read.destination == destination))
                continue;
            if(!read.completed)
                Error("logical destination reuse before pending DS write completion");
            read.release_event = event_;
        }
    }

    void ReuseBarrier(bool present, int keep)
    {
        if(!present)
            Error("missing LDS reuse barrier");
        else
            WaitDs(keep);
    }

    void Tail(int keep, int future_count, Destination future)
    {
        if(keep < 0 || keep > future_count)
            Error("stage tail exceeds future operand budget");
        WaitDs(keep);
        for(const auto index : ds_queue_)
            if(!(result_.reads[index].destination == future))
                Error("stage tail retains an operation other than the next operand");
    }

    void Exit(int iterations)
    {
        result_.pending_ds_at_exit      = static_cast<int>(ds_queue_.size());
        result_.pending_tensors_at_exit = static_cast<int>(tensor_queue_.size());
        if(!ds_queue_.empty())
            Error("missing terminal DS drain");
        if(!tensor_queue_.empty())
            Error("missing terminal TDM drain");
        for(const auto& read : result_.reads)
        {
            if(read.first_consumer_event != -1)
            {
                if(read.issue_event >= read.first_consumer_event ||
                   read.last_consumer_event < read.first_consumer_event ||
                   read.release_event <= read.last_consumer_event)
                    Error("producer/consumer/release lifetime ordering violated");
            }
            else if(read.destination == Destination{Operand::K, iterations, 0})
                ++result_.unused_next_k_reads;
            else
                Error("unexpected operand without a consumer");
        }
    }
};

template <typename Geometry, typename Schedule, bool IsQk, int Stage>
void RunStage(Machine& machine, Result& result, const Options<Geometry>& options, int iteration)
{
    using Kind             = ck_tile::FmhaTdmV128LoadKind;
    using Executor         = ck_tile::FmhaTdmV128ScheduleExecutor<Schedule>;
    constexpr int stages   = IsQk ? Geometry::kQkStages : Geometry::kPvStages;
    constexpr bool last    = Stage == stages - 1;
    constexpr bool loads_k = IsQk ? !last : last;
    constexpr int loads    = loads_k ? Geometry::kKSuLoadCount : Geometry::kVStageLoadCount;
    constexpr int wmmas    = IsQk ? Geometry::kQkWmmasPerStage : Geometry::kPvWmmasPerStage;
    const Destination current{IsQk ? Operand::K : Operand::V, iteration, Stage};
    const Destination next{loads_k ? Operand::K : Operand::V,
                           iteration + (!IsQk && last ? 1 : 0),
                           last ? 0 : Stage + 1};
    const Location consumer{iteration, IsQk, Stage, 0};
    if constexpr(!HasExactConsumers<Geometry>)
        machine.Consume(current,
                        IsQk ? Geometry::kKSuLoadCount : Geometry::kVStageLoadCount,
                        consumer,
                        options.require_compiler_consumer_completion);
    if constexpr(IsQk)
        machine.Consume({Operand::Q, 0, 0},
                        Geometry::kKSuLoadCount,
                        consumer,
                        options.require_compiler_consumer_completion);
    std::array<int, loads> accesses{};
    std::array<int, wmmas> coverage{};
    int current_wmma = -1;
    auto emit_wmma   = [&](auto stage, auto ordinal) {
        if constexpr(HasExactConsumers<Geometry>)
        {
            constexpr auto mapping = [] {
                if constexpr(IsQk)
                    return NativeBf16Operands<Geometry>::k;
                else
                    return NativeBf16Operands<Geometry>::v;
            }();
            constexpr int current_loads =
                IsQk ? Geometry::kKSuLoadCount : Geometry::kVStageLoadCount;
            for(int access = 0; access < current_loads; ++access)
                if(mapping.uses[decltype(ordinal)::value][access])
                    machine.Consume(current,
                                    current_loads,
                                      {iteration, IsQk, Stage, decltype(ordinal)::value},
                                    options.require_compiler_consumer_completion,
                                    access);
        }
        if(decltype(stage)::value != Stage || decltype(ordinal)::value < 0 ||
           decltype(ordinal)::value >= wmmas)
            machine.Error("WMMA ordinal or stage mismatch");
        else
            ++coverage[decltype(ordinal)::value];
        current_wmma = decltype(ordinal)::value;
        ++result.wmma_count;
    };
    auto emit_load = [&](auto kind, auto ordinal) {
        if constexpr(decltype(kind)::value != Kind::IgnoredLegacy)
        {
            if(decltype(kind)::value != (loads_k ? Kind::KRead : Kind::VRead))
                machine.Error("schedule load kind disagrees with logical destination");
            constexpr int access = decltype(ordinal)::value;
            if constexpr(access < 0 || access >= loads)
                machine.Error("scheduled access out of range");
            else
            {
                ++accesses[access];
                machine.IssueRead(next, access, {iteration, IsQk, Stage, current_wmma});
            }
        }
    };
    auto emit_point = [&](auto, auto, auto) {
        // Compiler fences neither retire DS/TDM work nor provide cross-wave visibility.
        ++result.compiler_fence_count;
    };
    if constexpr(IsQk)
        Executor::template ExecuteQkStage<Stage>(emit_wmma, emit_load, emit_point);
    else
        Executor::template ExecutePvStage<Stage>(emit_wmma, emit_load, emit_point);
    for(const auto count : coverage)
        if(count != 1)
            machine.Error("WMMA coverage is not exactly once");
    for(const auto count : accesses)
        if(count != 1)
            machine.Error("scheduled access coverage is not exactly once");
    if constexpr(!HasExactConsumers<Geometry>)
        machine.FinishConsumption(current);
    if constexpr(IsQk)
        machine.FinishConsumption({Operand::Q, 0, 0});
    machine.Release(current);
    if constexpr(IsQk)
        machine.Tail(options.qk_tails[Stage], loads, next);
    else
        machine.Tail(options.pv_tails[Stage], loads, next);
}

} // namespace detail

template <typename Geometry, typename Schedule>
Result ValidateActiveSchedule(const Options<Geometry>& options = {})
{
    static_assert(Geometry::kQkStages > 0 && Geometry::kPvStages > 0);
    Result result;
    result.exact_operand_consumers = detail::HasExactConsumers<Geometry>;
    detail::Machine machine(result);
    if constexpr(Schedule::kNumQkStages != Geometry::kQkStages ||
                 Schedule::kNumPvStages != Geometry::kPvStages ||
                 Schedule::kQkWmmasPerStage != Geometry::kQkWmmasPerStage ||
                 Schedule::kPvWmmasPerStage != Geometry::kPvWmmasPerStage)
    {
        machine.Error("Schedule/Geometry stage or WMMA count mismatch");
        return result;
    }
    else
    {
        if(options.iterations < 0 || options.k_tensor_wait < 0 || options.k_tensor_wait > 1 ||
           options.v_tensor_wait < 0 || options.v_tensor_wait > 1)
        {
            machine.Error("invalid iteration or tensor wait configuration");
            return result;
        }
        if(options.iterations == 0)
        {
            if(!options.empty_work_returns_before_prologue)
                machine.Error("zero iterations would enter the do/while body");
            return result;
        }

        const Destination q{Operand::Q, 0, 0};
        result.isa_obligations = {
            {ObligationKind::DestinationLayout,
             {},
             q,
             "For BF16 compile the gfx125 device-pass metadata assertions binding native host "
             "SFC/WMMA maps to actual policy types. Audit one DS instruction per event and Q "
             "read count. Other geometries are inventory-only, without per-access use proof."},
            {ObligationKind::PhysicalRegisterLease,
             {},
             q,
             "Prove allocated VGPR destinations survive asynchronous writes through completion, "
             "including copies, stage transitions, both parity paths, and unused next-K at exit."},
            {ObligationKind::PhysicalTensorIssueCount,
             {},
             q,
             "The queue counts logical TDM issues. Audit physical descriptor-box issue count "
             "and ordering before accepting tensorcnt=1 for each dtype/layout."},
            {ObligationKind::HardwareWaitAndBarrierLowering,
             {},
             q,
             "Verify wait_dscnt/tensorcnt and LDS visibility barriers at the modeled source "
             "boundaries, with actual reader counts and both loop exit paths."}};

        machine.IssueTensor(Operand::Q, 0);
        machine.WaitTensor(0);
        for(int access = 0; access < Geometry::kKSuLoadCount; ++access)
            machine.IssueRead(q, access, {});
        machine.ReuseBarrier(options.q_reuse_barrier, 0);
        machine.IssueTensor(Operand::K, 0);
        machine.IssueTensor(Operand::V, 0);
        machine.IssueTensor(Operand::K, 1);
        machine.WaitTensor(0);
        for(int access = 0; access < Geometry::kKSuLoadCount; ++access)
            machine.IssueRead({Operand::K, 0, 0}, access, {});

        for(int iteration = 0; iteration < options.iterations; ++iteration)
        {
            machine.ReuseBarrier(options.v_reuse_barrier, options.v_reuse_ds_limit);
            machine.IssueTensor(Operand::V, iteration + 1);
            ck_tile::static_for<0, Geometry::kQkStages, 1>{}([&](auto stage) {
                detail::RunStage<Geometry, Schedule, true, decltype(stage)::value>(
                    machine, result, options, iteration);
            });
            machine.WaitTensor(options.v_tensor_wait);
            machine.ReuseBarrier(options.k_reuse_barrier, options.k_reuse_ds_limit);
            machine.IssueTensor(Operand::K, iteration + 2);
            ck_tile::static_for<0, Geometry::kPvStages, 1>{}([&](auto stage) {
                detail::RunStage<Geometry, Schedule, false, decltype(stage)::value>(
                    machine, result, options, iteration);
            });
            machine.WaitTensor(options.k_tensor_wait);
            if(iteration + 1 < options.iterations)
                ++result.backedges;
        }
        if(options.epilogue_tensor_drain &&
           (options.k_tensor_wait != 0 || options.v_tensor_wait != 0))
            machine.WaitTensor(0);
        if(options.epilogue_ds_drain && options.pv_tails.back() != 0)
            machine.WaitDs(0);
        machine.Release(q);
        machine.Release({Operand::K, options.iterations, 0});
        machine.Exit(options.iterations);
        if(result.unused_next_k_reads != Geometry::kKSuLoadCount)
            machine.Error("terminal next-K producer coverage mismatch");
        return result;
    }
}

} // namespace tdm_v128_test::lifetime
