// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "tdm_fmha_v128_lifetime.hpp"

#include <iostream>
#include <stdexcept>

namespace {

namespace life = tdm_v128_test::lifetime;
using ck_tile::FmhaTdmV128ActiveSchedule;
using ck_tile::FmhaTdmV128ScheduleFor;
using Kind = ck_tile::FmhaTdmV128LoadKind;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
static_assert(life::detail::NativeBf16Operands<ck_tile::LegacyD192Geometry>::k.valid);
static_assert(life::detail::NativeBf16Operands<ck_tile::FmhaTdmD128Geometry>::v.valid);
#endif

void Check(bool condition, const char* message)
{
    if(!condition)
        throw std::runtime_error(message);
}

template <typename Geometry>
life::Options<Geometry> AllTails(int iterations)
{
    life::Options<Geometry> options;
    options.iterations    = iterations;
    options.k_tensor_wait = 1;
    options.v_tensor_wait = 1;
    options.qk_tails.fill(Geometry::kKSuLoadCount);
    options.qk_tails.back() = Geometry::kVStageLoadCount;
    options.pv_tails.fill(Geometry::kVStageLoadCount);
    options.pv_tails.back() = Geometry::kKSuLoadCount;
    return options;
}

// Protocol fixture only: this is not a supported FP8 Geometry, layout, or kernel.
// It matches the planned D192 FP8 K64 inventories to catch a hidden four-PV-stage assumption.
struct UnequalStageFixture
{
    static constexpr int kQkStages        = 4;
    static constexpr int kPvStages        = 2;
    static constexpr int kQkWmmasPerStage = 12;
    static constexpr int kPvWmmasPerStage = 16;
    static constexpr int kKSuLoadCount    = 12;
    static constexpr int kVStageLoadCount = 32;
};

template <typename Geometry, typename Schedule>
void PositiveMatrix()
{
    for(const int iterations : {0, 1, 2, 3, 4, 5, 6})
    {
        for(int mode = 0; mode < 3; ++mode)
        {
            auto options = AllTails<Geometry>(iterations);
            if(mode == 0)
            {
                options.qk_tails.fill(0);
                options.pv_tails.fill(0);
                options.k_tensor_wait = options.v_tensor_wait = 0;
            }
            else if(mode == 1)
            {
                options.qk_tails.fill(0);
                options.pv_tails.fill(0);
            }
            const auto result = life::ValidateActiveSchedule<Geometry, Schedule>(options);
            if(!result.LogicalContractHolds())
                throw std::runtime_error(result.errors.front());
            const int expected_wmmas =
                iterations * (Geometry::kQkStages * Geometry::kQkWmmasPerStage +
                              Geometry::kPvStages * Geometry::kPvWmmasPerStage);
            Check(result.wmma_count == expected_wmmas, "exact WMMA inventory");
            Check(result.compiler_fence_count == expected_wmmas * 3,
                  "compiler fence callbacks remain separate from hardware waits");
            Check(result.backedges == std::max(0, iterations - 1), "odd/even backedge count");
            Check(result.exact_operand_consumers == life::detail::HasExactConsumers<Geometry>,
                  "unsupported geometry must remain explicitly inventory-only");
            Check(result.pending_ds_at_exit == 0 && result.pending_tensors_at_exit == 0,
                  "terminal queues must be empty");
            Check(result.unused_next_k_reads == (iterations == 0 ? 0 : Geometry::kKSuLoadCount),
                  "unused last-PV next-K reads must be retained and drained");
            Check(result.tensor_issue_count == (iterations == 0 ? 0 : 4 + 2 * iterations),
                  "logical TDM inventory including prologue Q");
            if(iterations == 0)
            {
                Check(result.reads.empty() && result.isa_obligations.empty(),
                      "empty path exits before asynchronous work");
                continue;
            }
            Check(result.CountObligations(life::ObligationKind::PhysicalRegisterLease) == 1,
                  "logical success must retain physical destination-lifetime obligation");
            Check(result.CountObligations(life::ObligationKind::PhysicalTensorIssueCount) == 1,
                  "logical tensor issues must not be claimed as physical issue counts");
            if(mode == 2)
                Check(result.CountObligations(life::ObligationKind::CompilerConsumerCompletion) > 1,
                      "partial tails require explicit compiler completion proof obligations");
            for(const auto& read : result.reads)
            {
                Check(read.completed && read.release_event > read.issue_event,
                      "every logical register destination is retired before release");
                if(read.first_consumer_event == -1)
                    Check(read.destination == life::Destination{life::Operand::K, iterations, 0},
                          "only terminal lookahead may be unused");
                else if constexpr(life::detail::HasExactConsumers<Geometry>)
                {
                    if(read.destination.operand == life::Operand::Q)
                        continue;
                    const auto mapping = [&] {
                        using Metadata = life::detail::NativeBf16Operands<Geometry>;
                        if(read.destination.operand == life::Operand::K)
                            return std::pair<int, int>{Metadata::k.first[read.access],
                                                       Metadata::k.last[read.access]};
                        return std::pair<int, int>{Metadata::v.first[read.access],
                                                   Metadata::v.last[read.access]};
                    }();
                    Check(read.first_consumer.wmma == mapping.first &&
                              read.last_consumer.wmma == mapping.second,
                          "actual first/last WMMA must match SFC destination and B-slice "
                          "intersection");
                }
            }
        }
    }
}

template <typename Base, int Fault>
struct BrokenSchedule : Base
{
    template <ck_tile::index_t Stage, ck_tile::index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitQkRowHalf(Visitor&& visitor)
    {
        auto mutate = [&](auto kind, auto ordinal) {
            if constexpr(Stage == 0 && decltype(kind)::value == Kind::KRead &&
                         decltype(ordinal)::value == 0)
            {
                if constexpr(Fault == 0)
                    return;
                else if constexpr(Fault == 1)
                    visitor(kind, ck_tile::number<1>{});
                else
                    visitor(std::integral_constant<Kind, Kind::VRead>{}, ordinal);
            }
            else
                visitor(kind, ordinal);
        };
        Base::template VisitQkRowHalf<Stage, Wmma, SecondHalf>(mutate);
    }
};

void NegativeControls()
{
    using Geometry   = ck_tile::FmhaTdmD128Geometry;
    using Schedule   = FmhaTdmV128ScheduleFor<Geometry>;
    const auto valid = AllTails<Geometry>(3);
    Check(life::ValidateActiveSchedule<Geometry, BrokenSchedule<Schedule, 0>>(valid).HasError(
              "coverage"),
          "missing producer must fail");
    Check(life::ValidateActiveSchedule<Geometry, BrokenSchedule<Schedule, 1>>(valid).HasError(
              "duplicate logical destination"),
          "duplicate destination must fail");
    Check(life::ValidateActiveSchedule<Geometry, BrokenSchedule<Schedule, 2>>(valid).HasError(
              "load kind"),
          "wrong destination kind must fail");

    auto options                                 = valid;
    options.require_compiler_consumer_completion = false;
    Check(life::ValidateActiveSchedule<Geometry, Schedule>(options).HasError("consumer before DS"),
          "scheduler fences must not substitute for consumer completion");
    options                   = valid;
    options.epilogue_ds_drain = false;
    Check(life::ValidateActiveSchedule<Geometry, Schedule>(options).HasError("terminal DS drain"),
          "unused terminal next-K requires drain");
    options                       = valid;
    options.epilogue_tensor_drain = false;
    Check(life::ValidateActiveSchedule<Geometry, Schedule>(options).HasError("terminal TDM drain"),
          "partial tensor waits require terminal drain");
    options                 = valid;
    options.q_reuse_barrier = false;
    Check(life::ValidateActiveSchedule<Geometry, Schedule>(options).HasError("LDS overwrite"),
          "Q DS readers must finish before K0 overwrite");
    options                 = valid;
    options.k_reuse_barrier = false;
    Check(life::ValidateActiveSchedule<Geometry, Schedule>(options).HasError("LDS reuse barrier"),
          "cross-wave K reuse barrier must exist even when local reads are complete");
    options                 = valid;
    options.v_reuse_barrier = false;
    Check(life::ValidateActiveSchedule<Geometry, Schedule>(options).HasError("LDS reuse barrier"),
          "cross-wave V reuse barrier must exist even when local reads are complete");
    options = valid;
    ++options.qk_tails[0];
    Check(life::ValidateActiveSchedule<Geometry, Schedule>(options).HasError("tail exceeds"),
          "tail larger than future operand inventory must fail");
    options                                    = valid;
    options.iterations                         = 0;
    options.empty_work_returns_before_prologue = false;
    Check(life::ValidateActiveSchedule<Geometry, Schedule>(options).HasError("do/while"),
          "zero work must not enter first loop body");
    Check(life::ValidateActiveSchedule<Geometry, FmhaTdmV128ActiveSchedule<UnequalStageFixture>>()
              .HasError("stage or WMMA count mismatch"),
          "Schedule and Geometry must agree independently for QK and PV");

    auto unequal              = AllTails<UnequalStageFixture>(1);
    unequal.epilogue_ds_drain = false;
    Check(life::ValidateActiveSchedule<UnequalStageFixture,
                                       FmhaTdmV128ActiveSchedule<UnequalStageFixture>>(unequal)
              .HasError("terminal DS drain"),
          "two-stage PV must drain its selected last stage, not fixed stage three");
}

void QueueOrderingControls()
{
    life::Result result;
    life::detail::Machine machine(result);
    const life::Destination k{life::Operand::K, 0, 0};
    const life::Destination v{life::Operand::V, 0, 0};
    machine.IssueTensor(life::Operand::K, 0);
    machine.IssueRead(k, 0, {});
    Check(result.HasError("before TDM producer completion"), "TDM producer must complete first");

    life::Result reuse;
    life::detail::Machine second(reuse);
    second.IssueTensor(life::Operand::K, 0);
    second.IssueTensor(life::Operand::V, 0);
    second.WaitTensor(0);
    second.IssueRead(k, 0, {});
    second.IssueRead(v, 0, {});
    second.WaitDs(1);
    second.IssueTensor(life::Operand::K, 2);
    Check(reuse.LogicalContractHolds(), "younger unrelated V read may survive K LDS reuse");
    second.Release(v);
    Check(reuse.HasError("destination reuse before pending DS write"),
          "unused pending destination must not be released");
    second.IssueTensor(life::Operand::V, 2);
    Check(reuse.HasError("LDS overwrite before DS reader"),
          "future-count arithmetic alone does not permit same-bank reuse");

    life::Result order;
    life::detail::Machine third(order);
    third.IssueTensor(life::Operand::K, 0);
    third.WaitTensor(0);
    third.IssueRead(k, 0, {});
    third.Tail(1, 1, v);
    Check(order.HasError("other than the next operand"),
          "valid numerical tail count with wrong destination must fail");
}

void CurrentLegacyTuning()
{
    using Geometry = ck_tile::LegacyD192Geometry;
    using Schedule = FmhaTdmV128ScheduleFor<Geometry>;
    for(const int iterations : {0, 1, 2, 3, 4, 5, 6})
    {
        const auto options =
            life::OptionsFromTuning<Geometry, ck_tile::LegacyD192Tuning>(iterations);
        const auto result = life::ValidateActiveSchedule<Geometry, Schedule>(options);
        Check(result.LogicalContractHolds(), "actual legacy tuning must satisfy logical contract");
    }
}

void D128PhysicalConsumers()
{
    using Metadata = life::detail::NativeBf16Operands<ck_tile::FmhaTdmD128Geometry>;
    // Independently intersected DS destinations with WMMA operands in the captured
    // native-Max3 profile6/fmha-7 ISA, rather than deriving expectations from the SFC.
    constexpr std::array<int, 16> k_first = {0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14};
    constexpr std::array<int, 16> v_first = {0, 0, 4, 4, 1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 7, 7};
    for(int access = 0; access < 16; ++access)
    {
        Check(Metadata::k.first[access] == k_first[access] &&
                  Metadata::k.last[access] == k_first[access] + 1,
              "D128 K first/last consumers must match independent physical ISA evidence");
        Check(Metadata::v.first[access] == v_first[access] &&
                  Metadata::v.last[access] == v_first[access] + 8,
              "D128 V first/last consumers must match independent physical ISA evidence");
        for(int wmma = 0; wmma < 16; ++wmma)
        {
            Check(Metadata::k.uses[wmma][access] ==
                      (wmma == k_first[access] || wmma == k_first[access] + 1),
                  "D128 K consumer set must match independent physical ISA evidence");
            Check(Metadata::v.uses[wmma][access] ==
                      (wmma == v_first[access] || wmma == v_first[access] + 8),
                  "D128 V consumer set must match independent physical ISA evidence");
        }
    }
}

template <typename Geometry>
void FineGrainedConsumers()
{
    using Metadata = life::detail::NativeBf16Operands<Geometry>;
    life::Result result;
    life::detail::Machine machine(result);
    const life::Destination destination{life::Operand::K, 0, 0};
    machine.IssueTensor(life::Operand::K, 0);
    machine.WaitTensor(0);
    for(int access = 0; access < Geometry::kKSuLoadCount; ++access)
        machine.IssueRead(destination, access, {});
    for(int access = 0; access < Geometry::kKSuLoadCount; ++access)
        if(Metadata::k.uses[0][access])
            machine.Consume(destination, Geometry::kKSuLoadCount, {0, true, 0, 0}, true, access);
    Check(result.LogicalContractHolds(), "first-WMMA access completion contract");
    Check(machine.PendingDsCount() > 0 && machine.PendingDsCount() < Geometry::kKSuLoadCount,
          "first WMMA must not require an invented full-operand drain");
    for(const auto& obligation : result.isa_obligations)
        Check(
            obligation.access >= 0 && Metadata::k.uses[0][obligation.access] &&
                obligation.maximum_dscnt > 0,
            "consumer obligation must name only a used access and its permissible pending suffix");
    const auto print = [](const char* operand, const auto& mapping) {
        std::cout << "D" << Geometry::kHeadDimQK << ' ' << operand << " first=";
        for(const auto value : mapping.first)
            std::cout << value << ',';
        std::cout << " last=";
        for(const auto value : mapping.last)
            std::cout << value << ',';
        std::cout << '\n';
    };
    print("K", Metadata::k);
    print("V", Metadata::v);
}

} // namespace

int main()
{
    try
    {
        PositiveMatrix<ck_tile::LegacyD192Geometry,
                       FmhaTdmV128ScheduleFor<ck_tile::LegacyD192Geometry>>();
        PositiveMatrix<ck_tile::FmhaTdmD128Geometry,
                       FmhaTdmV128ScheduleFor<ck_tile::FmhaTdmD128Geometry>>();
        PositiveMatrix<UnequalStageFixture, FmhaTdmV128ActiveSchedule<UnequalStageFixture>>();
        NegativeControls();
        QueueOrderingControls();
        CurrentLegacyTuning();
        D128PhysicalConsumers();
        FineGrainedConsumers<ck_tile::LegacyD192Geometry>();
        FineGrainedConsumers<ck_tile::FmhaTdmD128Geometry>();
        std::cout
            << "V128 conditional logical lifetime model PASS: 70 configurations, "
               "zero/one/two/odd/even trips, unequal stages, negative controls\n"
               "NOT ISA acceptance: compiler per-access completion, device-pass metadata equality, "
               "physical register leases, physical TDM counts, and wait/barrier lowering "
               "remain proof obligations. Unequal-stage case is a protocol fixture only.\n";
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "V128 lifetime model FAIL: " << error.what() << '\n';
        return 1;
    }
}
