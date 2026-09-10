// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <iostream>

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_v128_schedule.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_schedule_executor.hpp"

namespace {

using namespace ck_tile;
using Legacy   = BlockFmhaPipelineQRKSVSTdmD192V128Schedule;
using Active   = FmhaTdmV128LegacySchedule<LegacyD192Geometry>;
using Executor = FmhaTdmV128ScheduleExecutor<Active>;
using Token    = FmhaD192ScheduleToken;
using Kind     = FmhaTdmV128LoadKind;

struct Event
{
    int kind;
    int value;

    bool operator==(const Event& other) const { return kind == other.kind && value == other.value; }
};

struct Trace
{
    std::array<Event, 512> events{};
    std::size_t size = 0;

    void Add(int kind, int value) { events.at(size++) = {kind, value}; }

    bool operator==(const Trace& other) const
    {
        if(size != other.size)
            return false;
        for(std::size_t i = 0; i < size; ++i)
            if(!(events[i] == other.events[i]))
                return false;
        return true;
    }
};

template <bool IsQk, index_t Stage>
bool ValidateStage()
{
    Trace actual, expected, callbacks, expected_callbacks;
    int seen[23]            = {};
    int accesses[24]        = {};
    constexpr bool loads_k  = IsQk ? Stage < 3 : Stage == 3;
    constexpr int num_loads = loads_k ? 24 : 16;
    auto emit_wmma          = [&](auto stage, auto wmma) {
        actual.Add(0, decltype(stage)::value * 100 + decltype(wmma)::value);
    };
    auto emit_point = [&](auto, auto, auto point) {
        actual.Add(1, static_cast<int>(decltype(point)::value));
    };
    auto softmax = [&](auto token, auto ordinal) {
        callbacks.Add(static_cast<int>(decltype(token)::value), decltype(ordinal)::value);
    };
    bool valid     = true;
    auto emit_load = [&](auto kind, auto access) {
        constexpr auto value  = decltype(kind)::value;
        constexpr int ordinal = decltype(access)::value;
        actual.Add(10 + static_cast<int>(value), ordinal);
        if constexpr(value != Kind::IgnoredLegacy)
        {
            static_assert(ordinal >= 0 && ordinal < num_loads);
            valid = valid && (value == (loads_k ? Kind::KRead : Kind::VRead));
            ++accesses[ordinal];
        }
        else if constexpr(IsQk)
        {
            decltype(kind)::EmitSoftmax(softmax);
        }
    };
    if constexpr(IsQk)
        Executor::ExecuteQkStage<Stage>(emit_wmma, emit_load, emit_point);
    else
        Executor::ExecutePvStage<Stage>(emit_wmma, emit_load, emit_point);

    // Derive expected window accesses from per-token counts within this stage,
    // independently of the adapter's subtraction of whole-table ordinals.
    constexpr int num_wmma = IsQk ? Legacy::kQkWmmasPerStage : Legacy::kPvWmmasPerStage;
    for(int wmma = 0; wmma < num_wmma; ++wmma)
    {
        const auto& rows = IsQk ? Legacy::kQkRows : Legacy::kPvRows;
        const int index  = Stage * num_wmma + wmma;
        const auto& row  = rows[index];
        expected.Add(0, Stage * 100 + wmma);
        expected.Add(1, static_cast<int>(FmhaTdmV128SchedulePoint::AfterWmma));
        for(int half = 0; half < 2; ++half)
        {
            const int first = half == 0 ? 0 : row.size / 2;
            const int last  = half == 0 ? row.size / 2 : row.size;
            for(int slot = first; slot < last; ++slot)
            {
                const auto token = row[slot];
                const int id     = static_cast<int>(token);
                int ordinal      = 0;
                for(int prev_row = 0; prev_row <= index; ++prev_row)
                    for(int prev_slot = 0;
                        prev_slot < (prev_row == index ? slot : rows[prev_row].size);
                        ++prev_slot)
                        ordinal += rows[prev_row][prev_slot] == token;
                Kind kind  = Kind::IgnoredLegacy;
                int access = ordinal;
                if(token >= Token::KM0 && token <= Token::KM3)
                {
                    kind   = Kind::KRead;
                    access = (id - static_cast<int>(Token::KM0)) * 6 + seen[id];
                }
                else if(token >= Token::VM0 && token <= Token::VM3)
                {
                    kind   = Kind::VRead;
                    access = (id - static_cast<int>(Token::VM0)) * 4 + seen[id];
                }
                else if constexpr(IsQk)
                {
                    if((token >= Token::P2M0 && token <= Token::P2M3) || token == Token::ORescale)
                        expected_callbacks.Add(id, ordinal);
                }
                ++seen[id];
                expected.Add(10 + static_cast<int>(kind), access);
            }
            expected.Add(1,
                         static_cast<int>(half == 0 ? FmhaTdmV128SchedulePoint::BetweenTokenHalves
                                                    : FmhaTdmV128SchedulePoint::AfterTokens));
        }
    }
    for(int i = 0; i < num_loads; ++i)
        valid = valid && accesses[i] == 1;
    valid = valid && actual == expected && callbacks == expected_callbacks;
    ++expected.events[0].value;
    return valid && !(actual == expected);
}

struct UnequalStages
{
    static constexpr index_t kNumQkStages     = 2;
    static constexpr index_t kNumPvStages     = 1;
    static constexpr index_t kQkWmmasPerStage = 2;
    static constexpr index_t kPvWmmasPerStage = 3;

    template <index_t, index_t, bool, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitQkRowHalf(Visitor&)
    {
    }
    template <index_t, index_t, bool, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvRowHalf(Visitor&)
    {
    }
};

bool ValidateUnequalStageCounts()
{
    using TestExecutor = FmhaTdmV128ScheduleExecutor<UnequalStages>;
    int wmma           = 0;
    int points         = 0;
    auto emit_wmma     = [&](auto, auto) { ++wmma; };
    auto emit_token    = [](auto, auto) {};
    auto emit_point    = [&](auto, auto, auto) { ++points; };
    TestExecutor::ExecuteQkStage<0>(emit_wmma, emit_token, emit_point);
    TestExecutor::ExecuteQkStage<1>(emit_wmma, emit_token, emit_point);
    TestExecutor::ExecutePvStage<0>(emit_wmma, emit_token, emit_point);
    return wmma == 7 && points == 21;
}

template <bool IsQk, index_t Stage>
bool ValidateD128Stage()
{
    using Schedule = FmhaTdmV128ScheduleFor<FmhaTdmD128Geometry>;
    using Exec     = FmhaTdmV128ScheduleExecutor<Schedule>;
    Trace actual, expected;
    auto wmma = [&](auto stage, auto ordinal) { actual.Add(0, stage * 100 + ordinal); };
    auto load = [&](auto kind, auto ordinal) {
        actual.Add(10 + static_cast<int>(decltype(kind)::value), ordinal);
    };
    auto point = [&](auto, auto, auto p) { actual.Add(1, static_cast<int>(decltype(p)::value)); };
    if constexpr(IsQk)
        Exec::template ExecuteQkStage<Stage>(wmma, load, point);
    else
        Exec::template ExecutePvStage<Stage>(wmma, load, point);

    // D128's independent inventory is one load after every native WMMA.
    constexpr Kind next = (IsQk ? Stage < 3 : Stage == 3) ? Kind::KRead : Kind::VRead;
    for(int i = 0; i < 16; ++i)
    {
        expected.Add(0, Stage * 100 + i);
        expected.Add(1, 0);
        expected.Add(1, 1);
        expected.Add(10 + static_cast<int>(next), i);
        expected.Add(1, 2);
    }
    const bool valid = actual == expected;
    ++expected.events[3].value;
    return valid && !(actual == expected);
}

template <int Wmmas, int Loads>
constexpr bool ValidateRowCoverage()
{
    constexpr auto rows = FmhaTdmV128ActiveRows<Wmmas, Loads>{};
    int ordinal         = 0;
    for(const auto& row : rows.rows)
    {
        if(row.size < 0 || row.size > rows.kMaxLoadsPerRow)
            return false;
        for(int i = 0; i < row.size; ++i)
            if(row.accesses[i] != ordinal++)
                return false;
    }
    return ordinal == Loads;
}

static_assert(ValidateRowCoverage<16, 16>() && ValidateRowCoverage<16, 24>() &&
              ValidateRowCoverage<12, 32>() && ValidateRowCoverage<24, 16>());
static_assert(std::is_same_v<FmhaTdmV128ScheduleFor<LegacyD192Geometry>, Active>);

} // namespace

int main()
{
    bool valid = ValidateUnequalStageCounts();
    ck_tile::static_for<0, 4, 1>{}([&](auto stage) {
        valid = ValidateStage<true, stage>() && valid;
        valid = ValidateStage<false, stage>() && valid;
        valid = ValidateD128Stage<true, stage>() && valid;
        valid = ValidateD128Stage<false, stage>() && valid;
    });
    std::cout << "V128 schedule: 160 WMMA rows, 160 loads, ignored slots and fences "
              << (valid ? "PASS" : "FAIL") << '\n';
    std::cout << "D128 schedule: 64 QK + 64 PV WMMA, 128 loads, future-operand ordinals "
              << (valid ? "PASS" : "FAIL") << '\n';
    return valid ? 0 : 1;
}
