// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Dynamic Q of the split-kv pingpong pipeline: each bf16 Q row is quantized to fp8 in the kernel
// (QuantizeQRows), scale = amax * (1 / 448), inv = v_rcp_f32(max(scale, 1e-20)), q = x * inv to
// fp8. A kernel runs QuantizeQRows on Q tiles and writes back, for every element of every thread,
// q with the scale and the reciprocal that thread used. The host checks the scales bit for bit,
// the reciprocals within the 1 ulp of v_rcp_f32 and alike on all copies of a row, and q bit for bit
// against x * (that reciprocal), all through the kernel's flush of f32 denormals.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha.hpp"

namespace {

using ck_tile::bf16_t;
using ck_tile::fp8_t;
using ck_tile::index_t;

// tile A of the fp8bf16 dynamic-Q split-kv instances (codegen/ops/fmha_fwd_splitkv.py)
using Shape  = ck_tile::TileFmhaShape<ck_tile::sequence<16, 64, 128, 128, 64, 128>,
                                      ck_tile::sequence<1, 4, 1>,
                                      ck_tile::sequence<16, 16, 128>,
                                      ck_tile::sequence<1, 4, 1>,
                                      ck_tile::sequence<16, 16, 64>,
                                      true>;
using Traits = ck_tile::TileFmhaFwdSplitKVTraits<true,
                                                 true,
                                                 false,
                                                 false,
                                                 false,
                                                 ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                 /*kHasBiasGrad=*/false,
                                                 true,
                                                 false,
                                                 false,
                                                 /*kHasUnevenSplits=*/true,
                                                 /*kMergeNumHeadGroupsSeqLenQ=*/false,
                                                 -1,
                                                 false,
                                                 ck_tile::BlockAttentionQuantScaleEnum::DYNAMIC_Q>;
using Problem =
    ck_tile::BlockFmhaFwdSplitKVPipelineProblem<fp8_t,
                                                fp8_t,
                                                fp8_t,
                                                float,
                                                float,
                                                float,
                                                float,
                                                fp8_t,
                                                float,
                                                float,
                                                Shape,
                                                false,
                                                ck_tile::ComposedAttention<0, true>,
                                                ck_tile::SimplifiedGenericAttentionMask<false>,
                                                Traits>;
// QuantizeQRows lives in the pingpong pipeline the split one derives from
using PingPong = typename ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVSTdmPingPong<Problem>::Base;
static_assert(PingPong::kDynamicQ);

constexpr index_t kRows     = PingPong::kM0;
constexpr index_t kHdim     = PingPong::kQKHeaddim;
constexpr float kFp8Max     = 448.f;
constexpr float kScaleFloor = 1e-20f;
constexpr index_t kPack     = 8; // values per pk8 conversion

// gfx1250 values, checked in the kernel: the host pass sees 64-lane warps and would double the
// block size (a launch over the kernel's launch bounds fails)
constexpr index_t kBlockSize      = 128;
constexpr index_t kSlotsPerThread = 64;
constexpr index_t kSlotsPerBlock  = kBlockSize * kSlotsPerThread;

// One Q element as a thread quantized it, with the row scale and reciprocal it used. Every element
// of every thread has its own slot: the warps and lane halves holding copies of a row all show.
struct QuantSlot
{
    index_t row;
    index_t col;
    float scale;
    float inv;
    fp8_t q;
};

// q_in: tiles of kRows x kHdim, row-major; one tile per block
__global__ void __launch_bounds__(kBlockSize) quantize_q_rows(const bf16_t* q_in, QuantSlot* slots)
{
#if defined(__gfx125__)
    using namespace ck_tile;
    const index_t row0 = blockIdx.x * kRows;

    constexpr auto q_dstr =
        PingPong::Policy::template MakeQRegTileDistribution<typename PingPong::Problem>();
    auto q_tile = make_static_distributed_tensor<bf16_t>(q_dstr);
    static_assert(PingPong::kBlockSize == kBlockSize);
    static_assert(decltype(q_tile)::get_thread_buffer_size() == kSlotsPerThread);
    constexpr auto spans = decltype(q_tile)::get_distributed_spans();
    const auto row_col   = [&](auto idx0, auto idx1) {
        const auto x = get_x_indices_from_distributed_indices(q_dstr, make_tuple(idx0, idx1));
        return make_tuple(row0 + x.at(number<0>{}), x.at(number<1>{}));
    };
    sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
            const auto rc                  = row_col(idx0, idx1);
            q_tile(make_tuple(idx0, idx1)) = q_in[rc.at(number<0>{}) * kHdim + rc.at(number<1>{})];
        });
    });

    const auto quantized = PingPong::QuantizeQRows(q_tile);
    const auto& q        = quantized.get(number<0>{});
    const auto& q_scale  = quantized.get(number<1>{});
    const auto& q_inv    = quantized.get(number<2>{});

    QuantSlot* slot = slots + (blockIdx.x * kBlockSize + threadIdx.x) * kSlotsPerThread;
    sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
            const auto rc = row_col(idx0, idx1);
            *slot++       = QuantSlot{rc.at(number<0>{}),
                                rc.at(number<1>{}),
                                q_scale[make_tuple(idx0)],
                                q_inv[make_tuple(idx0)],
                                q[make_tuple(idx0, idx1)]};
        });
    });
#else
    ck_tile::ignore = q_in;
    ck_tile::ignore = slots;
#endif
}

// x: groups of kPack floats, q: each group through the pk8 conversion of QuantizeQRows
__global__ void convert_pk8(const float* x, fp8_t* q, index_t groups)
{
#if defined(__gfx125__)
    using namespace ck_tile;
    const index_t group = blockIdx.x * blockDim.x + threadIdx.x;
    if(group < groups)
    {
        fp32x8_t values{};
        for(index_t i = 0; i < kPack; ++i)
            values[i] = x[group * kPack + i];
        // a scale operand of 1.0 is exponent 0, no scaling
        const auto packed =
            bit_cast<array<fp8_t, kPack>>(__builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(values, 1.f));
        for(index_t i = 0; i < kPack; ++i)
            q[group * kPack + i] = packed[i];
    }
#else
    ck_tile::ignore = x;
    ck_tile::ignore = q;
    ck_tile::ignore = groups;
#endif
}

bool has_gfx125()
{
    int devices = 0;
    return hipGetDeviceCount(&devices) == hipSuccess && 0 < devices &&
           ck_tile::is_gfx125_supported();
}

float ftz(float x) { return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.f, x) : x; }

uint32_t bits(float x) { return ck_tile::bit_cast<uint32_t>(x); }

uint8_t bits(fp8_t x) { return ck_tile::bit_cast<uint8_t>(x); }

bf16_t bf16_bits(uint16_t x) { return ck_tile::bit_cast<bf16_t>(x); }

bf16_t to_bf16(float x) { return ck_tile::type_convert<bf16_t>(x); }

// Q rows of kHdim values, padded with zero rows to whole tiles
struct QRows
{
    std::vector<bf16_t> values;

    index_t rows() const { return static_cast<index_t>(values.size()) / kHdim; }

    // values in (-amax, amax) and +-amax at col
    void add_amax(float amax, index_t col, std::mt19937& rng)
    {
        std::uniform_real_distribution<float> value(-amax, amax);
        for(index_t i = 0; i < kHdim; ++i)
        {
            bf16_t x = to_bf16(value(rng));
            if(!(std::abs(ck_tile::type_convert<float>(x)) < amax))
                x = to_bf16(0.f);
            values.push_back(i == col ? to_bf16(i % 2 == 0 ? amax : -amax) : x);
        }
    }
    void add(const std::vector<bf16_t>& row)
    {
        values.insert(values.end(), row.begin(), row.end());
    }
    void pad()
    {
        while(rows() % kRows != 0)
            add(std::vector<bf16_t>(kHdim, to_bf16(0.f)));
    }
};

// rows padded to whole tiles; one slot per element of each thread of each block
std::vector<QuantSlot> run_device(const QRows& rows)
{
    const index_t blocks = rows.rows() / kRows;
    std::vector<QuantSlot> slots(blocks * kSlotsPerBlock);
    ck_tile::DeviceMem q_in_buf(rows.values.size() * sizeof(bf16_t));
    ck_tile::DeviceMem slot_buf(slots.size() * sizeof(QuantSlot));
    q_in_buf.ToDevice(rows.values.data());
    slot_buf.SetBytePattern(0xff); // row -1 where a slot is left unwritten
    quantize_q_rows<<<blocks, kBlockSize>>>(static_cast<const bf16_t*>(q_in_buf.GetDeviceBuffer()),
                                            static_cast<QuantSlot*>(slot_buf.GetDeviceBuffer()));
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    slot_buf.FromDevice(slots.data());
    return slots;
}

// V_RCP_F32 is accurate to 1 ulp: |inv - exact| at most the float spacing in exact's binade
bool within_rcp_ulp(float inv, double exact)
{
    constexpr int kMantissaBits = std::numeric_limits<float>::digits - 1;
    const double ulp            = std::ldexp(1.0, std::ilogb(exact) - kMantissaBits);
    return std::isfinite(inv) && std::abs(static_cast<double>(inv) - exact) <= ulp;
}

std::string slot_name(std::size_t slot)
{
    return "block " + std::to_string(slot / kSlotsPerBlock) + " thread " +
           std::to_string(slot % kSlotsPerBlock / kSlotsPerThread) + " element " +
           std::to_string(slot % kSlotsPerThread);
}

// Checks every slot against the host and that the slots cover every value of rows; returns the
// number of values with x * inv > 448
index_t check_rows(QRows rows)
{
    rows.pad();
    const std::vector<QuantSlot> slots = run_device(rows);
    const index_t num_rows             = rows.rows();

    std::vector<float> scales(num_rows);
    for(index_t r = 0; r < num_rows; ++r)
    {
        float amax = 0.f;
        for(index_t i = 0; i < kHdim; ++i)
            amax =
                std::max(amax, std::abs(ck_tile::type_convert<float>(rows.values[r * kHdim + i])));
        scales[r] = ftz(amax * (1.f / kFp8Max));
    }

    std::vector<const QuantSlot*> first_copy(num_rows, nullptr);
    std::vector<bool> covered(rows.values.size(), false);
    index_t overshoot = 0;
    for(std::size_t s = 0; s < slots.size(); ++s)
    {
        const QuantSlot& slot = slots[s];
        if(!(0 <= slot.row && slot.row < num_rows && 0 <= slot.col && slot.col < kHdim))
        {
            ADD_FAILURE() << slot_name(s) << ": row " << slot.row << " col " << slot.col;
            continue;
        }
        const index_t r = slot.row;
        EXPECT_EQ(bits(slot.scale), bits(scales[r]))
            << slot_name(s) << " row " << r << ": scale " << slot.scale << ", host " << scales[r];

        const double exact = 1.0 / static_cast<double>(std::max(scales[r], kScaleFloor));
        EXPECT_TRUE(within_rcp_ulp(slot.inv, exact))
            << slot_name(s) << " row " << r << ": inv " << slot.inv << ", exact " << exact;

        if(first_copy[r] == nullptr)
            first_copy[r] = &slot;
        EXPECT_TRUE(bits(slot.scale) == bits(first_copy[r]->scale) &&
                    bits(slot.inv) == bits(first_copy[r]->inv))
            << slot_name(s) << " row " << r << ": scale " << slot.scale << " inv " << slot.inv
            << ", another copy of the row " << first_copy[r]->scale << " " << first_copy[r]->inv;

        const std::size_t element = r * kHdim + slot.col;
        const float x             = ck_tile::type_convert<float>(rows.values[element]);
        const float scaled        = ftz(ftz(x) * slot.inv);
        if(!covered[element])
            overshoot += kFp8Max < std::abs(scaled);
        covered[element] = true;
        // round to nearest even, and 448 past the fp8 maximum (see HostFp8Oracle)
        const fp8_t expected = ck_tile::type_convert<fp8_t>(scaled);
        EXPECT_EQ(bits(slot.q), bits(expected))
            << slot_name(s) << " row " << r << " col " << slot.col << ": x " << x << " * inv "
            << slot.inv << " = " << scaled << " -> " << ck_tile::type_convert<float>(slot.q)
            << ", host " << ck_tile::type_convert<float>(expected);
    }
    for(std::size_t e = 0; e < covered.size(); ++e)
        EXPECT_TRUE(covered[e]) << "row " << e / kHdim << " col " << e % kHdim << " in no slot";
    return overshoot;
}

#define SKIP_WITHOUT_GFX125() \
    if(!has_gfx125())         \
    GTEST_SKIP() << "the dynamic Q pipeline is gfx1250 only"

} // namespace

// The q oracle: the host fp8 conversion rounds to nearest even and saturates at 448.
TEST(TestCkTileFmhaFwdDynQQuant, HostFp8Oracle)
{
    const auto to_fp8 = [](float x) {
        return ck_tile::type_convert<float>(ck_tile::type_convert<fp8_t>(x));
    };
    EXPECT_EQ(to_fp8(1.0625f), 1.f);                              // tie, to the even 1
    EXPECT_EQ(to_fp8(1.1875f), 1.25f);                            // tie, to the even 1.25
    EXPECT_EQ(to_fp8(432.f), 448.f);                              // tie between 416 and 448
    EXPECT_EQ(to_fp8(std::ldexp(3.f, -10)), std::ldexp(1.f, -8)); // subnormal tie
    EXPECT_EQ(to_fp8(std::ldexp(1.f, -10)), 0.f);                 // half the smallest subnormal
    EXPECT_EQ(to_fp8(std::nextafter(448.f, 1000.f)), 448.f);
    EXPECT_EQ(to_fp8(470.f), 448.f);
}

// The reciprocal bound at exact reciprocals: the floats next to them pass, two spacings off fails.
TEST(TestCkTileFmhaFwdDynQQuant, RcpUlpBound)
{
    const float inf = std::numeric_limits<float>::infinity();
    for(const float exact : {1.f, 64.f})
    {
        const float up   = std::nextafter(exact, inf);
        const float down = std::nextafter(exact, 0.f);
        EXPECT_TRUE(within_rcp_ulp(exact, exact));
        EXPECT_TRUE(within_rcp_ulp(up, exact));
        EXPECT_TRUE(within_rcp_ulp(down, exact));
        // the floats below a power of 2 are half as far apart
        EXPECT_TRUE(within_rcp_ulp(std::nextafter(down, 0.f), exact));
        EXPECT_FALSE(within_rcp_ulp(std::nextafter(up, inf), exact));
        EXPECT_FALSE(within_rcp_ulp(std::nextafter(std::nextafter(down, 0.f), 0.f), exact));
    }
    EXPECT_FALSE(within_rcp_ulp(inf, 1e20));
    EXPECT_FALSE(within_rcp_ulp(std::numeric_limits<float>::quiet_NaN(), 1.0));
}

// Rows over 16 binades, each with its amax in another column, so that every lane and each half
// of a row's lanes holds a row maximum somewhere.
TEST(TestCkTileFmhaFwdDynQQuant, Range)
{
    SKIP_WITHOUT_GFX125();
    std::mt19937 rng(20261124);
    QRows rows;
    for(index_t r = 0; r < 4 * kHdim; ++r)
        rows.add_amax(std::ldexp(1.f, static_cast<int>(r % 16) - 8), r % kHdim, rng);
    check_rows(rows);
}

// Zero rows (+0 and -0), tiny and flushed scales: scale 0 or the f32 minimum normal, q 0.
TEST(TestCkTileFmhaFwdDynQQuant, TinyAndZero)
{
    SKIP_WITHOUT_GFX125();
    std::mt19937 rng(20261125);
    const float threshold = 1.75f * std::ldexp(1.f, -118); // amax of scale 2^-126
    const bf16_t below_threshold =
        bf16_bits(static_cast<uint16_t>(ck_tile::bit_cast<uint16_t>(to_bf16(threshold)) - 1));
    QRows rows;
    rows.add(std::vector<bf16_t>(kHdim, to_bf16(0.f)));
    rows.add(std::vector<bf16_t>(kHdim, to_bf16(-0.f)));
    rows.add(std::vector<bf16_t>(kHdim, bf16_bits(1))); // the bf16 minimum subnormal
    rows.add_amax(std::ldexp(1.f, -120), 3, rng);
    rows.add_amax(threshold, 17, rng);
    rows.add_amax(ck_tile::type_convert<float>(below_threshold), 18, rng);
    rows.add_amax(std::ldexp(1.f, -126), 64, rng);
    // amax 2^-100 among subnormal values
    std::vector<bf16_t> subnormals(kHdim);
    for(index_t i = 0; i < kHdim; ++i)
        subnormals[i] = bf16_bits(static_cast<uint16_t>(1 + i % 0x7f));
    subnormals[90] = to_bf16(std::ldexp(1.f, -100));
    rows.add(subnormals);
    // the scale an f32 subnormal: 2^-126 .. 2^-118
    for(int e = -126; e <= -118; ++e)
        rows.add_amax(std::ldexp(1.f, e), (e + 126) * 13, rng);
    check_rows(rows);
}

// Ties of fp8 at scale 1 (amax 448) and 2^-6 (amax 7): x * inv lands on them where inv is exact
// (Pk8RneTies takes the ties to the conversion directly).
TEST(TestCkTileFmhaFwdDynQQuant, Round)
{
    SKIP_WITHOUT_GFX125();
    std::vector<float> ties;
    float prev = 0.f;
    for(uint8_t b = 1; b <= 0x7e; ++b)
    {
        const float v = ck_tile::type_convert<float>(ck_tile::bit_cast<fp8_t>(b));
        ties.push_back((prev + v) / 2.f);
        prev = v;
    }
    QRows rows;
    for(const float amax : {448.f, 7.f})
        for(index_t r = 0; r < 2; ++r)
        {
            std::vector<bf16_t> row(kHdim);
            for(index_t i = 0; i < kHdim; ++i)
                row[i] = to_bf16((r == 0 ? 1.f : -1.f) * ties[i % ties.size()] * (amax / kFp8Max));
            row[kHdim - 1] = to_bf16(amax);
            rows.add(row);
        }
    check_rows(rows);
}

// The pk8 conversion on the midpoints of neighboring fp8 values, either sign: to the even code.
TEST(TestCkTileFmhaFwdDynQQuant, Pk8RneTies)
{
    SKIP_WITHOUT_GFX125();
    constexpr uint8_t kSign    = 0x80;
    constexpr uint8_t kMaxCode = 0x7e; // 448
    const auto value           = [](uint8_t code) {
        return ck_tile::type_convert<float>(ck_tile::bit_cast<fp8_t>(code));
    };
    std::vector<float> ties;
    std::vector<uint8_t> expected;
    for(const uint8_t sign : {uint8_t{0}, kSign})
        for(uint8_t upper = 1; upper <= kMaxCode; ++upper)
        {
            const uint8_t lower = upper - 1;
            const float tie     = (value(lower) + value(upper)) / 2.f;
            ties.push_back(sign == kSign ? -tie : tie);
            expected.push_back(sign | (lower % 2 == 0 ? lower : upper));
        }
    while(ties.size() % kPack != 0)
    {
        ties.push_back(0.f);
        expected.push_back(0);
    }

    const index_t groups       = static_cast<index_t>(ties.size()) / kPack;
    constexpr index_t kThreads = 64;
    ck_tile::DeviceMem x_buf(ties.size() * sizeof(float));
    ck_tile::DeviceMem q_buf(ties.size() * sizeof(fp8_t));
    x_buf.ToDevice(ties.data());
    convert_pk8<<<ck_tile::integer_divide_ceil(groups, kThreads), kThreads>>>(
        static_cast<const float*>(x_buf.GetDeviceBuffer()),
        static_cast<fp8_t*>(q_buf.GetDeviceBuffer()),
        groups);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    std::vector<fp8_t> q(ties.size());
    q_buf.FromDevice(q.data());
    for(std::size_t i = 0; i < ties.size(); ++i)
        EXPECT_EQ(bits(q[i]), expected[i])
            << "x " << ties[i] << " -> " << ck_tile::type_convert<float>(q[i]) << ", even "
            << value(expected[i]);
}

// Every bf16 amax mantissa: where the reciprocal rounds up, the row maximum lands past 448.
TEST(TestCkTileFmhaFwdDynQQuant, Overshoot)
{
    SKIP_WITHOUT_GFX125();
    std::mt19937 rng(20261126);
    QRows rows;
    for(const uint16_t exponent : {0x3f80, 0x4000, 0x4380}) // 1, 2, 256
        for(uint16_t m = 0; m < 0x80; ++m)
            rows.add_amax(ck_tile::type_convert<float>(bf16_bits(exponent | m)), m, rng);
    const index_t overshoot = check_rows(rows);
    std::cout << "values with x * inv > 448: " << overshoot << std::endl;
    RecordProperty("overshoot_values", static_cast<int>(overshoot));
}
