// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The target model's fused epilogues (include/ck_tile/ops/megakernel/epilogue/) in one persistent
// ck_tile::Megakernel launch on random data, checked against the host reference in
// epilogue_reference.hpp. Stages (TP8 rank shapes, M token rows):
//   0 GemmPhase       qkvg = X [M, 8192] * Wqkvg^T, Wqkvg [2304, 8192]
//   1 QkvRowOps       sliding layer: q / k norm + RoPE, q -> q1, K / V -> bf16 paged cache 1
//   2 QkvRowOps       global layer (NoPE): q -> q2, K / V -> fp8 paged cache 2      (after 0)
//   3 AttnGate        o_norm(attn) * sigmoid(gate of qkvg) -> o_proj input          (after 0)
//   4 ResidualNorm    post-attention: 2 fp32 partials summed, RMSNorm, residual gate, Centered
//   5 ResidualNorm    post-feedforward: InputScaled, residual gate (residual = x of 4), Centered
//   6..8 NormQuant    pre_expert_norm of a pooler output + fp8 per row / MXFP8 / MXFP4
// Exit code: 0 pass, 1 wrong result, 2 unsupported arguments, 3 error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/arg_parser.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/megakernel.hpp"
#include "ck_tile/ops/megakernel/epilogue/epilogues.hpp"

#include "epilogue_reference.hpp"

namespace {

using DataType    = ck_tile::bf16_t;
using AccDataType = float;
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using index_t     = ck_tile::index_t;

constexpr index_t kH      = 8192;
constexpr index_t kP      = 4096;
constexpr index_t kQkvN   = 2304;
constexpr index_t kQN     = 1024;
constexpr index_t kPage   = 16;
constexpr index_t kMaxPos = 131072;

// Same GEMM instance as example 58: comp_tdm_v3, 128 x 128 x 64 tile, 2 x 2 waves.
// NPerBlock = 128 keeps every 64-wide head inside one tile (see qkv_rowops_phase.hpp).
// A template like example 58's: its member types must not be instantiated in a host context.
template <index_t TileN>
struct GemmInstance
{
    using TileShape   = ck_tile::TileGemmShape<ck_tile::sequence<128, TileN, 64>,
                                               ck_tile::sequence<2, 2, 1>,
                                               ck_tile::sequence<16, 16, 32>>;
    using Partitioner = ck_tile::GemmTile1DPartitioner<TileShape>;
    using Traits      = ck_tile::
        TileGemmUniversalTraits<false, false, false, true, RowMajor, ColumnMajor, RowMajor>;
    using Problem =
        ck_tile::UniversalGemmPipelineProblem<DataType,
                                              DataType,
                                              AccDataType,
                                              TileShape,
                                              Traits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;
    using Pipeline        = ck_tile::GemmPipelineAgBgCrCompTDMV3<Problem>;
    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<DataType,
                                                             DataType,
                                                             ck_tile::tuple<>,
                                                             AccDataType,
                                                             DataType,
                                                             ck_tile::tuple<>,
                                                             RowMajor,
                                                             ck_tile::element_wise::PassThrough,
                                                             128,
                                                             TileN,
                                                             2,
                                                             2,
                                                             16,
                                                             16,
                                                             32,
                                                             false,
                                                             1,
                                                             false,
                                                             1,
                                                             1,
                                                             true>;
    using Epilogue        = ck_tile::TdmEpilogue<EpilogueProblem>;
    using Kernel          = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;
};

using QkvGemm            = GemmInstance<128>;
using GemmStage          = ck_tile::GemmPhase<QkvGemm::Kernel>;
constexpr index_t kWarps = 4; // the GEMM's 2 x 2 waves; Megakernel checks every phase agrees

using QkvOps    = ck_tile::MkQkvRowOpsPhase<kWarps, ck_tile::bf16_t, kPage>;
using QkvOpsFp8 = ck_tile::MkQkvRowOpsPhase<kWarps, ck_tile::fp8_t, kPage>;
using AttnGate  = ck_tile::MkAttnGatePhase<kWarps>;
using PostAttn = ck_tile::MkResidualNormPhase<kH, kWarps, ck_tile::MkFirstNorm::kRms, float, 2>;
using PostFfn  = ck_tile::MkResidualNormPhase<kH, kWarps, ck_tile::MkFirstNorm::kInputScaled>;
using QuantRow = ck_tile::MkNormQuantPhase<kP, kWarps, ck_tile::MkActQuant::kFp8Row>;
using QuantMxFp8 = ck_tile::MkNormQuantPhase<kP, kWarps, ck_tile::MkActQuant::kMxFp8>;
using QuantMxFp4 = ck_tile::MkNormQuantPhase<kP, kWarps, ck_tile::MkActQuant::kMxFp4>;

using Epilogues = ck_tile::Megakernel<ck_tile::megakernel_stage<GemmStage>,
                                      ck_tile::megakernel_stage<QkvOps, ck_tile::after_rows<0>>,
                                      ck_tile::megakernel_stage<QkvOpsFp8, ck_tile::after_rows<0>>,
                                      ck_tile::megakernel_stage<AttnGate, ck_tile::after_rows<0>>,
                                      ck_tile::megakernel_stage<PostAttn>,
                                      ck_tile::megakernel_stage<PostFfn, ck_tile::after_rows<4>>,
                                      ck_tile::megakernel_stage<QuantRow>,
                                      ck_tile::megakernel_stage<QuantMxFp8>,
                                      ck_tile::megakernel_stage<QuantMxFp4>>;

#define HIP_CHECK(expr)                                                                    \
    do                                                                                     \
    {                                                                                      \
        const hipError_t err_ = (expr);                                                    \
        if(err_ != hipSuccess)                                                             \
        {                                                                                  \
            throw std::runtime_error(std::string("HIP error ") + hipGetErrorString(err_) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));  \
        }                                                                                  \
    } while(0)

struct DeviceBuffer
{
    void* p      = nullptr;
    size_t bytes = 0;
    explicit DeviceBuffer(size_t n) : bytes(n) { HIP_CHECK(hipMalloc(&p, n)); }
    ~DeviceBuffer() { (void)hipFree(p); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* as() const
    {
        return static_cast<T*>(p);
    }
    template <typename T>
    void upload(const std::vector<T>& v)
    {
        HIP_CHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    template <typename T>
    std::vector<T> download() const
    {
        std::vector<T> v(bytes / sizeof(T));
        HIP_CHECK(hipMemcpy(v.data(), p, bytes, hipMemcpyDeviceToHost));
        return v;
    }
};

using mk_ref::bf16_bits;
using mk_ref::bf16_float;

std::vector<uint16_t> random_bf16(size_t n, float sigma, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, sigma);
    std::vector<uint16_t> v(n);
    for(auto& x : v)
        x = bf16_bits(dist(rng));
    return v;
}

std::vector<float> random_f32(size_t n, float sigma, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.f, sigma);
    std::vector<float> v(n);
    for(auto& x : v)
        x = dist(rng);
    return v;
}

std::vector<float> widen(const uint16_t* p, size_t n)
{
    std::vector<float> v(n);
    for(size_t i = 0; i < n; ++i)
        v[i] = bf16_float(p[i]);
    return v;
}

/// Counts and prints elements outside |got - ref| <= rel * |ref| + abs.
struct Checker
{
    const char* name;
    double rel, abs;
    long bad = 0, n = 0;
    void operator()(double got, double ref, long where)
    {
        ++n;
        if(std::fabs(got - ref) <= rel * std::fabs(ref) + abs)
            return;
        if(bad++ < 5)
            std::printf("  %s[%ld] = %g, ref %g\n", name, where, got, ref);
    }
    /// Accepts got anywhere in [lo, hi], widened by the tolerance of each end.
    void range(double got, double lo, double hi, double ref, long where)
    {
        ++n;
        if(got >= lo - (rel * std::fabs(lo) + abs) && got <= hi + (rel * std::fabs(hi) + abs))
            return;
        if(bad++ < 5)
            std::printf("  %s[%ld] = %g, ref %g (range [%g, %g])\n", name, where, got, ref, lo, hi);
    }
    int report() const
    {
        std::printf("  %-28s %ld / %ld outside tolerance\n", name, bad, n);
        return bad != 0;
    }
};

// The device helpers' host paths against the independent reference encoders.
int check_host_encoders()
{
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    long bad = 0;
    for(int i = 0; i < 200000; ++i)
    {
        const float x = nd(rng) * std::exp2(static_cast<float>(static_cast<int>(rng() % 40) - 20));
        bad += (ck_tile::mk_pack_fp8x4(x, 0.f, 0.f, 0.f) & 0xFF) != mk_ref::fp8_encode(x);
        bad += ck_tile::mk_e2m1_encode(x) != mk_ref::e2m1_encode(x);
        const float a = std::fabs(x);
        bad += ck_tile::mk_mx_scale_e8m0<8>(a) != mk_ref::mx_scale(a, 8);
        bad += ck_tile::mk_mx_scale_e8m0<2>(a) != mk_ref::mx_scale(a, 2);
    }
    std::printf("host encoders vs reference: %ld mismatches\n", bad);
    return bad != 0;
}

// A template, like example 58's run: the GEMM types must first be instantiated in a function.
template <typename MK>
int run(const ck_tile::ArgParser& arg)
{
    const index_t M     = arg.get_int("m");
    const int launches  = arg.get_int("launches");
    const int gemm_rows = arg.get_int("check_rows");
    const float k_scale = 0.05f, v_scale = 0.02f; // fp8 cache scales

    int failures = check_host_encoders();

    // ---- inputs --------------------------------------------------------------------------
    const auto x    = random_bf16(size_t(M) * kH, 1.f, 1);
    const auto wqkv = random_bf16(size_t(kQkvN) * kH, 1.f / std::sqrt(float(kH)), 2);
    const auto attn = random_bf16(size_t(M) * kQN, 0.05f, 3);
    const auto pa0  = random_f32(size_t(M) * kH, 3.f, 4); // o_proj partials of two ranks
    const auto pa1  = random_f32(size_t(M) * kH, 3.f, 5);
    const auto res0 = random_bf16(size_t(M) * kH, 2.f, 6);
    const auto g1   = random_f32(kH, 0.5f, 7);
    const auto w1n  = random_bf16(kH, 0.1f, 8);
    const auto mffn = random_bf16(size_t(M) * kH, 4.f, 9);
    const auto w2f  = random_bf16(kH, 0.1f, 10);
    const auto g2   = random_f32(kH, 0.5f, 11);
    const auto w2n  = random_bf16(kH, 0.1f, 12);
    const auto pool = random_bf16(size_t(M) * kP, 2.f, 13);
    const auto wpe  = random_bf16(kP, 0.1f, 14);

    // Pages: M tokens scattered over the slots of a few pages, row 1 is padding (slot -1).
    const index_t num_pages = (M + kPage - 1) / kPage + 2;
    std::vector<int32_t> slots(num_pages * kPage);
    std::iota(slots.begin(), slots.end(), 0);
    std::mt19937 rng(15);
    std::shuffle(slots.begin(), slots.end(), rng);
    slots.resize(M);
    if(M > 1)
        slots[1] = -1;
    std::vector<int32_t> pos(M);
    for(auto& p : pos)
        p = static_cast<int32_t>(rng() % kMaxPos);
    const auto rope = ck_tile::mk_make_rope_table(kMaxPos);

    const size_t cache_elems = size_t(num_pages) * 2 * kPage * 64;
    DeviceBuffer d_x(x.size() * 2), d_w(wqkv.size() * 2), d_qkvg(size_t(M) * kQkvN * 2);
    DeviceBuffer d_q1(size_t(M) * kQN * 2), d_q2(size_t(M) * kQN * 2);
    DeviceBuffer d_k1(cache_elems * 2), d_v1(cache_elems * 2), d_k2(cache_elems), d_v2(cache_elems);
    DeviceBuffer d_slots(M * 4), d_pos(M * 4), d_rope(rope.size() * 4);
    DeviceBuffer d_attn(attn.size() * 2), d_o(size_t(M) * kQN * 2);
    DeviceBuffer d_pa0(pa0.size() * 4), d_pa1(pa1.size() * 4), d_res0(res0.size() * 2);
    std::vector<float> ab1(2 * kH), ab2(2 * kH); // alpha then beta
    ck_tile::mk_residual_gate_coeffs(g1.data(), kH, ab1.data(), ab1.data() + kH);
    ck_tile::mk_residual_gate_coeffs(g2.data(), kH, ab2.data(), ab2.data() + kH);
    DeviceBuffer d_g1(2 * kH * 4), d_w1n(kH * 2), d_x1(size_t(M) * kH * 2),
        d_h1(size_t(M) * kH * 2);
    DeviceBuffer d_mffn(mffn.size() * 2), d_w2f(kH * 2), d_g2(2 * kH * 4), d_w2n(kH * 2);
    DeviceBuffer d_x2(size_t(M) * kH * 2), d_h2(size_t(M) * kH * 2);
    DeviceBuffer d_pool(pool.size() * 2), d_wpe(kP * 2), d_p(size_t(M) * kP * 2);
    DeviceBuffer d_qr(size_t(M) * kP), d_sr(M * 4);
    DeviceBuffer d_qm8(size_t(M) * kP), d_sm8(size_t(M) * kP / 32);
    DeviceBuffer d_qm4(size_t(M) * kP / 2), d_sm4(size_t(M) * kP / 32);

    d_x.upload(x), d_w.upload(wqkv), d_slots.upload(slots), d_pos.upload(pos), d_rope.upload(rope);
    d_attn.upload(attn), d_pa0.upload(pa0), d_pa1.upload(pa1), d_res0.upload(res0);
    d_g1.upload(ab1), d_w1n.upload(w1n), d_mffn.upload(mffn), d_w2f.upload(w2f), d_g2.upload(ab2);
    d_w2n.upload(w2n), d_pool.upload(pool), d_wpe.upload(wpe);

    // ---- phase arguments -----------------------------------------------------------------
    const auto gemm = QkvGemm::Kernel::MakeKernelArgs(
        ck_tile::GemmHostArgs(d_x.p, d_w.p, d_qkvg.p, 1, M, kQkvN, kH, kH, kH, kQkvN));

    QkvOps::Args qkv1{};
    qkv1.qkvg         = d_qkvg.as<ck_tile::bf16_t>();
    qkv1.qkvg_stride  = kQkvN;
    qkv1.q_out        = d_q1.as<ck_tile::bf16_t>();
    qkv1.q_stride     = kQN;
    qkv1.k_cache      = d_k1.as<ck_tile::bf16_t>();
    qkv1.v_cache      = d_v1.as<ck_tile::bf16_t>();
    qkv1.slot_mapping = d_slots.as<int32_t>();
    qkv1.positions    = d_pos.as<int32_t>();
    qkv1.rope_table   = d_rope.as<float>();
    qkv1.k_scale = qkv1.v_scale = 1.f;
    qkv1.M                      = M;
    qkv1.use_rope               = 1;

    QkvOpsFp8::Args qkv2{};
    qkv2.qkvg         = d_qkvg.as<ck_tile::bf16_t>();
    qkv2.qkvg_stride  = kQkvN;
    qkv2.q_out        = d_q2.as<ck_tile::bf16_t>();
    qkv2.q_stride     = kQN;
    qkv2.k_cache      = d_k2.as<ck_tile::fp8_t>();
    qkv2.v_cache      = d_v2.as<ck_tile::fp8_t>();
    qkv2.slot_mapping = d_slots.as<int32_t>();
    qkv2.positions    = nullptr;
    qkv2.rope_table   = nullptr;
    qkv2.k_scale      = k_scale;
    qkv2.v_scale      = v_scale;
    qkv2.M            = M;
    qkv2.use_rope     = 0;

    AttnGate::Args ag{d_attn.as<ck_tile::bf16_t>(),
                      kQN,
                      d_qkvg.as<ck_tile::bf16_t>() + QkvOps::kGateCol,
                      kQkvN,
                      d_o.as<ck_tile::bf16_t>(),
                      kQN,
                      M};

    PostAttn::Args pa{};
    pa.in[0]           = d_pa0.as<float>();
    pa.in[1]           = d_pa1.as<float>();
    pa.num_inputs      = 2;
    pa.in_stride       = kH;
    pa.first_w         = nullptr;
    pa.first_eps       = ck_tile::MkConstants::kPostNormEps;
    pa.gate_alpha      = d_g1.as<float>();
    pa.gate_beta       = d_g1.as<float>() + kH;
    pa.residual        = d_res0.as<ck_tile::bf16_t>();
    pa.residual_stride = kH;
    pa.x_out           = d_x1.as<ck_tile::bf16_t>();
    pa.x_stride        = kH;
    pa.next_w          = d_w1n.as<ck_tile::bf16_t>();
    pa.next_eps        = ck_tile::MkConstants::kRmsEps;
    pa.h_out           = d_h1.as<ck_tile::bf16_t>();
    pa.h_stride        = kH;
    pa.M               = M;

    PostFfn::Args pf{};
    pf.in[0]           = d_mffn.as<ck_tile::bf16_t>();
    pf.num_inputs      = 1;
    pf.in_stride       = kH;
    pf.first_w         = d_w2f.as<ck_tile::bf16_t>();
    pf.first_eps       = ck_tile::MkConstants::kPostNormEps;
    pf.gate_alpha      = d_g2.as<float>();
    pf.gate_beta       = d_g2.as<float>() + kH;
    pf.residual        = d_x1.as<ck_tile::bf16_t>(); // stage 4's new residual
    pf.residual_stride = kH;
    pf.x_out           = d_x2.as<ck_tile::bf16_t>();
    pf.x_stride        = kH;
    pf.next_w          = d_w2n.as<ck_tile::bf16_t>();
    pf.next_eps        = ck_tile::MkConstants::kRmsEps;
    pf.h_out           = d_h2.as<ck_tile::bf16_t>();
    pf.h_stride        = kH;
    pf.M               = M;

    auto quant_args = [&](void* p_out, void* q, index_t q_stride, void* s, index_t s_stride) {
        return ck_tile::MkNormQuantArgs{d_pool.as<ck_tile::bf16_t>(),
                                          kP,
                                          d_wpe.as<ck_tile::bf16_t>(),
                                          ck_tile::MkConstants::kRmsEps,
                                          static_cast<ck_tile::bf16_t*>(p_out),
                                          kP,
                                          q,
                                          q_stride,
                                          s,
                                          s_stride,
                                          M};
    };
    const auto qr  = quant_args(d_p.p, d_qr.p, kP, d_sr.p, 1);
    const auto qm8 = quant_args(nullptr, d_qm8.p, kP, d_sm8.p, kP / 32);
    const auto qm4 = quant_args(nullptr, d_qm4.p, kP / 2, d_sm4.p, kP / 32);

    const typename MK::ArgsTuple args{gemm, qkv1, qkv2, ag, pa, pf, qr, qm8, qm4};
    const index_t n_counters = MK::NumCounters(args);
    DeviceBuffer d_counters(size_t(n_counters) * 4);
    HIP_CHECK(hipMemset(d_counters.p, 0, d_counters.bytes));
    const auto kargs = MK::MakeKargs(args, d_counters.as<uint32_t>());
    if(!MK::IsSupportedArgument(kargs))
    {
        std::printf("megakernel arguments not supported\n");
        return 2;
    }

    const ck_tile::stream_config s{nullptr, false};
    // MK::BlockSize() is the host pass's kBlockSize, which assumes wave64 (4 waves -> 256
    // threads); the GEMM kernel's BlockSize() halves it at run time on wave32 targets.
    const dim3 block       = QkvGemm::Kernel::BlockSize();
    const index_t resident = MK::MaxResidentGridSize(s);
    index_t grid           = arg.get_int("grid");
    grid = std::min(grid > 0 ? grid : std::max<index_t>(GemmStage::NumWork(gemm), M), resident);
    std::printf("M %d: grid %d x %u threads (resident %d), LDS %d B, %d counters\n",
                M,
                grid,
                block.x,
                resident,
                static_cast<int>(MK::kSmemSize),
                n_counters);

    // Outputs start as NaN (0xFF bytes) so unwritten or skipped elements stay visible.
    DeviceBuffer* outs[] = {&d_qkvg,
                            &d_q1,
                            &d_q2,
                            &d_k1,
                            &d_v1,
                            &d_k2,
                            &d_v2,
                            &d_o,
                            &d_x1,
                            &d_h1,
                            &d_x2,
                            &d_h2,
                            &d_p,
                            &d_qr,
                            &d_sr,
                            &d_qm8,
                            &d_sm8,
                            &d_qm4,
                            &d_sm4};
    for(int l = 0; l < launches; ++l)
    {
        for(auto* b : outs)
            HIP_CHECK(hipMemset(b->p, 0xFF, b->bytes));
        ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(MK{}, dim3(grid), block, 0, kargs));
        HIP_CHECK(hipDeviceSynchronize());
    }

    // ---- checks --------------------------------------------------------------------------
    const auto qkvg = d_qkvg.download<uint16_t>();
    {
        // GEMM: a few rows against a double reference (tolerance of example 58).
        Checker c{"qkvg", 0, 0};
        for(int i = 0; i < gemm_rows && i < M; ++i)
        {
            const index_t r = i * (M - 1) / std::max(1, gemm_rows - 1);
            for(index_t n = 0; n < kQkvN; ++n)
            {
                double ref = 0, mag = 0;
                for(index_t k = 0; k < kH; ++k)
                {
                    const double p = double(bf16_float(x[size_t(r) * kH + k])) *
                                     bf16_float(wqkv[size_t(n) * kH + k]);
                    ref += p;
                    mag += std::fabs(p);
                }
                c.abs = mag * std::ldexp(1.0, -16);
                c.rel = std::ldexp(1.0, -7);
                c(bf16_float(qkvg[size_t(r) * kQkvN + n]), ref, long(r) * kQkvN + n);
            }
        }
        failures += c.report();
    }

    // q / K / V of both layers from the GPU's qkvg.
    {
        const auto q1 = d_q1.download<uint16_t>(), q2 = d_q2.download<uint16_t>();
        const auto k1 = d_k1.download<uint16_t>(), v1 = d_v1.download<uint16_t>();
        const auto k2 = d_k2.download<uint8_t>(), v2 = d_v2.download<uint8_t>();
        // Two bf16 ulps of the value plus one of the head RMS (RoPE sums can cancel).
        Checker cq1{"q (sliding)", std::ldexp(1.0, -7), mk_ref::kQkScale * std::ldexp(1.0, -7)};
        Checker cq2{"q (global)", std::ldexp(1.0, -7), mk_ref::kQkScale * std::ldexp(1.0, -7)};
        Checker ck1{"K cache bf16", std::ldexp(1.0, -7), std::ldexp(1.0, -7)};
        Checker cv1{"V cache bf16", 0, 0};
        // fp8: one e4m3 ulp (2^-3 relative), plus the bf16 input's ulp scaled.
        Checker ck2{"K cache fp8", std::ldexp(1.0, -3), std::ldexp(1.0, -9)};
        Checker cv2{"V cache fp8", std::ldexp(1.0, -3), std::ldexp(1.0, -9)};
        std::vector<char> written(cache_elems, 0);
        for(index_t r = 0; r < M; ++r)
        {
            const auto row = widen(&qkvg[size_t(r) * kQkvN], kQkvN);
            for(int layer = 0; layer < 2; ++layer)
            {
                const bool sliding = layer == 0;
                for(index_t h = 0; h < 16 + 2; ++h)
                {
                    float head[64];
                    std::copy(&row[h * 64], &row[h * 64] + 64, head);
                    mk_ref::qk_head(head, h < 16, sliding, pos[r]);
                    for(int d = 0; d < 64; ++d)
                    {
                        if(h < 16)
                        {
                            const size_t i = size_t(r) * kQN + h * 64 + d;
                            (sliding ? cq1
                                     : cq2)(bf16_float((sliding ? q1 : q2)[i]), head[d], long(i));
                            continue;
                        }
                        if(slots[r] < 0)
                            continue;
                        const index_t kv = h - 16;
                        const size_t dst =
                            ((size_t(slots[r] / kPage) * 2 + kv) * kPage + slots[r] % kPage) * 64 +
                            d;
                        const float v_val = row[QkvOps::kVCol + kv * 64 + d];
                        written[dst]      = 1;
                        if(sliding)
                        {
                            ck1(bf16_float(k1[dst]), head[d], long(dst));
                            cv1(bf16_float(v1[dst]), v_val, long(dst));
                        }
                        else
                        {
                            ck2(mk_ref::fp8_decode(k2[dst]) * k_scale,
                                mk_ref::fp8_decode(
                                    mk_ref::fp8_encode(head[d] * (1.f / k_scale))) *
                                    k_scale,
                                long(dst));
                            cv2(mk_ref::fp8_decode(v2[dst]) * v_scale,
                                mk_ref::fp8_decode(
                                    mk_ref::fp8_encode(v_val * (1.f / v_scale))) *
                                    v_scale,
                                long(dst));
                        }
                    }
                }
            }
        }
        long touched = 0;
        for(size_t i = 0; i < cache_elems; ++i)
            if(!written[i])
                touched += k1[i] != 0xFFFF || v1[i] != 0xFFFF || k2[i] != 0xFF || v2[i] != 0xFF;
        std::printf("  %-28s %ld\n", "cache slots written wrongly", touched);
        failures += cq1.report() + cq2.report() + ck1.report() + cv1.report() + ck2.report() +
                    cv2.report() + (touched != 0);
    }

    // Attention gate.
    {
        const auto o = d_o.download<uint16_t>();
        Checker c{"o_proj input", std::ldexp(1.0, -7), std::ldexp(1.0, -9)};
        for(index_t r = 0; r < M; ++r)
            for(index_t h = 0; h < 16; ++h)
            {
                float oh[64], gh[64];
                for(int d = 0; d < 64; ++d)
                {
                    oh[d] = bf16_float(attn[size_t(r) * kQN + h * 64 + d]);
                    gh[d] = bf16_float(qkvg[size_t(r) * kQkvN + QkvOps::kGateCol + h * 64 + d]);
                }
                mk_ref::attn_gate_head(oh, gh);
                for(int d = 0; d < 64; ++d)
                {
                    const size_t i = size_t(r) * kQN + h * 64 + d;
                    c(bf16_float(o[i]), oh[d], long(i));
                }
            }
        failures += c.report();
    }

    // Residual norms: post-attention, then post-feedforward on the GPU's x1.
    {
        const auto x1 = d_x1.download<uint16_t>(), h1 = d_h1.download<uint16_t>();
        const auto x2 = d_x2.download<uint16_t>(), h2 = d_h2.download<uint16_t>();
        const auto g1w = widen(w1n.data(), kH), w2fw = widen(w2f.data(), kH),
                   w2nw = widen(w2n.data(), kH);
        Checker cx1{"x post-attention", std::ldexp(1.0, -7), std::ldexp(1.0, -10)};
        Checker ch1{"h post-attention", std::ldexp(1.0, -7), std::ldexp(1.0, -10)};
        Checker cx2{"x post-feedforward", std::ldexp(1.0, -7), std::ldexp(1.0, -10)};
        Checker ch2{"h post-feedforward", std::ldexp(1.0, -7), std::ldexp(1.0, -10)};
        // x: the post-norm output is rounded to bf16 before the residual gate, so a few fp32 ulps
        // of error in the row's rsqrt (the GPU sums the squares as an fp32 tree and uses
        // v_rsq_f32; the reference sums in double) move elements that sit on a bf16 rounding
        // boundary by one ulp, and the gate's cancellation (alpha * res + beta * an) can amplify
        // that to several ulps of x. Accept x for any r within kRSlack (2^-18 relative), a few
        // times the fp32 tree error over 8192 terms and 2^10 below the bf16 spacing, so a wrong
        // norm (eps, width, a missing partial) still fails. (With a serial fp32 reference sum,
        // 10-14 elements per 131072 failed; the GPU matches the double reference exactly here.)
        // h: CenteredRMSNorm of the GPU's own x, so an accepted boundary flip of x does not
        // cascade (h has no bf16 rounding before its last, so the tolerance covers its r).
        constexpr float kRSlack = 1.f / 262144.f;
        std::vector<float> sum(kH), xr(kH), hr(kH), xlo(kH), xhi(kH), hg(kH);
        for(index_t r = 0; r < M; ++r)
        {
            const size_t o = size_t(r) * kH;
            for(index_t i = 0; i < kH; ++i)
                sum[i] = pa0[o + i] + pa1[o + i];
            const auto res = widen(&res0[o], kH);
            mk_ref::residual_norm_row(sum.data(),
                                        false,
                                        nullptr,
                                        g1.data(),
                                        res.data(),
                                        g1w.data(),
                                        kH,
                                        xr.data(),
                                        hr.data(),
                                        kRSlack,
                                        xlo.data(),
                                        xhi.data());
            const auto xg1 = widen(&x1[o], kH);
            mk_ref::centered_rmsnorm(xg1.data(), g1w.data(), kH, mk_ref::kRmsEps, hg.data());
            for(index_t i = 0; i < kH; ++i)
            {
                cx1.range(xg1[i], xlo[i], xhi[i], xr[i], long(o + i));
                ch1(bf16_float(h1[o + i]), hg[i], long(o + i));
            }
            const auto m    = widen(&mffn[o], kH);
            const auto res2 = widen(&x1[o], kH);
            mk_ref::residual_norm_row(m.data(),
                                        true,
                                        w2fw.data(),
                                        g2.data(),
                                        res2.data(),
                                        w2nw.data(),
                                        kH,
                                        xr.data(),
                                        hr.data(),
                                        kRSlack,
                                        xlo.data(),
                                        xhi.data());
            const auto xg2 = widen(&x2[o], kH);
            mk_ref::centered_rmsnorm(xg2.data(), w2nw.data(), kH, mk_ref::kRmsEps, hg.data());
            for(index_t i = 0; i < kH; ++i)
            {
                cx2.range(xg2[i], xlo[i], xhi[i], xr[i], long(o + i));
                ch2(bf16_float(h2[o + i]), hg[i], long(o + i));
            }
        }
        failures += cx1.report() + ch1.report() + cx2.report() + ch2.report();
    }

    // pre_expert_norm and the three quantisations. The codes are checked against the reference
    // encoders applied to the GPU's p, which makes them exact.
    {
        const auto p   = d_p.download<uint16_t>();
        const auto qrb = d_qr.download<uint8_t>();
        const auto srb = d_sr.download<float>();
        const auto q8 = d_qm8.download<uint8_t>(), s8 = d_sm8.download<uint8_t>();
        const auto q4 = d_qm4.download<uint8_t>(), s4 = d_sm4.download<uint8_t>();
        const auto w = widen(wpe.data(), kP);
        Checker cp{"p = pre_expert_norm", std::ldexp(1.0, -7), std::ldexp(1.0, -10)};
        long bad_row = 0, bad_mx8 = 0, bad_mx4 = 0;
        std::vector<float> pr(kP);
        for(index_t r = 0; r < M; ++r)
        {
            const auto in = widen(&pool[size_t(r) * kP], kP);
            mk_ref::centered_rmsnorm(in.data(), w.data(), kP, mk_ref::kRmsEps, pr.data());
            const auto pg = widen(&p[size_t(r) * kP], kP);
            float amax    = 0;
            for(index_t i = 0; i < kP; ++i)
            {
                cp(pg[i], pr[i], long(r) * kP + i);
                amax = std::max(amax, std::fabs(pg[i]));
            }
            const float inv = amax > 0 ? 448.f / amax : 1.f;
            bad_row += mk_ref::f2u(srb[r]) != mk_ref::f2u(amax > 0 ? amax / 448.f : 1.f);
            for(index_t i = 0; i < kP; ++i)
                bad_row += qrb[size_t(r) * kP + i] != mk_ref::fp8_encode(pg[i] * inv);
            for(index_t b = 0; b < kP / 32; ++b)
            {
                float bmax = 0;
                for(int j = 0; j < 32; ++j)
                    bmax = std::max(bmax, std::fabs(pg[b * 32 + j]));
                const uint8_t e8 = mk_ref::mx_scale(bmax, 8), e4 = mk_ref::mx_scale(bmax, 2);
                bad_mx8 += s8[size_t(r) * (kP / 32) + b] != e8;
                bad_mx4 += s4[size_t(r) * (kP / 32) + b] != e4;
                const float i8 = 1.f / mk_ref::mx_value(e8), i4 = 1.f / mk_ref::mx_value(e4);
                for(int j = 0; j < 32; ++j)
                {
                    const index_t i = b * 32 + j;
                    bad_mx8 += q8[size_t(r) * kP + i] != mk_ref::fp8_encode(pg[i] * i8);
                    const uint8_t byte = q4[size_t(r) * (kP / 2) + i / 2];
                    bad_mx4 +=
                        ((i % 2 ? byte >> 4 : byte) & 15) != mk_ref::e2m1_encode(pg[i] * i4);
                }
            }
        }
        failures += cp.report();
        std::printf("  %-28s %ld / %d codes+scales differ\n", "fp8 per row", bad_row, M * (kP + 1));
        std::printf("  %-28s %ld\n", "MXFP8 codes+scales differ", bad_mx8);
        std::printf("  %-28s %ld\n", "MXFP4 codes+scales differ", bad_mx4);
        failures += (bad_row != 0) + (bad_mx8 != 0) + (bad_mx4 != 0);
    }

    std::printf("%s (%d launches)\n", failures == 0 ? "PASS" : "FAIL", launches);
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("m", "16", "token rows: 8, 16 (one DFlash2 verify window) .. 64")
        .insert("grid", "0", "workgroups; 0 = max(GEMM tiles, M), capped at resident")
        .insert("launches", "3", "launches sharing one counter buffer (last one is checked)")
        .insert("check_rows", "3", "rows of the qkv GEMM checked against a double reference");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        return run<Epilogues>(arg);
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
