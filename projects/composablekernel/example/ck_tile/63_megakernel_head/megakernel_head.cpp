// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The target model's head for DFlash2 verify in one persistent ck_tile::Megakernel launch:
//   stage 0  xn     = RMSNorm(x, 1e-5, norm.w)                       x [M, H] bf16
//   stage 1  logits = 20 * tanh(xn @ lm_head^T * 0.1767767 / 20)      lm_head shard [N, H] bf16
//            fused into the GEMM epilogue: per-row argmax keys (u64 atomic max), optional fp32
//            logits
//   stage 2  argmax tokens, DFlash2 accept of each verify window, keys cleared for the next launch
// One TP rank is emulated: the shard is rows [rank * N, (rank + 1) * N) of a vocab of world * N.
//
// Data: random x, norm weight and lm_head, plus, on even rows, a planted "hot" token whose lm_head
// row is alpha * sign(xn row) (a clear maximum) with an exact duplicate row and a near duplicate
// (one element zeroed: a different fp32 sum that usually rounds to the same bf16 logit) at random
// ids, so the lowest-id tie rule is exercised both for exact and for bf16-rounding ties.
// Drafts follow the reference argmax for a random number of positions, then diverge.
//
// Checks:
//   1. xn against a double reference (at most 1 bf16 ulp).
//   2. Every row's token against double logits of the GPU's xn: no token may be surely larger,
//      or surely equal and of lower id, or an exact duplicate of lower id (head_reference.hpp).
//      max_logit lies within the reference bounds of that token; keys_out matches both.
//   3. Optional fp32 logits (-logits=1) within the reference bounds.
//   4. accepted / num_accepted equal the CPU accept of the GPU tokens.
//   5. -launches launches share one counter buffer and one key buffer without a reset; every
//      launch must give the same outputs and leave the keys zero.
// Exit code: 0 pass, 1 wrong result, 2 unsupported arguments, 3 error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
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
#include "ck_tile/ops/megakernel/head/head.hpp"

#include "head_reference.hpp"

namespace {

using DataType    = ck_tile::bf16_t;
using AccDataType = float;
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using index_t     = ck_tile::index_t;
using Numerics    = ck_tile::head_softcap_numerics;

constexpr index_t kWarpM = 16;
constexpr index_t kWarpN = 16;
constexpr index_t kWarpK = 32;

// comp_tdm_v3 as in example 58: A row-major, B column-major (lm_head [N, K]). The kernel's own
// epilogue is not run (HeadLogitsPhase has its own) but the kernel type needs one.
template <index_t TileM, index_t TileN, index_t TileK, index_t WarpsM, index_t WarpsN>
struct GemmInstance
{
    using TileShape   = ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                               ck_tile::sequence<WarpsM, WarpsN, 1>,
                                               ck_tile::sequence<kWarpM, kWarpN, kWarpK>>;
    using Partitioner = ck_tile::GemmTile1DPartitioner<TileShape>;
    using Traits      = ck_tile::TileGemmUniversalTraits<false, // kPadM
                                                         false, // kPadN
                                                         false, // kPadK
                                                         true,  // DoubleSmemBuffer
                                                         RowMajor,
                                                         ColumnMajor,
                                                         RowMajor>;
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
                                                             TileM,
                                                             TileN,
                                                             WarpsM,
                                                             WarpsN,
                                                             kWarpM,
                                                             kWarpN,
                                                             kWarpK,
                                                             false, // isCTransposed
                                                             1,     // kNumWaveGroups
                                                             false, // FixedVectorSize
                                                             1,
                                                             1,     // BlockedXDLN_PerWarp
                                                             true>; // DoubleSmemBuffer
    using Epilogue        = ck_tile::TdmEpilogue<EpilogueProblem>;
    using Kernel          = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;
};

// Weight streaming at small M: one M tile covers all rows, 4 waves side by side along N (64
// columns each). TODO: tune TileK / TileN for bytes in flight once GPUs are back.
using Gemm16 = GemmInstance<16, 256, 64, 1, 4>;
using Gemm64 = GemmInstance<64, 256, 64, 1, 4>;

// Four waves. Template arguments must not depend on the compile pass (the host pass sees 64-lane
// waves), or the host launches a kernel the device code object does not contain.
constexpr index_t kNumWaves = 4;

template <typename Gemm, Numerics N>
struct Head
{
    using GemmKernel                    = typename Gemm::Kernel;
    using Norm                          = ck_tile::HeadNormPhase<kNumWaves>;
    using Logits                        = ck_tile::HeadLogitsPhase<typename Gemm::Kernel, N>;
    using Accept                        = ck_tile::HeadAcceptPhase<kNumWaves>;
    using MK                            = ck_tile::Megakernel<ck_tile::megakernel_stage<Norm>,
                                                              ck_tile::megakernel_stage<Logits, ck_tile::after_rows<0>>,
                                                              ck_tile::megakernel_stage<Accept, ck_tile::after_all<1>>>;
    static constexpr Numerics kNumerics = N;
};

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
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { HIP_CHECK(hipMalloc(&p, std::max<size_t>(bytes, 4))); }
    ~DeviceBuffer() { (void)hipFree(p); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    T* as() const
    {
        return static_cast<T*>(p);
    }
};

template <typename T>
void to_device(DeviceBuffer& d, const std::vector<T>& h)
{
    HIP_CHECK(hipMemcpy(d.p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
}
template <typename T>
void to_host(std::vector<T>& h, const DeviceBuffer& d)
{
    HIP_CHECK(hipMemcpy(h.data(), d.p, h.size() * sizeof(T), hipMemcpyDeviceToHost));
}

struct Problem
{
    int M, H, N, world, rank, window;
    int vocab_offset() const { return rank * N; }
};

struct Data
{
    std::vector<uint16_t> x, w, wt, xn_ref;
    std::vector<int32_t> drafts;
    int planted = 0;
};

Data make_data(const Problem& p, uint64_t seed)
{
    using head_ref::bf16_to_float;
    using head_ref::float_to_bf16;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> scale(0.5f, 4.f);
    Data d;

    d.x.resize(size_t(p.M) * p.H);
    for(int r = 0; r < p.M; ++r)
    {
        const float s = scale(rng);
        for(int k = 0; k < p.H; ++k)
            d.x[size_t(r) * p.H + k] = float_to_bf16(s * nd(rng));
    }
    d.w.resize(p.H);
    for(auto& v : d.w)
        v = float_to_bf16(1.f + 0.1f * nd(rng));
    // Raw logits of std ~25: the softcap's tanh argument reaches ~0.9 at the random maximum.
    const float sw = 25.f / std::sqrt(static_cast<float>(p.H));
    d.wt.resize(size_t(p.N) * p.H);
    for(auto& v : d.wt)
        v = float_to_bf16(sw * nd(rng));

    d.xn_ref = head_ref::rmsnorm(d.x, d.w, p.M, p.H, ck_tile::MkHeadConstants::kNormEps);

    // Hot tokens on even rows: raw logit ~150, plus an exact and a near duplicate.
    std::set<int> used;
    auto fresh = [&]() {
        for(;;)
        {
            const int t = static_cast<int>(rng() % static_cast<uint64_t>(p.N));
            if(used.insert(t).second)
                return t;
        }
    };
    for(int r = 0; r < p.M && 3 * (d.planted + 1) <= p.N; r += 2)
    {
        const uint16_t* xr = &d.xn_ref[size_t(r) * p.H];
        double l1          = 0;
        int kmin           = 0;
        for(int k = 0; k < p.H; ++k)
        {
            l1 += std::fabs(bf16_to_float(xr[k]));
            if(std::fabs(bf16_to_float(xr[k])) < std::fabs(bf16_to_float(xr[kmin])))
                kmin = k;
        }
        const float alpha = static_cast<float>(150.0 / std::max(l1, 1e-6));
        const int hot = fresh(), dup = fresh(), near = fresh();
        for(int k = 0; k < p.H; ++k)
        {
            const uint16_t v             = float_to_bf16(bf16_to_float(xr[k]) < 0 ? -alpha : alpha);
            d.wt[size_t(hot) * p.H + k]  = v;
            d.wt[size_t(dup) * p.H + k]  = v;
            d.wt[size_t(near) * p.H + k] = k == kmin ? uint16_t(0) : v;
        }
        ++d.planted;
    }
    return d;
}

std::vector<int> all_rows(int M)
{
    std::vector<int> r(M);
    for(int i = 0; i < M; ++i)
        r[i] = i;
    return r;
}

template <typename HeadT>
int run(const ck_tile::ArgParser& arg, const Problem& p)
{
    using MK          = typename HeadT::MK;
    using Norm        = typename HeadT::Norm;
    using Logits      = typename HeadT::Logits;
    using Accept      = typename HeadT::Accept;
    constexpr auto nm = HeadT::kNumerics;
    const float mult  = ck_tile::MkHeadConstants::kOutMultiplier;
    const float cap   = ck_tile::MkHeadConstants::kSoftcap;

    const int launches    = arg.get_int("launches");
    const bool want_logit = arg.get_int("logits") != 0;
    const int seqs        = p.M / p.window;
    const int off         = p.vocab_offset();

    Data d = make_data(p, static_cast<uint64_t>(arg.get_int("seed")));

    // Drafts: follow the reference argmax (on the CPU xn) for a random prefix, then diverge.
    {
        const auto ref = head_ref::logits(d.xn_ref, d.wt, all_rows(p.M), p.N, p.H);
        std::vector<int32_t> ref_tok(p.M);
        for(int r = 0; r < p.M; ++r)
            ref_tok[r] = off + head_ref::check_argmax<nm>(ref[r], 0, mult, cap).ref_token;
        std::mt19937_64 rng(77);
        d.drafts.assign(size_t(seqs) * (p.window - 1), 0);
        for(int s = 0; s < seqs; ++s)
        {
            const int a = static_cast<int>(rng() % static_cast<uint64_t>(p.window));
            for(int i = 0; i < p.window - 1; ++i)
            {
                const int32_t t = ref_tok[s * p.window + i];
                d.drafts[size_t(s) * (p.window - 1) + i] =
                    i < a ? t
                          : (i == a ? (t + 1) % (p.world * p.N)
                                    : static_cast<int32_t>(rng() % uint64_t(p.world * p.N)));
            }
        }
    }

    DeviceBuffer d_x(d.x.size() * 2), d_w(d.w.size() * 2), d_wt(d.wt.size() * 2);
    DeviceBuffer d_xn(d.x.size() * 2), d_keys(size_t(p.M) * 8), d_keys_out(size_t(p.M) * 8);
    DeviceBuffer d_logits(want_logit ? size_t(p.M) * p.N * 4 : 0);
    DeviceBuffer d_tokens(size_t(p.M) * 4), d_maxl(size_t(p.M) * 4);
    DeviceBuffer d_drafts(d.drafts.size() * 4), d_acc(size_t(p.M) * 4), d_nacc(size_t(seqs) * 4);
    to_device(d_x, d.x);
    to_device(d_w, d.w);
    to_device(d_wt, d.wt);
    to_device(d_drafts, d.drafts);
    // Zeroed once; the accept stage clears them after every launch.
    HIP_CHECK(hipMemset(d_keys.p, 0, size_t(p.M) * 8));

    const typename Norm::Args norm{d_x.as<ck_tile::bf16_t>(),
                                   d_w.as<ck_tile::bf16_t>(),
                                   d_xn.as<ck_tile::bf16_t>(),
                                   p.M,
                                   p.H,
                                   p.H,
                                   p.H,
                                   ck_tile::MkHeadConstants::kNormEps};
    const auto gemm = HeadT::GemmKernel::MakeKernelArgs(
        ck_tile::GemmHostArgs(d_xn.p, d_wt.p, nullptr, 1, p.M, p.N, p.H, p.H, p.H, p.N));
    const typename Logits::Args logits{gemm,
                                       d_keys.as<uint64_t>(),
                                       want_logit ? d_logits.as<float>() : nullptr,
                                       p.N,
                                       off,
                                       mult,
                                       cap};
    const typename Accept::Args accept{d_keys.as<uint64_t>(),
                                       p.M,
                                       p.window,
                                       true,
                                       d_drafts.as<int32_t>(),
                                       d_tokens.as<int32_t>(),
                                       d_maxl.as<float>(),
                                       d_keys_out.as<uint64_t>(),
                                       d_acc.as<int32_t>(),
                                       d_nacc.as<int32_t>()};

    const typename MK::ArgsTuple args{norm, logits, accept};
    const index_t n_counters = MK::NumCounters(args);
    DeviceBuffer d_counters(size_t(n_counters) * sizeof(uint32_t));
    HIP_CHECK(hipMemset(d_counters.p, 0, size_t(n_counters) * sizeof(uint32_t)));
    const auto kargs = MK::MakeKargs(args, d_counters.as<uint32_t>());
    if(!MK::IsSupportedArgument(kargs))
    {
        std::printf("megakernel arguments not supported\n");
        return 2;
    }

    const ck_tile::stream_config s{nullptr, false};
    const index_t resident = MK::MaxResidentGridSize(s);
    index_t grid           = arg.get_int("grid");
    grid = grid > 0 ? std::min(grid, resident) : std::min(Logits::NumWork(logits), resident);
    std::printf("M %d H %d N %d (rank %d of %d, tokens [%d, %d)), window %d, %s, tile %dx%d: %d "
                "tiles, grid %d (resident %d), LDS %d B, %d planted rows\n",
                p.M,
                p.H,
                p.N,
                p.rank,
                p.world,
                off,
                off + p.N,
                p.window,
                nm == Numerics::hf_bf16 ? "hf_bf16" : "fp32",
                Logits::kMPerBlock,
                Logits::kNPerBlock,
                Logits::NumWork(logits),
                grid,
                resident,
                static_cast<int>(MK::kSmemSize),
                d.planted);

    std::vector<int32_t> tokens(p.M), acc(size_t(p.M)), nacc(seqs), tokens0, acc0, nacc0;
    std::vector<float> maxl(p.M);
    std::vector<uint64_t> keys_out(p.M), keys(p.M);
    int failures = 0;
    for(int l = 0; l < launches; ++l)
    {
        HIP_CHECK(hipMemset(d_tokens.p, 0xFF, size_t(p.M) * 4));
        HIP_CHECK(hipMemset(d_acc.p, 0x7F, size_t(p.M) * 4));
        HIP_CHECK(hipMemset(d_nacc.p, 0xFF, size_t(seqs) * 4));
        ck_tile::launch_kernel(
            s, ck_tile::make_kernel<1>(MK{}, dim3(grid), MK::BlockSize(), 0, kargs));
        HIP_CHECK(hipDeviceSynchronize());
        to_host(tokens, d_tokens);
        to_host(acc, d_acc);
        to_host(nacc, d_nacc);
        to_host(keys, d_keys);
        if(std::any_of(keys.begin(), keys.end(), [](uint64_t k) { return k != 0; }))
        {
            ++failures;
            std::printf("launch %d: keys not cleared\n", l);
        }
        if(l == 0)
        {
            tokens0 = tokens;
            acc0    = acc;
            nacc0   = nacc;
        }
        else if(tokens != tokens0 || acc != acc0 || nacc != nacc0)
        {
            ++failures;
            std::printf("launch %d: outputs differ from launch 0\n", l);
        }
    }
    to_host(maxl, d_maxl);
    to_host(keys_out, d_keys_out);

    // 1. xn.
    std::vector<uint16_t> xn(d.x.size());
    to_host(xn, d_xn);
    int bad_xn = 0;
    for(size_t i = 0; i < xn.size(); ++i)
    {
        const int dist = std::abs(int(int16_t(xn[i])) - int(int16_t(d.xn_ref[i])));
        if(dist > 1)
        {
            if(bad_xn < 5)
                std::printf("  xn[%zu] = %g, ref %g\n",
                            i,
                            head_ref::bf16_to_float(xn[i]),
                            head_ref::bf16_to_float(d.xn_ref[i]));
            ++bad_xn;
        }
    }

    // 2. Tokens against the reference logits of the GPU's xn.
    const auto ref = head_ref::logits(xn, d.wt, all_rows(p.M), p.N, p.H);
    int bad_tok = 0, ties = 0;
    for(int r = 0; r < p.M; ++r)
    {
        const int t  = tokens[r] - off;
        const auto c = head_ref::check_argmax<nm>(ref[r], t, mult, cap);
        ties += c.exact_ties > 0;
        bool ok = c.ok;
        if(ok)
        {
            const head_ref::CappedBounds<nm> b(
                ref[r].acc[t], head_ref::fp32_dot_err(ref[r].mag[t]), mult, cap);
            ok = maxl[r] >= b.lo && maxl[r] <= b.hi &&
                 keys_out[r] == ck_tile::head_argmax_key(maxl[r], tokens[r]);
        }
        if(!ok)
        {
            if(bad_tok < 5)
                std::printf("  row %d: token %d (logit %g), reference token %d\n",
                            r,
                            tokens[r],
                            static_cast<double>(maxl[r]),
                            off + c.ref_token);
            ++bad_tok;
        }
    }

    // 3. Optional logits.
    int bad_logits = 0;
    if(want_logit)
    {
        std::vector<float> lg(size_t(p.M) * p.N);
        to_host(lg, d_logits);
        for(int r = 0; r < p.M; ++r)
            for(int n = 0; n < p.N; ++n)
            {
                const head_ref::CappedBounds<nm> b(
                    ref[r].acc[n], head_ref::fp32_dot_err(ref[r].mag[n]), mult, cap);
                const float g = lg[size_t(r) * p.N + n];
                if(!(g >= b.lo && g <= b.hi))
                {
                    if(bad_logits < 5)
                        std::printf(
                            "  logits[%d, %d] = %g, bounds [%g, %g]\n", r, n, g, b.lo, b.hi);
                    ++bad_logits;
                }
            }
    }

    // 4. Accept.
    const auto acc_ref   = head_ref::accept(tokens, d.drafts, p.window);
    const int bad_accept = (acc != acc_ref.tokens) + (nacc != acc_ref.count);
    int emitted          = 0;
    for(int v : acc_ref.count)
        emitted += v;

    const bool pass =
        failures == 0 && bad_xn == 0 && bad_tok == 0 && bad_logits == 0 && bad_accept == 0;
    std::printf(
        "%s: %d/%d launches consistent, xn bad %d, tokens bad %d (%d rows with exact ties), "
        "logits bad %d, accept %s (%d tokens over %d windows)\n",
        pass ? "PASS" : "FAIL",
        launches - failures,
        launches,
        bad_xn,
        bad_tok,
        ties,
        bad_logits,
        bad_accept ? "WRONG" : "ok",
        emitted,
        seqs);
    return pass ? 0 : 1;
}

template <typename Gemm>
int dispatch_numerics(const ck_tile::ArgParser& arg, const Problem& p)
{
    const std::string n = arg.get_str("numerics");
    if(n == "hf_bf16")
        return run<Head<Gemm, Numerics::hf_bf16>>(arg, p);
    if(n == "fp32")
        return run<Head<Gemm, Numerics::fp32>>(arg, p);
    std::printf("unknown -numerics=%s\n", n.c_str());
    return 2;
}

} // namespace

int main(int argc, char* argv[])
{
    ck_tile::ArgParser arg;
    arg.insert("config", "small", "small (H 1024, vocab 6000) | full (H 8192, vocab 202048 TP8)")
        .insert("m", "16", "rows: window * sequences, <= 64")
        .insert("window", "16", "DFlash2 verify window (rows per sequence)")
        .insert("hidden", "0", "override H")
        .insert("vocab", "0", "override the total vocab (a multiple of -world)")
        .insert("world", "0", "override the TP world size (vocab shards)")
        .insert("rank", "-1", "emulated TP rank (default world - 1)")
        .insert("numerics", "hf_bf16", "softcap numerics: hf_bf16 | fp32")
        .insert("logits", "0", "also write the fp32 softcapped logits")
        .insert("grid", "0", "workgroups; 0 = min(tiles, resident)")
        .insert("launches", "3", "launches sharing one counter and key buffer")
        .insert("seed", "1", "data seed");
    if(!arg.parse(argc, argv))
        return 2;
    try
    {
        const std::string cfg = arg.get_str("config");
        Problem p{};
        int vocab = 0;
        if(cfg == "small")
        {
            p.H     = 1024;
            vocab   = 6000;
            p.world = 1;
        }
        else if(cfg == "full")
        {
            p.H     = 8192;
            vocab   = ck_tile::MkHeadConstants::kVocab;
            p.world = 8;
        }
        else
        {
            std::printf("unknown -config=%s\n", cfg.c_str());
            return 2;
        }
        p.H      = arg.get_int("hidden") > 0 ? arg.get_int("hidden") : p.H;
        vocab    = arg.get_int("vocab") > 0 ? arg.get_int("vocab") : vocab;
        p.world  = arg.get_int("world") > 0 ? arg.get_int("world") : p.world;
        p.M      = arg.get_int("m");
        p.window = arg.get_int("window");
        p.rank   = arg.get_int("rank") >= 0 ? arg.get_int("rank") : p.world - 1;
        if(vocab % p.world != 0 || p.rank >= p.world || p.M <= 0 || p.M > 64 || p.window <= 0 ||
           p.M % p.window != 0 || p.H % 8 != 0)
        {
            std::printf("bad problem: vocab %d world %d rank %d M %d window %d H %d\n",
                        vocab,
                        p.world,
                        p.rank,
                        p.M,
                        p.window,
                        p.H);
            return 2;
        }
        p.N = vocab / p.world;
        return p.M <= 16 ? dispatch_numerics<Gemm16>(arg, p) : dispatch_numerics<Gemm64>(arg, p);
    }
    catch(const std::exception& e)
    {
        std::printf("error: %s\n", e.what());
        return 3;
    }
}
