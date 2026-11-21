// Prototype: gfx1250 fp8 decode attention over sglang's SHUFFLE 5D paged KV cache.
//   K: [num_blocks][hkv][D/16][16 tok][16 B]   V: [num_blocks][hkv][1][D][16 tok]   (page = 16)
// One query token, GQA rows (<= 16) of one kv head form the 16 columns of S^T = K Q^T.
// Grid (num_splits, hkv, batch), 4 waves per CTA, workgroup cluster over the splits.
// Each CTA walks its page range in 64-token steps (one page per wave), 3-stage async ring.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v16i __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

constexpr int D      = 128;
constexpr int PAGE   = 16;
constexpr int NW     = 4;          // waves; one page each per step
constexpr int STEP   = NW * PAGE;  // tokens per step
constexpr int PAGEB  = PAGE * D;   // bytes of one page of one head (fp8)
constexpr int NST    = 3;          // ring stages
constexpr int ROWS   = 16;
constexpr float LOG2E = 1.4426950408889634f;

#define G1(T, p) ((__attribute__((address_space(1))) T*)(p))
#define L3(T, p) ((__attribute__((address_space(3))) T*)(p))

struct Args
{
    const unsigned short* q;     // [B][HQ][D] bf16
    const unsigned char* k;      // 5D
    const unsigned char* v;      // 5D
    const int* block_table;      // [B][max_blocks]
    const int* ctx_lens;         // [B]
    const float* k_descale;
    const float* v_descale;
    unsigned short* out;         // [B][HQ][D] bf16
    unsigned short* ws_o;        // [B][HKV][NS][ROWS][D] bf16
    float* ws_ml;                // [B][HKV][NS][ROWS][2]
    int hq, hkv, max_blocks, window, ns, pages_per_split;
    float sm_scale;
    unsigned long long* dbg;
};
#define STAMP(i)                                                                             \
    do                                                                                       \
    {                                                                                        \
        if(a.dbg && threadIdx.x == 0)                                                        \
            a.dbg[((size_t)blockIdx.z * gridDim.x + blockIdx.x) * 16 + (i)] = wall_clock64(); \
    } while(0)

__device__ inline float bf2f(unsigned short x) { return __uint_as_float((unsigned)x << 16); }
__device__ inline unsigned short f2bf(float f)
{
    unsigned u = __float_as_uint(f);
    return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

struct alignas(16) Smem
{
    alignas(16) unsigned char k[NST][NW][PAGEB];
    alignas(16) unsigned char v[NST][NW][PAGEB];
    alignas(16) unsigned char p[ROWS][STEP];
    alignas(16) float red[2][NW][ROWS]; // per-wave row max / row sum
};

// NS_ = splits = cluster size (compile time, so the workgroup ids need no IB_STS2 read)
template <int NS_>
__global__ __launch_bounds__(NW * 32) __attribute__((cluster_dims(NS_, 1, 1))) void decode5d(Args a)
{
    __shared__ alignas(16) Smem sm;
#ifdef HWSD_PREFETCH
    // the entry sequence newer gfx1250 compilers emit in every kernel
    asm volatile("s_mov_b64 s[64:65], 0\n\tv_nop\n\tglobal_prefetch_b8 v0, s[64:65] scope:SCOPE_SE" :::
                     "s64", "s65", "memory");
#endif
    STAMP(0);
    const int l = threadIdx.x % 32, w = __builtin_amdgcn_readfirstlane(threadIdx.x) / 32;
    const int n = l % 16, h = l / 16;
    const int s = blockIdx.x, g = blockIdx.y, b = blockIdx.z;
    const int gqa = a.hq / a.hkv;

    const int ctx  = a.ctx_lens[b];
    const int lo   = a.window > 0 ? max(0, ctx - a.window) : 0;
    const int pg0  = lo / PAGE + s * a.pages_per_split; // first page of this split
    const int pend = min((ctx + PAGE - 1) / PAGE, lo / PAGE + (s + 1) * a.pages_per_split);
    const int steps = pend > pg0 ? (pend - pg0 + NW - 1) / NW : 0;
    const int* bt = a.block_table + (size_t)b * a.max_blocks;
    // block ids of this split's pages, 32 per register (lane = page); readlane picks one
    auto load_bt = [&](int chunk) {
        const int pg = min(pg0 + chunk * 32 + l, max(pend - 1, 0));
        return bt[pg];
    };
    int bt_cur = load_bt(0), bt_next = load_bt(1);
    int cur_chunk = 0;

    // Q^T operand, row n: d chunks 2i+h (16 B fp8 each) after a per-row absmax quantization
    float qv[64];
    {
        const unsigned short* qrow = a.q + ((size_t)b * a.hq + g * gqa + min(n, gqa - 1)) * D;
        for(int i = 0; i < 4; ++i)
        {
            const int c = 2 * i + h;
            const v4i x0 = *reinterpret_cast<const v4i*>(qrow + c * 16);
            const v4i x1 = *reinterpret_cast<const v4i*>(qrow + c * 16 + 8);
            for(int e = 0; e < 4; ++e)
            {
                qv[i * 16 + 2 * e]         = bf2f(x0[e] & 0xffff);
                qv[i * 16 + 2 * e + 1]     = bf2f((unsigned)x0[e] >> 16);
                qv[i * 16 + 8 + 2 * e]     = bf2f(x1[e] & 0xffff);
                qv[i * 16 + 8 + 2 * e + 1] = bf2f((unsigned)x1[e] >> 16);
            }
        }
    }

    // issue the async copies of step t into stage st: wave w takes page pg0 + t*NW + w
    auto issue = [&](int t, int st) {
        const int idx  = t * NW + w;
        const int blk  = __builtin_amdgcn_readlane((idx >> 5) == cur_chunk ? bt_cur : bt_next, idx & 31);
        const unsigned char* ks = a.k + ((size_t)blk * a.hkv + g) * PAGEB;
        const unsigned char* vs = a.v + ((size_t)blk * a.hkv + g) * PAGEB;
        for(int i = 0; i < 4; ++i)
        {
            const int off = (i * 32 + l) * 16;
            __builtin_amdgcn_global_load_async_to_lds_b128(
                G1(v4i, ks + off), L3(v4i, &sm.k[st][w][off]), 0, 0);
            __builtin_amdgcn_global_load_async_to_lds_b128(
                G1(v4i, vs + off), L3(v4i, &sm.v[st][w][off]), 0, 0);
        }
    };
    for(int t = 0; t < min(steps, NST); ++t)
        issue(t, t);

    float qmax = 1e-20f;
    for(int i = 0; i < 64; ++i)
        qmax = fmaxf(qmax, fabsf(qv[i]));
    qmax = fmaxf(qmax, __shfl_xor(qmax, 16, 32));
    const float qinv = 448.f / qmax;
    v16i qf;
    for(int j = 0; j < 16; ++j)
    {
        int x = __builtin_amdgcn_cvt_pk_fp8_f32(qv[4 * j] * qinv, qv[4 * j + 1] * qinv, 0, false);
        qf[j] = __builtin_amdgcn_cvt_pk_fp8_f32(qv[4 * j + 2] * qinv, qv[4 * j + 3] * qinv, x, true);
    }
    // column n of S^T in log2 units
    const float scl = qmax / 448.f * *a.k_descale * a.sm_scale * LOG2E;

    // step t's data is complete in LDS for this wave: zero the V columns of tokens past the
    // context (arbitrary bytes there would turn P = 0 into NaN)
    auto land = [&](int t, int issued) {
        if(issued > t + 2)
            __builtin_amdgcn_s_wait_asynccnt(16);
        else if(issued > t + 1)
            __builtin_amdgcn_s_wait_asynccnt(8);
        else
            __builtin_amdgcn_s_wait_asynccnt(0);
        const int pg    = pg0 + t * NW + w;
        const int valid = __builtin_amdgcn_readfirstlane(min(max(ctx - pg * PAGE, 0), PAGE));
        if(valid < PAGE)
            for(int i = 0; i < 4; ++i)
            {
                unsigned char* row = &sm.v[t % NST][w][(i * 32 + l) * 16];
                for(int tok = valid; tok < PAGE; ++tok)
                    row[tok] = 0;
            }
    };

    STAMP(1);
    float m_run = -INFINITY, l_run = 0.f;
    v8f acc[2] = {v8f{0, 0, 0, 0, 0, 0, 0, 0}, v8f{0, 0, 0, 0, 0, 0, 0, 0}};
    int issued = min(steps, NST);
    if(steps > 0)
        land(0, issued);
    __syncthreads();

    // per step: QK -> barrier (row max) -> P -> barrier (P, row sums, next step's data) -> PV
    for(int t = 0; t < steps; ++t)
    {
        const int st = t % NST;
        if(t < 4) STAMP(2 + t);

        // S^T for this wave's page: A = K (row = token n, d chunks 2i+h)
        v16i kf;
        {
            const unsigned char* kb = &sm.k[st][w][0];
            for(int i = 0; i < 4; ++i)
            {
                const v4i x = *reinterpret_cast<const v4i*>(kb + (2 * i + h) * 256 + n * 16);
                kf[4 * i] = x[0]; kf[4 * i + 1] = x[1]; kf[4 * i + 2] = x[2]; kf[4 * i + 3] = x[3];
            }
        }
        v8f st8 = __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(kf, qf, 0, v8f{0, 0, 0, 0, 0, 0, 0, 0},
                                                              false, false);
        const int tok0 = (pg0 + t * NW + w) * PAGE + h * 8;
        float sv[8];
        float mx = -INFINITY;
        for(int r = 0; r < 8; ++r)
        {
            const int pos = tok0 + r;
            const bool ok = pos >= lo && pos < ctx && pos < pend * PAGE;
            sv[r]         = ok ? st8[r] * scl : -INFINITY;
            mx            = fmaxf(mx, sv[r]);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        if(h == 0)
            sm.red[0][w][n] = mx;
        __syncthreads();
        // every wave is past step t-1's PV: its stage takes step t+2
        if(t > 0 && (t * NW) % 32 == 0)
        {
            ++cur_chunk;
            bt_cur  = bt_next;
            bt_next = load_bt(cur_chunk + 1);
        }
        if(t >= 1 && t + 2 < steps)
        {
            issue(t + 2, (t + 2) % NST);
            ++issued;
        }
        float m_step = sm.red[0][0][n];
        for(int i = 1; i < NW; ++i)
            m_step = fmaxf(m_step, sm.red[0][i][n]);
        const float m_new = fmaxf(m_run, m_step);
        const float m_use = m_new == -INFINITY ? 0.f : m_new;
        const float alpha = __builtin_amdgcn_exp2f(m_run - m_use);
        // P scaled by 448: the row max lands on e4m3's largest normal
        float ps = 0.f;
        float pv[8];
        for(int r = 0; r < 8; ++r)
        {
            pv[r] = __builtin_amdgcn_exp2f(sv[r] - m_use) * 448.f;
            ps += pv[r];
        }
        ps = ps + __shfl_xor(ps, 16, 32);
        int p0 = __builtin_amdgcn_cvt_pk_fp8_f32(pv[0], pv[1], 0, false);
        p0     = __builtin_amdgcn_cvt_pk_fp8_f32(pv[2], pv[3], p0, true);
        int p1 = __builtin_amdgcn_cvt_pk_fp8_f32(pv[4], pv[5], 0, false);
        p1     = __builtin_amdgcn_cvt_pk_fp8_f32(pv[6], pv[7], p1, true);
        *reinterpret_cast<int2*>(&sm.p[n][w * PAGE + h * 8]) = make_int2(p0, p1);
        if(h == 0)
            sm.red[1][w][n] = ps;
        if(t + 1 < steps)
            land(t + 1, issued);
        __syncthreads();
        float psum = 0.f;
        for(int i = 0; i < NW; ++i)
            psum += sm.red[1][i][n];
        l_run = l_run * alpha + psum;
        m_run = m_new;

        // O^T += V^T P^T for d blocks w and w + 4; tokens of pages h and h + 2 on both operands
        v8i pf;
        {
            const v4i x = *reinterpret_cast<const v4i*>(&sm.p[n][h * PAGE]);
            const v4i y = *reinterpret_cast<const v4i*>(&sm.p[n][(h + 2) * PAGE]);
            pf = v8i{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
        }
        for(int bk = 0; bk < 2; ++bk)
        {
            const int dd = (bk * NW + w) * 16 + n;
            const v4i x  = *reinterpret_cast<const v4i*>(&sm.v[st][h][dd * 16]);
            const v4i y  = *reinterpret_cast<const v4i*>(&sm.v[st][h + 2][dd * 16]);
            const v8i vf{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
            acc[bk] = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(vf, pf, 0, acc[bk] * alpha, false,
                                                                 false);
        }
    }

    STAMP(6);
    // lane (n, h) holds O[row n][d = (bk*NW + w)*16 + h*8 + r]
    const float vd   = *a.v_descale;
    const bool rowok = n < gqa;
    if constexpr(NS_ == 1)
    {
        const float f = l_run > 0.f ? vd / l_run : 0.f;
        if(rowok)
            for(int bk = 0; bk < 2; ++bk)
            {
                int o[4];
                for(int e = 0; e < 4; ++e)
                    o[e] = f2bf(acc[bk][2 * e] * f) | ((int)f2bf(acc[bk][2 * e + 1] * f) << 16);
                unsigned short* dst = a.out + ((size_t)b * a.hq + g * gqa + n) * D +
                                      (bk * NW + w) * 16 + h * 8;
                *reinterpret_cast<v4i*>(dst) = v4i{o[0], o[1], o[2], o[3]};
            }
        return;
    }

    // partial: O / l in bf16, (m, l) in fp32
    const size_t part = (((size_t)b * a.hkv + g) * NS_ + s) * ROWS;
    {
        const float f = l_run > 0.f ? 1.f / l_run : 0.f;
        for(int bk = 0; bk < 2; ++bk)
        {
            int o[4];
            for(int e = 0; e < 4; ++e)
                o[e] = f2bf(acc[bk][2 * e] * f) | ((int)f2bf(acc[bk][2 * e + 1] * f) << 16);
            *reinterpret_cast<v4i*>(a.ws_o + (part + n) * D + (bk * NW + w) * 16 + h * 8) =
                v4i{o[0], o[1], o[2], o[3]};
        }
        if(w == 0 && h == 0)
            *reinterpret_cast<float2*>(a.ws_ml + (part + n) * 2) = make_float2(m_run, l_run);
    }
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    STAMP(7);
    __builtin_amdgcn_s_cluster_barrier();
    asm volatile("global_inv scope:SCOPE_SE" ::: "memory");
    STAMP(8);

    // merge: CTA s takes rows [s*rp, s*rp + rp); thread -> (row, 2 consecutive d)
    constexpr int rp = (ROWS + NS_ - 1) / NS_;
    for(int rr = threadIdx.x / 64; rr < rp; rr += NW * 32 / 64)
    {
        const int row = s * rp + rr;
        if(row >= gqa)
            break;
        const int d2      = (threadIdx.x % 64) * 2;
        const size_t base = ((size_t)b * a.hkv + g) * NS_ * ROWS;
        float2 ml[NS_];
        unsigned oi[NS_];
        for(int i = 0; i < NS_; ++i)
        {
            ml[i] = *reinterpret_cast<const float2*>(a.ws_ml + (base + i * ROWS + row) * 2);
            oi[i] = *reinterpret_cast<const unsigned*>(a.ws_o + (base + i * ROWS + row) * D + d2);
        }
        float M = -INFINITY;
        for(int i = 0; i < NS_; ++i)
            M = fmaxf(M, ml[i].x);
        float L = 0.f, o0 = 0.f, o1 = 0.f;
        for(int i = 0; i < NS_; ++i)
        {
            const float wgt = ml[i].y > 0.f ? ml[i].y * __builtin_amdgcn_exp2f(ml[i].x - M) : 0.f;
            L += wgt;
            o0 += wgt * bf2f(oi[i] & 0xffff);
            o1 += wgt * bf2f(oi[i] >> 16);
        }
        const float f = L > 0.f ? vd / L : 0.f;
        *reinterpret_cast<unsigned*>(a.out + ((size_t)b * a.hq + g * gqa + row) * D + d2) =
            f2bf(o0 * f) | ((unsigned)f2bf(o1 * f) << 16);
    }
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
    STAMP(9);
}

// ---------------- host ----------------
static float fp8_to_f(unsigned char x)
{
    const int sgn = x >> 7, e = (x >> 3) & 15, m = x & 7;
    float v = e == 0 ? ldexpf(m / 8.f, -6) : ldexpf(1.f + m / 8.f, e - 7);
    return sgn ? -v : v;
}
static float bf16_to_f(unsigned short x)
{
    unsigned u = (unsigned)x << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static unsigned short f_to_bf16(float f)
{
    unsigned u;
    memcpy(&u, &f, 4);
    return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

int main(int argc, char** argv)
{
    const int B      = argc > 1 ? atoi(argv[1]) : 1;
    const int CTX    = argc > 2 ? atoi(argv[2]) : 1024;
    const int WIN    = argc > 3 ? atoi(argv[3]) : 2048;
    const int PPS_in = argc > 4 ? atoi(argv[4]) : 16; // pages per split
    const bool check = argc > 5 ? atoi(argv[5]) : 1;
    const int HQ = 16, HKV = 1;

    const int lo_pages = WIN > 0 ? std::max(0, CTX - WIN) / PAGE : 0;
    const int span     = (CTX + PAGE - 1) / PAGE - lo_pages;
    int PPS = PPS_in;
    if(PPS <= 0)
    {
        // splits: at most 15 (cluster_dims holds 4 bits), about 288 CTAs in total, >= 1 step each
        const int target = std::max(1, std::min({15, std::max(1, 288 / (B * HKV)), (span + 3) / 4}));
        PPS              = ((span + target - 1) / target + 3) / 4 * 4;
    }
    while((span + PPS - 1) / PPS > 15)
        PPS *= 2;
    const int NS = (span + PPS - 1) / PPS;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int pps = (CTX + PAGE - 1) / PAGE, max_blocks = pps, NB = B * pps + 1;
    std::vector<unsigned short> hq((size_t)B * HQ * D);
    for(auto& x : hq) x = f_to_bf16(2.f * U(rng));
    // fp8 bytes drawn from finite e4m3 values in [-1, 1]
    std::vector<unsigned char> fin;
    for(int i = 0; i < 256; ++i)
        if((i & 0x7f) != 0x7f && fabsf(fp8_to_f(i)) <= 1.f)
            fin.push_back(i);
    std::uniform_int_distribution<int> Ui(0, fin.size() - 1);
    std::vector<unsigned char> hk((size_t)NB * HKV * PAGEB), hv((size_t)NB * HKV * PAGEB);
    for(auto& x : hk) x = fin[Ui(rng)];
    for(auto& x : hv) x = fin[Ui(rng)];
    // poison: bytes of the last page past the context are NaN codes
    std::vector<int> perm(NB - 1);
    std::iota(perm.begin(), perm.end(), 1);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int> bt((size_t)B * max_blocks), cl(B, CTX);
    for(int bb = 0; bb < B; ++bb)
        for(int p = 0; p < pps; ++p)
            bt[(size_t)bb * max_blocks + p] = perm[bb * pps + p];
    for(int bb = 0; bb < B; ++bb)
    {
        const int last = bt[(size_t)bb * max_blocks + pps - 1];
        for(int tok = CTX - (pps - 1) * PAGE; tok < PAGE; ++tok)
            for(int d = 0; d < D; ++d)
            {
                hk[(size_t)last * PAGEB + (d / 16) * 256 + tok * 16 + d % 16] = 0x7f;
                hv[(size_t)last * PAGEB + d * 16 + tok]                         = 0x7f;
            }
    }
    const float kd = 0.75f, vd = 1.25f, scale = 1.f / sqrtf((float)D);

    unsigned short *dq, *dout, *dws;
    unsigned char *dk, *dv;
    int *dbt, *dcl;
    float *dws_ml, *dsc;
    hipMalloc(&dq, hq.size() * 2);
    hipMalloc(&dk, hk.size());
    hipMalloc(&dv, hv.size());
    hipMalloc(&dbt, bt.size() * 4);
    hipMalloc(&dcl, cl.size() * 4);
    hipMalloc(&dout, (size_t)B * HQ * D * 2);
    hipMalloc(&dws, (size_t)B * HKV * NS * ROWS * D * 2);
    hipMalloc(&dws_ml, (size_t)B * HKV * NS * ROWS * 8);
    hipMalloc(&dsc, 8);
    hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(dk, hk.data(), hk.size(), hipMemcpyHostToDevice);
    hipMemcpy(dv, hv.data(), hv.size(), hipMemcpyHostToDevice);
    hipMemcpy(dbt, bt.data(), bt.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dcl, cl.data(), cl.size() * 4, hipMemcpyHostToDevice);
    float hs[2] = {kd, vd};
    hipMemcpy(dsc, hs, 8, hipMemcpyHostToDevice);

    unsigned long long* ddbg;
    hipMalloc(&ddbg, (size_t)B * NS * 16 * 8);
    Args a{dq, dk, dv, dbt, dcl, dsc, dsc + 1, dout, dws, dws_ml, HQ, HKV, max_blocks, WIN, NS, PPS, scale, nullptr};
    hipStream_t stream;
    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    hipLaunchConfig_t cfg{};
    cfg.gridDim  = dim3(NS, HKV, B);
    cfg.blockDim = dim3(NW * 32);
    cfg.stream   = stream;
    hipLaunchAttribute at[1];
    at[0].id               = hipLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = NS;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs              = at;
    cfg.numAttrs           = NS > 1 ? 1 : 0;
    auto launch = [&] {
        hipError_t e;
        switch(NS)
        {
        case 1: e = hipLaunchKernelEx(&cfg, decode5d<1>, a); break;
        case 2: e = hipLaunchKernelEx(&cfg, decode5d<2>, a); break;
        case 3: e = hipLaunchKernelEx(&cfg, decode5d<3>, a); break;
        case 4: e = hipLaunchKernelEx(&cfg, decode5d<4>, a); break;
        case 5: e = hipLaunchKernelEx(&cfg, decode5d<5>, a); break;
        case 6: e = hipLaunchKernelEx(&cfg, decode5d<6>, a); break;
        case 7: e = hipLaunchKernelEx(&cfg, decode5d<7>, a); break;
        case 8: e = hipLaunchKernelEx(&cfg, decode5d<8>, a); break;
        case 9: e = hipLaunchKernelEx(&cfg, decode5d<9>, a); break;
        case 10: e = hipLaunchKernelEx(&cfg, decode5d<10>, a); break;
        case 11: e = hipLaunchKernelEx(&cfg, decode5d<11>, a); break;
        case 12: e = hipLaunchKernelEx(&cfg, decode5d<12>, a); break;
        case 13: e = hipLaunchKernelEx(&cfg, decode5d<13>, a); break;
        case 14: e = hipLaunchKernelEx(&cfg, decode5d<14>, a); break;
        case 15: e = hipLaunchKernelEx(&cfg, decode5d<15>, a); break;
        default: e = hipErrorInvalidValue; break;
        }
        if(e != hipSuccess) { printf("launch: %s\n", hipGetErrorString(e)); exit(1); }
    };
    launch();
    hipStreamSynchronize(stream);

    if(check)
    {
        std::vector<unsigned short> ho((size_t)B * HQ * D);
        hipMemcpy(ho.data(), dout, ho.size() * 2, hipMemcpyDeviceToHost);
        double maxerr = 0, maxref = 0, dot = 0, nr = 0, ng = 0;
        const int lo = WIN > 0 ? std::max(0, CTX - WIN) : 0;
        for(int bb = 0; bb < B; ++bb)
            for(int hh = 0; hh < HQ; ++hh)
            {
                const int gg = hh / (HQ / HKV);
                std::vector<double> sc(CTX - lo);
                double mx = -1e300;
                for(int tok = lo; tok < CTX; ++tok)
                {
                    const int blk = bt[(size_t)bb * max_blocks + tok / PAGE], tp = tok % PAGE;
                    double accd   = 0;
                    for(int d = 0; d < D; ++d)
                        accd += bf16_to_f(hq[((size_t)bb * HQ + hh) * D + d]) *
                                fp8_to_f(hk[((size_t)blk * HKV + gg) * PAGEB + (d / 16) * 256 + tp * 16 + d % 16]);
                    sc[tok - lo] = accd * kd * scale;
                    mx           = std::max(mx, sc[tok - lo]);
                }
                double L = 0;
                std::vector<double> o(D, 0.0);
                for(int tok = lo; tok < CTX; ++tok)
                {
                    const int blk = bt[(size_t)bb * max_blocks + tok / PAGE], tp = tok % PAGE;
                    const double p = exp(sc[tok - lo] - mx);
                    L += p;
                    for(int d = 0; d < D; ++d)
                        o[d] += p * fp8_to_f(hv[((size_t)blk * HKV + gg) * PAGEB + d * 16 + tp]);
                }
                for(int d = 0; d < D; ++d)
                {
                    const double ref = o[d] / L * vd;
                    const double got = bf16_to_f(ho[((size_t)bb * HQ + hh) * D + d]);
                    maxerr           = std::max(maxerr, fabs(ref - got));
                    maxref           = std::max(maxref, fabs(ref));
                    dot += ref * got; nr += ref * ref; ng += got * got;
                }
            }
        const double cosv = dot / sqrt(nr * ng + 1e-300);
        printf("check: max abs err %.4g (max |ref| %.4g) cos %.6f %s\n", maxerr, maxref, cosv,
               (cosv > 0.999 && std::isfinite(maxerr)) ? "OK" : "BAD");
    }

    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    for(int i = 0; i < 20; ++i)
        launch();
    hipStreamSynchronize(stream);
    const int R = 100;
    hipEventRecord(e0, stream);
    for(int i = 0; i < R; ++i)
        launch();
    hipEventRecord(e1, stream);
    hipEventSynchronize(e1);
    float ms_e;
    hipEventElapsedTime(&ms_e, e0, e1);
    hipGraph_t gr;
    hipGraphExec_t ge;
    hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
    for(int i = 0; i < R; ++i)
        launch();
    hipStreamEndCapture(stream, &gr);
    hipGraphInstantiate(&ge, gr, nullptr, nullptr, 0);
    hipGraphLaunch(ge, stream);
    hipStreamSynchronize(stream);
    float best = 1e30f;
    for(int rep = 0; rep < 5; ++rep)
    {
        hipEventRecord(e0, stream);
        hipGraphLaunch(ge, stream);
        hipEventRecord(e1, stream);
        hipEventSynchronize(e1);
        float ms;
        hipEventElapsedTime(&ms, e0, e1);
        best = std::min(best, ms);
    }
    printf("B=%d ctx=%d win=%d splits=%d (pages/split %d): event %.2f us, graph %.2f us\n", B, CTX,
           WIN, NS, PPS, ms_e * 1e3f / R, best * 1e3f / R);
    if(getenv("STAMPS"))
    {
        a.dbg = ddbg;
        for(int it = 0; it < 3; ++it) launch();
        hipStreamSynchronize(stream);
        std::vector<unsigned long long> hd((size_t)B * NS * 16);
        hipMemcpy(hd.data(), ddbg, hd.size() * 8, hipMemcpyDeviceToHost);
        const char* nm[] = {"entry->Q", "Q->step0", "s0->s1", "s1->s2", "s2->s3", "s3->end", "part st", "cl bar", "merge"};
        int idx[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        for(int sp = 0; sp < NS; ++sp)
        {
            const unsigned long long* t = &hd[(size_t)sp * 16];
            printf("  split %2d:", sp);
            for(int k = 0; k < 9; ++k)
                if(t[idx[k]] && t[idx[k + 1]] && t[idx[k + 1]] >= t[idx[k]])
                    printf(" %s %.2f", nm[k], (t[idx[k + 1]] - t[idx[k]]) / 100.0);
            printf("  total %.2f\n", (t[9] - t[0]) / 100.0);
        }
    }
    return 0;
}
