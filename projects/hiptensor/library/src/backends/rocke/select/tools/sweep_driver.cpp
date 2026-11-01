/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// Selection sweep driver: runs the problems of a spec file (gen_corpus.py) through
// hiptensorCreatePlan on the rocKE backend (public API only).
//
//   (default)   one HIPTENSOR_ALGO_DEFAULT_PATIENT plan per problem: the library times its
//               candidates (HIPTENSOR_ROCKE_AUTOTUNE_TOPK=all: every one) and appends the
//               timings and its features to HIPTENSOR_ROCKE_SELECT_DUMP;
//   --default   one HIPTENSOR_ALGO_DEFAULT plan per problem, its selection log printed (SEL
//               lines); with HIPTENSOR_ROCKE_SELECT_DUMP the library writes every candidate's
//               features without a time (a problem pool for the next sweep round, or the picks of
//               an evaluation);
//   --jit       plans with HIPTENSOR_JIT_MODE_DEFAULT (the contraction JIT's candidate);
//   --cold R    contractions: after the plan above, every candidate (KERNEL_RANK 0, 1, ...) is
//               planned and timed cold: before each of R timed calls a READ of a --flush-mb
//               buffer (default 1024) evicts the caches (a write flush would leave dirty lines
//               whose write-back lands in the timed call); the repetitions are interleaved over
//               the candidates in a new random order every round. One JSON line per candidate goes to
//               --cold-out: {"tag", "rank", "key", "us" (median), "us_min", "reps", "rel_dev",
//               "bad", "ws"}. Operands hold uniform random values in [-1, 1); every candidate's
//               output (D filled with NaN first) is compared with the output of a reference
//               candidate (the f64 route, else the first candidate): rel_dev = ||D - D_ref|| /
//               ||D_ref|| over the elements finite in D_ref, bad = elements finite in D_ref and
//               not in D. train_select.py takes the cold times and drops disagreeing candidates.
//
//   --tag-suffix S  the problems' tags (dump and cold lines) are "<id>S", e.g. "#none" for a
//               HIPTENSOR_JIT_MODE_NONE sweep of problems whose JIT-mode sweep has the plain ids.
//
// usage: sweep_driver SPECS [--start I] [--end J] [--deadline S] [--progress F] [--default]
//                     [--jit] [--cold R --cold-out FILE] [--flush-mb MB] [--tag-suffix S]
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    hiptensorDataType_t dtypeOf(std::string const& s)
    {
        if(s == "f16")
            return HIPTENSOR_R_16F;
        if(s == "bf16")
            return HIPTENSOR_R_16BF;
        if(s == "f32")
            return HIPTENSOR_R_32F;
        if(s == "f64")
            return HIPTENSOR_R_64F;
        if(s == "c32")
            return HIPTENSOR_C_32F;
        return HIPTENSOR_C_64F;
    }
    hiptensorComputeDescriptor_t computeOf(std::string const& s)
    {
        if(s == "16F")
            return HIPTENSOR_COMPUTE_DESC_16F;
        if(s == "16BF")
            return HIPTENSOR_COMPUTE_DESC_16BF;
        if(s == "64F")
            return HIPTENSOR_COMPUTE_DESC_64F;
        if(s == "C32F")
            return HIPTENSOR_COMPUTE_DESC_C32F;
        if(s == "C64F")
            return HIPTENSOR_COMPUTE_DESC_C64F;
        return HIPTENSOR_COMPUTE_DESC_32F;
    }
    hiptensorOperator_t opOf(std::string const& s)
    {
        if(s == "MAX")
            return HIPTENSOR_OP_MAX;
        if(s == "MIN")
            return HIPTENSOR_OP_MIN;
        if(s == "MUL")
            return HIPTENSOR_OP_MUL;
        return HIPTENSOR_OP_ADD;
    }

    // element kinds of the device helpers: 0 f16, 1 bf16, 2 f32, 3 f64 (complex: 2 reals)
    int kindOf(hiptensorDataType_t t)
    {
        switch(t)
        {
        case HIPTENSOR_R_16F:
            return 0;
        case HIPTENSOR_R_16BF:
            return 1;
        case HIPTENSOR_R_32F:
        case HIPTENSOR_C_32F:
            return 2;
        default:
            return 3;
        }
    }
    int realsOf(hiptensorDataType_t t)
    {
        return t == HIPTENSOR_C_32F || t == HIPTENSOR_C_64F ? 2 : 1;
    }
    size_t realBytes(int kind)
    {
        return kind < 2 ? 2 : kind == 2 ? 4 : 8;
    }

    __device__ double loadReal(void const* p, size_t i, int kind)
    {
        switch(kind)
        {
        case 0:
            return double(__half2float(static_cast<__half const*>(p)[i]));
        case 1:
        {
            uint32_t const bits = uint32_t(static_cast<uint16_t const*>(p)[i]) << 16;
            return double(__uint_as_float(bits));
        }
        case 2:
            return double(static_cast<float const*>(p)[i]);
        default:
            return static_cast<double const*>(p)[i];
        }
    }

    __device__ uint64_t mix(uint64_t z)
    {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // uniform [-1, 1) values (rounded to the element type)
    __global__ void fillRandom(void* p, size_t n, int kind, uint64_t seed)
    {
        for(size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
            i += size_t(gridDim.x) * blockDim.x)
        {
            double const v
                = double(mix(seed * 0x9E3779B97F4A7C15ull + i + 1) >> 11) * 0x1.0p-53 * 2.0 - 1.0;
            switch(kind)
            {
            case 0:
                static_cast<__half*>(p)[i] = __float2half(float(v));
                break;
            case 1:
            {
                // round to nearest even
                uint32_t const bits = __float_as_uint(float(v));
                static_cast<uint16_t*>(p)[i]
                    = uint16_t((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
                break;
            }
            case 2:
                static_cast<float*>(p)[i] = float(v);
                break;
            default:
                static_cast<double*>(p)[i] = v;
                break;
            }
        }
    }

    // sums of (x - r)^2 and r^2 over the elements finite in r, and the count of those not
    // finite in x
    __global__ void compare(
        void const* x, void const* r, size_t n, int kind, double* sums, unsigned long long* bad)
    {
        double             d2 = 0.0, r2 = 0.0;
        unsigned long long b = 0;
        for(size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
            i += size_t(gridDim.x) * blockDim.x)
        {
            double const rv = loadReal(r, i, kind);
            if(!isfinite(rv))
            {
                continue;
            }
            double const xv = loadReal(x, i, kind);
            if(!isfinite(xv))
            {
                b++;
                continue;
            }
            d2 += (xv - rv) * (xv - rv);
            r2 += rv * rv;
        }
        atomicAdd(&sums[0], d2);
        atomicAdd(&sums[1], r2);
        atomicAdd(bad, b);
    }

    // a read of the flush buffer; the value goes to `out` only so that the loads stay
    __global__ void flushRead(uint4 const* __restrict__ p, size_t n, uint32_t* out)
    {
        uint32_t acc = 0;
        for(size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
            i += size_t(gridDim.x) * blockDim.x)
        {
            uint4 const v = p[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        if(acc == 0x9E3779B9u)
        {
            out[0] = acc;
        }
    }

    struct T
    {
        std::vector<int32_t> modes;
        std::vector<int64_t> ext, str;

        // elements from the first to the last one addressed
        size_t span() const
        {
            size_t s = 1;
            for(size_t i = 0; i < ext.size(); i++)
            {
                s += size_t(std::max<int64_t>(ext[i] - 1, 0)) * size_t(str[i]);
            }
            return s;
        }
    };

    T readTensor(std::istringstream& in, int n)
    {
        T t;
        t.modes.resize(n);
        t.ext.resize(n);
        t.str.resize(n);
        for(auto& m : t.modes)
            in >> m;
        for(auto& e : t.ext)
            in >> e;
        for(auto& s : t.str)
            in >> s;
        return t;
    }

    hiptensorTensorDescriptor_t describe(hiptensorHandle_t h, T const& t, hiptensorDataType_t type)
    {
        hiptensorTensorDescriptor_t d = nullptr;
        if(hiptensorCreateTensorDescriptor(h, &d, t.ext.size(), t.ext.data(), t.str.data(), type, 0)
           != HIPTENSOR_STATUS_SUCCESS)
            return nullptr;
        return d;
    }

    std::string lastSelection;
    void        callback(int32_t, char const*, char const* msg)
    {
        if(std::strstr(msg, "rocKE selection:") != nullptr)
            lastSelection = msg;
    }

    std::string field(std::string const& log, char const* key)
    {
        auto q = log.find(key);
        if(q == std::string::npos)
            return {};
        q += std::strlen(key);
        auto e = log.find(", ", q);
        return log.substr(q, e == std::string::npos ? std::string::npos : e - q);
    }

    // 1 and 1/2 in the compute type (hipTensor reads the scalars as the compute type)
    struct Scalars
    {
        alignas(16) unsigned char alpha[16] = {}, beta[16] = {};
    };
    Scalars scalarsFor(hiptensorComputeDescriptor_t c)
    {
        Scalars s;
        auto    put = [&c](unsigned char* b, double v) {
            if(c == HIPTENSOR_COMPUTE_DESC_16F)
            {
                uint16_t const h = v == 1.0 ? 0x3C00u : 0x3800u;
                std::memcpy(b, &h, 2);
            }
            else if(c == HIPTENSOR_COMPUTE_DESC_16BF)
            {
                uint16_t const h = v == 1.0 ? 0x3F80u : 0x3F00u;
                std::memcpy(b, &h, 2);
            }
            else if(c == HIPTENSOR_COMPUTE_DESC_64F || c == HIPTENSOR_COMPUTE_DESC_C64F)
            {
                std::memcpy(b, &v, 8);
            }
            else
            {
                float const f = float(v);
                std::memcpy(b, &f, 4);
            }
        };
        put(s.alpha, 1.0);
        put(s.beta, 0.5);
        return s;
    }

#define CHECK_HIP(x)                                                         \
    do                                                                       \
    {                                                                        \
        if((x) != hipSuccess)                                                \
        {                                                                    \
            std::fprintf(stderr, "HIP error %s at line %d\n", #x, __LINE__); \
            std::exit(4);                                                    \
        }                                                                    \
    } while(0)

    struct Cold
    {
        int                 reps   = 0;
        size_t              flushB = size_t(1024) << 20;
        bool                jit    = false;
        FILE*               out    = nullptr;
        void*               flush  = nullptr;
        uint32_t*           sink   = nullptr;
        double*             sums   = nullptr;
        unsigned long long* bad    = nullptr;
    };

    // Plans every candidate of a contraction (KERNEL_RANK 0, 1, ...) and times it cold; see the
    // top of the file.
    void coldContraction(hiptensorHandle_t              h,
                         hiptensorOperationDescriptor_t desc,
                         hiptensorComputeDescriptor_t   compute,
                         hiptensorDataType_t            type,
                         T const&                       A,
                         T const&                       B,
                         T const&                       D,
                         bool                           bilinear,
                         std::string const&             tag,
                         Cold&                          cold)
    {
        // the problem's candidates and their keys (plans only)
        auto plan = [&](int rank, hiptensorPlan_t& p, uint64_t& ws) -> std::string {
            p                              = nullptr;
            hiptensorPlanPreference_t pref = nullptr;
            hiptensorCreatePlanPreference(h,
                                          &pref,
                                          HIPTENSOR_ALGO_DEFAULT,
                                          cold.jit ? HIPTENSOR_JIT_MODE_DEFAULT
                                                   : HIPTENSOR_JIT_MODE_NONE);
            hiptensorCacheMode_t none = HIPTENSOR_CACHE_MODE_NONE;
            hiptensorPlanPreferenceSetAttribute(
                h, pref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &none, sizeof(none));
            int32_t const k = rank;
            hiptensorPlanPreferenceSetAttribute(
                h, pref, HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK, &k, sizeof(k));
            ws = 0;
            std::string key;
            if(hiptensorEstimateWorkspaceSize(h, desc, pref, HIPTENSOR_WORKSPACE_MAX, &ws)
               == HIPTENSOR_STATUS_SUCCESS)
            {
                lastSelection.clear();
                if(hiptensorCreatePlan(h, &p, desc, pref, ws) == HIPTENSOR_STATUS_SUCCESS)
                {
                    key = field(lastSelection, "Pick: ");
                    hiptensorPlanGetAttribute(
                        h, p, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, &ws, sizeof(ws));
                }
                else
                {
                    p = nullptr;
                }
            }
            hiptensorDestroyPlanPreference(pref);
            return key;
        };
        struct Cand
        {
            int             rank = 0;
            std::string     key;
            hiptensorPlan_t plan = nullptr;
            uint64_t        ws   = 0;
        };
        std::vector<Cand>     cands;
        std::set<std::string> keys;
        for(int rank = 0; rank < 64; rank++)
        {
            Cand c;
            c.rank = rank;
            c.key  = plan(rank, c.plan, c.ws);
            if(c.plan == nullptr)
            {
                break; // past the candidates
            }
            if(c.key.empty() || !keys.insert(c.key).second)
            {
                // a candidate that could not be built is replaced by the next one (+NEXT)
                hiptensorDestroyPlan(c.plan);
                continue;
            }
            cands.push_back(c);
        }
        if(cands.empty())
        {
            return;
        }
        int const    kind  = kindOf(type);
        int const    reals = realsOf(type);
        size_t const ea = A.span() * reals, eb = B.span() * reals, ed = D.span() * reals;
        size_t const eb_   = realBytes(kind);
        uint64_t     wsMax = 0;
        for(auto const& c : cands)
            wsMax = std::max(wsMax, c.ws);
        size_t freeB = 0, totalB = 0;
        CHECK_HIP(hipMemGetInfo(&freeB, &totalB));
        size_t const need = (ea + eb + 3 * ed) * eb_ + wsMax;
        if(need > freeB / 2)
        {
            std::printf("COLD_SKIP %s needs %zu bytes\n", tag.c_str(), need);
            for(auto& c : cands)
                hiptensorDestroyPlan(c.plan);
            return;
        }
        void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dRef = nullptr,
             *dW = nullptr;
        CHECK_HIP(hipMalloc(&dA, ea * eb_));
        CHECK_HIP(hipMalloc(&dB, eb * eb_));
        CHECK_HIP(hipMalloc(&dD, ed * eb_));
        CHECK_HIP(hipMalloc(&dRef, ed * eb_));
        if(bilinear)
            CHECK_HIP(hipMalloc(&dC, ed * eb_));
        if(wsMax > 0)
            CHECK_HIP(hipMalloc(&dW, wsMax));
        uint64_t const seed = std::hash<std::string>{}(tag);
        fillRandom<<<1024, 256>>>(dA, ea, kind, seed * 3 + 1);
        fillRandom<<<1024, 256>>>(dB, eb, kind, seed * 3 + 2);
        if(bilinear)
            fillRandom<<<1024, 256>>>(dC, ed, kind, seed * 3 + 3);
        CHECK_HIP(hipDeviceSynchronize());
        auto const sc  = scalarsFor(compute);
        auto       run = [&](Cand const& c, void* out) {
            return hiptensorContract(
                h, c.plan, sc.alpha, dA, dB, sc.beta, bilinear ? dC : nullptr, out, dW, c.ws, 0);
        };
        // the reference output: the f64 route when the problem has it
        size_t refIdx = 0;
        for(size_t i = 0; i < cands.size(); i++)
        {
            if(cands[i].key.rfind("f64:", 0) == 0 || cands[i].key.rfind("gett64:", 0) == 0)
            {
                refIdx = i;
                break;
            }
        }
        CHECK_HIP(hipMemset(dRef, 0xFF, ed * eb_));
        bool const refOk = run(cands[refIdx], dRef) == HIPTENSOR_STATUS_SUCCESS;
        CHECK_HIP(hipDeviceSynchronize());
        hipEvent_t e0, e1;
        CHECK_HIP(hipEventCreate(&e0));
        CHECK_HIP(hipEventCreate(&e1));
        // 1. every candidate once on a NaN-filled D (an element not written stays NaN), compared
        // with the reference, then once more untimed (its time decides its repetitions)
        size_t const                    nc = cands.size();
        std::vector<hiptensorStatus_t>  status(nc);
        std::vector<double>             relDev(nc, -1.0), warmMs(nc, 0.0);
        std::vector<unsigned long long> nbad(nc, 0);
        for(size_t i = 0; i < nc; i++)
        {
            CHECK_HIP(hipMemset(dD, 0xFF, ed * eb_));
            status[i] = run(cands[i], dD);
            CHECK_HIP(hipDeviceSynchronize());
            if(status[i] != HIPTENSOR_STATUS_SUCCESS)
            {
                continue;
            }
            if(refOk)
            {
                CHECK_HIP(hipMemset(cold.sums, 0, 2 * sizeof(double)));
                CHECK_HIP(hipMemset(cold.bad, 0, sizeof(unsigned long long)));
                compare<<<1024, 256>>>(dD, dRef, ed, kind, cold.sums, cold.bad);
                double sums[2] = {};
                CHECK_HIP(hipMemcpy(sums, cold.sums, sizeof(sums), hipMemcpyDeviceToHost));
                CHECK_HIP(hipMemcpy(&nbad[i], cold.bad, sizeof(nbad[i]), hipMemcpyDeviceToHost));
                relDev[i] = sums[1] > 0.0 ? std::sqrt(sums[0] / sums[1]) : std::sqrt(sums[0]);
            }
            auto const t0 = std::chrono::steady_clock::now();
            run(cands[i], dD);
            CHECK_HIP(hipDeviceSynchronize());
            warmMs[i]
                = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                      .count();
        }
        // 2. the timed repetitions interleaved over the candidates, in a new order every round
        // (no candidate is always timed first after the problem's setup); a slow candidate
        // (its time is far from the best) gets fewer repetitions
        std::vector<std::vector<float>> times(nc);
        std::vector<size_t>             order(nc);
        std::iota(order.begin(), order.end(), size_t(0));
        std::mt19937_64 rng(std::hash<std::string>{}(tag) ^ 0x5DEECE66Dull);
        for(int r = 0; r < cold.reps; r++)
        {
            std::shuffle(order.begin(), order.end(), rng);
            for(size_t i : order)
            {
                if(status[i] != HIPTENSOR_STATUS_SUCCESS || (warmMs[i] > 200.0 && r >= 2))
                {
                    continue;
                }
                flushRead<<<4096, 256>>>(
                    static_cast<uint4 const*>(cold.flush), cold.flushB / 16, cold.sink);
                CHECK_HIP(hipDeviceSynchronize());
                CHECK_HIP(hipEventRecord(e0, 0));
                run(cands[i], dD);
                CHECK_HIP(hipEventRecord(e1, 0));
                CHECK_HIP(hipEventSynchronize(e1));
                float ms = 0.0f;
                CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
                times[i].push_back(ms);
            }
        }
        for(size_t i = 0; i < nc; i++)
        {
            auto const& c = cands[i];
            auto&       t = times[i];
            std::sort(t.begin(), t.end());
            double median = -1.0, best = -1.0;
            if(!t.empty())
            {
                size_t const m = t.size();
                median         = 1000.0
                         * (m % 2 ? double(t[m / 2])
                                  : 0.5 * (double(t[m / 2 - 1]) + double(t[m / 2])));
                best = 1000.0 * double(t[0]);
            }
            std::fprintf(
                cold.out,
                "{\"tag\":\"%s\",\"rank\":%d,\"key\":\"%s\",\"status\":\"%s\",\"us\":%.4f,"
                "\"us_min\":%.4f,\"reps\":%zu,\"rel_dev\":%.6g,\"bad\":%llu,\"ref\":\"%s\","
                "\"ws\":%llu}\n",
                tag.c_str(),
                c.rank,
                c.key.c_str(),
                hiptensorGetErrorString(status[i]),
                median,
                best,
                t.size(),
                relDev[i],
                nbad[i],
                cands[refIdx].key.c_str(),
                static_cast<unsigned long long>(c.ws));
        }
        std::fflush(cold.out);
        CHECK_HIP(hipEventDestroy(e0));
        CHECK_HIP(hipEventDestroy(e1));
        for(auto& c : cands)
            hiptensorDestroyPlan(c.plan);
        for(void* p : {dA, dB, dC, dD, dRef, dW})
            if(p != nullptr)
                CHECK_HIP(hipFree(p));
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::fprintf(stderr,
                     "usage: %s SPECS [--start I] [--end J] [--deadline S] [--progress F] "
                     "[--default] [--jit] [--cold R --cold-out FILE] [--flush-mb MB] "
                     "[--tag-suffix S]\n",
                     argv[0]);
        return 2;
    }
    std::string specs = argv[1], progress, coldOut, tagSuffix;
    long        start = 0, end = 1L << 40;
    double      deadline   = 1e9;
    bool        defaultRun = false;
    Cold        cold;
    for(int i = 2; i < argc; i++)
    {
        std::string a = argv[i];
        if(a == "--start")
            start = std::atol(argv[++i]);
        else if(a == "--end")
            end = std::atol(argv[++i]);
        else if(a == "--deadline")
            deadline = std::atof(argv[++i]);
        else if(a == "--progress")
            progress = argv[++i];
        else if(a == "--default")
            defaultRun = true;
        else if(a == "--jit")
            cold.jit = true;
        else if(a == "--cold")
            cold.reps = std::max(1, std::atoi(argv[++i]));
        else if(a == "--cold-out")
            coldOut = argv[++i];
        else if(a == "--flush-mb")
            cold.flushB = size_t(std::atol(argv[++i])) << 20;
        else if(a == "--tag-suffix")
            tagSuffix = argv[++i];
    }
    if(cold.reps > 0)
    {
        if(coldOut.empty())
        {
            std::fprintf(stderr, "--cold needs --cold-out\n");
            return 2;
        }
        cold.out = std::fopen(coldOut.c_str(), "a");
        if(cold.out == nullptr)
        {
            std::fprintf(stderr, "cannot open %s\n", coldOut.c_str());
            return 2;
        }
        CHECK_HIP(hipMalloc(&cold.flush, cold.flushB));
        CHECK_HIP(hipMemset(cold.flush, 0x5A, cold.flushB));
        CHECK_HIP(hipMalloc(&cold.sink, sizeof(uint32_t)));
        CHECK_HIP(hipMalloc(&cold.sums, 2 * sizeof(double)));
        CHECK_HIP(hipMalloc(&cold.bad, sizeof(unsigned long long)));
    }
    std::string const dumpPath = std::getenv("HIPTENSOR_ROCKE_SELECT_DUMP") != nullptr
                                     ? std::getenv("HIPTENSOR_ROCKE_SELECT_DUMP")
                                     : "";
    auto              t0       = std::chrono::steady_clock::now();
    hiptensorHandle_t h        = nullptr;
    if(hiptensorCreate(&h) != HIPTENSOR_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "hiptensorCreate failed\n");
        return 3;
    }
    hiptensorLoggerSetCallback(&callback);
    hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE | HIPTENSOR_LOG_LEVEL_ERROR);

    std::ifstream in(specs);
    std::string   line;
    long          index = -1, done = 0, failed = 0;
    while(std::getline(in, line))
    {
        if(line.empty())
            continue;
        index++;
        if(index < start || index >= end)
            continue;
        double elapsed
            = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if(elapsed > deadline)
        {
            std::printf("DEADLINE next=%ld\n", index);
            break;
        }
        std::istringstream                       ss(line);
        std::string                              kind, dtype, compute;
        long                                     id   = 0;
        hiptensorOperationDescriptor_t           desc = nullptr;
        std::vector<hiptensorTensorDescriptor_t> descs;
        ss >> kind >> id >> dtype >> compute;
        auto              type = dtypeOf(dtype);
        hiptensorStatus_t st   = HIPTENSOR_STATUS_SUCCESS;
        T                 A, B, D;
        int               bil = 0;
        if(kind == "C")
        {
            int na = 0, nb = 0, nd = 0;
            ss >> bil >> na >> nb >> nd;
            A       = readTensor(ss, na);
            B       = readTensor(ss, nb);
            D       = readTensor(ss, nd);
            auto dA = describe(h, A, type), dB = describe(h, B, type), dD = describe(h, D, type);
            auto dC = bil ? describe(h, D, type) : nullptr;
            descs   = {dA, dB, dD, dC};
            st      = hiptensorCreateContraction(h,
                                            &desc,
                                            dA,
                                            A.modes.data(),
                                            HIPTENSOR_OP_IDENTITY,
                                            dB,
                                            B.modes.data(),
                                            HIPTENSOR_OP_IDENTITY,
                                            dC,
                                            bil ? D.modes.data() : nullptr,
                                            HIPTENSOR_OP_IDENTITY,
                                            dD,
                                            D.modes.data(),
                                            computeOf(compute));
        }
        else
        {
            std::string op;
            int         beta = 0, na = 0, nd = 0;
            ss >> op >> beta >> na >> nd;
            A       = readTensor(ss, na);
            D       = readTensor(ss, nd);
            auto dA = describe(h, A, type), dC = describe(h, D, type), dD = describe(h, D, type);
            descs = {dA, dC, dD};
            st    = hiptensorCreateReduction(h,
                                          &desc,
                                          dA,
                                          A.modes.data(),
                                          HIPTENSOR_OP_IDENTITY,
                                          dC,
                                          D.modes.data(),
                                          HIPTENSOR_OP_IDENTITY,
                                          dD,
                                          D.modes.data(),
                                          opOf(op),
                                          computeOf(compute));
        }
        if(st == HIPTENSOR_STATUS_SUCCESS)
        {
            std::string tag = std::to_string(id) + tagSuffix;
            setenv("HIPTENSOR_ROCKE_SELECT_DUMP_TAG", tag.c_str(), 1);
            if(!dumpPath.empty())
                setenv("HIPTENSOR_ROCKE_SELECT_DUMP", dumpPath.c_str(), 1);
            hiptensorPlanPreference_t pref = nullptr;
            hiptensorCreatePlanPreference(
                h,
                &pref,
                defaultRun ? HIPTENSOR_ALGO_DEFAULT : HIPTENSOR_ALGO_DEFAULT_PATIENT,
                cold.jit ? HIPTENSOR_JIT_MODE_DEFAULT : HIPTENSOR_JIT_MODE_NONE);
            hiptensorCacheMode_t none = HIPTENSOR_CACHE_MODE_NONE;
            hiptensorPlanPreferenceSetAttribute(
                h, pref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &none, sizeof(none));
            // the workspace every candidate runs with: the dump lists every candidate (a plan
            // with the DEFAULT estimate selects the same candidate among fewer)
            uint64_t ws = 0;
            st = hiptensorEstimateWorkspaceSize(h, desc, pref, HIPTENSOR_WORKSPACE_MAX, &ws);
            hiptensorPlan_t plan = nullptr;
            lastSelection.clear();
            double planUs = 0;
            if(st == HIPTENSOR_STATUS_SUCCESS)
            {
                auto p0 = std::chrono::steady_clock::now();
                st      = hiptensorCreatePlan(h, &plan, desc, pref, ws);
                planUs  = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()
                                                                   - p0)
                             .count();
            }
            if(plan != nullptr)
                hiptensorDestroyPlan(plan);
            hiptensorDestroyPlanPreference(pref);
            if(defaultRun)
                std::printf("SEL %ld plan_us=%.1f %s\n", id, planUs, lastSelection.c_str());
            if(st == HIPTENSOR_STATUS_SUCCESS && cold.reps > 0 && kind == "C")
            {
                // the per-candidate plans of the cold timing write no dump lines
                unsetenv("HIPTENSOR_ROCKE_SELECT_DUMP");
                coldContraction(h, desc, computeOf(compute), type, A, B, D, bil != 0, tag, cold);
            }
        }
        if(desc != nullptr)
            hiptensorDestroyOperationDescriptor(desc);
        for(auto d : descs)
            if(d != nullptr)
                hiptensorDestroyTensorDescriptor(d);
        if(st != HIPTENSOR_STATUS_SUCCESS)
        {
            failed++;
            std::printf("FAIL %ld %s\n", id, hiptensorGetErrorString(st));
        }
        done++;
        if(!progress.empty())
        {
            FILE* f = std::fopen(progress.c_str(), "w");
            if(f)
            {
                std::fprintf(f, "%ld\n", index + 1);
                std::fclose(f);
            }
        }
        std::fflush(stdout);
    }
    std::printf("SWEEP_END done=%ld failed=%ld last=%ld\n", done, failed, index);
    hiptensorLoggerSetCallback(nullptr);
    hiptensorDestroy(h);
    if(cold.out != nullptr)
    {
        std::fclose(cold.out);
        hipFree(cold.flush);
        hipFree(cold.sink);
        hipFree(cold.sums);
        hipFree(cold.bad);
    }
    return 0;
}
