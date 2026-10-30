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

// The rocKE backend's contraction JIT (rocke_gett_jit.hpp) through the public API
// (HIPTENSOR_BACKEND=rocke, no fallback, a build with HIPTENSOR_ENABLE_ROCKE_JIT): contractions
// the pack's GETT route tables do not hold take, with HIPTENSOR_JIT_MODE_DEFAULT, the route
// rocKE's tensor GETT planner chooses, built and compiled at plan creation.
//  - fixed problems of every route kind (the MN-layout GETT in place, with one or two operands
//    packed first, the K-contiguous route; ragged extents, padded strides, an extent-1 mode) and
//    seeded random problems (ranks 2-5 per tensor, random mode orders, padded strides, f16 /
//    bf16, scale / bilinear): every output is compared with a double-precision loop reference
//    after one rounding (integer data: the f32 sums are exact), with infinities and NaNs in the
//    operands (IEEE propagation), alpha or beta zero (the operand of the zero scale is not
//    read), and with a workspace or an operand that misses the 16-byte alignment (the plan's
//    fallback, the direct GETT); D's padding is never written;
//  - HIPTENSOR_JIT_MODE_NONE never compiles and never takes the route; a second plan of a
//    problem (or of the same memory layout described with other mode labels and orders) takes
//    the kernel cache; hiptensorWriteKernelCacheToFile / HIPTENSOR_KERNEL_CACHE_DIR carry the
//    kernels to a fresh process that may not compile;
//  - a kernel that cannot be built (compiling disabled, the time budget spent, missing or
//    damaged selection models) leaves the plan on the next candidate with the same results;
//  - the boundary to rocKE's planner: every problem of rocKE's tuned and AOT GETT route tables,
//    described to hipTensor with other mode labels, in other descriptor orders and with extent-1
//    modes, and planned by the JIT with the tables off (HIPTENSOR_ROCKE_GETT_TABLES=off), takes
//    exactly the table's launches (kernels, grids, blocks, pack steps, arguments) as rocKE's C++
//    engine (linked into this test) plans them for the table entry.
// RockeGettJitLatency.DISABLED_Report prints the JIT's costs (planner, cold compile per kernel,
// cache hits).

#include <gtest/gtest.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "api_test_helpers.hpp"

// rocKE's C++ engine (the route tables and their plans; this test links hiptensor_rocke_core)
#include "rocke/instance_tensor_gett.h"

namespace
{
    using hiptensor_test::ApiObjects;
    using hiptensor_test::ScopedEnvironmentVariable;

    // The rocKE backend alone. The route tests take the planner's route whenever the problem
    // has one: HIPTENSOR_ROCKE_SELECT=static orders the candidates by the rules (the JIT route
    // first) instead of the pack's selection model, which may rank another route first (the
    // RockeGettJitSelection tests select with models).
    struct RockeOnly : ::testing::Environment
    {
        void SetUp() override
        {
#ifdef _WIN32
            _putenv_s("HIPTENSOR_BACKEND", "rocke");
            _putenv_s("HIPTENSOR_BACKEND_FALLBACK", "0");
            _putenv_s("HIPTENSOR_ROCKE_SELECT", "static");
#else
            setenv("HIPTENSOR_BACKEND", "rocke", 1);
            setenv("HIPTENSOR_BACKEND_FALLBACK", "0", 1);
            setenv("HIPTENSOR_ROCKE_SELECT", "static", 1);
#endif
        }
    };
    auto const* const gEnvironment = ::testing::AddGlobalTestEnvironment(new RockeOnly);

    // The library's log lines from the capture's start (or its last clear()). Captures nest: an
    // inner one (runProblem's) sees its own lines, an outer one every line of its scope.
    class LogCapture
    {
    public:
        LogCapture()
        {
            std::scoped_lock lock(sMutex);
            if(sDepth++ == 0)
            {
                sLines.clear();
                hiptensorLoggerSetCallback(&LogCapture::callback);
                hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE
                                       | HIPTENSOR_LOG_LEVEL_ERROR);
            }
            mStart = sLines.size();
        }
        ~LogCapture()
        {
            std::scoped_lock lock(sMutex);
            if(--sDepth == 0)
            {
                hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_OFF);
                hiptensorLoggerSetCallback(nullptr);
            }
        }
        LogCapture(LogCapture const&)            = delete;
        LogCapture& operator=(LogCapture const&) = delete;
        void        clear()
        {
            std::scoped_lock lock(sMutex);
            mStart = sLines.size();
        }
        std::vector<std::string> lines() const
        {
            std::scoped_lock lock(sMutex);
            return std::vector<std::string>(sLines.begin() + long(std::min(mStart, sLines.size())),
                                            sLines.end());
        }
        bool contains(std::string const& text) const
        {
            auto all = lines();
            return std::any_of(all.begin(), all.end(), [&text](auto const& line) {
                return line.find(text) != std::string::npos;
            });
        }
        std::string last(std::string const& text) const
        {
            auto all = lines();
            for(auto it = all.rbegin(); it != all.rend(); ++it)
            {
                if(it->find(text) != std::string::npos)
                {
                    return *it;
                }
            }
            return "";
        }
        std::string all() const
        {
            std::string out;
            for(auto const& l : lines())
            {
                out += l + "\n";
            }
            return out;
        }

    private:
        static void callback(int32_t, char const* func, char const* msg)
        {
            std::scoped_lock lock(sMutex);
            sLines.push_back(std::string(func) + ": " + msg);
        }
        size_t                                 mStart = 0;
        static inline std::mutex               sMutex;
        static inline std::vector<std::string> sLines;
        static inline int                      sDepth = 0;
    };

    std::string deviceArch()
    {
        int             device = 0;
        hipDeviceProp_t props;
        if(hipGetDevice(&device) != hipSuccess
           || hipGetDeviceProperties(&props, device) != hipSuccess)
        {
            return "";
        }
        std::string name(props.gcnArchName);
        return name.substr(0, name.find(':'));
    }

    // The targets rocKE's tensor GETT planner serves (rocke::gett::planner::plans_arch).
    bool plannerTarget()
    {
        return deviceArch() == "gfx950";
    }

    std::string selfExecutable()
    {
#ifdef _WIN32
        return "";
#else
        char    buf[4096];
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        return n > 0 ? std::string(buf, size_t(n)) : "";
#endif
    }

    std::string tempPath(std::string const& name)
    {
        return ::testing::TempDir() + "hiptensor_rocke_gett_jit_" + name + "_"
#ifndef _WIN32
               + std::to_string(getpid())
#endif
            ;
    }

    // Runs one test of this binary in a fresh process; true when it passed.
    bool runChild(std::string const& filter)
    {
        auto exe = selfExecutable();
        if(exe.empty())
        {
            return false;
        }
        std::string cmd = "\"" + exe + "\" --gtest_filter=" + filter;
        return std::system(cmd.c_str()) == 0;
    }

    // ------------------------------------------------------------------ problems

    // A contraction in rocKE's terms: mode lists row-major (the last mode is the fastest),
    // element strides per mode (empty: packed in the list's order). The hipTensor tensors take
    // the reversed lists (column-major), so the memory layout is the one the lists name.
    struct Problem
    {
        std::string                    name;
        hiptensorDataType_t            type     = HIPTENSOR_R_16F;
        bool                           bilinear = false;
        std::vector<std::string>       a, b, d;
        std::map<std::string, int64_t> ext;
        std::map<std::string, int64_t> sa, sb, sd;
    };

    std::string describe(Problem const& p)
    {
        auto list
            = [&p](std::vector<std::string> const& m, std::map<std::string, int64_t> const& st) {
                  std::string out = "[";
                  for(size_t i = 0; i < m.size(); i++)
                  {
                      out += (i ? " " : "") + m[i] + "=" + std::to_string(p.ext.at(m[i]));
                      if(!st.empty())
                      {
                          out += "/" + std::to_string(st.at(m[i]));
                      }
                  }
                  return out + "]";
              };
        return p.name + (p.type == HIPTENSOR_R_16F ? " f16 " : " bf16 ")
               + (p.bilinear ? "bilinear " : "scale ") + "A" + list(p.a, p.sa) + " B"
               + list(p.b, p.sb) + " D" + list(p.d, p.sd);
    }

    uint16_t encode(double v, hiptensorDataType_t t)
    {
        uint16_t out = 0;
        if(t == HIPTENSOR_R_16F)
        {
            _Float16 h = static_cast<_Float16>(v);
            std::memcpy(&out, &h, 2);
        }
        else
        {
            hip_bfloat16 h(static_cast<float>(v));
            std::memcpy(&out, &h, 2);
        }
        return out;
    }

    double decode(uint16_t b, hiptensorDataType_t t)
    {
        if(t == HIPTENSOR_R_16F)
        {
            _Float16 h;
            std::memcpy(&h, &b, 2);
            return static_cast<double>(h);
        }
        hip_bfloat16 h;
        std::memcpy(&h, &b, 2);
        return static_cast<double>(static_cast<float>(h));
    }

    double roundTo(double v, hiptensorDataType_t t)
    {
        return decode(encode(v, t), t);
    }

    // A strided 16-bit tensor in host memory: its hipTensor modes (column-major), strides and
    // storage (the span of the strides).
    struct HostTensor
    {
        std::vector<std::string> names;
        std::vector<int32_t>     modes;
        std::vector<int64_t>     extents, strides;
        int64_t                  span = 1;
        std::vector<uint16_t>    bits;
    };

    int32_t modeId(std::string const& name)
    {
        // a stable distinct integer per name
        int32_t h = 7;
        for(char c : name)
        {
            h = h * 131 + c;
        }
        return h & 0x3fffffff;
    }

    HostTensor hostTensor(Problem const&                        p,
                          std::vector<std::string> const&       rowMajor,
                          std::map<std::string, int64_t> const& strides,
                          std::map<std::string, int32_t> const& ids)
    {
        HostTensor t;
        t.names        = std::vector<std::string>(rowMajor.rbegin(), rowMajor.rend());
        int64_t packed = 1;
        t.span         = 1;
        for(auto const& n : t.names)
        {
            t.modes.push_back(ids.at(n));
            t.extents.push_back(p.ext.at(n));
            int64_t s = strides.empty() ? packed : strides.at(n);
            t.strides.push_back(s);
            t.span += (p.ext.at(n) - 1) * s;
            packed *= p.ext.at(n);
        }
        return t;
    }

    // Integer values in [-2, 2], spread by a hash of the storage index.
    void fill(HostTensor& t, uint64_t seed, hiptensorDataType_t type)
    {
        t.bits.resize(size_t(t.span));
        for(int64_t i = 0; i < t.span; i++)
        {
            uint64_t h        = (uint64_t(i) + seed * 0x51ull) * 0x9E3779B97F4A7C15ull;
            t.bits[size_t(i)] = encode(double(int64_t((h >> 40) % 5) - 2), type);
        }
    }

    int64_t offsetOf(HostTensor const& t, std::map<std::string, int64_t> const& index)
    {
        int64_t off = 0;
        for(size_t i = 0; i < t.names.size(); i++)
        {
            auto it = index.find(t.names[i]);
            if(it != index.end())
            {
                off += it->second * t.strides[i];
            }
        }
        return off;
    }

    struct DeviceBuffer
    {
        void* p = nullptr;
        explicit DeviceBuffer(size_t bytes)
        {
            EXPECT_EQ(hipMalloc(&p, std::max<size_t>(bytes, 16)), hipSuccess);
        }
        ~DeviceBuffer()
        {
            static_cast<void>(hipFree(p));
        }
        DeviceBuffer(DeviceBuffer const&)            = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    };

    struct Stream
    {
        hipStream_t s = nullptr;
        Stream()
        {
            EXPECT_EQ(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), hipSuccess);
        }
        ~Stream()
        {
            static_cast<void>(hipStreamDestroy(s));
        }
    };

    // What a plan of a problem was.
    struct Outcome
    {
        bool        planned = false;
        std::string pick, source, reason;
        bool        jit = false; // the JIT GETT route
        std::string route; // the planner's "<kind>/<candidate>"
        std::string kernel; // the JIT GETT kernel
        int         kernels = 0, compiled = 0, hits = 0, fileHits = 0;
        double      jitMs = 0.0, planMs = 0.0;
        uint64_t    workspace = 0;
        bool        exact     = true;
        int64_t     checked   = 0;
    };

    std::string field(std::string const& line, std::string const& name)
    {
        auto at = line.find(name + ": ");
        if(at == std::string::npos)
        {
            return "";
        }
        at += name.size() + 2;
        auto end = line.find_first_of(" ,;", at);
        return line.substr(at, end == std::string::npos ? std::string::npos : end - at);
    }

    void readLog(LogCapture const& log, Outcome& o)
    {
        auto const sel = log.last("rocKE selection:");
        o.pick         = field(sel, "Pick");
        o.source       = field(sel, "Source");
        auto open      = sel.find("(", sel.find("Source: "));
        auto close     = sel.rfind("), Pick: ");
        if(open != std::string::npos && close != std::string::npos && close > open)
        {
            o.reason = sel.substr(open + 1, close - open - 1);
        }
        o.jit = o.pick.rfind("jit:", 0) == 0;
        if(o.jit)
        {
            o.kernel = o.pick.substr(4);
        }
        auto const jl = log.last("rocKE GETT JIT: ");
        if(!jl.empty())
        {
            int k = 0, c = 0, h = 0, f = 0;
            if(std::sscanf(jl.c_str() + jl.find("rocKE GETT JIT: "),
                           "rocKE GETT JIT: %d kernels (compiled %d, kernel cache hits %d, of "
                           "which from a file %d) in %lf ms",
                           &k,
                           &c,
                           &h,
                           &f,
                           &o.jitMs)
               == 5)
            {
                o.kernels  = k;
                o.compiled = c;
                o.hits     = h;
                o.fileHits = f;
            }
        }
        // "jit-gett:<kind>/<candidate>:<kernel>" (the program description in the plan's id)
        for(auto const& l : log.lines())
        {
            auto at = l.find("jit-gett:");
            if(at != std::string::npos)
            {
                auto rest = l.substr(at + 9);
                o.route   = rest.substr(0, rest.find(':'));
            }
        }
    }

    struct Variant
    {
        char const* what;
        double      alpha, beta;
        bool        poisonAB, poisonC;
        size_t      wsSkew; // bytes the workspace pointer is moved by
        size_t      aSkew; // elements A is moved by
    };

    struct RunOptions
    {
        hiptensorJitMode_t   jit  = HIPTENSOR_JIT_MODE_DEFAULT;
        hiptensorAlgo_t      algo = HIPTENSOR_ALGO_DEFAULT;
        std::vector<Variant> variants;
        int                  samples = 1024;
        ApiObjects*          api     = nullptr; // share a handle
        // a variant per plan: also check D's padding is never written
        bool checkPadding = true;
        // execute on this device (buffers and stream there; -1: the plan's device)
        int execDevice = -1;
    };

    // Makes a device current for its lifetime (nothing for -1), then restores the previous one.
    struct DeviceScope
    {
        int previous = -1;
        explicit DeviceScope(int device)
        {
            if(device >= 0)
            {
                EXPECT_EQ(hipGetDevice(&previous), hipSuccess);
                EXPECT_EQ(hipSetDevice(device), hipSuccess);
            }
        }
        ~DeviceScope()
        {
            if(previous >= 0)
            {
                static_cast<void>(hipSetDevice(previous));
            }
        }
        DeviceScope(DeviceScope const&)            = delete;
        DeviceScope& operator=(DeviceScope const&) = delete;
    };

    // Another device of the current device's architecture (-1: none).
    int otherDevice()
    {
        int current = 0, count = 0;
        if(hipGetDevice(&current) != hipSuccess || hipGetDeviceCount(&count) != hipSuccess)
        {
            return -1;
        }
        hipDeviceProp_t mine, other;
        if(hipGetDeviceProperties(&mine, current) != hipSuccess)
        {
            return -1;
        }
        for(int d = 0; d < count; d++)
        {
            if(d != current && hipGetDeviceProperties(&other, d) == hipSuccess
               && std::string(other.gcnArchName) == std::string(mine.gcnArchName))
            {
                return d;
            }
        }
        return -1;
    }

    std::vector<Variant> allVariants(bool bilinear)
    {
        std::vector<Variant> v = {
            {"plain", 1.0, -2.0, false, false, 0, 0},
            {"inf/nan operands", 2.0, -1.0, true, false, 0, 0},
            {"alpha 0, A and B poisoned", 0.0, -2.0, true, false, 0, 0},
            {"misaligned workspace", 1.0, 3.0, false, false, 8, 0},
            {"misaligned A", -1.0, 2.0, false, false, 0, 1},
        };
        if(bilinear)
        {
            v.push_back({"beta 0, C poisoned", 2.0, 0.0, false, true, 0, 0});
        }
        return v;
    }

    // Plans `p` with the JIT mode of `opt`, then runs the variants and compares every sampled
    // output (and outputs reading the poisoned elements) with the reference.
    Outcome runProblem(Problem const& p, RunOptions const& opt)
    {
        SCOPED_TRACE(describe(p));
        Outcome                     o;
        std::unique_ptr<ApiObjects> own;
        ApiObjects*                 api = opt.api;
        if(api == nullptr)
        {
            own = std::make_unique<ApiObjects>();
            api = own.get();
        }
        LogCapture                     log;
        std::map<std::string, int32_t> ids;
        for(auto const& [n, e] : p.ext)
        {
            ids[n] = modeId(n);
        }
        HostTensor A = hostTensor(p, p.a, p.sa, ids), B = hostTensor(p, p.b, p.sb, ids),
                   D = hostTensor(p, p.d, p.sd, ids), C = hostTensor(p, p.d, p.sd, ids);
        fill(A, 1, p.type);
        fill(B, 2, p.type);
        fill(C, 3, p.type);
        std::vector<std::string> kmodes;
        for(auto const& n : A.names)
        {
            if(std::find(D.names.begin(), D.names.end(), n) == D.names.end())
            {
                kmodes.push_back(n);
            }
        }
        std::vector<std::pair<int64_t, int64_t>> koffs = {{0, 0}};
        for(auto const& k : kmodes)
        {
            std::map<std::string, int64_t>           one = {{k, 1}};
            int64_t const                            sa = offsetOf(A, one), sb = offsetOf(B, one);
            std::vector<std::pair<int64_t, int64_t>> next;
            for(int64_t i = 0; i < p.ext.at(k); i++)
            {
                for(auto const& kk : koffs)
                {
                    next.emplace_back(kk.first + i * sa, kk.second + i * sb);
                }
            }
            koffs.swap(next);
        }
        auto                           dA   = api->tensor(A.extents, A.strides, p.type);
        auto                           dB   = api->tensor(B.extents, B.strides, p.type);
        auto                           dD   = api->tensor(D.extents, D.strides, p.type);
        hiptensorOperationDescriptor_t desc = nullptr;
        EXPECT_EQ(hiptensorCreateContraction(api->handle,
                                             &desc,
                                             dA,
                                             A.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dB,
                                             B.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             p.bilinear ? dD : nullptr,
                                             p.bilinear ? D.modes.data() : nullptr,
                                             HIPTENSOR_OP_IDENTITY,
                                             dD,
                                             D.modes.data(),
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        if(desc == nullptr)
        {
            return o;
        }
        api->keep(desc);
        hiptensorPlanPreference_t pref = nullptr;
        EXPECT_EQ(hiptensorCreatePlanPreference(api->handle, &pref, opt.algo, opt.jit),
                  HIPTENSOR_STATUS_SUCCESS);
        uint64_t estimate = 0;
        EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                      api->handle, desc, pref, HIPTENSOR_WORKSPACE_DEFAULT, &estimate),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        log.clear();
        hiptensorPlan_t plan   = nullptr;
        auto const      t0     = std::chrono::steady_clock::now();
        auto            status = hiptensorCreatePlan(api->handle, &plan, desc, pref, estimate);
        o.planMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                       .count();
        static_cast<void>(hiptensorDestroyPlanPreference(pref));
        EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << log.all();
        if(plan == nullptr)
        {
            return o;
        }
        api->plans.push_back(plan);
        o.planned = true;
        readLog(log, o);
        EXPECT_EQ(hiptensorPlanGetAttribute(api->handle,
                                            plan,
                                            HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                            &o.workspace,
                                            sizeof(o.workspace)),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_LE(o.workspace, estimate);
        // the planner's route goes first by the rules, or the cell's model ranks it first (or
        // the handle's plan cache names it, or the measured selection timed it fastest)
        if(o.jit && o.source != "PLAN_CACHE" && opt.algo == HIPTENSOR_ALGO_DEFAULT)
        {
            if(o.source == "MODEL")
            {
                EXPECT_EQ(o.reason.rfind("contraction.h@", 0), 0u) << log.all();
            }
            else
            {
                EXPECT_EQ(o.source, "TUNED") << log.all();
                EXPECT_EQ(o.reason, "rocKE's tensor GETT planner (JIT)") << log.all();
            }
        }

        // outputs: random samples, outputs reading A's first element (every M mode 0) and B's
        // last (every N mode at its last index)
        std::mt19937_64 rng(1234);
        int64_t         dCount = 1;
        for(auto e : D.extents)
        {
            dCount *= e;
        }
        auto indexOf = [&D](int64_t flat) {
            std::map<std::string, int64_t> index;
            for(size_t i = 0; i < D.names.size(); i++)
            {
                index[D.names[i]] = flat % D.extents[i];
                flat /= D.extents[i];
            }
            return index;
        };
        std::vector<std::map<std::string, int64_t>> outs;
        std::uniform_int_distribution<int64_t>      pick(0, dCount - 1);
        for(int s = 0; s < opt.samples; s++)
        {
            outs.push_back(indexOf(pick(rng)));
        }
        for(int s = 0; s < 64; s++)
        {
            auto index = indexOf(pick(rng));
            for(auto& [n, v] : index)
            {
                bool const m = std::find(A.names.begin(), A.names.end(), n) != A.names.end();
                if(s % 2 == 0 && m)
                {
                    v = 0;
                }
                if(s % 2 == 1 && !m)
                {
                    v = p.ext.at(n) - 1;
                }
            }
            outs.push_back(index);
        }
        // D's elements (the rest of its span is padding)
        std::vector<bool> inD;
        if(opt.checkPadding)
        {
            inD.assign(size_t(D.span), false);
            for(int64_t f = 0; f < dCount; f++)
            {
                inD[size_t(offsetOf(D, indexOf(f)))] = true;
            }
        }

        double const inf = std::numeric_limits<double>::infinity();
        double const nan = std::numeric_limits<double>::quiet_NaN();
        DeviceScope  onDevice(opt.execDevice);
        DeviceBuffer bufA(size_t(A.span + 8) * 2), bufB(size_t(B.span) * 2),
            bufC(size_t(C.span) * 2), bufD(size_t(D.span) * 2), bufW(o.workspace + 64);
        uint16_t const sentinel = 0x7c01; // a NaN pattern in f16; an ordinary value in bf16
        for(auto const& v : opt.variants)
        {
            SCOPED_TRACE(v.what);
            HostTensor a = A, b = B, cc = C;
            if(v.poisonAB)
            {
                a.bits[0]                  = encode(inf, p.type);
                b.bits[size_t(b.span - 1)] = encode(nan, p.type);
                a.bits[size_t(a.span / 2)] = encode(-inf, p.type);
            }
            if(v.poisonC)
            {
                for(auto& x : cc.bits)
                {
                    x = encode(nan, p.type);
                }
            }
            auto* pA = static_cast<uint16_t*>(bufA.p) + v.aSkew;
            EXPECT_EQ(hipMemcpy(pA, a.bits.data(), a.bits.size() * 2, hipMemcpyHostToDevice),
                      hipSuccess);
            EXPECT_EQ(hipMemcpy(bufB.p, b.bits.data(), b.bits.size() * 2, hipMemcpyHostToDevice),
                      hipSuccess);
            EXPECT_EQ(hipMemcpy(bufC.p, cc.bits.data(), cc.bits.size() * 2, hipMemcpyHostToDevice),
                      hipSuccess);
            std::vector<uint16_t> init(size_t(D.span), sentinel);
            EXPECT_EQ(hipMemcpy(bufD.p, init.data(), init.size() * 2, hipMemcpyHostToDevice),
                      hipSuccess);
            float  alpha = float(v.alpha), beta = float(v.beta);
            Stream stream;
            void*  w  = static_cast<uint8_t*>(bufW.p) + v.wsSkew;
            auto   st = hiptensorContract(api->handle,
                                        plan,
                                        &alpha,
                                        pA,
                                        bufB.p,
                                        &beta,
                                        p.bilinear ? bufC.p : nullptr,
                                        bufD.p,
                                        o.workspace > 0 ? w : nullptr,
                                        o.workspace,
                                        stream.s);
            EXPECT_EQ(st, HIPTENSOR_STATUS_SUCCESS) << log.all();
            EXPECT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            std::vector<uint16_t> got(size_t(D.span));
            EXPECT_EQ(hipMemcpy(got.data(), bufD.p, got.size() * 2, hipMemcpyDeviceToHost),
                      hipSuccess);
            int64_t bad = 0, nonfinite = 0;
            for(auto const& index : outs)
            {
                double sum = 0.0;
                if(v.alpha != 0.0)
                {
                    int64_t const ba = offsetOf(a, index), bb = offsetOf(b, index);
                    for(auto const& kk : koffs)
                    {
                        sum += decode(a.bits[size_t(ba + kk.first)], p.type)
                               * decode(b.bits[size_t(bb + kk.second)], p.type);
                    }
                }
                double        want = v.alpha != 0.0 ? v.alpha * sum : 0.0;
                int64_t const od   = offsetOf(D, index);
                if(p.bilinear && v.beta != 0.0)
                {
                    want += v.beta * decode(cc.bits[size_t(od)], p.type);
                }
                want       = roundTo(want, p.type);
                double out = decode(got[size_t(od)], p.type);
                nonfinite += !std::isfinite(want);
                bool same = std::isnan(want) ? std::isnan(out) : out == want;
                o.checked++;
                if(!same && bad++ < 5)
                {
                    ADD_FAILURE() << "output at " << od << " got " << out << " want " << want
                                  << " (pick " << o.pick << ")";
                }
            }
            if(opt.checkPadding)
            {
                int64_t written = 0;
                for(int64_t i = 0; i < D.span; i++)
                {
                    written += !inD[size_t(i)] && got[size_t(i)] != sentinel;
                }
                EXPECT_EQ(written, 0) << "padding of D written (pick " << o.pick << ")";
                bad += written;
            }
            EXPECT_EQ(bad, 0);
            o.exact = o.exact && bad == 0;
            if(v.poisonAB && v.alpha != 0.0)
            {
                EXPECT_GT(nonfinite, 0);
            }
            if(v.alpha == 0.0 || v.poisonC)
            {
                EXPECT_EQ(nonfinite, 0);
            }
        }
        return o;
    }

    // ------------------------------------------------------------------ fixed problems

    // Problems of every route kind outside the pack's route tables (the planner decides; the
    // expected kinds hold for the shipped planner and models on gfx950, as rocKE's own device
    // test of whole plans): the MN-layout GETT in place, A or both operands K-fastest (packed
    // first), ragged extents with an extent-1 mode and padded strides, and a long-K problem of
    // the K-contiguous route.
    std::vector<Problem> fixedProblems()
    {
        using M     = std::map<std::string, int64_t>;
        M const ext = {{"K0", 24}, {"K1", 24}, {"M0", 72}, {"M1", 20}, {"N0", 60}, {"N1", 12}};
        M const rag
            = {{"K0", 40}, {"K1", 3}, {"M0", 50}, {"M1", 7}, {"N0", 36}, {"N1", 5}, {"U", 1}};
        std::vector<Problem> out;
        for(auto type : {HIPTENSOR_R_16F, HIPTENSOR_R_16BF})
        {
            for(bool bil : {false, true})
            {
                std::string const t = std::string(type == HIPTENSOR_R_16F ? "f16" : "bf16")
                                      + (bil ? "_bilinear" : "_scale");
                out.push_back({"mn_" + t,
                               type,
                               bil,
                               {"K1", "K0", "M1", "M0"},
                               {"K1", "K0", "N1", "N0"},
                               {"N1", "N0", "M1", "M0"},
                               ext});
                out.push_back({"kfast_a_" + t,
                               type,
                               bil,
                               {"M1", "M0", "K1", "K0"},
                               {"K1", "K0", "N1", "N0"},
                               {"N1", "N0", "M1", "M0"},
                               ext});
                out.push_back({"kfast_ab_" + t,
                               type,
                               bil,
                               {"M1", "M0", "K1", "K0"},
                               {"N1", "N0", "K1", "K0"},
                               {"N1", "N0", "M1", "M0"},
                               ext});
                Problem r{"ragged_padded_" + t,
                          type,
                          bil,
                          {"K1", "U", "K0", "M1", "M0"},
                          {"K1", "K0", "N1", "N0"},
                          {"N1", "N0", "U", "M1", "M0"},
                          rag};
                r.sa = {{"M0", 1},
                        {"M1", 56},
                        {"K0", 56 * 7 + 8},
                        {"U", 7},
                        {"K1", (56 * 7 + 8) * 40 + 16}};
                r.sb = {{"N0", 1}, {"N1", 40}, {"K0", 40 * 5 + 24}, {"K1", (40 * 5 + 24) * 40}};
                r.sd = {{"M0", 1},
                        {"M1", 52},
                        {"U", 52 * 7},
                        {"N0", 52 * 7 + 4},
                        {"N1", (52 * 7 + 4) * 36 + 8}};
                out.push_back(r);
            }
        }
        // a long contraction with D's fastest run long: the K-contiguous route's domain
        out.push_back({"long_k_f16_scale",
                       HIPTENSOR_R_16F,
                       false,
                       {"M0", "K1", "K0"},
                       {"N0", "K1", "K0"},
                       {"M0", "N0"},
                       {{"K0", 512}, {"K1", 12}, {"M0", 384}, {"N0", 512}}});
        out.push_back({"long_k_bf16_bilinear",
                       HIPTENSOR_R_16BF,
                       true,
                       {"M0", "K1", "K0"},
                       {"N0", "K1", "K0"},
                       {"M0", "N0"},
                       {{"K0", 512}, {"K1", 10}, {"M0", 320}, {"N0", 768}}});
        // operands that already are the K-contiguous GETT's copies (packed in its operand mode
        // order): the planner runs that GETT on the user's A and B in place
        // (single/tensor_gett_kc; found with rocKE's planner, as the C++ planner decides):
        // two modes each; several modes with padded strides; extent-1 modes and padding
        out.push_back({"kc_in_place_f16_scale",
                       HIPTENSOR_R_16F,
                       false,
                       {"p", "b"},
                       {"o", "b"},
                       {"p", "o"},
                       {{"b", 64}, {"o", 1023}, {"p", 1248}}});
        Problem kc2{"kc_in_place_bf16_bilinear_padded",
                    HIPTENSOR_R_16BF,
                    true,
                    {"c", "h", "x", "e", "k"},
                    {"a", "x", "e", "k"},
                    {"c", "a", "h"},
                    {{"a", 133}, {"c", 224}, {"e", 8}, {"h", 45}, {"k", 2}, {"x", 6}}};
        kc2.sa = {{"k", 1}, {"e", 2}, {"x", 16}, {"h", 96}, {"c", 4320}};
        kc2.sb = {{"k", 1}, {"e", 2}, {"x", 16}, {"a", 96}};
        kc2.sd = {{"h", 1}, {"a", 48}, {"c", 6384}};
        out.push_back(kc2);
        Problem kc3{"kc_in_place_bf16_bilinear_unit_modes",
                    HIPTENSOR_R_16BF,
                    true,
                    {"a", "k", "z", "m"},
                    {"f", "x", "z", "g"},
                    {"x", "g", "a", "m", "f", "k"},
                    {{"a", 1}, {"f", 1}, {"g", 77}, {"k", 3}, {"m", 209}, {"x", 33}, {"z", 192}}};
        kc3.sa = {{"z", 1}, {"m", 192}, {"k", 40128}, {"a", 240768}};
        kc3.sb = {{"z", 1}, {"g", 192}, {"x", 14784}, {"f", 975744}};
        kc3.sd = {{"k", 1}, {"f", 3}, {"m", 3}, {"a", 627}, {"g", 627}, {"x", 48279}};
        out.push_back(kc3);
        return out;
    }

    // ------------------------------------------------------------------ random problems

    // Splits ~2^lg over `n` modes: extents that are multiples of 8 on the fastest positions
    // (`friendly`) or anything (odd, small).
    std::vector<int64_t> splitExtents(std::mt19937_64& rng, double lg, int n, bool friendly)
    {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double>                    cuts;
        for(int i = 0; i + 1 < n; i++)
        {
            cuts.push_back(u(rng) * lg);
        }
        std::sort(cuts.begin(), cuts.end());
        std::vector<int64_t> out;
        double               prev = 0.0;
        for(int i = 0; i < n; i++)
        {
            double const c = i + 1 < n ? cuts[size_t(i)] : lg;
            int64_t      e = std::max<int64_t>(1, std::llround(std::pow(2.0, c - prev)));
            prev           = c;
            if(friendly)
            {
                e = std::max<int64_t>(8, (e + 4) / 8 * 8);
            }
            else if(u(rng) < 0.3)
            {
                e |= 1;
            }
            out.push_back(std::min<int64_t>(e, 4096));
        }
        return out;
    }

    // A seeded random contraction: 1-2 M / N / K modes, random mode orders, padded strides,
    // tensors of at most 2^21 elements.
    Problem randomProblem(uint64_t seed)
    {
        std::mt19937_64                        rng(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        auto    pickInt = [&](int lo, int hi) { return lo + int(rng() % uint64_t(hi - lo + 1)); };
        Problem p;
        p.name        = "random_" + std::to_string(seed);
        p.type        = u(rng) < 0.5 ? HIPTENSOR_R_16F : HIPTENSOR_R_16BF;
        p.bilinear    = u(rng) < 0.5;
        int const  nm = pickInt(1, 2), nn = pickInt(1, 2), nk = pickInt(1, 2);
        bool const friendly = u(rng) < 0.7;
        while(true)
        {
            double const lm = 4.0 + u(rng) * 6.5, ln = 4.0 + u(rng) * 6.5, lk = 3.0 + u(rng) * 7.0;
            auto         em   = splitExtents(rng, lm, nm, friendly);
            auto         en   = splitExtents(rng, ln, nn, friendly);
            auto         ek   = splitExtents(rng, lk, nk, friendly);
            auto         prod = [](std::vector<int64_t> const& v) {
                int64_t x = 1;
                for(auto e : v)
                {
                    x *= e;
                }
                return x;
            };
            int64_t const M = prod(em), N = prod(en), K = prod(ek);
            if(M * K > (int64_t(1) << 21) || N * K > (int64_t(1) << 21)
               || M * N > (int64_t(1) << 21))
            {
                continue;
            }
            p.ext.clear();
            p.a.clear();
            p.b.clear();
            p.d.clear();
            for(int i = 0; i < nm; i++)
            {
                p.ext["m" + std::to_string(i)] = em[size_t(i)];
            }
            for(int i = 0; i < nn; i++)
            {
                p.ext["n" + std::to_string(i)] = en[size_t(i)];
            }
            for(int i = 0; i < nk; i++)
            {
                p.ext["k" + std::to_string(i)] = ek[size_t(i)];
            }
            break;
        }
        std::vector<std::string> ms, ns, ks;
        for(auto const& [n, e] : p.ext)
        {
            (n[0] == 'm' ? ms : n[0] == 'n' ? ns : ks).push_back(n);
        }
        auto order = [&](std::vector<std::string> v) {
            if(u(rng) < 0.8)
            {
                std::shuffle(v.begin(), v.end(), rng);
            }
            return v;
        };
        auto cat = [](std::vector<std::string> x, std::vector<std::string> const& y) {
            x.insert(x.end(), y.begin(), y.end());
            return x;
        };
        p.a = order(cat(ms, ks));
        p.b = order(cat(ns, ks));
        p.d = order(cat(ms, ns));
        // strides: packed, or a random memory order with padding
        auto strides = [&](std::vector<std::string> const& modes) {
            std::map<std::string, int64_t> st;
            if(u(rng) < 0.55)
            {
                return st;
            }
            auto mem = modes;
            if(u(rng) < 0.5)
            {
                std::shuffle(mem.begin(), mem.end(), rng);
            }
            int64_t acc = 1;
            for(auto it = mem.rbegin(); it != mem.rend(); ++it)
            {
                st[*it] = acc;
                acc *= p.ext.at(*it);
                if(u(rng) < 0.3)
                {
                    int64_t const pads[] = {1, 2, 8, 16, 64, 3};
                    acc += pads[rng() % 6];
                }
            }
            return st;
        };
        p.sa = strides(p.a);
        p.sb = strides(p.b);
        p.sd = strides(p.d);
        return p;
    }

    int envInt(char const* name, int fallback)
    {
        char const* v = std::getenv(name);
        return v != nullptr && *v != 0 ? std::atoi(v) : fallback;
    }
} // namespace

// The fixed problems: on gfx950 each takes the JIT route (the planner's route kinds), and every
// variant is exact; elsewhere the planner does not serve the target and the other routes are
// exact.
TEST(RockeGettJit, FixedRoutesExact)
{
    std::set<std::string> routes;
    for(auto const& p : fixedProblems())
    {
        RunOptions opt;
        opt.variants = allVariants(p.bilinear);
        auto o       = runProblem(p, opt);
        ASSERT_TRUE(o.planned) << describe(p);
        EXPECT_TRUE(o.exact) << describe(p);
        if(plannerTarget())
        {
            EXPECT_TRUE(o.jit) << describe(p) << " pick " << o.pick;
            routes.insert(o.route);
        }
        else
        {
            EXPECT_FALSE(o.jit) << o.pick;
        }
        std::printf("GETT-JIT-FIXED %s pick=%s route=%s kernels=%d compiled=%d plan_ms=%.1f\n",
                    p.name.c_str(),
                    o.pick.c_str(),
                    o.route.c_str(),
                    o.kernels,
                    o.compiled,
                    o.planMs);
    }
    if(plannerTarget())
    {
        // MN single launch, the pack hybrid, and the K-contiguous route (packed, and on the user
        // operands in place)
        std::set<std::string> kinds;
        for(auto const& r : routes)
        {
            kinds.insert(r.substr(0, r.find('/')));
            kinds.insert(r.substr(r.find('/') + 1));
        }
        EXPECT_TRUE(kinds.count("single") && kinds.count("pack_hybrid")
                    && kinds.count("tensor_gett_mn") && kinds.count("tensor_gett_kc"))
            << ::testing::PrintToString(routes);
        EXPECT_TRUE(routes.count("single/tensor_gett_kc") && routes.count("pack_hybrid/tensor_gett_kc"))
            << ::testing::PrintToString(routes);
    }
}

// Seeded random problems (HIPTENSOR_ROCKE_TEST_GETT_JIT_PROBLEMS, default 64; seed base
// HIPTENSOR_ROCKE_TEST_GETT_JIT_SEED): whatever route a plan takes, every output is exact; on
// gfx950 a share of them takes the JIT route, and its plans are exact with the scale and poison
// variants.
TEST(RockeGettJit, RandomProblemsExact)
{
    int const                  n    = envInt("HIPTENSOR_ROCKE_TEST_GETT_JIT_PROBLEMS", 64);
    uint64_t const             base = uint64_t(envInt("HIPTENSOR_ROCKE_TEST_GETT_JIT_SEED", 9100));
    int                        jit  = 0;
    std::map<std::string, int> routes;
    for(int i = 0; i < n; i++)
    {
        auto       p = randomProblem(base + uint64_t(i));
        RunOptions opt;
        opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0},
                        {"inf/nan operands", 2.0, -1.0, true, false, 0, 0},
                        {"alpha 0, A and B poisoned", 0.0, 3.0, true, false, 0, 0}};
        if(p.bilinear)
        {
            opt.variants.push_back({"beta 0, C poisoned", 2.0, 0.0, false, true, 0, 0});
        }
        auto o = runProblem(p, opt);
        ASSERT_TRUE(o.planned) << describe(p);
        EXPECT_TRUE(o.exact) << describe(p);
        jit += o.jit ? 1 : 0;
        routes[o.jit ? o.route : o.pick.substr(0, o.pick.find(':'))]++;
        std::printf("GETT-JIT-RANDOM %s pick=%s route=%s compiled=%d plan_ms=%.1f checked=%lld\n",
                    describe(p).c_str(),
                    o.pick.c_str(),
                    o.route.c_str(),
                    o.compiled,
                    o.planMs,
                    static_cast<long long>(o.checked));
        if(HasFatalFailure())
        {
            return;
        }
    }
    std::printf("GETT-JIT-RANDOM summary: %d of %d plans on the JIT route; %s\n",
                jit,
                n,
                ::testing::PrintToString(routes).c_str());
    if(plannerTarget() && n >= 16)
    {
        EXPECT_GE(jit, n / 4) << ::testing::PrintToString(routes);
    }
    else if(!plannerTarget())
    {
        EXPECT_EQ(jit, 0);
    }
}

// HIPTENSOR_JIT_MODE_NONE takes the pack's routes and never compiles (a fresh process: nothing in
// its kernel cache afterwards); HIPTENSOR_JIT_MODE_DEFAULT then compiles.
TEST(RockeGettJit, ModeNoneNeverCompiles)
{
    if(selfExecutable().empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "none");
    EXPECT_TRUE(runChild("RockeGettJitChild.ModeNone"));
}

TEST(RockeGettJitChild, ModeNone)
{
    char const* c = std::getenv("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD");
    if(c == nullptr || std::string(c) != "none")
    {
        GTEST_SKIP() << "run by RockeGettJit.ModeNoneNeverCompiles";
    }
    auto       p = fixedProblems()[0];
    LogCapture log;
    RunOptions opt;
    opt.jit      = HIPTENSOR_JIT_MODE_NONE;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    auto o       = runProblem(p, opt);
    EXPECT_TRUE(o.exact);
    EXPECT_FALSE(o.jit) << o.pick;
    EXPECT_FALSE(log.contains("rocKE GETT JIT")) << log.all();
    EXPECT_FALSE(log.contains("rocKE JIT: compiled")) << log.all();
    {
        ApiObjects api;
        auto       path = tempPath("none") + ".htk";
        ASSERT_EQ(hiptensorWriteKernelCacheToFile(api.handle, path.c_str()),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_TRUE(log.contains("wrote 0 entries")) << log.all();
        std::remove(path.c_str());
    }
    opt.jit = HIPTENSOR_JIT_MODE_DEFAULT;
    auto j  = runProblem(p, opt);
    EXPECT_TRUE(j.exact);
    if(plannerTarget())
    {
        EXPECT_TRUE(j.jit) << j.pick;
        EXPECT_GE(j.compiled, 1);
    }
}

// A second plan of a problem takes the kernel cache; so does a plan of the same memory layout
// described with other mode labels and another mode order (the JIT names the request
// canonically).
TEST(RockeGettJit, SecondPlanHitsTheCache)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    auto       p = fixedProblems()[2];
    RunOptions opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    auto first   = runProblem(p, opt);
    ASSERT_TRUE(first.jit) << first.pick;
    auto second = runProblem(p, opt);
    ASSERT_TRUE(second.jit) << second.pick;
    EXPECT_EQ(second.kernel, first.kernel);
    EXPECT_EQ(second.compiled, 0);
    EXPECT_EQ(second.hits, second.kernels);
    // relabelled (every name changes) and the descriptor modes listed in another order: the
    // strides keep the memory layout
    Problem q = p;
    q.name    = p.name + "_relabelled";
    std::map<std::string, std::string> rename;
    for(auto const& [n, e] : p.ext)
    {
        rename[n] = "z" + n;
    }
    q.ext.clear();
    for(auto const& [n, e] : p.ext)
    {
        q.ext[rename[n]] = e;
    }
    auto relist = [&](std::vector<std::string> const&       modes,
                      std::map<std::string, int64_t> const& st,
                      std::vector<std::string>&             out,
                      std::map<std::string, int64_t>&       outSt) {
        // packed strides of the original order, then the list reversed (another order)
        std::map<std::string, int64_t> packed;
        int64_t                        acc = 1;
        for(auto it = modes.rbegin(); it != modes.rend(); ++it)
        {
            packed[*it] = st.empty() ? acc : st.at(*it);
            acc *= p.ext.at(*it);
        }
        out.clear();
        outSt.clear();
        for(auto it = modes.rbegin(); it != modes.rend(); ++it)
        {
            out.push_back(rename[*it]);
            outSt[rename[*it]] = packed[*it];
        }
    };
    relist(p.a, p.sa, q.a, q.sa);
    relist(p.b, p.sb, q.b, q.sb);
    relist(p.d, p.sd, q.d, q.sd);
    auto third = runProblem(q, opt);
    ASSERT_TRUE(third.jit) << third.pick;
    EXPECT_TRUE(third.exact);
    EXPECT_EQ(third.kernel, first.kernel);
    EXPECT_EQ(third.compiled, 0);
}

// hiptensorWriteKernelCacheToFile in this process, hiptensorReadKernelCacheFromFile in a fresh one
// that may not compile: its JIT plans take every kernel from the file (exact results);
// HIPTENSOR_JIT_MODE_NONE plans there ignore the entries.
TEST(RockeGettJitCacheFile, RoundTripInAFreshProcess)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    if(selfExecutable().empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    RunOptions opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    int kernels  = 0;
    for(int i : {0, 2, 3})
    {
        auto o = runProblem(fixedProblems()[size_t(i)], opt);
        ASSERT_TRUE(o.jit) << o.pick;
        kernels += o.kernels;
    }
    ApiObjects api;
    auto       path = tempPath("roundtrip") + ".htk";
    ASSERT_EQ(hiptensorWriteKernelCacheToFile(api.handle, path.c_str()), HIPTENSOR_STATUS_SUCCESS);
    {
        ScopedEnvironmentVariable file("HIPTENSOR_ROCKE_TEST_GETT_JIT_FILE", path.c_str());
        ScopedEnvironmentVariable nocompile("HIPTENSOR_ROCKE_JIT_NO_COMPILE", "1");
        EXPECT_TRUE(runChild("RockeGettJitChild.ServesFromTheFile"));
    }
    std::remove(path.c_str());
}

TEST(RockeGettJitChild, ServesFromTheFile)
{
    char const* path = std::getenv("HIPTENSOR_ROCKE_TEST_GETT_JIT_FILE");
    if(path == nullptr)
    {
        GTEST_SKIP() << "run by RockeGettJitCacheFile.RoundTripInAFreshProcess";
    }
    {
        ApiObjects api;
        LogCapture log;
        ASSERT_EQ(hiptensorReadKernelCacheFromFile(api.handle, path), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_TRUE(log.contains("0 damaged")) << log.all();
    }
    RunOptions opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0},
                    {"misaligned workspace", 2.0, 1.0, false, false, 8, 0}};
    opt.jit      = HIPTENSOR_JIT_MODE_NONE;
    auto none    = runProblem(fixedProblems()[0], opt);
    EXPECT_FALSE(none.jit) << none.pick;
    EXPECT_TRUE(none.exact);
    opt.jit = HIPTENSOR_JIT_MODE_DEFAULT;
    for(int i : {0, 2, 3})
    {
        auto t0 = std::chrono::steady_clock::now();
        auto o  = runProblem(fixedProblems()[size_t(i)], opt);
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                      .count();
        EXPECT_TRUE(o.jit) << o.pick;
        EXPECT_TRUE(o.exact);
        EXPECT_EQ(o.compiled, 0);
        EXPECT_EQ(o.fileHits, o.kernels);
        std::printf(
            "GETT-JIT-FILE-HIT problem=%d kernels=%d jit_ms=%.3f plan_ms=%.3f case_ms=%.1f\n",
            i,
            o.kernels,
            o.jitMs,
            o.planMs,
            ms);
    }
}

#ifndef _WIN32
// HIPTENSOR_KERNEL_CACHE_DIR: a plan's compiles write one file per kernel there; a fresh process
// that may not compile and uses the same directory serves its plans from it.
TEST(RockeGettJitCacheDir, SharedAcrossProcesses)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    if(selfExecutable().empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    std::string dir = tempPath("dir");
    ASSERT_EQ(mkdir(dir.c_str(), 0700), 0);
    ScopedEnvironmentVariable cacheDir("HIPTENSOR_KERNEL_CACHE_DIR", dir.c_str());
    RunOptions                opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    // a problem no other test plans (its kernels are compiled here, not taken from memory)
    Problem p   = fixedProblems()[1];
    p.ext["M0"] = 80;
    p.name += "_dir";
    auto o = runProblem(p, opt);
    ASSERT_TRUE(o.jit) << o.pick;
    EXPECT_EQ(o.compiled, o.kernels);
    {
        ScopedEnvironmentVariable nocompile("HIPTENSOR_ROCKE_JIT_NO_COMPILE", "1");
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "dir");
        EXPECT_TRUE(runChild("RockeGettJitChild.ServesFromTheDirectory"));
    }
    // a spent compile budget still serves the directory's kernels (only compiling stops)
    {
        ScopedEnvironmentVariable budget("HIPTENSOR_JIT_MAX_MS", "0.000000001");
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "dir");
        EXPECT_TRUE(runChild("RockeGettJitChild.ServesFromTheDirectory"));
    }
    std::string rm = "rm -rf \"" + dir + "\"";
    EXPECT_EQ(std::system(rm.c_str()), 0);
}
#endif

TEST(RockeGettJitChild, ServesFromTheDirectory)
{
    char const* c = std::getenv("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD");
    if(c == nullptr || std::string(c) != "dir")
    {
        GTEST_SKIP() << "run by RockeGettJitCacheDir.SharedAcrossProcesses";
    }
    RunOptions opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    Problem p    = fixedProblems()[1];
    p.ext["M0"]  = 80;
    p.name += "_dir";
    auto o = runProblem(p, opt);
    EXPECT_TRUE(o.jit) << o.pick;
    EXPECT_TRUE(o.exact);
    EXPECT_EQ(o.compiled, 0);
    EXPECT_EQ(o.fileHits, o.kernels);
}

// A JIT route whose kernels cannot be built leaves the plan on the next candidate of the
// selection (same results): compiling disabled with an empty kernel cache, and a compile budget
// that is spent before the first kernel (fresh processes).
TEST(RockeGettJit, CompileFailureFallsBack)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    if(selfExecutable().empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    {
        ScopedEnvironmentVariable nocompile("HIPTENSOR_ROCKE_JIT_NO_COMPILE", "1");
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "fail");
        EXPECT_TRUE(runChild("RockeGettJitChild.FallsBack"));
    }
    {
        ScopedEnvironmentVariable budget("HIPTENSOR_JIT_MAX_MS", "0.000000001");
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "fail");
        EXPECT_TRUE(runChild("RockeGettJitChild.FallsBack"));
    }
}

TEST(RockeGettJitChild, FallsBack)
{
    char const* c = std::getenv("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD");
    if(c == nullptr || std::string(c) != "fail")
    {
        GTEST_SKIP() << "run by RockeGettJit.CompileFailureFallsBack";
    }
    LogCapture log;
    RunOptions opt;
    opt.variants = allVariants(true);
    for(int i : {0, 7})
    {
        auto o = runProblem(fixedProblems()[size_t(i)], opt);
        EXPECT_TRUE(o.planned);
        EXPECT_FALSE(o.jit) << o.pick;
        EXPECT_TRUE(o.exact);
        EXPECT_NE(o.source.find("+NEXT"), std::string::npos) << o.source;
        EXPECT_TRUE(log.contains("not built")) << log.all();
    }
}

// HIPTENSOR_ROCKE_DISABLE_TUNED (no route of rocKE's tables or planner) and
// HIPTENSOR_ROCKE_JIT=off (no JIT) keep contractions off the JIT route.
TEST(RockeGettJit, DisabledByKnobs)
{
    for(auto const& [name, value] :
        {std::pair{"HIPTENSOR_ROCKE_DISABLE_TUNED", "1"}, std::pair{"HIPTENSOR_ROCKE_JIT", "off"}})
    {
        ScopedEnvironmentVariable env(name, value);
        RunOptions                opt;
        opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
        auto o       = runProblem(fixedProblems()[0], opt);
        EXPECT_FALSE(o.jit) << name << " " << o.pick;
        EXPECT_TRUE(o.exact);
    }
}

#ifndef _WIN32
// The planner's selection models are data next to the pack (<target>/gemm_models with an index
// of SHA-256 digests): a pack directory without them, or with a model file that does not match
// its digest, has no JIT route (fresh processes; the plans take the other routes, exactly).
TEST(RockeGettJit, ModelsMissingOrDamaged)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    if(selfExecutable().empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    // the pack this library loads: a fresh process prints its path
    std::string pack;
    {
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "print");
        std::string               cmd
            = "\"" + selfExecutable() + "\" --gtest_filter=RockeGettJitChild.PrintsThePack";
        FILE* pipe = popen(cmd.c_str(), "r");
        ASSERT_NE(pipe, nullptr);
        std::string printed;
        char        buf[4096];
        while(std::fgets(buf, sizeof(buf), pipe) != nullptr)
        {
            printed += buf;
        }
        ASSERT_EQ(pclose(pipe), 0) << printed;
        std::string const tag = "ROCKE-PACK-PATH=";
        auto              at  = printed.find(tag);
        ASSERT_NE(at, std::string::npos) << printed;
        pack = printed.substr(at + tag.size());
        pack = pack.substr(0, pack.find('\n'));
    }
    auto const        slash  = pack.find_last_of('/');
    auto const        slash2 = pack.find_last_of('/', slash - 1);
    std::string const target = pack.substr(slash2 + 1, slash - slash2 - 1);
    std::string const models = pack.substr(0, slash) + "/gemm_models";
    std::string       dir    = tempPath("models");
    std::string cmd = "mkdir -p \"" + dir + "/" + target + "\" && cp \"" + pack + "\" \"" + dir
                      + "/" + target + "/\"";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    {
        ScopedEnvironmentVariable packPath("HIPTENSOR_ROCKE_PACK_PATH", dir.c_str());
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "nomodels");
        EXPECT_TRUE(runChild("RockeGettJitChild.NoJitRoute"));
    }
    // the models, one of them changed by a byte
    cmd = "cp -r \"" + models + "\" \"" + dir + "/" + target + "/\" && f=$(ls \"" + dir + "/"
          + target + "/gemm_models\"/*.model.txt | head -1) && printf 'x' | dd of=\"$f\" bs=1 "
          + "seek=100 conv=notrunc 2>/dev/null";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    {
        ScopedEnvironmentVariable packPath("HIPTENSOR_ROCKE_PACK_PATH", dir.c_str());
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "damaged");
        EXPECT_TRUE(runChild("RockeGettJitChild.NoJitRoute"));
    }
    // intact again: the route is back (the copy of the pack directory is complete)
    cmd = "rm -rf \"" + dir + "/" + target + "/gemm_models\" && cp -r \"" + models + "\" \"" + dir
          + "/" + target + "/\"";
    ASSERT_EQ(std::system(cmd.c_str()), 0);
    {
        ScopedEnvironmentVariable packPath("HIPTENSOR_ROCKE_PACK_PATH", dir.c_str());
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "intact");
        EXPECT_TRUE(runChild("RockeGettJitChild.NoJitRoute"));
    }
    cmd = "rm -rf \"" + dir + "\"";
    EXPECT_EQ(std::system(cmd.c_str()), 0);
}
#endif

TEST(RockeGettJitChild, PrintsThePack)
{
    char const* c = std::getenv("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD");
    if(c == nullptr || std::string(c) != "print")
    {
        GTEST_SKIP() << "run by RockeGettJit.ModelsMissingOrDamaged";
    }
    LogCapture log;
    RunOptions opt;
    opt.jit = HIPTENSOR_JIT_MODE_NONE;
    runProblem(fixedProblems()[0], opt);
    // "rocKE pack <path>: <n> kernels, pack id ..." (logged when the process loads the pack)
    for(auto const& l : log.lines())
    {
        auto at  = l.find("rocKE pack ");
        auto end = l.find(": ", at == std::string::npos ? 0 : at);
        if(at != std::string::npos && end != std::string::npos
           && l.find(" kernels, pack id ", end) != std::string::npos)
        {
            std::printf("ROCKE-PACK-PATH=%s\n", l.substr(at + 11, end - at - 11).c_str());
            return;
        }
    }
    ADD_FAILURE() << "no pack line\n" << log.all();
}

TEST(RockeGettJitChild, NoJitRoute)
{
    char const* c = std::getenv("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD");
    if(c == nullptr
       || (std::string(c) != "nomodels" && std::string(c) != "damaged"
           && std::string(c) != "intact"))
    {
        GTEST_SKIP() << "run by RockeGettJit.ModelsMissingOrDamaged";
    }
    LogCapture log;
    RunOptions opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    auto o       = runProblem(fixedProblems()[0], opt);
    EXPECT_TRUE(o.exact);
    if(std::string(c) == "intact")
    {
        EXPECT_TRUE(o.jit) << o.pick << "\n" << log.all();
        return;
    }
    EXPECT_FALSE(o.jit) << o.pick;
    EXPECT_TRUE(log.contains(std::string(c) == "nomodels" ? "has no GEMM selection models"
                                                          : "does not match its digest"))
        << log.all();
}

// A JIT plan executed on another device than the one it was created on (its kernels are loaded
// there only) runs its fallback, the direct GETT, with the same results (needs two devices of the
// architecture).
TEST(RockeGettJit, OtherDeviceRunsTheFallback)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    int const other = otherDevice();
    if(other < 0)
    {
        GTEST_SKIP() << "needs a second device of this architecture";
    }
    for(int i : {0, 2, 7})
    {
        RunOptions opt;
        opt.variants   = allVariants(true);
        opt.execDevice = other;
        auto o         = runProblem(fixedProblems()[size_t(i)], opt);
        EXPECT_TRUE(o.jit) << o.pick;
        EXPECT_TRUE(o.exact);
    }
}

// ------------------------------------------------------------------ selection with models

namespace
{
    // A selection model file <dir>/<target>/<cell>.model for HIPTENSOR_ROCKE_SELECT_MODELS (a
    // new directory per name: the library reads a model once per process and source).
    std::string
        writeSelectModel(std::string const& name, std::string const& cell, std::string const& text)
    {
        std::string const dir = tempPath("select_models_" + name);
        std::string const sub = dir + "/" + deviceArch();
        EXPECT_EQ(std::system(("mkdir -p \"" + sub + "\"").c_str()), 0);
        std::ofstream out(sub + "/" + cell + ".model");
        out << text << "\n";
        return dir;
    }

    // One tree on the feature "route": 0 for `route`, 1 for every other route.
    std::string favourRoute(int route)
    {
        char tree[96];
        std::snprintf(tree, sizeof(tree), "3;0 0;%d.5 %d.5;-1 -2;1 -3;1 0 1", route - 1, route);
        return tree;
    }

    // The dump lines (HIPTENSOR_ROCKE_SELECT_DUMP) of `tag` whose candidate key starts with
    // `prefix`.
    std::vector<std::string>
        dumpLines(std::string const& path, std::string const& tag, std::string const& prefix)
    {
        std::vector<std::string> out;
        std::ifstream            in(path);
        std::string              line;
        while(std::getline(in, line))
        {
            if(line.find("\"tag\":\"" + tag + "\"") != std::string::npos
               && line.find("\"key\":\"" + prefix) != std::string::npos)
            {
                out.push_back(line);
            }
        }
        return out;
    }
}

// A selection model that scores the JIT route (format v2, route 11) ranks it with the other
// candidates: picked when it predicts it fastest, not picked when it predicts another route
// fastest; a model that does not score it leaves the rule (the planner's route first).
TEST(RockeGettJitSelection, ModelScoresTheJitRoute)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    ScopedEnvironmentVariable select("HIPTENSOR_ROCKE_SELECT", nullptr);
    auto const                p = fixedProblems()[0];
    RunOptions                opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    auto with    = [&](std::string const& name, std::string const& text) {
        ScopedEnvironmentVariable models("HIPTENSOR_ROCKE_SELECT_MODELS",
                                         writeSelectModel(name, "contraction.h", text).c_str());
        return runProblem(p, opt);
    };
    auto const head = std::string("v2;schema=2;routes=3,6,7,8,11;margin=0;features=route;trees=1|");
    auto       jit  = with("jit", head + favourRoute(11));
    EXPECT_TRUE(jit.jit) << jit.pick;
    EXPECT_EQ(jit.source, "MODEL") << jit.reason;
    EXPECT_TRUE(jit.exact);
    auto direct = with("direct", head + favourRoute(8));
    EXPECT_FALSE(direct.jit) << direct.pick;
    EXPECT_EQ(direct.pick.rfind("dgett:", 0), 0u) << direct.pick;
    EXPECT_EQ(direct.source, "MODEL") << direct.reason;
    EXPECT_TRUE(direct.exact);
    // no compile for a JIT candidate the selection did not pick
    EXPECT_EQ(direct.kernels, 0);
    auto rule = with(
        "rule", "v2;schema=2;routes=3,6,7,8;margin=0;features=route;trees=1|" + favourRoute(8));
    EXPECT_TRUE(rule.jit) << rule.pick;
    EXPECT_EQ(rule.source, "TUNED");
    EXPECT_EQ(rule.reason, "rocKE's tensor GETT planner (JIT)");
    EXPECT_TRUE(rule.exact);
}

// The shipped models of the planner's target score the JIT route with the other routes: the
// plans' selection source is the model, and every JIT candidate's dump line carries a score.
TEST(RockeGettJitSelection, ShippedModelScoresTheJitRoute)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    ScopedEnvironmentVariable select("HIPTENSOR_ROCKE_SELECT", nullptr);
    std::string const         dump = tempPath("shipped_dump.jsonl");
    std::remove(dump.c_str());
    ScopedEnvironmentVariable out("HIPTENSOR_ROCKE_SELECT_DUMP", dump.c_str());
    ScopedEnvironmentVariable tag("HIPTENSOR_ROCKE_SELECT_DUMP_TAG", "shipped");
    RunOptions                opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    int jitPlans = 0;
    for(size_t i : {0, 1, 2, 16})
    {
        auto o = runProblem(fixedProblems()[i], opt);
        EXPECT_EQ(o.source, "MODEL") << o.pick << " (" << o.reason << ")";
        EXPECT_EQ(o.reason.rfind("contraction.h@", 0), 0u) << o.reason;
        EXPECT_TRUE(o.exact);
        jitPlans += o.jit ? 1 : 0;
    }
    auto const lines = dumpLines(dump, "shipped", "jit:");
    EXPECT_EQ(lines.size(), 4u);
    for(auto const& l : lines)
    {
        EXPECT_NE(l.find("\"source\":\"MODEL\""), std::string::npos) << l;
        EXPECT_NE(l.find("\"score\":"), std::string::npos) << l;
        EXPECT_EQ(l.find("\"us\":"), std::string::npos) << l;
    }
    std::printf("shipped model: %d of 4 plans took the JIT route\n", jitPlans);
    std::remove(dump.c_str());
}

// A JIT candidate has the same features before its launches are built (planned, not compiled)
// and after (built and timed by the measured selection), so the model ranks it the same way in
// either state.
TEST(RockeGettJitSelection, FeaturesBeforeAndAfterTheBuild)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    // a problem no other test plans: its kernels are not in this process's kernel cache
    Problem const     p{"features_f16",
                    HIPTENSOR_R_16F,
                    true,
                        {"K0", "M0"},
                        {"K0", "N0"},
                        {"N0", "M0"},
                        {{"K0", 320}, {"M0", 208}, {"N0", 176}}};
    std::string const dump = tempPath("features_dump.jsonl");
    std::remove(dump.c_str());
    ScopedEnvironmentVariable out("HIPTENSOR_ROCKE_SELECT_DUMP", dump.c_str());
    RunOptions                opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    {
        ScopedEnvironmentVariable tag("HIPTENSOR_ROCKE_SELECT_DUMP_TAG", "pending");
        ScopedEnvironmentVariable nocompile("HIPTENSOR_ROCKE_JIT_NO_COMPILE", "1");
        LogCapture                log;
        auto                      o = runProblem(p, opt);
        EXPECT_FALSE(o.jit) << o.pick;
        EXPECT_TRUE(log.contains("not built")) << log.all();
    }
    {
        ScopedEnvironmentVariable tag("HIPTENSOR_ROCKE_SELECT_DUMP_TAG", "built");
        ScopedEnvironmentVariable topk("HIPTENSOR_ROCKE_AUTOTUNE_TOPK", "all");
        RunOptions                patient = opt;
        patient.algo                      = HIPTENSOR_ALGO_DEFAULT_PATIENT;
        auto o                            = runProblem(p, patient);
        EXPECT_TRUE(o.exact);
    }
    auto const pending = dumpLines(dump, "pending", "jit:");
    auto const built   = dumpLines(dump, "built", "jit:");
    ASSERT_EQ(pending.size(), 1u) << "no JIT candidate";
    ASSERT_EQ(built.size(), 1u) << "the JIT candidate was not timed";
    EXPECT_EQ(built[0].find("\"us\":") != std::string::npos, true) << built[0];
    auto features = [](std::string const& l) { return l.substr(l.find("\"f\":{")); };
    EXPECT_EQ(features(pending[0]), features(built[0]));
    // the planned launches and the planner's route are features of the candidate
    EXPECT_EQ(features(pending[0]).find("\"gett_kind\":0"), std::string::npos) << pending[0];
    EXPECT_EQ(features(pending[0]).find("\"steps\":0"), std::string::npos) << pending[0];
    EXPECT_EQ(features(pending[0]).find("\"lg_pred_ns\":0"), std::string::npos) << pending[0];
    // schema 3: the bytes the route moves (packs, operands, D, C)
    EXPECT_EQ(features(pending[0]).find("\"lg_bytes\":0"), std::string::npos) << pending[0];
    EXPECT_EQ(features(pending[0]).find("\"intensity_q8\":0"), std::string::npos) << pending[0];
    std::remove(dump.c_str());
}

// The plan's mode is a feature of every candidate (schema 3, jit_mode): 1 when the plan
// specializes its elementwise launches with the JIT (HIPTENSOR_JIT_MODE_DEFAULT), 0 with
// HIPTENSOR_JIT_MODE_NONE and with the JIT turned off (HIPTENSOR_ROCKE_JIT=off); the candidates'
// times differ between the modes (the GEMM routes' packing and epilogue launches).
TEST(RockeGettJitSelection, PlanModeIsAFeature)
{
    std::string const dump = tempPath("mode_dump.jsonl");
    std::remove(dump.c_str());
    ScopedEnvironmentVariable out("HIPTENSOR_ROCKE_SELECT_DUMP", dump.c_str());
    auto const                p = fixedProblems()[1];
    RunOptions                opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    struct Mode
    {
        char const*        tag;
        hiptensorJitMode_t jit;
        char const*        env; // HIPTENSOR_ROCKE_JIT
        char const*        want;
    };
    for(auto const& m : {Mode{"mode_none", HIPTENSOR_JIT_MODE_NONE, nullptr, "\"jit_mode\":0"},
                         Mode{"mode_default", HIPTENSOR_JIT_MODE_DEFAULT, nullptr, "\"jit_mode\":1"},
                         Mode{"mode_off", HIPTENSOR_JIT_MODE_DEFAULT, "off", "\"jit_mode\":0"}})
    {
        ScopedEnvironmentVariable tag("HIPTENSOR_ROCKE_SELECT_DUMP_TAG", m.tag);
        ScopedEnvironmentVariable jit("HIPTENSOR_ROCKE_JIT", m.env);
        opt.jit = m.jit;
        auto o  = runProblem(p, opt);
        EXPECT_TRUE(o.exact) << m.tag;
        auto const lines = dumpLines(dump, m.tag, "");
        ASSERT_FALSE(lines.empty()) << m.tag;
        for(auto const& l : lines)
        {
            EXPECT_NE(l.find(m.want), std::string::npos) << m.tag << ": " << l;
        }
    }
    std::remove(dump.c_str());
}

// A v3 selection model that reads the planner's predicted time (lg_pred_ns) names the digest of
// the planner's GEMM selection models it was trained with (the SHA-256 of
// <pack dir>/gemm_models/index.txt): with the pack's digest it ranks the candidates; with another
// digest (or none) it is not used for a problem with a JIT candidate (the rules then put the
// planner's route first) and the plan log says why.
TEST(RockeGettJitSelection, ModelNamesThePlannersModels)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    if(selfExecutable().empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    ScopedEnvironmentVariable select("HIPTENSOR_ROCKE_SELECT", nullptr);
    // the pack's planner models: their index's digest
    std::string pack;
    {
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_GETT_JIT_CHILD", "print");
        FILE* pipe = popen(("\"" + selfExecutable()
                            + "\" --gtest_filter=RockeGettJitChild.PrintsThePack")
                               .c_str(),
                           "r");
        ASSERT_NE(pipe, nullptr);
        std::string printed;
        char        buf[4096];
        while(std::fgets(buf, sizeof(buf), pipe) != nullptr)
        {
            printed += buf;
        }
        ASSERT_EQ(pclose(pipe), 0) << printed;
        auto const at = printed.find("ROCKE-PACK-PATH=");
        ASSERT_NE(at, std::string::npos) << printed;
        pack = printed.substr(at + 16);
        pack = pack.substr(0, pack.find('\n'));
    }
    std::string const index = pack.substr(0, pack.find_last_of('/')) + "/gemm_models/index.txt";
    FILE*             sum   = popen(("sha256sum \"" + index + "\"").c_str(), "r");
    ASSERT_NE(sum, nullptr);
    char digest[65] = {};
    ASSERT_EQ(std::fscanf(sum, "%64s", digest), 1);
    ASSERT_EQ(pclose(sum), 0) << index;

    auto const p = fixedProblems()[0];
    RunOptions opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    auto with    = [&](std::string const& name, std::string const& gemm) {
        // the direct GETT (8) predicted fastest; lg_pred_ns read (a tree that never splits on it)
        auto const text = "v3;schema=3;routes=3,6,7,8,11;margin=0;gemm=" + gemm
                          + ";features=route,lg_pred_ns;trees=1|" + favourRoute(8);
        ScopedEnvironmentVariable models("HIPTENSOR_ROCKE_SELECT_MODELS",
                                         writeSelectModel(name, "contraction.h", text).c_str());
        LogCapture                log;
        auto                      o = runProblem(p, opt);
        return std::make_pair(o, log.last("rocKE selection:"));
    };
    auto [same, sameLine] = with("gemm_same", digest);
    EXPECT_EQ(same.source, "MODEL") << sameLine;
    EXPECT_EQ(same.pick.rfind("dgett:", 0), 0u) << sameLine;
    EXPECT_TRUE(same.exact);
    for(auto const& [name, gemm] : {std::pair<std::string, std::string>{"gemm_other", std::string(64, 'a')},
                                     {"gemm_none", "-"}})
    {
        auto [o, line] = with(name, gemm);
        EXPECT_TRUE(o.jit) << name << ": " << line;
        EXPECT_EQ(o.source, "TUNED") << name << ": " << line;
        EXPECT_NE(line.find("trained with other GEMM selection models"), std::string::npos)
            << name << ": " << line;
        EXPECT_TRUE(o.exact);
    }
}

// The JIT candidate's bytes moved are a feature from schema 3 (lg_bytes, intensity_q8); a model
// of schema 2 sees them as that schema had them (0). A model on lg_bytes alone (no candidate moves
// 0 bytes but the JIT candidate under schema 2): with schema 2 the JIT candidate is predicted
// worst and not picked; with schema 3 every candidate ties and the rules' order (the planner's
// route first) stands.
TEST(RockeGettJitSelection, EarlierSchemaSeesItsFeatures)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    ScopedEnvironmentVariable select("HIPTENSOR_ROCKE_SELECT", nullptr);
    auto const                p = fixedProblems()[0];
    RunOptions                opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    // lg_bytes <= 0.5: 1 (worst), else 0
    std::string const tree = "2;0;0.5;-1;-2;1 0";
    auto              with = [&](std::string const& name, int schema) {
        auto const text = "v3;schema=" + std::to_string(schema)
                          + ";routes=3,6,7,8,11;margin=0;gemm=-;features=lg_bytes;trees=1|"
                          + tree;
        ScopedEnvironmentVariable models("HIPTENSOR_ROCKE_SELECT_MODELS",
                                         writeSelectModel(name, "contraction.h", text).c_str());
        return runProblem(p, opt);
    };
    auto two = with("schema2_bytes", 2);
    EXPECT_EQ(two.source, "MODEL") << two.reason;
    EXPECT_FALSE(two.jit) << two.pick;
    EXPECT_TRUE(two.exact);
    auto three = with("schema3_bytes", 3);
    EXPECT_EQ(three.source, "MODEL") << three.reason;
    EXPECT_TRUE(three.jit) << three.pick;
    EXPECT_TRUE(three.exact);
}

// The JIT candidate's route feature (gett_kind) names the route rocKE's planner chose: 1 MN-layout
// GETT, 2 MN-layout GETT after tensor_pack copies, 3 K-contiguous GETT after copies, 4 ring body,
// 5 K-contiguous GETT on the operands in place.
TEST(RockeGettJitSelection, RouteFeatureNamesThePlannersRoute)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    std::string const dump = tempPath("route_dump.jsonl");
    std::remove(dump.c_str());
    ScopedEnvironmentVariable        out("HIPTENSOR_ROCKE_SELECT_DUMP", dump.c_str());
    std::map<std::string, int> const kinds = {{"single/tensor_gett_mn", 1},
                                              {"pack_hybrid/tensor_gett_mn", 2},
                                              {"pack_hybrid/tensor_gett_kc", 3},
                                              {"single/tensor_gett_t32s3", 4},
                                              {"pack_hybrid/tensor_gett_t32s3", 4},
                                              {"single/tensor_gett_kc", 5}};
    std::set<int>                    seen;
    RunOptions                       opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    // the fixed problems, and long contractions with both operands K-contiguous (the K-contiguous
    // route on the operands in place)
    auto problems = fixedProblems();
    for(auto type : {HIPTENSOR_R_16F, HIPTENSOR_R_16BF})
    {
        problems.push_back({type == HIPTENSOR_R_16F ? "kc_in_place_f16" : "kc_in_place_bf16",
                            type,
                            false,
                            {"M0", "K0"},
                            {"N0", "K0"},
                            {"M0", "N0"},
                            {{"K0", 8192}, {"M0", 256}, {"N0", 384}}});
    }
    for(auto const& p : problems)
    {
        ScopedEnvironmentVariable tag("HIPTENSOR_ROCKE_SELECT_DUMP_TAG", p.name.c_str());
        auto                      o = runProblem(p, opt);
        if(!o.jit)
        {
            continue;
        }
        auto const lines = dumpLines(dump, p.name, "jit:");
        ASSERT_EQ(lines.size(), 1u) << p.name;
        auto const it = kinds.find(o.route);
        ASSERT_NE(it, kinds.end()) << o.route;
        EXPECT_NE(lines[0].find("\"gett_kind\":" + std::to_string(it->second) + ","),
                  std::string::npos)
            << o.route << ": " << lines[0];
        seen.insert(it->second);
    }
    // the fixed problems take MN-layout, pack hybrid, K-contiguous (packed and in place) and
    // ring-body routes
    EXPECT_GE(seen.size(), 4u);
    EXPECT_TRUE(seen.count(5)) << "no plan took the K-contiguous GETT on the operands in place";
    std::remove(dump.c_str());
}

// The workspace estimate of a JIT plan's preference includes the JIT route's workspace (the
// operand packs), and MIN takes the least of the candidates.
TEST(RockeGettJit, WorkspaceEstimate)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    auto       p = fixedProblems()[2]; // both operands packed first
    RunOptions opt;
    opt.variants = {{"plain", 1.0, -2.0, false, false, 0, 0}};
    auto o       = runProblem(p, opt);
    ASSERT_TRUE(o.jit) << o.pick;
    EXPECT_GT(o.workspace, 0u);
}

// ------------------------------------------------------------------ the boundary to rocKE

namespace
{
    // The record of a route table entry's plan as rocKE's engine plans it, in the format of the
    // pack's GETT route records (pack/rocke_pack_builder.cpp gettPlanDescriptor) and of the
    // library's "rocKE GETT JIT plan" log line.
    std::string tableRecord(rocke::gett::TablePlan const& plan)
    {
        auto launch = [](std::string const&                        kernel,
                         int const*                                grid,
                         int                                       block,
                         std::map<std::string, std::string> const& binds) {
            std::string out = "|k=" + kernel + ";g=" + std::to_string(grid[0]) + ","
                              + std::to_string(grid[1]) + "," + std::to_string(grid[2])
                              + ";b=" + std::to_string(block) + ";a=";
            bool first = true;
            for(auto const& [k, v] : binds)
            {
                out += (first ? "" : ",") + k + "=" + v;
                first = false;
            }
            return out;
        };
        uint64_t              off = 0;
        std::vector<uint64_t> offsets;
        std::string           packs;
        for(auto const& s : plan.packs)
        {
            packs += launch(s.spec.kernel_name(),
                            s.grid,
                            s.block,
                            {{"X", s.operand + "+0"}, {"Y", "W+" + std::to_string(off)}});
            offsets.push_back(off);
            off += (uint64_t(s.workspace_bytes) + 255) / 256 * 256;
        }
        rocke::gett::Plan pl;
        std::string       why;
        if(!plan.gett.plan(&pl, &why))
        {
            return "no plan: " + why;
        }
        std::map<std::string, std::string> binds = {{"C", "O+0"},
                                                    {"M", "#" + std::to_string(pl.M)},
                                                    {"N", "#" + std::to_string(pl.N)},
                                                    {"K", "#" + std::to_string(pl.K)},
                                                    {"alpha", "@a"}};
        if(plan.gett.knobs.bilinear)
        {
            binds["Cin"]  = "C+0";
            binds["beta"] = "@b";
        }
        int src[2];
        rocke::gett::gett_operand_sources(plan, src);
        auto const ops = rocke::gett::launch_operands(plan.gett);
        binds["A"] = src[0] >= 0 ? "W+" + std::to_string(offsets[size_t(src[0])]) : ops.first + "+0";
        binds["B"]
            = src[1] >= 0 ? "W+" + std::to_string(offsets[size_t(src[1])]) : ops.second + "+0";
        int grid[3];
        rocke::gett::tensor_gett_grid(plan.gett, grid);
        return "v1;ws=" + std::to_string(off) + ";steps=" + std::to_string(plan.packs.size() + 1)
               + packs
               + launch(plan.gett.kernel_name(), grid, rocke::gett::tensor_gett_block(plan.gett), binds);
    }

    // The entry with its modes named as the request of the same memory layout is named
    // canonically: x0, x1, ... by first appearance in D, A, B, each tensor from its fastest mode
    // (the entries are packed row-major: the last mode is the fastest).
    rocke::gett::TunedEntry canonicalEntry(rocke::gett::TunedEntry const& e)
    {
        std::map<std::string, std::string> names;
        for(auto const* modes : {&e.modes_d, &e.modes_a, &e.modes_b})
        {
            for(auto it = modes->rbegin(); it != modes->rend(); ++it)
            {
                if(!names.count(*it))
                {
                    names[*it] = "x" + std::to_string(names.size());
                }
            }
        }
        rocke::gett::TunedEntry out = e;
        out.extents.clear();
        for(auto const& [m, x] : e.extents)
        {
            out.extents[names.at(m)] = x;
        }
        for(auto* modes : {&out.modes_a, &out.modes_b, &out.modes_d})
        {
            for(auto& m : *modes)
            {
                m = names.at(m);
            }
        }
        return out;
    }
} // namespace

// Every problem of rocKE's tuned and AOT GETT route tables, described to hipTensor (column-major
// descriptors of the entry's packed layout; per problem one of: the plain description, other mode
// labels, the descriptor modes listed in another order with their strides, an extent-1 mode in
// every tensor), and planned by the contraction JIT with the tables off: the logged launches are
// exactly the ones rocKE's C++ engine plans for the entry (table_plan; rocKE's tests check that
// they are Python's and that the table entries are rocKE's product routes), with the modes named
// as the planner request names them. So hipTensor's request of a contraction is the request
// rocKE's pack builder plans for it: the same modes, order, extents and strides, and the same
// decision of the planner (a tuned entry found by its key, an AOT entry re-planned).
TEST(RockeGettJitBoundary, TablesPlanLikeRocke)
{
    if(!plannerTarget())
    {
        GTEST_SKIP() << "the planner serves gfx950";
    }
    int             device = 0;
    hipDeviceProp_t props;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
    // the tables' entries were planned for the target's CU count (rocKE's arch_num_cus)
    if(props.multiProcessorCount != 256)
    {
        GTEST_SKIP() << "the route tables were planned for 256 CUs, the device has "
                     << props.multiProcessorCount;
    }
    ScopedEnvironmentVariable tables("HIPTENSOR_ROCKE_GETT_TABLES", "off");
    ApiObjects                api;
    std::mt19937_64           rng(77);
    int                       compared = 0, tunedEntries = 0;
    for(auto const* json : {rocke_tensor_gett_tuned_json(), rocke_tensor_gett_aot_json()})
    {
        std::vector<rocke::gett::TunedEntry> entries;
        std::string                          why;
        ASSERT_TRUE(rocke::gett::tuned_entries(json, "gfx950", &entries, &why)) << why;
        ASSERT_FALSE(entries.empty());
        tunedEntries += json == rocke_tensor_gett_tuned_json() ? int(entries.size()) : 0;
        for(auto const& entry : entries)
        {
            auto const              canon = canonicalEntry(entry);
            rocke::gett::TablePlan  plan;
            ASSERT_TRUE(rocke::gett::table_plan(canon, "gfx950", &plan, &why))
                << entry.key << ": " << why;
            auto const want    = tableRecord(plan);
            int const  variant = compared % 4;
            // the entry's tensors as hipTensor descriptors: modes reversed (column-major), packed
            // strides; mode ids: a fixed map, or (variant 1) other labels
            std::map<std::string, int32_t> ids;
            int32_t                        next = variant == 1 ? 900 + int32_t(rng() % 50) : 10;
            for(auto const& [m, x] : entry.extents)
            {
                ids[m] = next;
                next += variant == 1 ? 3 + int32_t(rng() % 5) : 1;
            }
            struct Desc
            {
                std::vector<int32_t> modes;
                std::vector<int64_t> extents, strides;
            };
            auto describeTensor = [&](std::vector<std::string> const& rowMajor) {
                Desc    d;
                int64_t s = 1;
                for(auto it = rowMajor.rbegin(); it != rowMajor.rend(); ++it)
                {
                    d.modes.push_back(ids.at(*it));
                    d.extents.push_back(entry.extents.at(*it));
                    d.strides.push_back(s);
                    s *= entry.extents.at(*it);
                }
                if(variant == 2)
                {
                    // the same layout listed in another order
                    std::vector<size_t> order(d.modes.size());
                    std::iota(order.begin(), order.end(), size_t(0));
                    std::shuffle(order.begin(), order.end(), rng);
                    Desc o;
                    for(auto i : order)
                    {
                        o.modes.push_back(d.modes[i]);
                        o.extents.push_back(d.extents[i]);
                        o.strides.push_back(d.strides[i]);
                    }
                    d = o;
                }
                return d;
            };
            auto A = describeTensor(entry.modes_a), B = describeTensor(entry.modes_b),
                 D = describeTensor(entry.modes_d);
            if(variant == 3)
            {
                // extent-1 modes: an M mode (A and D, at the front, stride 7) and an N mode (B
                // and D, at the back, a stride past the span); a mode of A, B and D would be a
                // batch mode, which the API refuses
                auto unit = [](Desc& d, int32_t mode, bool front, int64_t stride) {
                    auto at = front ? 0 : d.modes.size();
                    d.modes.insert(d.modes.begin() + long(at), mode);
                    d.extents.insert(d.extents.begin() + long(at), 1);
                    d.strides.insert(d.strides.begin() + long(at), stride);
                };
                // (the API admits at most six modes per group)
                auto count = [](std::vector<std::string> const& x, std::vector<std::string> const& y) {
                    return std::count_if(x.begin(), x.end(), [&y](std::string const& m) {
                        return std::find(y.begin(), y.end(), m) != y.end();
                    });
                };
                if(count(entry.modes_a, entry.modes_d) < 6)
                {
                    unit(A, 5, true, 7);
                    unit(D, 5, true, 7);
                }
                if(count(entry.modes_b, entry.modes_d) < 6)
                {
                    unit(B, 6, false, int64_t(1) << 20);
                    unit(D, 6, false, int64_t(1) << 20);
                }
            }
            auto const type = entry.dtype == "bf16" ? HIPTENSOR_R_16BF : HIPTENSOR_R_16F;
            auto       dA   = api.tensor(A.extents, A.strides, type);
            auto       dB   = api.tensor(B.extents, B.strides, type);
            auto       dD   = api.tensor(D.extents, D.strides, type);
            hiptensorOperationDescriptor_t desc = nullptr;
            ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                                 &desc,
                                                 dA,
                                                 A.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dB,
                                                 B.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 entry.bilinear ? dD : nullptr,
                                                 entry.bilinear ? D.modes.data() : nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dD,
                                                 D.modes.data(),
                                                 HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS)
                << entry.key;
            api.keep(desc);
            hiptensorPlanPreference_t pref = nullptr;
            ASSERT_EQ(hiptensorCreatePlanPreference(
                          api.handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_DEFAULT),
                      HIPTENSOR_STATUS_SUCCESS);
            LogCapture log;
            uint64_t   estimate = 0;
            // the estimate plans every candidate (the planner runs; nothing is compiled)
            EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                          api.handle, desc, pref, HIPTENSOR_WORKSPACE_MAX, &estimate),
                      HIPTENSOR_STATUS_SUCCESS);
            static_cast<void>(hiptensorDestroyPlanPreference(pref));
            auto const line = log.last("rocKE GETT JIT plan ");
            ASSERT_FALSE(line.empty()) << entry.key << " (variant " << variant << ")\n"
                                       << log.all();
            auto got = line.substr(line.find(": ", line.find("rocKE GETT JIT plan ")) + 2);
            while(!got.empty() && (got.back() == '\n' || got.back() == '\r' || got.back() == ' '))
            {
                got.pop_back();
            }
            EXPECT_EQ(got, want) << entry.key << " (" << entry.route << ", variant " << variant
                                 << ")";
            // no table candidate with the tables off
            EXPECT_EQ(log.all().find("tuned:"), std::string::npos) << entry.key;
            compared++;
            if(HasFatalFailure())
            {
                return;
            }
        }
    }
    std::printf("GETT-JIT-BOUNDARY %d table problems (%d tuned) planned as rocKE plans them\n",
                compared,
                tunedEntries);
    EXPECT_GE(compared, 200);
}

// The JIT's costs per problem (fixed problems and HIPTENSOR_ROCKE_TEST_GETT_JIT_LATENCY_PROBLEMS
// random ones): a first plan's time (planner, build and COMGR compile of the route's kernels; the
// first plan of the process also loads the selection models, logged on their own line), a second
// plan's (kernel cache hits), and the plan time of HIPTENSOR_JIT_MODE_NONE. Run with COMGR's own
// cache off (AMD_COMGR_CACHE=0) for cold compile times. Kernel times are measured by the
// enablement benchmarks (cold caches), not here.
TEST(RockeGettJitLatency, DISABLED_Report)
{
    int const            n    = envInt("HIPTENSOR_ROCKE_TEST_GETT_JIT_LATENCY_PROBLEMS", 24);
    uint64_t const       base = uint64_t(envInt("HIPTENSOR_ROCKE_TEST_GETT_JIT_SEED", 9100));
    std::vector<Problem> problems;
    if(envInt("HIPTENSOR_ROCKE_TEST_GETT_JIT_LATENCY_FIXED", 1) != 0)
    {
        problems = fixedProblems();
    }
    for(int i = 0; i < n; i++)
    {
        problems.push_back(randomProblem(base + uint64_t(i)));
    }
    LogCapture models;
    for(auto const& p : problems)
    {
        ApiObjects api;
        RunOptions opt;
        opt.api          = &api;
        opt.samples      = 64;
        opt.checkPadding = false;
        opt.variants     = {{"plain", 1.0, -2.0, false, false, 0, 0}};
        auto cold        = runProblem(p, opt);
        auto warm        = runProblem(p, opt);
        opt.jit          = HIPTENSOR_JIT_MODE_NONE;
        auto none        = runProblem(p, opt);
        std::printf("GETT-JIT-LATENCY %s | jit=%d route=%s kernels=%d compiled=%d "
                    "cold_plan_ms=%.2f cold_jit_ms=%.2f warm_plan_ms=%.3f warm_jit_ms=%.3f "
                    "hits=%d none_pick=%s none_plan_ms=%.3f exact=%d/%d/%d\n",
                    describe(p).c_str(),
                    cold.jit ? 1 : 0,
                    cold.route.c_str(),
                    cold.kernels,
                    cold.compiled,
                    cold.planMs,
                    cold.jitMs,
                    warm.planMs,
                    warm.jitMs,
                    warm.hits,
                    none.pick.c_str(),
                    none.planMs,
                    cold.exact ? 1 : 0,
                    warm.exact ? 1 : 0,
                    none.exact ? 1 : 0);
    }
    std::printf("GETT-JIT-MODELS %s\n", models.last("selection models").c_str());
}
