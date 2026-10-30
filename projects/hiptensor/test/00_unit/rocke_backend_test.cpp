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

// The rocKE backend through the public API (HIPTENSOR_BACKEND=rocke, no fallback): contractions
// of every data type (also of mixed data types), trinary contractions, permutations, elementwise
// binary and trinary operations and reductions with random mode orders, extents and padded
// strides, executed on a non-blocking stream, and tensors of 2 GiB and more.
//
// Every result is compared with a plain loop reference over all indices in double precision
// (complex double), not with the library's CPU reference:
//  - integer-valued data in a small range keep every intermediate exact in the compute type, so
//    the result must equal the reference rounded once to the output type, bit for bit;
//  - random real data are compared with an error bound of the compute type's accumulation and
//    one rounding to the output type.
// The plan log must name the rocKE backend for every case.

#include <gtest/gtest.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "api_test_helpers.hpp"

namespace
{
    using hiptensor_test::ApiObjects;
    using cplx = std::complex<double>;

    // Selects the rocKE backend without fallback for the whole test program.
    struct RockeOnly : ::testing::Environment
    {
        void SetUp() override
        {
#ifdef _WIN32
            _putenv_s("HIPTENSOR_BACKEND", "rocke");
            _putenv_s("HIPTENSOR_BACKEND_FALLBACK", "0");
#else
            setenv("HIPTENSOR_BACKEND", "rocke", 1);
            setenv("HIPTENSOR_BACKEND_FALLBACK", "0", 1);
#endif
        }
    };
    auto const* const gEnvironment = ::testing::AddGlobalTestEnvironment(new RockeOnly);

    // The JIT lines of the whole program ("rocKE JIT: <s> of <n> launches specialized ...;
    // <first decline reason>"): plans, launches, specialized launches, declines by reason.
    struct JitTotals
    {
        int64_t                        plans = 0, launches = 0, specialized = 0;
        std::map<std::string, int64_t> declines;
    };

    class LogCapture
    {
    public:
        LogCapture()
        {
            std::scoped_lock lock(sMutex);
            sLines.clear();
            hiptensorLoggerSetCallback(&LogCapture::callback);
            hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE
                                   | HIPTENSOR_LOG_LEVEL_ERROR);
        }
        ~LogCapture()
        {
            hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_OFF);
            hiptensorLoggerSetCallback(nullptr);
        }
        bool contains(std::string const& text) const
        {
            std::scoped_lock lock(sMutex);
            return std::any_of(sLines.begin(), sLines.end(), [&text](auto const& line) {
                return line.find(text) != std::string::npos;
            });
        }
        std::string all() const
        {
            std::scoped_lock lock(sMutex);
            std::string      out;
            for(auto const& l : sLines)
            {
                out += l + "\n";
            }
            return out;
        }

        // The last line that contains `text` ("" if none).
        std::string last(std::string const& text) const
        {
            std::scoped_lock lock(sMutex);
            for(auto it = sLines.rbegin(); it != sLines.rend(); ++it)
            {
                if(it->find(text) != std::string::npos)
                {
                    return *it;
                }
            }
            return "";
        }

        // The JIT lines of the whole program (JitTotals).
        static JitTotals jitTotals()
        {
            std::scoped_lock lock(sMutex);
            return sJit;
        }

    private:
        static void callback(int32_t, char const* func, char const* msg)
        {
            std::scoped_lock lock(sMutex);
            sLines.push_back(std::string(func) + ": " + msg);
            std::string const line = msg;
            auto const        at   = line.find("rocKE JIT: ");
            int               s = 0, n = 0;
            if(at != std::string::npos
               && std::sscanf(line.c_str() + at, "rocKE JIT: %d of %d launches", &s, &n) == 2)
            {
                sJit.plans++;
                sJit.launches += n;
                sJit.specialized += s;
                if(s < n)
                {
                    auto const note = line.find("; ", at);
                    sJit.declines[note == std::string::npos ? "?" : line.substr(note + 2)] += n - s;
                }
            }
        }
        static inline std::mutex               sMutex;
        static inline std::vector<std::string> sLines;
        static inline JitTotals                sJit;
    };

    // Prints the JIT totals of the program (HIPTENSOR_ROCKE_TEST_JIT=1) at its end.
    struct JitReport : ::testing::Environment
    {
        void TearDown() override
        {
            auto const t = LogCapture::jitTotals();
            if(t.plans == 0)
            {
                return;
            }
            std::printf("rocKE JIT totals: %lld of %lld launches specialized in %lld plans\n",
                        static_cast<long long>(t.specialized),
                        static_cast<long long>(t.launches),
                        static_cast<long long>(t.plans));
            for(auto const& [why, count] : t.declines)
            {
                std::printf("  declined %lld (first reason of a plan): %s\n",
                            static_cast<long long>(count),
                            why.c_str());
            }
        }
    };
    auto const* const gJitReport = ::testing::AddGlobalTestEnvironment(new JitReport);

    // HIPTENSOR_ROCKE_TEST_JIT=1 with HIPTENSOR_ROCKE_JIT=all: the JIT must specialize every
    // elementwise launch (every launch of the pack's generic kernel has a JIT class).
    bool jitAll()
    {
        char const* jit  = std::getenv("HIPTENSOR_ROCKE_TEST_JIT");
        char const* mode = std::getenv("HIPTENSOR_ROCKE_JIT");
        return jit != nullptr && std::string(jit) == "1" && mode != nullptr
               && std::string(mode) == "all";
    }

    void setEnv(char const* name, char const* value)
    {
#ifdef _WIN32
        _putenv_s(name, value == nullptr ? "" : value);
#else
        if(value == nullptr)
        {
            unsetenv(name);
        }
        else
        {
            setenv(name, value, 1);
        }
#endif
    }

    // Sets an environment variable for the life of the object.
    struct ScopedEnv
    {
        ScopedEnv(char const* name, char const* value)
            : mName(name)
        {
            if(char const* old = std::getenv(name))
            {
                mOld = old;
            }
            setEnv(name, value);
        }
        ~ScopedEnv()
        {
            setEnv(mName, mOld ? mOld->c_str() : nullptr);
        }
        char const*                mName;
        std::optional<std::string> mOld;
    };

    // The base name of the current device's architecture, e.g. "gfx950".
    // HIPTENSOR_ROCKE_CONTRACTION_ROUTE=f64: contractions on the f64 route only (no GEMM routes,
    // no GETT routes)
    bool f64RouteOnly()
    {
        char const* route = std::getenv("HIPTENSOR_ROCKE_CONTRACTION_ROUTE");
        return route != nullptr && std::string(route) == "f64";
    }

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

    // Another device of the current device's architecture (-1: none), to execute plans on a
    // device other than the one they were created on.
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

    // Makes `device` current for the scope's lifetime (nothing for -1), then restores the
    // device it found.
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

    bool isComplex(hiptensorDataType_t t)
    {
        return t == HIPTENSOR_C_32F || t == HIPTENSOR_C_64F;
    }

    size_t sizeOf(hiptensorDataType_t t)
    {
        switch(t)
        {
        case HIPTENSOR_R_16F:
        case HIPTENSOR_R_16BF:
            return 2;
        case HIPTENSOR_R_32F:
            return 4;
        case HIPTENSOR_R_64F:
        case HIPTENSOR_C_32F:
            return 8;
        default:
            return 16;
        }
    }

    // Rounds a value once to a data type (the stored value, as double).
    double roundTo(double v, hiptensorDataType_t t)
    {
        switch(t)
        {
        case HIPTENSOR_R_16F:
        {
            // two steps would round twice: round from double directly
            return static_cast<double>(static_cast<_Float16>(v));
        }
        case HIPTENSOR_R_16BF:
        {
            float f = static_cast<float>(v);
            // the double -> float step is exact for the values the exact cases produce
            return static_cast<double>(static_cast<float>(hip_bfloat16(f)));
        }
        case HIPTENSOR_R_32F:
        case HIPTENSOR_C_32F:
            return static_cast<double>(static_cast<float>(v));
        default:
            return v;
        }
    }

    double unitRoundoff(hiptensorDataType_t t)
    {
        switch(t)
        {
        case HIPTENSOR_R_16F:
            return std::ldexp(1.0, -11);
        case HIPTENSOR_R_16BF:
            return std::ldexp(1.0, -8);
        case HIPTENSOR_R_32F:
        case HIPTENSOR_C_32F:
            return std::ldexp(1.0, -24);
        default:
            return std::ldexp(1.0, -53);
        }
    }

    // A strided tensor in host memory: modes, extents, strides (elements) and the values of every
    // stored element (padding included) as complex doubles.
    struct Tensor
    {
        hiptensorDataType_t  type = HIPTENSOR_R_32F;
        std::vector<int32_t> modes;
        std::vector<int64_t> extents;
        std::vector<int64_t> strides;
        std::vector<cplx>    values; // storage order

        int64_t storage() const
        {
            int64_t span = 1;
            for(size_t i = 0; i < extents.size(); i++)
            {
                span += (extents[i] - 1) * strides[i];
            }
            return span;
        }

        std::vector<uint8_t> bytes() const
        {
            std::vector<uint8_t> out(values.size() * sizeOf(type));
            for(size_t i = 0; i < values.size(); i++)
            {
                auto* p = out.data() + i * sizeOf(type);
                switch(type)
                {
                case HIPTENSOR_R_16F:
                {
                    _Float16 v = static_cast<_Float16>(values[i].real());
                    std::memcpy(p, &v, 2);
                    break;
                }
                case HIPTENSOR_R_16BF:
                {
                    hip_bfloat16 v(static_cast<float>(values[i].real()));
                    std::memcpy(p, &v, 2);
                    break;
                }
                case HIPTENSOR_R_32F:
                {
                    float v = static_cast<float>(values[i].real());
                    std::memcpy(p, &v, 4);
                    break;
                }
                case HIPTENSOR_R_64F:
                {
                    double v = values[i].real();
                    std::memcpy(p, &v, 8);
                    break;
                }
                case HIPTENSOR_C_32F:
                {
                    float v[2] = {static_cast<float>(values[i].real()),
                                  static_cast<float>(values[i].imag())};
                    std::memcpy(p, v, 8);
                    break;
                }
                default:
                {
                    double v[2] = {values[i].real(), values[i].imag()};
                    std::memcpy(p, v, 16);
                    break;
                }
                }
            }
            return out;
        }

        void fromBytes(std::vector<uint8_t> const& in)
        {
            for(size_t i = 0; i < values.size(); i++)
            {
                auto const* p = in.data() + i * sizeOf(type);
                switch(type)
                {
                case HIPTENSOR_R_16F:
                {
                    _Float16 v;
                    std::memcpy(&v, p, 2);
                    values[i] = static_cast<double>(v);
                    break;
                }
                case HIPTENSOR_R_16BF:
                {
                    hip_bfloat16 v;
                    std::memcpy(&v, p, 2);
                    values[i] = static_cast<double>(static_cast<float>(v));
                    break;
                }
                case HIPTENSOR_R_32F:
                {
                    float v;
                    std::memcpy(&v, p, 4);
                    values[i] = v;
                    break;
                }
                case HIPTENSOR_R_64F:
                {
                    double v;
                    std::memcpy(&v, p, 8);
                    values[i] = v;
                    break;
                }
                case HIPTENSOR_C_32F:
                {
                    float v[2];
                    std::memcpy(v, p, 8);
                    values[i] = cplx(v[0], v[1]);
                    break;
                }
                default:
                {
                    double v[2];
                    std::memcpy(v, p, 16);
                    values[i] = cplx(v[0], v[1]);
                    break;
                }
                }
            }
        }

        int64_t offset(std::map<int32_t, int64_t> const& index) const
        {
            int64_t off = 0;
            for(size_t i = 0; i < modes.size(); i++)
            {
                off += index.at(modes[i]) * strides[i];
            }
            return off;
        }
    };

    // A tensor with the given modes stored in a random mode order with random row padding,
    // filled with integers in [-range, range] (exact) or uniform values in [-1, 1].
    Tensor makeTensor(hiptensorDataType_t               type,
                      std::vector<int32_t> const&       modes,
                      std::map<int32_t, int64_t> const& extents,
                      std::mt19937&                     rng,
                      bool                              exact,
                      int                               range = 3)
    {
        Tensor t;
        t.type  = type;
        t.modes = modes;
        for(auto m : modes)
        {
            t.extents.push_back(extents.at(m));
        }
        std::vector<size_t> order(modes.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        t.strides.assign(modes.size(), 0);
        int64_t stride = 1;
        for(size_t j = 0; j < order.size(); j++)
        {
            t.strides[order[j]] = stride;
            stride *= t.extents[order[j]] + (j == 0 && rng() % 3 == 0 ? 3 : 0);
        }
        t.values.resize(std::max<int64_t>(t.storage(), 1));
        std::uniform_int_distribution<int>     ints(-range, range);
        std::uniform_real_distribution<double> reals(-1.0, 1.0);
        for(auto& v : t.values)
        {
            double re = exact ? ints(rng) : reals(rng);
            double im = isComplex(type) ? (exact ? ints(rng) : reals(rng)) : 0.0;
            // store what the device sees
            v = cplx(roundTo(re, type), roundTo(im, type));
        }
        return t;
    }

    // Calls fn(index) for every index of `modes` over `extents`.
    template <typename F>
    void forEachIndex(std::vector<int32_t> const&       modes,
                      std::map<int32_t, int64_t> const& extents,
                      F&&                               fn)
    {
        std::map<int32_t, int64_t> index;
        for(auto m : modes)
        {
            index[m] = 0;
        }
        while(true)
        {
            fn(index);
            size_t i = 0;
            for(; i < modes.size(); i++)
            {
                if(++index[modes[i]] < extents.at(modes[i]))
                {
                    break;
                }
                index[modes[i]] = 0;
            }
            if(i == modes.size())
            {
                return;
            }
        }
    }

    // Uploads, runs (planned with the estimated workspace, on a non-blocking stream), downloads.
    struct Device
    {
        std::vector<void*> buffers;
        ~Device()
        {
            for(auto p : buffers)
            {
                static_cast<void>(hipFree(p));
            }
        }
        // `skew`: the tensor starts that many elements past the allocation (misaligned for
        // vector accesses)
        void* upload(Tensor const& t, size_t skew = 0)
        {
            auto         b   = t.bytes();
            size_t const pre = skew * sizeOf(t.type);
            void*        p   = nullptr;
            EXPECT_EQ(hipMalloc(&p, std::max<size_t>(b.size() + pre, 16)), hipSuccess);
            auto* const data = static_cast<uint8_t*>(p) + pre;
            EXPECT_EQ(hipMemcpy(data, b.data(), b.size(), hipMemcpyHostToDevice), hipSuccess);
            buffers.push_back(p);
            return data;
        }
        void download(void* p, Tensor& t)
        {
            std::vector<uint8_t> b(t.values.size() * sizeOf(t.type));
            EXPECT_EQ(hipMemcpy(b.data(), p, b.size(), hipMemcpyDeviceToHost), hipSuccess);
            t.fromBytes(b);
        }
    };

    hiptensorTensorDescriptor_t describe(ApiObjects& api, Tensor const& t)
    {
        return api.tensor(t.extents, t.strides, t.type);
    }

    // Scalars of the compute type: real ones as float or double, complex ones as pairs.
    struct Scalar
    {
        double      re = 0.0, im = 0.0;
        uint8_t     bytes[16]{};
        void const* as(hiptensorDataType_t type)
        {
            switch(type)
            {
            case HIPTENSOR_R_16F:
            {
                _Float16 v = static_cast<_Float16>(re);
                std::memcpy(bytes, &v, 2);
                break;
            }
            case HIPTENSOR_R_16BF:
            {
                hip_bfloat16 v(static_cast<float>(re));
                std::memcpy(bytes, &v, 2);
                break;
            }
            case HIPTENSOR_R_32F:
            {
                float v = static_cast<float>(re);
                std::memcpy(bytes, &v, 4);
                break;
            }
            case HIPTENSOR_R_64F:
                std::memcpy(bytes, &re, 8);
                break;
            case HIPTENSOR_C_32F:
            {
                float v[2] = {static_cast<float>(re), static_cast<float>(im)};
                std::memcpy(bytes, v, 8);
                break;
            }
            default:
            {
                double v[2] = {re, im};
                std::memcpy(bytes, v, 16);
                break;
            }
            }
            return bytes;
        }
        cplx value() const
        {
            return cplx(re, im);
        }
    };

    hiptensorDataType_t scalarTypeOf(hiptensorComputeDescriptor_t compute)
    {
        switch(compute)
        {
        case HIPTENSOR_COMPUTE_DESC_16F:
            return HIPTENSOR_R_16F;
        case HIPTENSOR_COMPUTE_DESC_16BF:
            return HIPTENSOR_R_16BF;
        case HIPTENSOR_COMPUTE_DESC_64F:
            return HIPTENSOR_R_64F;
        case HIPTENSOR_COMPUTE_DESC_C32F:
            return HIPTENSOR_C_32F;
        case HIPTENSOR_COMPUTE_DESC_C64F:
            return HIPTENSOR_C_64F;
        default:
            return HIPTENSOR_R_32F;
        }
    }

    // Plans with the estimated workspace; returns the plan or nullptr (with a failed expectation).
    hiptensorPlan_t planOf(ApiObjects&                    api,
                           hiptensorOperationDescriptor_t desc,
                           uint64_t&                      workspaceSize,
                           LogCapture const&              log)
    {
        workspaceSize = 0;
        EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                      api.handle, desc, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        hiptensorPlan_t plan   = nullptr;
        auto            status = api.plan(desc, plan, workspaceSize);
        EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << log.all();
        EXPECT_TRUE(log.contains("Backend: rocke")) << log.all();
        uint64_t required = 0;
        if(plan != nullptr)
        {
            EXPECT_EQ(hiptensorPlanGetAttribute(api.handle,
                                                plan,
                                                HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                                &required,
                                                sizeof(required)),
                      HIPTENSOR_STATUS_SUCCESS);
            EXPECT_LE(required, workspaceSize);
            workspaceSize = required;
        }
        return status == HIPTENSOR_STATUS_SUCCESS ? plan : nullptr;
    }

    // How a case plans: the plan preference (KERNEL_RANK, algorithm, plan-cache mode), the
    // workspace preference of the estimate the plan is limited to, and optionally the library
    // objects to plan with (to share a handle and its plan cache between cases).
    struct Choice
    {
        int32_t                       rank      = 0;
        hiptensorAlgo_t               algo      = HIPTENSOR_ALGO_DEFAULT;
        hiptensorCacheMode_t          cacheMode = HIPTENSOR_CACHE_MODE_PEDANTIC;
        hiptensorWorksizePreference_t workspace = HIPTENSOR_WORKSPACE_DEFAULT;
        ApiObjects*                   api       = nullptr;
        // plan with a workspace limit of zero (the estimate is still queried)
        bool zeroWorkspace = false;
        // A, B, C and D one element past aligned addresses
        size_t skew = 0;
        // execute on this device (buffers, workspace and stream there; -1: the device the plan
        // is created on)
        int device = -1;
    };

    // What the rocKE backend's selection log line says about a plan, and its workspace.
    struct SelectionLog
    {
        int         candidates = 0;
        std::string pick;
        std::string source;
        std::string reason; // what decided the source (the rule, the model, ...)
        std::string launches; // kernel families of the program's launches, '+'-separated
        uint64_t    estimate = 0;
        uint64_t    required = 0;
        std::string line; // the whole selection log line
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

    SelectionLog selectionOf(LogCapture const& log)
    {
        SelectionLog out;
        std::string  all = log.all(), line;
        for(size_t start = 0; start < all.size();)
        {
            auto end = all.find('\n', start);
            auto l = all.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if(l.find("rocKE selection:") != std::string::npos)
            {
                line = l;
            }
            start = end == std::string::npos ? all.size() : end + 1;
        }
        out.candidates = std::atoi(field(line, "Candidates").c_str());
        out.pick       = field(line, "Pick");
        out.source     = field(line, "Source");
        out.launches   = field(line, "Launches");
        out.line       = line;
        // "Source: <source> (<reason>), Pick: ..."
        auto const open  = line.find("(", line.find("Source: "));
        auto const close = line.rfind("), Pick: ");
        if(open != std::string::npos && close != std::string::npos && close > open)
        {
            out.reason = line.substr(open + 1, close - open - 1);
        }
        return out;
    }

    // planOf with a Choice: the plan or nullptr (with a failed expectation).
    hiptensorPlan_t planWith(ApiObjects&                    api,
                             hiptensorOperationDescriptor_t desc,
                             Choice const&                  choice,
                             LogCapture const&              log,
                             uint64_t&                      workspaceSize,
                             SelectionLog*                  selection)
    {
        hiptensorPlanPreference_t pref = nullptr;
        // HIPTENSOR_ROCKE_TEST_JIT=1: every plan with HIPTENSOR_JIT_MODE_DEFAULT (the rocKE JIT's
        // specialized kernels against the same references)
        char const* jit     = std::getenv("HIPTENSOR_ROCKE_TEST_JIT");
        auto const  jitMode = jit != nullptr && std::string(jit) == "1" ? HIPTENSOR_JIT_MODE_DEFAULT
                                                                        : HIPTENSOR_JIT_MODE_NONE;
        EXPECT_EQ(hiptensorCreatePlanPreference(api.handle, &pref, choice.algo, jitMode),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(api.handle,
                                                      pref,
                                                      HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK,
                                                      &choice.rank,
                                                      sizeof(choice.rank)),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(hiptensorPlanPreferenceSetAttribute(api.handle,
                                                      pref,
                                                      HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE,
                                                      &choice.cacheMode,
                                                      sizeof(choice.cacheMode)),
                  HIPTENSOR_STATUS_SUCCESS);
        workspaceSize = 0;
        EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                      api.handle, desc, pref, choice.workspace, &workspaceSize),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        if(choice.zeroWorkspace)
        {
            workspaceSize = 0;
        }
        hiptensorPlan_t plan   = nullptr;
        auto            status = hiptensorCreatePlan(api.handle, &plan, desc, pref, workspaceSize);
        static_cast<void>(hiptensorDestroyPlanPreference(pref));
        EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << log.all();
        if(plan == nullptr)
        {
            return nullptr;
        }
        api.plans.push_back(plan);
        EXPECT_TRUE(log.contains("Backend: rocke")) << log.all();
        if(jitMode == HIPTENSOR_JIT_MODE_DEFAULT && jitAll())
        {
            auto const line = log.last("rocKE JIT: ");
            int        s = 0, n = 0;
            if(!line.empty()
               && std::sscanf(line.c_str() + line.find("rocKE JIT: "),
                              "rocKE JIT: %d of %d launches",
                              &s,
                              &n)
                      == 2)
            {
                EXPECT_EQ(s, n) << line;
            }
        }
        uint64_t required = 0;
        EXPECT_EQ(
            hiptensorPlanGetAttribute(
                api.handle, plan, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, &required, sizeof(required)),
            HIPTENSOR_STATUS_SUCCESS);
        EXPECT_LE(required, workspaceSize);
        if(selection != nullptr)
        {
            *selection          = selectionOf(log);
            selection->estimate = workspaceSize;
            selection->required = required;
        }
        workspaceSize = required;
        return status == HIPTENSOR_STATUS_SUCCESS ? plan : nullptr;
    }

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

    // Compares every element of `got` with `want` (both padded storage): exact cases bit for
    // bit after one rounding of the reference, the others within `bound[i]`.
    void compare(Tensor const&              got,
                 std::vector<cplx> const&   want,
                 std::vector<double> const& bound,
                 std::vector<bool> const&   written,
                 bool                       exact,
                 std::string const&         what)
    {
        int64_t bad = 0;
        for(size_t i = 0; i < got.values.size(); i++)
        {
            if(!written[i])
            {
                continue;
            }
            cplx w
                = exact ? cplx(roundTo(want[i].real(), got.type), roundTo(want[i].imag(), got.type))
                        : want[i];
            double err = std::abs(got.values[i] - w);
            bool   ok  = exact ? (got.values[i] == w) : (err <= bound[i]);
            if(!ok && bad++ < 5)
            {
                ADD_FAILURE() << what << ": element " << i << " got " << got.values[i] << " want "
                              << w << " (bound " << (exact ? 0.0 : bound[i]) << ")";
            }
        }
        EXPECT_EQ(bad, 0) << what;
    }

    // ---------------------------------------------------------------------- contraction

    struct ContractionCase
    {
        hiptensorDataType_t          type;
        hiptensorComputeDescriptor_t compute;
        int                          seed;
        bool                         exact;
        bool                         bilinear;
        bool                         scale; // no C (hipTensor's scale contraction)
        hiptensorOperator_t          opA = HIPTENSOR_OP_IDENTITY;
        // > 0: groupRank M, N and K modes each, interleaved so that no two modes merge
        // (A = m0 k0 m1 k1 ..., B = n0 k0 n1 k1 ..., D = m0 n0 m1 n1 ...): more modes than the
        // generic kernels' slots
        int groupRank = 0;
        // > 0: the extent of m0, n0 and k0 (groupRank)
        int64_t groupFirst = 0;
        // an outer product: no contracted mode, A = m p, B = n, D = n m p (groupRank 0)
        bool outer = false;
    };

    void runContraction(ContractionCase const& c,
                        Choice const&          choice    = {},
                        SelectionLog*          selection = nullptr)
    {
        std::mt19937              rng(c.seed);
        std::optional<ApiObjects> own;
        if(choice.api == nullptr)
        {
            own.emplace();
        }
        ApiObjects& api = choice.api != nullptr ? *choice.api : *own;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        // the direct GETT rounds operands to a 16-bit compute type on every target; the f64 route
        // alone declines them
        if(f64RouteOnly() && c.type == HIPTENSOR_R_32F && c.compute != HIPTENSOR_COMPUTE_DESC_32F)
        {
            GTEST_SKIP() << "the f64 route declines f32 data with a 16-bit compute type";
        }
        LogCapture log;
        // modes: m0 m1 n0 k0 k1 with random extents, some not multiples of any tile (hipTensor
        // refuses batched modes, shared by A, B and D); with c.outer no contracted mode
        std::vector<int64_t>       pick = {1, 3, 5, 8, 17, 33, 64, 70};
        std::map<int32_t, int64_t> ext;
        std::vector<int32_t>       ma, mb, md, kmodes;
        if(c.groupRank > 0)
        {
            for(int i = 0; i < c.groupRank; i++)
            {
                // extents 2, and 3 for one mode of each group: small enough for the loop
                // reference, not a power of two
                int32_t const m = 'a' + i, n = 'h' + i, k = 'o' + i;
                int64_t const e = i == int(rng() % c.groupRank) ? 3 : 2;
                ext[m]          = e;
                ext[n]          = i == 0 ? 3 : 2;
                ext[k]          = i == c.groupRank - 1 ? 3 : 2;
                if(i == 0 && c.groupFirst > 0)
                {
                    ext[m] = ext[n] = ext[k] = c.groupFirst;
                }
                ma.insert(ma.end(), {m, k});
                mb.insert(mb.end(), {n, k});
                md.insert(md.end(), {m, n});
                kmodes.push_back(k);
            }
        }
        else
        {
            for(auto mode : {'m', 'p', 'n', 'k', 'q'})
            {
                ext[mode] = pick[rng() % pick.size()];
            }
            if(!c.exact)
            {
                ext['k'] = 96;
                ext['q'] = 3;
            }
            ma     = {'m', 'k', 'p', 'q'};
            mb     = {'k', 'n', 'q'};
            md     = {'n', 'm', 'p'};
            kmodes = {'k', 'q'};
            if(c.outer)
            {
                ma     = {'m', 'p'};
                mb     = {'n'};
                kmodes = {};
            }
            std::shuffle(ma.begin(), ma.end(), rng);
            std::shuffle(mb.begin(), mb.end(), rng);
            std::shuffle(md.begin(), md.end(), rng);
        }
        auto A = makeTensor(c.type, ma, ext, rng, c.exact);
        auto B = makeTensor(c.type, mb, ext, rng, c.exact);
        auto C = makeTensor(c.type, md, ext, rng, c.exact);
        auto D = makeTensor(c.type, md, ext, rng, c.exact);
        // complex contractions need C with D's layout
        if(rng() % 2 || isComplex(c.type))
        {
            D.strides = C.strides; // C and D with one layout
            D.values.assign(C.values.size(), cplx(0, 0));
        }
        auto   scalarType = scalarTypeOf(c.compute);
        Scalar alpha, beta;
        alpha.re = c.exact ? 2.0 : 0.75;
        beta.re  = c.bilinear && !c.scale ? (c.exact ? -1.0 : 0.5) : 0.0;
        if(isComplex(c.type))
        {
            alpha.im = 1.0;
            beta.im  = c.bilinear && !c.scale ? 2.0 : 0.0;
        }

        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dB = describe(api, B), dD = describe(api, D);
        auto dC = c.scale ? nullptr : describe(api, C);
        ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                             &desc,
                                             dA,
                                             A.modes.data(),
                                             c.opA,
                                             dB,
                                             B.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dC,
                                             c.scale ? nullptr : C.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dD,
                                             D.modes.data(),
                                             c.compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t workspaceSize = 0;
        auto     plan          = planWith(api, desc, choice, log, workspaceSize, selection);
        if(plan == nullptr)
        {
            return;
        }
        DeviceScope onDevice(choice.device);
        Device      dev;
        void*       pA = dev.upload(A, choice.skew);
        void*       pB = dev.upload(B, choice.skew);
        void*       pC = c.scale ? nullptr : dev.upload(C, choice.skew);
        void*       pD = dev.upload(D, choice.skew);
        void*       ws = nullptr;
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            dev.buffers.push_back(ws);
        }
        Stream stream;
        ASSERT_EQ(hiptensorContract(api.handle,
                                    plan,
                                    alpha.as(scalarType),
                                    pA,
                                    pB,
                                    beta.as(scalarType),
                                    pC,
                                    pD,
                                    ws,
                                    workspaceSize,
                                    stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);

        // reference: the inputs as the compute type sees them
        auto inputType = c.type;
        if(c.compute == HIPTENSOR_COMPUTE_DESC_16F)
        {
            inputType = HIPTENSOR_R_16F;
        }
        else if(c.compute == HIPTENSOR_COMPUTE_DESC_16BF)
        {
            inputType = HIPTENSOR_R_16BF;
        }
        std::vector<cplx>   want(D.values.size(), cplx(0, 0));
        std::vector<double> bound(D.values.size(), 0.0);
        std::vector<bool>   written(D.values.size(), false);
        int64_t             kSize = 1;
        for(auto k : kmodes)
        {
            kSize *= ext[k];
        }
        double const uc = isComplex(c.type) || c.type == HIPTENSOR_R_64F ? unitRoundoff(c.type)
                                                                         : std::ldexp(1.0, -24);
        double const uo = unitRoundoff(D.type);
        forEachIndex(md, ext, [&](auto index) {
            cplx   sum  = 0;
            double mags = 0;
            forEachIndex(kmodes, ext, [&](auto kindex) {
                auto full = index;
                for(auto& kv : kindex)
                {
                    full[kv.first] = kv.second;
                }
                cplx a = A.values[A.offset(full)];
                cplx b = B.values[B.offset(full)];
                a      = cplx(roundTo(a.real(), inputType), roundTo(a.imag(), inputType));
                b      = cplx(roundTo(b.real(), inputType), roundTo(b.imag(), inputType));
                if(c.opA == HIPTENSOR_OP_CONJ)
                {
                    a = std::conj(a);
                }
                else if(c.opA == HIPTENSOR_OP_RELU)
                {
                    a = cplx(std::max(a.real(), 0.0), 0.0);
                }
                sum += a * b;
                mags += std::abs(a) * std::abs(b);
            });
            auto off   = D.offset(index);
            cplx cval  = c.scale ? cplx(0, 0) : C.values[C.offset(index)];
            want[off]  = alpha.value() * sum + beta.value() * cval;
            bound[off] = (kSize + 4) * uc * std::abs(alpha.value()) * mags * 2
                         + 2 * uo * std::abs(want[off]) + 4 * uc * std::abs(beta.value() * cval);
            written[off] = true;
        });
        compare(got, want, bound, written, c.exact, "contraction");
    }

    class RockeContraction : public ::testing::TestWithParam<ContractionCase>
    {
    };

    TEST_P(RockeContraction, MatchesLoopReference)
    {
        runContraction(GetParam());
    }

    std::vector<ContractionCase> contractionCases()
    {
        std::vector<ContractionCase> out;
        struct Row
        {
            hiptensorDataType_t          type;
            hiptensorComputeDescriptor_t compute;
        };
        Row const rows[] = {{HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F},
                            {HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_32F},
                            {HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F},
                            {HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_16F},
                            {HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_16BF},
                            {HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F},
                            {HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_32F},
                            {HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F},
                            {HIPTENSOR_C_64F, HIPTENSOR_COMPUTE_DESC_C64F}};
        int       seed   = 100;
        for(auto const& r : rows)
        {
            for(int i = 0; i < 4; i++)
            {
                out.push_back({r.type, r.compute, seed++, true, i % 2 == 1, i == 2});
            }
            out.push_back({r.type, r.compute, seed++, false, true, false});
            if(isComplex(r.type))
            {
                out.push_back({r.type, r.compute, seed++, true, true, false, HIPTENSOR_OP_CONJ});
            }
            // 10 and 12 modes per tensor, past the generic kernels' 8 slots
            if(r.compute != HIPTENSOR_COMPUTE_DESC_16F && r.compute != HIPTENSOR_COMPUTE_DESC_16BF
               && !(r.type == HIPTENSOR_R_64F && r.compute == HIPTENSOR_COMPUTE_DESC_32F))
            {
                out.push_back(
                    {r.type, r.compute, seed++, true, true, false, HIPTENSOR_OP_IDENTITY, 5});
                out.push_back(
                    {r.type, r.compute, seed++, true, false, true, HIPTENSOR_OP_IDENTITY, 6});
            }
        }
        // outer products (no contracted mode), scale and bilinear
        seed = 190;
        for(auto const& r : rows)
        {
            for(bool scale : {false, true})
            {
                ContractionCase o{r.type, r.compute, seed++, true, !scale, scale};
                o.outer = true;
                out.push_back(o);
            }
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke, RockeContraction, ::testing::ValuesIn(contractionCases()));

    // Large contractions D[m,n] = alpha * A[k,m] B[k,n] + beta * C[m,n] (packed column-major),
    // sized so that the GEMM bodies for many output tiles run (data-parallel and stream-K), with
    // 2048 sampled outputs compared with the loop reference (integer data: exact).
    struct LargeCase
    {
        hiptensorDataType_t          type;
        hiptensorComputeDescriptor_t compute;
        int64_t                      M, N, K;
    };

    class RockeLargeContraction : public ::testing::TestWithParam<LargeCase>
    {
    };

    TEST_P(RockeLargeContraction, SampledOutputs)
    {
        auto const c = GetParam();
        ApiObjects api;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture                 log;
        std::map<int32_t, int64_t> ext    = {{'m', c.M}, {'n', c.N}, {'k', c.K}};
        auto                       packed = [&](std::vector<int32_t> const& modes, int seed) {
            Tensor t;
            t.type    = c.type;
            t.modes   = modes;
            int64_t s = 1;
            for(auto m : modes)
            {
                t.extents.push_back(ext[m]);
                t.strides.push_back(s);
                s *= ext[m];
            }
            t.values.resize(s);
            for(int64_t i = 0; i < s; i++)
            {
                double re = double((i * 7 + seed) % 5) - 2;
                double im = isComplex(c.type) ? double((i * 3 + seed) % 5) - 2 : 0.0;
                t.values[i] = cplx(re, im);
            }
            return t;
        };
        auto A = packed({'k', 'm'}, 1), B = packed({'k', 'n'}, 2), C = packed({'m', 'n'}, 3);
        auto D                              = C;
        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dB = describe(api, B), dC = describe(api, C),
             dD = describe(api, D);
        ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                             &desc,
                                             dA,
                                             A.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dB,
                                             B.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dC,
                                             C.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dD,
                                             D.modes.data(),
                                             c.compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t workspaceSize = 0;
        auto     plan          = planOf(api, desc, workspaceSize, log);
        ASSERT_NE(plan, nullptr);
        Device dev;
        void*  pA = dev.upload(A);
        void*  pB = dev.upload(B);
        void*  pC = dev.upload(C);
        void*  pD = dev.upload(D);
        void*  ws = nullptr;
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            dev.buffers.push_back(ws);
        }
        auto   scalarType = scalarTypeOf(c.compute);
        Scalar alpha, beta;
        alpha.re = 2.0;
        beta.re  = -1.0;
        if(isComplex(c.type))
        {
            alpha.im = 1.0;
            beta.im  = 1.0;
        }
        Stream stream;
        // twice: the second execution reuses the plan (stream-K flags zeroed again)
        for(int run = 0; run < 2; run++)
        {
            ASSERT_EQ(hiptensorContract(api.handle,
                                        plan,
                                        alpha.as(scalarType),
                                        pA,
                                        pB,
                                        beta.as(scalarType),
                                        pC,
                                        pD,
                                        ws,
                                        workspaceSize,
                                        stream.s),
                      HIPTENSOR_STATUS_SUCCESS)
                << log.all();
        }
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);
        std::mt19937                           rng(11);
        std::uniform_int_distribution<int64_t> pm(0, c.M - 1), pn(0, c.N - 1);
        int64_t                                bad = 0;
        for(int s = 0; s < 2048; s++)
        {
            int64_t m = pm(rng), n = pn(rng);
            cplx    sum = 0;
            for(int64_t k = 0; k < c.K; k++)
            {
                sum += A.values[k + c.K * m] * B.values[k + c.K * n];
            }
            cplx w = alpha.value() * sum + beta.value() * C.values[m + c.M * n];
            w      = cplx(roundTo(w.real(), c.type), roundTo(w.imag(), c.type));
            if(got.values[m + c.M * n] != w && bad++ < 5)
            {
                ADD_FAILURE() << "D[" << m << "," << n << "] got " << got.values[m + c.M * n]
                              << " want " << w;
            }
        }
        EXPECT_EQ(bad, 0) << log.all();
    }

    INSTANTIATE_TEST_SUITE_P(
        Rocke,
        RockeLargeContraction,
        ::testing::Values(LargeCase{HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, 4096, 4096, 256},
                          LargeCase{HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, 8192, 8192, 256},
                          LargeCase{HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_32F, 4096, 4096, 192},
                          LargeCase{HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_32F, 8192, 8192, 128},
                          LargeCase{HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, 4096, 4096, 128},
                          LargeCase{HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, 8192, 8192, 96},
                          LargeCase{HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, 2048, 2048, 64},
                          LargeCase{HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, 6144, 4096, 64},
                          LargeCase{HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, 8192, 4096, 32},
                          LargeCase{HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F, 2048, 2048, 4096},
                          LargeCase{
                              HIPTENSOR_C_64F, HIPTENSOR_COMPUTE_DESC_C64F, 1024, 1024, 2048}));

    // Tensors past 2^31 elements (the kernels' 64-bit offsets): a permutation of an f16 tensor and
    // a reduction of an f32 tensor of 65537 x 32769 elements, sampled outputs compared with the
    // loop reference (integer data: exact).
    TEST(RockeLargeIndex, PermutationAndReduction)
    {
        size_t freeBytes = 0, totalBytes = 0;
        ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);
        if(freeBytes < (size_t(24) << 30))
        {
            GTEST_SKIP() << "needs 24 GiB of free device memory";
        }
        ApiObjects    api;
        LogCapture    log;
        int64_t const M = 65537, N = 32769, count = M * N;
        int32_t const modeA[] = {'m', 'n'}, modeB[] = {'n', 'm'}, modeD[] = {'m'};
        Stream        stream;
        {
            std::vector<_Float16> a(count);
            for(int64_t i = 0; i < count; i++)
            {
                a[i] = static_cast<_Float16>(double(i % 2039) - 1019);
            }
            auto                           dA   = api.tensor({M, N}, {}, HIPTENSOR_R_16F);
            auto                           dB   = api.tensor({N, M}, {}, HIPTENSOR_R_16F);
            hiptensorOperationDescriptor_t desc = nullptr;
            ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                                 &desc,
                                                 dA,
                                                 modeA,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dB,
                                                 modeB,
                                                 HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(desc);
            uint64_t ws   = 0;
            auto     plan = planOf(api, desc, ws, log);
            ASSERT_NE(plan, nullptr);
            EXPECT_TRUE(log.contains("_i64_")) << log.all();
            void *pA = nullptr, *pB = nullptr;
            ASSERT_EQ(hipMalloc(&pA, count * 2), hipSuccess);
            ASSERT_EQ(hipMalloc(&pB, count * 2), hipSuccess);
            ASSERT_EQ(hipMemcpy(pA, a.data(), count * 2, hipMemcpyHostToDevice), hipSuccess);
            float alpha = -1.0f;
            ASSERT_EQ(hiptensorPermute(api.handle, plan, &alpha, pA, pB, stream.s),
                      HIPTENSOR_STATUS_SUCCESS);
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            std::vector<_Float16> b(count);
            ASSERT_EQ(hipMemcpy(b.data(), pB, count * 2, hipMemcpyDeviceToHost), hipSuccess);
            static_cast<void>(hipFree(pA));
            static_cast<void>(hipFree(pB));
            std::mt19937                           rng(3);
            std::uniform_int_distribution<int64_t> pm(0, M - 1), pn(0, N - 1);
            int64_t                                bad = 0;
            for(int s = 0; s < 4096; s++)
            {
                int64_t m   = s < 2 ? (s ? M - 1 : 0) : pm(rng),
                        n   = s < 2 ? (s ? N - 1 : 0) : pn(rng);
                double want = -double(a[m + M * n]);
                if(double(b[n + N * m]) != want && bad++ < 5)
                {
                    ADD_FAILURE() << "B[" << n << "," << m << "] = " << double(b[n + N * m])
                                  << ", want " << want;
                }
            }
            EXPECT_EQ(bad, 0);
        }
        {
            std::vector<float> a(count);
            for(int64_t i = 0; i < count; i++)
            {
                a[i] = float(i % 5) - 2.0f;
            }
            std::vector<float> c(M);
            for(int64_t i = 0; i < M; i++)
            {
                c[i] = float(i % 3);
            }
            auto                           dA   = api.tensor({M, N}, {}, HIPTENSOR_R_32F);
            auto                           dC   = api.tensor({M}, {}, HIPTENSOR_R_32F);
            hiptensorOperationDescriptor_t desc = nullptr;
            ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                               &desc,
                                               dA,
                                               modeA,
                                               HIPTENSOR_OP_IDENTITY,
                                               dC,
                                               modeD,
                                               HIPTENSOR_OP_IDENTITY,
                                               dC,
                                               modeD,
                                               HIPTENSOR_OP_ADD,
                                               HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(desc);
            uint64_t ws   = 0;
            auto     plan = planOf(api, desc, ws, log);
            ASSERT_NE(plan, nullptr);
            void *pA = nullptr, *pC = nullptr, *pD = nullptr, *pW = nullptr;
            ASSERT_EQ(hipMalloc(&pA, count * 4), hipSuccess);
            ASSERT_EQ(hipMalloc(&pC, M * 4), hipSuccess);
            ASSERT_EQ(hipMalloc(&pD, M * 4), hipSuccess);
            if(ws > 0)
            {
                ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
            }
            ASSERT_EQ(hipMemcpy(pA, a.data(), count * 4, hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(pC, c.data(), M * 4, hipMemcpyHostToDevice), hipSuccess);
            float alpha = 2.0f, beta = -1.0f;
            ASSERT_EQ(
                hiptensorReduce(api.handle, plan, &alpha, pA, &beta, pC, pD, pW, ws, stream.s),
                HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            std::vector<float> d(M);
            ASSERT_EQ(hipMemcpy(d.data(), pD, M * 4, hipMemcpyDeviceToHost), hipSuccess);
            for(auto* p : {pA, pC, pD, pW})
            {
                static_cast<void>(hipFree(p));
            }
            std::mt19937                           rng(5);
            std::uniform_int_distribution<int64_t> pm(0, M - 1);
            int64_t                                bad = 0;
            for(int s = 0; s < 512; s++)
            {
                int64_t m   = s == 0 ? M - 1 : pm(rng);
                double  sum = 0;
                for(int64_t n = 0; n < N; n++)
                {
                    sum += a[m + M * n];
                }
                double want = 2.0 * sum - c[m];
                if(double(d[m]) != want && bad++ < 5)
                {
                    ADD_FAILURE() << "D[" << m << "] = " << d[m] << ", want " << want;
                }
            }
            EXPECT_EQ(bad, 0);
        }
    }

    // A problem of rocKE's tuned GETT route table (rank-8 operands, packed column-major): the
    // plan takes the tuned route when the pack has it. 4096 sampled outputs are compared with
    // the loop reference (integer data: exact).
    TEST(RockeTunedGett, SampledOutputs)
    {
        ApiObjects                 api;
        LogCapture                 log;
        std::map<int32_t, int64_t> ext    = {{'a', 12},
                                             {'b', 8},
                                             {'c', 8},
                                             {'d', 6}, // M0..M3
                                             {'e', 10},
                                             {'f', 8},
                                             {'g', 6},
                                             {'h', 8}, // N0..N3
                                             {'i', 6},
                                             {'j', 6},
                                             {'k', 4},
                                             {'l', 4}}; // K0..K3
        std::vector<int32_t>       ma     = {'a', 'b', 'c', 'd', 'i', 'j', 'k', 'l'};
        std::vector<int32_t>       mb     = {'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'};
        std::vector<int32_t>       md     = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
        auto                       packed = [&](std::vector<int32_t> const& modes) {
            Tensor t;
            t.type    = HIPTENSOR_R_16F;
            t.modes   = modes;
            int64_t s = 1;
            for(auto m : modes)
            {
                t.extents.push_back(ext[m]);
                t.strides.push_back(s);
                s *= ext[m];
            }
            t.values.resize(s);
            return t;
        };
        auto A = packed(ma), B = packed(mb), C = packed(md), D = packed(md);
        for(size_t i = 0; i < A.values.size(); i++)
        {
            A.values[i] = double(int64_t(i * 7 % 5) - 2);
        }
        for(size_t i = 0; i < B.values.size(); i++)
        {
            B.values[i] = double(int64_t(i * 3 % 5) - 2);
        }
        for(size_t i = 0; i < C.values.size(); i++)
        {
            C.values[i] = double(int64_t(i % 7) - 3);
        }
        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dB = describe(api, B), dC = describe(api, C),
             dD = describe(api, D);
        ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                             &desc,
                                             dA,
                                             A.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dB,
                                             B.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dC,
                                             C.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dD,
                                             D.modes.data(),
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t workspaceSize = 0;
        auto     plan          = planOf(api, desc, workspaceSize, log);
        ASSERT_NE(plan, nullptr);
        if(!log.contains("tuned-gett"))
        {
            GTEST_SKIP() << "the pack has no tuned GETT route for this problem";
        }
        // the tuned route table decides before the selection model
        EXPECT_TRUE(log.contains("Source: TUNED")) << log.all();
        Device dev;
        void*  pA = dev.upload(A);
        void*  pB = dev.upload(B);
        void*  pC = dev.upload(C);
        void*  pD = dev.upload(D);
        void*  ws = nullptr;
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            dev.buffers.push_back(ws);
        }
        float  alpha = 1.0f, beta = -2.0f;
        Stream stream;
        ASSERT_EQ(hiptensorContract(
                      api.handle, plan, &alpha, pA, pB, &beta, pC, pD, ws, workspaceSize, stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);
        std::mt19937                           rng(7);
        std::uniform_int_distribution<int64_t> pick(0, int64_t(D.values.size()) - 1);
        std::vector<int32_t>                   kmodes = {'i', 'j', 'k', 'l'};
        int64_t                                bad    = 0;
        for(int s = 0; s < 4096; s++)
        {
            int64_t                    flat = pick(rng);
            std::map<int32_t, int64_t> index;
            int64_t                    rest = flat;
            for(auto m : md)
            {
                index[m] = rest % ext[m];
                rest /= ext[m];
            }
            double sum = 0;
            forEachIndex(kmodes, ext, [&](auto kindex) {
                auto full = index;
                for(auto& kv : kindex)
                {
                    full[kv.first] = kv.second;
                }
                sum += A.values[A.offset(full)].real() * B.values[B.offset(full)].real();
            });
            double want = roundTo(alpha * sum + beta * C.values[flat].real(), HIPTENSOR_R_16F);
            if(got.values[flat].real() != want && bad++ < 5)
            {
                ADD_FAILURE() << "element " << flat << " got " << got.values[flat].real()
                              << " want " << want;
            }
        }
        EXPECT_EQ(bad, 0);
    }

    // ---------------------------------------------------------------- AOT GETT route table

    // Problems of rocKE's AOT GETT route table (one per route kind: the MN-layout GETT reading
    // the operands in place, with B or both operands packed first, and the K-contiguous route
    // with both operands packed, A's or B's pack first; f16 and bf16, scale and bilinear). Mode
    // lists are rocKE's (row-major: the last mode is the fastest); the hipTensor tensors are the
    // packed column-major tensors of the reversed lists. The plan must take the route of the
    // pack (on gfx950, whose pack carries the table) and match the f64 loop reference bit for
    // bit after one rounding (integer data: the f32 sums are exact) at sampled outputs, at
    // outputs that read an infinity or a NaN (IEEE propagation), with alpha or beta zero (the
    // operand whose scale is zero is not read: its NaN / infinity never reaches D), and with a
    // workspace not aligned to 16 bytes (the plan's fallback).
    struct AotGettCase
    {
        char const*                    name;
        hiptensorDataType_t            type;
        bool                           bilinear;
        std::vector<std::string>       modesA, modesB, modesD;
        std::map<std::string, int64_t> extents;
    };

    // A dense column-major 16-bit tensor in host memory (bits) with its hipTensor modes.
    struct AotTensor
    {
        hiptensorDataType_t      type;
        std::vector<int32_t>     modes;
        std::vector<std::string> names;
        std::vector<int64_t>     extents, strides;
        std::vector<uint16_t>    bits;
        int64_t                  size = 1;
    };

    uint16_t aotEncode(double v, hiptensorDataType_t t)
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

    double aotDecode(uint16_t b, hiptensorDataType_t t)
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

    int32_t aotModeId(std::string const& name)
    {
        return int32_t(name[0]) * 64 + std::stoi(name.substr(1));
    }

    AotTensor aotTensor(AotGettCase const& c, std::vector<std::string> const& rocke)
    {
        AotTensor t;
        t.type  = c.type;
        t.names = std::vector<std::string>(rocke.rbegin(), rocke.rend());
        for(auto const& n : t.names)
        {
            t.modes.push_back(aotModeId(n));
            t.extents.push_back(c.extents.at(n));
            t.strides.push_back(t.size);
            t.size *= c.extents.at(n);
        }
        return t;
    }

    // Integer values in [-2, 2], spread by a multiplicative hash of the storage index.
    void aotFill(AotTensor& t, uint64_t seed)
    {
        t.bits.resize(size_t(t.size));
        for(int64_t i = 0; i < t.size; i++)
        {
            uint64_t h        = (uint64_t(i) + seed) * 0x9E3779B97F4A7C15ull;
            t.bits[size_t(i)] = aotEncode(double(int64_t((h >> 40) % 5) - 2), t.type);
        }
    }

    int64_t aotOffset(AotTensor const& t, std::map<std::string, int64_t> const& index)
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

    // Plans `c`, then runs it once per scale / poison / workspace variant and compares sampled
    // outputs with the reference.
    void runAotGettCase(AotGettCase const& c)
    {
        SCOPED_TRACE(c.name);
        ApiObjects api;
        LogCapture log;
        AotTensor  A = aotTensor(c, c.modesA), B = aotTensor(c, c.modesB),
                  D = aotTensor(c, c.modesD), C = aotTensor(c, c.modesD);
        aotFill(A, 1);
        aotFill(B, 2);
        aotFill(C, 3);
        std::vector<std::string> kmodes;
        for(auto const& n : A.names)
        {
            if(std::find(D.names.begin(), D.names.end(), n) == D.names.end())
            {
                kmodes.push_back(n);
            }
        }
        // every K index as (offset in A, offset in B)
        std::vector<std::pair<int64_t, int64_t>> koffs = {{0, 0}};
        for(auto const& k : kmodes)
        {
            std::map<std::string, int64_t>           one = {{k, 1}};
            int64_t const                            sa = aotOffset(A, one), sb = aotOffset(B, one);
            std::vector<std::pair<int64_t, int64_t>> next;
            for(int64_t i = 0; i < c.extents.at(k); i++)
            {
                for(auto const& o : koffs)
                {
                    next.emplace_back(o.first + i * sa, o.second + i * sb);
                }
            }
            koffs.swap(next);
        }
        auto dA = api.tensor(A.extents, A.strides, c.type);
        auto dB = api.tensor(B.extents, B.strides, c.type);
        auto dD = api.tensor(D.extents, D.strides, c.type);
        auto dC = c.bilinear ? api.tensor(C.extents, C.strides, c.type) : nullptr;
        hiptensorOperationDescriptor_t desc = nullptr;
        ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                             &desc,
                                             dA,
                                             A.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dB,
                                             B.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dC,
                                             c.bilinear ? C.modes.data() : nullptr,
                                             HIPTENSOR_OP_IDENTITY,
                                             dD,
                                             D.modes.data(),
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t workspaceSize = 0;
        auto     plan          = planOf(api, desc, workspaceSize, log);
        ASSERT_NE(plan, nullptr);
        if(!log.contains("tuned-gett"))
        {
            // gfx950's pack carries the AOT route table: the route must be there
            ASSERT_NE(deviceArch(), "gfx950") << log.all();
            GTEST_SKIP() << "the pack has no AOT GETT route for this problem";
        }
        EXPECT_TRUE(log.contains("Source: TUNED")) << log.all();

        // poison: an infinity at A's first element and a NaN at B's last; C (bilinear) gets an
        // infinity and a NaN in the variants that do not read it
        std::vector<int64_t> outs;
        {
            std::mt19937                           rng(11);
            std::uniform_int_distribution<int64_t> pick(0, D.size - 1);
            for(int s = 0; s < 4096; s++)
            {
                outs.push_back(pick(rng));
            }
            // outputs reading A's first element (every M mode 0) or B's last (every N mode
            // last)
            for(int s = 0; s < 128; s++)
            {
                std::map<std::string, int64_t> index;
                int64_t                        r = pick(rng);
                for(size_t i = 0; i < D.names.size(); i++)
                {
                    int64_t v = r % D.extents[i];
                    r /= D.extents[i];
                    if(s % 2 == 0 && D.names[i][0] == 'M')
                    {
                        v = 0;
                    }
                    if(s % 2 == 1 && D.names[i][0] == 'N')
                    {
                        v = D.extents[i] - 1;
                    }
                    index[D.names[i]] = v;
                }
                outs.push_back(aotOffset(D, index));
            }
        }
        double const inf = std::numeric_limits<double>::infinity();
        double const nan = std::numeric_limits<double>::quiet_NaN();

        Device dev;
        auto   upload = [&](AotTensor const& t) {
            void* p = nullptr;
            EXPECT_EQ(hipMalloc(&p, size_t(t.size) * 2), hipSuccess);
            EXPECT_EQ(hipMemcpy(p, t.bits.data(), size_t(t.size) * 2, hipMemcpyHostToDevice),
                      hipSuccess);
            dev.buffers.push_back(p);
            return p;
        };
        void* pD = nullptr;
        ASSERT_EQ(hipMalloc(&pD, size_t(D.size) * 2), hipSuccess);
        dev.buffers.push_back(pD);
        void* ws = nullptr;
        ASSERT_EQ(hipMalloc(&ws, workspaceSize + 16), hipSuccess);
        dev.buffers.push_back(ws);

        struct Variant
        {
            char const* what;
            double      alpha, beta;
            bool        poisonAB, poisonC, misalignWs;
        };
        std::vector<Variant> variants = {
            {"plain", 1.0, -2.0, false, false, false},
            {"inf/nan operands", 2.0, -1.0, true, false, false},
            {"alpha 0, A and B poisoned", 0.0, -2.0, true, false, false},
            {"misaligned workspace", 1.0, 3.0, false, false, true},
        };
        if(c.bilinear)
        {
            variants.push_back({"beta 0, C poisoned", 2.0, 0.0, false, true, false});
        }
        for(auto const& v : variants)
        {
            SCOPED_TRACE(v.what);
            AotTensor a = A, b = B, cc = C;
            if(v.poisonAB)
            {
                a.bits[0]                  = aotEncode(inf, c.type);
                b.bits[size_t(b.size - 1)] = aotEncode(nan, c.type);
                a.bits[size_t(a.size / 2)] = aotEncode(-inf, c.type);
            }
            if(v.poisonC)
            {
                for(int64_t o : outs)
                {
                    cc.bits[size_t(o)] = aotEncode(o % 2 ? nan : inf, c.type);
                }
            }
            void* pA = upload(a);
            void* pB = upload(b);
            void* pC = c.bilinear ? upload(cc) : nullptr;
            ASSERT_EQ(hipMemset(pD, 0x7f, size_t(D.size) * 2), hipSuccess);
            float  alpha = float(v.alpha), beta = float(v.beta);
            Stream stream;
            void*  w = v.misalignWs ? static_cast<uint8_t*>(ws) + 8 : ws;
            ASSERT_EQ(
                hiptensorContract(
                    api.handle, plan, &alpha, pA, pB, &beta, pC, pD, w, workspaceSize, stream.s),
                HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            std::vector<uint16_t> got(size_t(D.size));
            ASSERT_EQ(hipMemcpy(got.data(), pD, got.size() * 2, hipMemcpyDeviceToHost), hipSuccess);
            int64_t bad = 0, nonfinite = 0;
            for(int64_t flat : outs)
            {
                std::map<std::string, int64_t> index;
                int64_t                        rest = flat;
                for(size_t i = 0; i < D.names.size(); i++)
                {
                    index[D.names[i]] = rest % D.extents[i];
                    rest /= D.extents[i];
                }
                double sum = 0;
                if(v.alpha != 0.0)
                {
                    int64_t const baseA = aotOffset(a, index), baseB = aotOffset(b, index);
                    for(auto const& kk : koffs)
                    {
                        sum += aotDecode(a.bits[size_t(baseA + kk.first)], c.type)
                               * aotDecode(b.bits[size_t(baseB + kk.second)], c.type);
                    }
                }
                double want = v.alpha != 0.0 ? v.alpha * sum : 0.0;
                if(c.bilinear && v.beta != 0.0)
                {
                    want += v.beta * aotDecode(cc.bits[size_t(flat)], c.type);
                }
                want       = roundTo(want, c.type);
                double out = aotDecode(got[size_t(flat)], c.type);
                nonfinite += !std::isfinite(want);
                bool same = std::isnan(want) ? std::isnan(out) : out == want;
                if(!same && bad++ < 5)
                {
                    ADD_FAILURE() << "element " << flat << " got " << out << " want " << want;
                }
            }
            EXPECT_EQ(bad, 0);
            // the infinity / NaN variants reach the outputs that read them; the zero-scale
            // variants never do
            if(v.poisonAB && v.alpha != 0.0)
            {
                EXPECT_GT(nonfinite, 0);
            }
            if(v.alpha == 0.0 || v.poisonC)
            {
                EXPECT_EQ(nonfinite, 0);
            }
        }
    }

    std::vector<AotGettCase> aotGettCases()
    {
        using M = std::map<std::string, int64_t>;
        return {
            {"mn f16 bilinear",
             HIPTENSOR_R_16F,
             true,
             {"K1", "M1", "K0", "M0"},
             {"K1", "N1", "K0", "N0"},
             {"N1", "M1", "N0", "M0"},
             M{{"K0", 24}, {"K1", 24}, {"M0", 48}, {"M1", 32}, {"N0", 40}, {"N1", 36}}},
            {"mn + B pack, bf16 scale",
             HIPTENSOR_R_16BF,
             false,
             {"K2", "M2", "K1", "M1", "K0", "M0"},
             {"K2", "N2", "K1", "N1", "K0", "N0"},
             {"N2", "M2", "N1", "M1", "N0", "M0"},
             M{{"K0", 12},
               {"K1", 8},
               {"K2", 6},
               {"M0", 24},
               {"M1", 16},
               {"M2", 8},
               {"N0", 20},
               {"N1", 12},
               {"N2", 8}}},
            {"mn + A, B packs, f16 bilinear",
             HIPTENSOR_R_16F,
             true,
             {"K4", "M4", "K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"},
             {"K4", "N4", "K3", "N3", "K2", "N2", "K1", "N1", "K0", "N0"},
             {"N4", "M4", "N3", "M3", "N2", "M2", "N1", "M1", "N0", "M0"},
             M{{"K0", 6},
               {"K1", 4},
               {"K2", 4},
               {"K3", 4},
               {"K4", 2},
               {"M0", 10},
               {"M1", 8},
               {"M2", 8},
               {"M3", 6},
               {"M4", 4},
               {"N0", 10},
               {"N1", 8},
               {"N2", 6},
               {"N3", 8},
               {"N4", 4}}},
            {"K-contiguous A, B, bf16 bilinear",
             HIPTENSOR_R_16BF,
             true,
             {"K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"},
             {"K3", "N3", "K2", "N2", "K1", "N1", "K0", "N0"},
             {"N3", "M3", "N2", "M2", "N1", "M1", "N0", "M0"},
             M{{"K0", 6},
               {"K1", 6},
               {"K2", 4},
               {"K3", 4},
               {"M0", 12},
               {"M1", 8},
               {"M2", 8},
               {"M3", 6},
               {"N0", 10},
               {"N1", 8},
               {"N2", 6},
               {"N3", 8}}},
            {"K-contiguous A, B, f16 scale",
             HIPTENSOR_R_16F,
             false,
             {"K3", "M3", "K2", "M2", "K1", "M1", "K0", "M0"},
             {"K3", "N3", "K2", "N2", "K1", "N1", "K0", "N0"},
             {"N3", "M3", "N2", "M2", "N1", "M1", "N0", "M0"},
             M{{"K0", 6},
               {"K1", 6},
               {"K2", 4},
               {"K3", 4},
               {"M0", 12},
               {"M1", 8},
               {"M2", 8},
               {"M3", 6},
               {"N0", 10},
               {"N1", 8},
               {"N2", 6},
               {"N3", 8}}},
            {"K-contiguous B, A, f16 scale",
             HIPTENSOR_R_16F,
             false,
             {"K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"},
             {"K0", "K1", "K2", "K3", "N3", "N2", "N1", "N0"},
             {"N3", "N2", "N1", "N0", "M3", "M2", "M1", "M0"},
             M{{"K0", 8},
               {"K1", 8},
               {"K2", 6},
               {"K3", 4},
               {"M0", 16},
               {"M1", 12},
               {"M2", 10},
               {"M3", 8},
               {"N0", 14},
               {"N1", 10},
               {"N2", 12},
               {"N3", 8}}},
        };
    }

    class RockeAotGett : public ::testing::TestWithParam<int>
    {
    };

    TEST_P(RockeAotGett, Outputs)
    {
        if(f64RouteOnly())
        {
            GTEST_SKIP() << "HIPTENSOR_ROCKE_CONTRACTION_ROUTE=f64";
        }
        runAotGettCase(aotGettCases()[size_t(GetParam())]);
    }

    INSTANTIATE_TEST_SUITE_P(Routes,
                             RockeAotGett,
                             ::testing::Range(0, int(aotGettCases().size())),
                             [](::testing::TestParamInfo<int> const& info) {
                                 std::string n = aotGettCases()[size_t(info.param)].name;
                                 for(auto& ch : n)
                                 {
                                     if(!std::isalnum(static_cast<unsigned char>(ch)))
                                     {
                                         ch = '_';
                                     }
                                 }
                                 return n;
                             });

    // ---------------------------------------------------------------------- elementwise

    struct ElementwiseCase
    {
        int                 arity; // 1 permutation, 2 binary, 3 trinary
        hiptensorDataType_t in;
        hiptensorDataType_t out;
        hiptensorDataType_t scalar;
        int                 seed;
    };

    double unaryRef(hiptensorOperator_t op, double x)
    {
        switch(op)
        {
        case HIPTENSOR_OP_NEG:
            return -x;
        case HIPTENSOR_OP_ABS:
            return std::fabs(x);
        case HIPTENSOR_OP_RELU:
            return x > 0 ? x : 0;
        case HIPTENSOR_OP_IDENTITY:
        case HIPTENSOR_OP_CONJ:
            return x;
        default:
            // the exact operators only (RockeOperatorReduction checks the others within bounds)
            ADD_FAILURE() << "unaryRef: operator " << int(op) << " is not an exact operator";
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    double binaryRef(hiptensorOperator_t op, double a, double c)
    {
        switch(op)
        {
        case HIPTENSOR_OP_MUL:
            return a * c;
        case HIPTENSOR_OP_MAX:
            return std::max(a, c);
        case HIPTENSOR_OP_MIN:
            return std::min(a, c);
        default:
            return a + c;
        }
    }

    void runElementwise(ElementwiseCase const& c)
    {
        std::mt19937 rng(c.seed);
        ApiObjects   api;
        if(!api.supports(c.in))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture                 log;
        int                        rank = 2 + rng() % 4;
        std::vector<int32_t>       modes;
        std::map<int32_t, int64_t> ext;
        std::vector<int64_t>       pick = {1, 2, 3, 7, 16, 31};
        for(int i = 0; i < rank; i++)
        {
            modes.push_back('a' + i);
            ext['a' + i] = pick[rng() % pick.size()];
        }
        ext['a']      = 33;
        auto shuffled = [&] {
            auto m = modes;
            std::shuffle(m.begin(), m.end(), rng);
            return m;
        };
        std::vector<Tensor> ins;
        for(int i = 0; i < c.arity; i++)
        {
            ins.push_back(makeTensor(c.in, shuffled(), ext, rng, true));
        }
        auto D = makeTensor(c.out, shuffled(), ext, rng, true);
        std::fill(D.values.begin(), D.values.end(), cplx(0, 0));
        hiptensorOperator_t const ops[]
            = {HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_NEG, HIPTENSOR_OP_ABS, HIPTENSOR_OP_RELU};
        hiptensorOperator_t const bops[]
            = {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL, HIPTENSOR_OP_MAX, HIPTENSOR_OP_MIN};
        std::vector<hiptensorOperator_t> uop;
        for(int i = 0; i < c.arity; i++)
        {
            uop.push_back(ops[rng() % 4]);
        }
        auto         opAB  = bops[rng() % 4];
        auto         opABC = bops[rng() % 4];
        Scalar       s[3];
        double const svals[] = {2.0, -1.0, 0.5};
        for(int i = 0; i < 3; i++)
        {
            s[i].re = svals[(i + c.seed) % 3];
        }
        auto compute = c.scalar == HIPTENSOR_R_64F    ? HIPTENSOR_COMPUTE_DESC_64F
                       : c.scalar == HIPTENSOR_R_16F  ? HIPTENSOR_COMPUTE_DESC_16F
                       : c.scalar == HIPTENSOR_R_16BF ? HIPTENSOR_COMPUTE_DESC_16BF
                                                      : HIPTENSOR_COMPUTE_DESC_32F;
        hiptensorOperationDescriptor_t           desc = nullptr;
        std::vector<hiptensorTensorDescriptor_t> d;
        for(auto const& t : ins)
        {
            d.push_back(describe(api, t));
        }
        auto              dD = describe(api, D);
        hiptensorStatus_t created;
        if(c.arity == 1)
        {
            created = hiptensorCreatePermutation(
                api.handle, &desc, d[0], ins[0].modes.data(), uop[0], dD, D.modes.data(), compute);
        }
        else if(c.arity == 2)
        {
            created = hiptensorCreateElementwiseBinary(api.handle,
                                                       &desc,
                                                       d[0],
                                                       ins[0].modes.data(),
                                                       uop[0],
                                                       d[1],
                                                       ins[1].modes.data(),
                                                       uop[1],
                                                       dD,
                                                       D.modes.data(),
                                                       opAB,
                                                       compute);
        }
        else
        {
            created = hiptensorCreateElementwiseTrinary(api.handle,
                                                        &desc,
                                                        d[0],
                                                        ins[0].modes.data(),
                                                        uop[0],
                                                        d[1],
                                                        ins[1].modes.data(),
                                                        uop[1],
                                                        d[2],
                                                        ins[2].modes.data(),
                                                        uop[2],
                                                        dD,
                                                        D.modes.data(),
                                                        opAB,
                                                        opABC,
                                                        compute);
        }
        ASSERT_EQ(created, HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t workspaceSize = 0;
        auto     plan          = planOf(api, desc, workspaceSize, log);
        if(plan == nullptr)
        {
            return;
        }
        Device             dev;
        std::vector<void*> p;
        for(auto const& t : ins)
        {
            p.push_back(dev.upload(t));
        }
        void*             pD = dev.upload(D);
        Stream            stream;
        hiptensorStatus_t status;
        if(c.arity == 1)
        {
            status = hiptensorPermute(api.handle, plan, s[0].as(c.scalar), p[0], pD, stream.s);
        }
        else if(c.arity == 2)
        {
            status = hiptensorElementwiseBinaryExecute(
                api.handle, plan, s[0].as(c.scalar), p[0], s[2].as(c.scalar), p[1], pD, stream.s);
        }
        else
        {
            Scalar a = s[0], b = s[1], g = s[2];
            status = hiptensorElementwiseTrinaryExecute(api.handle,
                                                        plan,
                                                        a.as(c.scalar),
                                                        p[0],
                                                        b.as(c.scalar),
                                                        p[1],
                                                        g.as(c.scalar),
                                                        p[2],
                                                        pD,
                                                        stream.s);
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);
        std::vector<cplx>   want(D.values.size(), 0);
        std::vector<double> bound(D.values.size(), 0);
        std::vector<bool>   written(D.values.size(), false);
        double const        scale[] = {s[0].re, c.arity == 3 ? s[1].re : s[2].re, s[2].re};
        forEachIndex(modes, ext, [&](auto index) {
            std::vector<double> t;
            for(int i = 0; i < c.arity; i++)
            {
                double sc = c.arity == 2 && i == 1 ? s[2].re : scale[i];
                t.push_back(sc * unaryRef(uop[i], ins[i].values[ins[i].offset(index)].real()));
            }
            double r = t[0];
            if(c.arity >= 2)
            {
                r = binaryRef(opAB, t[0], t[1]);
            }
            if(c.arity == 3)
            {
                r = binaryRef(opABC, r, t[2]);
            }
            auto off     = D.offset(index);
            want[off]    = r;
            written[off] = true;
        });
        compare(got, want, bound, written, true, "elementwise");
    }

    class RockeElementwise : public ::testing::TestWithParam<ElementwiseCase>
    {
    };

    TEST_P(RockeElementwise, MatchesLoopReference)
    {
        runElementwise(GetParam());
    }

    std::vector<ElementwiseCase> elementwiseCases()
    {
        std::vector<ElementwiseCase> out;
        int                          seed = 500;
        struct Row
        {
            int                 arity;
            hiptensorDataType_t in, out, scalar;
        };
        Row const rows[] = {{1, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                            {1, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_32F},
                            {1, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                            {1, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F},
                            {2, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                            {2, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_32F},
                            {2, HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_R_32F},
                            {2, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                            {2, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F},
                            {3, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                            {3, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                            {3, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F}};
        for(auto const& r : rows)
        {
            for(int i = 0; i < 3; i++)
            {
                out.push_back({r.arity, r.in, r.out, r.scalar, seed++});
            }
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke, RockeElementwise, ::testing::ValuesIn(elementwiseCases()));

    // ---------------------------------------------------------------------- reduction

    struct ReductionCase
    {
        hiptensorDataType_t          type;
        hiptensorComputeDescriptor_t compute;
        hiptensorOperator_t          op;
        int                          seed;
    };

    void runReduction(ReductionCase const& c,
                      Choice const&        choice    = {},
                      SelectionLog*        selection = nullptr)
    {
        std::mt19937              rng(c.seed);
        std::optional<ApiObjects> own;
        if(choice.api == nullptr)
        {
            own.emplace();
        }
        ApiObjects& api = choice.api != nullptr ? *choice.api : *own;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture                 log;
        int                        rank = 2 + rng() % 4;
        std::vector<int32_t>       modes;
        std::map<int32_t, int64_t> ext;
        std::vector<int64_t>       pick = c.op == HIPTENSOR_OP_MUL
                                              ? std::vector<int64_t>{1, 2, 3, 4}
                                              : std::vector<int64_t>{2, 3, 7, 16, 65};
        for(int i = 0; i < rank; i++)
        {
            modes.push_back('a' + i);
            ext['a' + i] = pick[rng() % pick.size()];
        }
        auto ma = modes;
        std::shuffle(ma.begin(), ma.end(), rng);
        int                  kept = 1 + rng() % (rank - 1);
        std::vector<int32_t> md(ma.begin(), ma.begin() + kept);
        std::shuffle(md.begin(), md.end(), rng);
        std::vector<int32_t> red;
        for(auto m : modes)
        {
            if(std::find(md.begin(), md.end(), m) == md.end())
            {
                red.push_back(m);
            }
        }
        // products of values in [-1, 1] stay exact (and finite) for any count
        int  range = c.op == HIPTENSOR_OP_MUL ? 1 : 3;
        auto A     = makeTensor(c.type, ma, ext, rng, true, range);
        auto C     = makeTensor(c.type, md, ext, rng, true);
        auto D     = C;
        std::fill(D.values.begin(), D.values.end(), cplx(0, 0));
        auto   scalarType = scalarTypeOf(c.compute);
        Scalar alpha, beta;
        alpha.re = 2.0;
        beta.re  = (c.seed % 2) ? -1.0 : 0.0;

        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dC = describe(api, C), dD = describe(api, D);
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &desc,
                                           dA,
                                           A.modes.data(),
                                           HIPTENSOR_OP_IDENTITY,
                                           dC,
                                           C.modes.data(),
                                           HIPTENSOR_OP_IDENTITY,
                                           dD,
                                           D.modes.data(),
                                           c.op,
                                           c.compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t workspaceSize = 0;
        auto     plan          = planWith(api, desc, choice, log, workspaceSize, selection);
        if(plan == nullptr)
        {
            return;
        }
        Device dev;
        void*  pA = dev.upload(A);
        void*  pC = dev.upload(C);
        void*  pD = dev.upload(D);
        void*  ws = nullptr;
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            dev.buffers.push_back(ws);
        }
        Stream stream;
        ASSERT_EQ(hiptensorReduce(api.handle,
                                  plan,
                                  alpha.as(scalarType),
                                  pA,
                                  beta.as(scalarType),
                                  pC,
                                  pD,
                                  ws,
                                  workspaceSize,
                                  stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);
        std::vector<cplx>   want(D.values.size(), 0);
        std::vector<double> bound(D.values.size(), 0);
        std::vector<bool>   written(D.values.size(), false);
        forEachIndex(md, ext, [&](auto index) {
            double r = c.op == HIPTENSOR_OP_MUL   ? 1.0
                       : c.op == HIPTENSOR_OP_MAX ? -INFINITY
                       : c.op == HIPTENSOR_OP_MIN ? INFINITY
                                                  : 0.0;
            forEachIndex(red, ext, [&](auto rindex) {
                auto full = index;
                for(auto& kv : rindex)
                {
                    full[kv.first] = kv.second;
                }
                r = binaryRef(c.op, r, A.values[A.offset(full)].real());
            });
            auto off     = D.offset(index);
            want[off]    = alpha.re * r + beta.re * C.values[C.offset(index)].real();
            written[off] = true;
        });
        compare(got, want, bound, written, true, "reduction");
    }

    class RockeReduction : public ::testing::TestWithParam<ReductionCase>
    {
    };

    TEST_P(RockeReduction, MatchesLoopReference)
    {
        runReduction(GetParam());
    }

    std::vector<ReductionCase> reductionCases()
    {
        std::vector<ReductionCase> out;
        int                        seed = 900;
        struct Row
        {
            hiptensorDataType_t          type;
            hiptensorComputeDescriptor_t compute;
        };
        Row const rows[] = {{HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F},
                            {HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_32F},
                            {HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F},
                            {HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F}};
        for(auto const& r : rows)
        {
            for(auto op : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL, HIPTENSOR_OP_MAX, HIPTENSOR_OP_MIN})
            {
                for(int i = 0; i < 3; i++)
                {
                    out.push_back({r.type, r.compute, op, seed++});
                }
            }
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke, RockeReduction, ::testing::ValuesIn(reductionCases()));

    // --------------------------------------------------------- tuned reduction classes

    // MAX / MIN of the reduction reference: NaN propagates (hipTensor's reductions)
    double reduceRef(hiptensorOperator_t op, double r, double x)
    {
        if((op == HIPTENSOR_OP_MAX || op == HIPTENSOR_OP_MIN) && (std::isnan(r) || std::isnan(x)))
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return binaryRef(op, r, x);
    }

    double reduceIdentity(hiptensorOperator_t op)
    {
        return op == HIPTENSOR_OP_MUL   ? 1.0
               : op == HIPTENSOR_OP_MAX ? -INFINITY
               : op == HIPTENSOR_OP_MIN ? INFINITY
                                        : 0.0;
    }

    // A reduction the tuned classes serve: A's modes / extents (A stored in a random mode order
    // with padding), D's modes (stored in a random order), the operators, scales, NaN data and
    // misaligned buffers. Integer data: every result is exact after one rounding.
    struct TunedReductionCase
    {
        hiptensorDataType_t  type;
        hiptensorOperator_t  op;
        std::string          modesA;
        std::string          modesD;
        std::vector<int64_t> extents; // per mode of modesA
        int                  seed;
        hiptensorOperator_t  opA     = HIPTENSOR_OP_IDENTITY;
        hiptensorOperator_t  opC     = HIPTENSOR_OP_IDENTITY;
        double               alpha   = 1.0;
        double               beta    = 0.0;
        int                  nanA    = 0; // NaN elements of A
        bool                 nanC    = false; // C all NaN (beta 0: never read)
        bool                 allNanA = false; // A all NaN (alpha 0: never read)
        size_t               skew    = 0; // A, C, D this many elements past aligned addresses
    };

    std::ostream& operator<<(std::ostream& os, TunedReductionCase const& c)
    {
        return os << "type " << int(c.type) << " op " << int(c.op) << " A " << c.modesA << " D '"
                  << c.modesD << "' seed " << c.seed;
    }

    class RockeTunedReduction : public ::testing::TestWithParam<TunedReductionCase>
    {
    };

    TEST_P(RockeTunedReduction, ExactAndPicked)
    {
        auto const   c = GetParam();
        std::mt19937 rng(c.seed);
        ApiObjects   api;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture                 log;
        std::vector<int32_t>       ma(c.modesA.begin(), c.modesA.end());
        std::vector<int32_t>       md(c.modesD.begin(), c.modesD.end());
        std::map<int32_t, int64_t> ext;
        for(size_t i = 0; i < ma.size(); i++)
        {
            ext[ma[i]] = c.extents[i];
        }
        std::vector<int32_t> red;
        for(auto m : ma)
        {
            if(std::find(md.begin(), md.end(), m) == md.end())
            {
                red.push_back(m);
            }
        }
        int  range = c.op == HIPTENSOR_OP_MUL ? 1 : 3;
        auto A     = makeTensor(c.type, ma, ext, rng, true, range);
        auto C     = makeTensor(c.type, md, ext, rng, true);
        auto D     = C;
        std::fill(D.values.begin(), D.values.end(), cplx(-77, 0));
        if(c.allNanA)
        {
            std::fill(A.values.begin(), A.values.end(), cplx(NAN, 0));
        }
        for(int i = 0; i < c.nanA; i++)
        {
            A.values[rng() % A.values.size()] = cplx(NAN, 0);
        }
        auto Cread = C;
        if(c.nanC)
        {
            std::fill(C.values.begin(), C.values.end(), cplx(NAN, 0));
        }
        auto const scalarType = c.type == HIPTENSOR_R_64F ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
        auto const compute
            = c.type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
        Scalar alpha, beta;
        alpha.re = c.alpha;
        beta.re  = c.beta;

        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dC = describe(api, C), dD = describe(api, D);
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &desc,
                                           dA,
                                           A.modes.data(),
                                           c.opA,
                                           dC,
                                           C.modes.data(),
                                           c.opC,
                                           dD,
                                           D.modes.data(),
                                           c.op,
                                           compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t     workspaceSize = 0;
        SelectionLog sel;
        auto         plan = planWith(api, desc, {}, log, workspaceSize, &sel);
        ASSERT_NE(plan, nullptr);
        // the pack of every rocKE target carries the tuned classes: the rule picks one
        EXPECT_EQ(sel.pick.rfind("rreduce:", 0), 0u) << sel.pick << "\n" << log.all();
        Device dev;
        void*  pA = dev.upload(A, c.skew);
        void*  pC = dev.upload(C, c.skew);
        void*  pD = dev.upload(D, c.skew);
        void*  ws = nullptr;
        if(workspaceSize > 0)
        {
            // NaN-filled: a partial that is not written before it is read shows
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            ASSERT_EQ(hipMemset(ws, 0xFF, workspaceSize), hipSuccess);
            dev.buffers.push_back(ws);
        }
        Stream stream;
        ASSERT_EQ(hiptensorReduce(api.handle,
                                  plan,
                                  alpha.as(scalarType),
                                  pA,
                                  beta.as(scalarType),
                                  pC,
                                  pD,
                                  ws,
                                  workspaceSize,
                                  stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);
        int64_t bad = 0;
        forEachIndex(md, ext, [&](auto index) {
            double r = reduceIdentity(c.op);
            if(c.alpha != 0.0)
            {
                forEachIndex(red, ext, [&](auto rindex) {
                    auto full = index;
                    for(auto& kv : rindex)
                    {
                        full[kv.first] = kv.second;
                    }
                    r = reduceRef(c.op, r, unaryRef(c.opA, A.values[A.offset(full)].real()));
                });
            }
            double want = c.alpha != 0.0 ? c.alpha * r : 0.0;
            if(c.beta != 0.0)
            {
                want += c.beta * unaryRef(c.opC, Cread.values[Cread.offset(index)].real());
            }
            want            = roundTo(want, c.type);
            double const g  = got.values[got.offset(index)].real();
            bool const   ok = std::isnan(want) ? std::isnan(g) : g == want;
            if(!ok && bad++ < 5)
            {
                ADD_FAILURE() << "output " << got.offset(index) << " got " << g << " want " << want
                              << " (" << sel.pick << ")";
            }
        });
        EXPECT_EQ(bad, 0) << sel.pick;
    }

    std::vector<TunedReductionCase> tunedReductionCases()
    {
        std::vector<TunedReductionCase> out;
        int                             seed = 2600;
        auto                            add  = [&](TunedReductionCase c) {
            c.seed = seed++;
            out.push_back(c);
        };
        for(auto type : {HIPTENSOR_R_16F, HIPTENSOR_R_16BF, HIPTENSOR_R_32F, HIPTENSOR_R_64F})
        {
            for(auto op : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL, HIPTENSOR_OP_MAX, HIPTENSOR_OP_MIN})
            {
                // the innermost stored mode reduced (row classes) or kept (col classes), a full
                // reduction (split partials), two reduced runs, three kept runs
                add({type, op, "ab", "b", {4096, 96}, 0});
                add({type, op, "ab", "a", {96, 3000}, 0});
                add({type, op, "a", "", {1 << 20}, 0});
                add({type, op, "abc", "ca", {64, 129, 48}, 0});
                add({type, op, "abcde", "bd", {24, 9, 16, 7, 33}, 0});
                // scales and operators: beta with opC, a NaN C that beta 0 never reads, an A of
                // NaN that alpha 0 never reads
                add({type,
                     op,
                     "ab",
                     "b",
                     {512, 40},
                     0,
                     HIPTENSOR_OP_IDENTITY,
                     HIPTENSOR_OP_NEG,
                     2.0,
                     -1.0});
                add({type,
                     op,
                     "ab",
                     "a",
                     {40, 512},
                     0,
                     HIPTENSOR_OP_IDENTITY,
                     HIPTENSOR_OP_IDENTITY,
                     0.5,
                     0.0,
                     0,
                     true});
                add({type,
                     op,
                     "ab",
                     "a",
                     {40, 512},
                     0,
                     HIPTENSOR_OP_IDENTITY,
                     HIPTENSOR_OP_ABS,
                     0.0,
                     2.0,
                     0,
                     false,
                     true});
                // NaN reaches exactly the outputs whose reduction reads one (MAX / MIN too)
                add({type,
                     op,
                     "abc",
                     "c",
                     {256, 3, 64},
                     0,
                     HIPTENSOR_OP_IDENTITY,
                     HIPTENSOR_OP_IDENTITY,
                     1.0,
                     0.0,
                     5});
                add({type,
                     op,
                     "abc",
                     "a",
                     {64, 3, 256},
                     0,
                     HIPTENSOR_OP_IDENTITY,
                     HIPTENSOR_OP_IDENTITY,
                     1.0,
                     0.0,
                     5});
                // opA (the una classes)
                for(auto opA : {HIPTENSOR_OP_ABS, HIPTENSOR_OP_NEG, HIPTENSOR_OP_RELU})
                {
                    add({type,
                         op,
                         "abc",
                         op == HIPTENSOR_OP_MUL ? "c" : "b",
                         {300, 33, 20},
                         0,
                         opA});
                }
                // A, C and D one element past aligned addresses: the vector classes fall back
                add({type,
                     op,
                     "ab",
                     "b",
                     {1024, 64},
                     0,
                     HIPTENSOR_OP_IDENTITY,
                     HIPTENSOR_OP_IDENTITY,
                     1.0,
                     1.0,
                     0,
                     false,
                     false,
                     1});
                add({type,
                     op,
                     "ab",
                     "a",
                     {64, 1024},
                     0,
                     HIPTENSOR_OP_IDENTITY,
                     HIPTENSOR_OP_IDENTITY,
                     1.0,
                     1.0,
                     0,
                     false,
                     false,
                     1});
            }
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeTunedReduction,
                             ::testing::ValuesIn(tunedReductionCases()));

    // The planner's untuned choice is rocKE's (kernels.common.hiptensor_aot.rreduce_host
    // .untuned_choices): the first candidate of these problems (packed column-major A and D, not in
    // the route table) is the one rocKE's test_hiptensor_aot_rreduce_table.py PICKS_GFX950 expects
    // for gfx950 with 256 compute units. Regenerate both lists when the table's palette or
    // prototypes change.
    struct TunedPickCase
    {
        hiptensorDataType_t  type;
        hiptensorOperator_t  op;
        std::string          modesA;
        std::string          modesD;
        std::vector<int64_t> extents;
        bool                 una;
        std::string          pick;
    };

    std::ostream& operator<<(std::ostream& os, TunedPickCase const& c)
    {
        return os << "type " << int(c.type) << " op " << int(c.op) << " A " << c.modesA << " D '"
                  << c.modesD << "'";
    }

    class RockeTunedPick : public ::testing::TestWithParam<TunedPickCase>
    {
    };

    TEST_P(RockeTunedPick, SameAsRocke)
    {
        auto const      c      = GetParam();
        int             device = 0;
        hipDeviceProp_t props;
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
        if(deviceArch() != "gfx950" || props.multiProcessorCount != 256)
        {
            GTEST_SKIP() << "the expected picks are rocKE's for gfx950 with 256 compute units";
        }
        ApiObjects api;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture                 log;
        std::map<int32_t, int64_t> ext;
        Tensor                     A, D;
        A.type = D.type = c.type;
        int64_t stride  = 1;
        for(size_t i = 0; i < c.modesA.size(); i++)
        {
            ext[c.modesA[i]] = c.extents[i];
            A.modes.push_back(c.modesA[i]);
            A.extents.push_back(c.extents[i]);
            A.strides.push_back(stride);
            stride *= c.extents[i];
        }
        stride = 1;
        for(char m : c.modesD)
        {
            D.modes.push_back(m);
            D.extents.push_back(ext.at(m));
            D.strides.push_back(stride);
            stride *= ext.at(m);
        }
        auto const compute
            = c.type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
        hiptensorOperationDescriptor_t desc = nullptr;
        auto                           dA = describe(api, A), dD = describe(api, D);
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &desc,
                                           dA,
                                           A.modes.data(),
                                           c.una ? HIPTENSOR_OP_ABS : HIPTENSOR_OP_IDENTITY,
                                           dD,
                                           D.modes.data(),
                                           HIPTENSOR_OP_IDENTITY,
                                           dD,
                                           D.modes.data(),
                                           c.op,
                                           compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t     workspaceSize = 0;
        SelectionLog sel;
        auto         plan = planWith(api, desc, {}, log, workspaceSize, &sel);
        ASSERT_NE(plan, nullptr);
        EXPECT_EQ(sel.pick, c.pick) << log.all();
    }

    INSTANTIATE_TEST_SUITE_P(
        Rocke,
        RockeTunedPick,
        ::testing::Values(
            TunedPickCase{
                HIPTENSOR_R_16F,
                HIPTENSOR_OP_ADD,
                "ab",
                "b",
                {3000, 70},
                false,
                "rreduce:hiptensor_aot_rreduce_row_add_f16_f16_r1_b256_v8_l256_u4_nt_i32:split2"},
            TunedPickCase{
                HIPTENSOR_R_32F,
                HIPTENSOR_OP_MAX,
                "ab",
                "a",
                {70, 3000},
                false,
                "rreduce:hiptensor_aot_rreduce_col_max_f32_f32_r1_b512_v2_l32_u8_nt_i32:split128"},
            TunedPickCase{
                HIPTENSOR_R_16BF,
                HIPTENSOR_OP_MIN,
                "abc",
                "b",
                {100, 50, 30},
                false,
                "rreduce:hiptensor_aot_rreduce_row_min_bf16_bf16_r2_b512_v4_l512_u4_nt_i32:split2"},
            TunedPickCase{
                HIPTENSOR_R_64F,
                HIPTENSOR_OP_ADD,
                "abcd",
                "bd",
                {17, 33, 9, 40},
                false,
                "rreduce:hiptensor_aot_rreduce_row_add_f64_f64_r8_b256_v1_l1_u4_nt_i32:split128"},
            TunedPickCase{
                HIPTENSOR_R_32F,
                HIPTENSOR_OP_ADD,
                "a",
                "",
                {1000003},
                false,
                "rreduce:hiptensor_aot_rreduce_row_add_f32_f32_r8_b256_v1_l1_u4_nt_i32:split1024"},
            TunedPickCase{HIPTENSOR_R_16F,
                          HIPTENSOR_OP_MAX,
                          "abc",
                          "ac",
                          {8, 2000, 12},
                          false,
                          "rreduce:hiptensor_aot_rreduce_col_max_f16_f16_r2_b256_v8_l1_w8_u4_nt_"
                          "i32:split128"},
            TunedPickCase{HIPTENSOR_R_32F,
                          HIPTENSOR_OP_MUL,
                          "abc",
                          "c",
                          {640, 3, 50},
                          true,
                          "rreduce:hiptensor_aot_rreduce_row_mul_f32_f32_r2_b256_v4_l32_u8_nt_una_"
                          "i32:split8"},
            TunedPickCase{
                HIPTENSOR_R_64F,
                HIPTENSOR_OP_MAX,
                "abcde",
                "eb",
                {6, 7, 8, 9, 10},
                false,
                "rreduce:hiptensor_aot_rreduce_row_max_f64_f64_r2_b1024_v2_l16_u4_nt_i32:split8"}));

    // The problems of rocKE's reduction route table run with their tuned knobs (the reduction doc
    // rows, packed column-major, as hipTensor's benchmarks describe them): device-format data
    // generated in place (no per-element host records), an exact reference over A's storage.
    // With a workspace limit of zero or the minimum estimate they keep the tuned kernels with
    // the split lowered to fit (no split at zero); HIPTENSOR_ROCKE_DISABLE_TUNED runs the generic
    // kernels instead.
    enum class TableWorkspace
    {
        Estimate, // the default estimate
        Zero, // a workspace limit of zero
        Minimum, // HIPTENSOR_WORKSPACE_MIN
        DisableTuned, // HIPTENSOR_ROCKE_DISABLE_TUNED set
    };

    struct TableReductionCase
    {
        hiptensorDataType_t  type;
        hiptensorOperator_t  op;
        std::string          modesA;
        std::string          modesD;
        std::vector<int64_t> extents;
        TableWorkspace       ws = TableWorkspace::Estimate;
    };

    std::ostream& operator<<(std::ostream& os, TableReductionCase const& c)
    {
        return os << "type " << int(c.type) << " op " << int(c.op) << " A " << c.modesA << " D '"
                  << c.modesD << "' workspace " << int(c.ws);
    }

    class RockeTableReduction : public ::testing::TestWithParam<TableReductionCase>
    {
    };

    // value i of the generated A: an integer in [-2, 2] (+-1 for products)
    double tableValue(uint64_t i, bool pm1)
    {
        uint64_t h = (i + 0x51ED27ull) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
        return pm1 ? ((h >> 7) & 1 ? 1.0 : -1.0) : double(int64_t((h >> 7) % 5) - 2);
    }

    void putRaw(uint8_t* p, hiptensorDataType_t t, double v)
    {
        switch(t)
        {
        case HIPTENSOR_R_16F:
        {
            _Float16 x = static_cast<_Float16>(v);
            std::memcpy(p, &x, 2);
            break;
        }
        case HIPTENSOR_R_16BF:
        {
            hip_bfloat16 x(static_cast<float>(v));
            std::memcpy(p, &x, 2);
            break;
        }
        case HIPTENSOR_R_32F:
        {
            float x = static_cast<float>(v);
            std::memcpy(p, &x, 4);
            break;
        }
        default:
            std::memcpy(p, &v, 8);
            break;
        }
    }

    double getRaw(uint8_t const* p, hiptensorDataType_t t)
    {
        switch(t)
        {
        case HIPTENSOR_R_16F:
        {
            _Float16 x;
            std::memcpy(&x, p, 2);
            return static_cast<double>(x);
        }
        case HIPTENSOR_R_16BF:
        {
            hip_bfloat16 x;
            std::memcpy(&x, p, 2);
            return static_cast<double>(static_cast<float>(x));
        }
        case HIPTENSOR_R_32F:
        {
            float x;
            std::memcpy(&x, p, 4);
            return x;
        }
        default:
        {
            double x;
            std::memcpy(&x, p, 8);
            return x;
        }
        }
    }

    TEST_P(RockeTableReduction, TunedRouteExact)
    {
        auto const c = GetParam();
        ApiObjects api;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture           log;
        size_t const         eb = sizeOf(c.type);
        std::vector<int64_t> extD, strideA(c.modesA.size()), strideD;
        int64_t              nA = 1;
        for(size_t i = 0; i < c.modesA.size(); i++)
        {
            strideA[i] = nA;
            nA *= c.extents[i];
        }
        int64_t nD = 1;
        // D's stride of each mode of A (0: reduced)
        std::vector<int64_t> dOfA(c.modesA.size(), 0);
        for(char m : c.modesD)
        {
            auto i  = c.modesA.find(m);
            dOfA[i] = nD;
            extD.push_back(c.extents[i]);
            strideD.push_back(nD);
            nD *= c.extents[i];
        }
        std::vector<int32_t>           ma(c.modesA.begin(), c.modesA.end());
        std::vector<int32_t>           md(c.modesD.begin(), c.modesD.end());
        hiptensorOperationDescriptor_t desc = nullptr;
        auto                           dA   = api.tensor(c.extents, strideA, c.type);
        auto                           dD   = api.tensor(extD, strideD, c.type);
        auto const                     compute
            = c.type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &desc,
                                           dA,
                                           ma.data(),
                                           HIPTENSOR_OP_IDENTITY,
                                           dD,
                                           md.data(),
                                           HIPTENSOR_OP_IDENTITY,
                                           dD,
                                           md.data(),
                                           c.op,
                                           compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t     workspaceSize = 0;
        SelectionLog sel;
        Choice       choice;
        choice.zeroWorkspace = c.ws == TableWorkspace::Zero;
        if(c.ws == TableWorkspace::Minimum)
        {
            choice.workspace = HIPTENSOR_WORKSPACE_MIN;
        }
        std::unique_ptr<ScopedEnv> disable;
        if(c.ws == TableWorkspace::DisableTuned)
        {
            disable = std::make_unique<ScopedEnv>("HIPTENSOR_ROCKE_DISABLE_TUNED", "1");
        }
        auto plan = planWith(api, desc, choice, log, workspaceSize, &sel);
        disable.reset();
        ASSERT_NE(plan, nullptr);
        if(c.ws == TableWorkspace::DisableTuned)
        {
            // the generic kernels only
            EXPECT_EQ(sel.pick.rfind("reduce:", 0), 0u) << sel.pick << "\n" << log.all();
        }
        else if(deviceArch() == "gfx950")
        {
            // the table's tuned route, its split lowered to the workspace limit
            EXPECT_EQ(sel.source, "TUNED") << log.all();
            EXPECT_EQ(sel.pick.rfind("rreduce:tuned:", 0), 0u) << sel.pick;
        }
        else
        {
            // the untuned choice, its split lowered to the workspace limit
            EXPECT_EQ(sel.pick.rfind("rreduce:", 0), 0u) << sel.pick << "\n" << log.all();
        }
        if(c.ws == TableWorkspace::Zero)
        {
            EXPECT_EQ(workspaceSize, 0u);
            EXPECT_EQ(sel.required, 0u) << sel.pick;
            EXPECT_EQ(sel.pick.substr(sel.pick.size() - 7), ":split1") << sel.pick;
        }
        EXPECT_LE(sel.required, workspaceSize) << sel.pick;
        bool const           pm1 = c.op == HIPTENSOR_OP_MUL;
        std::vector<uint8_t> a(size_t(nA) * eb);
        for(int64_t i = 0; i < nA; i++)
        {
            putRaw(a.data() + size_t(i) * eb, c.type, tableValue(uint64_t(i), pm1));
        }
        void *pA = nullptr, *pC = nullptr, *pD = nullptr, *ws = nullptr;
        ASSERT_EQ(hipMalloc(&pA, a.size()), hipSuccess);
        ASSERT_EQ(hipMalloc(&pC, size_t(nD) * eb), hipSuccess);
        ASSERT_EQ(hipMalloc(&pD, size_t(nD) * eb), hipSuccess);
        // C all NaN: beta 0 never reads it
        ASSERT_EQ(hipMemset(pC, 0xFF, size_t(nD) * eb), hipSuccess);
        ASSERT_EQ(hipMemset(pD, 0xFF, size_t(nD) * eb), hipSuccess);
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            ASSERT_EQ(hipMemset(ws, 0xFF, workspaceSize), hipSuccess);
        }
        ASSERT_EQ(hipMemcpy(pA, a.data(), a.size(), hipMemcpyHostToDevice), hipSuccess);
        Stream stream;
        Scalar alpha, beta;
        alpha.re              = 1.0;
        beta.re               = 0.0;
        auto const scalarType = c.type == HIPTENSOR_R_64F ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
        auto const status     = hiptensorReduce(api.handle,
                                            plan,
                                            alpha.as(scalarType),
                                            pA,
                                            beta.as(scalarType),
                                            pC,
                                            pD,
                                            ws,
                                            workspaceSize,
                                            stream.s);
        EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << log.all();
        EXPECT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        std::vector<uint8_t> d(size_t(nD) * eb);
        EXPECT_EQ(hipMemcpy(d.data(), pD, d.size(), hipMemcpyDeviceToHost), hipSuccess);
        for(void* p : {pA, pC, pD, ws})
        {
            static_cast<void>(hipFree(p));
        }
        // reference: one pass over A's storage, the D offset kept with an odometer
        std::vector<double>  r(size_t(nD), reduceIdentity(c.op));
        std::vector<int64_t> idx(c.modesA.size(), 0);
        int64_t              off = 0;
        for(int64_t i = 0; i < nA; i++)
        {
            double const x = getRaw(a.data() + size_t(i) * eb, c.type);
            r[size_t(off)] = reduceRef(c.op, r[size_t(off)], x);
            for(size_t m = 0; m < idx.size(); m++)
            {
                if(++idx[m] < c.extents[m])
                {
                    off += dOfA[m];
                    break;
                }
                off -= dOfA[m] * (c.extents[m] - 1);
                idx[m] = 0;
            }
        }
        int64_t bad = 0;
        for(int64_t o = 0; o < nD; o++)
        {
            double const want = roundTo(r[size_t(o)], c.type);
            double const got  = getRaw(d.data() + size_t(o) * eb, c.type);
            if(got != want && bad++ < 5)
            {
                ADD_FAILURE() << "output " << o << " got " << got << " want " << want;
            }
        }
        EXPECT_EQ(bad, 0) << sel.pick;
    }

    std::vector<TableReductionCase> tableReductionCases()
    {
        // the reduction doc rows (hipTensor's benchmark descriptors: packed column-major)
        struct Shape
        {
            std::string          a, d;
            std::vector<int64_t> ext;
        };
        Shape const                     r1{"a", "", {16777216}};
        Shape const                     r2{"ab", "a", {16384, 16384}};
        Shape const                     r3{"abc", "ac", {512, 1024, 1024}};
        Shape const                     r4{"abcd", "abd", {128, 128, 128, 128}};
        Shape const                     r5{"abcde", "ace", {64, 64, 64, 32, 32}};
        Shape const                     r6{"abcdef", "ace", {32, 32, 32, 32, 16, 16}};
        std::vector<TableReductionCase> out;
        auto add = [&](hiptensorDataType_t t, hiptensorOperator_t op, Shape const& s) {
            out.push_back({t, op, s.a, s.d, s.ext});
        };
        for(auto t : {HIPTENSOR_R_16F, HIPTENSOR_R_16BF, HIPTENSOR_R_32F, HIPTENSOR_R_64F})
        {
            for(auto op : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL, HIPTENSOR_OP_MAX, HIPTENSOR_OP_MIN})
            {
                add(t, op, r1);
            }
        }
        add(HIPTENSOR_R_16F, HIPTENSOR_OP_ADD, r2);
        add(HIPTENSOR_R_64F, HIPTENSOR_OP_MAX, r2);
        add(HIPTENSOR_R_32F, HIPTENSOR_OP_MIN, r3);
        add(HIPTENSOR_R_16BF, HIPTENSOR_OP_MUL, r4);
        add(HIPTENSOR_R_16F, HIPTENSOR_OP_MAX, r5);
        add(HIPTENSOR_R_32F, HIPTENSOR_OP_ADD, r6);
        // with a workspace limit of zero: every doc row (f32 sums), the full reduction of
        // f16 and f64 data, a MAX; with the minimum estimate and with the tuned kernels disabled:
        // a few rows
        for(auto const* s : {&r1, &r2, &r3, &r4, &r5, &r6})
        {
            out.push_back(
                {HIPTENSOR_R_32F, HIPTENSOR_OP_ADD, s->a, s->d, s->ext, TableWorkspace::Zero});
        }
        out.push_back(
            {HIPTENSOR_R_16F, HIPTENSOR_OP_ADD, r1.a, r1.d, r1.ext, TableWorkspace::Zero});
        out.push_back(
            {HIPTENSOR_R_64F, HIPTENSOR_OP_ADD, r1.a, r1.d, r1.ext, TableWorkspace::Zero});
        out.push_back(
            {HIPTENSOR_R_64F, HIPTENSOR_OP_MAX, r6.a, r6.d, r6.ext, TableWorkspace::Zero});
        for(auto const* s : {&r1, &r2, &r4})
        {
            out.push_back(
                {HIPTENSOR_R_32F, HIPTENSOR_OP_ADD, s->a, s->d, s->ext, TableWorkspace::Minimum});
        }
        out.push_back(
            {HIPTENSOR_R_16F, HIPTENSOR_OP_ADD, r1.a, r1.d, r1.ext, TableWorkspace::Minimum});
        for(auto const* s : {&r1, &r5})
        {
            out.push_back({HIPTENSOR_R_32F,
                           HIPTENSOR_OP_ADD,
                           s->a,
                           s->d,
                           s->ext,
                           TableWorkspace::DisableTuned});
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeTableReduction,
                             ::testing::ValuesIn(tableReductionCases()));

    // The operators of the tuned classes that are not exact (square root, reciprocal, sigmoid,
    // tanh, exp, log), as opA (the una classes) and as opC, on data with each operator's special
    // arguments (NaN, infinities, signed zeros, -1 outside the domain of sqrt and log, 0 at the
    // pole of the reciprocal and log): against a long double reference, within the error bounds
    // of the programmer's guide (ulp of the compute type per operator, then the sum and the one
    // rounding to D). NaN must be NaN, infinities and zeros exact.
    struct OperatorReductionCase
    {
        hiptensorDataType_t type;
        hiptensorOperator_t op; // ADD or MAX
        hiptensorOperator_t unary;
        bool                onC; // the operator on C (opA identity), else on A
        bool                row; // A's innermost mode reduced (row classes), else kept (col)
    };

    std::ostream& operator<<(std::ostream& os, OperatorReductionCase const& c)
    {
        return os << "type " << int(c.type) << " op " << int(c.op) << " unary " << int(c.unary)
                  << (c.onC ? " on C" : " on A") << (c.row ? " row" : " col");
    }

    // The programmer's guide's bound of `u` in ulp of the compute type (f32, or f64 for f64).
    double operatorUlp(hiptensorOperator_t u, bool f64)
    {
        switch(u)
        {
        case HIPTENSOR_OP_SQRT:
        case HIPTENSOR_OP_RCP:
            return 0.5;
        case HIPTENSOR_OP_EXP:
            return 1.0;
        case HIPTENSOR_OP_LOG:
            return f64 ? 1.0 : 2.25;
        case HIPTENSOR_OP_TANH:
            return f64 ? 1.0 : 2.0;
        case HIPTENSOR_OP_SIGMOID:
            return f64 ? 2.0 : 3.0;
        default:
            return 0.0;
        }
    }

    long double operatorRef(hiptensorOperator_t u, long double x)
    {
        switch(u)
        {
        case HIPTENSOR_OP_SQRT:
            return std::sqrt(x);
        case HIPTENSOR_OP_RCP:
            return 1.0L / x;
        case HIPTENSOR_OP_SIGMOID:
            // 1 / (1 + e^-x): 0 at -inf, 1 at +inf, NaN for NaN
            return 1.0L / (1.0L + std::exp(-x));
        case HIPTENSOR_OP_TANH:
            return std::tanh(x);
        case HIPTENSOR_OP_EXP:
            return std::exp(x);
        case HIPTENSOR_OP_LOG:
            return std::log(x);
        default:
            return x;
        }
    }

    // An argument of `u` inside its domain whose result keeps sums of 64 terms far from the f16
    // range (and the compute type's normal range: no subnormal results).
    double operatorArg(hiptensorOperator_t u, std::mt19937& rng)
    {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        double const                           x = unit(rng);
        switch(u)
        {
        case HIPTENSOR_OP_SQRT:
            return 100.0 * x;
        case HIPTENSOR_OP_RCP:
            return (rng() & 1 ? 1.0 : -1.0) * (0.25 + 7.75 * x);
        case HIPTENSOR_OP_SIGMOID:
            return -30.0 + 60.0 * x;
        case HIPTENSOR_OP_TANH:
            return -5.0 + 10.0 * x;
        case HIPTENSOR_OP_EXP:
            return -4.0 + 8.0 * x;
        default: // LOG
            return 1.0e-3 + 100.0 * x;
        }
    }

    class RockeOperatorReduction : public ::testing::TestWithParam<OperatorReductionCase>
    {
    };

    TEST_P(RockeOperatorReduction, WithinTheBounds)
    {
        auto const c = GetParam();
        ApiObjects api;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture   log;
        std::mt19937 rng(7000 + int(c.unary) * 16 + int(c.type) * 4 + c.onC * 2 + c.row);
        // A "ab" packed: row = reduce a (innermost) into D "b", col = keep a, reduce b
        int64_t const        nOut = c.row ? 24 : 48, nRed = c.row ? 64 : 40;
        std::vector<int64_t> ext
            = c.row ? std::vector<int64_t>{nRed, nOut} : std::vector<int64_t>{nOut, nRed};
        std::string const modesD = c.row ? "b" : "a";
        size_t const      eb     = sizeOf(c.type);
        bool const        f64    = c.type == HIPTENSOR_R_64F;
        // the special arguments, one per output (outputs past them: regular arguments only)
        double const special[] = {std::nan(""), HUGE_VAL, -HUGE_VAL, 0.0, -0.0, -1.0};
        size_t const nSpecial  = std::size(special);
        // A and C as stored (the data type's values), as long double
        std::vector<long double> a(static_cast<size_t>(nOut * nRed)), cv(static_cast<size_t>(nOut));
        std::vector<uint8_t>     rawA(a.size() * eb), rawC(cv.size() * eb);
        auto                     at = [&](int64_t o, int64_t r) {
            // A's storage offset of output o, reduced index r
            return size_t(c.row ? r + nRed * o : o + nOut * r);
        };
        auto store = [&](std::vector<uint8_t>& raw, size_t i, double v) {
            putRaw(raw.data() + i * eb, c.type, v);
            return static_cast<long double>(getRaw(raw.data() + i * eb, c.type));
        };
        std::uniform_int_distribution<int> small(-2, 2);
        for(int64_t o = 0; o < nOut; o++)
        {
            int64_t const where = int64_t(rng() % uint32_t(nRed));
            for(int64_t r = 0; r < nRed; r++)
            {
                double v = c.onC ? double(small(rng)) : operatorArg(c.unary, rng);
                if(!c.onC && size_t(o) < nSpecial && r == where)
                {
                    v = special[o];
                }
                a[at(o, r)] = store(rawA, at(o, r), v);
            }
            double v      = c.onC ? (size_t(o) < nSpecial ? special[o] : operatorArg(c.unary, rng))
                                  : double(small(rng));
            cv[size_t(o)] = store(rawC, size_t(o), v);
        }
        std::vector<int64_t> strideA = {1, ext[0]};
        std::vector<int32_t> ma      = {'a', 'b'};
        std::vector<int32_t> md(modesD.begin(), modesD.end());
        auto const           dA = api.tensor(ext, strideA, c.type);
        auto const           dD = api.tensor({nOut}, {1}, c.type);
        auto const compute      = f64 ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
        hiptensorOperationDescriptor_t desc = nullptr;
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &desc,
                                           dA,
                                           ma.data(),
                                           c.onC ? HIPTENSOR_OP_IDENTITY : c.unary,
                                           dD,
                                           md.data(),
                                           c.onC ? c.unary : HIPTENSOR_OP_IDENTITY,
                                           dD,
                                           md.data(),
                                           c.op,
                                           compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t     workspaceSize = 0;
        SelectionLog sel;
        auto         plan = planWith(api, desc, {}, log, workspaceSize, &sel);
        ASSERT_NE(plan, nullptr);
        // the tuned classes run these operators (opA: the una classes)
        EXPECT_EQ(sel.pick.rfind("rreduce:", 0), 0u) << sel.pick << "\n" << log.all();
        void *pA = nullptr, *pC = nullptr, *pD = nullptr, *ws = nullptr;
        ASSERT_EQ(hipMalloc(&pA, rawA.size()), hipSuccess);
        ASSERT_EQ(hipMalloc(&pC, rawC.size()), hipSuccess);
        ASSERT_EQ(hipMalloc(&pD, rawC.size()), hipSuccess);
        ASSERT_EQ(hipMemcpy(pA, rawA.data(), rawA.size(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(pC, rawC.data(), rawC.size(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(pD, 0xFF, rawC.size()), hipSuccess);
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            ASSERT_EQ(hipMemset(ws, 0xFF, workspaceSize), hipSuccess);
        }
        // opA: D = sum / max of opA(A) (beta 0, C NaN-free but unread); opC: D = sum / max of A +
        // opC(C)
        double const beta       = c.onC ? 1.0 : 0.0;
        auto const   scalarType = f64 ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
        Scalar       alpha, betaS;
        alpha.re = 1.0;
        betaS.re = beta;
        Stream stream;
        EXPECT_EQ(hiptensorReduce(api.handle,
                                  plan,
                                  alpha.as(scalarType),
                                  pA,
                                  betaS.as(scalarType),
                                  pC,
                                  pD,
                                  ws,
                                  workspaceSize,
                                  stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        EXPECT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        std::vector<uint8_t> rawD(rawC.size());
        EXPECT_EQ(hipMemcpy(rawD.data(), pD, rawD.size(), hipMemcpyDeviceToHost), hipSuccess);
        for(void* p : {pA, pC, pD, ws})
        {
            static_cast<void>(hipFree(p));
        }
        long double const uc     = f64 ? std::ldexp(1.0L, -53) : std::ldexp(1.0L, -24);
        long double const ud     = unitRoundoff(c.type);
        long double const floorD = c.type == HIPTENSOR_R_16F    ? std::ldexp(1.0L, -24)
                                   : c.type == HIPTENSOR_R_16BF ? std::ldexp(1.0L, -133)
                                   : c.type == HIPTENSOR_R_32F  ? std::ldexp(1.0L, -149)
                                                                : std::ldexp(1.0L, -1074);
        long double const ulpOp  = operatorUlp(c.unary, f64);
        bool const        isMax  = c.op == HIPTENSOR_OP_MAX;
        int64_t           bad = 0, specials = 0;
        for(int64_t o = 0; o < nOut; o++)
        {
            // terms of the reduction (opA applied), their size, the C term
            long double r = isMax ? -HUGE_VALL : 0.0L, size = 0.0L;
            bool        nan = false;
            for(int64_t rr = 0; rr < nRed; rr++)
            {
                long double const t = c.onC ? a[at(o, rr)] : operatorRef(c.unary, a[at(o, rr)]);
                nan                 = nan || std::isnan(t);
                r                   = isMax ? std::max(r, t) : r + t;
                size                = isMax ? std::max(size, std::fabs(t)) : size + std::fabs(t);
            }
            if(nan)
            {
                r = std::numeric_limits<long double>::quiet_NaN();
            }
            long double const ct   = c.onC ? operatorRef(c.unary, cv[size_t(o)]) : 0.0L;
            long double const want = r + beta * ct;
            // the operator's error on each term (or the C term), the sum's and D's rounding
            long double const opErr  = 2.0L * ulpOp * uc * (c.onC ? std::fabs(ct) : size);
            long double const sumErr = (isMax ? 2.0L : static_cast<long double>(nRed + 2)) * uc
                                       * (size + std::fabs(beta * ct));
            long double const tol = opErr + sumErr + ud * std::fabs(want) + floorD;
            double const      got = getRaw(rawD.data() + size_t(o) * eb, c.type);
            bool              ok;
            if(std::isnan(want))
            {
                ok = std::isnan(got);
                specials++;
            }
            else if(std::isinf(want))
            {
                ok = static_cast<long double>(got) == want;
                specials++;
            }
            else
            {
                ok = !std::isnan(got) && std::fabs(static_cast<long double>(got) - want) <= tol;
            }
            if(!ok && bad++ < 5)
            {
                ADD_FAILURE() << "output " << o << " got " << got << " want "
                              << static_cast<double>(want) << " tol " << static_cast<double>(tol)
                              << " (" << sel.pick << ")";
            }
        }
        EXPECT_EQ(bad, 0) << sel.pick;
        // the special arguments reached the outputs (a NaN argument at least gives NaN)
        EXPECT_GE(specials, 1) << sel.pick;
    }

    std::vector<OperatorReductionCase> operatorReductionCases()
    {
        std::vector<OperatorReductionCase> out;
        for(auto type : {HIPTENSOR_R_16F, HIPTENSOR_R_16BF, HIPTENSOR_R_32F, HIPTENSOR_R_64F})
        {
            for(auto unary : {HIPTENSOR_OP_SQRT,
                              HIPTENSOR_OP_RCP,
                              HIPTENSOR_OP_SIGMOID,
                              HIPTENSOR_OP_TANH,
                              HIPTENSOR_OP_EXP,
                              HIPTENSOR_OP_LOG})
            {
                for(bool onC : {false, true})
                {
                    for(auto op : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MAX})
                    {
                        for(bool row : {true, false})
                        {
                            out.push_back({type, op, unary, onC, row});
                        }
                    }
                }
            }
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeOperatorReduction,
                             ::testing::ValuesIn(operatorReductionCases()));

    // ---------------------------------------------------------------------- selection

    // Every candidate the selector ranks for a problem (KERNEL_RANK 0, 1, ...) is a different
    // kernel program and computes the exact result (integer data).
    class RockeEveryContractionCandidate : public ::testing::TestWithParam<ContractionCase>
    {
    };

    TEST_P(RockeEveryContractionCandidate, Exact)
    {
        auto const c = GetParam();
        // with the largest workspace estimate every candidate is admitted
        Choice all;
        all.workspace = HIPTENSOR_WORKSPACE_MAX;
        SelectionLog first;
        runContraction(c, all, &first);
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        ASSERT_GE(first.candidates, 1);
        std::set<std::string> picks{first.pick};
        bool                  gemm   = first.pick.rfind("gemm:", 0) == 0;
        bool                  gett32 = first.pick.rfind("gett32:", 0) == 0;
        bool                  f64    = first.pick.rfind("f64:", 0) == 0;
        for(int32_t r = 1; r < first.candidates; r++)
        {
            Choice choice = all;
            choice.rank   = r;
            SelectionLog s;
            runContraction(c, choice, &s);
            ASSERT_FALSE(HasFailure()) << "KERNEL_RANK " << r << ": " << s.pick;
            EXPECT_EQ(s.source, "KERNEL_RANK");
            EXPECT_EQ(s.candidates, first.candidates);
            picks.insert(s.pick);
            gemm   = gemm || s.pick.rfind("gemm:", 0) == 0;
            gett32 = gett32 || s.pick.rfind("gett32:", 0) == 0;
            f64    = f64 || s.pick.rfind("f64:", 0) == 0;
        }
        EXPECT_EQ(picks.size(), size_t(first.candidates));
        bool const native16 = c.type == HIPTENSOR_R_32F && c.compute != HIPTENSOR_COMPUTE_DESC_32F;
        bool const real64   = c.type == HIPTENSOR_R_64F || c.type == HIPTENSOR_C_64F;
        // f32 / complex f32 data (and 16-bit data with an operand operator) with the 32F / C32F
        // compute type: the native f32 route (the default HIPTENSOR_ROCKE_F32_RECIPE) where the
        // pack has it, and the f64 route
        bool const f32Recipe = (c.type == HIPTENSOR_R_32F || c.type == HIPTENSOR_C_32F
                                || c.opA != HIPTENSOR_OP_IDENTITY)
                               && !native16 && !real64;
        bool const nativeF32 = f32Recipe && std::getenv("HIPTENSOR_ROCKE_F32_RECIPE") == nullptr;
        if(nativeF32)
        {
            EXPECT_EQ(gett32, deviceArch() == "gfx950" || deviceArch() == "gfx942");
            EXPECT_FALSE(gemm);
            EXPECT_TRUE(f64);
        }
        else if(deviceArch() == "gfx950" && !real64)
        {
            // the GEMM route of the type and (except for 16-bit compute) the f64 route
            EXPECT_TRUE(gemm);
            EXPECT_EQ(f64, !native16);
        }
        if(real64 || deviceArch() != "gfx950")
        {
            // f64 strided GEMM classes
            EXPECT_GE(first.candidates, native16 ? 1 : 2);
        }
    }

    std::vector<ContractionCase> everyCandidateCases()
    {
        struct Row
        {
            hiptensorDataType_t          type;
            hiptensorComputeDescriptor_t compute;
            hiptensorOperator_t          opA;
        };
        Row const rows[] = {{HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, HIPTENSOR_OP_IDENTITY},
                            {HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_32F, HIPTENSOR_OP_IDENTITY},
                            {HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, HIPTENSOR_OP_RELU},
                            {HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, HIPTENSOR_OP_IDENTITY},
                            {HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_16F, HIPTENSOR_OP_IDENTITY},
                            {HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_16BF, HIPTENSOR_OP_IDENTITY},
                            {HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, HIPTENSOR_OP_IDENTITY},
                            {HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F, HIPTENSOR_OP_IDENTITY},
                            {HIPTENSOR_C_64F, HIPTENSOR_COMPUTE_DESC_C64F, HIPTENSOR_OP_IDENTITY}};
        std::vector<ContractionCase> out;
        int                          seed = 1300;
        for(auto const& r : rows)
        {
            for(bool scale : {false, true})
            {
                out.push_back({r.type, r.compute, seed++, true, !scale, scale, r.opA});
            }
        }
        // outer products (no contracted mode): every route
        seed = 1350;
        for(auto const& r : rows)
        {
            if(r.opA == HIPTENSOR_OP_IDENTITY)
            {
                ContractionCase c{r.type, r.compute, seed++, true, seed % 2 == 0, seed % 2 != 0};
                c.outer = true;
                out.push_back(c);
            }
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeEveryContractionCandidate,
                             ::testing::ValuesIn(everyCandidateCases()));

    // Every reduction candidate (row and col kernels, split degrees) computes the exact result.
    class RockeEveryReductionCandidate : public ::testing::TestWithParam<ReductionCase>
    {
    };

    TEST_P(RockeEveryReductionCandidate, Exact)
    {
        auto const c = GetParam();
        Choice     all;
        all.workspace = HIPTENSOR_WORKSPACE_MAX;
        SelectionLog first;
        runReduction(c, all, &first);
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        ASSERT_GE(first.candidates, 1);
        std::set<std::string> picks{first.pick};
        for(int32_t r = 1; r < first.candidates; r++)
        {
            Choice choice = all;
            choice.rank   = r;
            SelectionLog s;
            runReduction(c, choice, &s);
            ASSERT_FALSE(HasFailure()) << "KERNEL_RANK " << r << ": " << s.pick;
            picks.insert(s.pick);
        }
        EXPECT_EQ(picks.size(), size_t(first.candidates));
    }

    std::vector<ReductionCase> everyReductionCases()
    {
        std::vector<ReductionCase> out;
        int                        seed = 1400;
        for(auto type : {HIPTENSOR_R_16F, HIPTENSOR_R_32F, HIPTENSOR_R_64F})
        {
            auto compute
                = type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
            for(auto op : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MAX})
            {
                for(int i = 0; i < 2; i++)
                {
                    out.push_back({type, compute, op, seed++});
                }
            }
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeEveryReductionCandidate,
                             ::testing::ValuesIn(everyReductionCases()));

    // The default selection is deterministic; on gfx950 the pack's selection model decides,
    // HIPTENSOR_ROCKE_SELECT=static selects by rocKE's rules.
    TEST(RockeSelection, ModelStaticAndDeterministic)
    {
        ContractionCase const c{
            HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, 1501, true, true, false};
        // with room for every candidate both selections see the same candidates
        Choice all;
        all.workspace = HIPTENSOR_WORKSPACE_MAX;
        SelectionLog a, b;
        runContraction(c, all, &a);
        runContraction(c, all, &b);
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        EXPECT_EQ(a.pick, b.pick);
        EXPECT_TRUE(a.source == "MODEL" || a.source == "STATIC" || a.source == "RULE") << a.source;
        // a rule names itself and picks its route: the ring GETT when the layout admits it, the
        // direct GETT on targets without the GEMM routes, when the cell's model does not score
        // their routes; otherwise the model decides (gfx950's model of this class scores every
        // route: the ring GETT and the direct GETT too)
        bool const rule = a.source == "RULE";
        if(rule)
        {
            bool const ring = a.reason == "the ring GETT rule";
            EXPECT_TRUE(ring || a.reason == "the direct GETT rule") << a.reason;
            EXPECT_EQ(a.pick.rfind(ring ? "rgett:" : "dgett:", 0), 0u)
                << a.pick << " (" << a.reason << ")";
        }
        if(deviceArch() == "gfx950")
        {
            EXPECT_EQ(a.source, "MODEL") << a.reason;
            EXPECT_EQ(a.reason.rfind("contraction.h@", 0), 0u) << a.reason;
        }
        SelectionLog st;
        {
            ScopedEnv env("HIPTENSOR_ROCKE_SELECT", "static");
            runContraction(c, all, &st);
        }
        // without a model the rules order every route (the ring GETT and the direct GETT go
        // first as above); a rule of the model selection applies to the static one too
        EXPECT_TRUE(st.source == "RULE" || st.source == "STATIC") << st.source;
        if(rule)
        {
            EXPECT_EQ(st.source, "RULE");
            EXPECT_EQ(st.pick, a.pick);
        }
        EXPECT_EQ(st.candidates, a.candidates);

        // reductions: the tuned reduction rule picks a tuned reduction class when one serves the
        // problem (the rule applies to the static selection too); otherwise their models decide
        ReductionCase const r{HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, HIPTENSOR_OP_ADD, 1502};
        SelectionLog        red, redStatic;
        runReduction(r, {}, &red);
        {
            ScopedEnv env("HIPTENSOR_ROCKE_SELECT", "static");
            runReduction(r, {}, &redStatic);
        }
        EXPECT_TRUE(red.source == "MODEL" || red.source == "STATIC" || red.source == "RULE")
            << red.source;
        if(red.source == "RULE")
        {
            EXPECT_EQ(red.reason, "the tuned reduction rule");
            EXPECT_EQ(red.pick.rfind("rreduce:", 0), 0u) << red.pick;
            EXPECT_EQ(redStatic.source, "RULE");
            EXPECT_EQ(redStatic.pick, red.pick);
        }
        else
        {
            EXPECT_EQ(redStatic.source, "STATIC");
            if(deviceArch() == "gfx950" && red.candidates > 1)
            {
                EXPECT_EQ(red.source, "MODEL");
            }
        }
    }

    // A selection model file <dir>/<target>/<cell>.model for HIPTENSOR_ROCKE_SELECT_MODELS (a new
    // directory per name: the library reads a model once per process and source).
    std::string
        writeSelectModel(std::string const& name, std::string const& cell, std::string const& text)
    {
        std::string const dir = ::testing::TempDir() + "rocke_select_models_" + name;
        std::string const sub = dir + "/" + deviceArch();
        EXPECT_EQ(std::system(("mkdir -p \"" + sub + "\"").c_str()), 0);
        FILE* f = std::fopen((sub + "/" + cell + ".model").c_str(), "w");
        EXPECT_NE(f, nullptr);
        if(f != nullptr)
        {
            std::fputs((text + "\n").c_str(), f);
            std::fclose(f);
        }
        return dir;
    }

    // One tree on the feature "route": 0 for `route`, 1 for every other route.
    std::string favourRoute(int route)
    {
        char tree[96];
        std::snprintf(tree, sizeof(tree), "3;0 0;%d.5 %d.5;-1 -2;1 -3;1 0 1", route - 1, route);
        return tree;
    }

    // A model scores the routes it was trained on: a format v2 model names them (here the ring
    // GETT (7) and the direct GETT (8) among them), a format v1 model scores the GEMM and f64
    // routes only and the rules order the others; a model the library cannot use (malformed
    // routes, a later feature schema, an unknown feature) leaves the rules' selection.
    // HIPTENSOR_ROCKE_SELECT_MODELS replaces the pack's models.
    TEST(RockeSelection, ModelScoresItsRoutes)
    {
        // on gfx950 a problem the ring GETT serves (its rule is then the rules' selection)
        ContractionCase c{HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, 1503, true, true, false};
        Choice          all;
        all.workspace = HIPTENSOR_WORKSPACE_MAX;
        SelectionLog rules;
        for(int seed = 1503; seed < 1563; seed++)
        {
            c.seed = seed;
            ScopedEnv env("HIPTENSOR_ROCKE_SELECT", "static");
            runContraction(c, all, &rules);
            if(HasFatalFailure() || IsSkipped() || deviceArch() != "gfx950"
               || rules.reason == "the ring GETT rule")
            {
                break;
            }
        }
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        if(deviceArch() == "gfx950")
        {
            ASSERT_EQ(rules.reason, "the ring GETT rule") << "no problem the ring GETT serves";
        }
        ASSERT_GE(rules.candidates, 2);
        auto const cell = std::string("contraction.h");
        auto       with = [&](std::string const& name, std::string const& text) {
            SelectionLog out;
            ScopedEnv    env("HIPTENSOR_ROCKE_SELECT_MODELS",
                          writeSelectModel(name, cell, text).c_str());
            runContraction(c, all, &out);
            EXPECT_EQ(out.candidates, rules.candidates) << name;
            return out;
        };
        auto const tree = favourRoute(8);
        // every route scored: the direct GETT (every contraction has it) wins
        auto v2 = with("v2_all",
                       "v2;schema=2;routes=3,6,7,8,9;margin=0;features=route;trees=1|" + tree);
        EXPECT_EQ(v2.source, "MODEL") << v2.reason;
        EXPECT_EQ(v2.reason.rfind(cell + "@", 0), 0u) << v2.reason;
        EXPECT_EQ(v2.pick.rfind("dgett:", 0), 0u) << v2.pick;
        // schema 1 features with the v2 format: the same
        auto s1 = with("v2_schema1",
                       "v2;schema=1;routes=3,6,7,8,9;margin=0;features=route;trees=1|" + tree);
        EXPECT_EQ(s1.source, "MODEL") << s1.reason;
        EXPECT_EQ(s1.pick, v2.pick);
        // a margin: the model's best (the direct GETT, predicted 1 better than every other route)
        // replaces the first candidate of the static order (never the direct GETT, the last
        // route of the static order) only when its prediction is lower by more than the margin
        auto keep = with("v2_margin_keep",
                         "v2;schema=2;routes=3,6,7,8,9;margin=1.5;features=route;trees=1|" + tree);
        EXPECT_EQ(keep.source, "MODEL") << keep.reason;
        EXPECT_NE(keep.pick.rfind("dgett:", 0), 0u) << keep.pick;
        // a gap equal to the margin keeps the static order's candidate too
        auto equal = with("v2_margin_equal",
                          "v2;schema=2;routes=3,6,7,8,9;margin=1;features=route;trees=1|" + tree);
        EXPECT_EQ(equal.source, "MODEL") << equal.reason;
        EXPECT_EQ(equal.pick, keep.pick);
        auto take = with("v2_margin_take",
                         "v2;schema=2;routes=3,6,7,8,9;margin=0.5;features=route;trees=1|" + tree);
        EXPECT_EQ(take.source, "MODEL") << take.reason;
        EXPECT_EQ(take.pick.rfind("dgett:", 0), 0u) << take.pick;
        // the ring GETT not scored: its rule applies when it is a candidate
        bool const ring   = rules.pick.rfind("rgett:", 0) == 0;
        auto       noRing = with("v2_no_ring",
                           "v2;schema=2;routes=3,6,8,9;margin=0;features=route;trees=1|" + tree);
        if(ring)
        {
            EXPECT_EQ(noRing.source, "RULE") << noRing.reason;
            EXPECT_EQ(noRing.reason, "the ring GETT rule");
            EXPECT_EQ(noRing.pick, rules.pick);
            // a model that scores the ring GETT picks it when it predicts it fastest
            auto r = with("v2_ring",
                          "v2;schema=2;routes=3,6,7,8,9;margin=0;features=route;trees=1|"
                              + favourRoute(7));
            EXPECT_EQ(r.source, "MODEL") << r.reason;
            EXPECT_EQ(r.pick.rfind("rgett:", 0), 0u) << r.pick;
        }
        else
        {
            EXPECT_EQ(noRing.source, "MODEL") << noRing.reason;
            EXPECT_EQ(noRing.pick, v2.pick);
        }
        // format v1: the direct GETT and the ring GETT are not scored
        auto v1 = with("v1", "v1;schema=1;features=route;trees=1|" + tree);
        if(rules.source == "RULE")
        {
            EXPECT_EQ(v1.source, "RULE") << v1.reason;
            EXPECT_EQ(v1.pick, rules.pick);
        }
        else
        {
            EXPECT_EQ(v1.source, "MODEL") << v1.reason;
            EXPECT_NE(v1.pick.rfind("dgett:", 0), 0u) << v1.pick;
        }
        // models the library does not use: the rules' selection
        for(auto const& [name, text] :
            {std::pair<std::string, std::string>{
                 "bad_routes", "v2;schema=2;routes=3,x;margin=0;features=route;trees=1|" + tree},
             {"bad_route_id", "v2;schema=2;routes=3,99;margin=0;features=route;trees=1|" + tree},
             {"later_schema",
              "v2;schema=4;routes=3,6,7,8,9;margin=0;features=route;trees=1|" + tree},
             {"v3_bad_gemm",
              "v3;schema=3;routes=3,6,7,8,9;margin=0;gemm=abc;features=route;trees=1|" + tree},
             {"v3_no_gemm_field",
              "v3;schema=3;routes=3,6,7,8,9;margin=0;features=route;trees=1|" + tree},
             {"unknown_feature",
              "v2;schema=2;routes=3,6,7,8,9;margin=0;features=nosuch;trees=1|" + tree},
             {"no_routes_field", "v2;schema=2;features=route;trees=1|" + tree},
             {"no_margin_field", "v2;schema=2;routes=3,6,7,8,9;features=route;trees=1|" + tree},
             {"bad_margin", "v2;schema=2;routes=3,6,7,8,9;margin=x;features=route;trees=1|" + tree},
             {"negative_margin",
              "v2;schema=2;routes=3,6,7,8,9;margin=-1;features=route;trees=1|" + tree},
             {"empty_margin",
              "v2;schema=2;routes=3,6,7,8,9;margin=;features=route;trees=1|" + tree},
             {"bad_tree",
              "v2;schema=2;routes=3,6,7,8,9;margin=0;features=route;trees=1|3;0 0;7.5"}})
        {
            auto bad = with(name, text);
            EXPECT_EQ(bad.pick, rules.pick) << name;
            EXPECT_EQ(bad.source, rules.source) << name;
            if(rules.source == "STATIC")
            {
                EXPECT_NE(bad.reason.find("not used"), std::string::npos)
                    << name << ": " << bad.reason;
            }
        }
        // no file for the cell in the directory: no model
        {
            SelectionLog none;
            ScopedEnv    env("HIPTENSOR_ROCKE_SELECT_MODELS",
                          (::testing::TempDir() + "rocke_select_models_empty").c_str());
            runContraction(c, all, &none);
            EXPECT_EQ(none.pick, rules.pick);
            EXPECT_EQ(none.source, rules.source);
        }
        // format v3 (the planner's GEMM selection models named; a model that does not read the
        // planner's predicted time names none): as v2
        auto v3 = with("v3", "v3;schema=3;routes=3,6,7,8,9;margin=0;gemm=-;features=route;trees=1|"
                                 + tree);
        EXPECT_EQ(v3.source, "MODEL") << v3.reason;
        EXPECT_EQ(v3.pick, v2.pick);
    }

    // The workspace is known at plan creation: MIN <= DEFAULT <= MAX, the plan's required
    // workspace is the estimate it was created with (DEFAULT) or at most it (MIN, MAX), and the
    // plans of every estimate compute the exact result.
    TEST(RockeSelection, WorkspaceAtPlanTime)
    {
        for(auto type : {HIPTENSOR_R_32F, HIPTENSOR_R_16BF, HIPTENSOR_R_64F})
        {
            ContractionCase const c{type,
                                    type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F
                                                            : HIPTENSOR_COMPUTE_DESC_32F,
                                    1601,
                                    true,
                                    true,
                                    false};
            SelectionLog          mn, df, mx;
            Choice                choice;
            choice.workspace = HIPTENSOR_WORKSPACE_MIN;
            runContraction(c, choice, &mn);
            choice.workspace = HIPTENSOR_WORKSPACE_DEFAULT;
            runContraction(c, choice, &df);
            choice.workspace = HIPTENSOR_WORKSPACE_MAX;
            runContraction(c, choice, &mx);
            if(HasFailure() || IsSkipped())
            {
                return;
            }
            EXPECT_LE(mn.estimate, df.estimate);
            EXPECT_LE(df.estimate, mx.estimate);
            EXPECT_EQ(df.required, df.estimate);
            EXPECT_LE(mn.required, mn.estimate);
            EXPECT_LE(mx.required, mx.estimate);
        }
    }

    // HIPTENSOR_ALGO_DEFAULT_PATIENT times the candidates and stores the winner in the plan
    // cache: a later plan of the same problem on the handle (and on a handle that read the plan
    // cache file) takes it from the cache. The timings are written to the dump file.
    TEST(RockeSelection, AutotunePlanCache)
    {
        ContractionCase const c{
            HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, 1701, true, true, false};
        std::string const dump = ::testing::TempDir() + "rocke_select_dump.jsonl";
        std::string const file = ::testing::TempDir() + "rocke_select_plan_cache.bin";
        std::remove(dump.c_str());
        std::remove(file.c_str());
        ApiObjects   api;
        SelectionLog tuned, cached, reread, uncached;
        {
            ScopedEnv topk("HIPTENSOR_ROCKE_AUTOTUNE_TOPK", "all");
            ScopedEnv out("HIPTENSOR_ROCKE_SELECT_DUMP", dump.c_str());
            ScopedEnv tag("HIPTENSOR_ROCKE_SELECT_DUMP_TAG", "unit");
            Choice    choice;
            choice.algo = HIPTENSOR_ALGO_DEFAULT_PATIENT;
            choice.api  = &api;
            runContraction(c, choice, &tuned);
        }
        if(HasFailure() || IsSkipped())
        {
            return;
        }
        if(tuned.candidates < 2)
        {
            GTEST_SKIP() << "one candidate: nothing to time";
        }
        EXPECT_EQ(tuned.source, "AUTOTUNE");
        int   lines = 0;
        FILE* f     = std::fopen(dump.c_str(), "r");
        ASSERT_NE(f, nullptr);
        char buf[8192];
        while(std::fgets(buf, sizeof(buf), f) != nullptr)
        {
            std::string line(buf);
            if(line.find("\"us\":") != std::string::npos
               && line.find("\"tag\":\"unit\"") != std::string::npos)
            {
                lines++;
            }
        }
        std::fclose(f);
        EXPECT_EQ(lines, tuned.candidates);

        Choice choice;
        choice.api = &api;
        runContraction(c, choice, &cached);
        EXPECT_EQ(cached.source, "PLAN_CACHE");
        EXPECT_EQ(cached.pick, tuned.pick);

        ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(api.handle, file.c_str()),
                  HIPTENSOR_STATUS_SUCCESS);
        ApiObjects other;
        uint32_t   read = 0;
        ASSERT_EQ(hiptensorHandleReadPlanCacheFromFile(other.handle, file.c_str(), &read),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_GE(read, 1u);
        choice.api = &other;
        runContraction(c, choice, &reread);
        EXPECT_EQ(reread.source, "PLAN_CACHE");
        EXPECT_EQ(reread.pick, tuned.pick);

        choice.cacheMode = HIPTENSOR_CACHE_MODE_NONE;
        runContraction(c, choice, &uncached);
        EXPECT_NE(uncached.source, "PLAN_CACHE");
        std::remove(dump.c_str());
        std::remove(file.c_str());
    }

    // KERNEL_RANK past the candidates is NOT_SUPPORTED (no fallback in this test program).
    TEST(RockeSelection, KernelRankPastTheCandidates)
    {
        ApiObjects                     api;
        LogCapture                     log;
        int64_t const                  lens[] = {33, 7};
        int32_t const                  ma[] = {'a', 'b'}, mb[] = {'b', 'a'};
        auto                           dA   = api.tensor({lens[0], lens[1]}, {}, HIPTENSOR_R_32F);
        auto                           dB   = api.tensor({lens[1], lens[0]}, {}, HIPTENSOR_R_32F);
        hiptensorOperationDescriptor_t desc = nullptr;
        ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                             &desc,
                                             dA,
                                             ma,
                                             HIPTENSOR_OP_IDENTITY,
                                             dB,
                                             mb,
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        hiptensorPlanPreference_t pref = nullptr;
        ASSERT_EQ(hiptensorCreatePlanPreference(
                      api.handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);
        int32_t const rank = 1;
        ASSERT_EQ(hiptensorPlanPreferenceSetAttribute(
                      api.handle, pref, HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK, &rank, sizeof(rank)),
                  HIPTENSOR_STATUS_SUCCESS);
        hiptensorPlan_t plan   = nullptr;
        auto            status = hiptensorCreatePlan(api.handle, &plan, desc, pref, 0);
        static_cast<void>(hiptensorDestroyPlanPreference(pref));
        if(plan != nullptr)
        {
            api.plans.push_back(plan);
        }
        EXPECT_EQ(status, HIPTENSOR_STATUS_NOT_SUPPORTED) << log.all();
        EXPECT_TRUE(log.contains("KERNEL_RANK 1 is past the 1 rocKE candidates")) << log.all();
    }

    // ---------------------------------------------------------------------- one rounding

    // `v` rounded once to nearest even to data type `t` (f16, bf16, f32; complex f32 per
    // component), with subnormals and overflow to infinity: an independent reference of the
    // conversions to a compute type (roundTo above rounds to bf16 through f32).
    double roundOnce(double v, hiptensorDataType_t t)
    {
        int p = 0, emin = 0, emax = 0;
        switch(t)
        {
        case HIPTENSOR_R_16F:
            p = 11, emin = -14, emax = 15;
            break;
        case HIPTENSOR_R_16BF:
            p = 8, emin = -126, emax = 127;
            break;
        case HIPTENSOR_R_32F:
        case HIPTENSOR_C_32F:
            p = 24, emin = -126, emax = 127;
            break;
        default:
            return v;
        }
        if(!std::isfinite(v) || v == 0.0)
        {
            return v;
        }
        int e = 0;
        std::frexp(std::fabs(v), &e);
        int const    q   = std::max(e - 1, emin) - (p - 1); // exponent of the last bit
        double const r   = std::ldexp(std::nearbyint(std::ldexp(std::fabs(v), -q)), q);
        double const top = std::ldexp(2.0 - std::ldexp(1.0, 1 - p), emax);
        return std::copysign(r > top ? std::numeric_limits<double>::infinity() : r, v);
    }

    cplx roundOnce(cplx v, hiptensorDataType_t t)
    {
        return cplx(roundOnce(v.real(), t), roundOnce(v.imag(), t));
    }

    // got against want per component: NaN matches NaN, an infinity itself, a finite value the
    // same value (exact) or one within `bound`.
    bool matches(cplx got, cplx want, double bound, bool exact)
    {
        auto one = [&](double g, double w) {
            if(std::isnan(w))
            {
                return std::isnan(g);
            }
            if(std::isinf(w))
            {
                return g == w;
            }
            return exact ? g == w : std::fabs(g - w) <= bound;
        };
        return one(got.real(), want.real()) && one(got.imag(), want.imag());
    }

    // The precision a compute type gives an operand: rounded once to f16 / bf16 for the 16-bit
    // compute types; exact otherwise (f16, bf16 and f32 values are f32 values, and f64 operands
    // take the f64 route).
    hiptensorDataType_t operandPrecision(hiptensorComputeDescriptor_t compute)
    {
        return compute == HIPTENSOR_COMPUTE_DESC_16F    ? HIPTENSOR_R_16F
               : compute == HIPTENSOR_COMPUTE_DESC_16BF ? HIPTENSOR_R_16BF
                                                        : HIPTENSOR_R_64F;
    }

    // The unit roundoff of the products' accumulation.
    double accumulationRoundoff(hiptensorComputeDescriptor_t compute)
    {
        return compute == HIPTENSOR_COMPUTE_DESC_64F || compute == HIPTENSOR_COMPUTE_DESC_C64F
                   ? std::ldexp(1.0, -53)
                   : std::ldexp(1.0, -24);
    }

    cplx applyOp(hiptensorOperator_t op, cplx x)
    {
        switch(op)
        {
        case HIPTENSOR_OP_CONJ:
            return std::conj(x);
        case HIPTENSOR_OP_RELU:
            return cplx(std::max(x.real(), 0.0), 0.0);
        case HIPTENSOR_OP_NEG:
            return -x;
        default:
            return x;
        }
    }

    // ---------------------------------------------------------------------- f32 recipes

    // HIPTENSOR_ROCKE_F32_RECIPE: the routes each recipe admits (the candidate of every
    // KERNEL_RANK) and the accuracy class of its results against a loop reference in double, for
    // f32, complex f32 and mixed real data with the 32F / C32F compute type.
    struct RecipeCase
    {
        char const*         recipe; // nullptr: the variable unset (native)
        hiptensorDataType_t a, b, d;
        bool                scale; // no C
        int                 seed;
    };

    std::ostream& operator<<(std::ostream& os, RecipeCase const& c)
    {
        return os << (c.recipe != nullptr ? c.recipe : "unset") << "_" << int(c.a) << "_"
                  << int(c.b) << "_" << int(c.d) << (c.scale ? "_scale" : "_bilinear");
    }

    // The accuracy class of a recipe (programmers-guide.rst): products per element product,
    // the relative error of a product's representation, the accumulation's unit roundoff.
    struct RecipeClass
    {
        int    products;
        double rep;
        double uc;
    };

    RecipeClass recipeClass(std::string const& recipe)
    {
        double const u = std::ldexp(1.0, -24);
        if(recipe == "f64")
        {
            return {1, 0.0, std::ldexp(1.0, -53)};
        }
        if(recipe == "x9")
        {
            // three bf16 parts, every part product (each exact, summed in f32)
            return {9, 0.0, u};
        }
        if(recipe == "x6")
        {
            // three bf16 parts (|a - hi - mid - lo| <= 2^-24 |a|), the three smallest products
            // (at most 2^-23 |a b|) dropped
            return {6, std::ldexp(1.0, -21), u};
        }
        if(recipe == "x3" || recipe == "x4")
        {
            // two bf16 parts: |a - hi - mid| <= 2^-16 |a| per operand; x3 also drops mid * mid
            return {recipe == "x3" ? 3 : 4, std::ldexp(1.0, -14), u};
        }
        return {1, 0.0, u}; // native: f32 products
    }

    // The direct GETT's compute class in a candidate key (dgett:hiptensor_aot_dgett_<r|c>_<comp>_...).
    bool dgettComputesF64(std::string const& key)
    {
        return key.find("_dgett_r_f64_") != std::string::npos
               || key.find("_dgett_c_f64_") != std::string::npos;
    }

    // The candidate keys a recipe admits for the case's data (the f64 route always; the direct
    // GETT, f32 products and accumulation, under every recipe of the f32 class or below, f64
    // under the f64 recipe).
    bool recipeAdmits(std::string const& recipe, bool cx, std::string const& key)
    {
        if(key.rfind("f64:", 0) == 0)
        {
            return true;
        }
        if(key.rfind("dgett:", 0) == 0)
        {
            return recipe != "f64" || dgettComputesF64(key);
        }
        if(recipe == "f64")
        {
            return false;
        }
        if(recipe == "native")
        {
            return key.rfind("gett32:", 0) == 0;
        }
        std::string const route = cx ? (recipe == "x6" ? "cf32" : "cf32" + recipe) : recipe;
        return key.rfind("gemm:hiptensor_" + route + "_", 0) == 0;
    }

    void runRecipe(RecipeCase const& c, Choice const& choice, SelectionLog* selection)
    {
        ScopedEnv env("HIPTENSOR_ROCKE_F32_RECIPE", c.recipe);
        // unset or an unknown value: native
        std::string recipe = c.recipe != nullptr ? c.recipe : "native";
        if(recipe != "x3" && recipe != "x4" && recipe != "x6" && recipe != "x9" && recipe != "f64")
        {
            recipe = "native";
        }
        bool const                cx = isComplex(c.d);
        std::mt19937              rng(c.seed);
        std::optional<ApiObjects> own;
        if(choice.api == nullptr)
        {
            own.emplace();
        }
        ApiObjects& api = choice.api != nullptr ? *choice.api : *own;
        for(auto t : {c.a, c.b, c.d})
        {
            if(!api.supports(t))
            {
                GTEST_SKIP() << "data type not supported on this device";
            }
        }
        LogCapture                 log;
        std::map<int32_t, int64_t> ext{{'m', 33}, {'p', 5}, {'n', 70}, {'k', 96}, {'q', 3}};
        std::vector<int32_t> ma = {'m', 'k', 'p', 'q'}, mb = {'k', 'n', 'q'}, md = {'n', 'm', 'p'},
                             kmodes = {'k', 'q'};
        std::shuffle(ma.begin(), ma.end(), rng);
        std::shuffle(mb.begin(), mb.end(), rng);
        std::shuffle(md.begin(), md.end(), rng);
        auto A    = makeTensor(c.a, ma, ext, rng, false);
        auto B    = makeTensor(c.b, mb, ext, rng, false);
        auto C    = makeTensor(c.d, md, ext, rng, false);
        auto D    = makeTensor(c.d, md, ext, rng, false);
        D.strides = C.strides;
        D.values.assign(C.values.size(), cplx(0, 0));
        auto const compute    = cx ? HIPTENSOR_COMPUTE_DESC_C32F : HIPTENSOR_COMPUTE_DESC_32F;
        auto       scalarType = scalarTypeOf(compute);
        Scalar     alpha, beta;
        alpha.re = 0.75;
        beta.re  = c.scale ? 0.0 : 0.5;
        if(cx)
        {
            alpha.im = 1.0;
            beta.im  = c.scale ? 0.0 : -0.25;
        }
        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dB = describe(api, B), dD = describe(api, D);
        auto dC = c.scale ? nullptr : describe(api, C);
        ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                             &desc,
                                             dA,
                                             A.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dB,
                                             B.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dC,
                                             c.scale ? nullptr : C.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dD,
                                             D.modes.data(),
                                             compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t     workspaceSize = 0;
        SelectionLog sel;
        auto         plan = planWith(api, desc, choice, log, workspaceSize, &sel);
        if(plan == nullptr)
        {
            return;
        }
        if(selection != nullptr)
        {
            *selection = sel;
        }
        // the recipe's routes only (and the f64 route), never a less accurate one
        EXPECT_TRUE(recipeAdmits(recipe, cx, sel.pick)) << recipe << ": " << sel.pick;
        Device dev;
        void*  pA = dev.upload(A);
        void*  pB = dev.upload(B);
        void*  pC = c.scale ? nullptr : dev.upload(C);
        void*  pD = dev.upload(D);
        void*  ws = nullptr;
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            dev.buffers.push_back(ws);
        }
        Stream stream;
        ASSERT_EQ(hiptensorContract(api.handle,
                                    plan,
                                    alpha.as(scalarType),
                                    pA,
                                    pB,
                                    beta.as(scalarType),
                                    pC,
                                    pD,
                                    ws,
                                    workspaceSize,
                                    stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);

        // the class bound: (2 (n K + 4) u_c + rep) |alpha| S + 2 u_d |d| + 4 u_c |beta c|, K the
        // real products per component (2 per complex product), n the products per product
        auto const          cls   = recipeClass(recipe);
        double const        uo    = unitRoundoff(c.d);
        int64_t             kSize = ext['k'] * ext['q'] * (cx ? 2 : 1);
        std::vector<cplx>   want(D.values.size(), cplx(0, 0));
        std::vector<double> bound(D.values.size(), 0.0);
        std::vector<bool>   written(D.values.size(), false);
        forEachIndex(md, ext, [&](auto index) {
            cplx   sum  = 0;
            double mags = 0;
            forEachIndex(kmodes, ext, [&](auto kindex) {
                auto full = index;
                for(auto& kv : kindex)
                {
                    full[kv.first] = kv.second;
                }
                cplx const a = A.values[A.offset(full)];
                cplx const b = B.values[B.offset(full)];
                sum += a * b;
                mags += (std::fabs(a.real()) + std::fabs(a.imag()))
                        * (std::fabs(b.real()) + std::fabs(b.imag()));
            });
            auto off   = D.offset(index);
            cplx cval  = c.scale ? cplx(0, 0) : C.values[C.offset(index)];
            want[off]  = alpha.value() * sum + beta.value() * cval;
            bound[off] = (2 * (cls.products * kSize + 4) * cls.uc + cls.rep)
                             * std::abs(alpha.value()) * mags
                         + 2 * uo * std::abs(want[off])
                         + 4 * cls.uc * std::abs(beta.value() * cval);
            written[off] = true;
        });
        compare(got, want, bound, written, false, recipe + " contraction (" + sel.pick + ")");
    }

    class RockeF32Recipe : public ::testing::TestWithParam<RecipeCase>
    {
    };

    // Every candidate of the recipe (KERNEL_RANK 0, 1, ...) is one of its routes or the f64
    // route, and meets the recipe's accuracy class.
    TEST_P(RockeF32Recipe, EveryCandidateWithinItsClass)
    {
        auto const c = GetParam();
        Choice     all;
        all.workspace = HIPTENSOR_WORKSPACE_MAX;
        SelectionLog first;
        runRecipe(c, all, &first);
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        ASSERT_GE(first.candidates, 1);
        std::string const recipe = c.recipe != nullptr ? c.recipe : "native";
        bool const        gemm   = deviceArch() == "gfx950";
        bool const        f32    = deviceArch() == "gfx950" || deviceArch() == "gfx942";
        bool const        own    = recipe == "native" ? f32 : recipe != "f64" && gemm;
        // the recipe's own routes (not the f64 route or the direct GETT, which every recipe
        // lists)
        auto const ownRoute = [](std::string const& pick) {
            return pick.rfind("f64:", 0) != 0 && pick.rfind("dgett:", 0) != 0;
        };
        // rocKE's rules order the candidates of every recipe but x6 (whose selection model may
        // rank the f64 route first): the recipe's own route first where the pack has it
        if(recipe != "x6")
        {
            EXPECT_EQ(ownRoute(first.pick), own) << recipe << ": " << first.pick;
        }
        bool                  ownSeen = ownRoute(first.pick);
        std::set<std::string> picks{first.pick};
        for(int32_t r = 1; r < first.candidates; r++)
        {
            Choice choice = all;
            choice.rank   = r;
            SelectionLog s;
            runRecipe(c, choice, &s);
            ASSERT_FALSE(HasFailure()) << "KERNEL_RANK " << r << ": " << s.pick;
            EXPECT_EQ(s.candidates, first.candidates);
            picks.insert(s.pick);
            ownSeen = ownSeen || ownRoute(s.pick);
        }
        EXPECT_EQ(picks.size(), size_t(first.candidates));
        // the recipe's own route is a candidate exactly where the pack has its kernels
        EXPECT_EQ(ownSeen, own) << recipe;
    }

    std::vector<RecipeCase> recipeCases()
    {
        auto const              r16 = HIPTENSOR_R_16F, r32 = HIPTENSOR_R_32F, c32 = HIPTENSOR_C_32F;
        std::vector<RecipeCase> out;
        int                     seed = 4100;
        for(char const* recipe : {(char const*)nullptr, "native", "x3", "x4", "x6", "x9", "f64"})
        {
            for(bool scale : {false, true})
            {
                out.push_back({recipe, r32, r32, r32, scale, seed++});
                out.push_back({recipe, c32, c32, c32, scale, seed++});
            }
            out.push_back({recipe, r16, r32, r16, false, seed++});
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke, RockeF32Recipe, ::testing::ValuesIn(recipeCases()));

    // One full-mantissa product per element (f32 values in [1, 2) with random low bits, the
    // other K terms zero): native and f64 store the product rounded once, x9 and x6 within a
    // few f32 roundings of it, and x3 / x4 visibly lose low bits (the recipes really differ),
    // within 2^-14.
    TEST(RockeF32Recipes, OneProductPerElement)
    {
        ApiObjects                             api;
        int64_t const                          M = 64, N = 48, K = 4;
        std::mt19937                           rng(4200);
        std::uniform_int_distribution<int64_t> mant(1, (int64_t(1) << 23) - 1);
        std::vector<float>                     a(M * K, 0.0f), b(N * K, 0.0f);
        for(int64_t m = 0; m < M; m++)
        {
            a[m * K] = float(1.0 + std::ldexp(double(mant(rng)), -23));
        }
        for(int64_t n = 0; n < N; n++)
        {
            b[n * K] = float(1.0 + std::ldexp(double(mant(rng)), -23));
        }
        // A [K][M] -> modes (k, m) with k fastest: A[k + m K]; B (k, n); D (m, n), m fastest
        std::vector<int32_t> ma{'k', 'm'}, mb{'k', 'n'}, md{'m', 'n'};
        std::vector<int64_t> ea{K, M}, eb{K, N}, ed{M, N};
        auto                 dA = api.tensor(ea, {}, HIPTENSOR_R_32F);
        auto                 dB = api.tensor(eb, {}, HIPTENSOR_R_32F);
        auto                 dD = api.tensor(ed, {}, HIPTENSOR_R_32F);
        float *              pA = nullptr, *pB = nullptr, *pD = nullptr;
        ASSERT_EQ(hipMalloc(&pA, a.size() * 4), hipSuccess);
        ASSERT_EQ(hipMalloc(&pB, b.size() * 4), hipSuccess);
        ASSERT_EQ(hipMalloc(&pD, M * N * 4), hipSuccess);
        ASSERT_EQ(hipMemcpy(pA, a.data(), a.size() * 4, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(pB, b.data(), b.size() * 4, hipMemcpyHostToDevice), hipSuccess);
        bool const gemm = deviceArch() == "gfx950";
        for(char const* recipe : {"native", "x9", "f64", "x6", "x4", "x3"})
        {
            ScopedEnv                      env("HIPTENSOR_ROCKE_F32_RECIPE", recipe);
            LogCapture                     log;
            hiptensorOperationDescriptor_t desc = nullptr;
            ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                                 &desc,
                                                 dA,
                                                 ma.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dB,
                                                 mb.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dD,
                                                 md.data(),
                                                 HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(desc);
            uint64_t     ws = 0;
            SelectionLog sel;
            auto         plan = planWith(api, desc, Choice{}, log, ws, &sel);
            ASSERT_NE(plan, nullptr);
            void* work = nullptr;
            if(ws > 0)
            {
                ASSERT_EQ(hipMalloc(&work, ws), hipSuccess);
            }
            float  one = 1.0f, zero = 0.0f;
            Stream stream;
            ASSERT_EQ(hiptensorContract(
                          api.handle, plan, &one, pA, pB, &zero, nullptr, pD, work, ws, stream.s),
                      HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            std::vector<float> d(M * N);
            ASSERT_EQ(hipMemcpy(d.data(), pD, d.size() * 4, hipMemcpyDeviceToHost), hipSuccess);
            if(work != nullptr)
            {
                static_cast<void>(hipFree(work));
            }
            std::string const r       = recipe;
            bool const        ownGemm = r != "native" && r != "f64" && gemm;
            int64_t           inexact = 0;
            double            worst   = 0.0;
            for(int64_t n = 0; n < N; n++)
            {
                for(int64_t m = 0; m < M; m++)
                {
                    double const exact = double(a[m * K]) * double(b[n * K]); // exact in double
                    double const once  = double(float(exact));
                    double const g     = double(d[m + n * M]);
                    inexact += g != once;
                    worst = std::max(worst, std::fabs(g - exact) / exact);
                }
            }
            if(r == "native" || r == "f64" || !ownGemm)
            {
                // f32 products, f64 (or the f64 route standing in): one rounding
                EXPECT_EQ(inexact, 0) << r << ": " << sel.pick;
            }
            else if(r == "x9" || r == "x6")
            {
                // the part products summed in f32 (x6: without the three smallest)
                EXPECT_LE(worst, std::ldexp(1.0, -20)) << r << ": " << sel.pick;
            }
            else
            {
                // two bf16 parts: about 16 significant bits per operand
                EXPECT_GT(inexact, 0) << r << ": " << sel.pick;
                EXPECT_GT(worst, std::ldexp(1.0, -22)) << r << ": " << sel.pick;
                EXPECT_LE(worst, std::ldexp(1.0, -14)) << r << ": " << sel.pick;
            }
        }
        for(auto* p : {pA, pB, pD})
        {
            static_cast<void>(hipFree(p));
        }
    }

    // An unknown recipe is the default (native), and the plan log says so.
    TEST(RockeF32Recipes, UnknownRecipeIsNative)
    {
        RecipeCase   c{"x7", HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F, false, 4300};
        LogCapture   outer;
        SelectionLog sel;
        runRecipe(c, Choice{}, &sel);
        if(IsSkipped())
        {
            return;
        }
        bool const f32 = deviceArch() == "gfx950" || deviceArch() == "gfx942";
        EXPECT_EQ(sel.pick.rfind("gett32:", 0) == 0, f32) << sel.pick;
        EXPECT_TRUE(outer.contains("HIPTENSOR_ROCKE_F32_RECIPE=x7 is not one of")) << outer.all();
    }

    // ---------------------------------------------------------------------- mixed data types

    // A contraction whose A, B, C and D have different data types. The operands are rounded
    // once to the compute type's precision (operandPrecision), the products accumulate in the
    // compute type and the result is rounded once to D's type. C keeps its own type.
    struct MixedCase
    {
        hiptensorDataType_t          a, b, c, d;
        hiptensorComputeDescriptor_t compute;
        bool                         exact;
        bool                         scale; // no C
        int                          seed;
    };

    void runMixed(MixedCase const& c, Choice const& choice = {}, SelectionLog* selection = nullptr)
    {
        std::mt19937              rng(c.seed);
        std::optional<ApiObjects> own;
        if(choice.api == nullptr)
        {
            own.emplace();
        }
        ApiObjects& api = choice.api != nullptr ? *choice.api : *own;
        for(auto t : {c.a, c.b, c.c, c.d})
        {
            if(!api.supports(t))
            {
                GTEST_SKIP() << "data type not supported on this device";
            }
        }
        if(f64RouteOnly()
           && (c.compute == HIPTENSOR_COMPUTE_DESC_16F || c.compute == HIPTENSOR_COMPUTE_DESC_16BF))
        {
            GTEST_SKIP() << "the f64 route declines 16-bit compute types of mixed operands";
        }
        LogCapture                 log;
        std::vector<int64_t>       pick = {1, 3, 5, 8, 17, 33, 64, 70};
        std::map<int32_t, int64_t> ext;
        for(auto mode : {'m', 'p', 'n', 'k', 'q'})
        {
            ext[mode] = pick[rng() % pick.size()];
        }
        if(!c.exact)
        {
            ext['k'] = 96;
            ext['q'] = 3;
        }
        std::vector<int32_t> ma = {'m', 'k', 'p', 'q'}, mb = {'k', 'n', 'q'}, md = {'n', 'm', 'p'},
                             kmodes = {'k', 'q'};
        std::shuffle(ma.begin(), ma.end(), rng);
        std::shuffle(mb.begin(), mb.end(), rng);
        std::shuffle(md.begin(), md.end(), rng);
        auto A = makeTensor(c.a, ma, ext, rng, c.exact);
        auto B = makeTensor(c.b, mb, ext, rng, c.exact);
        auto C = makeTensor(c.c, md, ext, rng, c.exact);
        auto D = makeTensor(c.d, md, ext, rng, c.exact);
        if(isComplex(c.d) || rng() % 2)
        {
            // complex contractions need C with D's layout
            D.strides = C.strides;
            D.values.assign(C.values.size(), cplx(0, 0));
        }
        auto   scalarType = scalarTypeOf(c.compute);
        Scalar alpha, beta;
        alpha.re = c.exact ? 2.0 : 0.75;
        beta.re  = c.scale ? 0.0 : (c.exact ? -1.0 : 0.5);
        if(isComplex(c.d))
        {
            alpha.im = 1.0;
            beta.im  = c.scale ? 0.0 : 2.0;
        }
        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dB = describe(api, B), dD = describe(api, D);
        auto dC = c.scale ? nullptr : describe(api, C);
        ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                             &desc,
                                             dA,
                                             A.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dB,
                                             B.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dC,
                                             c.scale ? nullptr : C.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dD,
                                             D.modes.data(),
                                             c.compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t workspaceSize = 0;
        auto     plan          = planWith(api, desc, choice, log, workspaceSize, selection);
        if(plan == nullptr)
        {
            return;
        }
        Device dev;
        void*  pA = dev.upload(A);
        void*  pB = dev.upload(B);
        void*  pC = c.scale ? nullptr : dev.upload(C);
        void*  pD = dev.upload(D);
        void*  ws = nullptr;
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            dev.buffers.push_back(ws);
        }
        Stream stream;
        ASSERT_EQ(hiptensorContract(api.handle,
                                    plan,
                                    alpha.as(scalarType),
                                    pA,
                                    pB,
                                    beta.as(scalarType),
                                    pC,
                                    pD,
                                    ws,
                                    workspaceSize,
                                    stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);

        auto const          prec  = operandPrecision(c.compute);
        double const        uc    = accumulationRoundoff(c.compute);
        double const        uo    = unitRoundoff(D.type);
        int64_t             kSize = ext['k'] * ext['q'];
        std::vector<cplx>   want(D.values.size(), cplx(0, 0));
        std::vector<double> bound(D.values.size(), 0.0);
        std::vector<bool>   written(D.values.size(), false);
        forEachIndex(md, ext, [&](auto index) {
            cplx   sum  = 0;
            double mags = 0;
            forEachIndex(kmodes, ext, [&](auto kindex) {
                auto full = index;
                for(auto& kv : kindex)
                {
                    full[kv.first] = kv.second;
                }
                cplx const a = roundOnce(A.values[A.offset(full)], prec);
                cplx const b = roundOnce(B.values[B.offset(full)], prec);
                sum += a * b;
                mags += std::abs(a) * std::abs(b);
            });
            auto off   = D.offset(index);
            cplx cval  = c.scale ? cplx(0, 0) : C.values[C.offset(index)];
            want[off]  = alpha.value() * sum + beta.value() * cval;
            bound[off] = (kSize + 4) * uc * std::abs(alpha.value()) * mags * 2
                         + 2 * uo * std::abs(want[off]) + 4 * uc * std::abs(beta.value() * cval);
            written[off] = true;
        });
        int64_t bad = 0;
        for(size_t i = 0; i < got.values.size(); i++)
        {
            if(!written[i])
            {
                continue;
            }
            // exact data: the reference rounded once to D's type (f64 types keep it)
            cplx const w = c.exact ? roundOnce(want[i], D.type) : want[i];
            if(!matches(got.values[i], w, bound[i], c.exact) && bad++ < 5)
            {
                ADD_FAILURE() << "mixed contraction: element " << i << " got " << got.values[i]
                              << " want " << w << " (bound " << bound[i] << ")";
            }
        }
        EXPECT_EQ(bad, 0) << log.all();
    }

    class RockeMixedContraction : public ::testing::TestWithParam<MixedCase>
    {
    };

    TEST_P(RockeMixedContraction, MatchesLoopReference)
    {
        runMixed(GetParam());
    }

    // Every candidate of an exact mixed case (KERNEL_RANK 0, 1, ...) computes the exact result.
    TEST_P(RockeMixedContraction, EveryCandidateExact)
    {
        auto c = GetParam();
        if(!c.exact)
        {
            GTEST_SKIP() << "random data: covered by MatchesLoopReference";
        }
        Choice all;
        all.workspace = HIPTENSOR_WORKSPACE_MAX;
        SelectionLog first;
        runMixed(c, all, &first);
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        ASSERT_GE(first.candidates, 1);
        std::set<std::string> picks{first.pick};
        for(int32_t r = 1; r < first.candidates; r++)
        {
            Choice choice = all;
            choice.rank   = r;
            SelectionLog s;
            runMixed(c, choice, &s);
            ASSERT_FALSE(HasFailure()) << "KERNEL_RANK " << r << ": " << s.pick;
            picks.insert(s.pick);
        }
        EXPECT_EQ(picks.size(), size_t(first.candidates));
    }

    std::vector<MixedCase> mixedCases()
    {
        struct Row
        {
            hiptensorDataType_t          a, b, c, d;
            hiptensorComputeDescriptor_t compute;
        };
        auto const r16 = HIPTENSOR_R_16F, rb16 = HIPTENSOR_R_16BF, r32 = HIPTENSOR_R_32F,
                   r64 = HIPTENSOR_R_64F, c32 = HIPTENSOR_C_32F, c64 = HIPTENSOR_C_64F;
        Row const rows[] = {
            // 32F: x6 (mixed or f32 operands), h16 (equal 16-bit operands), the f64 route
            {r16, r32, r32, r32, HIPTENSOR_COMPUTE_DESC_32F},
            {rb16, r16, r16, r16, HIPTENSOR_COMPUTE_DESC_32F},
            {r16, r16, r32, r32, HIPTENSOR_COMPUTE_DESC_32F},
            {rb16, rb16, rb16, r32, HIPTENSOR_COMPUTE_DESC_32F},
            {r32, r32, r16, r64, HIPTENSOR_COMPUTE_DESC_32F},
            {r32, r64, r64, r32, HIPTENSOR_COMPUTE_DESC_32F},
            // 16-bit compute types: every operand rounded once
            {r32, r16, r16, r16, HIPTENSOR_COMPUTE_DESC_16F},
            {r64, r32, r32, r16, HIPTENSOR_COMPUTE_DESC_16F},
            {r64, r64, r64, rb16, HIPTENSOR_COMPUTE_DESC_16BF},
            {r16, rb16, r32, rb16, HIPTENSOR_COMPUTE_DESC_16BF},
            // 64F: the f64 route (D written directly, or through T when C's type differs)
            {r16, r32, r64, r64, HIPTENSOR_COMPUTE_DESC_64F},
            {r64, r64, r32, r64, HIPTENSOR_COMPUTE_DESC_64F},
            // complex
            {c32, c64, c64, c64, HIPTENSOR_COMPUTE_DESC_C64F},
            {c64, c32, c32, c32, HIPTENSOR_COMPUTE_DESC_C32F},
            {c32, c32, c32, c64, HIPTENSOR_COMPUTE_DESC_C32F},
            {c32, c32, c64, c32, HIPTENSOR_COMPUTE_DESC_C32F},
        };
        std::vector<MixedCase> out;
        int                    seed = 2100;
        for(auto const& r : rows)
        {
            out.push_back({r.a, r.b, r.c, r.d, r.compute, true, false, seed++});
            out.push_back({r.a, r.b, r.c, r.d, r.compute, true, true, seed++});
            out.push_back({r.a, r.b, r.c, r.d, r.compute, false, false, seed++});
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke, RockeMixedContraction, ::testing::ValuesIn(mixedCases()));

    // Operands of another type than a 16-bit compute type are rounded once to it, also from
    // f64 (no rounding through f32) and to infinity past its range; NaN propagates.
    // D[m, n] = A[m, 0] * 1 + A[m, 1] * 0 with A's values next to the rounding boundaries.
    TEST(RockeMixed, OperandsRoundOnceToTheComputeType)
    {
        if(f64RouteOnly())
        {
            GTEST_SKIP() << "the f64 route declines 16-bit compute types of mixed operands";
        }
        for(auto compute : {HIPTENSOR_COMPUTE_DESC_16F, HIPTENSOR_COMPUTE_DESC_16BF})
        {
            auto const          prec   = operandPrecision(compute);
            int const           p      = prec == HIPTENSOR_R_16F ? 11 : 8;
            int const           emin   = prec == HIPTENSOR_R_16F ? -14 : -126;
            int const           emax   = prec == HIPTENSOR_R_16F ? 15 : 127;
            std::vector<double> probes = {0.0,
                                          -0.0,
                                          1.0,
                                          std::numeric_limits<double>::infinity(),
                                          std::numeric_limits<double>::quiet_NaN(),
                                          1e300,
                                          -1e300};
            std::mt19937        rng(7);
            for(int e : {emin - p, emin - 2, emin, -3, 0, 5, emax - 1, emax})
            {
                double const ulp = std::ldexp(1.0, std::max(e, emin) - (p - 1));
                for(int i = 0; i < 6; i++)
                {
                    double const base = (e >= emin ? std::ldexp(1.0, e) : 0.0)
                                        + double(rng() % (1u << (p - 1))) * ulp;
                    double const mid = base + ulp / 2;
                    for(double d : {-std::ldexp(1.0, -12),
                                    -std::ldexp(1.0, -30),
                                    -std::ldexp(1.0, -38),
                                    0.0,
                                    std::ldexp(1.0, -38),
                                    std::ldexp(1.0, -30),
                                    std::ldexp(1.0, -12)})
                    {
                        probes.push_back(mid + ulp * d);
                        probes.push_back(-(mid + ulp * d));
                    }
                }
            }
            double const top = std::ldexp(2.0 - std::ldexp(1.0, 1 - p), emax);
            for(double v : {top, top + std::ldexp(1.0, emax - p), top * (1 + 1e-12)})
            {
                probes.push_back(v);
                probes.push_back(-v);
            }
            int64_t const M = int64_t(probes.size()), N = 3;
            ApiObjects    api;
            LogCapture    log;
            // A[m, k] (f64, m fastest), B[k, n] (f32), D[m, n] (f32 holds every 16-bit value)
            std::vector<double> a(M * 2, 0.0);
            for(int64_t m = 0; m < M; m++)
            {
                a[m]     = probes[m];
                a[m + M] = 3.0;
            }
            std::vector<float> b(2 * N);
            for(int64_t n = 0; n < N; n++)
            {
                b[0 + 2 * n] = 1.0f;
                b[1 + 2 * n] = 0.0f;
            }
            int32_t const                  ma[] = {'m', 'k'}, mb[] = {'k', 'n'}, md[] = {'m', 'n'};
            auto                           dA   = api.tensor({M, 2}, {}, HIPTENSOR_R_64F);
            auto                           dB   = api.tensor({2, N}, {}, HIPTENSOR_R_32F);
            auto                           dD   = api.tensor({M, N}, {}, HIPTENSOR_R_32F);
            hiptensorOperationDescriptor_t desc = nullptr;
            ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                                 &desc,
                                                 dA,
                                                 ma,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dB,
                                                 mb,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dD,
                                                 md,
                                                 compute),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(desc);
            uint64_t ws   = 0;
            auto     plan = planOf(api, desc, ws, log);
            ASSERT_NE(plan, nullptr);
            void *pA = nullptr, *pB = nullptr, *pD = nullptr, *pW = nullptr;
            ASSERT_EQ(hipMalloc(&pA, a.size() * 8), hipSuccess);
            ASSERT_EQ(hipMalloc(&pB, b.size() * 4), hipSuccess);
            ASSERT_EQ(hipMalloc(&pD, M * N * 4), hipSuccess);
            if(ws > 0)
            {
                ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
            }
            ASSERT_EQ(hipMemcpy(pA, a.data(), a.size() * 8, hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(pB, b.data(), b.size() * 4, hipMemcpyHostToDevice), hipSuccess);
            Scalar alpha, beta;
            alpha.re = 1.0;
            Stream stream;
            ASSERT_EQ(hiptensorContract(api.handle,
                                        plan,
                                        alpha.as(scalarTypeOf(compute)),
                                        pA,
                                        pB,
                                        beta.as(scalarTypeOf(compute)),
                                        nullptr,
                                        pD,
                                        pW,
                                        ws,
                                        stream.s),
                      HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            std::vector<float> d(M * N);
            ASSERT_EQ(hipMemcpy(d.data(), pD, d.size() * 4, hipMemcpyDeviceToHost), hipSuccess);
            for(auto* ptr : {pA, pB, pD, pW})
            {
                static_cast<void>(hipFree(ptr));
            }
            int64_t bad = 0;
            for(int64_t m = 0; m < M; m++)
            {
                double const want = roundOnce(probes[m], prec);
                for(int64_t n = 0; n < N; n++)
                {
                    double const g = d[m + M * n];
                    if(!matches(g, want, 0.0, true) && bad++ < 8)
                    {
                        ADD_FAILURE() << "compute " << compute << ": A = " << probes[m] << " gave "
                                      << g << ", want " << want;
                    }
                }
            }
            EXPECT_EQ(bad, 0) << log.all();
        }
    }

    // ---------------------------------------------------------------------- trinary contraction

    // E = alpha * opA(A) * B * opC(C) + beta * opD(D) in hipTensor's two steps: T = opA(A) * B,
    // stored in E's data type, then E = alpha * T * opC(C) + beta * opD(D); every operand of a
    // step rounded to the precision of a 16-bit compute type. Modes: A {m, k, j}, B {k, u},
    // C {u, j, n}, D and E {m, n}: step 1 contracts k, T has {m, j, u}, step 2 contracts j, u.
    struct TrinaryCase
    {
        hiptensorDataType_t          type;
        hiptensorComputeDescriptor_t compute;
        int                          seed;
        bool                         exact;
        bool                         bilinear; // D given
        hiptensorOperator_t          opA     = HIPTENSOR_OP_IDENTITY;
        hiptensorOperator_t          opC     = HIPTENSOR_OP_IDENTITY;
        hiptensorOperator_t          opD     = HIPTENSOR_OP_IDENTITY;
        bool                         inPlace = false; // D is E's buffer
        // extents of m, k, j, u, n (empty: drawn from the seed)
        std::vector<int64_t> extents = {};
    };

    void runTrinary(TrinaryCase const& c,
                    Choice const&      choice    = {},
                    SelectionLog*      selection = nullptr)
    {
        std::mt19937              rng(c.seed);
        std::optional<ApiObjects> own;
        if(choice.api == nullptr)
        {
            own.emplace();
        }
        ApiObjects& api = choice.api != nullptr ? *choice.api : *own;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        if(f64RouteOnly() && c.type == HIPTENSOR_R_32F && c.compute != HIPTENSOR_COMPUTE_DESC_32F)
        {
            GTEST_SKIP() << "the f64 route declines f32 data with a 16-bit compute type";
        }
        LogCapture                 log;
        std::vector<int64_t>       pick = {1, 2, 3, 5, 8, 17};
        std::map<int32_t, int64_t> ext;
        for(auto mode : {'m', 'k', 'j', 'u', 'n'})
        {
            ext[mode] = pick[rng() % pick.size()];
        }
        if(!c.exact)
        {
            ext['k'] = 48;
            ext['u'] = 6;
        }
        if(c.extents.size() == 5)
        {
            ext = {{'m', c.extents[0]},
                   {'k', c.extents[1]},
                   {'j', c.extents[2]},
                   {'u', c.extents[3]},
                   {'n', c.extents[4]}};
        }
        std::vector<int32_t> ma = {'m', 'k', 'j'}, mb = {'k', 'u'}, mc = {'u', 'j', 'n'},
                             me = {'m', 'n'};
        std::shuffle(ma.begin(), ma.end(), rng);
        std::shuffle(mb.begin(), mb.end(), rng);
        std::shuffle(mc.begin(), mc.end(), rng);
        std::shuffle(me.begin(), me.end(), rng);
        auto A = makeTensor(c.type, ma, ext, rng, c.exact, 2);
        auto B = makeTensor(c.type, mb, ext, rng, c.exact, 2);
        auto C = makeTensor(c.type, mc, ext, rng, c.exact, 2);
        auto D = makeTensor(c.type, me, ext, rng, c.exact, 2);
        auto E = makeTensor(c.type, me, ext, rng, c.exact, 2);
        if(isComplex(c.type) || c.inPlace || rng() % 2)
        {
            // complex contractions need D with E's layout
            E.strides = D.strides;
            E.values  = c.inPlace ? D.values : std::vector<cplx>(D.values.size(), cplx(0, 0));
        }
        auto   scalarType = scalarTypeOf(c.compute);
        Scalar alpha, beta;
        alpha.re = c.exact ? 2.0 : 0.75;
        beta.re  = c.bilinear ? (c.exact ? -1.0 : 0.5) : 0.0;
        if(isComplex(c.type))
        {
            alpha.im = 1.0;
            beta.im  = c.bilinear ? 2.0 : 0.0;
        }
        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dB = describe(api, B), dC = describe(api, C);
        auto dE = describe(api, E);
        auto dD = c.bilinear ? describe(api, D) : nullptr;
        ASSERT_EQ(hiptensorCreateContractionTrinary(api.handle,
                                                    &desc,
                                                    dA,
                                                    A.modes.data(),
                                                    c.opA,
                                                    dB,
                                                    B.modes.data(),
                                                    HIPTENSOR_OP_IDENTITY,
                                                    dC,
                                                    C.modes.data(),
                                                    c.opC,
                                                    dD,
                                                    c.bilinear ? D.modes.data() : nullptr,
                                                    c.opD,
                                                    dE,
                                                    E.modes.data(),
                                                    c.compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t workspaceSize = 0;
        auto     plan          = planWith(api, desc, choice, log, workspaceSize, selection);
        if(plan == nullptr)
        {
            return;
        }
        // the intermediate T needs workspace whatever the steps run on
        int64_t tCount = ext['m'] * ext['j'] * ext['u'];
        EXPECT_GE(workspaceSize, uint64_t(tCount) * sizeOf(c.type));
        DeviceScope onDevice(choice.device);
        Device      dev;
        void*       pA = dev.upload(A, choice.skew);
        void*       pB = dev.upload(B, choice.skew);
        void*       pC = dev.upload(C, choice.skew);
        void*       pE = dev.upload(E, choice.skew);
        void*       pD = c.inPlace ? pE : (c.bilinear ? dev.upload(D, choice.skew) : nullptr);
        void*       ws = nullptr;
        if(workspaceSize > 0)
        {
            // NaN in every byte: nothing of the workspace's earlier contents may reach E
            ASSERT_EQ(hipMalloc(&ws, workspaceSize), hipSuccess);
            ASSERT_EQ(hipMemset(ws, 0xFF, workspaceSize), hipSuccess);
            dev.buffers.push_back(ws);
        }
        Stream stream;
        ASSERT_EQ(hiptensorContractTrinary(api.handle,
                                           plan,
                                           alpha.as(scalarType),
                                           pA,
                                           pB,
                                           pC,
                                           beta.as(scalarType),
                                           pD,
                                           pE,
                                           ws,
                                           workspaceSize,
                                           stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = E;
        dev.download(pE, got);

        // reference: the two steps, T rounded to E's data type
        auto const   prec   = operandPrecision(c.compute);
        double const uc     = isComplex(c.type) || c.type == HIPTENSOR_R_64F
                                  ? accumulationRoundoff(c.compute)
                                  : std::ldexp(1.0, -24);
        double const ut     = unitRoundoff(c.type); // T and E
        double const up     = prec == HIPTENSOR_R_64F ? 0.0 : unitRoundoff(prec);
        auto const   tStore = isComplex(c.type) ? HIPTENSOR_C_32F : c.type;
        std::map<std::tuple<int64_t, int64_t, int64_t>, std::pair<cplx, double>> T; // value, bound
        for(int64_t m = 0; m < ext['m']; m++)
        {
            for(int64_t j = 0; j < ext['j']; j++)
            {
                for(int64_t u = 0; u < ext['u']; u++)
                {
                    cplx   sum  = 0;
                    double mags = 0;
                    for(int64_t k = 0; k < ext['k']; k++)
                    {
                        std::map<int32_t, int64_t> i{{'m', m}, {'k', k}, {'j', j}, {'u', u}};
                        cplx const a = roundOnce(applyOp(c.opA, A.values[A.offset(i)]), prec);
                        cplx const b = roundOnce(B.values[B.offset(i)], prec);
                        sum += a * b;
                        mags += std::abs(a) * std::abs(b);
                    }
                    cplx const   t     = c.type == HIPTENSOR_R_64F || c.type == HIPTENSOR_C_64F
                                             ? sum
                                             : roundOnce(sum, tStore);
                    double const accum = 2 * (ext['k'] + 4) * uc * mags;
                    T[{m, j, u}]       = {t, accum * (1 + 2 * ut) + 2 * ut * std::abs(sum)};
                }
            }
        }
        std::vector<cplx>   want(E.values.size(), cplx(0, 0));
        std::vector<double> bound(E.values.size(), 0.0);
        std::vector<bool>   written(E.values.size(), false);
        int64_t const       k2 = ext['j'] * ext['u'];
        for(int64_t m = 0; m < ext['m']; m++)
        {
            for(int64_t n = 0; n < ext['n']; n++)
            {
                std::map<int32_t, int64_t> i{{'m', m}, {'n', n}};
                cplx                       sum  = 0;
                double                     mags = 0, carried = 0;
                for(int64_t j = 0; j < ext['j']; j++)
                {
                    for(int64_t u = 0; u < ext['u']; u++)
                    {
                        i['j']              = j;
                        i['u']              = u;
                        auto const& [t, bt] = T[{m, j, u}];
                        cplx const tc       = roundOnce(t, prec);
                        cplx const cc = roundOnce(applyOp(c.opC, C.values[C.offset(i)]), prec);
                        sum += tc * cc;
                        mags += std::abs(tc) * std::abs(cc);
                        carried += (bt + 2 * up * std::abs(t)) * std::abs(cc);
                    }
                }
                cplx const dval = c.bilinear ? applyOp(c.opD, D.values[D.offset(i)]) : cplx(0, 0);
                auto const off  = E.offset(i);
                want[off]       = alpha.value() * sum + beta.value() * dval;
                bound[off]      = std::abs(alpha.value()) * (2 * (k2 + 4) * uc * mags + carried)
                             + 2 * ut * std::abs(want[off])
                             + 4 * uc * std::abs(beta.value() * dval);
                written[off] = true;
            }
        }
        int64_t bad = 0;
        for(size_t i = 0; i < got.values.size(); i++)
        {
            if(!written[i])
            {
                continue;
            }
            cplx const w = c.exact && c.type != HIPTENSOR_R_64F && c.type != HIPTENSOR_C_64F
                               ? roundOnce(want[i], tStore)
                               : want[i];
            if(!matches(got.values[i], w, bound[i], c.exact) && bad++ < 5)
            {
                ADD_FAILURE() << "trinary contraction: element " << i << " got " << got.values[i]
                              << " want " << w << " (bound " << bound[i] << ")";
            }
        }
        EXPECT_EQ(bad, 0) << log.all();
    }

    class RockeTrinaryContraction : public ::testing::TestWithParam<TrinaryCase>
    {
    };

    TEST_P(RockeTrinaryContraction, MatchesLoopReference)
    {
        runTrinary(GetParam());
    }

    // Every candidate (a pair of a candidate of each step) computes the exact result, with
    // aligned buffers and with A, B, C, D and E one element past aligned addresses (the
    // fallbacks of the steps' launches and programs).
    TEST_P(RockeTrinaryContraction, EveryCandidateExact)
    {
        auto const c = GetParam();
        if(!c.exact)
        {
            GTEST_SKIP() << "random data: covered by MatchesLoopReference";
        }
        for(size_t skew : {size_t(0), size_t(1)})
        {
            Choice all;
            all.workspace = HIPTENSOR_WORKSPACE_MAX;
            all.skew      = skew;
            SelectionLog first;
            runTrinary(c, all, &first);
            if(HasFatalFailure() || IsSkipped())
            {
                return;
            }
            ASSERT_GE(first.candidates, 1);
            EXPECT_EQ(first.pick.rfind("tri:", 0), 0u) << first.pick;
            std::set<std::string> picks{first.pick};
            for(int32_t r = 1; r < first.candidates; r++)
            {
                Choice choice = all;
                choice.rank   = r;
                SelectionLog s;
                runTrinary(c, choice, &s);
                ASSERT_FALSE(HasFailure())
                    << "KERNEL_RANK " << r << " skew " << skew << ": " << s.pick;
                EXPECT_EQ(s.candidates, first.candidates);
                picks.insert(s.pick);
            }
            EXPECT_EQ(picks.size(), size_t(first.candidates));
        }
    }

    std::vector<TrinaryCase> trinaryCases()
    {
        struct Row
        {
            hiptensorDataType_t          type;
            hiptensorComputeDescriptor_t compute;
        };
        Row const                rows[] = {{HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F},
                                           {HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_32F},
                                           {HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F},
                                           {HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_16F},
                                           {HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F},
                                           {HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F},
                                           {HIPTENSOR_C_64F, HIPTENSOR_COMPUTE_DESC_C64F}};
        std::vector<TrinaryCase> out;
        int                      seed = 2300;
        for(auto const& r : rows)
        {
            bool const cx = isComplex(r.type);
            auto const op = cx ? HIPTENSOR_OP_CONJ : HIPTENSOR_OP_RELU;
            auto const od = cx ? HIPTENSOR_OP_CONJ : HIPTENSOR_OP_NEG;
            out.push_back({r.type, r.compute, seed++, true, true});
            out.push_back({r.type, r.compute, seed++, true, false});
            out.push_back({r.type, r.compute, seed++, true, true, op, op, od});
            out.push_back({r.type,
                           r.compute,
                           seed++,
                           true,
                           true,
                           HIPTENSOR_OP_IDENTITY,
                           HIPTENSOR_OP_IDENTITY,
                           HIPTENSOR_OP_IDENTITY,
                           true});
            out.push_back({r.type, r.compute, seed++, false, true});
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke, RockeTrinaryContraction, ::testing::ValuesIn(trinaryCases()));

    // The API's checks of a trinary contraction's execution: alpha, A, B, C and E are needed,
    // D when beta is not zero, and the workspace the plan requires.
    TEST(RockeTrinary, ExecutionChecks)
    {
        ApiObjects    api;
        LogCapture    log;
        int64_t const M = 9, K = 5, U = 4, N = 7;
        int32_t const ma[] = {'m', 'k'}, mb[] = {'k', 'u'}, mc[] = {'u', 'n'}, me[] = {'m', 'n'};
        auto          f32 = HIPTENSOR_R_32F;
        auto          dA = api.tensor({M, K}, {}, f32), dB = api.tensor({K, U}, {}, f32);
        auto          dC = api.tensor({U, N}, {}, f32), dE = api.tensor({M, N}, {}, f32);
        hiptensorOperationDescriptor_t desc = nullptr;
        ASSERT_EQ(hiptensorCreateContractionTrinary(api.handle,
                                                    &desc,
                                                    dA,
                                                    ma,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    dB,
                                                    mb,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    dC,
                                                    mc,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    dE,
                                                    me,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    dE,
                                                    me,
                                                    HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t ws   = 0;
        auto     plan = planOf(api, desc, ws, log);
        ASSERT_NE(plan, nullptr);
        ASSERT_GE(ws, uint64_t(M * U * 4));
        std::vector<float> a(M * K, 1.0f), b(K * U, 1.0f), cc(U * N, 1.0f), e(M * N, 2.0f);
        void *             pA = nullptr, *pB = nullptr, *pC = nullptr, *pE = nullptr, *pW = nullptr;
        for(auto [p, v] : {std::make_pair(&pA, &a),
                           std::make_pair(&pB, &b),
                           std::make_pair(&pC, &cc),
                           std::make_pair(&pE, &e)})
        {
            ASSERT_EQ(hipMalloc(p, v->size() * 4), hipSuccess);
            ASSERT_EQ(hipMemcpy(*p, v->data(), v->size() * 4, hipMemcpyHostToDevice), hipSuccess);
        }
        ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
        float const one = 1.0f, zero = 0.0f;
        auto        run = [&](void const* alpha,
                       void const* A,
                       void const* C,
                       void const* beta,
                       void const* D,
                       void*       W,
                       uint64_t    size) {
            return hiptensorContractTrinary(
                api.handle, plan, alpha, A, pB, C, beta, D, pE, W, size, nullptr);
        };
        EXPECT_EQ(run(nullptr, pA, pC, &one, pE, pW, ws), HIPTENSOR_STATUS_INVALID_VALUE);
        EXPECT_EQ(run(&one, nullptr, pC, &one, pE, pW, ws), HIPTENSOR_STATUS_INVALID_VALUE);
        EXPECT_EQ(run(&one, pA, nullptr, &one, pE, pW, ws), HIPTENSOR_STATUS_INVALID_VALUE);
        EXPECT_EQ(run(&one, pA, pC, &one, nullptr, pW, ws), HIPTENSOR_STATUS_INVALID_VALUE);
        EXPECT_EQ(run(&one, pA, pC, &one, pE, pW, ws - 1), HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE);
        EXPECT_EQ(run(&one, pA, pC, &one, pE, nullptr, ws),
                  HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE);
        // beta 0: D is not read (may be null); E = A * B * C = K * U everywhere
        ASSERT_EQ(run(&one, pA, pC, &zero, nullptr, pW, ws), HIPTENSOR_STATUS_SUCCESS) << log.all();
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        ASSERT_EQ(hipMemcpy(e.data(), pE, e.size() * 4, hipMemcpyDeviceToHost), hipSuccess);
        for(auto v : e)
        {
            EXPECT_EQ(v, float(K * U));
        }
        for(auto* p : {pA, pB, pC, pE, pW})
        {
            static_cast<void>(hipFree(p));
        }
    }

    // Trinary contractions whose steps run tiled operand packs and epilogues (extents and strides
    // that keep the tiled kernel's vectors aligned, on the emulated-f32 and 16-bit GEMM routes of
    // gfx950): every candidate is exact with aligned buffers, with every buffer one element past
    // an aligned address (each tiled launch then runs the generic launches it replaced, which
    // address T in the workspace and the trinary's tensors exactly as the tiled launch does), and
    // executed on another device of the same architecture than the plan was created on, when the
    // machine has one.
    TEST(RockeTrinary, TiledStepLaunches)
    {
        int const        other   = otherDevice();
        std::vector<int> devices = {-1};
        if(other >= 0)
        {
            devices.push_back(other);
        }
        bool tiled = false;
        int  seed  = 4300;
        for(auto [type, compute] : {std::pair{HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F},
                                    std::pair{HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F}})
        {
            for(int layout = 0; layout < 3; layout++)
            {
                TrinaryCase c{type, compute, seed++, true, layout != 1};
                c.extents = {32, 16, 16, 8, 32};
                for(int device : devices)
                {
                    for(size_t skew : {size_t(0), size_t(1)})
                    {
                        for(int32_t rank = 0;; rank++)
                        {
                            Choice choice;
                            choice.rank      = rank;
                            choice.skew      = skew;
                            choice.device    = device;
                            choice.workspace = HIPTENSOR_WORKSPACE_MAX;
                            SelectionLog sel;
                            runTrinary(c, choice, &sel);
                            if(IsSkipped())
                            {
                                return;
                            }
                            ASSERT_FALSE(HasFailure())
                                << "seed " << c.seed << " KERNEL_RANK " << rank << " skew " << skew
                                << " device " << device << ": " << sel.pick;
                            tiled = tiled || sel.launches.find("ewt") != std::string::npos;
                            if(rank + 1 >= sel.candidates)
                            {
                                break;
                            }
                        }
                    }
                }
            }
        }
        // gfx950's GEMM routes run tiled packs and epilogues (other targets may too: the packs
        // and epilogues of the f64 and native f32 routes)
        if(deviceArch() == "gfx950")
        {
            EXPECT_TRUE(tiled);
        }
        RecordProperty("executed_on_another_device", other >= 0 ? 1 : 0);
    }

    // The intermediate T has E's data type: with f32 A and B and an f16 E, T = A * B is rounded
    // to f16 before the second step (integer data whose T values are past f16's integer
    // range, so the rounding of T changes E; everything else is exact). Every candidate.
    TEST(RockeTrinary, IntermediateInOutputType)
    {
        ApiObjects    api;
        int64_t const M = 5, K = 701, U = 2, N = 3;
        int32_t const ma[] = {'m', 'k'}, mb[] = {'k', 'u'}, mc[] = {'u', 'n'}, me[] = {'m', 'n'};
        auto          dA                    = api.tensor({M, K}, {}, HIPTENSOR_R_32F),
             dB                             = api.tensor({K, U}, {}, HIPTENSOR_R_32F);
        auto dC                             = api.tensor({U, N}, {}, HIPTENSOR_R_32F),
             dE                             = api.tensor({M, N}, {}, HIPTENSOR_R_16F);
        hiptensorOperationDescriptor_t desc = nullptr;
        ASSERT_EQ(hiptensorCreateContractionTrinary(api.handle,
                                                    &desc,
                                                    dA,
                                                    ma,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    dB,
                                                    mb,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    dC,
                                                    mc,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    nullptr,
                                                    nullptr,
                                                    HIPTENSOR_OP_IDENTITY,
                                                    dE,
                                                    me,
                                                    HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        std::vector<float> a(M * K), b(K * U), c(U * N);
        for(int64_t i = 0; i < M * K; i++)
        {
            a[i] = float(1 + (i * 7) % 3);
        }
        for(int64_t i = 0; i < K * U; i++)
        {
            b[i] = float(1 + (i * 5) % 3);
        }
        for(int64_t i = 0; i < U * N; i++)
        {
            c[i] = float(1 + i % 2);
        }
        // reference: T rounded once to f16, then E = sum_u T * C rounded once to f16
        std::vector<double> want(M * N);
        int64_t             rounded = 0;
        for(int64_t m = 0; m < M; m++)
        {
            for(int64_t n = 0; n < N; n++)
            {
                double e = 0;
                for(int64_t u = 0; u < U; u++)
                {
                    double t = 0;
                    for(int64_t k = 0; k < K; k++)
                    {
                        t += double(a[m + M * k]) * double(b[k + K * u]);
                    }
                    double const tr = roundOnce(t, HIPTENSOR_R_16F);
                    rounded += tr != t ? 1 : 0;
                    e += tr * double(c[u + U * n]);
                }
                want[m + M * n] = roundOnce(e, HIPTENSOR_R_16F);
            }
        }
        ASSERT_GT(rounded, 0) << "the case must round T";
        void *pA = nullptr, *pB = nullptr, *pC = nullptr, *pE = nullptr;
        ASSERT_EQ(hipMalloc(&pA, a.size() * 4), hipSuccess);
        ASSERT_EQ(hipMalloc(&pB, b.size() * 4), hipSuccess);
        ASSERT_EQ(hipMalloc(&pC, c.size() * 4), hipSuccess);
        ASSERT_EQ(hipMalloc(&pE, M * N * 2), hipSuccess);
        ASSERT_EQ(hipMemcpy(pA, a.data(), a.size() * 4, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(pB, b.data(), b.size() * 4, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(pC, c.data(), c.size() * 4, hipMemcpyHostToDevice), hipSuccess);
        int32_t candidates = 1;
        for(int32_t rank = 0; rank < candidates; rank++)
        {
            LogCapture log;
            Choice     choice;
            choice.rank      = rank;
            choice.workspace = HIPTENSOR_WORKSPACE_MAX;
            choice.api       = &api;
            uint64_t     ws  = 0;
            SelectionLog sel;
            auto         plan = planWith(api, desc, choice, log, ws, &sel);
            ASSERT_NE(plan, nullptr);
            candidates = sel.candidates;
            void* pW   = nullptr;
            ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
            float const alpha = 1.0f, beta = 0.0f;
            ASSERT_EQ(
                hiptensorContractTrinary(
                    api.handle, plan, &alpha, pA, pB, pC, &beta, nullptr, pE, pW, ws, nullptr),
                HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            static_cast<void>(hipFree(pW));
            std::vector<_Float16> e(M * N);
            ASSERT_EQ(hipMemcpy(e.data(), pE, e.size() * 2, hipMemcpyDeviceToHost), hipSuccess);
            for(int64_t i = 0; i < M * N; i++)
            {
                EXPECT_EQ(double(e[i]), want[i]) << sel.pick << " E element " << i;
            }
        }
        for(auto* p : {pA, pB, pC, pE})
        {
            static_cast<void>(hipFree(p));
        }
    }

    // NaN in an input reaches exactly the outputs it contributes to; NaN in D does not reach E
    // when beta is zero; an infinity in A gives infinite outputs (f64 and f16 data).
    TEST(RockeTrinary, NonFiniteValues)
    {
        for(auto type : {HIPTENSOR_R_64F, HIPTENSOR_R_16F})
        {
            ApiObjects api;
            if(!api.supports(type))
            {
                continue;
            }
            LogCapture    log;
            int64_t const M = 6, K = 4, U = 3, N = 5;
            int32_t const ma[] = {'m', 'k'}, mb[] = {'k', 'u'}, mc[] = {'u', 'n'},
                          me[] = {'m', 'n'};
            auto dA = api.tensor({M, K}, {}, type), dB = api.tensor({K, U}, {}, type);
            auto dC = api.tensor({U, N}, {}, type), dE = api.tensor({M, N}, {}, type);
            auto compute
                = type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
            hiptensorOperationDescriptor_t desc = nullptr;
            ASSERT_EQ(hiptensorCreateContractionTrinary(api.handle,
                                                        &desc,
                                                        dA,
                                                        ma,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        dB,
                                                        mb,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        dC,
                                                        mc,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        dE,
                                                        me,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        dE,
                                                        me,
                                                        compute),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(desc);
            uint64_t ws   = 0;
            auto     plan = planOf(api, desc, ws, log);
            ASSERT_NE(plan, nullptr);
            double const nan = std::numeric_limits<double>::quiet_NaN();
            double const inf = std::numeric_limits<double>::infinity();
            Tensor       A, B, C, D, E;
            auto         packed = [&](Tensor& t, std::vector<int32_t> modes, int64_t r, int64_t c) {
                t.type    = type;
                t.modes   = modes;
                t.extents = {r, c};
                t.strides = {1, r};
                t.values.assign(r * c, cplx(1.0, 0.0));
            };
            packed(A, {'m', 'k'}, M, K);
            packed(B, {'k', 'u'}, K, U);
            packed(C, {'u', 'n'}, U, N);
            packed(D, {'m', 'n'}, M, N);
            packed(E, {'m', 'n'}, M, N);
            A.values[1 + M * 2] = nan; // row m = 1
            A.values[3 + M * 0] = inf; // row m = 3
            for(auto& v : D.values)
            {
                v = nan; // beta 0: not read
            }
            Device dev;
            void*  pA = dev.upload(A);
            void*  pB = dev.upload(B);
            void*  pC = dev.upload(C);
            void*  pD = dev.upload(D);
            void*  pE = dev.upload(E);
            void*  pW = nullptr;
            if(ws > 0)
            {
                ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
                dev.buffers.push_back(pW);
            }
            Scalar alpha, beta;
            alpha.re = 1.0;
            Stream stream;
            ASSERT_EQ(hiptensorContractTrinary(api.handle,
                                               plan,
                                               alpha.as(scalarTypeOf(compute)),
                                               pA,
                                               pB,
                                               pC,
                                               beta.as(scalarTypeOf(compute)),
                                               pD,
                                               pE,
                                               pW,
                                               ws,
                                               stream.s),
                      HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            Tensor got = E;
            dev.download(pE, got);
            for(int64_t m = 0; m < M; m++)
            {
                for(int64_t n = 0; n < N; n++)
                {
                    double const g    = got.values[m + M * n].real();
                    double const want = m == 1 ? nan : m == 3 ? inf : double(K * U);
                    EXPECT_TRUE(matches(g, want, 0.0, true))
                        << "type " << type << " E[" << m << "," << n << "] = " << g << "\n"
                        << log.all();
                }
            }
        }
    }

    // ---------------------------------------------------------------------- tensors of 2 GiB

    // Tensors of 2 GiB and more, on the 64-bit-address paths: the 64-bit-address GEMM body
    // (f16, f32, complex f32), the f64 GEMM in host-iterated launches over M (and over K for a
    // row of A past 2 GiB), elementwise pieces of a mode past 2^31 elements, and reductions
    // over more than 2^31 reduced elements or outputs. Integer data: sampled outputs exact.
    // Every candidate of a contraction (KERNEL_RANK) is checked.
    bool hugeMemory(uint64_t bytes)
    {
        size_t freeBytes = 0, totalBytes = 0;
        return hipMemGetInfo(&freeBytes, &totalBytes) == hipSuccess && freeBytes >= bytes;
    }

    // Values of a huge tensor: small integers (-1, 0, 1) from a hash of the element index (no
    // period, so an element read from a wrong offset shows).
    int64_t hugeInt(int64_t i, int seed)
    {
        uint64_t z = uint64_t(i) * 4 + uint64_t(seed) + 0x9E3779B97F4A7C15ull;
        z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        return int64_t(z % 3) - 1;
    }

    double hugeValue(int64_t i, int seed)
    {
        return double(hugeInt(i, seed));
    }

    template <typename T>
    std::vector<T> hugeFill(int64_t count, int seed)
    {
        std::vector<T> v(count);
        for(int64_t i = 0; i < count; i++)
        {
            v[i] = static_cast<T>(hugeValue(i, seed));
        }
        return v;
    }

    struct HugeContractionCase
    {
        hiptensorDataType_t          type;
        hiptensorComputeDescriptor_t compute;
        int64_t                      M, N, K;
        bool                         bilinear;
        char const*                  recipe = nullptr; // HIPTENSOR_ROCKE_F32_RECIPE
    };

    // D[m, n] = alpha * sum_k A[k, m] B[k, n] + beta * C[m, n], packed column-major, every
    // candidate; 512 sampled outputs against the loop reference (exact).
    void runHugeContraction(HugeContractionCase const& c)
    {
        ScopedEnv  recipe("HIPTENSOR_ROCKE_F32_RECIPE", c.recipe);
        ApiObjects api;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        bool const     cx     = isComplex(c.type);
        size_t const   es     = sizeOf(c.type);
        uint64_t const aBytes = uint64_t(c.M) * c.K * es, bBytes = uint64_t(c.N) * c.K * es;
        if(!hugeMemory(4 * (aBytes + bBytes) + (uint64_t(16) << 30)))
        {
            GTEST_SKIP() << "not enough free device memory for tensors of 2 GiB";
        }
        // host values (complex: interleaved components)
        int64_t const comps  = cx ? 2 : 1;
        auto          hostOf = [&](int64_t count, int seed) {
            std::vector<uint8_t> bytes(count * es);
            for(int64_t i = 0; i < count * comps; i++)
            {
                double const v = hugeValue(i, seed);
                switch(c.type)
                {
                case HIPTENSOR_R_16F:
                {
                    _Float16 h = static_cast<_Float16>(v);
                    std::memcpy(bytes.data() + 2 * i, &h, 2);
                    break;
                }
                case HIPTENSOR_R_32F:
                case HIPTENSOR_C_32F:
                {
                    float f = static_cast<float>(v);
                    std::memcpy(bytes.data() + 4 * i, &f, 4);
                    break;
                }
                default:
                    std::memcpy(bytes.data() + 8 * i, &v, 8);
                    break;
                }
            }
            return bytes;
        };
        auto value = [&](int64_t i, int seed) {
            return cx ? cplx(hugeValue(2 * i, seed), hugeValue(2 * i + 1, seed))
                      : cplx(hugeValue(i, seed), 0.0);
        };
        auto  hA = hostOf(c.M * c.K, 1), hB = hostOf(c.N * c.K, 2), hC = hostOf(c.M * c.N, 3);
        void *pA = nullptr, *pB = nullptr, *pC = nullptr, *pD = nullptr;
        ASSERT_EQ(hipMalloc(&pA, hA.size()), hipSuccess);
        ASSERT_EQ(hipMalloc(&pB, hB.size()), hipSuccess);
        ASSERT_EQ(hipMalloc(&pC, hC.size()), hipSuccess);
        ASSERT_EQ(hipMalloc(&pD, hC.size()), hipSuccess);
        ASSERT_EQ(hipMemcpy(pA, hA.data(), hA.size(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(pB, hB.data(), hB.size(), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(pC, hC.data(), hC.size(), hipMemcpyHostToDevice), hipSuccess);
        hA.clear();
        hA.shrink_to_fit();
        hB.clear();
        hB.shrink_to_fit();
        int32_t const ma[] = {'k', 'm'}, mb[] = {'k', 'n'}, md[] = {'m', 'n'};
        auto dA = api.tensor({c.K, c.M}, {}, c.type), dB = api.tensor({c.K, c.N}, {}, c.type);
        auto dD                             = api.tensor({c.M, c.N}, {}, c.type);
        hiptensorOperationDescriptor_t desc = nullptr;
        ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                             &desc,
                                             dA,
                                             ma,
                                             HIPTENSOR_OP_IDENTITY,
                                             dB,
                                             mb,
                                             HIPTENSOR_OP_IDENTITY,
                                             c.bilinear ? dD : nullptr,
                                             c.bilinear ? md : nullptr,
                                             HIPTENSOR_OP_IDENTITY,
                                             dD,
                                             md,
                                             c.compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        Scalar alpha, beta;
        alpha.re = 2.0;
        beta.re  = c.bilinear ? -1.0 : 0.0;
        if(cx)
        {
            alpha.im = 1.0;
            beta.im  = c.bilinear ? 1.0 : 0.0;
        }
        auto const scalarType = scalarTypeOf(c.compute);
        // sampled outputs (the corners first) and their exact values
        std::mt19937                             rng(17);
        std::uniform_int_distribution<int64_t>   pm(0, c.M - 1), pn(0, c.N - 1);
        std::vector<std::pair<int64_t, int64_t>> samples;
        if(c.M * c.N <= 512)
        {
            for(int64_t m = 0; m < c.M; m++)
            {
                for(int64_t n = 0; n < c.N; n++)
                {
                    samples.emplace_back(m, n);
                }
            }
        }
        else
        {
            samples = {{0, 0}, {c.M - 1, c.N - 1}};
            while(samples.size() < 512)
            {
                samples.emplace_back(pm(rng), pn(rng));
            }
        }
        // exact integer sums (complex: of the component products)
        std::vector<cplx> want;
        for(auto [m, n] : samples)
        {
            int64_t re = 0, im = 0;
            for(int64_t k = 0; k < c.K; k++)
            {
                int64_t const ia = k + c.K * m, ib = k + c.K * n;
                if(cx)
                {
                    int64_t const ar = hugeInt(2 * ia, 1), ai = hugeInt(2 * ia + 1, 1);
                    int64_t const br = hugeInt(2 * ib, 2), bi = hugeInt(2 * ib + 1, 2);
                    re += ar * br - ai * bi;
                    im += ar * bi + ai * br;
                }
                else
                {
                    re += hugeInt(ia, 1) * hugeInt(ib, 2);
                }
            }
            cplx const sum{double(re), double(im)};
            cplx       w
                = alpha.value() * sum + (c.bilinear ? beta.value() * value(m + c.M * n, 3) : 0.0);
            want.push_back(roundOnce(w, cx ? HIPTENSOR_C_32F : c.type));
        }
        int32_t               candidates = 1;
        std::set<std::string> picks;
        for(int32_t rank = 0; rank < candidates; rank++)
        {
            LogCapture log;
            Choice     choice;
            choice.rank      = rank;
            choice.workspace = HIPTENSOR_WORKSPACE_MAX;
            choice.api       = &api;
            uint64_t     ws  = 0;
            SelectionLog sel;
            auto         plan = planWith(api, desc, choice, log, ws, &sel);
            ASSERT_NE(plan, nullptr) << "KERNEL_RANK " << rank;
            candidates = sel.candidates;
            picks.insert(sel.pick);
            void* pW = nullptr;
            if(ws > 0)
            {
                ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess) << ws;
            }
            ASSERT_EQ(hipMemset(pD, 0xFF, hC.size()), hipSuccess);
            Stream stream;
            ASSERT_EQ(hiptensorContract(api.handle,
                                        plan,
                                        alpha.as(scalarType),
                                        pA,
                                        pB,
                                        beta.as(scalarType),
                                        c.bilinear ? pC : nullptr,
                                        pD,
                                        pW,
                                        ws,
                                        stream.s),
                      HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            static_cast<void>(hipFree(pW));
            int64_t bad = 0;
            for(size_t s = 0; s < samples.size(); s++)
            {
                auto [m, n] = samples[s];
                uint8_t raw[16];
                ASSERT_EQ(hipMemcpy(raw,
                                    static_cast<uint8_t*>(pD) + (m + c.M * n) * es,
                                    es,
                                    hipMemcpyDeviceToHost),
                          hipSuccess);
                Tensor one;
                one.type = c.type;
                one.values.resize(1);
                one.fromBytes(std::vector<uint8_t>(raw, raw + es));
                if(!matches(one.values[0], want[s], 0.0, true) && bad++ < 5)
                {
                    ADD_FAILURE() << sel.pick << ": D[" << m << "," << n << "] = " << one.values[0]
                                  << ", want " << want[s];
                }
            }
            EXPECT_EQ(bad, 0) << sel.pick << "\n" << log.all();
        }
        EXPECT_EQ(picks.size(), size_t(candidates));
        for(auto* p : {pA, pB, pC, pD})
        {
            static_cast<void>(hipFree(p));
        }
    }

    class RockeHugeContraction : public ::testing::TestWithParam<HugeContractionCase>
    {
    };

    TEST_P(RockeHugeContraction, SampledOutputsEveryCandidate)
    {
        runHugeContraction(GetParam());
    }

    INSTANTIATE_TEST_SUITE_P(
        Rocke,
        RockeHugeContraction,
        ::testing::Values(
            // A of 2.15 GB: the 64-bit-address h16 body and the f64 route in row ranges
            HugeContractionCase{
                HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, 65536, 256, 16400, true},
            // A of 2.16 GB: the native f32 GEMM in row ranges (and the f64 route)
            HugeContractionCase{
                HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, 33000, 64, 16384, true},
            // the same with HIPTENSOR_ROCKE_F32_RECIPE=x6: the 64-bit-address x6 body
            HugeContractionCase{
                HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, 33000, 64, 16384, true, "x6"},
            // D and C of 2.24 GB: the ring GETT's classes for D past 2^31 bytes (A packed), the
            // direct GETT with 64-bit offsets, the 64-bit-address h16 body
            HugeContractionCase{
                HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, 34000, 33000, 64, true},
            // A of 2.68 GB: the native complex f32 GEMM in row ranges
            HugeContractionCase{
                HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, 20480, 64, 16384, true},
            // the same with HIPTENSOR_ROCKE_F32_RECIPE=x6: the 64-bit-address cf32 body
            HugeContractionCase{
                HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, 20480, 64, 16384, true, "x6"},
            // D and C of more than 2^30 complex elements (8.6 GB): the complex elementwise
            // launches with 64-bit offsets (their 32-bit offsets index components)
            HugeContractionCase{
                HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, 33000, 32600, 64, true},
            // M past 65535 rows of 256-row tiles (A of 2.18 GB, T of more than 2^31 elements):
            // the 64-bit-address h16 body in two row ranges
            HugeContractionCase{
                HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, 17000001, 96, 64, true},
            // A of 2.42 GB: the f64 GEMM in row ranges
            HugeContractionCase{
                HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F, 36864, 64, 8192, false},
            // rows of A and B past 2 GiB (K = 2^28 + 1024): K ranges accumulated in f64, with
            // and without C
            HugeContractionCase{
                HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F, 2, 3, (int64_t(1) << 28) + 1024, true},
            HugeContractionCase{HIPTENSOR_R_64F,
                                HIPTENSOR_COMPUTE_DESC_64F,
                                2,
                                3,
                                (int64_t(1) << 28) + 1024,
                                false}));

    // A mode of more than 2^31 elements (B = -A, f16, 4.3 GB each): pieces of 2^30.
    TEST(RockeHugeTensor, PermutationOfALongMode)
    {
        int64_t const count = (int64_t(1) << 31) + 77;
        if(!hugeMemory(uint64_t(count) * 4 + (uint64_t(4) << 30)))
        {
            GTEST_SKIP() << "not enough free device memory for tensors of 2 GiB";
        }
        ApiObjects                     api;
        LogCapture                     log;
        int32_t const                  mode[] = {'m'};
        auto                           dA     = api.tensor({count}, {}, HIPTENSOR_R_16F);
        hiptensorOperationDescriptor_t desc   = nullptr;
        ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                             &desc,
                                             dA,
                                             mode,
                                             HIPTENSOR_OP_IDENTITY,
                                             dA,
                                             mode,
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t ws   = 0;
        auto     plan = planOf(api, desc, ws, log);
        ASSERT_NE(plan, nullptr);
        auto  a  = hugeFill<_Float16>(count, 5);
        void *pA = nullptr, *pB = nullptr;
        ASSERT_EQ(hipMalloc(&pA, count * 2), hipSuccess);
        ASSERT_EQ(hipMalloc(&pB, count * 2), hipSuccess);
        ASSERT_EQ(hipMemcpy(pA, a.data(), count * 2, hipMemcpyHostToDevice), hipSuccess);
        float const alpha = -1.0f;
        Stream      stream;
        ASSERT_EQ(hiptensorPermute(api.handle, plan, &alpha, pA, pB, stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        std::vector<_Float16> b(count);
        ASSERT_EQ(hipMemcpy(b.data(), pB, count * 2, hipMemcpyDeviceToHost), hipSuccess);
        static_cast<void>(hipFree(pA));
        static_cast<void>(hipFree(pB));
        int64_t bad = 0;
        for(int64_t i = 0; i < count; i++)
        {
            if(double(b[i]) != -double(a[i]) && bad++ < 5)
            {
                ADD_FAILURE() << "B[" << i << "] = " << double(b[i]) << ", want " << -double(a[i]);
            }
        }
        EXPECT_EQ(bad, 0) << log.all();
    }

    // Reductions past 2^31 elements: a sum over more than 2^31 reduced elements (partials of
    // host-iterated pieces), and more than 2^31 outputs (pieces of the kept modes).
    TEST(RockeHugeTensor, Reductions)
    {
        int64_t const R = (int64_t(1) << 31) + 33, O = (int64_t(1) << 31) + 5;
        if(!hugeMemory(uint64_t(O) * 2 * 3 + (uint64_t(8) << 30)))
        {
            GTEST_SKIP() << "not enough free device memory for tensors of 2 GiB";
        }
        ApiObjects api;
        Stream     stream;
        {
            // D[c] = 2 * sum_r A[r, c] - C[c], A f16 [R, 2] (r fastest): +1 and -1 in
            // alternating runs of 64 with a few 3s, so the partial sums of every order the
            // kernels take (64 lanes reading 64 consecutive elements, contiguous ranges) are
            // small integers (exact in f32)
            LogCapture            log;
            std::vector<_Float16> a(R * 2);
            for(int64_t i = 0; i < R * 2; i++)
            {
                a[i] = static_cast<_Float16>(i % 1000003 == 0 ? 3.0 : ((i >> 6) & 1 ? -1.0 : 1.0));
            }
            int32_t const                  ma[] = {'r', 'c'}, mc[] = {'c'};
            auto                           dA   = api.tensor({R, 2}, {}, HIPTENSOR_R_16F);
            auto                           dC   = api.tensor({2}, {}, HIPTENSOR_R_16F);
            hiptensorOperationDescriptor_t desc = nullptr;
            ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                               &desc,
                                               dA,
                                               ma,
                                               HIPTENSOR_OP_IDENTITY,
                                               dC,
                                               mc,
                                               HIPTENSOR_OP_IDENTITY,
                                               dC,
                                               mc,
                                               HIPTENSOR_OP_ADD,
                                               HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(desc);
            uint64_t ws   = 0;
            auto     plan = planOf(api, desc, ws, log);
            ASSERT_NE(plan, nullptr);
            _Float16 c[2] = {static_cast<_Float16>(5.0), static_cast<_Float16>(-7.0)}, d[2];
            void *   pA = nullptr, *pC = nullptr, *pD = nullptr, *pW = nullptr;
            ASSERT_EQ(hipMalloc(&pA, a.size() * 2), hipSuccess);
            ASSERT_EQ(hipMalloc(&pC, 4), hipSuccess);
            ASSERT_EQ(hipMalloc(&pD, 4), hipSuccess);
            if(ws > 0)
            {
                ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
            }
            ASSERT_EQ(hipMemcpy(pA, a.data(), a.size() * 2, hipMemcpyHostToDevice), hipSuccess);
            ASSERT_EQ(hipMemcpy(pC, c, 4, hipMemcpyHostToDevice), hipSuccess);
            float const alpha = 2.0f, beta = -1.0f;
            ASSERT_EQ(
                hiptensorReduce(api.handle, plan, &alpha, pA, &beta, pC, pD, pW, ws, stream.s),
                HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            ASSERT_EQ(hipMemcpy(d, pD, 4, hipMemcpyDeviceToHost), hipSuccess);
            for(auto* p : {pA, pC, pD, pW})
            {
                static_cast<void>(hipFree(p));
            }
            for(int64_t col = 0; col < 2; col++)
            {
                double sum = 0;
                for(int64_t r = 0; r < R; r++)
                {
                    sum += double(a[r + R * col]);
                }
                double const want = roundOnce(2.0 * sum - double(c[col]), HIPTENSOR_R_16F);
                EXPECT_EQ(double(d[col]), want) << "column " << col << "\n" << log.all();
            }
        }
        {
            // D[o] = max(A[o, 0], A[o, 1]), more than 2^31 outputs (f16)
            LogCapture                     log;
            auto                           a    = hugeFill<_Float16>(O * 2, 9);
            int32_t const                  ma[] = {'o', 'c'}, md[] = {'o'};
            auto                           dA   = api.tensor({O, 2}, {}, HIPTENSOR_R_16F);
            auto                           dD   = api.tensor({O}, {}, HIPTENSOR_R_16F);
            hiptensorOperationDescriptor_t desc = nullptr;
            ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                               &desc,
                                               dA,
                                               ma,
                                               HIPTENSOR_OP_IDENTITY,
                                               dD,
                                               md,
                                               HIPTENSOR_OP_IDENTITY,
                                               dD,
                                               md,
                                               HIPTENSOR_OP_MAX,
                                               HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(desc);
            uint64_t ws   = 0;
            auto     plan = planOf(api, desc, ws, log);
            ASSERT_NE(plan, nullptr);
            void *pA = nullptr, *pD = nullptr, *pW = nullptr;
            ASSERT_EQ(hipMalloc(&pA, a.size() * 2), hipSuccess);
            ASSERT_EQ(hipMalloc(&pD, O * 2), hipSuccess);
            if(ws > 0)
            {
                ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
            }
            ASSERT_EQ(hipMemcpy(pA, a.data(), a.size() * 2, hipMemcpyHostToDevice), hipSuccess);
            float const alpha = 1.0f, beta = 0.0f;
            ASSERT_EQ(
                hiptensorReduce(api.handle, plan, &alpha, pA, &beta, nullptr, pD, pW, ws, stream.s),
                HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            std::vector<_Float16> d(O);
            ASSERT_EQ(hipMemcpy(d.data(), pD, O * 2, hipMemcpyDeviceToHost), hipSuccess);
            for(auto* p : {pA, pD, pW})
            {
                static_cast<void>(hipFree(p));
            }
            int64_t bad = 0;
            for(int64_t o = 0; o < O; o++)
            {
                double const want = std::max(double(a[o]), double(a[o + O]));
                if(double(d[o]) != want && bad++ < 5)
                {
                    ADD_FAILURE() << "D[" << o << "] = " << double(d[o]) << ", want " << want;
                }
            }
            EXPECT_EQ(bad, 0) << log.all();
        }
    }

    // A reduction with more kept and more reduced modes than the kernels' 8 slots (9 of each,
    // interleaved so that none merge): pieces of the kept modes are separate reductions, pieces
    // of the reduced modes write partials that the final kernel combines. Exact (integer data).
    TEST(RockeReductionPieces, MoreThanEightKeptAndReducedModes)
    {
        ApiObjects                 api;
        LogCapture                 log;
        int const                  pairs = 9;
        std::vector<int32_t>       modesA, modesD;
        std::vector<int64_t>       extA, strideA, extD, strideD;
        int64_t                    sa = 1, sd = 1;
        std::map<int32_t, int64_t> ext;
        for(int i = 0; i < pairs; i++)
        {
            int32_t const k = 'a' + i, r = 'n' + i;
            int64_t const ek = i == 4 ? 3 : 2, er = i == 6 ? 3 : 2;
            ext[k] = ek;
            ext[r] = er;
            modesA.insert(modesA.end(), {k, r});
            extA.insert(extA.end(), {ek, er});
            strideA.insert(strideA.end(), {sa, sa * ek});
            sa *= ek * er;
            modesD.push_back(k);
            extD.push_back(ek);
            strideD.push_back(sd);
            sd *= ek;
        }
        std::vector<float> a(sa), c(sd);
        for(int64_t i = 0; i < sa; i++)
        {
            a[i] = float(int64_t(hugeInt(i, 21)));
        }
        for(int64_t i = 0; i < sd; i++)
        {
            c[i] = float(i % 5) - 2.0f;
        }
        auto                           dA   = api.tensor(extA, strideA, HIPTENSOR_R_32F);
        auto                           dD   = api.tensor(extD, strideD, HIPTENSOR_R_32F);
        hiptensorOperationDescriptor_t desc = nullptr;
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &desc,
                                           dA,
                                           modesA.data(),
                                           HIPTENSOR_OP_IDENTITY,
                                           dD,
                                           modesD.data(),
                                           HIPTENSOR_OP_IDENTITY,
                                           dD,
                                           modesD.data(),
                                           HIPTENSOR_OP_ADD,
                                           HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        void *pA = nullptr, *pC = nullptr, *pD = nullptr;
        ASSERT_EQ(hipMalloc(&pA, a.size() * 4), hipSuccess);
        ASSERT_EQ(hipMalloc(&pC, c.size() * 4), hipSuccess);
        ASSERT_EQ(hipMalloc(&pD, c.size() * 4), hipSuccess);
        ASSERT_EQ(hipMemcpy(pA, a.data(), a.size() * 4, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(pC, c.data(), c.size() * 4, hipMemcpyHostToDevice), hipSuccess);
        // reference: D[k] = 2 * sum_r A[k, r] - C[k]
        std::vector<double> want(sd, 0.0);
        for(int64_t i = 0; i < sa; i++)
        {
            int64_t rest = i, d = 0;
            for(int j = 0; j < pairs; j++)
            {
                int64_t const ik = rest % extA[2 * j];
                rest /= extA[2 * j];
                rest /= extA[2 * j + 1];
                d += ik * strideD[j];
            }
            want[d] += 2.0 * a[i];
        }
        for(int64_t i = 0; i < sd; i++)
        {
            want[i] -= c[i];
        }
        int32_t candidates = 1;
        for(int32_t rank = 0; rank < candidates; rank++)
        {
            Choice choice;
            choice.rank      = rank;
            choice.workspace = HIPTENSOR_WORKSPACE_MAX;
            choice.api       = &api;
            uint64_t     ws  = 0;
            SelectionLog sel;
            auto         plan = planWith(api, desc, choice, log, ws, &sel);
            ASSERT_NE(plan, nullptr);
            candidates = sel.candidates;
            void* pW   = nullptr;
            if(ws > 0)
            {
                ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
            }
            ASSERT_EQ(hipMemset(pD, 0xFF, c.size() * 4), hipSuccess);
            float const alpha = 2.0f, beta = -1.0f;
            ASSERT_EQ(hiptensorReduce(api.handle, plan, &alpha, pA, &beta, pC, pD, pW, ws, nullptr),
                      HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            static_cast<void>(hipFree(pW));
            std::vector<float> d(sd);
            ASSERT_EQ(hipMemcpy(d.data(), pD, d.size() * 4, hipMemcpyDeviceToHost), hipSuccess);
            int64_t bad = 0;
            for(int64_t i = 0; i < sd; i++)
            {
                if(double(d[i]) != want[i] && bad++ < 5)
                {
                    ADD_FAILURE() << sel.pick << ": D[" << i << "] = " << d[i] << ", want "
                                  << want[i];
                }
            }
            EXPECT_EQ(bad, 0) << log.all();
        }
        for(auto* p : {pA, pC, pD})
        {
            static_cast<void>(hipFree(p));
        }
    }

    // ---------------------------------------------------------------------- unary operators

    // Every hipTensor unary operator through the rocKE kernels: permutations of every data type,
    // elementwise binary and trinary operations with an operator per input, reductions with opA
    // and opC, contractions with opA, opB and opC (each route), trinary contractions with opA
    // to opD, and the special values. References in long double; the bound of each operator is
    // its documented maximum error in ulps of the compute type (programmer's guide).

    hiptensorOperator_t const kUnaryOps[]
        = {HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_SQRT,    HIPTENSOR_OP_RELU, HIPTENSOR_OP_CONJ,
           HIPTENSOR_OP_RCP,      HIPTENSOR_OP_SIGMOID, HIPTENSOR_OP_TANH, HIPTENSOR_OP_EXP,
           HIPTENSOR_OP_LOG,      HIPTENSOR_OP_ABS,     HIPTENSOR_OP_NEG,  HIPTENSOR_OP_SIN,
           HIPTENSOR_OP_COS,      HIPTENSOR_OP_TAN,     HIPTENSOR_OP_SINH, HIPTENSOR_OP_COSH,
           HIPTENSOR_OP_ASIN,     HIPTENSOR_OP_ACOS,    HIPTENSOR_OP_ATAN, HIPTENSOR_OP_ASINH,
           HIPTENSOR_OP_ACOSH,    HIPTENSOR_OP_ATANH,   HIPTENSOR_OP_CEIL, HIPTENSOR_OP_FLOOR};

    long double unaryL(hiptensorOperator_t op, long double x)
    {
        switch(op)
        {
        case HIPTENSOR_OP_SQRT:
            return std::sqrt(x);
        case HIPTENSOR_OP_RELU:
            return x > 0 ? x : 0.0L;
        case HIPTENSOR_OP_RCP:
            return 1.0L / x;
        case HIPTENSOR_OP_SIGMOID:
            return 1.0L / (1.0L + std::exp(-x));
        case HIPTENSOR_OP_TANH:
            return std::tanh(x);
        case HIPTENSOR_OP_EXP:
            return std::exp(x);
        case HIPTENSOR_OP_LOG:
            return std::log(x);
        case HIPTENSOR_OP_ABS:
            return std::fabs(x);
        case HIPTENSOR_OP_NEG:
            return -x;
        case HIPTENSOR_OP_SIN:
            return std::sin(x);
        case HIPTENSOR_OP_COS:
            return std::cos(x);
        case HIPTENSOR_OP_TAN:
            return std::tan(x);
        case HIPTENSOR_OP_SINH:
            return std::sinh(x);
        case HIPTENSOR_OP_COSH:
            return std::cosh(x);
        case HIPTENSOR_OP_ASIN:
            return std::asin(x);
        case HIPTENSOR_OP_ACOS:
            return std::acos(x);
        case HIPTENSOR_OP_ATAN:
            return std::atan(x);
        case HIPTENSOR_OP_ASINH:
            return std::asinh(x);
        case HIPTENSOR_OP_ACOSH:
            return std::acosh(x);
        case HIPTENSOR_OP_ATANH:
            return std::atanh(x);
        case HIPTENSOR_OP_CEIL:
            return std::ceil(x);
        case HIPTENSOR_OP_FLOOR:
            return std::floor(x);
        default: // IDENTITY, CONJ of a real value
            return x;
        }
    }

    // The documented maximum error of an operator in ulps of the compute type (f32 or f64),
    // against the exact result of the compute-type operand.
    double opUlps(hiptensorOperator_t op, bool f64)
    {
        switch(op)
        {
        case HIPTENSOR_OP_IDENTITY:
        case HIPTENSOR_OP_CONJ:
        case HIPTENSOR_OP_RELU:
        case HIPTENSOR_OP_ABS:
        case HIPTENSOR_OP_NEG:
        case HIPTENSOR_OP_CEIL:
        case HIPTENSOR_OP_FLOOR:
            return 0.0;
        case HIPTENSOR_OP_SQRT:
        case HIPTENSOR_OP_RCP:
            return 0.5;
        case HIPTENSOR_OP_EXP:
            return 1.0;
        case HIPTENSOR_OP_LOG:
            return f64 ? 1.0 : 2.25;
        case HIPTENSOR_OP_TANH:
            return f64 ? 1.0 : 2.0;
        case HIPTENSOR_OP_SIGMOID:
            return f64 ? 2.0 : 3.0;
        default: // the math operators: f64 programs (f32: rounded once from f64)
            return f64 ? 0.65 : 0.501;
        }
    }

    // ulp of |v| in a data type (subnormal spacing below the normal range).
    double ulpIn(double v, hiptensorDataType_t t)
    {
        int bits = 52, emin = -1022;
        switch(t)
        {
        case HIPTENSOR_R_16F:
            bits = 10, emin = -14;
            break;
        case HIPTENSOR_R_16BF:
            bits = 7, emin = -126;
            break;
        case HIPTENSOR_R_32F:
        case HIPTENSOR_C_32F:
            bits = 23, emin = -126;
            break;
        default:
            break;
        }
        int e = 0;
        std::frexp(std::fabs(v), &e); // |v| = m 2^e, m in [0.5, 1)
        return std::ldexp(1.0, std::max(e - 1, emin) - bits);
    }

    // A value of the operator's domain, rounded to `type` (where the result is finite).
    double domainValue(hiptensorOperator_t op, std::mt19937& rng, hiptensorDataType_t type)
    {
        auto u = [&](double lo, double hi) {
            return std::uniform_real_distribution<double>(lo, hi)(rng);
        };
        double sign = rng() % 2 ? 1.0 : -1.0;
        double v    = 0;
        switch(op)
        {
        case HIPTENSOR_OP_SQRT:
        case HIPTENSOR_OP_LOG:
            v = std::ldexp(u(1.0, 2.0), int(rng() % 24) - 12);
            break;
        case HIPTENSOR_OP_RCP:
            v = sign * std::ldexp(u(1.0, 2.0), int(rng() % 16) - 8);
            break;
        case HIPTENSOR_OP_ASIN:
        case HIPTENSOR_OP_ACOS:
        case HIPTENSOR_OP_ATANH:
            v = u(-0.97, 0.97);
            break;
        case HIPTENSOR_OP_ACOSH:
            v = 1.0 + std::ldexp(u(1.0, 2.0), int(rng() % 16) - 10);
            break;
        case HIPTENSOR_OP_EXP:
        case HIPTENSOR_OP_SINH:
        case HIPTENSOR_OP_COSH:
            v = u(-10.0, 10.0);
            break;
        case HIPTENSOR_OP_SIN:
        case HIPTENSOR_OP_COS:
            v = rng() % 4 ? u(-8.0, 8.0) : u(-3000.0, 3000.0);
            break;
        case HIPTENSOR_OP_TAN:
            v = u(-1.5, 1.5);
            break;
        case HIPTENSOR_OP_CEIL:
        case HIPTENSOR_OP_FLOOR:
            v = rng() % 4 ? u(-20.0, 20.0) : double(int(rng() % 21) - 10) + 0.5;
            break;
        default:
            v = u(-4.0, 4.0);
            break;
        }
        return roundTo(v, type);
    }

    // A tensor of `modes` (random storage order and padding) with values of op's domain.
    Tensor domainTensor(hiptensorDataType_t               type,
                        std::vector<int32_t> const&       modes,
                        std::map<int32_t, int64_t> const& extents,
                        std::mt19937&                     rng,
                        hiptensorOperator_t               op)
    {
        auto t = makeTensor(type, modes, extents, rng, true);
        for(auto& v : t.values)
        {
            v = domainValue(op, rng, type);
        }
        return t;
    }

    bool isF64Compute(hiptensorComputeDescriptor_t c)
    {
        return c == HIPTENSOR_COMPUTE_DESC_64F || c == HIPTENSOR_COMPUTE_DESC_C64F;
    }

    // A value and the bound of its error (each term of the references below).
    // A reference value (long double: the exact result to far below an f64 ulp) and the bound of
    // the kernel's error against it (each term of the references below).
    struct Approx
    {
        long double v = 0;
        double      e = 0;
    };

    // scale * op(x) in the compute type: the operator's error and the product's rounding.
    Approx unaryTerm(hiptensorOperator_t op, double x, double scale, bool f64)
    {
        hiptensorDataType_t const ct = f64 ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
        long double const         y  = unaryL(op, x);
        Approx                    a;
        a.v = scale * y;
        a.e = std::fabs(scale) * opUlps(op, f64) * ulpIn(double(y), ct)
              + 0.5 * ulpIn(double(a.v), ct);
        return a;
    }

    Approx binaryTerm(hiptensorOperator_t op, Approx p, Approx q, bool f64)
    {
        hiptensorDataType_t const ct = f64 ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
        Approx                    r;
        switch(op)
        {
        case HIPTENSOR_OP_MUL:
            r.v = p.v * q.v;
            r.e = double(std::fabs(p.v)) * q.e + double(std::fabs(q.v)) * p.e + p.e * q.e;
            break;
        case HIPTENSOR_OP_MAX:
            r.v = std::max(p.v, q.v);
            r.e = std::max(p.e, q.e);
            return r;
        case HIPTENSOR_OP_MIN:
            r.v = std::min(p.v, q.v);
            r.e = std::max(p.e, q.e);
            return r;
        default:
            r.v = p.v + q.v;
            r.e = p.e + q.e;
            break;
        }
        r.e += 0.5 * ulpIn(double(r.v), ct);
        return r;
    }

    double maxOf(hiptensorDataType_t t)
    {
        switch(t)
        {
        case HIPTENSOR_R_16F:
            return 65504.0;
        case HIPTENSOR_R_16BF:
            return 3.3895313892515355e38;
        case HIPTENSOR_R_32F:
            return std::numeric_limits<float>::max();
        default:
            return std::numeric_limits<double>::max();
        }
    }

    // |got - want| within the bound plus one rounding to the output type (exact NaN /
    // infinity agreement where the reference is not finite, an infinity of the reference's sign
    // where it rounds past the output type's range).
    bool withinBound(double got, Approx want, hiptensorDataType_t out, bool narrower)
    {
        if(std::isnan(want.v) || std::isinf(want.v))
        {
            return std::isnan(want.v) ? std::isnan(got) : got == double(want.v);
        }
        double const tol = want.e + (narrower ? ulpIn(double(want.v), out) : 0.0);
        // the largest finite value plus half its ulp: values past it round to infinity
        double const edge = maxOf(out) + 0.5 * ulpIn(maxOf(out), out);
        if(std::fabs(want.v) - tol > edge)
        {
            return std::isinf(got) && (got > 0) == (want.v > 0);
        }
        if(std::isinf(got) && std::fabs(want.v) + tol >= edge)
        {
            return (got > 0) == (want.v > 0);
        }
        return std::fabs(got - want.v) <= tol;
    }

    struct UnaryCase
    {
        hiptensorOperator_t          op;
        hiptensorDataType_t          in, out;
        hiptensorComputeDescriptor_t compute;
        int                          seed;
    };

    std::string unaryName(::testing::TestParamInfo<UnaryCase> const& info)
    {
        auto t = [](hiptensorDataType_t x) {
            return x == HIPTENSOR_R_16F    ? std::string("f16")
                   : x == HIPTENSOR_R_16BF ? std::string("bf16")
                   : x == HIPTENSOR_R_32F  ? std::string("f32")
                                           : std::string("f64");
        };
        return "op" + std::to_string(int(info.param.op)) + "_" + t(info.param.in) + "_"
               + t(info.param.out) + "_c" + std::to_string(int(info.param.compute));
    }

    // D = opA(A) as a permutation (random mode orders and padding).
    class RockeUnaryPermutation : public ::testing::TestWithParam<UnaryCase>
    {
    };

    TEST_P(RockeUnaryPermutation, WithinTheOperatorsBound)
    {
        auto const   c = GetParam();
        std::mt19937 rng(c.seed);
        ApiObjects   api;
        if(!api.supports(c.in) || !api.supports(c.out))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture                 log;
        std::vector<int32_t>       modes = {'a', 'b', 'c'};
        std::map<int32_t, int64_t> ext   = {{'a', 37}, {'b', 5}, {'c', 9}};
        auto                       A     = domainTensor(c.in, modes, ext, rng, c.op);
        auto                       mD    = modes;
        std::shuffle(mD.begin(), mD.end(), rng);
        auto D = makeTensor(c.out, mD, ext, rng, true);
        std::fill(D.values.begin(), D.values.end(), cplx(0, 0));
        hiptensorOperationDescriptor_t desc = nullptr;
        auto                           dA = describe(api, A), dD = describe(api, D);
        ASSERT_EQ(hiptensorCreatePermutation(
                      api.handle, &desc, dA, A.modes.data(), c.op, dD, D.modes.data(), c.compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t ws   = 0;
        auto     plan = planOf(api, desc, ws, log);
        ASSERT_NE(plan, nullptr);
        Device dev;
        void*  pA = dev.upload(A);
        void*  pD = dev.upload(D);
        Stream stream;
        Scalar one;
        one.re = 1.0;
        ASSERT_EQ(
            hiptensorPermute(api.handle, plan, one.as(scalarTypeOf(c.compute)), pA, pD, stream.s),
            HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);
        bool const f64      = isF64Compute(c.compute);
        bool const narrower = sizeOf(c.out) < (f64 ? 8u : 4u);
        int64_t    bad      = 0;
        forEachIndex(modes, ext, [&](auto index) {
            double x    = A.values[A.offset(index)].real();
            auto   want = unaryTerm(c.op, x, 1.0, f64);
            double g    = got.values[D.offset(index)].real();
            if(!withinBound(g, want, c.out, narrower) && bad++ < 5)
            {
                ADD_FAILURE() << "op " << int(c.op) << " x " << x << " got " << g << " want "
                              << want.v << " (bound " << want.e << ")";
            }
        });
        EXPECT_EQ(bad, 0);
    }

    std::vector<UnaryCase> unaryCases()
    {
        std::vector<UnaryCase> out;
        struct Row
        {
            hiptensorDataType_t          in, out;
            hiptensorComputeDescriptor_t compute;
        };
        // the API's permutation rows (bf16 data: the elementwise suites below)
        Row const rows[] = {{HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F},
                            {HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_16F},
                            {HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F},
                            {HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F},
                            {HIPTENSOR_R_16F, HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F}};
        int       seed   = 4100;
        for(auto op : kUnaryOps)
        {
            for(auto const& r : rows)
            {
                out.push_back({op, r.in, r.out, r.compute, seed++});
            }
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeUnaryPermutation,
                             ::testing::ValuesIn(unaryCases()),
                             unaryName);

    // The special values of every operator, exact: NaN, infinities, signed zeros, subnormals,
    // arguments outside the domain (C semantics; RELU(x) = x > 0 ? x : 0).
    TEST(RockeUnary, SpecialValues)
    {
        ApiObjects api;
        for(auto type : {HIPTENSOR_R_32F, HIPTENSOR_R_64F})
        {
            double const        sub = type == HIPTENSOR_R_32F ? std::ldexp(1.0, -149)
                                                              : std::numeric_limits<double>::denorm_min();
            std::vector<double> xs
                = {0.0, -0.0, INFINITY, -INFINITY, NAN, sub, -sub, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5};
            for(auto op : kUnaryOps)
            {
                LogCapture log;
                Tensor     A;
                A.type    = type;
                A.modes   = {'i'};
                A.extents = {int64_t(xs.size())};
                A.strides = {1};
                for(double x : xs)
                {
                    A.values.push_back(x);
                }
                Tensor D = A;
                std::fill(D.values.begin(), D.values.end(), cplx(7, 0));
                auto compute = type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F
                                                       : HIPTENSOR_COMPUTE_DESC_32F;
                hiptensorOperationDescriptor_t desc = nullptr;
                auto                           dA   = describe(api, A);
                ASSERT_EQ(
                    hiptensorCreatePermutation(
                        api.handle, &desc, dA, A.modes.data(), op, dA, D.modes.data(), compute),
                    HIPTENSOR_STATUS_SUCCESS);
                api.keep(desc);
                uint64_t ws   = 0;
                auto     plan = planOf(api, desc, ws, log);
                ASSERT_NE(plan, nullptr);
                Device dev;
                void*  pA = dev.upload(A);
                void*  pD = dev.upload(D);
                Scalar one;
                one.re = 1.0;
                ASSERT_EQ(hiptensorPermute(api.handle, plan, one.as(type), pA, pD, nullptr),
                          HIPTENSOR_STATUS_SUCCESS);
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                Tensor got = D;
                dev.download(pD, got);
                bool const f64 = type == HIPTENSOR_R_64F;
                for(size_t i = 0; i < xs.size(); i++)
                {
                    double const w = roundTo(double(unaryL(op, xs[i])), type);
                    double const g = got.values[i].real();
                    if(std::isnan(w) || std::isinf(w) || w == 0)
                    {
                        // exact, the sign of a zero included
                        bool const same = std::isnan(w)
                                              ? std::isnan(g)
                                              : (g == w && std::signbit(g) == std::signbit(w));
                        EXPECT_TRUE(same) << "op " << int(op) << " type " << int(type) << " x "
                                          << xs[i] << " got " << g << " want " << w;
                    }
                    else
                    {
                        EXPECT_TRUE(withinBound(g, unaryTerm(op, xs[i], 1.0, f64), type, false))
                            << "op " << int(op) << " type " << int(type) << " x " << xs[i]
                            << " got " << g << " want " << w;
                    }
                }
            }
        }
    }

    // Elementwise binary and trinary operations with an operator per input (any of them).
    struct UnaryElementwiseCase
    {
        int                 arity;
        hiptensorDataType_t type;
        int                 seed;
    };

    class RockeUnaryElementwise : public ::testing::TestWithParam<UnaryElementwiseCase>
    {
    };

    TEST_P(RockeUnaryElementwise, OperatorPerInput)
    {
        auto const   c = GetParam();
        std::mt19937 rng(c.seed);
        ApiObjects   api;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture                 log;
        std::vector<int32_t>       modes    = {'a', 'b', 'c'};
        std::map<int32_t, int64_t> ext      = {{'a', 33}, {'b', 6}, {'c', 7}};
        auto                       shuffled = [&] {
            auto m = modes;
            std::shuffle(m.begin(), m.end(), rng);
            return m;
        };
        std::vector<hiptensorOperator_t> uop;
        std::vector<Tensor>              ins;
        for(int i = 0; i < c.arity; i++)
        {
            uop.push_back(kUnaryOps[rng() % std::size(kUnaryOps)]);
            ins.push_back(domainTensor(c.type, shuffled(), ext, rng, uop.back()));
        }
        auto D = makeTensor(c.type, shuffled(), ext, rng, true);
        std::fill(D.values.begin(), D.values.end(), cplx(0, 0));
        hiptensorOperator_t const bops[]
            = {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL, HIPTENSOR_OP_MAX, HIPTENSOR_OP_MIN};
        auto opAB    = bops[rng() % 4];
        auto opABC   = bops[rng() % 4];
        bool f64     = c.type == HIPTENSOR_R_64F;
        auto compute = f64 ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
        auto st      = f64 ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
        if(c.arity == 3 && c.type == HIPTENSOR_R_16F)
        {
            // the API's trinary row of f16 data takes f16 scalars (the kernels compute in f32)
            compute = HIPTENSOR_COMPUTE_DESC_16F;
            st      = HIPTENSOR_R_16F;
        }
        std::vector<hiptensorTensorDescriptor_t> d;
        for(auto const& t : ins)
        {
            d.push_back(describe(api, t));
        }
        auto                           dD   = describe(api, D);
        hiptensorOperationDescriptor_t desc = nullptr;
        if(c.arity == 2)
        {
            ASSERT_EQ(hiptensorCreateElementwiseBinary(api.handle,
                                                       &desc,
                                                       d[0],
                                                       ins[0].modes.data(),
                                                       uop[0],
                                                       d[1],
                                                       ins[1].modes.data(),
                                                       uop[1],
                                                       dD,
                                                       D.modes.data(),
                                                       opAB,
                                                       compute),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        else
        {
            ASSERT_EQ(hiptensorCreateElementwiseTrinary(api.handle,
                                                        &desc,
                                                        d[0],
                                                        ins[0].modes.data(),
                                                        uop[0],
                                                        d[1],
                                                        ins[1].modes.data(),
                                                        uop[1],
                                                        d[2],
                                                        ins[2].modes.data(),
                                                        uop[2],
                                                        dD,
                                                        D.modes.data(),
                                                        opAB,
                                                        opABC,
                                                        compute),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        api.keep(desc);
        uint64_t ws   = 0;
        auto     plan = planOf(api, desc, ws, log);
        ASSERT_NE(plan, nullptr);
        Device             dev;
        std::vector<void*> p;
        for(auto const& t : ins)
        {
            p.push_back(dev.upload(t));
        }
        void*  pD = dev.upload(D);
        Scalar s[3];
        s[0].re = 1.5;
        s[1].re = -0.5;
        s[2].re = 2.0;
        Stream stream;
        if(c.arity == 2)
        {
            ASSERT_EQ(hiptensorElementwiseBinaryExecute(
                          api.handle, plan, s[0].as(st), p[0], s[1].as(st), p[1], pD, stream.s),
                      HIPTENSOR_STATUS_SUCCESS)
                << log.all();
        }
        else
        {
            Scalar a = s[0], b = s[1], g = s[2];
            ASSERT_EQ(
                hiptensorElementwiseTrinaryExecute(
                    api.handle, plan, a.as(st), p[0], b.as(st), p[1], g.as(st), p[2], pD, stream.s),
                HIPTENSOR_STATUS_SUCCESS)
                << log.all();
        }
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);
        bool const narrower = sizeOf(c.type) < (f64 ? 8u : 4u);
        int64_t    bad      = 0;
        forEachIndex(modes, ext, [&](auto index) {
            std::vector<Approx> t;
            for(int i = 0; i < c.arity; i++)
            {
                t.push_back(
                    unaryTerm(uop[i], ins[i].values[ins[i].offset(index)].real(), s[i].re, f64));
            }
            Approx r = binaryTerm(opAB, t[0], t[1], f64);
            if(c.arity == 3)
            {
                r = binaryTerm(opABC, r, t[2], f64);
            }
            double g = got.values[D.offset(index)].real();
            if(!withinBound(g, r, c.type, narrower) && bad++ < 5)
            {
                ADD_FAILURE() << "ops " << int(uop[0]) << "," << int(uop[1]) << " got " << g
                              << " want " << r.v << " (bound " << r.e << ")";
            }
        });
        EXPECT_EQ(bad, 0);
    }

    std::vector<UnaryElementwiseCase> unaryElementwiseCases()
    {
        std::vector<UnaryElementwiseCase> out;
        int                               seed = 4400;
        for(int arity : {2, 3})
        {
            for(auto t : {HIPTENSOR_R_16F, HIPTENSOR_R_16BF, HIPTENSOR_R_32F, HIPTENSOR_R_64F})
            {
                if(arity == 3 && t == HIPTENSOR_R_16BF)
                {
                    continue; // not an API row
                }
                for(int i = 0; i < 4; i++)
                {
                    out.push_back({arity, t, seed++});
                }
            }
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeUnaryElementwise,
                             ::testing::ValuesIn(unaryElementwiseCases()));

    // Reductions with opA and opC: sum and max of opA(A), D = alpha r + beta opC(C).
    struct UnaryReductionCase
    {
        hiptensorOperator_t opA, opC, opR;
        hiptensorDataType_t type;
        int                 seed;
    };

    class RockeUnaryReduction : public ::testing::TestWithParam<UnaryReductionCase>
    {
    };

    TEST_P(RockeUnaryReduction, WithinTheBound)
    {
        auto const   c = GetParam();
        std::mt19937 rng(c.seed);
        ApiObjects   api;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        LogCapture                 log;
        std::map<int32_t, int64_t> ext = {{'a', 70}, {'b', 9}, {'c', 5}};
        std::vector<int32_t>       ma  = {'a', 'b', 'c'};
        std::shuffle(ma.begin(), ma.end(), rng);
        std::vector<int32_t> md  = {'b'};
        std::vector<int32_t> red = {'a', 'c'};
        auto                 A   = domainTensor(c.type, ma, ext, rng, c.opA);
        auto                 C   = domainTensor(c.type, md, ext, rng, c.opC);
        auto                 D   = C;
        std::fill(D.values.begin(), D.values.end(), cplx(0, 0));
        bool const f64     = c.type == HIPTENSOR_R_64F;
        auto       compute = f64 ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
        auto       st      = f64 ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dC = describe(api, C), dD = describe(api, D);
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &desc,
                                           dA,
                                           A.modes.data(),
                                           c.opA,
                                           dC,
                                           C.modes.data(),
                                           c.opC,
                                           dD,
                                           D.modes.data(),
                                           c.opR,
                                           compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        uint64_t ws   = 0;
        auto     plan = planOf(api, desc, ws, log);
        ASSERT_NE(plan, nullptr);
        Device dev;
        void*  pA = dev.upload(A);
        void*  pC = dev.upload(C);
        void*  pD = dev.upload(D);
        void*  pW = nullptr;
        if(ws > 0)
        {
            ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
            dev.buffers.push_back(pW);
        }
        Scalar alpha, beta;
        alpha.re = 0.75;
        beta.re  = -1.25;
        Stream stream;
        ASSERT_EQ(hiptensorReduce(
                      api.handle, plan, alpha.as(st), pA, beta.as(st), pC, pD, pW, ws, stream.s),
                  HIPTENSOR_STATUS_SUCCESS)
            << log.all();
        ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        dev.download(pD, got);
        hiptensorDataType_t const ct       = f64 ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
        bool const                narrower = sizeOf(c.type) < (f64 ? 8u : 4u);
        int64_t                   bad      = 0;
        forEachIndex(md, ext, [&](auto index) {
            Approx r;
            r.v       = c.opR == HIPTENSOR_OP_MAX ? -INFINITY : 0.0;
            double ab = 0;
            int    n  = 0;
            forEachIndex(red, ext, [&](auto rindex) {
                auto full = index;
                for(auto& kv : rindex)
                {
                    full[kv.first] = kv.second;
                }
                auto t = unaryTerm(c.opA, A.values[A.offset(full)].real(), 1.0, f64);
                if(c.opR == HIPTENSOR_OP_MAX)
                {
                    r.v = std::max(r.v, t.v);
                    r.e = std::max(r.e, t.e);
                }
                else
                {
                    r.v += t.v;
                    r.e += t.e;
                    ab += double(std::fabs(t.v));
                    n++;
                }
            });
            // the summation (any order, split partials) in the compute type
            r.e += 2.0 * n * (f64 ? std::ldexp(1.0, -53) : std::ldexp(1.0, -24)) * ab;
            Approx a;
            a.v          = alpha.re * r.v;
            a.e          = std::fabs(alpha.re) * r.e + 0.5 * ulpIn(double(a.v), ct);
            auto   cterm = unaryTerm(c.opC, C.values[C.offset(index)].real(), beta.re, f64);
            Approx w     = binaryTerm(HIPTENSOR_OP_ADD, a, cterm, f64);
            double g     = got.values[D.offset(index)].real();
            if(!withinBound(g, w, c.type, narrower) && bad++ < 5)
            {
                ADD_FAILURE() << "got " << g << " want " << double(w.v) << " (bound " << w.e << ")";
            }
        });
        EXPECT_EQ(bad, 0) << log.all();
    }

    std::vector<UnaryReductionCase> unaryReductionCases()
    {
        std::vector<UnaryReductionCase> out;
        int                             seed      = 4600;
        hiptensorOperator_t const       mathOps[] = {HIPTENSOR_OP_SIN,
                                                     HIPTENSOR_OP_COS,
                                                     HIPTENSOR_OP_TAN,
                                                     HIPTENSOR_OP_SINH,
                                                     HIPTENSOR_OP_COSH,
                                                     HIPTENSOR_OP_ASIN,
                                                     HIPTENSOR_OP_ACOS,
                                                     HIPTENSOR_OP_ATAN,
                                                     HIPTENSOR_OP_ASINH,
                                                     HIPTENSOR_OP_ACOSH,
                                                     HIPTENSOR_OP_ATANH,
                                                     HIPTENSOR_OP_CEIL,
                                                     HIPTENSOR_OP_FLOOR};
        hiptensorDataType_t const       types[]
            = {HIPTENSOR_R_16F, HIPTENSOR_R_16BF, HIPTENSOR_R_32F, HIPTENSOR_R_64F};
        int i = 0;
        for(auto op : mathOps)
        {
            auto t = types[i++ % 4];
            out.push_back({op, HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_ADD, t, seed++});
            out.push_back({op, mathOps[(i + 5) % 13], HIPTENSOR_OP_MAX, t, seed++});
            out.push_back({HIPTENSOR_OP_SQRT, op, HIPTENSOR_OP_ADD, types[i % 4], seed++});
        }
        return out;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeUnaryReduction,
                             ::testing::ValuesIn(unaryReductionCases()));

    // Contractions with opA, opB and opC: each route of the data type (the GEMM routes on
    // gfx950, the f64 route), with every candidate of the selection.
    struct UnaryContractionCase
    {
        hiptensorDataType_t          type;
        hiptensorComputeDescriptor_t compute;
        hiptensorOperator_t          opA, opB, opC;
        int                          seed;
    };

    class RockeUnaryContraction : public ::testing::TestWithParam<UnaryContractionCase>
    {
    };

    TEST_P(RockeUnaryContraction, EveryCandidateWithinTheBound)
    {
        auto const c = GetParam();
        ApiObjects api;
        if(!api.supports(c.type))
        {
            GTEST_SKIP() << "data type not supported on this device";
        }
        if((c.compute == HIPTENSOR_COMPUTE_DESC_16F || c.compute == HIPTENSOR_COMPUTE_DESC_16BF)
           && f64RouteOnly())
        {
            GTEST_SKIP() << "the f64 route declines 16-bit compute types of other operand types";
        }
        std::mt19937               rng(c.seed);
        std::map<int32_t, int64_t> ext = {{'m', 21}, {'n', 17}, {'k', 40}, {'q', 3}};
        std::vector<int32_t>       ma = {'m', 'k', 'q'}, mb = {'q', 'n', 'k'}, md = {'n', 'm'};
        std::shuffle(ma.begin(), ma.end(), rng);
        std::shuffle(mb.begin(), mb.end(), rng);
        auto A = domainTensor(c.type, ma, ext, rng, c.opA);
        auto B = domainTensor(c.type, mb, ext, rng, c.opB);
        auto C = domainTensor(c.type, md, ext, rng, c.opC);
        auto D = C;
        std::fill(D.values.begin(), D.values.end(), cplx(0, 0));
        bool const f64  = isF64Compute(c.compute) || c.type == HIPTENSOR_R_64F;
        auto       st   = scalarTypeOf(c.compute);
        double     uIn  = 0; // the operand rounding of a 16-bit compute type
        double     uAcc = f64 ? std::ldexp(1.0, -53) : std::ldexp(1.0, -24);
        if(c.compute == HIPTENSOR_COMPUTE_DESC_16F || c.compute == HIPTENSOR_COMPUTE_DESC_16BF)
        {
            uIn = unitRoundoff(st);
        }
        hiptensorOperationDescriptor_t desc = nullptr;
        auto dA = describe(api, A), dB = describe(api, B), dC = describe(api, C);
        auto dD = describe(api, D);
        ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                             &desc,
                                             dA,
                                             A.modes.data(),
                                             c.opA,
                                             dB,
                                             B.modes.data(),
                                             c.opB,
                                             dC,
                                             C.modes.data(),
                                             c.opC,
                                             dD,
                                             D.modes.data(),
                                             c.compute),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(desc);
        Choice all;
        all.workspace      = HIPTENSOR_WORKSPACE_MAX;
        int32_t candidates = 1;
        for(int32_t rank = 0; rank < candidates; rank++)
        {
            all.rank = rank;
            LogCapture   log;
            SelectionLog sel;
            uint64_t     ws   = 0;
            auto         plan = planWith(api, desc, all, log, ws, &sel);
            ASSERT_NE(plan, nullptr) << "KERNEL_RANK " << rank;
            if(rank == 0)
            {
                candidates = sel.candidates;
                ASSERT_GE(candidates, 1);
            }
            Device dev;
            void*  pA = dev.upload(A);
            void*  pB = dev.upload(B);
            void*  pC = dev.upload(C);
            void*  pD = dev.upload(D);
            void*  pW = nullptr;
            if(ws > 0)
            {
                ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
                dev.buffers.push_back(pW);
            }
            Scalar alpha, beta;
            alpha.re = 0.75;
            beta.re  = 0.5;
            Stream stream;
            ASSERT_EQ(
                hiptensorContract(
                    api.handle, plan, alpha.as(st), pA, pB, beta.as(st), pC, pD, pW, ws, stream.s),
                HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            Tensor got = D;
            dev.download(pD, got);
            double const eA  = 2 * opUlps(c.opA, f64) * uAcc + 2 * uIn;
            double const eB  = 2 * opUlps(c.opB, f64) * uAcc + 2 * uIn;
            double const eC  = 2 * opUlps(c.opC, f64) * uAcc;
            int64_t      bad = 0, count = 0;
            forEachIndex(md, ext, [&](auto index) {
                long double sum  = 0;
                double      mags = 0;
                forEachIndex(std::vector<int32_t>{'k', 'q'}, ext, [&](auto kindex) {
                    auto full = index;
                    for(auto& kv : kindex)
                    {
                        full[kv.first] = kv.second;
                    }
                    long double a = unaryL(c.opA, A.values[A.offset(full)].real());
                    long double b = unaryL(c.opB, B.values[B.offset(full)].real());
                    sum += a * b;
                    mags += double(std::fabs(a * b));
                });
                double const cv    = double(unaryL(c.opC, C.values[C.offset(index)].real()));
                double const want  = double(alpha.re * sum) + beta.re * cv;
                int64_t      K     = ext['k'] * ext['q'];
                double       bound = std::fabs(alpha.re) * mags * (2 * (K + 4) * uAcc + eA + eB)
                               + std::fabs(beta.re * cv) * (4 * uAcc + eC)
                               + 2 * unitRoundoff(D.type) * std::fabs(want) + 1e-300;
                double g = got.values[D.offset(index)].real();
                count++;
                if(!(std::fabs(g - want) <= bound) && bad++ < 5)
                {
                    ADD_FAILURE() << "candidate " << sel.pick << " got " << g << " want " << want
                                  << " (bound " << bound << ")";
                }
            });
            EXPECT_EQ(bad, 0) << sel.pick;
            EXPECT_EQ(count, ext['m'] * ext['n']);
        }
    }

    std::vector<UnaryContractionCase> unaryContractionCases()
    {
        auto const f16 = HIPTENSOR_R_16F, bf16 = HIPTENSOR_R_16BF, f32 = HIPTENSOR_R_32F,
                   f64 = HIPTENSOR_R_64F;
        auto const c32 = HIPTENSOR_COMPUTE_DESC_32F, c64 = HIPTENSOR_COMPUTE_DESC_64F,
                   c16 = HIPTENSOR_COMPUTE_DESC_16F;
        return {
            {f32, c32, HIPTENSOR_OP_SIN, HIPTENSOR_OP_COS, HIPTENSOR_OP_TANH, 4700},
            {f32, c32, HIPTENSOR_OP_ATAN, HIPTENSOR_OP_EXP, HIPTENSOR_OP_FLOOR, 4701},
            {f16, c32, HIPTENSOR_OP_SINH, HIPTENSOR_OP_ASIN, HIPTENSOR_OP_COSH, 4702},
            {bf16, c32, HIPTENSOR_OP_ACOSH, HIPTENSOR_OP_LOG, HIPTENSOR_OP_ATANH, 4703},
            {f16, c16, HIPTENSOR_OP_COS, HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_SIN, 4704},
            {f64, c64, HIPTENSOR_OP_TAN, HIPTENSOR_OP_ASINH, HIPTENSOR_OP_ACOS, 4705},
            {f64, c64, HIPTENSOR_OP_CEIL, HIPTENSOR_OP_SIGMOID, HIPTENSOR_OP_RELU, 4706},
            {f32, c32, HIPTENSOR_OP_CONJ, HIPTENSOR_OP_RCP, HIPTENSOR_OP_SQRT, 4707},
        };
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeUnaryContraction,
                             ::testing::ValuesIn(unaryContractionCases()));

    // CONJ of a real tensor is the identity (it takes the identity's routes); operators other
    // than IDENTITY and CONJ on complex tensors are declined (neither backend defines them).
    TEST(RockeUnary, ConjOfRealAndOperatorsOfComplex)
    {
        ApiObjects    api;
        int64_t const M = 6, K = 5, N = 4;
        int32_t const ma[] = {'m', 'k'}, mb[] = {'k', 'n'}, md[] = {'m', 'n'};
        for(auto type : {HIPTENSOR_C_32F, HIPTENSOR_C_64F})
        {
            auto compute = type == HIPTENSOR_C_32F ? HIPTENSOR_COMPUTE_DESC_C32F
                                                   : HIPTENSOR_COMPUTE_DESC_C64F;
            auto dA = api.tensor({M, K}, {}, type), dB = api.tensor({K, N}, {}, type);
            auto dD = api.tensor({M, N}, {}, type);
            for(auto op : {HIPTENSOR_OP_SIN, HIPTENSOR_OP_ABS, HIPTENSOR_OP_FLOOR})
            {
                LogCapture                     log;
                hiptensorOperationDescriptor_t desc = nullptr;
                ASSERT_EQ(hiptensorCreateContraction(api.handle,
                                                     &desc,
                                                     dA,
                                                     ma,
                                                     op,
                                                     dB,
                                                     mb,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     nullptr,
                                                     nullptr,
                                                     HIPTENSOR_OP_IDENTITY,
                                                     dD,
                                                     md,
                                                     compute),
                          HIPTENSOR_STATUS_SUCCESS);
                api.keep(desc);
                hiptensorPlan_t plan = nullptr;
                uint64_t        ws   = 0;
                EXPECT_EQ(api.plan(desc, plan, ws), HIPTENSOR_STATUS_NOT_SUPPORTED) << log.all();
            }
        }
        // CONJ of real data: the same bits as IDENTITY (exact integer data)
        for(auto type : {HIPTENSOR_R_16F, HIPTENSOR_R_32F, HIPTENSOR_R_64F})
        {
            ContractionCase c{type,
                              type == HIPTENSOR_R_64F ? HIPTENSOR_COMPUTE_DESC_64F
                                                      : HIPTENSOR_COMPUTE_DESC_32F,
                              4800 + int(type),
                              true,
                              true,
                              false,
                              HIPTENSOR_OP_CONJ};
            runContraction(c);
        }
    }

    // Trinary contractions with an operator on every input (the quick tier's COS rows), f32
    // and f64 data.
    TEST(RockeUnary, TrinaryContractionOperators)
    {
        for(auto type : {HIPTENSOR_R_32F, HIPTENSOR_R_64F})
        {
            ApiObjects                 api;
            LogCapture                 log;
            std::mt19937               rng(4900 + int(type));
            bool const                 f64 = type == HIPTENSOR_R_64F;
            std::map<int32_t, int64_t> ext = {{'m', 9}, {'k', 13}, {'u', 7}, {'n', 11}};
            hiptensorOperator_t const  opA = HIPTENSOR_OP_COS, opB = HIPTENSOR_OP_ATAN,
                                      opC = HIPTENSOR_OP_SINH, opD = HIPTENSOR_OP_FLOOR;
            auto A = domainTensor(type, {'m', 'k'}, ext, rng, opA);
            auto B = domainTensor(type, {'k', 'u'}, ext, rng, opB);
            auto C = domainTensor(type, {'u', 'n'}, ext, rng, opC);
            auto D = domainTensor(type, {'n', 'm'}, ext, rng, opD);
            auto E = makeTensor(type, {'m', 'n'}, ext, rng, true);
            std::fill(E.values.begin(), E.values.end(), cplx(0, 0));
            auto compute = f64 ? HIPTENSOR_COMPUTE_DESC_64F : HIPTENSOR_COMPUTE_DESC_32F;
            hiptensorOperationDescriptor_t desc = nullptr;
            auto dA = describe(api, A), dB = describe(api, B), dC = describe(api, C);
            auto dD = describe(api, D), dE = describe(api, E);
            ASSERT_EQ(hiptensorCreateContractionTrinary(api.handle,
                                                        &desc,
                                                        dA,
                                                        A.modes.data(),
                                                        opA,
                                                        dB,
                                                        B.modes.data(),
                                                        opB,
                                                        dC,
                                                        C.modes.data(),
                                                        opC,
                                                        dD,
                                                        D.modes.data(),
                                                        opD,
                                                        dE,
                                                        E.modes.data(),
                                                        compute),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(desc);
            uint64_t ws   = 0;
            auto     plan = planOf(api, desc, ws, log);
            ASSERT_NE(plan, nullptr);
            Device dev;
            void*  pA = dev.upload(A);
            void*  pB = dev.upload(B);
            void*  pC = dev.upload(C);
            void*  pD = dev.upload(D);
            void*  pE = dev.upload(E);
            void*  pW = nullptr;
            ASSERT_EQ(hipMalloc(&pW, ws), hipSuccess);
            dev.buffers.push_back(pW);
            Scalar alpha, beta;
            alpha.re = 0.5;
            beta.re  = -1.0;
            auto st  = f64 ? HIPTENSOR_R_64F : HIPTENSOR_R_32F;
            ASSERT_EQ(hiptensorContractTrinary(api.handle,
                                               plan,
                                               alpha.as(st),
                                               pA,
                                               pB,
                                               pC,
                                               beta.as(st),
                                               pD,
                                               pE,
                                               pW,
                                               ws,
                                               nullptr),
                      HIPTENSOR_STATUS_SUCCESS)
                << log.all();
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            Tensor got = E;
            dev.download(pE, got);
            double const u  = f64 ? std::ldexp(1.0, -53) : std::ldexp(1.0, -24);
            double const eA = 2 * opUlps(opA, f64) * u, eB = 2 * opUlps(opB, f64) * u;
            double const eC  = 2 * opUlps(opC, f64) * u;
            int64_t      bad = 0;
            forEachIndex(std::vector<int32_t>{'m', 'n'}, ext, [&](auto index) {
                // T[m, u] = opA(A) opB(B) (rounded to E's type), then E = alpha T opC(C) + ...
                long double sum = 0;
                double      mag = 0, tErr = 0;
                for(int64_t uu = 0; uu < ext['u']; uu++)
                {
                    long double t  = 0;
                    double      tm = 0;
                    for(int64_t kk = 0; kk < ext['k']; kk++)
                    {
                        std::map<int32_t, int64_t> full
                            = {{'m', index.at('m')}, {'n', index.at('n')}, {'k', kk}, {'u', uu}};
                        long double a = unaryL(opA, A.values[A.offset(full)].real());
                        long double b = unaryL(opB, B.values[B.offset(full)].real());
                        t += a * b;
                        tm += double(std::fabs(a * b));
                    }
                    std::map<int32_t, int64_t> full = {{'n', index.at('n')}, {'u', uu}};
                    long double                c    = unaryL(opC, C.values[C.offset(full)].real());
                    double                     tb
                        = tm * (2 * (ext['k'] + 4) * u + eA + eB) + 2 * u * double(std::fabs(t));
                    sum += t * c;
                    mag += double(std::fabs(t * c));
                    tErr += tb * double(std::fabs(c));
                }
                double const dv   = double(unaryL(opD, D.values[D.offset(index)].real()));
                double const want = double(alpha.re * sum) + beta.re * dv;
                double const bound
                    = std::fabs(alpha.re) * (mag * (2 * (ext['u'] + 4) * u + eC) + tErr)
                      + 4 * u * std::fabs(beta.re * dv) + 2 * u * std::fabs(want) + 1e-300;
                double g = got.values[E.offset(index)].real();
                if(!(std::fabs(g - want) <= bound) && bad++ < 5)
                {
                    ADD_FAILURE() << "got " << g << " want " << want << " (bound " << bound << ")";
                }
            });
            EXPECT_EQ(bad, 0) << log.all();
        }
    }
    // ---------------------------------------------------------- in place, without workspace

    // Every contraction plans with a workspace limit of zero: the direct GETT (or the ring GETT)
    // reads the operands and writes D through their own strides, against the loop reference of
    // each case (every data type and compute type, scale and bilinear, conjugate, 10 and 12
    // modes per tensor).
    class RockeZeroWorkspaceContraction : public ::testing::TestWithParam<ContractionCase>
    {
    };

    TEST_P(RockeZeroWorkspaceContraction, MatchesLoopReference)
    {
        if(f64RouteOnly())
        {
            GTEST_SKIP() << "the f64 route needs workspace";
        }
        Choice none;
        none.zeroWorkspace = true;
        SelectionLog s;
        runContraction(GetParam(), none, &s);
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        EXPECT_EQ(s.required, 0u);
        EXPECT_TRUE(s.pick.rfind("dgett:", 0) == 0 || s.pick.rfind("rgett:", 0) == 0) << s.pick;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke,
                             RockeZeroWorkspaceContraction,
                             ::testing::ValuesIn(contractionCases()));

    // Mixed data types (A, B, C, D of different types, 16-bit compute types from f32 and f64)
    // without workspace.
    class RockeZeroWorkspaceMixed : public ::testing::TestWithParam<MixedCase>
    {
    };

    TEST_P(RockeZeroWorkspaceMixed, MatchesLoopReference)
    {
        if(f64RouteOnly())
        {
            GTEST_SKIP() << "the f64 route needs workspace";
        }
        Choice none;
        none.zeroWorkspace = true;
        SelectionLog s;
        runMixed(GetParam(), none, &s);
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        EXPECT_EQ(s.required, 0u);
        EXPECT_TRUE(s.pick.rfind("dgett:", 0) == 0 || s.pick.rfind("rgett:", 0) == 0) << s.pick;
    }

    INSTANTIATE_TEST_SUITE_P(Rocke, RockeZeroWorkspaceMixed, ::testing::ValuesIn(mixedCases()));

    // D[m, n] = alpha * sum_k A[m, k] B[n, k] + beta * C[m, n], every tensor packed column-major
    // (the ring GETT's layout on gfx950), integer data with a NaN in A, an infinity in B and a
    // NaN in C; `offset` elements past each allocation's start (the buffers then miss the
    // alignment of vector accesses). Runs every candidate (KERNEL_RANK) with the given scalars
    // and compares with the IEEE reference bit for bit (NaN positions included). Returns the
    // picks.
    std::vector<std::string> runInPlace(hiptensorDataType_t          type,
                                        hiptensorComputeDescriptor_t compute,
                                        double                       alphaRe,
                                        double                       betaRe,
                                        int64_t                      offset,
                                        bool                         specials,
                                        std::vector<std::string>*    launches = nullptr)
    {
        std::vector<std::string> picks;
        ApiObjects               probe;
        if(!probe.supports(type))
        {
            return picks;
        }
        int64_t const M = 264, N = 200, K = 96;
        std::mt19937  rng(int(type) * 7 + int(offset));
        auto          make = [&](std::vector<int32_t> modes, std::vector<int64_t> ext) {
            Tensor t;
            t.type    = type;
            t.modes   = modes;
            t.extents = ext;
            t.strides = {1, ext[0]};
            t.values.resize(ext[0] * ext[1]);
            std::uniform_int_distribution<int> ints(-3, 3);
            for(auto& v : t.values)
            {
                v = cplx(ints(rng), isComplex(type) ? ints(rng) : 0);
            }
            return t;
        };
        auto A = make({'m', 'k'}, {M, K});
        auto B = make({'n', 'k'}, {N, K});
        auto C = make({'m', 'n'}, {M, N});
        auto D = make({'m', 'n'}, {M, N});
        if(specials)
        {
            // a NaN in A and C; an infinity in B only where it is not used (alpha == 0): the
            // emulated-f32 GEMM route splits an infinity into parts whose sum is NaN
            double const nan = std::numeric_limits<double>::quiet_NaN();
            A.values[5]      = cplx(nan, 0);
            C.values[11]     = cplx(nan, 0);
            if(alphaRe == 0.0)
            {
                B.values[7] = cplx(std::numeric_limits<double>::infinity(), 0);
            }
        }
        auto const scalarType = scalarTypeOf(compute);
        Scalar     alpha, beta;
        alpha.re = alphaRe;
        beta.re  = betaRe;
        // reference (IEEE double; integer data: exact)
        std::vector<cplx> want(D.values.size());
        for(int64_t m = 0; m < M; m++)
        {
            for(int64_t n = 0; n < N; n++)
            {
                // the components' products written out (no recovery of infinities); real scales
                double sr = 0, si = 0;
                for(int64_t k = 0; k < K; k++)
                {
                    auto const a = A.values[m + k * M], b = B.values[n + k * N];
                    if(!isComplex(type))
                    {
                        sr += a.real() * b.real();
                        continue;
                    }
                    sr += a.real() * b.real() - a.imag() * b.imag();
                    si += a.real() * b.imag() + a.imag() * b.real();
                }
                cplx       v    = alphaRe == 0.0 ? cplx(0, 0) : cplx(alphaRe * sr, alphaRe * si);
                cplx const c    = C.values[m + n * M];
                v               = betaRe == 0.0
                                      ? v
                                      : cplx(v.real() + betaRe * c.real(), v.imag() + betaRe * c.imag());
                want[m + n * M] = cplx(roundTo(v.real(), type), roundTo(v.imag(), type));
            }
        }
        for(int32_t rank = 0;; rank++)
        {
            ApiObjects                     api;
            LogCapture                     log;
            hiptensorOperationDescriptor_t desc = nullptr;
            auto dA = describe(api, A), dB = describe(api, B), dC = describe(api, C),
                 dD = describe(api, D);
            EXPECT_EQ(hiptensorCreateContraction(api.handle,
                                                 &desc,
                                                 dA,
                                                 A.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dB,
                                                 B.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dC,
                                                 C.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dD,
                                                 D.modes.data(),
                                                 compute),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(desc);
            hiptensorPlanPreference_t pref = nullptr;
            EXPECT_EQ(hiptensorCreatePlanPreference(
                          api.handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                      HIPTENSOR_STATUS_SUCCESS);
            EXPECT_EQ(
                hiptensorPlanPreferenceSetAttribute(
                    api.handle, pref, HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK, &rank, sizeof(rank)),
                HIPTENSOR_STATUS_SUCCESS);
            uint64_t ws = 0;
            EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                          api.handle, desc, pref, HIPTENSOR_WORKSPACE_MAX, &ws),
                      HIPTENSOR_STATUS_SUCCESS);
            hiptensorPlan_t plan   = nullptr;
            auto            status = hiptensorCreatePlan(api.handle, &plan, desc, pref, ws);
            static_cast<void>(hiptensorDestroyPlanPreference(pref));
            if(status == HIPTENSOR_STATUS_NOT_SUPPORTED && rank > 0)
            {
                break; // past the candidates
            }
            EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << log.all();
            if(plan == nullptr)
            {
                break;
            }
            api.plans.push_back(plan);
            auto const sel = selectionOf(log);
            picks.push_back(sel.pick);
            if(launches != nullptr)
            {
                launches->push_back(sel.launches);
            }
            uint64_t required = 0;
            EXPECT_EQ(hiptensorPlanGetAttribute(api.handle,
                                                plan,
                                                HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                                &required,
                                                sizeof(required)),
                      HIPTENSOR_STATUS_SUCCESS);
            // buffers `offset` elements past their allocation
            Device dev;
            auto   at = [&](Tensor const& t) {
                Tensor s = t;
                s.values.insert(s.values.begin(), offset, cplx(0, 0));
                return static_cast<uint8_t*>(dev.upload(s)) + offset * sizeOf(type);
            };
            void* pA = at(A);
            void* pB = at(B);
            void* pC = at(C);
            void* pD = at(D);
            void* w  = nullptr;
            if(required > 0)
            {
                EXPECT_EQ(hipMalloc(&w, required), hipSuccess);
                dev.buffers.push_back(w);
            }
            Stream stream;
            EXPECT_EQ(hiptensorContract(api.handle,
                                        plan,
                                        alpha.as(scalarType),
                                        pA,
                                        pB,
                                        beta.as(scalarType),
                                        pC,
                                        pD,
                                        w,
                                        required,
                                        stream.s),
                      HIPTENSOR_STATUS_SUCCESS)
                << sel.pick;
            EXPECT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
            Tensor got = D;
            got.values.insert(got.values.begin(), offset, cplx(0, 0));
            dev.download(static_cast<uint8_t*>(pD) - offset * sizeOf(type), got);
            got.values.erase(got.values.begin(), got.values.begin() + offset);
            int64_t bad = 0;
            for(size_t i = 0; i < want.size(); i++)
            {
                auto const g = got.values[i], x = want[i];
                // a complex NaN is an element with a NaN component (a complex scale times
                // (NaN, 0) is (NaN, NaN), the reference's real scale gives (NaN, 0))
                bool const xNan = std::isnan(x.real()) || std::isnan(x.imag());
                bool const gNan = std::isnan(g.real()) || std::isnan(g.imag());
                bool const same = isComplex(type)
                                      ? (xNan ? gNan : g == x)
                                      : (std::isnan(g.real()) == std::isnan(x.real()))
                                            && (std::isnan(x.real()) || g.real() == x.real());
                if(!same && bad++ < 5)
                {
                    ADD_FAILURE() << sel.pick << " alpha " << alphaRe << " beta " << betaRe
                                  << " offset " << offset << ": element " << i << " got " << g
                                  << " want " << x;
                }
            }
            EXPECT_EQ(bad, 0) << sel.pick;
        }
        return picks;
    }

    // Zero scales: alpha == 0 does not use A and B (a NaN or infinity in them does not reach D),
    // beta == 0 does not read C (its NaN does not reach D); with both scales nonzero the NaN and
    // the infinity reach exactly their outputs. Every candidate: the ring GETT and the tuned
    // plans, whose kernels read every term, take the direct GETT for zero scales.
    TEST(RockeInPlaceGett, ZeroScalesDoNotReadTheirTerms)
    {
        struct Row
        {
            hiptensorDataType_t          type;
            hiptensorComputeDescriptor_t compute;
        };
        for(auto const& r : {Row{HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F},
                             Row{HIPTENSOR_R_16BF, HIPTENSOR_COMPUTE_DESC_32F},
                             Row{HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F},
                             Row{HIPTENSOR_R_64F, HIPTENSOR_COMPUTE_DESC_64F},
                             Row{HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F}})
        {
            for(auto [alpha, beta] :
                {std::pair{0.0, 2.0}, std::pair{2.0, 0.0}, std::pair{2.0, -1.0}})
            {
                auto picks = runInPlace(r.type, r.compute, alpha, beta, 0, true);
                if(r.type == HIPTENSOR_R_16F || r.type == HIPTENSOR_R_16BF)
                {
                    bool ring = false;
                    for(auto const& p : picks)
                    {
                        ring = ring || p.rfind("rgett:", 0) == 0;
                    }
                    EXPECT_EQ(ring, deviceArch() == "gfx950");
                }
                ASSERT_FALSE(HasFailure()) << "type " << int(r.type);
            }
        }
    }

    // Buffers that miss the alignment of the ring GETT's vector accesses (one element past an
    // aligned address): the plan's fallback (the direct GETT) runs, with the same results.
    TEST(RockeInPlaceGett, UnalignedBuffers)
    {
        for(auto type : {HIPTENSOR_R_16F, HIPTENSOR_R_16BF})
        {
            auto picks = runInPlace(type, HIPTENSOR_COMPUTE_DESC_32F, 2.0, -1.0, 1, false);
            ASSERT_FALSE(picks.empty());
            if(deviceArch() == "gfx950")
            {
                EXPECT_EQ(picks.front().rfind("rgett:", 0), 0u) << picks.front();
            }
        }
        // the other data types: every candidate with element-aligned buffers
        runInPlace(HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, 2.0, -1.0, 1, false);
        runInPlace(HIPTENSOR_C_32F, HIPTENSOR_COMPUTE_DESC_C32F, 2.0, -1.0, 1, false);
    }
    // The operand packs and the epilogue of the emulated-f32 GEMM route read their input along
    // another mode than they write (A and B fastest along M and N, packed fastest along K; the
    // result fastest along N, D along M): they run on the tiled elementwise kernel, with the
    // generic launch as the fallback of buffers not aligned to its vectors. Every candidate is
    // exact with aligned and unaligned buffers.
    TEST(RockeInPlaceGett, TiledPackAndEpilogueLaunches)
    {
        // the emulated-f32 GEMM route (f32 data takes the native route by default)
        ScopedEnv recipe("HIPTENSOR_ROCKE_F32_RECIPE", "x6");
        for(int64_t offset : {int64_t(0), int64_t(1)})
        {
            std::vector<std::string> launches;
            auto                     picks = runInPlace(
                HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, 2.0, -1.0, offset, false, &launches);
            ASSERT_FALSE(HasFailure());
            ASSERT_EQ(picks.size(), launches.size());
            bool tiledGemm = false;
            for(size_t i = 0; i < picks.size(); i++)
            {
                if(picks[i].rfind("gemm:", 0) == 0)
                {
                    // pack A, pack B, the GEMM, the epilogue
                    EXPECT_EQ(launches[i].rfind("ewt+ewt+gemm+ewt", 0), 0u)
                        << picks[i] << ": " << launches[i];
                    tiledGemm = true;
                }
            }
            // the GEMM routes are gfx950's
            EXPECT_EQ(tiledGemm, deviceArch() == "gfx950");
        }
    }
    // Plans executed on another device of the same architecture than the one they were created
    // on: every candidate of a contraction with tiled packs and epilogues of many modes (the
    // pack kernels load on the executing device; their generic fallbacks run only for unaligned
    // buffers) and of a 16-bit contraction (every route of its data type), exact, aligned and
    // not; with the JIT (HIPTENSOR_ROCKE_TEST_JIT=1) the specialized launches run the pack
    // launches they replaced.
    TEST(RockeInPlaceGett, OtherDevice)
    {
        int const other = otherDevice();
        if(other < 0)
        {
            GTEST_SKIP() << "one device of this architecture";
        }
        ContractionCase many{HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, 4100, true, true, false};
        many.groupRank  = 6;
        many.groupFirst = 4;
        ContractionCase const half{
            HIPTENSOR_R_16F, HIPTENSOR_COMPUTE_DESC_32F, 1501, true, true, false};
        for(auto const& c : {many, half})
        {
            for(size_t skew : {size_t(0), size_t(1)})
            {
                for(int32_t rank = 0;; rank++)
                {
                    Choice choice;
                    choice.rank      = rank;
                    choice.skew      = skew;
                    choice.device    = other;
                    choice.workspace = HIPTENSOR_WORKSPACE_MAX;
                    SelectionLog sel;
                    runContraction(c, choice, &sel);
                    ASSERT_FALSE(HasFailure()) << "seed " << c.seed << " KERNEL_RANK " << rank
                                               << " skew " << skew << ": " << sel.pick;
                    if(rank + 1 >= sel.candidates)
                    {
                        break;
                    }
                }
            }
        }
    }

    // Operands and D of 12 interleaved modes (6 per group, none merge): a pack or epilogue of the
    // emulated-f32 GEMM route that the generic kernel needs several launches for (more modes than
    // its slots) runs as one launch of the tiled kernel, with those launches as the fallback of
    // buffers not aligned to its vectors. Every candidate is exact, aligned and not.
    TEST(RockeInPlaceGett, TiledLaunchesOfManyModes)
    {
        // the emulated-f32 GEMM route (f32 data takes the native route by default)
        ScopedEnv       recipe("HIPTENSOR_ROCKE_F32_RECIPE", "x6");
        ContractionCase c{HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, 4100, true, true, false};
        c.groupRank  = 6;
        c.groupFirst = 4;
        for(size_t skew : {size_t(0), size_t(1)})
        {
            bool tiledGemm = false;
            for(int32_t rank = 0;; rank++)
            {
                Choice choice;
                choice.rank      = rank;
                choice.skew      = skew;
                choice.workspace = HIPTENSOR_WORKSPACE_MAX;
                SelectionLog sel;
                runContraction(c, choice, &sel);
                ASSERT_FALSE(HasFailure()) << "KERNEL_RANK " << rank << " skew " << skew;
                if(sel.pick.rfind("gemm:", 0) == 0)
                {
                    // one launch per operand pack and for the epilogue (the generic kernel
                    // would need a launch per index of the modes past its slots)
                    EXPECT_EQ(std::count(sel.launches.begin(), sel.launches.end(), '+'), 3)
                        << sel.pick << ": " << sel.launches;
                    EXPECT_NE(sel.launches.find("ewt"), std::string::npos) << sel.launches;
                    tiledGemm = true;
                }
                if(rank + 1 >= sel.candidates)
                {
                    break;
                }
            }
            EXPECT_EQ(tiledGemm, deviceArch() == "gfx950");
        }
    }

    // ---------------------------------------------------------------- tiled elementwise

    // An elementwise case for the tiled kernel: D[a, b, c, d] (mode a fastest) with input 0
    // fastest along b (read along the transposed mode), the other inputs fastest along a, b or a
    // batch mode; extents of a and b multiples of every vector width. `rank`: KERNEL_RANK (0 the
    // tiled kernel, 1 the generic one); `skew`: every buffer one element past an aligned address
    // (the tiled program then runs its fallback); `zeroScale`: an input whose scale is zero and
    // holds a NaN (it must not be read).
    struct TiledCase
    {
        int                 arity;
        hiptensorDataType_t in, out, scalar;
        int                 seed;
        int32_t             rank      = 0;
        size_t              skew      = 0;
        int                 zeroScale = -1;
        // false: input 0 laid out as D (the vector class of the generic kernel, not the tiled
        // kernel)
        bool transposed = true;
        // extents of a (D's fastest mode) and b (input 0's fastest mode) and of the batch mode c
        int64_t ea = 72, eb = 40, ec = 3;
        // every input laid out as D (one contiguous run: the flat kernel on gfx942 / gfx950)
        bool flat = false;
    };

    // A packed tensor stored in `order` (fastest first), integer values in [-3, 3].
    Tensor packedTensor(hiptensorDataType_t               type,
                        std::vector<int32_t> const&       modes,
                        std::vector<int32_t> const&       order,
                        std::map<int32_t, int64_t> const& ext,
                        std::mt19937&                     rng)
    {
        Tensor t;
        t.type  = type;
        t.modes = modes;
        for(auto m : modes)
        {
            t.extents.push_back(ext.at(m));
        }
        t.strides.assign(modes.size(), 0);
        int64_t s = 1;
        for(auto m : order)
        {
            auto i       = std::find(modes.begin(), modes.end(), m) - modes.begin();
            t.strides[i] = s;
            s *= ext.at(m);
        }
        t.values.resize(s);
        std::uniform_int_distribution<int> ints(-3, 3);
        for(auto& v : t.values)
        {
            v = cplx(ints(rng), 0);
        }
        return t;
    }

    // Returns the pick.
    std::string runTiled(TiledCase const& c)
    {
        std::mt19937 rng(c.seed);
        ApiObjects   api;
        if(!api.supports(c.in))
        {
            return "";
        }
        LogCapture                        log;
        std::map<int32_t, int64_t>        ext   = {{'a', c.ea}, {'b', c.eb}, {'c', c.ec}, {'d', 5}};
        std::vector<int32_t>              modes = {'a', 'b', 'c', 'd'};
        std::vector<Tensor>               ins;
        std::vector<std::vector<int32_t>> orders
            = {{'b', 'a', 'd', 'c'}, {'a', 'c', 'b', 'd'}, {'c', 'b', 'a', 'd'}};
        for(int i = 0; i < c.arity; i++)
        {
            auto order = orders[i == 0 ? 0 : 1 + rng() % 2];
            if(i > 0 && rng() % 3 == 0)
            {
                order = orders[0];
            }
            if(!c.transposed)
            {
                order = i == 0 ? std::vector<int32_t>{'a', 'b', 'c', 'd'} : orders[1 + rng() % 2];
            }
            if(c.flat)
            {
                order = {'a', 'b', 'c', 'd'};
            }
            ins.push_back(packedTensor(c.in, modes, order, ext, rng));
        }
        auto D = packedTensor(c.out, modes, {'a', 'b', 'c', 'd'}, ext, rng);
        std::fill(D.values.begin(), D.values.end(), cplx(0, 0));
        if(c.zeroScale >= 0)
        {
            ins[c.zeroScale].values[3] = cplx(std::numeric_limits<double>::quiet_NaN(), 0);
        }
        hiptensorOperator_t const ops[]
            = {HIPTENSOR_OP_IDENTITY, HIPTENSOR_OP_NEG, HIPTENSOR_OP_ABS, HIPTENSOR_OP_RELU};
        hiptensorOperator_t const bops[]
            = {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL, HIPTENSOR_OP_MAX, HIPTENSOR_OP_MIN};
        std::vector<hiptensorOperator_t> uop;
        for(int i = 0; i < c.arity; i++)
        {
            uop.push_back(ops[rng() % 4]);
        }
        auto   opAB = bops[rng() % 4], opABC = bops[rng() % 4];
        Scalar s[3];
        for(int i = 0; i < 3; i++)
        {
            s[i].re = (i + c.seed) % 3 == 0 ? 2.0 : (i + c.seed) % 3 == 1 ? -1.0 : 0.5;
        }
        // the scale of input i: alpha (A), beta (trinary B) or gamma (C)
        auto scaleOf = [&](int i) -> Scalar& { return c.arity == 2 && i == 1 ? s[2] : s[i]; };
        if(c.zeroScale >= 0)
        {
            scaleOf(c.zeroScale).re = 0.0;
        }
        auto compute = c.scalar == HIPTENSOR_R_64F    ? HIPTENSOR_COMPUTE_DESC_64F
                       : c.scalar == HIPTENSOR_R_16F  ? HIPTENSOR_COMPUTE_DESC_16F
                       : c.scalar == HIPTENSOR_R_16BF ? HIPTENSOR_COMPUTE_DESC_16BF
                                                      : HIPTENSOR_COMPUTE_DESC_32F;
        hiptensorOperationDescriptor_t           desc = nullptr;
        std::vector<hiptensorTensorDescriptor_t> d;
        for(auto const& t : ins)
        {
            d.push_back(describe(api, t));
        }
        auto dD = describe(api, D);
        if(c.arity == 1)
        {
            EXPECT_EQ(hiptensorCreatePermutation(
                          api.handle, &desc, d[0], modes.data(), uop[0], dD, modes.data(), compute),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        else if(c.arity == 2)
        {
            EXPECT_EQ(hiptensorCreateElementwiseBinary(api.handle,
                                                       &desc,
                                                       d[0],
                                                       modes.data(),
                                                       uop[0],
                                                       d[1],
                                                       modes.data(),
                                                       uop[1],
                                                       dD,
                                                       modes.data(),
                                                       opAB,
                                                       compute),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        else
        {
            EXPECT_EQ(hiptensorCreateElementwiseTrinary(api.handle,
                                                        &desc,
                                                        d[0],
                                                        modes.data(),
                                                        uop[0],
                                                        d[1],
                                                        modes.data(),
                                                        uop[1],
                                                        d[2],
                                                        modes.data(),
                                                        uop[2],
                                                        dD,
                                                        modes.data(),
                                                        opAB,
                                                        opABC,
                                                        compute),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        api.keep(desc);
        Choice choice;
        choice.rank     = c.rank;
        uint64_t     ws = 0;
        SelectionLog sel;
        auto         plan = planWith(api, desc, choice, log, ws, &sel);
        if(plan == nullptr)
        {
            return "";
        }
        Device dev;
        auto   at = [&](Tensor const& t) {
            Tensor x = t;
            x.values.insert(x.values.begin(), c.skew, cplx(0, 0));
            return static_cast<uint8_t*>(dev.upload(x)) + c.skew * sizeOf(t.type);
        };
        std::vector<void*> p;
        for(auto const& t : ins)
        {
            p.push_back(at(t));
        }
        void*             pD = at(D);
        Stream            stream;
        hiptensorStatus_t status;
        if(c.arity == 1)
        {
            status = hiptensorPermute(api.handle, plan, s[0].as(c.scalar), p[0], pD, stream.s);
        }
        else if(c.arity == 2)
        {
            status = hiptensorElementwiseBinaryExecute(
                api.handle, plan, s[0].as(c.scalar), p[0], s[2].as(c.scalar), p[1], pD, stream.s);
        }
        else
        {
            Scalar a = s[0], b = s[1], g = s[2];
            status = hiptensorElementwiseTrinaryExecute(api.handle,
                                                        plan,
                                                        a.as(c.scalar),
                                                        p[0],
                                                        b.as(c.scalar),
                                                        p[1],
                                                        g.as(c.scalar),
                                                        p[2],
                                                        pD,
                                                        stream.s);
        }
        EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << log.all();
        EXPECT_EQ(hipStreamSynchronize(stream.s), hipSuccess);
        Tensor got = D;
        got.values.insert(got.values.begin(), c.skew, cplx(0, 0));
        dev.download(static_cast<uint8_t*>(pD) - c.skew * sizeOf(D.type), got);
        got.values.erase(got.values.begin(), got.values.begin() + c.skew);
        std::vector<cplx>   want(D.values.size(), 0);
        std::vector<double> bound(D.values.size(), 0);
        std::vector<bool>   written(D.values.size(), false);
        forEachIndex(modes, ext, [&](auto index) {
            std::vector<double> t;
            for(int i = 0; i < c.arity; i++)
            {
                double const sc = scaleOf(i).re;
                // an input whose scale is zero is not read (its NaN does not reach D)
                t.push_back(
                    sc == 0.0 ? 0.0
                              : sc * unaryRef(uop[i], ins[i].values[ins[i].offset(index)].real()));
            }
            double r = t[0];
            if(c.arity >= 2)
            {
                r = binaryRef(opAB, t[0], t[1]);
            }
            if(c.arity == 3)
            {
                r = binaryRef(opABC, r, t[2]);
            }
            auto off     = D.offset(index);
            want[off]    = r;
            written[off] = true;
        });
        compare(got, want, bound, written, true, "tiled elementwise " + sel.pick);
        return sel.pick;
    }

    // Planes smaller than half a tile (16 x 16, 8 x 24): the folded tiled kernel (several planes
    // of consecutive batch indices per tile, a batch count that is not a multiple of them), with
    // aligned and unaligned buffers and an input whose scale is zero; exact.
    TEST(RockeTiledElementwise, FoldedSmallPlanes)
    {
        struct Row
        {
            int                 arity;
            hiptensorDataType_t in, out, scalar;
        };
        Row const rows[] = {{1, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                            {1, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                            {2, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                            {2, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F},
                            {3, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F}};
        int       seed   = 3100;
        for(auto const& r : rows)
        {
            for(auto [ea, eb] : {std::pair<int64_t, int64_t>{16, 16}, {8, 24}})
            {
                for(size_t skew : {size_t(0), size_t(1)})
                {
                    TiledCase c{r.arity, r.in, r.out, r.scalar, seed++, 0, skew};
                    c.ea      = ea;
                    c.eb      = eb;
                    c.ec      = 7;
                    auto pick = runTiled(c);
                    if(skew == 0)
                    {
                        EXPECT_EQ(pick.rfind("ewt:", 0), 0u) << pick;
                        EXPECT_EQ(pick.substr(pick.size() - 2), "_f") << pick;
                    }
                }
                TiledCase z{r.arity, r.in, r.out, r.scalar, seed++, 0, 0, r.arity - 1};
                z.ea = ea;
                z.eb = eb;
                z.ec = 7;
                runTiled(z);
            }
        }
    }

    // The tiled kernel (KERNEL_RANK 0) and the generic one (1) compute the same exact results for
    // every data type row, with aligned and unaligned buffers (the tiled program's fallback) and
    // with an input whose scale is zero.
    TEST(RockeTiledElementwise, EveryCandidateExact)
    {
        struct Row
        {
            int                 arity;
            hiptensorDataType_t in, out, scalar;
        };
        Row const rows[] = {{1, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                            {1, HIPTENSOR_R_16F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                            {1, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                            {1, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F},
                            {2, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_32F},
                            {2, HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_R_32F},
                            {2, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                            {2, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F},
                            {3, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                            {3, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                            {3, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F}};
        int       seed   = 2900;
        for(auto const& r : rows)
        {
            for(size_t skew : {size_t(0), size_t(1)})
            {
                // the tiled kernel, then (when another input is read along D's fastest mode) the
                // vector class of the generic kernel, then the generic kernel itself
                for(int32_t rank = 0; rank < 3; rank++)
                {
                    TiledCase c{r.arity, r.in, r.out, r.scalar, seed, rank, skew};
                    auto      pick = runTiled(c);
                    if(rank == 0 && skew == 0)
                    {
                        EXPECT_EQ(pick.rfind("ewt:", 0), 0u) << pick;
                    }
                    if(rank > 0 && pick.rfind("ewv:", 0) != 0)
                    {
                        EXPECT_EQ(pick.rfind("ew:", 0), 0u) << pick;
                        break;
                    }
                    EXPECT_TRUE(rank < 2) << pick;
                }
                seed++;
            }
            for(int z = 0; z < r.arity; z++)
            {
                TiledCase c{r.arity, r.in, r.out, r.scalar, seed++, 0, 0, z};
                runTiled(c);
            }
            // input 0 laid out as D: the vector class (rank 0) and the generic kernel (1)
            for(int32_t rank : {0, 1})
            {
                for(size_t skew : {size_t(0), size_t(1)})
                {
                    TiledCase c{r.arity, r.in, r.out, r.scalar, seed++, rank, skew, -1, false};
                    auto      pick = runTiled(c);
                    if(rank == 0 && skew == 0)
                    {
                        // a 16-bit permutation laid out as D is one contiguous run: the flat
                        // kernel on gfx942 / gfx950
                        bool const flat = r.arity == 1 && r.in == r.out
                                          && (r.in == HIPTENSOR_R_16F || r.in == HIPTENSOR_R_16BF)
                                          && (deviceArch() == "gfx942" || deviceArch() == "gfx950");
                        EXPECT_EQ(pick.rfind(flat ? "ewf:" : "ewv:", 0), 0u) << pick;
                    }
                }
            }
            ASSERT_FALSE(HasFailure()) << "arity " << r.arity << " type " << int(r.in);
        }
    }

    // Every tensor laid out as D (one contiguous run of elements in the same order): on gfx942 and
    // gfx950 for 16-bit rows the flat kernel (KERNEL_RANK 0; kernels.common.hiptensor_aot.ewf),
    // then the vector class of the generic kernel and the generic kernel, all exact for every data
    // type row (other rows: the vector class first), with aligned and unaligned buffers (the flat program's fallback), with an input whose
    // scale is zero (its NaN is not read), with an element count whose last vectors leave part of
    // the last block idle, and with HIPTENSOR_ROCKE_DISABLE_TUNED (no flat kernel).
    TEST(RockeFlatElementwise, EveryRowExact)
    {
        struct Row
        {
            int                 arity;
            hiptensorDataType_t in, out, scalar;
        };
        Row const  rows[] = {{1, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                             {1, HIPTENSOR_R_16F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                             {1, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                             {1, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F},
                             {2, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_32F},
                             {2, HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_R_32F},
                             {2, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                             {2, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F},
                             {3, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                             {3, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                             {3, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F}};
        bool const arch16 = deviceArch() == "gfx942" || deviceArch() == "gfx950";
        int        seed   = 3300;
        for(auto const& r : rows)
        {
            bool const flatTarget
                = arch16 && r.in == r.out && (r.in == HIPTENSOR_R_16F || r.in == HIPTENSOR_R_16BF);
            for(auto ec : {int64_t(3), int64_t(1)})
            {
                for(size_t skew : {size_t(0), size_t(1)})
                {
                    // the flat kernel (flat targets), the vector class, the generic kernel
                    int32_t const ranks = flatTarget ? 3 : 2;
                    for(int32_t rank = 0; rank < ranks; rank++)
                    {
                        TiledCase c{r.arity, r.in, r.out, r.scalar, seed++, rank, skew};
                        c.ec      = ec; // 72 x 40 x 1 x 5: 14400 elements
                        c.flat    = true;
                        auto pick = runTiled(c);
                        if(pick.empty())
                        {
                            break; // data type not supported on this device
                        }
                        char const* want = rank + (flatTarget ? 0 : 1) == 0   ? "ewf:"
                                           : rank + (flatTarget ? 0 : 1) == 1 ? "ewv:"
                                                                              : "ew:";
                        EXPECT_EQ(pick.rfind(want, 0), 0u) << pick;
                    }
                }
            }
            for(int z = 0; z < r.arity; z++)
            {
                TiledCase c{r.arity, r.in, r.out, r.scalar, seed++, 0, 0, z};
                c.flat    = true;
                auto pick = runTiled(c);
                if(flatTarget && !pick.empty())
                {
                    EXPECT_EQ(pick.rfind("ewf:", 0), 0u) << pick;
                }
            }
            {
                ScopedEnv off("HIPTENSOR_ROCKE_DISABLE_TUNED", "1");
                TiledCase c{r.arity, r.in, r.out, r.scalar, seed++};
                c.flat    = true;
                auto pick = runTiled(c);
                if(!pick.empty())
                {
                    EXPECT_EQ(pick.rfind("ewv:", 0), 0u) << pick;
                }
            }
            ASSERT_FALSE(HasFailure()) << "arity " << r.arity << " type " << int(r.in);
        }
    }
} // namespace
