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
// of every data type, permutations, elementwise binary and trinary operations and reductions with
// random mode orders, extents and padded strides, executed on a non-blocking stream.
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
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string>
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

    private:
        static void callback(int32_t, char const* func, char const* msg)
        {
            std::scoped_lock lock(sMutex);
            sLines.push_back(std::string(func) + ": " + msg);
        }
        static inline std::mutex               sMutex;
        static inline std::vector<std::string> sLines;
    };

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
        void* upload(Tensor const& t)
        {
            auto  b = t.bytes();
            void* p = nullptr;
            EXPECT_EQ(hipMalloc(&p, std::max<size_t>(b.size(), 16)), hipSuccess);
            EXPECT_EQ(hipMemcpy(p, b.data(), b.size(), hipMemcpyHostToDevice), hipSuccess);
            buffers.push_back(p);
            return p;
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
    };

    // What the rocKE backend's selection log line says about a plan, and its workspace.
    struct SelectionLog
    {
        int         candidates = 0;
        std::string pick;
        std::string source;
        uint64_t    estimate = 0;
        uint64_t    required = 0;
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
        char const* route    = std::getenv("HIPTENSOR_ROCKE_CONTRACTION_ROUTE");
        bool const  f64Route = (route != nullptr && std::string(route) == "f64")
                              || deviceArch() != "gfx950"; // the 16-bit GEMM routes are gfx950's
        if(f64Route && c.type == HIPTENSOR_R_32F && c.compute != HIPTENSOR_COMPUTE_DESC_32F)
        {
            GTEST_SKIP() << "the f64 route declines f32 data with a 16-bit compute type";
        }
        LogCapture log;
        // modes: m0 m1 n0 k0 k1 with random extents, some not multiples of any tile (hipTensor
        // refuses batched modes, shared by A, B and D)
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
        default:
            return x;
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
        bool                  gemm = first.pick.rfind("gemm:", 0) == 0;
        bool                  f64  = first.pick.rfind("f64:", 0) == 0;
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
            gemm = gemm || s.pick.rfind("gemm:", 0) == 0;
            f64  = f64 || s.pick.rfind("f64:", 0) == 0;
        }
        EXPECT_EQ(picks.size(), size_t(first.candidates));
        bool const native16 = c.type == HIPTENSOR_R_32F && c.compute != HIPTENSOR_COMPUTE_DESC_32F;
        bool const real64   = c.type == HIPTENSOR_R_64F || c.type == HIPTENSOR_C_64F;
        if(deviceArch() == "gfx950" && !real64)
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
        EXPECT_TRUE(a.source == "MODEL" || a.source == "STATIC") << a.source;
        if(deviceArch() == "gfx950")
        {
            EXPECT_EQ(a.source, "MODEL");
        }
        SelectionLog st;
        {
            ScopedEnv env("HIPTENSOR_ROCKE_SELECT", "static");
            runContraction(c, all, &st);
        }
        EXPECT_EQ(st.source, "STATIC");
        EXPECT_EQ(st.candidates, a.candidates);

        // reductions have models too
        ReductionCase const r{HIPTENSOR_R_32F, HIPTENSOR_COMPUTE_DESC_32F, HIPTENSOR_OP_ADD, 1502};
        SelectionLog        red;
        runReduction(r, {}, &red);
        EXPECT_TRUE(red.source == "MODEL" || red.source == "STATIC") << red.source;
        if(deviceArch() == "gfx950" && red.candidates > 1)
        {
            EXPECT_EQ(red.source, "MODEL");
        }
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
} // namespace
