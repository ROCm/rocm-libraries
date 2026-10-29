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

// The rocKE backend's JIT and kernel cache through the public API (HIPTENSOR_BACKEND=rocke, no
// fallback, a build with HIPTENSOR_ENABLE_ROCKE_JIT):
//  - HIPTENSOR_JIT_MODE_NONE never compiles; HIPTENSOR_JIT_MODE_DEFAULT specializes the
//    elementwise launches of permutations, elementwise operations and contractions;
//  - every result is compared bit for bit with a plain loop reference in double precision
//    (integer-valued data, exact in the compute and output types), including buffers that are
//    not aligned for the specialized kernels' vector accesses;
//  - a second plan of a problem takes the kernel cache, concurrent plans compile a kernel once;
//  - the JIT needs no file besides the kernel pack (it builds with rocKE's C++ kernel builders);
//  - the launch kinds of every pack row are specialized (math rows with gathered inputs and their
//    vector classes, mixed-type contraction epilogues) and match the pack kernels bit for bit;
//  - hiptensorWriteKernelCacheToFile / hiptensorReadKernelCacheFromFile: a fresh process that may
//    not compile serves the plans from the file; files of another format version, device class
//    or kernel ABI are refused with HIPTENSOR_STATUS_INVALID_VALUE, unreadable, damaged or
//    untrusted ones with HIPTENSOR_STATUS_IO_ERROR, damaged entries are skipped, an empty file
//    reads as nothing; spec keys of another key scheme are dropped (the entries are served by
//    identity, and written again once used).
// RockeJitLatency.DISABLED_Report prints plan-creation and kernel times (JIT and pack kernels).

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "api_test_helpers.hpp"

namespace
{
    using hiptensor_test::ApiObjects;
    using hiptensor_test::ScopedEnvironmentVariable;

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
        void clear()
        {
            std::scoped_lock lock(sMutex);
            sLines.clear();
        }
        std::vector<std::string> lines() const
        {
            std::scoped_lock lock(sMutex);
            return sLines;
        }
        bool contains(std::string const& text) const
        {
            auto all = lines();
            return std::any_of(all.begin(), all.end(), [&text](auto const& line) {
                return line.find(text) != std::string::npos;
            });
        }
        int count(std::string const& text) const
        {
            auto all = lines();
            return static_cast<int>(std::count_if(all.begin(), all.end(), [&text](auto const& l) {
                return l.find(text) != std::string::npos;
            }));
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
        static inline std::mutex               sMutex;
        static inline std::vector<std::string> sLines;
    };

    // The numbers of the last "rocKE JIT: <s> of <n> launches specialized (compiled <c>, kernel
    // cache hits <h>, of which from a file <f>)" line.
    struct JitLine
    {
        bool   found       = false;
        int    specialized = 0, launches = 0, compiled = 0, hits = 0, fileHits = 0;
        double ms = 0.0; // time the plan spent in the JIT
    };

    JitLine jitOf(LogCapture const& log)
    {
        JitLine out;
        for(auto const& l : log.lines())
        {
            auto at = l.find("rocKE JIT: ");
            if(at == std::string::npos)
            {
                continue;
            }
            JitLine j;
            if(std::sscanf(l.c_str() + at,
                           "rocKE JIT: %d of %d launches specialized (compiled %d, kernel cache "
                           "hits %d, of which from a file %d) in %lf ms",
                           &j.specialized,
                           &j.launches,
                           &j.compiled,
                           &j.hits,
                           &j.fileHits,
                           &j.ms)
               == 6)
            {
                j.found = true;
                out     = j;
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ data

    enum class Type
    {
        F16,
        F32,
    };

    hiptensorDataType_t dataType(Type t)
    {
        return t == Type::F16 ? HIPTENSOR_R_16F : HIPTENSOR_R_32F;
    }

    size_t bytesOf(Type t)
    {
        return t == Type::F16 ? 2 : 4;
    }

    // A device tensor of integer values, `offset` elements past an aligned allocation (to test
    // misaligned buffers).
    struct Tensor
    {
        Type                 type;
        std::vector<int64_t> lens;
        std::vector<double>  values;
        size_t               offset = 0;
        void*                base   = nullptr;

        Tensor(Type t, std::vector<int64_t> l, size_t off = 0)
            : type(t)
            , lens(std::move(l))
            , values(hiptensor_test::elementCount(lens), 0.0)
            , offset(off)
        {
            EXPECT_EQ(hipMalloc(&base, (values.size() + offset + 8) * bytesOf(type)), hipSuccess);
        }
        ~Tensor()
        {
            static_cast<void>(hipFree(base));
        }
        Tensor(Tensor const&)            = delete;
        Tensor& operator=(Tensor const&) = delete;

        void* data() const
        {
            return static_cast<uint8_t*>(base) + offset * bytesOf(type);
        }

        void fill(std::mt19937& rng, int lo, int hi)
        {
            std::uniform_int_distribution<int> dist(lo, hi);
            for(auto& v : values)
            {
                v = dist(rng);
            }
            upload();
        }

        void upload()
        {
            std::vector<uint8_t> host(values.size() * bytesOf(type));
            for(size_t i = 0; i < values.size(); i++)
            {
                if(type == Type::F16)
                {
                    _Float16 h = static_cast<_Float16>(values[i]);
                    std::memcpy(host.data() + 2 * i, &h, 2);
                }
                else
                {
                    float f = static_cast<float>(values[i]);
                    std::memcpy(host.data() + 4 * i, &f, 4);
                }
            }
            EXPECT_EQ(hipMemcpy(data(), host.data(), host.size(), hipMemcpyHostToDevice),
                      hipSuccess);
        }

        // Fills the device buffer with NaN (unwritten outputs show up).
        void poison()
        {
            EXPECT_EQ(hipMemset(data(), 0xFF, values.size() * bytesOf(type)), hipSuccess);
        }

        std::vector<double> download() const
        {
            std::vector<uint8_t> host(values.size() * bytesOf(type));
            EXPECT_EQ(hipMemcpy(host.data(), data(), host.size(), hipMemcpyDeviceToHost),
                      hipSuccess);
            std::vector<double> out(values.size());
            for(size_t i = 0; i < values.size(); i++)
            {
                if(type == Type::F16)
                {
                    _Float16 h;
                    std::memcpy(&h, host.data() + 2 * i, 2);
                    out[i] = static_cast<double>(h);
                }
                else
                {
                    float f;
                    std::memcpy(&f, host.data() + 4 * i, 4);
                    out[i] = f;
                }
            }
            return out;
        }
    };

    // Packed column-major index of a coordinate given in the tensor's own mode order.
    int64_t linear(std::vector<int64_t> const& lens, std::vector<int64_t> const& coord)
    {
        int64_t idx = 0, stride = 1;
        for(size_t i = 0; i < lens.size(); i++)
        {
            idx += coord[i] * stride;
            stride *= lens[i];
        }
        return idx;
    }

    // Visits every coordinate of `lens` (mode 0 fastest).
    template <typename F>
    void forEach(std::vector<int64_t> const& lens, F&& f)
    {
        std::vector<int64_t> c(lens.size(), 0);
        int64_t const        n = hiptensor_test::elementCount(lens);
        for(int64_t i = 0; i < n; i++)
        {
            f(c);
            for(size_t k = 0; k < c.size() && ++c[k] == lens[k]; k++)
            {
                c[k] = 0;
            }
        }
    }

    void expectExact(Tensor const& got, std::vector<double> const& want, std::string const& what)
    {
        auto    values = got.download();
        int64_t bad    = 0;
        for(size_t i = 0; i < want.size(); i++)
        {
            if(values[i] != want[i] && bad++ < 5)
            {
                ADD_FAILURE() << what << ": element " << i << " is " << values[i] << ", want "
                              << want[i];
            }
        }
        EXPECT_EQ(bad, 0) << what;
    }

    // `rank` >= 0: HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK (the candidate at that place of the
    // selection order)
    hiptensorPlan_t makePlan(ApiObjects&                    api,
                             hiptensorOperationDescriptor_t desc,
                             hiptensorJitMode_t             jit,
                             uint64_t*                      workspace = nullptr,
                             int32_t                        rank      = -1)
    {
        hiptensorPlanPreference_t pref = nullptr;
        EXPECT_EQ(hiptensorCreatePlanPreference(api.handle, &pref, HIPTENSOR_ALGO_DEFAULT, jit),
                  HIPTENSOR_STATUS_SUCCESS);
        if(rank >= 0)
        {
            EXPECT_EQ(
                hiptensorPlanPreferenceSetAttribute(
                    api.handle, pref, HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK, &rank, sizeof(rank)),
                HIPTENSOR_STATUS_SUCCESS);
        }
        uint64_t ws = 0;
        EXPECT_EQ(hiptensorEstimateWorkspaceSize(
                      api.handle, desc, pref, HIPTENSOR_WORKSPACE_DEFAULT, &ws),
                  HIPTENSOR_STATUS_SUCCESS);
        hiptensorPlan_t plan   = nullptr;
        auto            status = hiptensorCreatePlan(api.handle, &plan, desc, pref, ws);
        static_cast<void>(hiptensorDestroyPlanPreference(pref));
        EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS);
        if(plan != nullptr)
        {
            api.plans.push_back(plan);
            uint64_t required = 0;
            EXPECT_EQ(hiptensorPlanGetAttribute(api.handle,
                                                plan,
                                                HIPTENSOR_PLAN_REQUIRED_WORKSPACE,
                                                &required,
                                                sizeof(required)),
                      HIPTENSOR_STATUS_SUCCESS);
            if(workspace != nullptr)
            {
                *workspace = required;
            }
        }
        return plan;
    }

    std::vector<int32_t> iota32(size_t n)
    {
        std::vector<int32_t> m(n);
        std::iota(m.begin(), m.end(), 0);
        return m;
    }

    // ------------------------------------------------------------------ problems

    // One elementwise problem: D[modes 0..r) = opAC(alpha * A, gamma * C) (binary), alpha * A
    // (permutation) or opABC(opAB(alpha * A, beta * B), gamma * C) (trinary); every input with
    // its own mode order.
    struct Case
    {
        std::string                       name;
        int                               arity = 2;
        std::vector<int64_t>              lens; // of D, in D's mode order 0..r-1
        std::vector<std::vector<int32_t>> inputModes; // per input
        Type                              in   = Type::F32;
        Type                              out  = Type::F32;
        hiptensorOperator_t               op   = HIPTENSOR_OP_ADD;
        size_t                            skew = 0; // element offset of every buffer
    };

    std::vector<int64_t> lensOf(Case const& c, std::vector<int32_t> const& modes)
    {
        std::vector<int64_t> l;
        for(auto m : modes)
        {
            l.push_back(c.lens[m]);
        }
        return l;
    }

    double applyOp(hiptensorOperator_t op, double a, double b)
    {
        switch(op)
        {
        case HIPTENSOR_OP_MUL:
            return a * b;
        case HIPTENSOR_OP_MAX:
            return std::max(a, b);
        case HIPTENSOR_OP_MIN:
            return std::min(a, b);
        default:
            return a + b;
        }
    }

    // Plans the case with `jit`, runs it and checks the result. Returns the plan's JIT line.
    JitLine runCase(Case const& c, hiptensorJitMode_t jit, std::mt19937& rng)
    {
        ApiObjects                               api;
        LogCapture                               log;
        std::vector<std::unique_ptr<Tensor>>     inputs;
        std::vector<hiptensorTensorDescriptor_t> descs;
        for(auto const& modes : c.inputModes)
        {
            inputs.push_back(std::make_unique<Tensor>(c.in, lensOf(c, modes), c.skew));
            inputs.back()->fill(rng, -8, 8);
            descs.push_back(api.tensor(inputs.back()->lens, {}, dataType(c.in)));
        }
        Tensor d(c.out, c.lens, c.skew);
        d.poison();
        auto                           dDesc   = api.tensor(c.lens, {}, dataType(c.out));
        auto const                     dModes  = iota32(c.lens.size());
        hiptensorOperationDescriptor_t op      = nullptr;
        auto const                     compute = HIPTENSOR_COMPUTE_DESC_32F;
        if(c.arity == 1)
        {
            EXPECT_EQ(hiptensorCreatePermutation(api.handle,
                                                 &op,
                                                 descs[0],
                                                 c.inputModes[0].data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 dDesc,
                                                 dModes.data(),
                                                 compute),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        else if(c.arity == 2)
        {
            EXPECT_EQ(hiptensorCreateElementwiseBinary(api.handle,
                                                       &op,
                                                       descs[0],
                                                       c.inputModes[0].data(),
                                                       HIPTENSOR_OP_IDENTITY,
                                                       descs[1],
                                                       c.inputModes[1].data(),
                                                       HIPTENSOR_OP_IDENTITY,
                                                       dDesc,
                                                       dModes.data(),
                                                       c.op,
                                                       compute),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        else
        {
            EXPECT_EQ(hiptensorCreateElementwiseTrinary(api.handle,
                                                        &op,
                                                        descs[0],
                                                        c.inputModes[0].data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descs[1],
                                                        c.inputModes[1].data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descs[2],
                                                        c.inputModes[2].data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        dDesc,
                                                        dModes.data(),
                                                        c.op,
                                                        HIPTENSOR_OP_ADD,
                                                        compute),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        api.keep(op);
        auto plan = makePlan(api, op, jit);
        if(plan == nullptr)
        {
            ADD_FAILURE() << c.name << ": no plan\n" << log.all();
            return {};
        }
        float const alpha = 2.0f, beta = -1.0f, gamma = -3.0f;
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), hipSuccess);
        hiptensorStatus_t status;
        if(c.arity == 1)
        {
            status
                = hiptensorPermute(api.handle, plan, &alpha, inputs[0]->data(), d.data(), stream);
        }
        else if(c.arity == 2)
        {
            status = hiptensorElementwiseBinaryExecute(api.handle,
                                                       plan,
                                                       &alpha,
                                                       inputs[0]->data(),
                                                       &gamma,
                                                       inputs[1]->data(),
                                                       d.data(),
                                                       stream);
        }
        else
        {
            status = hiptensorElementwiseTrinaryExecute(api.handle,
                                                        plan,
                                                        &alpha,
                                                        inputs[0]->data(),
                                                        &beta,
                                                        inputs[1]->data(),
                                                        &gamma,
                                                        inputs[2]->data(),
                                                        d.data(),
                                                        stream);
        }
        EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << c.name;
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);

        std::vector<double> want(d.values.size());
        forEach(c.lens, [&](std::vector<int64_t> const& coord) {
            std::vector<double> v;
            for(size_t t = 0; t < inputs.size(); t++)
            {
                std::vector<int64_t> ic;
                for(auto m : c.inputModes[t])
                {
                    ic.push_back(coord[m]);
                }
                v.push_back(inputs[t]->values[linear(inputs[t]->lens, ic)]);
            }
            double r;
            if(c.arity == 1)
            {
                r = alpha * v[0];
            }
            else if(c.arity == 2)
            {
                r = applyOp(c.op, alpha * v[0], gamma * v[1]);
            }
            else
            {
                r = applyOp(c.op, alpha * v[0], beta * v[1]) + gamma * v[2];
            }
            want[linear(c.lens, coord)] = r;
        });
        expectExact(d, want, c.name);
        EXPECT_TRUE(log.contains("Backend: rocke")) << log.all();
        if(jit == HIPTENSOR_JIT_MODE_NONE)
        {
            EXPECT_FALSE(log.contains("rocKE JIT")) << c.name << "\n" << log.all();
        }
        return jitOf(log);
    }

    // f16 contraction with f32 compute: D[m0, n0, m1] = alpha * A[k0, m0, k1, m1] B[n0, k1, k0]
    // + beta * C (integer data, exact in f16). `rank`: as makePlan.
    JitLine runContraction(hiptensorJitMode_t jit,
                           std::mt19937&      rng,
                           size_t             skew = 0,
                           int32_t            rank = -1)
    {
        ApiObjects api;
        LogCapture log;
        // m0, n0, m1, k0, k1
        std::vector<int64_t> const ext = {24, 40, 6, 5, 3};
        std::vector<int32_t> const ma = {3, 0, 4, 2}, mb = {1, 4, 3}, md = {0, 1, 2};
        auto                       lensOfModes = [&](std::vector<int32_t> const& m) {
            std::vector<int64_t> l;
            for(auto x : m)
            {
                l.push_back(ext[x]);
            }
            return l;
        };
        Tensor a(Type::F16, lensOfModes(ma), skew), b(Type::F16, lensOfModes(mb), skew);
        Tensor c(Type::F16, lensOfModes(md), skew), d(Type::F16, lensOfModes(md), skew);
        a.fill(rng, -3, 3);
        b.fill(rng, -3, 3);
        c.fill(rng, -8, 8);
        d.poison();
        auto                           da = api.tensor(a.lens, {}, HIPTENSOR_R_16F);
        auto                           db = api.tensor(b.lens, {}, HIPTENSOR_R_16F);
        auto                           dc = api.tensor(c.lens, {}, HIPTENSOR_R_16F);
        auto                           dd = api.tensor(d.lens, {}, HIPTENSOR_R_16F);
        hiptensorOperationDescriptor_t op = nullptr;
        EXPECT_EQ(hiptensorCreateContraction(api.handle,
                                             &op,
                                             da,
                                             ma.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             db,
                                             mb.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dc,
                                             md.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dd,
                                             md.data(),
                                             HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        uint64_t ws   = 0;
        auto     plan = makePlan(api, op, jit, &ws, rank);
        if(plan == nullptr)
        {
            ADD_FAILURE() << "contraction: no plan\n" << log.all();
            return {};
        }
        void* work = nullptr;
        if(ws > 0)
        {
            EXPECT_EQ(hipMalloc(&work, ws), hipSuccess);
        }
        float const alpha = 2.0f, beta = -1.0f;
        EXPECT_EQ(hiptensorContract(api.handle,
                                    plan,
                                    &alpha,
                                    a.data(),
                                    b.data(),
                                    &beta,
                                    c.data(),
                                    d.data(),
                                    work,
                                    ws,
                                    nullptr),
                  HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        if(work != nullptr)
        {
            static_cast<void>(hipFree(work));
        }
        std::vector<double> want(d.values.size());
        forEach({ext[0], ext[1], ext[2]}, [&](std::vector<int64_t> const& mc) {
            double s = 0.0;
            for(int64_t k0 = 0; k0 < ext[3]; k0++)
            {
                for(int64_t k1 = 0; k1 < ext[4]; k1++)
                {
                    s += a.values[linear(a.lens, {k0, mc[0], k1, mc[2]})]
                         * b.values[linear(b.lens, {mc[1], k1, k0})];
                }
            }
            auto at  = linear(d.lens, mc);
            want[at] = alpha * s + beta * c.values[at];
        });
        expectExact(d, want, "contraction");
        if(jit == HIPTENSOR_JIT_MODE_NONE)
        {
            EXPECT_FALSE(log.contains("rocKE JIT")) << log.all();
        }
        return jitOf(log);
    }

    // The cases: vector classes (all inputs packed like D), gathered inputs (permuted modes),
    // every arity, f16 and f32, mixed types, a fully transposing permute (a vector form with
    // gathered loads only, not promoted by default) and a layout without a vector form.
    std::vector<Case> cases()
    {
        std::vector<Case> out;
        out.push_back(
            {"binary_f16_packed", 2, {64, 48, 5}, {{0, 1, 2}, {0, 1, 2}}, Type::F16, Type::F16});
        out.push_back({"binary_f32_gather",
                       2,
                       {32, 24, 7},
                       {{0, 1, 2}, {2, 0, 1}},
                       Type::F32,
                       Type::F32,
                       HIPTENSOR_OP_MAX});
        out.push_back({"permute_f16_to_f32", 1, {16, 33}, {{0, 1}}, Type::F16, Type::F32});
        // the input's fastest mode moves: a vector form with gathered loads only, which is not
        // promoted by default
        out.push_back(
            {"permute_f32_transposing", 1, {48, 40, 3}, {{2, 1, 0}}, Type::F32, Type::F32});
        out.push_back(
            {"permute_f16_inner_kept", 1, {64, 9, 11}, {{0, 2, 1}}, Type::F16, Type::F16});
        // B is gathered (fastest along mode 1, an extent that is no multiple of the vector
        // width: the tiled transposing kernel does not take it, the JIT's vector form does)
        out.push_back({"trinary_f32",
                       3,
                       {24, 21, 3},
                       {{0, 1, 2}, {1, 0, 2}, {0, 1, 2}},
                       Type::F32,
                       Type::F32,
                       HIPTENSOR_OP_MIN});
        // D's fastest extent 13 is no multiple of any vector width: no vector form
        out.push_back(
            {"binary_f32_odd", 2, {13, 17, 3}, {{0, 1, 2}, {0, 1, 2}}, Type::F32, Type::F32});
        return out;
    }
} // namespace

// Without HIPTENSOR_JIT_MODE_DEFAULT the backend never compiles: no JIT log line, exact results.
TEST(RockeJit, ModeNoneNeverCompiles)
{
    std::mt19937 rng(11);
    for(auto const& c : cases())
    {
        auto j = runCase(c, HIPTENSOR_JIT_MODE_NONE, rng);
        EXPECT_FALSE(j.found) << c.name;
    }
    auto j = runContraction(HIPTENSOR_JIT_MODE_NONE, rng);
    EXPECT_FALSE(j.found);
}

// HIPTENSOR_JIT_MODE_DEFAULT specializes every launch with a vector form and at least one input
// read with vector loads; the results are exact, also with buffers one element off the vector
// alignment (the pack kernel then runs).
TEST(RockeJit, DefaultSpecializesExactly)
{
    std::mt19937 rng(12);
    for(auto skew : {size_t(0), size_t(1)})
    {
        for(auto c : cases())
        {
            c.skew = skew;
            c.name += skew ? "_skew" : "";
            auto j = runCase(c, HIPTENSOR_JIT_MODE_DEFAULT, rng);
            ASSERT_TRUE(j.found) << c.name << ": the library has no rocKE JIT";
            EXPECT_EQ(j.launches, 1) << c.name;
            bool const promoted = c.name.find("odd") == std::string::npos
                                  && c.name.find("transposing") == std::string::npos;
            EXPECT_EQ(j.specialized, promoted ? 1 : 0) << c.name;
        }
        // the first pick (on targets with the direct GETT: that kernel, which has no elementwise
        // launch) is exact; the next candidate packs its operands with elementwise launches,
        // which the JIT specializes
        static_cast<void>(runContraction(HIPTENSOR_JIT_MODE_DEFAULT, rng, skew));
        auto j = runContraction(HIPTENSOR_JIT_MODE_DEFAULT, rng, skew, 1);
        EXPECT_TRUE(j.found);
        EXPECT_GE(j.launches, 1);
    }
}

// HIPTENSOR_ROCKE_JIT=all also specializes launches without a vector form (constant geometry
// only) and vector launches whose inputs are all gathered; HIPTENSOR_ROCKE_JIT=0 turns the JIT
// off.
TEST(RockeJit, PromotionKnobs)
{
    std::mt19937 rng(13);
    for(auto const& c : cases())
    {
        if(c.name.find("odd") == std::string::npos
           && c.name.find("transposing") == std::string::npos)
        {
            continue;
        }
        ScopedEnvironmentVariable all("HIPTENSOR_ROCKE_JIT", "all");
        auto                      j = runCase(c, HIPTENSOR_JIT_MODE_DEFAULT, rng);
        ASSERT_TRUE(j.found) << c.name << ": the library has no rocKE JIT";
        EXPECT_EQ(j.launches, 1) << c.name;
        EXPECT_EQ(j.specialized, 1) << c.name;
    }
    {
        ScopedEnvironmentVariable off("HIPTENSOR_ROCKE_JIT", "0");
        auto                      j = runCase(cases().front(), HIPTENSOR_JIT_MODE_DEFAULT, rng);
        EXPECT_EQ(j.specialized, 0);
    }
}

namespace
{
    // Values for the bit-for-bit comparisons of JIT and kernel-pack launches: integers, halves,
    // signed zeros, NaN, infinities, f32 subnormals, values that overflow f16 and tiny ones.
    void fillSpecial(Tensor& t, std::mt19937& rng)
    {
        static double const                special[] = {std::nan(""),
                                                        HUGE_VAL,
                                                        -HUGE_VAL,
                                                        -0.0,
                                                        0.0,
                                                        0.5,
                                                        -2.75,
                                                        7.0e4,
                                                        -3.0e38,
                                                        1.0e-41,
                                                        1.0e-6,
                                                        0.999};
        std::uniform_int_distribution<int> pick(0, 3 * int(std::size(special)) - 1);
        std::uniform_int_distribution<int> small(-8, 8);
        for(auto& v : t.values)
        {
            int const k = pick(rng);
            v           = k < int(std::size(special)) ? special[k] : small(rng);
        }
        t.upload();
    }

    // Whether the f16 (2 bytes) or f32 (4 bytes) element at `p` is a NaN.
    bool isNanBits(uint8_t const* p, size_t bytes)
    {
        if(bytes == 2)
        {
            uint16_t v;
            std::memcpy(&v, p, 2);
            return (v & 0x7C00u) == 0x7C00u && (v & 0x03FFu) != 0;
        }
        uint32_t v;
        std::memcpy(&v, p, 4);
        return (v & 0x7F800000u) == 0x7F800000u && (v & 0x007FFFFFu) != 0;
    }

    std::vector<uint8_t> tensorBytes(Tensor const& t)
    {
        std::vector<uint8_t> host(t.values.size() * bytesOf(t.type));
        EXPECT_EQ(hipMemcpy(host.data(), t.data(), host.size(), hipMemcpyDeviceToHost), hipSuccess);
        return host;
    }

    // One problem of the launch kinds the JIT gained, planned with `jit` (KERNEL_RANK `rank`),
    // run on special values (the same values for every call of a `kind`); D's bytes, the plan's
    // JIT line and the number of candidates.
    struct WidenedRun
    {
        std::vector<uint8_t> d;
        size_t               elem = 4; // bytes per element of D
        JitLine              jit;
        int                  candidates = 0;
        bool                 planned    = false;
    };

    WidenedRun runWidened(int kind, hiptensorJitMode_t jit, int32_t rank)
    {
        WidenedRun                               out;
        ApiObjects                               api;
        LogCapture                               log;
        std::mt19937                             rng(9100 + kind);
        hiptensorOperationDescriptor_t           op = nullptr;
        std::vector<std::unique_ptr<Tensor>>     in;
        std::vector<hiptensorTensorDescriptor_t> desc;
        auto                                     input = [&](Type t, std::vector<int64_t> lens) {
            in.push_back(std::make_unique<Tensor>(t, std::move(lens)));
            fillSpecial(*in.back(), rng);
            desc.push_back(api.tensor(in.back()->lens, {}, dataType(in.back()->type)));
        };
        std::unique_ptr<Tensor> d;
        if(kind == 0 || kind == 1)
        {
            // permutation with a math operator: kind 0 f32, the input fastest along D's mode 1
            // (gathered: the vec 1 class of the math row); kind 1 f16 packed (its vector class)
            Type const                 t = kind == 0 ? Type::F32 : Type::F16;
            std::vector<int64_t> const lens
                = kind == 0 ? std::vector<int64_t>{33, 20, 3} : std::vector<int64_t>{64, 9, 5};
            std::vector<int32_t> const ma
                = kind == 0 ? std::vector<int32_t>{1, 0, 2} : std::vector<int32_t>{0, 1, 2};
            std::vector<int64_t> la;
            for(auto m : ma)
            {
                la.push_back(lens[m]);
            }
            input(t, la);
            d             = std::make_unique<Tensor>(t, lens);
            auto const md = iota32(3);
            EXPECT_EQ(hiptensorCreatePermutation(api.handle,
                                                 &op,
                                                 desc[0],
                                                 ma.data(),
                                                 kind == 0 ? HIPTENSOR_OP_SIN : HIPTENSOR_OP_COS,
                                                 api.tensor(lens, {}, dataType(t)),
                                                 md.data(),
                                                 HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        else if(kind == 2)
        {
            // binary f32 with math operators, C gathered: the vec 1 class of the math row
            std::vector<int64_t> const lens = {32, 24, 7};
            std::vector<int32_t> const mc   = {2, 0, 1};
            input(Type::F32, lens);
            input(Type::F32, {7, 32, 24});
            d             = std::make_unique<Tensor>(Type::F32, lens);
            auto const md = iota32(3);
            EXPECT_EQ(hiptensorCreateElementwiseBinary(api.handle,
                                                       &op,
                                                       desc[0],
                                                       md.data(),
                                                       HIPTENSOR_OP_ATAN,
                                                       desc[1],
                                                       mc.data(),
                                                       HIPTENSOR_OP_FLOOR,
                                                       api.tensor(lens, {}, HIPTENSOR_R_32F),
                                                       md.data(),
                                                       HIPTENSOR_OP_MAX,
                                                       HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        else
        {
            // contraction of f16 A, B and C into f32 D: its epilogue (f32 result, f16 C, f32 D)
            // is a mixed-type row. D[m0, n0, m1] = A[k0, m0, k1, m1] B[n0, k1, k0] + C
            std::vector<int64_t> const ext = {24, 40, 6, 5, 3};
            std::vector<int32_t> const ma = {3, 0, 4, 2}, mb = {1, 4, 3}, md = {0, 1, 2};
            auto                       lensOf = [&](std::vector<int32_t> const& m) {
                std::vector<int64_t> l;
                for(auto x : m)
                {
                    l.push_back(ext[x]);
                }
                return l;
            };
            input(Type::F16, lensOf(ma));
            input(Type::F16, lensOf(mb));
            input(Type::F16, lensOf(md));
            d = std::make_unique<Tensor>(Type::F32, lensOf(md));
            EXPECT_EQ(hiptensorCreateContraction(api.handle,
                                                 &op,
                                                 desc[0],
                                                 ma.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 desc[1],
                                                 mb.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 desc[2],
                                                 md.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 api.tensor(d->lens, {}, HIPTENSOR_R_32F),
                                                 md.data(),
                                                 HIPTENSOR_COMPUTE_DESC_32F),
                      HIPTENSOR_STATUS_SUCCESS);
        }
        api.keep(op);
        d->poison();
        uint64_t ws   = 0;
        auto     plan = makePlan(api, op, jit, &ws, rank);
        if(plan == nullptr)
        {
            return out;
        }
        out.planned = true;
        for(auto const& l : log.lines())
        {
            auto at = l.find("Candidates: ");
            if(at != std::string::npos)
            {
                out.candidates = std::atoi(l.c_str() + at + 12);
            }
        }
        void* work = nullptr;
        if(ws > 0)
        {
            EXPECT_EQ(hipMalloc(&work, ws), hipSuccess);
        }
        float const       alpha = 1.5f, beta = -1.0f, gamma = 2.0f;
        hiptensorStatus_t status;
        if(kind <= 1)
        {
            status = hiptensorPermute(api.handle, plan, &alpha, in[0]->data(), d->data(), nullptr);
        }
        else if(kind == 2)
        {
            status = hiptensorElementwiseBinaryExecute(
                api.handle, plan, &alpha, in[0]->data(), &gamma, in[1]->data(), d->data(), nullptr);
        }
        else
        {
            status = hiptensorContract(api.handle,
                                       plan,
                                       &alpha,
                                       in[0]->data(),
                                       in[1]->data(),
                                       &beta,
                                       in[2]->data(),
                                       d->data(),
                                       work,
                                       ws,
                                       nullptr);
        }
        EXPECT_EQ(status, HIPTENSOR_STATUS_SUCCESS) << "kind " << kind;
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        if(work != nullptr)
        {
            static_cast<void>(hipFree(work));
        }
        out.d    = tensorBytes(*d);
        out.elem = bytesOf(d->type);
        out.jit  = jitOf(log);
        return out;
    }
} // namespace

// The launch kinds the JIT gained with HIPTENSOR_ROCKE_JIT=all (the math rows with a gathered
// input: their vec 1 class; the math rows' vector classes; the epilogue of a mixed-type
// contraction, every candidate) are specialized, and write the same bits as the kernel-pack
// kernels they replace (a NaN equals a NaN of any sign and payload), on data with NaN,
// infinities, signed zeros, subnormals and values that overflow the output type.
TEST(RockeJit, WidenedClassesMatchThePackKernels)
{
    ScopedEnvironmentVariable all("HIPTENSOR_ROCKE_JIT", "all");
    for(int kind = 0; kind < 4; kind++)
    {
        int  withLaunches = 0;
        int  candidates   = 1;
        auto ranks        = kind == 3 ? 64 : 1;
        for(int32_t rank = 0; rank < std::min(candidates, ranks); rank++)
        {
            auto const pack = runWidened(kind, HIPTENSOR_JIT_MODE_NONE, kind == 3 ? rank : -1);
            auto const jit  = runWidened(kind, HIPTENSOR_JIT_MODE_DEFAULT, kind == 3 ? rank : -1);
            ASSERT_TRUE(pack.planned && jit.planned) << "kind " << kind << " rank " << rank;
            candidates = kind == 3 ? pack.candidates : 1;
            // a candidate whose own results vary between runs (a split reduction in another
            // order) is no reference for a bit-for-bit comparison
            if(kind == 3 && runWidened(kind, HIPTENSOR_JIT_MODE_NONE, rank).d != pack.d)
            {
                continue;
            }
            EXPECT_FALSE(pack.jit.found) << "kind " << kind;
            if(!jit.jit.found)
            {
                EXPECT_EQ(kind, 3) << "kind " << kind << ": no elementwise launch";
                continue;
            }
            withLaunches++;
            EXPECT_GE(jit.jit.launches, 1) << "kind " << kind << " rank " << rank;
            EXPECT_EQ(jit.jit.specialized, jit.jit.launches) << "kind " << kind << " rank " << rank;
            ASSERT_EQ(pack.d.size(), jit.d.size());
            // a NaN equals a NaN: which operand's NaN an operation of two NaN returns (sign and
            // payload) is unspecified, and the compiler may commute the operands
            size_t bad = 0;
            for(size_t i = 0; i + pack.elem <= pack.d.size(); i += pack.elem)
            {
                bool const differ = std::memcmp(&pack.d[i], &jit.d[i], pack.elem) != 0;
                bad += differ
                               && !(isNanBits(&pack.d[i], pack.elem)
                                    && isNanBits(&jit.d[i], pack.elem))
                           ? 1
                           : 0;
            }
            EXPECT_EQ(bad, 0u) << "kind " << kind << " rank " << rank
                               << ": elements that differ between the JIT and the pack kernel";
        }
        EXPECT_GE(withLaunches, 1) << "kind " << kind << ": no plan with an elementwise launch";
    }
}

// A second plan of a problem takes the kernel cache; so does a problem that differs only in the
// values the kernel does not depend on (the scalars, the buffers).
TEST(RockeJit, SecondPlanHitsTheCache)
{
    std::mt19937 rng(14);
    Case c{"binary_f32_cache", 2, {40, 56, 2}, {{0, 1, 2}, {0, 1, 2}}, Type::F32, Type::F32};
    auto first  = runCase(c, HIPTENSOR_JIT_MODE_DEFAULT, rng);
    auto second = runCase(c, HIPTENSOR_JIT_MODE_DEFAULT, rng);
    EXPECT_EQ(first.specialized, 1);
    EXPECT_EQ(second.specialized, 1);
    EXPECT_EQ(second.compiled, 0);
    EXPECT_EQ(second.hits, 1);
}

// Concurrent plan creation of one new problem compiles its kernel once.
TEST(RockeJit, ConcurrentPlansCompileOnce)
{
    LogCapture               log;
    std::vector<std::thread> threads;
    std::vector<int>         ok(8, 0);
    for(int t = 0; t < 8; t++)
    {
        threads.emplace_back([t, &ok] {
            ApiObjects                     api;
            auto                           a  = api.tensor({72, 44, 3}, {}, HIPTENSOR_R_32F);
            auto                           d  = api.tensor({72, 44, 3}, {}, HIPTENSOR_R_32F);
            auto const                     m  = iota32(3);
            hiptensorOperationDescriptor_t op = nullptr;
            if(hiptensorCreatePermutation(api.handle,
                                          &op,
                                          a,
                                          m.data(),
                                          HIPTENSOR_OP_IDENTITY,
                                          d,
                                          m.data(),
                                          HIPTENSOR_COMPUTE_DESC_32F)
               != HIPTENSOR_STATUS_SUCCESS)
            {
                return;
            }
            api.keep(op);
            hiptensorPlanPreference_t pref = nullptr;
            hiptensorCreatePlanPreference(
                api.handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_DEFAULT);
            hiptensorPlan_t plan = nullptr;
            ok[t] = hiptensorCreatePlan(api.handle, &plan, op, pref, 0) == HIPTENSOR_STATUS_SUCCESS;
            hiptensorDestroyPlanPreference(pref);
            if(plan != nullptr)
            {
                api.plans.push_back(plan);
            }
        });
    }
    for(auto& th : threads)
    {
        th.join();
    }
    EXPECT_EQ(std::count(ok.begin(), ok.end(), 1), 8);
    EXPECT_EQ(log.count("rocKE JIT: compiled ew:hiptensor_jit_ew1_r_f32_f32_i32_b256_v4_u"), 1)
        << log.all();
    EXPECT_EQ(log.count("1 of 1 launches specialized"), 8) << log.all();
}

namespace
{
    // ------------------------------------------------------------------ cache files

    std::string tempFile(std::string const& name)
    {
        return ::testing::TempDir() + "hiptensor_rocke_jit_" + name + "_"
#ifndef _WIN32
               + std::to_string(getpid())
#endif
               + ".htk";
    }

    std::vector<uint8_t> readBytes(std::string const& path)
    {
        std::ifstream f(path, std::ios::binary);
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                                    std::istreambuf_iterator<char>());
    }

    void writeBytes(std::string const& path, std::vector<uint8_t> const& bytes)
    {
        std::remove(path.c_str());
        {
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<char const*>(bytes.data()), std::streamsize(bytes.size()));
        }
#ifndef _WIN32
        chmod(path.c_str(), 0600);
#endif
    }

    // SHA-256 (FIPS 180-4), for rewriting the header digest of edited files.
    std::array<uint8_t, 32> sha256(uint8_t const* data, size_t size)
    {
        static constexpr uint32_t k[64]
            = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
               0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
               0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
               0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
               0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
               0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
               0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
               0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
               0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
               0xc67178f2};
        uint32_t             h[8] = {0x6a09e667,
                                     0xbb67ae85,
                                     0x3c6ef372,
                                     0xa54ff53a,
                                     0x510e527f,
                                     0x9b05688c,
                                     0x1f83d9ab,
                                     0x5be0cd19};
        std::vector<uint8_t> m(data, data + size);
        m.push_back(0x80);
        while(m.size() % 64 != 56)
        {
            m.push_back(0);
        }
        uint64_t bits = uint64_t(size) * 8;
        for(int i = 7; i >= 0; i--)
        {
            m.push_back(uint8_t(bits >> (8 * i)));
        }
        auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
        for(size_t off = 0; off < m.size(); off += 64)
        {
            uint32_t w[64];
            for(int i = 0; i < 16; i++)
            {
                w[i] = uint32_t(m[off + 4 * i]) << 24 | uint32_t(m[off + 4 * i + 1]) << 16
                       | uint32_t(m[off + 4 * i + 2]) << 8 | uint32_t(m[off + 4 * i + 3]);
            }
            for(int i = 16; i < 64; i++)
            {
                uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i]        = w[i - 16] + s0 + w[i - 7] + s1;
            }
            uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6],
                     hh = h[7];
            for(int i = 0; i < 64; i++)
            {
                uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
                uint32_t ch = (e & f) ^ (~e & g);
                uint32_t t1 = hh + s1 + ch + k[i] + w[i];
                uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
                uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
                uint32_t t2 = s0 + mj;
                hh          = g;
                g           = f;
                f           = e;
                e           = d + t1;
                d           = c;
                c           = b;
                b           = a;
                a           = t1 + t2;
            }
            h[0] += a;
            h[1] += b;
            h[2] += c;
            h[3] += d;
            h[4] += e;
            h[5] += f;
            h[6] += g;
            h[7] += hh;
        }
        std::array<uint8_t, 32> out;
        for(int i = 0; i < 8; i++)
        {
            for(int j = 0; j < 4; j++)
            {
                out[4 * i + j] = uint8_t(h[i] >> (24 - 8 * j));
            }
        }
        return out;
    }

    // Field offsets of a kernel cache file header (see rocke_jit.cpp).
    struct Header
    {
        size_t minReader = 12, count = 20, arch = 24, cuCount = 0, abi = 0, shaAt = 0;
        bool   ok = false;
    };

    Header parseHeader(std::vector<uint8_t> const& b)
    {
        Header h;
        size_t at  = 24;
        auto   u32 = [&](size_t p) {
            uint32_t v = 0;
            std::memcpy(&v, b.data() + p, 4);
            return v;
        };
        auto str = [&]() {
            if(at + 4 > b.size())
            {
                return false;
            }
            at += 4 + u32(at);
            return at <= b.size();
        };
        if(b.size() < 64 || !str() || !str())
        {
            return h;
        }
        h.cuCount = at;
        at += 4 + 1 + 1;
        h.abi = at;
        at += 4;
        if(!str() || !str())
        {
            return h;
        }
        at += 8;
        if(!str())
        {
            return h;
        }
        h.shaAt = at;
        h.ok    = at + 32 <= b.size();
        return h;
    }

    void rehash(std::vector<uint8_t>& b, Header const& h)
    {
        auto d = sha256(b.data(), h.shaAt);
        std::memcpy(b.data() + h.shaAt, d.data(), 32);
    }

    void put32(std::vector<uint8_t>& b, size_t at, uint32_t v)
    {
        std::memcpy(b.data() + at, &v, 4);
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

    // Rewrites the compiler id of every entry of a kernel cache file (and so its identity and
    // entry checksum), as if another compiler had written the entries.
    std::vector<uint8_t> withCompiler(std::vector<uint8_t> const& in, std::string const& compiler)
    {
        auto const h = parseHeader(in);
        if(!h.ok)
        {
            return {};
        }
        uint32_t count = 0;
        std::memcpy(&count, in.data() + h.count, 4);
        std::vector<uint8_t> out(in.begin(), in.begin() + h.shaAt + 32);
        size_t               at    = h.shaAt + 32;
        auto                 put32 = [](std::vector<uint8_t>& b, uint32_t v) {
            for(int i = 0; i < 4; i++)
            {
                b.push_back(uint8_t(v >> (8 * i)));
            }
        };
        auto putStr = [&](std::vector<uint8_t>& b, std::string const& s) {
            put32(b, uint32_t(s.size()));
            b.insert(b.end(), s.begin(), s.end());
        };
        for(uint32_t e = 0; e < count; e++)
        {
            uint64_t size = 0;
            std::memcpy(&size, in.data() + at, 8);
            size_t p   = at + 8 + 32; // past the size and the entry id
            auto   str = [&](std::string& s) {
                uint32_t n = 0;
                std::memcpy(&n, in.data() + p, 4);
                s.assign(reinterpret_cast<char const*>(in.data() + p + 4), n);
                p += 4 + n;
            };
            std::string isa, oldCompiler;
            str(isa);
            uint8_t const generic = in[p++];
            str(oldCompiler);
            size_t const optionsAt = p;
            uint32_t     nopt      = 0;
            std::memcpy(&nopt, in.data() + p, 4);
            p += 4;
            std::vector<std::string> options(nopt);
            for(auto& o : options)
            {
                str(o);
            }
            uint8_t const fresh = in[p++];
            size_t const  rest  = p; // the LLVM IR digest and the rest of the entry
            // identity: SHA-256 of the length-prefixed (tag, isa, LLVM IR digest, options,
            // fresh-process flag, compiler id)
            std::vector<uint8_t> id;
            putStr(id, "hiptensor-kernel-cache-entry/v1");
            putStr(id, isa);
            id.insert(id.end(), in.begin() + rest, in.begin() + rest + 32);
            id.insert(id.end(), in.begin() + optionsAt, in.begin() + rest - 1);
            id.push_back(fresh);
            putStr(id, compiler);
            auto                 digest = sha256(id.data(), id.size());
            std::vector<uint8_t> body(digest.begin(), digest.end());
            putStr(body, isa);
            body.push_back(generic);
            putStr(body, compiler);
            body.insert(body.end(), in.begin() + optionsAt, in.begin() + rest);
            body.insert(body.end(), in.begin() + rest, in.begin() + at + size - 32);
            std::vector<uint8_t> framed;
            uint64_t const       newSize = 8 + body.size() + 32;
            for(int i = 0; i < 8; i++)
            {
                framed.push_back(uint8_t(newSize >> (8 * i)));
            }
            framed.insert(framed.end(), body.begin(), body.end());
            auto check = sha256(framed.data(), framed.size());
            framed.insert(framed.end(), check.begin(), check.end());
            out.insert(out.end(), framed.begin(), framed.end());
            at += size;
        }
        return out;
    }

    // Replaces `from` by `to` (the same length) in every entry of a kernel cache file and
    // rewrites the entries' checksums: the spec keys of the entries name another key scheme.
    std::vector<uint8_t>
        withKeyText(std::vector<uint8_t> in, std::string const& from, std::string const& to)
    {
        auto const h = parseHeader(in);
        if(!h.ok || from.size() != to.size())
        {
            return {};
        }
        uint32_t count = 0;
        std::memcpy(&count, in.data() + h.count, 4);
        size_t at = h.shaAt + 32;
        for(uint32_t e = 0; e < count; e++)
        {
            uint64_t size = 0;
            std::memcpy(&size, in.data() + at, 8);
            auto const begin = in.begin() + at, end = begin + (size - 32);
            for(auto it = std::search(begin, end, from.begin(), from.end()); it != end;
                it      = std::search(it, end, from.begin(), from.end()))
            {
                it = std::copy(to.begin(), to.end(), it);
            }
            auto check = sha256(in.data() + at, size - 32);
            std::memcpy(in.data() + at + size - 32, check.data(), 32);
            at += size;
        }
        return in;
    }

    uint32_t entryCount(std::vector<uint8_t> const& b)
    {
        auto const h     = parseHeader(b);
        uint32_t   count = 0;
        if(h.ok)
        {
            std::memcpy(&count, b.data() + h.count, 4);
        }
        return count;
    }

    bool contains(std::vector<uint8_t> const& b, std::string const& text)
    {
        return std::search(b.begin(), b.end(), text.begin(), text.end()) != b.end();
    }

    // Problems the file round trip carries (three classes and a contraction).
    std::vector<Case> fileCases()
    {
        return {
            {"file_binary_f16", 2, {96, 40, 3}, {{0, 1, 2}, {0, 1, 2}}, Type::F16, Type::F16},
            {"file_permute_f32", 1, {40, 36, 4}, {{0, 2, 1}}, Type::F32, Type::F32},
            {"file_trinary_f32",
             3,
             {36, 27, 2},
             {{0, 1, 2}, {0, 1, 2}, {1, 0, 2}},
             Type::F32,
             Type::F32},
        };
    }
} // namespace

// Write in this process, read in a fresh one that may not compile (HIPTENSOR_ROCKE_JIT_NO_COMPILE):
// its JIT plans take every kernel from the file and compute exact results.
TEST(RockeJitCacheFile, RoundTripInAFreshProcess)
{
    auto exe = selfExecutable();
    if(exe.empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    std::mt19937 rng(21);
    for(auto const& c : fileCases())
    {
        EXPECT_EQ(runCase(c, HIPTENSOR_JIT_MODE_DEFAULT, rng).specialized, 1) << c.name;
    }
    auto       contraction = runContraction(HIPTENSOR_JIT_MODE_DEFAULT, rng);
    ApiObjects api;
    auto       path = tempFile("roundtrip");
    ASSERT_EQ(hiptensorWriteKernelCacheToFile(api.handle, path.c_str()), HIPTENSOR_STATUS_SUCCESS);
    {
        ScopedEnvironmentVariable file("HIPTENSOR_ROCKE_TEST_CACHE_FILE", path.c_str());
        ScopedEnvironmentVariable nocompile("HIPTENSOR_ROCKE_JIT_NO_COMPILE", "1");
        ScopedEnvironmentVariable expect("HIPTENSOR_ROCKE_TEST_CONTRACTION_JIT",
                                         std::to_string(contraction.specialized).c_str());
        std::string cmd = "\"" + exe + "\" --gtest_filter=RockeJitChild.ServesFromTheFile";
        EXPECT_EQ(std::system(cmd.c_str()), 0);
    }
    std::remove(path.c_str());
}

// The child of RoundTripInAFreshProcess (skipped when run on its own).
TEST(RockeJitChild, ServesFromTheFile)
{
    char const* path = std::getenv("HIPTENSOR_ROCKE_TEST_CACHE_FILE");
    if(path == nullptr)
    {
        GTEST_SKIP() << "run by RockeJitCacheFile.RoundTripInAFreshProcess";
    }
    std::mt19937 rng(22);
    // before the file is read nothing can be specialized (compiling is disabled)
    auto before = runCase(fileCases()[0], HIPTENSOR_JIT_MODE_DEFAULT, rng);
    EXPECT_EQ(before.specialized, 0);
    {
        ApiObjects api;
        LogCapture log;
        ASSERT_EQ(hiptensorReadKernelCacheFromFile(api.handle, path), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_TRUE(log.contains("0 damaged")) << log.all();
    }
    // HIPTENSOR_JIT_MODE_NONE ignores the entries
    EXPECT_FALSE(runCase(fileCases()[1], HIPTENSOR_JIT_MODE_NONE, rng).found);
    for(auto const& c : fileCases())
    {
        auto t0 = std::chrono::steady_clock::now();
        auto j  = runCase(c, HIPTENSOR_JIT_MODE_DEFAULT, rng);
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                      .count();
        EXPECT_EQ(j.specialized, 1) << c.name;
        EXPECT_EQ(j.compiled, 0) << c.name;
        EXPECT_EQ(j.fileHits, 1) << c.name;
        std::printf(
            "JIT-FILE-HIT case=%s jit_ms=%.3f total_case_ms=%.3f\n", c.name.c_str(), j.ms, ms);
    }
    auto contraction = runContraction(HIPTENSOR_JIT_MODE_DEFAULT, rng);
    if(char const* want = std::getenv("HIPTENSOR_ROCKE_TEST_CONTRACTION_JIT"))
    {
        EXPECT_EQ(contraction.specialized, std::atoi(want));
        EXPECT_EQ(contraction.fileHits, contraction.specialized);
    }
}

#ifndef _WIN32
// HIPTENSOR_KERNEL_CACHE_DIR: a compile writes one file per kernel there; a fresh process that may
// not compile and uses the same directory serves its plans from it.
TEST(RockeJitCacheDir, SharedAcrossProcesses)
{
    auto exe = selfExecutable();
    if(exe.empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    std::string dir = ::testing::TempDir() + "hiptensor_rocke_jit_dir_" + std::to_string(getpid());
    ASSERT_EQ(mkdir(dir.c_str(), 0700), 0);
    ScopedEnvironmentVariable cacheDir("HIPTENSOR_KERNEL_CACHE_DIR", dir.c_str());
    std::mt19937              rng(51);
    Case c{"dir_binary_f32", 2, {44, 52, 3}, {{0, 1, 2}, {2, 0, 1}}, Type::F32, Type::F32};
    auto j = runCase(c, HIPTENSOR_JIT_MODE_DEFAULT, rng);
    EXPECT_EQ(j.specialized, 1);
    EXPECT_EQ(j.compiled, 1);
    int files = 0;
    if(auto* d = opendir(dir.c_str()))
    {
        while(auto* e = readdir(d))
        {
            files += std::string(e->d_name).find(".htk") != std::string::npos ? 1 : 0;
        }
        closedir(d);
    }
    EXPECT_EQ(files, 1);
    {
        ScopedEnvironmentVariable nocompile("HIPTENSOR_ROCKE_JIT_NO_COMPILE", "1");
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_CACHE_DIR_CHILD", "1");
        std::string cmd = "\"" + exe + "\" --gtest_filter=RockeJitChild.ServesFromTheDirectory";
        EXPECT_EQ(std::system(cmd.c_str()), 0);
    }
    std::string rm = "rm -rf \"" + dir + "\"";
    EXPECT_EQ(std::system(rm.c_str()), 0);
}
#endif

// The child of RockeJitCacheDir.SharedAcrossProcesses (skipped when run on its own).
TEST(RockeJitChild, ServesFromTheDirectory)
{
    if(std::getenv("HIPTENSOR_ROCKE_TEST_CACHE_DIR_CHILD") == nullptr)
    {
        GTEST_SKIP() << "run by RockeJitCacheDir.SharedAcrossProcesses";
    }
    std::mt19937 rng(52);
    Case         c{"dir_binary_f32", 2, {44, 52, 3}, {{0, 1, 2}, {2, 0, 1}}, Type::F32, Type::F32};
    auto         j = runCase(c, HIPTENSOR_JIT_MODE_DEFAULT, rng);
    EXPECT_EQ(j.specialized, 1);
    EXPECT_EQ(j.compiled, 0);
    EXPECT_EQ(j.fileHits, 1);
}

#ifndef _WIN32
// The JIT builds its kernels with rocKE's C++ kernel builders and needs no file besides the kernel
// pack: a fresh process whose pack directory (HIPTENSOR_ROCKE_PACK_PATH) holds only
// <target>/rocke.htpk specializes its plans and computes exact results.
TEST(RockeJitPack, SpecializesFromThePackAlone)
{
    auto exe = selfExecutable();
    if(exe.empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    // the pack this library loads: a fresh process prints its path
    std::string printed;
    {
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_PRINT_PACK", "1");
        std::string cmd  = "\"" + exe + "\" --gtest_filter=RockeJitChild.PrintsThePack";
        FILE*       pipe = popen(cmd.c_str(), "r");
        ASSERT_NE(pipe, nullptr);
        char buf[4096];
        while(std::fgets(buf, sizeof(buf), pipe) != nullptr)
        {
            printed += buf;
        }
        ASSERT_EQ(pclose(pipe), 0) << printed;
    }
    std::string const tag = "ROCKE-PACK-PATH=";
    auto              at  = printed.find(tag);
    ASSERT_NE(at, std::string::npos) << printed;
    std::string pack = printed.substr(at + tag.size());
    pack             = pack.substr(0, pack.find('\n'));
    // <dir>/<target>/rocke.htpk
    auto const slash  = pack.find_last_of('/');
    auto const slash2 = slash == std::string::npos ? slash : pack.find_last_of('/', slash - 1);
    ASSERT_NE(slash2, std::string::npos) << pack;
    std::string const target = pack.substr(slash2 + 1, slash - slash2 - 1);

    std::string dir = ::testing::TempDir() + "hiptensor_rocke_jit_pack_" + std::to_string(getpid());
    ASSERT_EQ(mkdir(dir.c_str(), 0700), 0);
    ASSERT_EQ(mkdir((dir + "/" + target).c_str(), 0700), 0);
    writeBytes(dir + "/" + target + "/rocke.htpk", readBytes(pack));
    {
        ScopedEnvironmentVariable packPath("HIPTENSOR_ROCKE_PACK_PATH", dir.c_str());
        ScopedEnvironmentVariable child("HIPTENSOR_ROCKE_TEST_PACK_ALONE", "1");
        std::string cmd = "\"" + exe + "\" --gtest_filter=RockeJitChild.SpecializesFromThePack";
        EXPECT_EQ(std::system(cmd.c_str()), 0);
    }
    std::string rm = "rm -rf \"" + dir + "\"";
    EXPECT_EQ(std::system(rm.c_str()), 0);
}
#endif

// The children of RockeJitPack.SpecializesFromThePackAlone (skipped when run on their own).
TEST(RockeJitChild, PrintsThePack)
{
    if(std::getenv("HIPTENSOR_ROCKE_TEST_PRINT_PACK") == nullptr)
    {
        GTEST_SKIP() << "run by RockeJitPack.SpecializesFromThePackAlone";
    }
    LogCapture                     log;
    ApiObjects                     api;
    auto                           a  = api.tensor({8, 8}, {}, HIPTENSOR_R_32F);
    auto                           d  = api.tensor({8, 8}, {}, HIPTENSOR_R_32F);
    auto const                     m  = iota32(2);
    hiptensorOperationDescriptor_t op = nullptr;
    ASSERT_EQ(hiptensorCreatePermutation(api.handle,
                                         &op,
                                         a,
                                         m.data(),
                                         HIPTENSOR_OP_IDENTITY,
                                         d,
                                         m.data(),
                                         HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    api.keep(op);
    ASSERT_NE(makePlan(api, op, HIPTENSOR_JIT_MODE_NONE), nullptr) << log.all();
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

TEST(RockeJitChild, SpecializesFromThePack)
{
    if(std::getenv("HIPTENSOR_ROCKE_TEST_PACK_ALONE") == nullptr)
    {
        GTEST_SKIP() << "run by RockeJitPack.SpecializesFromThePackAlone";
    }
    std::mt19937 rng(61);
    for(auto const& c : cases())
    {
        auto j = runCase(c, HIPTENSOR_JIT_MODE_DEFAULT, rng);
        ASSERT_TRUE(j.found) << c.name;
        bool const promoted = c.name.find("odd") == std::string::npos
                              && c.name.find("transposing") == std::string::npos;
        EXPECT_EQ(j.specialized, promoted ? 1 : 0) << c.name;
        EXPECT_EQ(j.compiled, promoted ? 1 : 0) << c.name;
    }
}

// Entries another compiler wrote are kept but not used: a fresh process that may not compile
// reads such a file and plans without them, unless HIPTENSOR_KERNEL_CACHE_ACCEPT_STALE=1.
TEST(RockeJitCacheFile, OtherCompilerEntriesAreNotUsed)
{
    auto exe = selfExecutable();
    if(exe.empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    std::mt19937 rng(71);
    ASSERT_EQ(runCase(fileCases()[1], HIPTENSOR_JIT_MODE_DEFAULT, rng).specialized, 1);
    ApiObjects api;
    auto       good = tempFile("stale_src");
    ASSERT_EQ(hiptensorWriteKernelCacheToFile(api.handle, good.c_str()), HIPTENSOR_STATUS_SUCCESS);
    auto stale = withCompiler(readBytes(good), "llvm-0.0.0;comgr-0.0");
    ASSERT_FALSE(stale.empty());
    auto path = tempFile("stale");
    writeBytes(path, stale);
    for(char const* accept : {"0", "1"})
    {
        ScopedEnvironmentVariable file("HIPTENSOR_ROCKE_TEST_CACHE_FILE", path.c_str());
        ScopedEnvironmentVariable nocompile("HIPTENSOR_ROCKE_JIT_NO_COMPILE", "1");
        ScopedEnvironmentVariable acceptStale("HIPTENSOR_KERNEL_CACHE_ACCEPT_STALE", accept);
        std::string               cmd = "\"" + exe + "\" --gtest_filter=RockeJitChild.StaleEntries";
        EXPECT_EQ(std::system(cmd.c_str()), 0) << "HIPTENSOR_KERNEL_CACHE_ACCEPT_STALE=" << accept;
    }
    std::remove(good.c_str());
    std::remove(path.c_str());
}

// The child of RockeJitCacheFile.OtherCompilerEntriesAreNotUsed (skipped when run on its own).
TEST(RockeJitChild, StaleEntries)
{
    char const* path   = std::getenv("HIPTENSOR_ROCKE_TEST_CACHE_FILE");
    char const* accept = std::getenv("HIPTENSOR_KERNEL_CACHE_ACCEPT_STALE");
    if(path == nullptr || accept == nullptr)
    {
        GTEST_SKIP() << "run by RockeJitCacheFile.OtherCompilerEntriesAreNotUsed";
    }
    {
        ApiObjects api;
        LogCapture log;
        ASSERT_EQ(hiptensorReadKernelCacheFromFile(api.handle, path), HIPTENSOR_STATUS_SUCCESS);
        // every entry of the file names another compiler
        EXPECT_FALSE(log.contains(", 0 of another compiler")) << log.all();
        EXPECT_TRUE(log.contains(std::string(accept) == "1" ? "(used: " : "(kept, not used;"))
            << log.all();
    }
    std::mt19937 rng(72);
    auto         j = runCase(fileCases()[1], HIPTENSOR_JIT_MODE_DEFAULT, rng);
    EXPECT_EQ(j.specialized, std::string(accept) == "1" ? 1 : 0);
    EXPECT_EQ(j.compiled, 0);
}

// Entries whose spec keys belong to another key scheme (hiptensor-rocke-jit/v1, the recipe JIT's)
// lose those keys when read: a fresh process serves them only by identity (no compile), and
// writing leaves an entry out until a plan has used it.
TEST(RockeJitCacheFile, ForeignSpecKeysAreDropped)
{
    auto exe = selfExecutable();
    if(exe.empty())
    {
        GTEST_SKIP() << "needs /proc/self/exe";
    }
    std::mt19937 rng(73);
    ASSERT_EQ(runCase(fileCases()[1], HIPTENSOR_JIT_MODE_DEFAULT, rng).specialized, 1);
    ApiObjects api;
    auto       good = tempFile("keys_src");
    ASSERT_EQ(hiptensorWriteKernelCacheToFile(api.handle, good.c_str()), HIPTENSOR_STATUS_SUCCESS);
    auto const src = readBytes(good);
    ASSERT_TRUE(contains(src, "hiptensor-rocke-jit/v2|"));
    auto foreign = withKeyText(src, "hiptensor-rocke-jit/v2|", "hiptensor-rocke-jit/v1|");
    ASSERT_FALSE(foreign.empty());
    ASSERT_FALSE(contains(foreign, "hiptensor-rocke-jit/v2|"));
    auto path = tempFile("keys");
    writeBytes(path, foreign);
    {
        ScopedEnvironmentVariable file("HIPTENSOR_ROCKE_TEST_CACHE_FILE", path.c_str());
        ScopedEnvironmentVariable nocompile("HIPTENSOR_ROCKE_JIT_NO_COMPILE", "1");
        std::string cmd = "\"" + exe + "\" --gtest_filter=RockeJitChild.ForeignSpecKeys";
        EXPECT_EQ(std::system(cmd.c_str()), 0);
    }
    std::remove(good.c_str());
    std::remove(path.c_str());
}

// The child of RockeJitCacheFile.ForeignSpecKeysAreDropped (skipped when run on its own).
TEST(RockeJitChild, ForeignSpecKeys)
{
    char const* path = std::getenv("HIPTENSOR_ROCKE_TEST_CACHE_FILE");
    if(path == nullptr || std::getenv("HIPTENSOR_ROCKE_JIT_NO_COMPILE") == nullptr)
    {
        GTEST_SKIP() << "run by RockeJitCacheFile.ForeignSpecKeysAreDropped";
    }
    ApiObjects api;
    ASSERT_EQ(hiptensorReadKernelCacheFromFile(api.handle, path), HIPTENSOR_STATUS_SUCCESS);
    // not used yet: no key of this scheme, so not written
    auto unused = tempFile("keys_unused");
    ASSERT_EQ(hiptensorWriteKernelCacheToFile(api.handle, unused.c_str()),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(entryCount(readBytes(unused)), 0u);
    // served by identity, without a compile
    std::mt19937 rng(73);
    auto         j = runCase(fileCases()[1], HIPTENSOR_JIT_MODE_DEFAULT, rng);
    EXPECT_EQ(j.specialized, 1);
    EXPECT_EQ(j.compiled, 0);
    // used: written with its key of this scheme only
    auto used = tempFile("keys_used");
    ASSERT_EQ(hiptensorWriteKernelCacheToFile(api.handle, used.c_str()), HIPTENSOR_STATUS_SUCCESS);
    auto const bytes = readBytes(used);
    EXPECT_GE(entryCount(bytes), 1u);
    EXPECT_TRUE(contains(bytes, "hiptensor-rocke-jit/v2|"));
    EXPECT_FALSE(contains(bytes, "hiptensor-rocke-jit/v1|"));
    std::remove(unused.c_str());
    std::remove(used.c_str());
}

// Header checks: format version, device class and kernel ABI are INVALID_VALUE; a damaged or
// truncated header, a bad magic, a missing or untrusted file are IO_ERROR; an empty file reads as
// nothing; a damaged entry is skipped.
TEST(RockeJitCacheFile, Rejections)
{
    std::mt19937 rng(31);
    ASSERT_EQ(runCase(fileCases()[0], HIPTENSOR_JIT_MODE_DEFAULT, rng).specialized, 1);
    ApiObjects api;
    auto       good = tempFile("good");
    ASSERT_EQ(hiptensorWriteKernelCacheToFile(api.handle, good.c_str()), HIPTENSOR_STATUS_SUCCESS);
    auto const bytes = readBytes(good);
    auto const h     = parseHeader(bytes);
    ASSERT_TRUE(h.ok);
    uint32_t count = 0;
    std::memcpy(&count, bytes.data() + h.count, 4);
    ASSERT_GE(count, 1u);
    EXPECT_EQ(hiptensorReadKernelCacheFromFile(api.handle, good.c_str()), HIPTENSOR_STATUS_SUCCESS);

    auto check = [&](char const* what, std::vector<uint8_t> const& b, hiptensorStatus_t want) {
        auto path = tempFile(what);
        writeBytes(path, b);
        EXPECT_EQ(hiptensorReadKernelCacheFromFile(api.handle, path.c_str()), want) << what;
        std::remove(path.c_str());
    };
    {
        auto b = bytes;
        put32(b, h.minReader, 99);
        rehash(b, h);
        check("min_reader", b, HIPTENSOR_STATUS_INVALID_VALUE);
    }
    {
        auto b = bytes;
        b[h.arch + 4 + 3] ^= 0x01; // gfx9x0 -> another architecture name
        rehash(b, h);
        check("arch", b, HIPTENSOR_STATUS_INVALID_VALUE);
    }
    {
        auto     b  = bytes;
        uint32_t cu = 0;
        std::memcpy(&cu, b.data() + h.cuCount, 4);
        put32(b, h.cuCount, cu + 1);
        rehash(b, h);
        check("cu_count", b, HIPTENSOR_STATUS_INVALID_VALUE);
    }
    {
        auto b = bytes;
        put32(b, h.abi, 77);
        rehash(b, h);
        check("kernel_abi", b, HIPTENSOR_STATUS_INVALID_VALUE);
    }
    {
        auto b = bytes;
        b[h.shaAt] ^= 0xFF;
        check("header_sha", b, HIPTENSOR_STATUS_IO_ERROR);
    }
    {
        auto b = bytes;
        b[0]   = 'X';
        check("magic", b, HIPTENSOR_STATUS_IO_ERROR);
    }
    check("truncated",
          std::vector<uint8_t>(bytes.begin(), bytes.begin() + 30),
          HIPTENSOR_STATUS_IO_ERROR);
    check("empty", {}, HIPTENSOR_STATUS_SUCCESS);
    {
        LogCapture log;
        auto       b = bytes;
        b[b.size() - 40] ^= 0x5A; // inside the last entry's code object
        check("entry", b, HIPTENSOR_STATUS_SUCCESS);
        EXPECT_TRUE(log.contains("1 damaged (skipped)")) << log.all();
    }
    auto missing = tempFile("missing");
    std::remove(missing.c_str());
    EXPECT_EQ(hiptensorReadKernelCacheFromFile(api.handle, missing.c_str()),
              HIPTENSOR_STATUS_IO_ERROR);
#ifndef _WIN32
    if(getuid() != 0)
    {
        auto open = tempFile("group_writable");
        writeBytes(open, bytes);
        chmod(open.c_str(), 0666);
        EXPECT_EQ(hiptensorReadKernelCacheFromFile(api.handle, open.c_str()),
                  HIPTENSOR_STATUS_IO_ERROR);
        std::remove(open.c_str());
    }
#endif
    EXPECT_EQ(hiptensorWriteKernelCacheToFile(api.handle, "/nonexistent-dir/x.htk"),
              HIPTENSOR_STATUS_IO_ERROR);
    std::remove(good.c_str());
}

// Plan-creation and kernel times of JIT and pack kernels (printed; run on purpose with
// --gtest_also_run_disabled_tests, with AMD_COMGR_CACHE=0 for cold compiles).
TEST(RockeJitLatency, DISABLED_Report)
{
    std::vector<Case> shapes = {
        // footprints of 0.5 to 1 GiB, past the last-level cache
        {"lat_binary_f16", 2, {2048, 1024, 64}, {{0, 1, 2}, {0, 1, 2}}, Type::F16, Type::F16},
        {"lat_binary_f32_gather",
         2,
         {1024, 1024, 64},
         {{0, 1, 2}, {1, 0, 2}},
         Type::F32,
         Type::F32},
        {"lat_permute_f16", 1, {2048, 512, 128}, {{1, 0, 2}}, Type::F16, Type::F16},
        {"lat_permute_f32", 1, {1024, 512, 128}, {{2, 1, 0}}, Type::F32, Type::F32},
        {"lat_permute_f16_inner_kept", 1, {2048, 256, 256}, {{0, 2, 1}}, Type::F16, Type::F16},
        {"lat_trinary_f32",
         3,
         {1024, 512, 64},
         {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}},
         Type::F32,
         Type::F32},
        {"lat_permute_f16_to_f32", 1, {2048, 512, 64}, {{0, 1, 2}}, Type::F16, Type::F32},
    };
    // every launch JIT-compiled, also those the default rule leaves on the pack kernel
    ScopedEnvironmentVariable all("HIPTENSOR_ROCKE_JIT", "all");
    auto                      now = [] { return std::chrono::steady_clock::now(); };
    auto                      ms
        = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    for(auto const& c : shapes)
    {
        ApiObjects                               api;
        LogCapture                               log;
        std::vector<std::unique_ptr<Tensor>>     inputs;
        std::vector<hiptensorTensorDescriptor_t> descs;
        std::mt19937                             rng(41);
        for(auto const& modes : c.inputModes)
        {
            inputs.push_back(std::make_unique<Tensor>(c.in, lensOf(c, modes)));
            inputs.back()->fill(rng, -4, 4);
            descs.push_back(api.tensor(inputs.back()->lens, {}, dataType(c.in)));
        }
        Tensor                         d(c.out, c.lens);
        auto                           dDesc  = api.tensor(c.lens, {}, dataType(c.out));
        auto                           dModes = iota32(c.lens.size());
        hiptensorOperationDescriptor_t op     = nullptr;
        if(c.arity == 1)
        {
            hiptensorCreatePermutation(api.handle,
                                       &op,
                                       descs[0],
                                       c.inputModes[0].data(),
                                       HIPTENSOR_OP_IDENTITY,
                                       dDesc,
                                       dModes.data(),
                                       HIPTENSOR_COMPUTE_DESC_32F);
        }
        else if(c.arity == 2)
        {
            hiptensorCreateElementwiseBinary(api.handle,
                                             &op,
                                             descs[0],
                                             c.inputModes[0].data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             descs[1],
                                             c.inputModes[1].data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             dDesc,
                                             dModes.data(),
                                             HIPTENSOR_OP_ADD,
                                             HIPTENSOR_COMPUTE_DESC_32F);
        }
        else
        {
            hiptensorCreateElementwiseTrinary(api.handle,
                                              &op,
                                              descs[0],
                                              c.inputModes[0].data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              descs[1],
                                              c.inputModes[1].data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              descs[2],
                                              c.inputModes[2].data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              dDesc,
                                              dModes.data(),
                                              HIPTENSOR_OP_ADD,
                                              HIPTENSOR_OP_ADD,
                                              HIPTENSOR_COMPUTE_DESC_32F);
        }
        api.keep(op);
        auto t0    = now();
        auto aot   = makePlan(api, op, HIPTENSOR_JIT_MODE_NONE);
        auto t1    = now();
        auto cold  = makePlan(api, op, HIPTENSOR_JIT_MODE_DEFAULT);
        auto t2    = now();
        auto jCold = jitOf(log);
        log.clear();
        auto        hit  = makePlan(api, op, HIPTENSOR_JIT_MODE_DEFAULT);
        auto        t3   = now();
        auto        jHit = jitOf(log);
        std::string compiled;
        for(auto const& l : log.lines())
        {
            if(l.find("rocKE JIT: compiled") != std::string::npos)
            {
                compiled = l.substr(l.find("rocKE JIT: compiled"));
            }
        }
        std::printf("JIT-LATENCY case=%s plan_aot_ms=%.3f plan_jit_cold_ms=%.3f "
                    "plan_jit_hit_ms=%.3f jit_cold_ms=%.3f jit_hit_ms=%.3f specialized=%d | %s\n",
                    c.name.c_str(),
                    ms(t0, t1),
                    ms(t1, t2),
                    ms(t2, t3),
                    jCold.ms,
                    jHit.ms,
                    jCold.specialized,
                    compiled.c_str());
        // kernel times: median of 21 runs each, interleaved
        float const alpha = 1.0f, beta = 1.0f, gamma = 1.0f;
        auto        run = [&](hiptensorPlan_t plan) {
            if(c.arity == 1)
            {
                return hiptensorPermute(
                    api.handle, plan, &alpha, inputs[0]->data(), d.data(), nullptr);
            }
            if(c.arity == 2)
            {
                return hiptensorElementwiseBinaryExecute(api.handle,
                                                         plan,
                                                         &alpha,
                                                         inputs[0]->data(),
                                                         &gamma,
                                                         inputs[1]->data(),
                                                         d.data(),
                                                         nullptr);
            }
            return hiptensorElementwiseTrinaryExecute(api.handle,
                                                      plan,
                                                      &alpha,
                                                      inputs[0]->data(),
                                                      &beta,
                                                      inputs[1]->data(),
                                                      &gamma,
                                                      inputs[2]->data(),
                                                      d.data(),
                                                      nullptr);
        };
        hipEvent_t e0, e1;
        EXPECT_EQ(hipEventCreate(&e0), hipSuccess);
        EXPECT_EQ(hipEventCreate(&e1), hipSuccess);
        std::vector<float> ta, tj;
        for(int r = 0; r < 21; r++)
        {
            for(int which = 0; which < 2; which++)
            {
                EXPECT_EQ(hipEventRecord(e0, nullptr), hipSuccess);
                run(which == 0 ? aot : cold);
                EXPECT_EQ(hipEventRecord(e1, nullptr), hipSuccess);
                EXPECT_EQ(hipEventSynchronize(e1), hipSuccess);
                float t = 0;
                EXPECT_EQ(hipEventElapsedTime(&t, e0, e1), hipSuccess);
                (which == 0 ? ta : tj).push_back(t);
            }
        }
        static_cast<void>(hipEventDestroy(e0));
        static_cast<void>(hipEventDestroy(e1));
        std::sort(ta.begin(), ta.end());
        std::sort(tj.begin(), tj.end());
        size_t bytes = d.values.size() * bytesOf(c.out);
        for(auto const& in : inputs)
        {
            bytes += in->values.size() * bytesOf(in->type);
        }
        std::printf("JIT-KERNEL case=%s bytes=%zu aot_us=%.2f jit_us=%.2f ratio=%.3f\n",
                    c.name.c_str(),
                    bytes,
                    1000.0 * ta[10],
                    1000.0 * tj[10],
                    ta[10] / tj[10]);
        static_cast<void>(hit);
    }

    // contractions whose operands are packed and whose result is written back by elementwise
    // launches: D[m0, n0, m1] = A[k0, m0, k1, m1] B[n0, k1, k0] + C
    for(auto type : {Type::F16, Type::F32})
    {
        ApiObjects                 api;
        LogCapture                 log;
        std::vector<int64_t> const ext = {64, 1024, 32, 32, 32}; // m0, n0, m1, k0, k1
        std::vector<int32_t> const ma = {3, 0, 4, 2}, mb = {1, 4, 3}, md = {0, 1, 2};
        auto                       lensOfModes = [&](std::vector<int32_t> const& m) {
            std::vector<int64_t> l;
            for(auto x : m)
            {
                l.push_back(ext[x]);
            }
            return l;
        };
        std::mt19937 rng(61);
        Tensor       a(type, lensOfModes(ma)), b(type, lensOfModes(mb)), cc(type, lensOfModes(md)),
            d(type, lensOfModes(md));
        a.fill(rng, -2, 2);
        b.fill(rng, -2, 2);
        cc.fill(rng, -2, 2);
        auto                           dt = dataType(type);
        hiptensorOperationDescriptor_t op = nullptr;
        hiptensorCreateContraction(api.handle,
                                   &op,
                                   api.tensor(a.lens, {}, dt),
                                   ma.data(),
                                   HIPTENSOR_OP_IDENTITY,
                                   api.tensor(b.lens, {}, dt),
                                   mb.data(),
                                   HIPTENSOR_OP_IDENTITY,
                                   api.tensor(cc.lens, {}, dt),
                                   md.data(),
                                   HIPTENSOR_OP_IDENTITY,
                                   api.tensor(d.lens, {}, dt),
                                   md.data(),
                                   HIPTENSOR_COMPUTE_DESC_32F);
        api.keep(op);
        uint64_t wsA = 0, wsJ = 0;
        auto     t0  = now();
        auto     aot = makePlan(api, op, HIPTENSOR_JIT_MODE_NONE, &wsA);
        auto     t1  = now();
        auto     jit = makePlan(api, op, HIPTENSOR_JIT_MODE_DEFAULT, &wsJ);
        auto     t2  = now();
        auto     j   = jitOf(log);
        void*    ws  = nullptr;
        EXPECT_EQ(hipMalloc(&ws, std::max<uint64_t>(std::max(wsA, wsJ), 1)), hipSuccess);
        float const one = 1.0f;
        hipEvent_t  e0, e1;
        EXPECT_EQ(hipEventCreate(&e0), hipSuccess);
        EXPECT_EQ(hipEventCreate(&e1), hipSuccess);
        std::vector<float> ta, tj;
        for(int r = 0; r < 21; r++)
        {
            for(int which = 0; which < 2; which++)
            {
                EXPECT_EQ(hipEventRecord(e0, nullptr), hipSuccess);
                hiptensorContract(api.handle,
                                  which == 0 ? aot : jit,
                                  &one,
                                  a.data(),
                                  b.data(),
                                  &one,
                                  cc.data(),
                                  d.data(),
                                  ws,
                                  which == 0 ? wsA : wsJ,
                                  nullptr);
                EXPECT_EQ(hipEventRecord(e1, nullptr), hipSuccess);
                EXPECT_EQ(hipEventSynchronize(e1), hipSuccess);
                float t = 0;
                EXPECT_EQ(hipEventElapsedTime(&t, e0, e1), hipSuccess);
                (which == 0 ? ta : tj).push_back(t);
            }
        }
        static_cast<void>(hipEventDestroy(e0));
        static_cast<void>(hipEventDestroy(e1));
        static_cast<void>(hipFree(ws));
        std::sort(ta.begin(), ta.end());
        std::sort(tj.begin(), tj.end());
        char const* name = type == Type::F16 ? "f16" : "f32";
        std::printf("JIT-LATENCY case=contraction_%s plan_aot_ms=%.3f plan_jit_cold_ms=%.3f "
                    "launches=%d specialized=%d compiled=%d\n",
                    name,
                    ms(t0, t1),
                    ms(t1, t2),
                    j.launches,
                    j.specialized,
                    j.compiled);
        std::printf("JIT-KERNEL case=contraction_%s aot_us=%.2f jit_us=%.2f ratio=%.3f\n",
                    name,
                    1000.0 * ta[10],
                    1000.0 * tj[10],
                    ta[10] / tj[10]);
    }
}
