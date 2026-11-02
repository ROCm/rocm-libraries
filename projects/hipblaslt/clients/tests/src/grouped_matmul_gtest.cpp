// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Grouped GEMM through hipblasLtMatmul (HIPBLASLT_BATCH_MODE_GROUPED layouts).

#include <gtest/gtest.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace
{
    struct GroupShape
    {
        int64_t m;
        int64_t n;
        int64_t k;
    };

    enum class Scalars
    {
        Host,
        DeviceShared,
        DevicePerGroup
    };

    struct Case
    {
        hipDataType             type = HIP_R_16F;
        hipblasOperation_t      opA  = HIPBLAS_OP_N;
        hipblasOperation_t      opB  = HIPBLAS_OP_N;
        std::vector<GroupShape> groups;
        int32_t                 width = HIPBLASLT_INTEGER_WIDTH_32;
        // D/C rows and the M side of A use the scalar ROWS/COLS attribute (all groups share m),
        // and every ld array is null with the scalar LD left at 0 (packed).
        bool fixedM = false;
        // The K side of A and B uses the scalar ROWS/COLS attribute (all groups share k).
        bool    fixedK        = false;
        Scalars scalars       = Scalars::Host;
        bool    totalColsHint = false;
        bool    totalRowsHint = false;
        // Also set the average rows and cols hints.
        bool averageHints = false;
        // Second call with these shapes (each no larger than in groups); needs !fixedM.
        std::vector<GroupShape> rerun;
    };

    constexpr size_t kWorkspace = size_t{64} << 20;

    float toFloat(hipDataType type, uint16_t v)
    {
        if(type == HIP_R_16F)
        {
            __half h;
            memcpy(&h, &v, sizeof(v));
            return __half2float(h);
        }
        hip_bfloat16 b;
        memcpy(&b, &v, sizeof(v));
        return static_cast<float>(b);
    }

    uint16_t fromFloat(hipDataType type, float f)
    {
        uint16_t v;
        if(type == HIP_R_16F)
        {
            __half h = __float2half(f);
            memcpy(&v, &h, sizeof(v));
        }
        else
        {
            hip_bfloat16 b(f);
            memcpy(&v, &b, sizeof(v));
        }
        return v;
    }

    template <typename T>
    void* upload(const std::vector<T>& host)
    {
        void* dev = nullptr;
        EXPECT_EQ(hipMalloc(&dev, std::max<size_t>(host.size() * sizeof(T), 1)), hipSuccess);
        if(!host.empty())
            EXPECT_EQ(hipMemcpy(dev, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
                      hipSuccess);
        return dev;
    }

    void* uploadDims(const std::vector<int64_t>& v, int32_t width)
    {
        if(width == HIPBLASLT_INTEGER_WIDTH_64)
            return upload(v);
        std::vector<int32_t> narrow(v.begin(), v.end());
        return upload(narrow);
    }

    // One grouped matrix: per-group shape and ld, packed into one buffer.
    struct GroupedMatrix
    {
        std::vector<int64_t> rows, cols, ld;
        std::vector<size_t> offset;
        std::vector<uint16_t> hostData;
        // Device copy of hostData, with the same layout. This is what hipBLASLt reads and writes.
        void* deviceData = nullptr;
        // Host vector of device addresses: ptrs[g] = deviceData + offset[g].
        std::vector<void*> ptrs;
        // Device copy of ptrs. Grouped hipblasLtMatmul takes this as the A/B/C/D argument.
        void* ptrArray = nullptr;

        void init(const std::vector<int64_t>& r,
                  const std::vector<int64_t>& c,
                  bool                        packed,
                  hipDataType                 type,
                  std::mt19937&               rng)
        {
            rows = r;
            cols = c;
            ld.resize(r.size());
            offset.resize(r.size());
            size_t total = 0;
            for(size_t g = 0; g < r.size(); g++)
            {
                ld[g]     = std::max<int64_t>(r[g], 1) + (packed ? 0 : int64_t(g % 3));
                offset[g] = total;
                total += size_t(ld[g]) * size_t(c[g]);
            }
            std::uniform_int_distribution<int> dist(-1, 1);
            hostData.resize(std::max<size_t>(total, 1));
            for(auto& v : hostData)
                v = fromFloat(type, float(dist(rng)));
            deviceData = upload(hostData);
            ptrs.resize(r.size());
            for(size_t g = 0; g < r.size(); g++)
                ptrs[g] = static_cast<uint16_t*>(deviceData) + offset[g];
            ptrArray = upload(ptrs);
        }

        float at(hipDataType type, size_t g, int64_t i, int64_t j) const
        {
            return toFloat(type, hostData[offset[g] + size_t(j * ld[g] + i)]);
        }

        GroupedMatrix()                                = default;
        GroupedMatrix(const GroupedMatrix&)            = delete;
        GroupedMatrix& operator=(const GroupedMatrix&) = delete;
        ~GroupedMatrix()
        {
            static_cast<void>(hipFree(deviceData));
            static_cast<void>(hipFree(ptrArray));
        }
    };

    // Runs f at scope exit, including the early returns of GTEST_SKIP and failed ASSERTs.
    template <typename F>
    struct ScopeExit
    {
        F f;
        ~ScopeExit()
        {
            f();
        }
    };
    template <typename F>
    ScopeExit(F) -> ScopeExit<F>;

    class GroupedMatmul : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t  handle = nullptr;
        std::vector<void*> deviceAllocs;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            for(void* p : deviceAllocs)
                static_cast<void>(hipFree(p));
            if(handle)
                hipblasLtDestroy(handle);
        }

        void* keep(void* p)
        {
            deviceAllocs.push_back(p);
            return p;
        }

        hipblasLtMatrixLayout_t makeLayout(hipDataType                 type,
                                           int                         groupCount,
                                           const std::vector<int64_t>* rows,
                                           const std::vector<int64_t>* cols,
                                           const std::vector<int64_t>* ld,
                                           int32_t                     width,
                                           uint64_t                    fixedRows = 0,
                                           uint64_t                    fixedCols = 0)
        {
            const void* r = rows ? keep(uploadDims(*rows, width)) : nullptr;
            const void* c = cols ? keep(uploadDims(*cols, width)) : nullptr;
            const void* l = ld ? keep(uploadDims(*ld, width)) : nullptr;
            hipblasLtMatrixLayout_t layout = nullptr;
            EXPECT_EQ(hipblasLtGroupedMatrixLayoutCreate(&layout, type, groupCount, r, c, l),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                          layout,
                          HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH,
                          &width,
                          sizeof(width)),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                          layout,
                          HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
                          &width,
                          sizeof(width)),
                      HIPBLAS_STATUS_SUCCESS);
            if(!rows)
                EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                              layout, HIPBLASLT_MATRIX_LAYOUT_ROWS, &fixedRows, sizeof(fixedRows)),
                          HIPBLAS_STATUS_SUCCESS);
            if(!cols)
                EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                              layout, HIPBLASLT_MATRIX_LAYOUT_COLS, &fixedCols, sizeof(fixedCols)),
                          HIPBLAS_STATUS_SUCCESS);
            return layout;
        }

        void run(const Case& tc)
        {
            const int    groups = int(tc.groups.size());
            std::mt19937 rng(1234 + groups);

            std::vector<int64_t> m, n, k;
            for(auto& s : tc.groups)
            {
                m.push_back(s.m);
                n.push_back(s.n);
                k.push_back(s.k);
            }
            const bool transA = tc.opA != HIPBLAS_OP_N;
            const bool transB = tc.opB != HIPBLAS_OP_N;

            // Buffers and leading dimensions fit the larger of the two calls' shapes.
            std::vector<int64_t> capM = m, capN = n, capK = k;
            for(size_t g = 0; g < tc.rerun.size(); g++)
            {
                capM[g] = std::max(capM[g], tc.rerun[g].m);
                capN[g] = std::max(capN[g], tc.rerun[g].n);
                capK[g] = std::max(capK[g], tc.rerun[g].k);
            }
            GroupedMatrix A, B, C, D;
            A.init(transA ? capK : capM, transA ? capM : capK, tc.fixedM, tc.type, rng);
            B.init(transB ? capN : capK, transB ? capK : capN, tc.fixedM, tc.type, rng);
            C.init(capM, capN, tc.fixedM, tc.type, rng);
            D.init(capM, capN, tc.fixedM, tc.type, rng);
            A.rows = transA ? k : m;
            A.cols = transA ? m : k;
            B.rows = transB ? n : k;
            B.cols = transB ? k : n;
            C.rows = D.rows = m;
            C.cols = D.cols = n;

            std::vector<float> alphas(groups), betas(groups);
            const float        alphaChoices[] = {1.f, 2.f, -1.f};
            const float        betaChoices[]  = {0.f, 1.f, -2.f};
            for(int g = 0; g < groups; g++)
            {
                bool perGroup = tc.scalars == Scalars::DevicePerGroup;
                alphas[g]     = perGroup ? alphaChoices[g % 3] : 2.f;
                betas[g]      = perGroup ? betaChoices[g % 3] : 1.f;
            }

            enum Dim
            {
                DimM,
                DimN,
                DimK
            };
            auto fixed
                = [&](Dim d) { return (d == DimM && tc.fixedM) || (d == DimK && tc.fixedK); };
            auto layout = [&](const GroupedMatrix& x, Dim rowsDim, Dim colsDim) {
                auto scalar = [&](Dim d) {
                    return fixed(d) ? uint64_t(d == DimM ? m[0] : k[0]) : uint64_t(0);
                };
                return makeLayout(tc.type,
                                  groups,
                                  fixed(rowsDim) ? nullptr : &x.rows,
                                  fixed(colsDim) ? nullptr : &x.cols,
                                  tc.fixedM ? nullptr : &x.ld,
                                  tc.width,
                                  scalar(rowsDim),
                                  scalar(colsDim));
            };
            auto layA = layout(A, transA ? DimK : DimM, transA ? DimM : DimK);
            auto layB = layout(B, transB ? DimN : DimK, transB ? DimK : DimN);
            auto layC = layout(C, DimM, DimN);
            auto layD = layout(D, DimM, DimN);

            hipblasLtMatmulDesc_t       desc   = nullptr;
            hipblasLtMatmulPreference_t pref   = nullptr;
            hipStream_t                 stream = nullptr;
            ScopeExit                   cleanup{[&] {
                if(stream)
                    static_cast<void>(hipStreamDestroy(stream));
                if(pref)
                    hipblasLtMatmulPreferenceDestroy(pref);
                if(desc)
                    hipblasLtMatmulDescDestroy(desc);
                for(auto l : {layA, layB, layC, layD})
                    hipblasLtMatrixLayoutDestroy(l);
            }};
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                          desc, HIPBLASLT_MATMUL_DESC_TRANSA, &tc.opA, sizeof(tc.opA)),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                          desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tc.opB, sizeof(tc.opB)),
                      HIPBLAS_STATUS_SUCCESS);

            const void* alphaArg = &alphas[0];
            const void* betaArg  = &betas[0];
            if(tc.scalars != Scalars::Host)
            {
                hipblasLtPointerMode_t mode = HIPBLASLT_POINTER_MODE_DEVICE;
                ASSERT_EQ(hipblasLtMatmulDescSetAttribute(
                              desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)),
                          HIPBLAS_STATUS_SUCCESS);
                float* dAlpha = static_cast<float*>(keep(upload(alphas)));
                float* dBeta  = static_cast<float*>(keep(upload(betas)));
                if(tc.scalars == Scalars::DeviceShared)
                {
                    alphaArg = dAlpha;
                    betaArg  = dBeta;
                }
                else
                {
                    int64_t stride = 1;
                    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                                                              HIPBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE,
                                                              &stride,
                                                              sizeof(stride)),
                              HIPBLAS_STATUS_SUCCESS);
                    ASSERT_EQ(hipblasLtMatmulDescSetAttribute(desc,
                                                              HIPBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE,
                                                              &stride,
                                                              sizeof(stride)),
                              HIPBLAS_STATUS_SUCCESS);
                    std::vector<const float*> ap(groups), bp(groups);
                    for(int g = 0; g < groups; g++)
                    {
                        ap[g] = dAlpha + g;
                        bp[g] = dBeta + g;
                    }
                    alphaArg = keep(upload(ap));
                    betaArg  = keep(upload(bp));
                }
            }

            ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
            uint64_t wsLimit = kWorkspace;
            ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsLimit, sizeof(wsLimit)),
                      HIPBLAS_STATUS_SUCCESS);
            auto setHint = [&](hipblasLtMatmulPreferenceAttributes_t attr, int64_t v) {
                ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref, attr, &v, sizeof(v)),
                          HIPBLAS_STATUS_SUCCESS);
            };
            int64_t maxM   = *std::max_element(m.begin(), m.end());
            int64_t maxN   = *std::max_element(n.begin(), n.end());
            int64_t totalN = std::accumulate(n.begin(), n.end(), int64_t(0));
            int64_t totalM = std::accumulate(m.begin(), m.end(), int64_t(0));
            if(tc.totalRowsHint)
                setHint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_ROWS_EXT, totalM);
            else if(!tc.fixedM)
                setHint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_ROWS_EXT, maxM);
            if(tc.totalColsHint)
                setHint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_COLS_EXT, totalN);
            else
                setHint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_COLS_EXT, maxN);
            setHint(HIPBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, 32);
            if(tc.averageHints)
            {
                setHint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, totalM / groups);
                setHint(HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, totalN / groups);
            }

            hipblasLtMatmulHeuristicResult_t result{};
            int                              count = 0;
            ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                          handle, desc, layA, layB, layC, layD, pref, 1, &result, &count),
                      HIPBLAS_STATUS_SUCCESS);
            if(count == 0)
                GTEST_SKIP() << "No grouped solution in the loaded library";
            ASSERT_LE(result.workspaceSize, kWorkspace);

            void* ws = nullptr;
            ASSERT_EQ(hipMalloc(&ws, kWorkspace), hipSuccess);
            keep(ws);

            ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
            auto matmulAndCheck = [&]() {
                ASSERT_EQ(hipblasLtMatmul(handle,
                                          desc,
                                          alphaArg,
                                          A.ptrArray,
                                          layA,
                                          B.ptrArray,
                                          layB,
                                          betaArg,
                                          C.ptrArray,
                                          layC,
                                          D.ptrArray,
                                          layD,
                                          &result.algo,
                                          ws,
                                          kWorkspace,
                                          stream),
                          HIPBLAS_STATUS_SUCCESS);
                ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

                std::vector<uint16_t> out(D.hostData.size());
                ASSERT_EQ(hipMemcpy(out.data(),
                                    D.deviceData,
                                    out.size() * sizeof(uint16_t),
                                    hipMemcpyDeviceToHost),
                          hipSuccess);

                for(int g = 0; g < groups; g++)
                {
                    SCOPED_TRACE(testing::Message() << "group " << g << " m=" << m[g]
                                                    << " n=" << n[g] << " k=" << k[g]);
                    int mismatches = 0;
                    for(int64_t j = 0; j < n[g]; j++)
                        for(int64_t i = 0; i < m[g]; i++)
                        {
                            float acc = 0.f;
                            for(int64_t p = 0; p < k[g]; p++)
                            {
                                float a = transA ? A.at(tc.type, g, p, i) : A.at(tc.type, g, i, p);
                                float b = transB ? B.at(tc.type, g, j, p) : B.at(tc.type, g, p, j);
                                acc += a * b;
                            }
                            float expect = alphas[g] * acc + betas[g] * C.at(tc.type, g, i, j);
                            float got
                                = toFloat(tc.type, out[D.offset[g] + size_t(j * D.ld[g] + i)]);
                            if(got != expect && mismatches++ < 5)
                                ADD_FAILURE() << "D(" << i << "," << j << ") = " << got
                                              << ", expected " << expect;
                        }
                    EXPECT_EQ(mismatches, 0);
                }
            };
            matmulAndCheck();

            if(!tc.rerun.empty())
            {
                // New shapes go straight into the device arrays the layouts point to; the
                // heuristic is not run again.
                for(int g = 0; g < groups; g++)
                {
                    m[g] = tc.rerun[g].m;
                    n[g] = tc.rerun[g].n;
                    k[g] = tc.rerun[g].k;
                }
                auto rewrite = [&](hipblasLtMatrixLayout_t           layout,
                                   hipblasLtMatrixLayoutAttribute_t attr,
                                   const std::vector<int64_t>&      values) {
                    void*  array   = nullptr;
                    size_t written = 0;
                    ASSERT_EQ(hipblasLtMatrixLayoutGetAttribute(
                                  layout, attr, &array, sizeof(array), &written),
                              HIPBLAS_STATUS_SUCCESS);
                    ASSERT_NE(array, nullptr);
                    std::vector<int32_t> narrow(values.begin(), values.end());
                    bool                 wide  = tc.width == HIPBLASLT_INTEGER_WIDTH_64;
                    const void*          src   = wide ? static_cast<const void*>(values.data())
                                                      : static_cast<const void*>(narrow.data());
                    size_t               bytes = values.size() * (wide ? 8 : 4);
                    ASSERT_EQ(hipMemcpy(array, src, bytes, hipMemcpyHostToDevice), hipSuccess);
                };
                const auto rowsAttr = HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY;
                const auto colsAttr = HIPBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY;
                rewrite(layA, rowsAttr, transA ? k : m);
                rewrite(layA, colsAttr, transA ? m : k);
                rewrite(layB, rowsAttr, transB ? n : k);
                rewrite(layB, colsAttr, transB ? k : n);
                for(auto layout : {layC, layD})
                {
                    rewrite(layout, rowsAttr, m);
                    rewrite(layout, colsAttr, n);
                }
                matmulAndCheck();
            }
        }
    };

    // The 1 x 1 x 1 group comes first, so a kernel that needs a minimum M would read before
    // the start of A. Skips where every NN kernel needs one (gfx950 FP16 needs M >= 8).
    TEST_F(GroupedMatmul, smoke_Fp16NN)
    {
        Case tc;
        tc.groups = {{1, 1, 1}, {64, 48, 32}, {17, 5, 64}, {128, 96, 16}, {33, 70, 40}};
        run(tc);
    }

    TEST_F(GroupedMatmul, smoke_EmptyGroupsAndZeroK)
    {
        Case tc;
        tc.opA    = HIPBLAS_OP_T;
        tc.groups = {{0, 16, 16}, {32, 32, 0}, {24, 0, 8}, {40, 20, 24}, {16, 16, 0}, {8, 0, 8}};
        run(tc);
    }

    TEST_F(GroupedMatmul, smoke_Fp16TN64BitDims)
    {
        Case tc;
        tc.opA    = HIPBLAS_OP_T;
        tc.width  = HIPBLASLT_INTEGER_WIDTH_64;
        tc.groups = {{48, 40, 32}, {16, 72, 8}, {96, 8, 64}};
        run(tc);
    }

    // Skips where every TT kernel needs a minimum N (gfx950 FP16 does).
    TEST_F(GroupedMatmul, smoke_Fp16TT)
    {
        Case tc;
        tc.opA    = HIPBLAS_OP_T;
        tc.opB    = HIPBLAS_OP_T;
        tc.groups = {{40, 24, 16}, {8, 64, 48}, {72, 72, 8}};
        run(tc);
    }

    // Skips where the library has no grouped BF16 kernels (gfx950 ships FP16 only).
    TEST_F(GroupedMatmul, smoke_Bf16NN)
    {
        Case tc;
        tc.type   = HIP_R_16BF;
        tc.groups = {{48, 40, 32}, {16, 72, 8}, {96, 8, 64}};
        run(tc);
    }

    // Shapes change on the device between calls; the launch setup from the first call is
    // reused because the hints and the algo are unchanged.
    TEST_F(GroupedMatmul, smoke_ShapesChangeBetweenCalls)
    {
        Case tc;
        tc.opA    = HIPBLAS_OP_T;
        tc.groups = {{64, 64, 32}, {48, 32, 64}, {96, 80, 16}, {32, 16, 32}};
        tc.rerun  = {{16, 64, 8}, {48, 0, 64}, {5, 3, 0}, {32, 16, 31}};
        run(tc);
    }

    // All groups share M: the rows arrays of A, C and D are null so the scalar ROWS applies,
    // and every ld array is null with LD = 0 (packed).
    TEST_F(GroupedMatmul, smoke_Fp16NNFixedMPacked)
    {
        Case tc;
        tc.fixedM = true;
        tc.groups = {{64, 10, 32}, {64, 0, 32}, {64, 77, 16}, {64, 3, 64}};
        run(tc);
    }

    // Very uneven N bounded only by the max-total-cols hint, with no per-group max.
    TEST_F(GroupedMatmul, smoke_TotalColsHint)
    {
        Case tc;
        tc.fixedM        = true;
        tc.totalColsHint = true;
        tc.groups        = {{96, 1, 32}, {96, 40, 32}, {96, 0, 32}, {96, 130, 32}, {96, 9, 32},
                            {96, 64, 32}, {96, 2, 32}, {96, 31, 32}};
        run(tc);
    }

    // Columns are redistributed across groups with the same total, as in an MoE layer.
    TEST_F(GroupedMatmul, smoke_TotalColsHintRedistributed)
    {
        Case tc;
        tc.opA           = HIPBLAS_OP_T;
        tc.totalColsHint = true;
        tc.groups        = {{80, 70, 32}, {80, 70, 32}, {80, 70, 32}, {80, 70, 32}};
        tc.rerun         = {{80, 0, 32}, {80, 270, 32}, {80, 3, 32}, {80, 7, 32}};
        run(tc);
    }

    // Device pointer mode: one alpha/beta shared by all groups, then per-group alpha/beta
    // through pointer arrays (ALPHA/BETA_BATCH_STRIDE = 1).
    TEST_F(GroupedMatmul, smoke_DeviceScalars)
    {
        for(Scalars s : {Scalars::DeviceShared, Scalars::DevicePerGroup})
        {
            SCOPED_TRACE(int(s));
            Case tc;
            tc.opA     = HIPBLAS_OP_T;
            tc.scalars = s;
            tc.groups  = {{32, 32, 32}, {16, 48, 8}, {40, 8, 24}, {8, 8, 0}};
            run(tc);
        }
    }

    // Very uneven M bounded only by the max-total-rows hint, with the average hints set.
    TEST_F(GroupedMatmul, smoke_TotalRowsAndAverageHints)
    {
        Case tc;
        tc.opA           = HIPBLAS_OP_T;
        tc.totalRowsHint = true;
        tc.averageHints  = true;
        tc.groups        = {{1, 48, 32}, {90, 48, 32}, {0, 48, 32}, {33, 48, 32}, {130, 48, 32}};
        run(tc);
    }

    // All groups share K: the K side of A and B uses the scalar ROWS/COLS attribute.
    TEST_F(GroupedMatmul, smoke_Fp16TNFixedK)
    {
        Case tc;
        tc.opA    = HIPBLAS_OP_T;
        tc.fixedK = true;
        tc.groups = {{64, 48, 40}, {17, 5, 40}, {128, 96, 40}, {0, 8, 40}};
        run(tc);
    }

    TEST_F(GroupedMatmul, smoke_AttributeValidation)
    {
        // A negative group count is rejected.
        hipblasLtMatrixLayout_t layout = nullptr;
        ASSERT_EQ(hipblasLtGroupedMatrixLayoutCreate(&layout, HIP_R_16F, -1, nullptr, nullptr, nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
        ASSERT_EQ(hipblasLtGroupedMatrixLayoutCreate(&layout, HIP_R_16F, 4, nullptr, nullptr, nullptr),
                  HIPBLAS_STATUS_SUCCESS);

        // A grouped layout reports the grouped batch mode and cannot be switched out of it.
        int32_t mode = 0;
        size_t  written = 0;
        ASSERT_EQ(hipblasLtMatrixLayoutGetAttribute(
                      layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &mode, sizeof(mode), &written),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(mode, HIPBLASLT_BATCH_MODE_GROUPED);
        int32_t strided = HIPBLASLT_BATCH_MODE_STRIDED;
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &strided, sizeof(strided)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        // The integer width defaults to 32-bit.
        int32_t width = -1;
        ASSERT_EQ(hipblasLtMatrixLayoutGetAttribute(layout,
                                                    HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH,
                                                    &width,
                                                    sizeof(width),
                                                    &written),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(width, HIPBLASLT_INTEGER_WIDTH_32);
        // Widths are enum values, not bit counts.
        for(int32_t badWidth : {2, 32, 64})
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(layout,
                                                        HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
                                                        &badWidth,
                                                        sizeof(badWidth)),
                      HIPBLAS_STATUS_INVALID_VALUE);
        // A valid width reads back.
        int32_t wide = HIPBLASLT_INTEGER_WIDTH_64;
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(layout,
                                                    HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
                                                    &wide,
                                                    sizeof(wide)),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutGetAttribute(layout,
                                                    HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
                                                    &width,
                                                    sizeof(width),
                                                    &written),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(width, HIPBLASLT_INTEGER_WIDTH_64);
        // A dimension array pointer reads back.
        const void* rowsArray = reinterpret_cast<const void*>(0x1234);
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      layout, HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY, &rowsArray, sizeof(rowsArray)),
                  HIPBLAS_STATUS_SUCCESS);
        const void* readBack = nullptr;
        EXPECT_EQ(hipblasLtMatrixLayoutGetAttribute(layout,
                                                    HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY,
                                                    &readBack,
                                                    sizeof(readBack),
                                                    &written),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(readBack, rowsArray);
        hipblasLtMatrixLayoutDestroy(layout);

        // A regular layout cannot be switched to grouped mode or given grouped arrays.
        hipblasLtMatrixLayout_t regular = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&regular, HIP_R_16F, 8, 8, 8), HIPBLAS_STATUS_SUCCESS);
        int32_t grouped = HIPBLASLT_BATCH_MODE_GROUPED;
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      regular, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &grouped, sizeof(grouped)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      regular, HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY, &rowsArray, sizeof(rowsArray)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        hipblasLtMatrixLayoutDestroy(regular);

        // Alpha/beta batch strides accept only 0 or 1; a valid stride reads back.
        hipblasLtMatmulDesc_t desc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        int64_t stride = 2;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE, &stride, sizeof(stride)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        stride = 1;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE, &stride, sizeof(stride)),
                  HIPBLAS_STATUS_SUCCESS);
        int64_t strideOut = 0;
        EXPECT_EQ(hipblasLtMatmulDescGetAttribute(desc,
                                                  HIPBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE,
                                                  &strideOut,
                                                  sizeof(strideOut),
                                                  &written),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(strideOut, 1);
        hipblasLtMatmulDescDestroy(desc);

        // Size hints reject negative values; a valid hint reads back.
        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        int64_t hint = -1;
        EXPECT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                      pref, HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_ROWS_EXT, &hint, sizeof(hint)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        hint = 4096;
        EXPECT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                      pref, HIPBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, &hint, sizeof(hint)),
                  HIPBLAS_STATUS_SUCCESS);
        int64_t hintOut = 0;
        EXPECT_EQ(hipblasLtMatmulPreferenceGetAttribute(pref,
                                                        HIPBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM,
                                                        &hintOut,
                                                        sizeof(hintOut),
                                                        &written),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hintOut, 4096);
        hipblasLtMatmulPreferenceDestroy(pref);
    }

    // No GPU work: every case is rejected before a kernel is selected or launched.
    TEST_F(GroupedMatmul, smoke_RejectsInvalidCalls)
    {
        std::vector<int64_t> dims = {8, 16};
        auto layA = makeLayout(HIP_R_16F, 2, &dims, &dims, nullptr, HIPBLASLT_INTEGER_WIDTH_32);
        auto layB = makeLayout(HIP_R_16F, 2, &dims, &dims, nullptr, HIPBLASLT_INTEGER_WIDTH_32);
        auto layD = makeLayout(HIP_R_16F, 2, &dims, &dims, nullptr, HIPBLASLT_INTEGER_WIDTH_32);
        auto layD3
            = makeLayout(HIP_R_16F, 3, nullptr, nullptr, nullptr, HIPBLASLT_INTEGER_WIDTH_32, 8, 8);
        hipblasLtMatrixLayout_t regular = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&regular, HIP_R_16F, 8, 8, 8), HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulDesc_t desc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        uint64_t wsLimit = kWorkspace;
        ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsLimit, sizeof(wsLimit)),
                  HIPBLAS_STATUS_SUCCESS);

        hipblasLtMatmulHeuristicResult_t result{};
        int                              count = 0;
        // D rows and cols come from arrays but no maximum hint is set.
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, layA, layB, layD, layD, pref, 1, &result, &count),
                  HIPBLAS_STATUS_INVALID_VALUE);
        // Grouped A and B with regular C and D.
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, layA, layB, regular, regular, pref, 1, &result, &count),
                  HIPBLAS_STATUS_INVALID_VALUE);
        // Group counts differ between layouts (2 vs 3).
        EXPECT_EQ(hipblasLtMatmulAlgoGetHeuristic(
                      handle, desc, layA, layB, layD3, layD3, pref, 1, &result, &count),
                  HIPBLAS_STATUS_INVALID_VALUE);

        float alpha = 1.f, beta = 0.f;
        void* fake  = reinterpret_cast<void*>(0x1000);
        // No heuristic has run with this descriptor, so it holds no size hints.
        hipblasLtMatmulAlgo_t algo{};
        EXPECT_EQ(hipblasLtMatmul(handle,
                                  desc,
                                  &alpha,
                                  fake,
                                  layA,
                                  fake,
                                  layB,
                                  &beta,
                                  fake,
                                  layD,
                                  fake,
                                  layD,
                                  &algo,
                                  fake,
                                  kWorkspace,
                                  nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);

        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        for(auto l : {layA, layB, layD, layD3, regular})
            hipblasLtMatrixLayoutDestroy(l);
    }

    // No GPU work: unsupported layouts, descriptors and calls are rejected before a kernel
    // is selected or launched.
    TEST_F(GroupedMatmul, smoke_RejectsUnsupportedSetups)
    {
        auto fixedLayout = [&](hipDataType type, int groupCount) {
            return makeLayout(
                type, groupCount, nullptr, nullptr, nullptr, HIPBLASLT_INTEGER_WIDTH_32, 8, 8);
        };
        auto    lay      = fixedLayout(HIP_R_16F, 3);
        auto    layF32   = fixedLayout(HIP_R_32F, 3);
        auto    layRow   = fixedLayout(HIP_R_16F, 3);
        auto    layEmpty = fixedLayout(HIP_R_16F, 0);
        int32_t order    = HIPBLASLT_ORDER_ROW;
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(
                      layRow, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                  HIPBLAS_STATUS_SUCCESS);

        auto makePref = [](uint64_t wsLimit) {
            hipblasLtMatmulPreference_t p = nullptr;
            EXPECT_EQ(hipblasLtMatmulPreferenceCreate(&p), HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(hipblasLtMatmulPreferenceSetAttribute(
                          p, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsLimit, sizeof(wsLimit)),
                      HIPBLAS_STATUS_SUCCESS);
            return p;
        };
        auto pref     = makePref(kWorkspace);
        auto tinyPref = makePref(16);

        std::vector<hipblasLtMatmulDesc_t> descs;
        auto makeDesc = [&](hipblasComputeType_t compute, hipDataType scaleType) {
            hipblasLtMatmulDesc_t d = nullptr;
            EXPECT_EQ(hipblasLtMatmulDescCreate(&d, compute, scaleType), HIPBLAS_STATUS_SUCCESS);
            descs.push_back(d);
            return d;
        };
        hipblasLtMatmulHeuristicResult_t result{};
        int                              count     = 0;
        auto                             heuristic = [&](hipblasLtMatmulDesc_t       d,
                             hipblasLtMatrixLayout_t     a,
                             hipblasLtMatrixLayout_t     c,
                             hipblasLtMatrixLayout_t     dl,
                             hipblasLtMatmulPreference_t p) {
            return hipblasLtMatmulAlgoGetHeuristic(handle, d, a, a, c, dl, p, 1, &result, &count);
        };

        auto desc = makeDesc(HIPBLAS_COMPUTE_32F, HIP_R_32F);
        // Row-major layouts.
        EXPECT_EQ(heuristic(desc, layRow, lay, lay, pref), HIPBLAS_STATUS_INVALID_VALUE);
        // C and D types differ.
        EXPECT_NE(heuristic(desc, lay, layF32, lay, pref), HIPBLAS_STATUS_SUCCESS);
        // No groups.
        EXPECT_EQ(heuristic(desc, layEmpty, layEmpty, layEmpty, pref),
                  HIPBLAS_STATUS_INVALID_VALUE);
        // A scale type other than FP32.
        EXPECT_EQ(heuristic(makeDesc(HIPBLAS_COMPUTE_16F, HIP_R_16F), lay, lay, lay, pref),
                  HIPBLAS_STATUS_INVALID_VALUE);
        // An epilogue.
        auto                epiDesc = makeDesc(HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasLtEpilogue_t relu    = HIPBLASLT_EPILOGUE_RELU;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      epiDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &relu, sizeof(relu)),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(epiDesc, lay, lay, lay, pref), HIPBLAS_STATUS_INVALID_VALUE);
        // A pointer mode other than host or device.
        auto                   vecDesc = makeDesc(HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasLtPointerMode_t vecMode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      vecDesc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &vecMode, sizeof(vecMode)),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(vecDesc, lay, lay, lay, pref), HIPBLAS_STATUS_INVALID_VALUE);
        // Per-group alpha in host pointer mode.
        auto    strideDesc = makeDesc(HIPBLAS_COMPUTE_32F, HIP_R_32F);
        int64_t stride     = 1;
        EXPECT_EQ(
            hipblasLtMatmulDescSetAttribute(
                strideDesc, HIPBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE, &stride, sizeof(stride)),
            HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(heuristic(strideDesc, lay, lay, lay, pref), HIPBLAS_STATUS_INVALID_VALUE);
        // A workspace limit below the record space returns no algo rather than an error.
        EXPECT_EQ(heuristic(desc, lay, lay, lay, tinyPref), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(count, 0);

        // The heuristic above recorded the hints on desc, so hipblasLtMatmul gets past them.
        float                 alpha = 1.f, beta = 0.f;
        void*                 fake = reinterpret_cast<void*>(0x1000);
        hipblasLtMatmulAlgo_t algo{};
        auto matmul = [&](hipblasLtMatrixLayout_t l, const void* a, void* workspace) {
            return hipblasLtMatmul(handle,
                                   desc,
                                   &alpha,
                                   a,
                                   l,
                                   fake,
                                   l,
                                   &beta,
                                   fake,
                                   l,
                                   fake,
                                   l,
                                   &algo,
                                   workspace,
                                   kWorkspace,
                                   nullptr);
        };
        // No groups is a no-op, even without data pointers.
        EXPECT_EQ(matmul(layEmpty, nullptr, nullptr), HIPBLAS_STATUS_SUCCESS);
        // A null data pointer.
        EXPECT_EQ(matmul(lay, nullptr, fake), HIPBLAS_STATUS_INVALID_VALUE);
        // No workspace for the records.
        EXPECT_EQ(matmul(lay, fake, nullptr), HIPBLAS_STATUS_INVALID_VALUE);

        for(auto d : descs)
            hipblasLtMatmulDescDestroy(d);
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulPreferenceDestroy(tinyPref);
        for(auto l : {lay, layF32, layRow, layEmpty})
            hipblasLtMatrixLayoutDestroy(l);
    }

    // Grouped attributes reject buffers too small for their values.
    TEST_F(GroupedMatmul, smoke_AttributeBufferSizes)
    {
        EXPECT_EQ(
            hipblasLtGroupedMatrixLayoutCreate(nullptr, HIP_R_16F, 2, nullptr, nullptr, nullptr),
            HIPBLAS_STATUS_INVALID_VALUE);

        hipblasLtMatrixLayout_t layout = nullptr;
        ASSERT_EQ(
            hipblasLtGroupedMatrixLayoutCreate(&layout, HIP_R_16F, 2, nullptr, nullptr, nullptr),
            HIPBLAS_STATUS_SUCCESS);
        const void* array   = reinterpret_cast<const void*>(0x1234);
        const void* back    = nullptr;
        int32_t     width   = HIPBLASLT_INTEGER_WIDTH_32;
        size_t      written = 0;
        for(auto attr :
            {HIPBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY, HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY})
        {
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(layout, attr, &array, sizeof(array)),
                      HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(
                hipblasLtMatrixLayoutGetAttribute(layout, attr, &back, sizeof(back), &written),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(back, array);
            EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(layout, attr, &array, 4),
                      HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_EQ(hipblasLtMatrixLayoutGetAttribute(layout, attr, &back, 4, &written),
                      HIPBLAS_STATUS_INVALID_VALUE);
        }
        const auto widthAttr = HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH;
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(layout, widthAttr, &width, 2),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtMatrixLayoutGetAttribute(layout, widthAttr, &width, 2, &written),
                  HIPBLAS_STATUS_INVALID_VALUE);
        hipblasLtMatrixLayoutDestroy(layout);

        // The integer widths apply only to grouped layouts.
        hipblasLtMatrixLayout_t regular = nullptr;
        ASSERT_EQ(hipblasLtMatrixLayoutCreate(&regular, HIP_R_16F, 8, 8, 8),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipblasLtMatrixLayoutSetAttribute(regular, widthAttr, &width, sizeof(width)),
                  HIPBLAS_STATUS_INVALID_VALUE);
        hipblasLtMatrixLayoutDestroy(regular);

        hipblasLtMatmulDesc_t desc = nullptr;
        ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                  HIPBLAS_STATUS_SUCCESS);
        int64_t stride = 1;
        EXPECT_EQ(hipblasLtMatmulDescSetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE, &stride, 4),
                  HIPBLAS_STATUS_INVALID_VALUE);
        EXPECT_EQ(hipblasLtMatmulDescGetAttribute(
                      desc, HIPBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE, &stride, 4, &written),
                  HIPBLAS_STATUS_INVALID_VALUE);
        hipblasLtMatmulDescDestroy(desc);

        hipblasLtMatmulPreference_t pref = nullptr;
        ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
        int64_t hint = 64;
        for(auto attr : {HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS,
                         HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS,
                         HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_ROWS_EXT})
        {
            EXPECT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref, attr, &hint, sizeof(hint)),
                      HIPBLAS_STATUS_SUCCESS);
            int64_t out = 0;
            EXPECT_EQ(
                hipblasLtMatmulPreferenceGetAttribute(pref, attr, &out, sizeof(out), &written),
                HIPBLAS_STATUS_SUCCESS);
            EXPECT_EQ(out, hint);
            EXPECT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref, attr, &hint, 4),
                      HIPBLAS_STATUS_INVALID_VALUE);
            EXPECT_EQ(hipblasLtMatmulPreferenceGetAttribute(pref, attr, &out, 4, &written),
                      HIPBLAS_STATUS_INVALID_VALUE);
        }
        hipblasLtMatmulPreferenceDestroy(pref);
    }
} // namespace
