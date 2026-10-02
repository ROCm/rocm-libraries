// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests for the fast_check verifier in clients/common/include/fast_check.hpp. The
// FastCheck_pre_checkin suite is host-only: it builds small integer GEMM problems on the CPU,
// computes D exactly, injects a fault, and checks that fast_check reports it. The
// FastCheckDevice_pre_checkin suite exercises the padding poison, scan and probe-sum kernels on
// small device buffers and launches no hipBLASLt kernels. The _pre_checkin suffix puts both suites
// in the standard test tier, which selects tests by name. Together they take about 0.5 s on
// gfx942.

#include <gtest/gtest.h>

#include "fast_check.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
    // A small strided-batched problem held in float, with integer values in the integer_exact
    // ranges: A and C in [0, 2], B in [-2, 2], scale in [1, 3], bias in [-3, 3].
    struct HostProblem
    {
        int64_t M, N, K, batch;
        bool    transA, transB;
        int64_t lda, ldb, ldc, ldd;
        float   alpha, beta;
        bool    with_scale, with_bias;

        std::vector<float> A, B, C, D, scale, bias;

        HostProblem(int64_t m,
                    int64_t n,
                    int64_t k,
                    int64_t b,
                    bool    ta,
                    bool    tb,
                    float   al,
                    float   be,
                    bool    sc,
                    bool    bi)
            : M(m)
            , N(n)
            , K(k)
            , batch(b)
            , transA(ta)
            , transB(tb)
            , alpha(al)
            , beta(be)
            , with_scale(sc)
            , with_bias(bi)
        {
            // Leading dimensions larger than the rows, so the padding is never read.
            lda            = (transA ? K : M) + 3;
            ldb            = (transB ? N : K) + 2;
            ldc            = M + 1;
            ldd            = M + 5;
            uint32_t state = 12345;
            auto     next  = [&](int lo, int hi) {
                state = state * 1664525u + 1013904223u;
                return float(lo + int((state >> 8) % uint32_t(hi - lo + 1)));
            };
            A.assign(size_t(stride_a() * batch), 99.f);
            B.assign(size_t(stride_b() * batch), 99.f);
            C.assign(size_t(stride_c() * batch), 99.f);
            for(int64_t s = 0; s < batch; s++)
            {
                for(int64_t c = 0; c < (transA ? M : K); c++)
                    for(int64_t r = 0; r < (transA ? K : M); r++)
                        A[size_t(s * stride_a() + c * lda + r)] = next(0, 2);
                for(int64_t c = 0; c < (transB ? K : N); c++)
                    for(int64_t r = 0; r < (transB ? N : K); r++)
                        B[size_t(s * stride_b() + c * ldb + r)] = next(-2, 2);
                for(int64_t c = 0; c < N; c++)
                    for(int64_t r = 0; r < M; r++)
                        C[size_t(s * stride_c() + c * ldc + r)] = next(0, 2);
            }
            scale.assign(size_t(M), 1.f);
            if(with_scale)
                for(auto& v : scale)
                    v = next(1, 3);
            bias.assign(size_t(M * batch), 0.f);
            if(with_bias)
                for(auto& v : bias)
                    v = next(-3, 3);
            D.assign(size_t(stride_d() * batch), 99.f);
            for(int64_t s = 0; s < batch; s++)
                for(int64_t j = 0; j < N; j++)
                    for(int64_t i = 0; i < M; i++)
                        D[size_t(s * stride_d() + j * ldd + i)] = float(exact(s, i, j));
        }

        int64_t stride_a() const
        {
            return lda * (transA ? M : K);
        }
        int64_t stride_b() const
        {
            return ldb * (transB ? K : N);
        }
        int64_t stride_c() const
        {
            return ldc * N;
        }
        int64_t stride_d() const
        {
            return ldd * N + 7; // a gap between batches
        }

        double a(int64_t s, int64_t i, int64_t k) const
        {
            return A[size_t(s * stride_a() + (transA ? i * lda + k : k * lda + i))];
        }
        double b(int64_t s, int64_t k, int64_t j) const
        {
            return B[size_t(s * stride_b() + (transB ? k * ldb + j : j * ldb + k))];
        }
        double exact(int64_t s, int64_t i, int64_t j) const
        {
            double acc = 0;
            for(int64_t k = 0; k < K; k++)
                acc += a(s, i, k) * b(s, k, j);
            return alpha * scale[size_t(i)] * acc + beta * C[size_t(s * stride_c() + j * ldc + i)]
                   + bias[size_t(s * M + i)];
        }
        float& d(int64_t s, int64_t i, int64_t j)
        {
            return D[size_t(s * stride_d() + j * ldd + i)];
        }

        FastCheckProblem problem() const
        {
            FastCheckProblem p;
            p.M           = M;
            p.N           = N;
            p.K           = K;
            p.batch_count = batch;
            p.transA      = transA;
            p.transB      = transB;
            p.A           = {A.data(), HIP_R_32F, transA ? K : M, transA ? M : K, lda, stride_a()};
            p.B           = {B.data(), HIP_R_32F, transB ? N : K, transB ? K : N, ldb, stride_b()};
            p.C           = {C.data(), HIP_R_32F, M, N, ldc, stride_c()};
            p.D           = {D.data(), HIP_R_32F, M, N, ldd, stride_d()};
            p.alpha       = alpha;
            p.beta        = beta;
            if(with_scale)
                p.scale_alpha_vec = scale.data();
            if(with_bias)
            {
                p.bias        = bias.data();
                p.bias_stride = M;
            }
            p.seed = 7;
            return p;
        }
    };

    HostProblem default_problem(bool transA = false, bool transB = false)
    {
        return HostProblem(37, 29, 23, 2, transA, transB, 2.f, -2.f, true, true);
    }

    // ------------------------------------------------------------------------------------------
    // fast_check_gemm on host data
    // ------------------------------------------------------------------------------------------

    // Guards against false failures: a correct D must pass for every transpose combination, with
    // alpha, beta, scaleAlpha_vector, bias and strided batches all in play.
    TEST(FastCheck_pre_checkin, correct_result_passes_all_transposes)
    {
        for(bool ta : {false, true})
            for(bool tb : {false, true})
            {
                HostProblem hp  = default_problem(ta, tb);
                auto        res = fast_check_gemm(hp.problem());
                EXPECT_TRUE(res.passed) << "transA=" << ta << " transB=" << tb << "\n"
                                        << res.message;
            }
    }

    // Fails if the row or column probe skips any element, or if a mismatch is not located.
    TEST(FastCheck_pre_checkin, single_wrong_element_is_found_and_located)
    {
        HostProblem hp = default_problem();
        hp.d(1, 17, 11) += 1;
        auto res = fast_check_gemm(hp.problem());
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("1 rows and 1 columns"), std::string::npos) << res.message;
        EXPECT_NE(res.message.find("batch 1, row 17, col 11"), std::string::npos) << res.message;
        // The offset uses D's layout: batch * stride_d + col * ldd + row.
        int64_t offset = hp.stride_d() + 11 * hp.ldd + 17;
        EXPECT_NE(res.message.find("element offset " + std::to_string(offset)), std::string::npos)
            << res.message;
    }

    // A plain checksum (all-ones probe) misses errors that sum to zero within a row; the random
    // probe does not. Fails if the probe vector degenerates to constant entries.
    TEST(FastCheck_pre_checkin, errors_that_cancel_in_a_plain_row_sum_are_found)
    {
        HostProblem hp = default_problem();
        hp.d(0, 5, 3) += 1;
        hp.d(0, 5, 4) -= 1;
        auto res = fast_check_gemm(hp.problem());
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("row 5, col 3"), std::string::npos) << res.message;
    }

    // Models an off-by-one row index in the store: every column of row i holds row i+1's value.
    TEST(FastCheck_pre_checkin, row_shifted_by_one_is_found)
    {
        HostProblem hp = default_problem();
        for(int64_t j = 0; j < hp.N; j++)
            hp.d(0, 9, j) = hp.d(0, 10, j);
        auto res = fast_check_gemm(hp.problem());
        EXPECT_FALSE(res.passed);
    }

    // Models a wrong column base pointer: column 8 is written with column 7's values.
    TEST(FastCheck_pre_checkin, column_duplicated_from_neighbor_is_found)
    {
        HostProblem hp = default_problem();
        for(int64_t i = 0; i < hp.M; i++)
            hp.d(1, i, 8) = hp.d(1, i, 7);
        auto res = fast_check_gemm(hp.problem());
        EXPECT_FALSE(res.passed);
    }

    // An element still holding the NaN sentinel must be reported directly.
    TEST(FastCheck_pre_checkin, non_finite_element_is_reported)
    {
        HostProblem hp       = default_problem();
        uint32_t    sentinel = uint32_t(fast_check_sentinel_bits(HIP_R_32F));
        std::memcpy(&hp.d(0, 0, 0), &sentinel, sizeof(sentinel));
        auto res = fast_check_gemm(hp.problem());
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("not finite"), std::string::npos) << res.message;
    }

    // The probe and the expected-value terms must use C only when beta is nonzero, and must not
    // read C at all otherwise (C may be uninitialized when beta == 0).
    TEST(FastCheck_pre_checkin, beta_zero_ignores_c)
    {
        HostProblem hp(16, 9, 5, 1, false, true, 2.f, 0.f, false, false);
        for(auto& v : hp.C)
            v = 1e30f;
        auto res = fast_check_gemm(hp.problem());
        EXPECT_TRUE(res.passed) << res.message;
    }

    // bf16 holds integers exactly only below 256. A correct kernel stores the rounded value of a
    // larger result, which must pass; a stored value that is not the correct rounding must fail.
    TEST(FastCheck_pre_checkin, rounding_above_the_exact_range_is_modelled)
    {
        const int64_t             M = 4, N = 3, K = 2;
        std::vector<float>        A = {100, 100, 1, 1, 100, 100, 1, 1}; // M x K
        std::vector<float>        B = {3, 0, 0, 1, 1, 0}; // K x N
        std::vector<hip_bfloat16> D(size_t(M * N));
        for(int64_t j = 0; j < N; j++)
            for(int64_t i = 0; i < M; i++)
            {
                double acc = 0;
                for(int64_t k = 0; k < K; k++)
                    acc += A[size_t(k * M + i)] * B[size_t(j * K + k)];
                D[size_t(j * M + i)] = hip_bfloat16(float(2 * acc + 1)); // e.g. 601 -> 600
            }
        FastCheckProblem p;
        p.M = M, p.N = N, p.K = K;
        p.A     = {A.data(), HIP_R_32F, M, K, M, M * K};
        p.B     = {B.data(), HIP_R_32F, K, N, K, K * N};
        p.C     = {A.data(), HIP_R_32F, M, N, M, M * N};
        p.D     = {D.data(), HIP_R_16BF, M, N, M, M * N};
        p.alpha = 2;
        std::vector<float> bias(size_t(M), 1.f);
        p.bias      = bias.data();
        p.bias_type = HIP_R_32F;

        auto res = fast_check_gemm(p);
        EXPECT_TRUE(res.passed) << res.message;

        D[0]           = hip_bfloat16(float(float(D[0]) + 4)); // a representable but wrong value
        auto res_wrong = fast_check_gemm(p);
        ASSERT_FALSE(res_wrong.passed);
        EXPECT_NE(res_wrong.message.find("row 0, col 0"), std::string::npos) << res_wrong.message;
    }

    TEST(FastCheck_pre_checkin, non_integer_alpha_is_rejected)
    {
        HostProblem hp = default_problem();
        auto        p  = hp.problem();
        p.alpha        = 1.5;
        auto res       = fast_check_gemm(p);
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("integer alpha"), std::string::npos) << res.message;
    }

    // The placement record must flag a buffer that crosses a 4 GiB boundary, give the boundary,
    // and leave buffers that stay inside one 4 GiB window unflagged.
    TEST(FastCheck_pre_checkin, placement_record_flags_4gib_crossings)
    {
        const uint64_t four_gib = uint64_t(1) << 32;
        auto           at = [](uint64_t a) { return reinterpret_cast<const void*>(uintptr_t(a)); };
        std::string    record = fast_check_describe_buffers({
            {"A", at(3 * four_gib - 0x600000), 0x1000000}, // crosses 3 * 2^32
            {"B", at(5 * four_gib), 0x1000}, // starts exactly on a boundary
            {"C", at(7 * four_gib - 0x1000), 0x1000}, // ends exactly on a boundary
            {"unused", nullptr, 0},
        });
        EXPECT_NE(record.find("A: 0x2ffa00000 to 0x300a00000 (16777216 bytes), crosses a 4 GiB "
                              "boundary at 0x300000000"),
                  std::string::npos)
            << record;
        EXPECT_NE(record.find("B: 0x500000000 to 0x500001000 (4096 bytes)\n"), std::string::npos)
            << record;
        EXPECT_EQ(record.find("C: 0x6fffff000 to 0x700000000 (4096 bytes), crosses"),
                  std::string::npos)
            << record;
        EXPECT_EQ(record.find("unused"), std::string::npos) << record;
    }

    // The sentinel must be a NaN wherever the type has one, so an unwritten element is non-finite.
    // FNUZ fp8 has a single NaN, 0x80, which all-ones bytes are not.
    TEST(FastCheck_pre_checkin, sentinel_is_nan_where_the_type_has_one)
    {
        EXPECT_EQ(fast_check_sentinel_bits(HIP_R_32F), 0xffffffffu);
        EXPECT_EQ(fast_check_sentinel_bits(HIP_R_16BF), 0xffffu);
        for(hipDataType t : {HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E5M2_FNUZ})
        {
            uint8_t bits = uint8_t(fast_check_sentinel_bits(t));
            EXPECT_EQ(bits, 0x80);
            hipblaslt_f8_fnuz f;
            std::memcpy(&f, &bits, 1);
            EXPECT_TRUE(f.is_nan());
        }
        EXPECT_EQ(fast_check_sentinel_bits(HIP_R_8I), 0x80u);
        EXPECT_EQ(fast_check_sentinel_bits(HIP_R_32I), 0x80000000u);
    }

    TEST(FastCheck_pre_checkin, probe_entries_are_nonzero_residues_and_depend_on_seed)
    {
        int differ = 0;
        for(uint64_t i = 0; i < 1000; i++)
        {
            uint64_t v = fast_check_probe(3, i);
            EXPECT_GE(v, 1u);
            EXPECT_LT(v, kFastCheckModulus);
            EXPECT_EQ(v, fast_check_probe(3, i));
            differ += v != fast_check_probe(4, i);
        }
        EXPECT_EQ(differ, 1000);
    }

    // ------------------------------------------------------------------------------------------
    // fast_check_result_device: D in device memory
    // ------------------------------------------------------------------------------------------

    // Copies D, with its padded leading dimension and batch gap, to the device and runs the
    // device pass against the same expected sums the host pass uses.
    template <typename T>
    FastCheckResult device_result(FastCheckProblem p, const std::vector<T>& d_host)
    {
        T* d = nullptr;
        EXPECT_EQ(hipMalloc(&d, d_host.size() * sizeof(T)), hipSuccess);
        EXPECT_EQ(hipMemcpy(d, d_host.data(), d_host.size() * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);
        auto e   = fast_check_expected(p);
        p.D.data = d;
        auto res = fast_check_result_device(p, e, 0);
        (void)hipFree(d);
        return res;
    }

    // The device pass must reach the same verdict as the host pass, for a correct D and for each
    // fault, and must locate the fault the same way.
    TEST(FastCheckDevice_pre_checkin, device_pass_agrees_with_host_pass)
    {
        struct Fault
        {
            const char* name;
            void (*inject)(HostProblem&);
            const char* located; // text both messages must contain, or nullptr
        };
        const Fault faults[] = {
            {"none", [](HostProblem&) {}, nullptr},
            {"single element",
             [](HostProblem& hp) { hp.d(1, 17, 11) += 1; },
             "batch 1, row 17, col 11"},
            {"cancelling pair",
             [](HostProblem& hp) {
                 hp.d(0, 5, 3) += 1;
                 hp.d(0, 5, 4) -= 1;
             },
             "row 5, col 3"},
            {"row shift",
             [](HostProblem& hp) {
                 for(int64_t j = 0; j < hp.N; j++)
                     hp.d(0, 9, j) = hp.d(0, 10, j);
             },
             nullptr},
            {"sentinel left in D",
             [](HostProblem& hp) {
                 uint32_t sentinel = uint32_t(fast_check_sentinel_bits(HIP_R_32F));
                 std::memcpy(&hp.d(1, 2, 3), &sentinel, sizeof(sentinel));
             },
             "batch 1, row 2, col 3"},
        };
        for(bool ta : {false, true})
            for(const auto& f : faults)
            {
                HostProblem hp = default_problem(ta, !ta);
                f.inject(hp);
                auto host   = fast_check_gemm(hp.problem());
                auto device = device_result(hp.problem(), hp.D);
                EXPECT_EQ(host.passed, device.passed)
                    << f.name << "\nhost: " << host.message << "\ndevice: " << device.message;
                EXPECT_EQ(device.passed, std::string(f.name) == "none") << f.name << "\n"
                                                                        << device.message;
                if(f.located)
                {
                    EXPECT_NE(host.message.find(f.located), std::string::npos) << host.message;
                    EXPECT_NE(device.message.find(f.located), std::string::npos) << device.message;
                }
            }
    }

    // bf16 results above 256 take the device pass's per-element path: a correctly rounded value
    // passes and a wrong one fails with its location.
    TEST(FastCheckDevice_pre_checkin, device_pass_models_bf16_rounding)
    {
        const int64_t             M = 4, N = 3, K = 2;
        std::vector<float>        A = {100, 100, 1, 1, 100, 100, 1, 1};
        std::vector<float>        B = {3, 0, 0, 1, 1, 0};
        std::vector<float>        bias(size_t(M), 1.f);
        std::vector<hip_bfloat16> D(size_t(M * N));
        for(int64_t j = 0; j < N; j++)
            for(int64_t i = 0; i < M; i++)
            {
                double acc = 0;
                for(int64_t k = 0; k < K; k++)
                    acc += A[size_t(k * M + i)] * B[size_t(j * K + k)];
                D[size_t(j * M + i)] = hip_bfloat16(float(2 * acc + 1));
            }
        FastCheckProblem p;
        p.M = M, p.N = N, p.K = K;
        p.A     = {A.data(), HIP_R_32F, M, K, M, M * K};
        p.B     = {B.data(), HIP_R_32F, K, N, K, K * N};
        p.C     = {A.data(), HIP_R_32F, M, N, M, M * N};
        p.D     = {nullptr, HIP_R_16BF, M, N, M, M * N};
        p.alpha = 2;
        p.bias  = bias.data();

        auto res = device_result(p, D);
        EXPECT_TRUE(res.passed) << res.message;
        D[0] = hip_bfloat16(float(float(D[0]) + 4));
        res  = device_result(p, D);
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("row 0, col 0"), std::string::npos) << res.message;
    }

    // fp8 outputs are not converted on the device; the device call copies D and uses the host
    // pass, and must still find a wrong element.
    TEST(FastCheckDevice_pre_checkin, fp8_output_falls_back_to_the_host_pass)
    {
        HostProblem hp(24, 10, 2, 1, false, false, 1.f, 0.f, false, false); // |D| <= 8
        std::vector<hipblaslt_f8_fnuz> d8(hp.D.size());
        for(size_t n = 0; n < hp.D.size(); n++)
            d8[n] = hipblaslt_f8_fnuz(hp.D[n]);
        FastCheckProblem p = hp.problem();
        p.D.type           = HIP_R_8F_E4M3_FNUZ;

        auto res = device_result(p, d8);
        EXPECT_TRUE(res.passed) << res.message;
        d8[size_t(3 * hp.ldd + 7)] = hipblaslt_f8_fnuz(hp.D[size_t(3 * hp.ldd + 7)] + 1.f);
        res                        = device_result(p, d8);
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("row 7, col 3"), std::string::npos) << res.message;
    }

    // ------------------------------------------------------------------------------------------
    // Device padding helpers
    // ------------------------------------------------------------------------------------------

    struct DeviceMatrix
    {
        // 5 x 3 region, ld 8, two batches with stride 30, and 7 elements after the last batch.
        static constexpr int64_t rows = 5, cols = 3, ld = 8, stride = 30, batch = 2;
        static constexpr size_t  total = size_t(stride * batch + 7);

        float* d = nullptr;

        DeviceMatrix()
        {
            EXPECT_EQ(hipMalloc(&d, total * sizeof(float)), hipSuccess);
        }
        ~DeviceMatrix()
        {
            (void)hipFree(d);
        }

        FastCheckMatrix matrix() const
        {
            return {d, HIP_R_32F, rows, cols, ld, stride};
        }

        static bool in_region(size_t idx)
        {
            int64_t b = int64_t(idx) / stride, rem = int64_t(idx) % stride;
            return b < batch && rem / ld < cols && rem % ld < rows;
        }

        std::vector<float> read() const
        {
            std::vector<float> h(total);
            EXPECT_EQ(hipMemcpy(h.data(), d, total * sizeof(float), hipMemcpyDeviceToHost),
                      hipSuccess);
            return h;
        }

        void write(const std::vector<float>& h)
        {
            EXPECT_EQ(hipMemcpy(d, h.data(), total * sizeof(float), hipMemcpyHostToDevice),
                      hipSuccess);
        }

        void set(size_t idx, float v)
        {
            EXPECT_EQ(hipMemcpy(d + idx, &v, sizeof(float), hipMemcpyHostToDevice), hipSuccess);
        }
    };

    // Poison must cover all padding (column tails, batch gap, tail after the last batch) and
    // leave the region untouched.
    TEST(FastCheckDevice_pre_checkin, poison_fills_exactly_the_padding)
    {
        DeviceMatrix       m;
        std::vector<float> h(DeviceMatrix::total, 1.f);
        m.write(h);
        fast_check_poison_padding_device(m.matrix(), DeviceMatrix::batch, DeviceMatrix::total, 0);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        auto out = m.read();
        for(size_t idx = 0; idx < DeviceMatrix::total; idx++)
            EXPECT_EQ(out[idx], DeviceMatrix::in_region(idx) ? 1.f : kFastCheckPoisonValue)
                << "offset " << idx;

        auto res = fast_check_scan_padding_device(
            m.matrix(), DeviceMatrix::batch, DeviceMatrix::total, true, 0);
        EXPECT_TRUE(res.passed) << res.message;
    }

    // A write into padding must be reported with its offset, in the poisoned (c_equal_d) mode and
    // in the sentinel mode.
    TEST(FastCheckDevice_pre_checkin, scan_reports_out_of_bounds_writes)
    {
        DeviceMatrix m;
        // Offset 36 is batch 1, row 6, col 0: column padding of batch 1.
        const size_t pad_offset = 36;
        ASSERT_FALSE(DeviceMatrix::in_region(pad_offset));

        std::vector<float> h(DeviceMatrix::total, 1.f);
        m.write(h);
        fast_check_poison_padding_device(m.matrix(), DeviceMatrix::batch, DeviceMatrix::total, 0);
        m.set(pad_offset, 2.f);
        auto res = fast_check_scan_padding_device(
            m.matrix(), DeviceMatrix::batch, DeviceMatrix::total, true, 0);
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("1 elements outside"), std::string::npos) << res.message;
        EXPECT_NE(res.message.find("element offset 36"), std::string::npos) << res.message;
        EXPECT_NE(res.message.find("batch 1, row 6, col 0"), std::string::npos) << res.message;

        // Sentinel mode: fill everything, write the region, then stray into the batch gap.
        fast_check_fill_sentinel_device(m.d, HIP_R_32F, DeviceMatrix::total, 0);
        std::vector<float> s = m.read();
        for(size_t idx = 0; idx < DeviceMatrix::total; idx++)
            if(DeviceMatrix::in_region(idx))
                s[idx] = 3.f;
        m.write(s);
        res = fast_check_scan_padding_device(
            m.matrix(), DeviceMatrix::batch, DeviceMatrix::total, false, 0);
        EXPECT_TRUE(res.passed) << res.message;
        m.set(DeviceMatrix::total - 1, 3.f);
        res = fast_check_scan_padding_device(
            m.matrix(), DeviceMatrix::batch, DeviceMatrix::total, false, 0);
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("element offset " + std::to_string(DeviceMatrix::total - 1)),
                  std::string::npos)
            << res.message;
    }

    // An element of the region that still holds the sentinel was never stored by the kernel.
    TEST(FastCheckDevice_pre_checkin, scan_reports_unwritten_elements)
    {
        DeviceMatrix m;
        fast_check_fill_sentinel_device(m.d, HIP_R_32F, DeviceMatrix::total, 0);
        std::vector<float> s = m.read();
        for(size_t idx = 0; idx < DeviceMatrix::total; idx++)
            if(DeviceMatrix::in_region(idx) && idx != 33)
                s[idx] = 3.f;
        m.write(s);
        auto res = fast_check_scan_padding_device(
            m.matrix(), DeviceMatrix::batch, DeviceMatrix::total, false, 0);
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("1 elements inside the region were never written"),
                  std::string::npos)
            << res.message;
        EXPECT_NE(res.message.find("element offset 33"), std::string::npos) << res.message;
    }

    // Integer outputs saturate, and their sentinel is a valid value. A correct saturated result
    // must pass; a wrong saturated value, and an element still holding the sentinel where the
    // correct value is -1, must both fail, on the host pass and on the device pass.
    TEST(FastCheckDevice_pre_checkin, integer_output_saturation_and_sentinel)
    {
        // D = A * B with int8 data, an int32 accumulator and an int8 output. With B = {1, 1} the
        // exact results are 200, -200, -1 and 5; the first two saturate to 127 and -128.
        const int64_t       M = 4, N = 1, K = 2;
        std::vector<int8_t> A = {100, -100, -1, 2, 100, -100, 0, 3}; // M x K
        std::vector<int8_t> B = {1, 1}; // K x N
        std::vector<int8_t> D = {127, -128, -1, 5}; // M x N
        FastCheckProblem    p;
        p.M = M, p.N = N, p.K = K;
        p.A            = {A.data(), HIP_R_8I, M, K, M, M * K};
        p.B            = {B.data(), HIP_R_8I, K, N, K, K * N};
        p.C            = {D.data(), HIP_R_8I, M, N, M, M * N};
        p.D            = {D.data(), HIP_R_8I, M, N, M, M * N};
        p.compute_type = HIP_R_32I;

        auto host = fast_check_gemm(p);
        EXPECT_TRUE(host.passed) << host.message;
        auto device = device_result(p, D);
        EXPECT_TRUE(device.passed) << device.message;

        std::vector<int8_t> wrong = D;
        wrong[0]                  = 126; // 200 saturates to 127, not 126
        p.D.data                  = wrong.data();
        host                      = fast_check_gemm(p);
        ASSERT_FALSE(host.passed);
        EXPECT_NE(host.message.find("row 0, col 0"), std::string::npos) << host.message;
        device = device_result(p, wrong);
        ASSERT_FALSE(device.passed);
        EXPECT_NE(device.message.find("row 0, col 0"), std::string::npos) << device.message;

        std::vector<int8_t> unwritten = D;
        unwritten[2]                  = int8_t(fast_check_sentinel_bits(HIP_R_8I)); // expected -1
        p.D.data                      = unwritten.data();
        host                          = fast_check_gemm(p);
        ASSERT_FALSE(host.passed);
        EXPECT_NE(host.message.find("row 2, col 0"), std::string::npos) << host.message;
        device = device_result(p, unwritten);
        ASSERT_FALSE(device.passed);
        EXPECT_NE(device.message.find("row 2, col 0"), std::string::npos) << device.message;
    }

    // With the FNUZ sentinel (0x80, a NaN), the device scan reports an unwritten element of an
    // FNUZ fp8 output. An all-ones sentinel is a valid FNUZ value and would hide it.
    TEST(FastCheckDevice_pre_checkin, scan_reports_unwritten_fnuz_elements)
    {
        const int64_t rows = 4, cols = 4, ld = 8;
        const size_t  total = size_t(ld * cols);
        uint8_t*      d     = nullptr;
        ASSERT_EQ(hipMalloc(&d, total), hipSuccess);
        FastCheckMatrix m{d, HIP_R_8F_E4M3_FNUZ, rows, cols, ld, 0};
        fast_check_fill_sentinel_device(d, HIP_R_8F_E4M3_FNUZ, total, 0);
        std::vector<uint8_t> h(total);
        ASSERT_EQ(hipMemcpy(h.data(), d, total, hipMemcpyDeviceToHost), hipSuccess);
        for(int64_t j = 0; j < cols; j++)
            for(int64_t i = 0; i < rows; i++)
                if(!(i == 3 && j == 2))
                    h[size_t(j * ld + i)] = 0x08; // a finite FNUZ value
        ASSERT_EQ(hipMemcpy(d, h.data(), total, hipMemcpyHostToDevice), hipSuccess);
        auto res = fast_check_scan_padding_device(m, 1, total, false, 0);
        (void)hipFree(d);
        ASSERT_FALSE(res.passed);
        EXPECT_NE(res.message.find("1 elements inside the region were never written"),
                  std::string::npos)
            << res.message;
        EXPECT_NE(res.message.find("element offset 19"), std::string::npos) << res.message;
    }

    TEST(FastCheckDevice_pre_checkin, copy_region_to_host_drops_the_padding)
    {
        DeviceMatrix       m;
        std::vector<float> h(DeviceMatrix::total);
        for(size_t idx = 0; idx < DeviceMatrix::total; idx++)
            h[idx] = float(idx);
        m.write(h);
        std::vector<float> compact(
            size_t(DeviceMatrix::rows * DeviceMatrix::cols * DeviceMatrix::batch));
        ASSERT_EQ(
            fast_check_copy_region_to_host(compact.data(), m.matrix(), DeviceMatrix::batch, 0),
            hipSuccess);
        size_t n = 0;
        for(int64_t b = 0; b < DeviceMatrix::batch; b++)
            for(int64_t j = 0; j < DeviceMatrix::cols; j++)
                for(int64_t i = 0; i < DeviceMatrix::rows; i++)
                    EXPECT_EQ(compact[n++],
                              float(b * DeviceMatrix::stride + j * DeviceMatrix::ld + i));
    }
} // namespace
