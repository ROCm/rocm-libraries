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

    // ------------------------------------------------------------------------------------------
    // Every supported type
    // ------------------------------------------------------------------------------------------

    struct TypeCase
    {
        hipDataType type;
        size_t      size;
        const char* name;
    };

    const TypeCase kTypes[] = {{HIP_R_64F, 8, "f64"},
                               {HIP_R_32F, 4, "f32"},
                               {HIP_R_16F, 2, "f16"},
                               {HIP_R_16BF, 2, "bf16"},
                               {HIP_R_32I, 4, "i32"},
                               {HIP_R_8I, 1, "i8"},
                               {HIP_R_8F_E4M3, 1, "f8"},
                               {HIP_R_8F_E5M2, 1, "bf8"},
                               {HIP_R_8F_E4M3_FNUZ, 1, "f8_fnuz"},
                               {HIP_R_8F_E5M2_FNUZ, 1, "bf8_fnuz"}};

    // Stores v in type t, rounding to nearest even, as the GPU stores a result.
    void store_as(hipDataType t, double v, void* dst)
    {
        auto put = [dst](auto x) { std::memcpy(dst, &x, sizeof(x)); };
        switch(t)
        {
        case HIP_R_64F:
            return put(v);
        case HIP_R_32F:
            return put(float(v));
        case HIP_R_16F:
            return put(hipblasLtHalf(float(v)));
        case HIP_R_16BF:
            return put(hip_bfloat16(float(v)));
        case HIP_R_32I:
            return put(int32_t(v));
        case HIP_R_8I:
            return put(int8_t(v));
        case HIP_R_8F_E4M3:
            return put(hipblaslt_f8(float(v)));
        case HIP_R_8F_E5M2:
            return put(hipblaslt_bf8(float(v)));
        case HIP_R_8F_E4M3_FNUZ:
            return put(hipblaslt_f8_fnuz(float(v)));
        case HIP_R_8F_E5M2_FNUZ:
            return put(hipblaslt_bf8_fnuz(float(v)));
        default:
            FAIL() << "no conversion for type " << int(t);
        }
    }

    std::vector<char> convert(const std::vector<float>& v, const TypeCase& t)
    {
        std::vector<char> out(v.size() * t.size);
        for(size_t i = 0; i < v.size(); i++)
            store_as(t.type, v[i], out.data() + i * t.size);
        return out;
    }

    // A, B, C and D all in one type. K = 3 keeps every result within 14, which every type holds,
    // though E5M2 rounds the ones above 8; D is stored the way the GPU rounds it.
    struct TypedProblem
    {
        TypeCase          t;
        HostProblem       hp{13, 11, 3, 2, false, true, 1.f, 1.f, false, false};
        std::vector<char> A, B, C, D;

        explicit TypedProblem(const TypeCase& tc)
            : t(tc)
            , A(convert(hp.A, tc))
            , B(convert(hp.B, tc))
            , C(convert(hp.C, tc))
            , D(convert(hp.D, tc))
        {
        }

        FastCheckProblem problem()
        {
            FastCheckProblem p = hp.problem();
            p.A.data = A.data(), p.B.data = B.data(), p.C.data = C.data(), p.D.data = D.data();
            p.A.type = p.B.type = p.C.type = p.D.type = t.type;
            p.compute_type = t.type == HIP_R_8I || t.type == HIP_R_32I ? HIP_R_32I : HIP_R_32F;
            return p;
        }

        char* d(int64_t s, int64_t i, int64_t j)
        {
            return D.data() + size_t(s * hp.stride_d() + j * hp.ldd + i) * t.size;
        }
    };

    FastCheckResult device_result_bytes(FastCheckProblem p, const std::vector<char>& d_host)
    {
        void* d = nullptr;
        EXPECT_EQ(hipMalloc(&d, d_host.size()), hipSuccess);
        EXPECT_EQ(hipMemcpy(d, d_host.data(), d_host.size(), hipMemcpyHostToDevice), hipSuccess);
        auto e   = fast_check_expected(p);
        p.D.data = d;
        auto res = fast_check_result_device(p, e, 0);
        (void)hipFree(d);
        return res;
    }

    // In every supported type, a correct D passes the host and device passes, and a wrong
    // element and an element left holding the sentinel are both located by each.
    TEST(FastCheckDevice_pre_checkin, every_type_passes_and_locates_faults)
    {
        for(const auto& tc : kTypes)
        {
            TypedProblem tp(tc);
            auto         host = fast_check_gemm(tp.problem());
            EXPECT_TRUE(host.passed) << tc.name << "\n" << host.message;
            auto device = device_result_bytes(tp.problem(), tp.D);
            EXPECT_TRUE(device.passed) << tc.name << "\n" << device.message;

            // Exactly 0 and 1 differ in every type.
            TypedProblem wrong(tc);
            store_as(tc.type, wrong.hp.d(1, 7, 5) == 0 ? 1 : 0, wrong.d(1, 7, 5));
            host = fast_check_gemm(wrong.problem());
            EXPECT_FALSE(host.passed) << tc.name;
            EXPECT_NE(host.message.find("batch 1, row 7, col 5"), std::string::npos)
                << tc.name << "\n"
                << host.message;
            device = device_result_bytes(wrong.problem(), wrong.D);
            EXPECT_FALSE(device.passed) << tc.name;
            EXPECT_NE(device.message.find("batch 1, row 7, col 5"), std::string::npos)
                << tc.name << "\n"
                << device.message;

            TypedProblem unwritten(tc);
            uint64_t     sentinel = fast_check_sentinel_bits(tc.type);
            std::memcpy(unwritten.d(0, 2, 3), &sentinel, tc.size);
            host = fast_check_gemm(unwritten.problem());
            EXPECT_FALSE(host.passed) << tc.name;
            EXPECT_NE(host.message.find("batch 0, row 2, col 3"), std::string::npos)
                << tc.name << "\n"
                << host.message;
            device = device_result_bytes(unwritten.problem(), unwritten.D);
            EXPECT_FALSE(device.passed) << tc.name;
            EXPECT_NE(device.message.find("batch 0, row 2, col 3"), std::string::npos)
                << tc.name << "\n"
                << device.message;
        }
    }

    // In every supported type, poison covers exactly the padding, the scan passes it and reports
    // a changed padding element and an unwritten one, and the region copy drops the padding.
    TEST(FastCheckDevice_pre_checkin, every_type_poisons_scans_and_copies)
    {
        using DM = DeviceMatrix;
        for(const auto& tc : kTypes)
        {
            const size_t bytes = DM::total * tc.size;
            char*        d     = nullptr;
            ASSERT_EQ(hipMalloc(&d, bytes), hipSuccess);
            FastCheckMatrix   m{d, tc.type, DM::rows, DM::cols, DM::ld, DM::stride};
            std::vector<char> one(tc.size), h(bytes);
            store_as(tc.type, 1, one.data());
            for(size_t idx = 0; idx < DM::total; idx++)
                std::memcpy(h.data() + idx * tc.size, one.data(), tc.size);
            ASSERT_EQ(hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice), hipSuccess);

            fast_check_poison_padding_device(m, DM::batch, DM::total, 0);
            auto res = fast_check_scan_padding_device(m, DM::batch, DM::total, true, 0);
            EXPECT_TRUE(res.passed) << tc.name << "\n" << res.message;
            ASSERT_EQ(hipMemcpy(d + 36 * tc.size, h.data(), tc.size, hipMemcpyHostToDevice),
                      hipSuccess);
            res = fast_check_scan_padding_device(m, DM::batch, DM::total, true, 0);
            EXPECT_FALSE(res.passed) << tc.name;
            EXPECT_NE(res.message.find("batch 1, row 6, col 0"), std::string::npos)
                << tc.name << "\n"
                << res.message;

            fast_check_fill_sentinel_device(d, tc.type, DM::total, 0);
            std::vector<char> s(bytes);
            ASSERT_EQ(hipMemcpy(s.data(), d, bytes, hipMemcpyDeviceToHost), hipSuccess);
            for(size_t idx = 0; idx < DM::total; idx++)
                if(DM::in_region(idx) && idx != 33)
                    std::memcpy(s.data() + idx * tc.size, one.data(), tc.size);
            ASSERT_EQ(hipMemcpy(d, s.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
            res = fast_check_scan_padding_device(m, DM::batch, DM::total, false, 0);
            // An integer sentinel is a valid value, so the scan cannot tell it was never
            // written; the value pass recomputes it exactly instead.
            if(tc.type == HIP_R_8I || tc.type == HIP_R_32I)
                EXPECT_TRUE(res.passed) << tc.name << "\n" << res.message;
            else
            {
                EXPECT_FALSE(res.passed) << tc.name;
                EXPECT_NE(res.message.find("element offset 33"), std::string::npos)
                    << tc.name << "\n"
                    << res.message;
            }

            std::vector<char> compact(size_t(DM::rows * DM::cols * DM::batch) * tc.size);
            ASSERT_EQ(fast_check_copy_region_to_host(compact.data(), m, DM::batch, 0), hipSuccess);
            size_t n = 0;
            for(int64_t b = 0; b < DM::batch; b++)
                for(int64_t j = 0; j < DM::cols; j++)
                    for(int64_t i = 0; i < DM::rows; i++, n++)
                        EXPECT_EQ(std::memcmp(
                                      compact.data() + n * tc.size,
                                      s.data() + size_t(b * DM::stride + j * DM::ld + i) * tc.size,
                                      tc.size),
                                  0)
                            << tc.name << " batch " << b << ", row " << i << ", col " << j;
            (void)hipFree(d);
        }
    }

    // Configurations fast_check cannot check exactly are refused with a reason, before any sums.
    TEST(FastCheck_pre_checkin, unsupported_configurations_are_refused)
    {
        auto refused = [](const FastCheckProblem& p, const char* reason) {
            auto res = fast_check_gemm(p);
            EXPECT_FALSE(res.passed) << reason;
            EXPECT_NE(res.message.find(reason), std::string::npos) << res.message;
        };
        HostProblem hp = default_problem();

        FastCheckProblem p = hp.problem();
        p.D.type           = HIP_C_32F;
        refused(p, "does not support data type");

        p              = hp.problem();
        p.compute_type = HIP_C_32F;
        refused(p, "does not support compute type");

        std::vector<float> scale = hp.scale;
        scale[3]                 = 1.5f;
        p                        = hp.problem();
        p.scale_alpha_vec        = scale.data();
        refused(p, "integer scaleAlpha_vector");

        std::vector<float> bias = hp.bias;
        bias[4]                 = 0.25f;
        p                       = hp.problem();
        p.bias                  = bias.data();
        refused(p, "integer bias");

        HostProblem fractional = default_problem();
        fractional.A[0]        = 0.5f;
        refused(fractional.problem(), "non-integer value in A, B or C");
    }

    // A type fast_check does not support is never scanned as clean, the fills leave it alone,
    // and the device pass returns the refusal from fast_check_expected.
    TEST(FastCheckDevice_pre_checkin, unsupported_types_are_never_clean)
    {
        DeviceMatrix       m;
        std::vector<float> h(DeviceMatrix::total, 1.f);
        m.write(h);
        FastCheckMatrix complex = m.matrix();
        complex.type            = HIP_C_32F;
        fast_check_fill_sentinel_device(m.d, HIP_C_32F, DeviceMatrix::total, 0);
        fast_check_poison_padding_device(complex, DeviceMatrix::batch, DeviceMatrix::total, 0);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        EXPECT_EQ(m.read(), h);
        auto res = fast_check_scan_padding_device(
            complex, DeviceMatrix::batch, DeviceMatrix::total, true, 0);
        EXPECT_FALSE(res.passed);
        EXPECT_NE(res.message.find("scan failed"), std::string::npos) << res.message;

        HostProblem      hp = default_problem();
        FastCheckProblem p  = hp.problem();
        p.D.type            = HIP_C_32F;
        res                 = fast_check_result_device(p, fast_check_expected(p), 0);
        EXPECT_FALSE(res.passed);
        EXPECT_NE(res.message.find("does not support data type"), std::string::npos) << res.message;
    }

    // An exact result beyond the range the compute type holds exactly is reported as such (or
    // refused up front, once fast_check bounds the results): its GPU value depends on the
    // summation order, so it is neither right nor wrong.
    TEST(FastCheck_pre_checkin, results_beyond_the_compute_range_are_reported)
    {
        // 100 * 100 + 100 * 100 = 20000, which f16 stores exactly but f16 arithmetic does not
        // hold exactly (above 2^11).
        std::vector<hipblasLtHalf> A(2, hipblasLtHalf(100.f)), B(2, hipblasLtHalf(100.f)),
            D(1, hipblasLtHalf(20000.f));
        FastCheckProblem p;
        p.M = 1, p.N = 1, p.K = 2;
        p.A            = {A.data(), HIP_R_16F, 1, 2, 1, 2};
        p.B            = {B.data(), HIP_R_16F, 2, 1, 2, 2};
        p.C            = {D.data(), HIP_R_16F, 1, 1, 1, 1};
        p.D            = {D.data(), HIP_R_16F, 1, 1, 1, 1};
        p.compute_type = HIP_R_16F;
        auto res       = fast_check_gemm(p);
        EXPECT_FALSE(res.passed);
        EXPECT_NE(res.message.find("compute type holds"), std::string::npos) << res.message;
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
