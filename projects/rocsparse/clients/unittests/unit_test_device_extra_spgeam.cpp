/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */

//
// Device unit tests for library/src/extra/rocsparse_spgeam.cpp.
//
// The generic rocsparse_spgeam API is driven through
//  - the combined analysis/compute flow and the split symbolic/numeric flow,
//  - the output matrix C created before or after the analysis,
//  - host and device scalars,
//  - index base one (and mixed bases) for A, B and C,
//  - the stage state machine (each stage after the listed stage sequences,
//    including the numeric-only flow: numeric analysis + numeric compute on a
//    fresh descriptor with a user supplied sparsity pattern of C), and
//  - the argument checks.
// Every successful flow compares the whole C (row pointer, column indices and
// values) against a host reference C = alpha * A + beta * B.
//
#include "unit_test_extra_common.hpp"

#include <memory>

using namespace extra_ut;

namespace
{
    using stage_t = rocsparse_spgeam_stage;

    const char* stage_name(stage_t s)
    {
        switch(s)
        {
        case rocsparse_spgeam_stage_analysis:
            return "analysis";
        case rocsparse_spgeam_stage_compute:
            return "compute";
        case rocsparse_spgeam_stage_symbolic_analysis:
            return "symbolic_analysis";
        case rocsparse_spgeam_stage_symbolic_compute:
            return "symbolic_compute";
        case rocsparse_spgeam_stage_numeric_analysis:
            return "numeric_analysis";
        case rocsparse_spgeam_stage_numeric_compute:
            return "numeric_compute";
        }
        return "invalid";
    }

    constexpr stage_t SA = rocsparse_spgeam_stage_symbolic_analysis;
    constexpr stage_t SC = rocsparse_spgeam_stage_symbolic_compute;
    constexpr stage_t NA = rocsparse_spgeam_stage_numeric_analysis;
    constexpr stage_t NC = rocsparse_spgeam_stage_numeric_compute;
    constexpr stage_t AN = rocsparse_spgeam_stage_analysis;
    constexpr stage_t CO = rocsparse_spgeam_stage_compute;

    bool is_analysis_stage(stage_t s)
    {
        return s == SA || s == AN;
    }

    // One SpGEAM problem: A, B (CSR), optional C (CSR) and the descriptor.
    template <typename T, typename I, typename J>
    struct Geam
    {
        rocsparse_handle       handle = nullptr;
        SpgeamDescrH           descr_h;
        rocsparse_spgeam_descr descr = nullptr;
        Csr                    A, B;
        Bsr                    want;
        DevBuf<I>              pA, pB, pC;
        DevBuf<J>              cA, cB, cC;
        DevBuf<T>              vA, vB, vC;
        SpmatH                 sA, sB, sC;
        DevBuf<char>           tmp;

        rocsparse_index_base bA = rocsparse_index_base_zero, bB = rocsparse_index_base_zero,
                             bC = rocsparse_index_base_zero;

        std::unique_ptr<Scalar<T>>        alpha, beta;
        std::unique_ptr<PointerModeGuard> mode;

        int64_t nnz_C = -1;

        void init(rocsparse_handle h, bool device, uint64_t seed, int64_t m = 9, int64_t n = 7)
        {
            handle = h;
            Rng rng(seed);
            A = random_csr<T>(m, n, 4, rng);
            B = random_csr<T>(m, n, 4, rng);
            ASSERT_GT(A.nnz(), 0);
            ASSERT_GT(B.nnz(), 0);

            mode = std::make_unique<PointerModeGuard>(h, mode_of(device));
            alpha.reset(
                new Scalar<T>(is_complex_t<T>::value ? cplx(2.0, 1.0) : cplx(2.0, 0.0), device));
            beta.reset(
                new Scalar<T>(is_complex_t<T>::value ? cplx(-1.0, 0.5) : cplx(-1.0, 0.0), device));
            want = bsr_geam(to_c<T>(alpha->h), bsr_from_csr(A), to_c<T>(beta->h), bsr_from_csr(B));

            ASSERT_NO_FATAL_FAILURE((make_csr_descr<T, I, J>(sA, A, pA, cA, vA, bA)));
            ASSERT_NO_FATAL_FAILURE((make_csr_descr<T, I, J>(sB, B, pB, cB, vB, bB)));

            descr = descr_h.d;
            ASSERT_NE(descr, nullptr);
            set_inputs(descr);
        }

        void set_inputs(rocsparse_spgeam_descr d)
        {
            const rocsparse_datatype dt = rocsparse_ut::dt_of<T>();
            spgeam_set_inputs(handle, d, dt, dt, alpha->get(), beta->get());
        }

        // C with room for nnz entries (the library fills the row pointer).
        // With `with_pattern` the user supplies the final sparsity pattern
        // instead (numeric-only flow: no symbolic stage fills it).
        void create_C(int64_t nnz, bool with_pattern = false)
        {
            ASSERT_EQ(sC.p, nullptr);
            if(with_pattern)
            {
                ASSERT_EQ(nnz, want.nnzb());
                pC = up_idx<I>(want.ptr, static_cast<int64_t>(bC));
                cC = up_idx<J>(want.col, static_cast<int64_t>(bC));
            }
            else
            {
                pC = DevBuf<I>(std::vector<I>(static_cast<size_t>(A.m) + 1, I(-7)));
                cC = DevBuf<J>(static_cast<size_t>(nnz));
            }
            vC = DevBuf<T>(static_cast<size_t>(nnz));
            ASSERT_TRUE(pC.ptr && cC.ptr && vC.ptr);
            ASSERT_EQ(rocsparse_create_csr_descr(&sC.p,
                                                 A.m,
                                                 A.n,
                                                 nnz,
                                                 pC.ptr,
                                                 cC.ptr,
                                                 vC.ptr,
                                                 rocsparse_ut::it_of<I>(),
                                                 rocsparse_ut::it_of<J>(),
                                                 bC,
                                                 rocsparse_ut::dt_of<T>()),
                      rocsparse_status_success);
        }

        void clear_values()
        {
            if(vC.ptr != nullptr)
                ASSERT_EQ(hipMemset(vC.ptr, 0, vC.n * sizeof(T)), hipSuccess);
        }

        // Buffer size + execution of one stage. C is only passed if requested
        // and available.
        rocsparse_status run(stage_t stage, bool pass_C)
        {
            rocsparse_spmat_descr C  = (pass_C && sC.p != nullptr) ? sC.p : nullptr;
            size_t                bs = 0;
            rocsparse_status      st
                = rocsparse_spgeam_buffer_size(handle, descr, sA.p, sB.p, C, stage, &bs, nullptr);
            if(st != rocsparse_status_success)
                return st;
            tmp = DevBuf<char>(bs);
            st  = rocsparse_spgeam(handle, descr, sA.p, sB.p, C, stage, bs, tmp.ptr, nullptr);
            if(st == rocsparse_status_success && is_analysis_stage(stage))
            {
                // Without C the number of non-zeros is an output of the
                // descriptor, with C it is written into the matrix itself.
                int64_t nnz = -1;
                if(C == nullptr)
                {
                    EXPECT_EQ(
                        rocsparse_spgeam_get_output(
                            handle, descr, rocsparse_spgeam_output_nnz, &nnz, sizeof(nnz), nullptr),
                        rocsparse_status_success);
                }
                else
                {
                    int64_t rows = 0, cols = 0;
                    EXPECT_EQ(rocsparse_spmat_get_size(C, &rows, &cols, &nnz),
                              rocsparse_status_success);
                }
                nnz_C = nnz;
            }
            return st;
        }

        // Run a stage that must succeed. The first analysis creates C if it
        // does not exist yet.
        void ok(stage_t stage, bool pass_C)
        {
            if(stage == NC || stage == CO || stage == NA || stage == SC)
                clear_values();
            ASSERT_EQ(run(stage, pass_C), rocsparse_status_success) << stage_name(stage);
            if(is_analysis_stage(stage))
            {
                ASSERT_EQ(nnz_C, want.nnzb()) << stage_name(stage);
                if(sC.p == nullptr)
                    ASSERT_NO_FATAL_FAILURE(create_C(nnz_C));
            }
        }

        void check_C(bool values = true)
        {
            ASSERT_NE(sC.p, nullptr);
            const std::vector<I> hp = pC.host();
            std::vector<J>       hc = cC.host();
            std::vector<T>       hv = vC.host();
            sort_rows(hp, static_cast<int64_t>(bC), hc, hv);
            expect_idx_equal<I>(hp, want.ptr, static_cast<int64_t>(bC), "C row_ptr");
            expect_idx_equal<J>(hc, want.col, static_cast<int64_t>(bC), "C col_ind");
            if(values)
                expect_vec_equal<T>(hv, want.val, "C val");
        }
    };

    struct Bases
    {
        rocsparse_index_base a = rocsparse_index_base_zero, b = rocsparse_index_base_zero,
                             c = rocsparse_index_base_zero;
    };

    constexpr rocsparse_index_base Z = rocsparse_index_base_zero;
    constexpr rocsparse_index_base O = rocsparse_index_base_one;

    // kind 0: analysis (C created afterwards) + compute
    // kind 1: symbolic / numeric split flow with repeated numeric stages
    // kind 2: C created first, analysis + compute
    // kind 3: C created first, symbolic / numeric split flow
    template <typename T, typename I, typename J>
    void flow(rocsparse_handle handle, bool device, int kind, uint64_t seed, Bases bases = Bases())
    {
        SCOPED_TRACE(testing::Message() << "device=" << device << " kind=" << kind
                                        << " type=" << static_cast<int>(rocsparse_ut::dt_of<T>())
                                        << " bases=" << bases.a << bases.b << bases.c);
        Geam<T, I, J> g;
        g.bA = bases.a;
        g.bB = bases.b;
        g.bC = bases.c;
        ASSERT_NO_FATAL_FAILURE(g.init(handle, device, seed));
        const bool c_first = kind >= 2;
        if(c_first)
            ASSERT_NO_FATAL_FAILURE(g.create_C(g.want.nnzb()));

        if(kind % 2 == 0)
        {
            ASSERT_NO_FATAL_FAILURE(g.ok(AN, c_first));
            ASSERT_NO_FATAL_FAILURE(g.ok(CO, true));
            ASSERT_NO_FATAL_FAILURE(g.check_C());
        }
        else
        {
            ASSERT_NO_FATAL_FAILURE(g.ok(SA, c_first));
            ASSERT_NO_FATAL_FAILURE(g.ok(SC, true));
            ASSERT_NO_FATAL_FAILURE(g.check_C(false));
            ASSERT_NO_FATAL_FAILURE(g.ok(NA, true));
            ASSERT_NO_FATAL_FAILURE(g.ok(NC, true));
            ASSERT_NO_FATAL_FAILURE(g.check_C());
            // The numeric stages can be repeated.
            ASSERT_NO_FATAL_FAILURE(g.ok(NC, true));
            ASSERT_NO_FATAL_FAILURE(g.check_C());
            ASSERT_NO_FATAL_FAILURE(g.ok(NA, true));
            ASSERT_NO_FATAL_FAILURE(g.ok(NC, true));
            ASSERT_NO_FATAL_FAILURE(g.check_C());
        }
    }

    template <typename T, typename I, typename J>
    void flows(rocsparse_handle handle, uint64_t seed, Bases bases = Bases())
    {
        for(int device = 0; device < 2; ++device)
            for(int kind = 0; kind < 4; ++kind)
            {
                ASSERT_NO_FATAL_FAILURE((flow<T, I, J>(handle, device != 0, kind, seed, bases)));
                ++seed;
            }
    }

    struct ExtraSpgeam : public rocsparse_ut::HandleTest
    {
    };
}

// Index base one for A, B and C, and mixed bases.
TEST_F(ExtraSpgeam, flows_index_base_one)
{
    flows<float, int32_t, int32_t>(handle, 141, Bases{O, O, O});
    flows<double, int64_t, int32_t>(handle, 161, Bases{O, Z, O});
    flows<float, int32_t, int32_t>(handle, 181, Bases{Z, O, Z});
    flows<rocsparse_double_complex, int32_t, int32_t>(handle, 201, Bases{O, O, Z});
}

// The stage sequences listed in `paths` (ending in every possible "last
// executed stage") against every next stage. The orders the header documents
// must succeed (and compute the right C when they produce values); mixing the
// combined and the split stages, and a compute stage without its analysis,
// must fail. Every other order is run without checking the result.
// Not run: the numeric compute stage as the very first stage on a fresh
// descriptor (it needs the pattern of C and the numeric analysis). The
// numeric analysis as the first stage is the numeric-only flow.
TEST_F(ExtraSpgeam, stage_state_machine)
{
    using G = Geam<float, int32_t, int32_t>;

    const std::vector<std::vector<stage_t>> paths
        = {{}, {SA}, {AN}, {AN, CO}, {SA, SC}, {SA, SC, NA}, {SA, SC, NA, NC}, {NA}, {NA, NC}};
    const std::vector<stage_t> nexts = {SA, SC, NA, NC, AN, CO};

    enum class Expect
    {
        success,
        error,
        unspecified
    };
    auto expect = [](const std::vector<stage_t>& path, stage_t next) {
        const bool combined_next = next == AN || next == CO;
        bool       combined = false, split = false, seen_AN = false, seen_SA = false;
        for(stage_t s : path)
        {
            (s == AN || s == CO ? combined : split) = true;
            seen_AN                                 = seen_AN || s == AN;
            seen_SA                                 = seen_SA || s == SA;
        }
        if((combined && !combined_next) || (split && combined_next) || (next == CO && !seen_AN)
           || (next == SC && !seen_SA))
            return Expect::error;
        if(path.empty())
            return (next == SA || next == AN || next == NA) ? Expect::success : Expect::unspecified;
        const stage_t last = path.back();
        if((last == SA && next == SC) || (last == AN && next == CO) || (last == SC && next == NA)
           || (last == NA && next == NC) || (last == NC && (next == NA || next == NC)))
            return Expect::success;
        return Expect::unspecified;
    };

    for(const auto& path : paths)
        for(stage_t next : nexts)
        {
            if(path.empty() && next == NC)
                continue;

            std::string desc = "after [";
            for(stage_t s : path)
                desc += std::string(stage_name(s)) + " ";
            desc += std::string("] next=") + stage_name(next);
            SCOPED_TRACE(desc);

            G g;
            ASSERT_NO_FATAL_FAILURE(g.init(handle, false, 7));
            // The output matrix has to exist for the stages that need it. A
            // sequence starting with the numeric analysis needs the pattern
            // of C from the user.
            const bool numeric_first
                = (path.empty() && next == NA) || (!path.empty() && path[0] == NA);
            if(path.empty())
                ASSERT_NO_FATAL_FAILURE(g.create_C(g.want.nnzb(), numeric_first));
            else if(numeric_first)
                ASSERT_NO_FATAL_FAILURE(g.create_C(g.want.nnzb(), true));
            for(stage_t s : path)
                ASSERT_NO_FATAL_FAILURE(g.ok(s, s != SA && s != AN));

            if(next == SC || next == NC || next == CO || next == NA)
                g.clear_values();
            const rocsparse_status st = g.run(next, true);
            switch(expect(path, next))
            {
            case Expect::success:
                ASSERT_EQ(st, rocsparse_status_success);
                if(next == NC || next == CO)
                    ASSERT_NO_FATAL_FAILURE(g.check_C());
                if(is_analysis_stage(next))
                    ASSERT_EQ(g.nnz_C, g.want.nnzb());
                break;
            case Expect::error:
                EXPECT_NE(st, rocsparse_status_success);
                break;
            case Expect::unspecified:
                break;
            }
        }
}

TEST_F(ExtraSpgeam, argument_errors)
{
    using G = Geam<float, int32_t, int32_t>;
    G g;
    ASSERT_NO_FATAL_FAILURE(g.init(handle, false, 3));
    ASSERT_NO_FATAL_FAILURE(g.create_C(g.want.nnzb()));

    size_t       bs = 0;
    DevBuf<char> buf(size_t(4096));
    using Sp = rocsparse_const_spmat_descr;
    auto buffer_size
        = [&](rocsparse_handle       h,
              rocsparse_spgeam_descr d,
              Sp                     a,
              Sp                     b,
              rocsparse_spmat_descr  c,
              stage_t s) { return rocsparse_spgeam_buffer_size(h, d, a, b, c, s, &bs, nullptr); };
    auto execute = [&](rocsparse_handle       h,
                       rocsparse_spgeam_descr d,
                       Sp                     a,
                       Sp                     b,
                       rocsparse_spmat_descr  c,
                       stage_t                s,
                       size_t                 size = 0,
                       void*                  temp = nullptr) {
        return rocsparse_spgeam(h, d, a, b, c, s, size, temp, nullptr);
    };
    // Both entry points share the operand checks.
    auto both = [&](Sp a, Sp b, rocsparse_spmat_descr c, stage_t s, Expected want) {
        EXPECT_TRUE(status_is(buffer_size(handle, g.descr, a, b, c, s), want));
        EXPECT_TRUE(status_is(execute(handle, g.descr, a, b, c, s), want));
    };
    constexpr auto        NULLP = rocsparse_status_invalid_pointer;
    rocsparse_spmat_descr A = g.sA.p, B = g.sB.p, C = g.sC.p;

    EXPECT_EQ(buffer_size(nullptr, g.descr, A, B, C, CO), rocsparse_status_invalid_handle);
    EXPECT_EQ(execute(nullptr, g.descr, A, B, C, AN), rocsparse_status_invalid_handle);
    EXPECT_EQ(buffer_size(handle, nullptr, A, B, C, CO), NULLP);
    EXPECT_EQ(execute(handle, nullptr, A, B, C, AN), NULLP);
    both(nullptr, B, C, AN, NULLP);
    both(A, nullptr, C, AN, NULLP);
    EXPECT_TRUE(status_is(buffer_size(handle, g.descr, A, B, nullptr, CO), any_error));
    for(stage_t s : {SC, NC, CO})
        EXPECT_EQ(execute(handle, g.descr, A, B, nullptr, s), NULLP) << stage_name(s);
    EXPECT_TRUE(status_is(execute(handle, g.descr, A, B, C, AN, 16, nullptr), any_error));
    const auto bad_stage = static_cast<stage_t>(99);
    EXPECT_TRUE(status_is(buffer_size(handle, g.descr, A, B, C, bad_stage), any_error));
    EXPECT_TRUE(status_is(execute(handle, g.descr, A, B, C, bad_stage, 16, buf.ptr), any_error));

    // ---- mismatching operands ----
    {
        // B stored as COO while A is CSR.
        std::vector<int64_t> rows;
        for(int64_t i = 0; i < g.B.m; ++i)
            for(int64_t p = g.B.ptr[i]; p < g.B.ptr[i + 1]; ++p)
                rows.push_back(i);
        DevBuf<int32_t> coo_row = up_idx<int32_t>(rows);
        SpmatH          coo, coo_a;
        for(SpmatH* m : {&coo, &coo_a})
            ASSERT_EQ(rocsparse_create_coo_descr(&m->p,
                                                 g.B.m,
                                                 g.B.n,
                                                 g.B.nnz(),
                                                 coo_row.ptr,
                                                 g.cB.ptr,
                                                 g.vB.ptr,
                                                 rocsparse_indextype_i32,
                                                 rocsparse_index_base_zero,
                                                 rocsparse_datatype_f32_r),
                      rocsparse_status_success);
        both(A, coo.p, C, AN, any_error);
        // A and B both COO: only CSR is supported.
        EXPECT_TRUE(
            status_is(buffer_size(handle, g.descr, coo_a.p, coo.p, nullptr, AN), any_error));
        // C stored as COO.
        both(A, B, coo.p, CO, any_error);
    }
    {
        // Value type of B (and of C) different from the compute type.
        DevBuf<int32_t> p, c;
        DevBuf<double>  v;
        SpmatH          bd;
        ASSERT_NO_FATAL_FAILURE((make_csr_descr<double, int32_t, int32_t>(bd, g.B, p, c, v)));
        both(A, bd.p, C, CO, any_error);
        both(bd.p, B, C, CO, any_error);
        both(A, B, bd.p, CO, any_error);
    }
    {
        // Index types of B (and of C) different from the ones of A.
        DevBuf<int64_t> p64, c64;
        DevBuf<int32_t> p32, c32;
        DevBuf<float>   v_row64, v_col64;
        SpmatH          b_row64, b_col64;
        ASSERT_NO_FATAL_FAILURE(
            (make_csr_descr<float, int64_t, int32_t>(b_row64, g.B, p64, c32, v_row64)));
        ASSERT_NO_FATAL_FAILURE(
            (make_csr_descr<float, int32_t, int64_t>(b_col64, g.B, p32, c64, v_col64)));
        for(rocsparse_spmat_descr wrong : {b_row64.p, b_col64.p})
        {
            both(A, wrong, C, CO, any_error);
            both(A, B, wrong, CO, any_error);
        }
    }

    // ---- invalid descriptor inputs ----
    {
        // An invalid value is either rejected by set_input or by spgeam.
        auto expect_with
            = [&](rocsparse_spgeam_input input, const void* data, size_t size, const char* what) {
                  SpgeamDescrH h;
                  ASSERT_NE(h.d, nullptr);
                  g.set_inputs(h.d);
                  if(rocsparse_spgeam_set_input(handle, h.d, input, data, size, nullptr)
                         != rocsparse_status_success
                     || buffer_size(handle, h.d, A, B, C, AN) != rocsparse_status_success)
                      return;
                  DevBuf<char> tmp(bs);
                  EXPECT_TRUE(status_is(execute(handle, h.d, A, B, C, AN, bs, tmp.ptr), any_error))
                      << what;
              };
        const auto bad_alg = static_cast<rocsparse_spgeam_alg>(99);
        const auto bad_op  = static_cast<rocsparse_operation>(99);
        const auto bad_dt  = static_cast<rocsparse_datatype>(99);
        expect_with(rocsparse_spgeam_input_alg, &bad_alg, sizeof(bad_alg), "alg");
        expect_with(rocsparse_spgeam_input_operation_A, &bad_op, sizeof(bad_op), "operation A");
        expect_with(rocsparse_spgeam_input_operation_B, &bad_op, sizeof(bad_op), "operation B");
        expect_with(rocsparse_spgeam_input_scalar_datatype, &bad_dt, sizeof(bad_dt), "scalar type");
    }

    // ---- set_input / get_output sizes ----
    {
        SpgeamDescrH h;
        ASSERT_NE(h.d, nullptr);
        float x = 1.0f;
        for(auto input : {rocsparse_spgeam_input_scalar_alpha, rocsparse_spgeam_input_scalar_beta})
            EXPECT_EQ(rocsparse_spgeam_set_input(handle, h.d, input, &x, 1, nullptr),
                      rocsparse_status_invalid_size);
        int32_t nnz32 = 0;
        EXPECT_EQ(rocsparse_spgeam_get_output(
                      handle, h.d, rocsparse_spgeam_output_nnz, &nnz32, sizeof(nnz32), nullptr),
                  rocsparse_status_invalid_size);
    }

    // The session is still usable after the failed calls.
    ASSERT_NO_FATAL_FAILURE(g.ok(AN, true));
    ASSERT_NO_FATAL_FAILURE(g.ok(CO, true));
    ASSERT_NO_FATAL_FAILURE(g.check_C());
}
