// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fast_check.hpp"
#include "d_vector.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace
{
    constexpr uint64_t P = kFastCheckModulus;

    // ------------------------------------------------------------------------------------------
    // Arithmetic modulo P = 2^61 - 1
    // ------------------------------------------------------------------------------------------

    // x mod P for any 128-bit x. Folding uses 2^61 == 1 (mod P).
    inline uint64_t mod_reduce(unsigned __int128 x)
    {
        unsigned __int128 y = (x & P) + (x >> 61);
        y                   = (y & P) + (y >> 61);
        uint64_t z          = uint64_t(y);
        return z >= P ? z - P : z;
    }

    __host__ __device__ inline uint64_t mod_add(uint64_t a, uint64_t b)
    {
        uint64_t s = a + b;
        return s >= P ? s - P : s;
    }

    inline uint64_t mod_mul(uint64_t a, uint64_t b)
    {
        return mod_reduce((unsigned __int128)a * b);
    }

    // Residue of v. Values strictly between -P and P, which covers every value the probe sums
    // see, need no division.
    __host__ __device__ inline uint64_t to_mod(int64_t v)
    {
        constexpr int64_t p = int64_t(P);
        if(v > -p && v < p)
            return v < 0 ? uint64_t(v + p) : uint64_t(v);
        int64_t r = v % p;
        return r < 0 ? uint64_t(r + p) : uint64_t(r);
    }

    // Sums products of residues. Each product is below 2^122, so 32 of them fit in the 128-bit
    // sum on top of a reduced remainder.
    struct ModAcc
    {
        unsigned __int128 sum   = 0;
        uint32_t          terms = 0;

        void add(uint64_t a, uint64_t b)
        {
            sum += (unsigned __int128)a * b;
            if(++terms == 32)
            {
                sum   = mod_reduce(sum);
                terms = 0;
            }
        }

        uint64_t value() const
        {
            return mod_reduce(sum);
        }
    };

    inline uint64_t splitmix64(uint64_t x)
    {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    // ------------------------------------------------------------------------------------------
    // Element types
    // ------------------------------------------------------------------------------------------

    size_t element_size(hipDataType t)
    {
        switch(t)
        {
        case HIP_R_64F:
            return 8;
        case HIP_R_32F:
        case HIP_R_32I:
            return 4;
        case HIP_R_16F:
        case HIP_R_16BF:
            return 2;
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_8F_E5M2_FNUZ:
        case HIP_R_8F_E4M3:
        case HIP_R_8F_E5M2:
        case HIP_R_8I:
            return 1;
        default:
            return 0;
        }
    }

    template <typename T>
    inline double to_double(T x)
    {
        if constexpr(std::is_same_v<T, hipblaslt_f8_fnuz> || std::is_same_v<T, hipblaslt_bf8_fnuz>
                     || std::is_same_v<T, hipblaslt_f8> || std::is_same_v<T, hipblaslt_bf8>)
            return float(static_cast<hipblasLtHalf>(x));
        else if constexpr(std::is_same_v<T, hipblasLtHalf> || std::is_same_v<T, hip_bfloat16>)
            return float(x);
        else
            return double(x);
    }

    // Calls f with a typed null pointer selecting the C++ type for t; returns false when t is
    // not supported.
    template <typename F>
    bool dispatch_type(hipDataType t, F&& f)
    {
        switch(t)
        {
        case HIP_R_32F:
            f(static_cast<const float*>(nullptr));
            return true;
        case HIP_R_64F:
            f(static_cast<const double*>(nullptr));
            return true;
        case HIP_R_16F:
            f(static_cast<const hipblasLtHalf*>(nullptr));
            return true;
        case HIP_R_16BF:
            f(static_cast<const hip_bfloat16*>(nullptr));
            return true;
        case HIP_R_8F_E4M3_FNUZ:
            f(static_cast<const hipblaslt_f8_fnuz*>(nullptr));
            return true;
        case HIP_R_8F_E5M2_FNUZ:
            f(static_cast<const hipblaslt_bf8_fnuz*>(nullptr));
            return true;
        case HIP_R_8F_E4M3:
            f(static_cast<const hipblaslt_f8*>(nullptr));
            return true;
        case HIP_R_8F_E5M2:
            f(static_cast<const hipblaslt_bf8*>(nullptr));
            return true;
        case HIP_R_32I:
            f(static_cast<const int32_t*>(nullptr));
            return true;
        case HIP_R_8I:
            f(static_cast<const int8_t*>(nullptr));
            return true;
        default:
            return false;
        }
    }

    inline double load(const void* p, hipDataType t, size_t i)
    {
        double v = std::numeric_limits<double>::quiet_NaN();
        dispatch_type(t, [&](auto typed_null) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(typed_null)>>;
            v       = to_double(static_cast<const T*>(p)[i]);
        });
        return v;
    }

    // The value a type stores for an exactly computed integer, after round-to-nearest-even
    // (floating types) or saturation (integer types).
    double round_to_type(double v, hipDataType t)
    {
        switch(t)
        {
        case HIP_R_32F:
            return double(float(v));
        case HIP_R_64F:
            return v;
        case HIP_R_16F:
            return double(float(hipblasLtHalf(float(v))));
        case HIP_R_16BF:
            return double(float(hip_bfloat16(float(v))));
        case HIP_R_8F_E4M3_FNUZ:
            return to_double(hipblaslt_f8_fnuz(float(v)));
        case HIP_R_8F_E5M2_FNUZ:
            return to_double(hipblaslt_bf8_fnuz(float(v)));
        case HIP_R_8F_E4M3:
            return to_double(hipblaslt_f8(float(v)));
        case HIP_R_8F_E5M2:
            return to_double(hipblaslt_bf8(float(v)));
        case HIP_R_32I:
            return std::clamp(v, -2147483648.0, 2147483647.0);
        case HIP_R_8I:
            return std::clamp(v, -128.0, 127.0);
        default:
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Every integer with magnitude below this value is exactly representable in the type.
    double exact_limit(hipDataType t)
    {
        switch(t)
        {
        case HIP_R_64F:
            return 0x1p53;
        case HIP_R_32F:
            return 0x1p24;
        case HIP_R_32I:
            return 0x1p31;
        case HIP_R_16F:
            return 0x1p11;
        case HIP_R_16BF:
            return 0x1p8;
        case HIP_R_8I:
            return 128;
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_8F_E4M3:
            return 0x1p4;
        case HIP_R_8F_E5M2_FNUZ:
        case HIP_R_8F_E5M2:
            return 0x1p3;
        default:
            return 0;
        }
    }

    // Stored values below this magnitude take the fast path. For integer types it leaves out the
    // extremes: a correct result reaches them only by saturating, and the sentinel uses the
    // minimum, so those elements are always recomputed exactly.
    double fast_limit(hipDataType t)
    {
        switch(t)
        {
        case HIP_R_8I:
            return 127;
        case HIP_R_32I:
            return 2147483647.0;
        default:
            return exact_limit(t);
        }
    }

    inline bool is_exact_integer(double v)
    {
        return std::isfinite(v) && v == std::trunc(v) && std::fabs(v) < 0x1p61;
    }

    // ------------------------------------------------------------------------------------------
    // Host matrix views and products
    // ------------------------------------------------------------------------------------------

    // A strided view of a matrix: element (r, c) is at base[r * rs + c * cs].
    struct View
    {
        const void* base;
        hipDataType type;
        int64_t     rs;
        int64_t     cs;

        double at(int64_t r, int64_t c) const
        {
            return load(base, type, size_t(r * rs + c * cs));
        }

        View transposed() const
        {
            return {base, type, cs, rs};
        }
    };

    View batch_view(const FastCheckMatrix& m, int64_t batch, bool trans)
    {
        // A buffer that is not read (C when beta is 0) may be null and must not be offset.
        const void* base = m.data ? static_cast<const char*>(m.data)
                                        + size_t(batch) * size_t(m.stride) * element_size(m.type)
                                  : nullptr;
        return trans ? View{base, m.type, m.ld, 1} : View{base, m.type, 1, m.ld};
    }

    // Calls f(r, c, |m(r, c)|) for every element, each row r from one thread only, in an order
    // that walks memory contiguously.
    template <typename F>
    void for_each_abs_by_row(const View& m, int64_t rows, int64_t cols, F&& f)
    {
        if(std::llabs(m.rs) <= std::llabs(m.cs))
        {
            constexpr int64_t block = 256;
#pragma omp parallel for schedule(dynamic)
            for(int64_t r0 = 0; r0 < rows; r0 += block)
                for(int64_t c = 0; c < cols; c++)
                    for(int64_t r = r0; r < std::min(rows, r0 + block); r++)
                        f(r, c, std::fabs(m.at(r, c)));
        }
        else
        {
#pragma omp parallel for schedule(dynamic)
            for(int64_t r = 0; r < rows; r++)
                for(int64_t c = 0; c < cols; c++)
                    f(r, c, std::fabs(m.at(r, c)));
        }
    }

    // out[r] = max over c of |m(r, c)|.
    std::vector<double> abs_row_max(const View& m, int64_t rows, int64_t cols)
    {
        std::vector<double> out(size_t(rows), 0);
        for_each_abs_by_row(m, rows, cols, [&](int64_t r, int64_t, double x) {
            out[size_t(r)] = std::max(out[size_t(r)], x);
        });
        return out;
    }

    // out[r] = sum over c of |m(r, c)| * v[c], with v[c] >= 0.
    std::vector<double>
        abs_mul_right(const View& m, int64_t rows, int64_t cols, const std::vector<double>& v)
    {
        std::vector<double> out(size_t(rows), 0);
        for_each_abs_by_row(m, rows, cols, [&](int64_t r, int64_t c, double x) {
            out[size_t(r)] += x * v[size_t(c)];
        });
        return out;
    }

    // out[r] = sum over c of m(r, c) * v[c], modulo P. Sets non_integer when an element is not an
    // exact integer.
    void mul_right(const View&                  m,
                   int64_t                      rows,
                   int64_t                      cols,
                   const std::vector<uint64_t>& v,
                   std::vector<uint64_t>&       out,
                   std::atomic<bool>&           non_integer)
    {
        out.assign(size_t(rows), 0);
        auto element = [&](int64_t r, int64_t c) -> uint64_t {
            double x = m.at(r, c);
            if(!is_exact_integer(x))
            {
                non_integer = true;
                return 0;
            }
            return to_mod(int64_t(x));
        };

        if(std::llabs(m.rs) <= std::llabs(m.cs))
        {
            // Rows are adjacent in memory: walk each column over a block of rows.
            constexpr int64_t block = 256;
#pragma omp parallel for schedule(dynamic)
            for(int64_t r0 = 0; r0 < rows; r0 += block)
            {
                int64_t             r1 = std::min(rows, r0 + block);
                std::vector<ModAcc> acc(size_t(r1 - r0));
                for(int64_t c = 0; c < cols; c++)
                    for(int64_t r = r0; r < r1; r++)
                        acc[size_t(r - r0)].add(element(r, c), v[size_t(c)]);
                for(int64_t r = r0; r < r1; r++)
                    out[size_t(r)] = acc[size_t(r - r0)].value();
            }
        }
        else
        {
#pragma omp parallel for schedule(dynamic, 64)
            for(int64_t r = 0; r < rows; r++)
            {
                ModAcc acc;
                for(int64_t c = 0; c < cols; c++)
                    acc.add(element(r, c), v[size_t(c)]);
                out[size_t(r)] = acc.value();
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // Comparison and reporting, shared by the host and device passes over D
    // ------------------------------------------------------------------------------------------

    struct BadElement
    {
        int64_t batch;
        int64_t row;
        int64_t col;
        double  expected; // value a correct kernel stores
        double  got;
    };

    constexpr size_t kMaxReported = 8;

    // The inputs of one batch, for recomputing single elements exactly.
    struct BatchInputs
    {
        View                        opA; // M x K
        View                        opB; // K x N
        View                        C; // M x N
        const std::vector<int64_t>& scale;
        const std::vector<int64_t>& col_scale;
        const std::vector<int64_t>& bias;
        int64_t                     K;
        int64_t                     alpha;
        int64_t                     beta;

        // The exact result, saturated to the int64 range: each factor may be close to 2^61, and
        // a result that does not fit is far beyond every compute type's exact range anyway.
        int64_t exact(int64_t i, int64_t j) const
        {
            __int128 acc = 0;
            for(int64_t k = 0; k < K; k++)
                acc = mul_add(int64_t(opA.at(i, k)), int64_t(opB.at(k, j)), acc);
            const __int128 factor
                = mul_add(mul_add(alpha, scale[size_t(i)], 0), col_scale[size_t(j)], 0);
            __int128 v = mul_add(factor, acc, bias[size_t(i)]);
            if(beta != 0)
                v = mul_add(beta, int64_t(C.at(i, j)), v);
            return int64_t(std::clamp<__int128>(v, INT64_MIN, INT64_MAX));
        }

        // a * b + c, saturating at the int128 limits instead of overflowing.
        static __int128 mul_add(__int128 a, __int128 b, __int128 c)
        {
            constexpr __int128 hi = ~(__int128(1) << 127), lo = -hi - 1;
            __int128           p;
            if(__builtin_mul_overflow(a, b, &p))
                return (a < 0) != (b < 0) ? lo : hi;
            if(__builtin_add_overflow(p, c, &p))
                return c < 0 ? lo : hi;
            return p;
        }
    };

    BatchInputs batch_inputs(const FastCheckProblem& p, const FastCheckExpected& e, int64_t b)
    {
        return {batch_view(p.A, b, p.transA),
                batch_view(p.B, b, p.transB),
                batch_view(p.C, b, false),
                e.scale,
                e.col_scale,
                e.bias[size_t(b)],
                p.K,
                e.alpha,
                e.beta};
    }

    std::string format_offset(const FastCheckProblem& p, const BadElement& e)
    {
        int64_t            ld     = p.device_ldd ? p.device_ldd : p.D.ld;
        int64_t            stride = p.device_stride_d ? p.device_stride_d : p.D.stride;
        int64_t            offset = e.batch * stride + e.col * ld + e.row;
        std::ostringstream s;
        s << "batch " << e.batch << ", row " << e.row << ", col " << e.col << ": expected "
          << e.expected << ", got " << e.got << " (element offset " << offset << " = 0x" << std::hex
          << offset << std::dec;
        if(offset >= (int64_t(1) << 32))
            s << ", at or above 2^32";
        else if(offset >= (int64_t(1) << 31))
            s << ", at or above 2^31";
        s << ")";
        return s.str();
    }

    // Collects the outcome of every batch and formats the result.
    struct Report
    {
        const FastCheckProblem& p;
        const int64_t           slow_budget;
        const double            acc_limit;

        std::atomic<int64_t>    slow_elements{0};
        std::atomic<bool>       over_budget{false};
        std::atomic<int64_t>    beyond_compute_range{0};
        std::vector<BadElement> bad_elements;
        int64_t                 bad_element_count = 0;
        int64_t                 bad_row_count     = 0;
        int64_t                 bad_col_count     = 0;
        std::vector<BadElement> located;

        explicit Report(const FastCheckProblem& problem)
            : p(problem)
            // Elements outside the range D's type stores exactly each cost a K-length dot
            // product; the budget keeps that to about 2^34 multiply-adds.
            , slow_budget(
                  std::max<int64_t>(1, (int64_t(1) << 34) / std::max<int64_t>(problem.K, 1)))
            , acc_limit(exact_limit(problem.compute_type))
        {
        }

        // An element whose stored value is outside the range D's type holds exactly, or is not an
        // integer. Compares it with the exact result and returns the value for the probe sums.
        int64_t slow_element(const BatchInputs&       in,
                             int64_t                  b,
                             int64_t                  i,
                             int64_t                  j,
                             double                   stored,
                             std::vector<BadElement>& bad,
                             int64_t&                 bad_count)
        {
            if(over_budget || ++slow_elements > slow_budget)
            {
                over_budget = true;
                return 0;
            }
            int64_t value   = in.exact(i, j);
            double  rounded = round_to_type(double(value), p.D.type);
            if(std::fabs(double(value)) >= acc_limit)
                ++beyond_compute_range;
            else if(!(rounded == stored) && bad_count++ < int64_t(kMaxReported))
                bad.push_back({b, i, j, rounded, stored});
            return value;
        }

        void add_bad(const std::vector<BadElement>& bad, int64_t count)
        {
            bad_element_count += count;
            for(auto& el : bad)
                if(bad_elements.size() < kMaxReported)
                    bad_elements.push_back(el);
        }

        // Compares one batch's probe sums with the expected ones and locates mismatched
        // elements where a bad row meets a bad column. stored_at reads one element of D.
        template <typename StoredAt>
        void compare(const FastCheckExpected&     e,
                     const BatchInputs&           in,
                     int64_t                      b,
                     const std::vector<uint64_t>& got_row,
                     const std::vector<uint64_t>& got_col,
                     StoredAt&&                   stored_at)
        {
            std::vector<int64_t> bad_rows, bad_cols;
            for(int64_t i = 0; i < p.M; i++)
                if(e.row_sums[size_t(b)][size_t(i)] != got_row[size_t(i)])
                    bad_rows.push_back(i);
            for(int64_t j = 0; j < p.N; j++)
                if(e.col_sums[size_t(b)][size_t(j)] != got_col[size_t(j)])
                    bad_cols.push_back(j);
            bad_row_count += int64_t(bad_rows.size());
            bad_col_count += int64_t(bad_cols.size());

            const size_t search = 64;
            for(size_t a = 0;
                a < std::min(search, bad_rows.size()) && located.size() < kMaxReported;
                a++)
                for(size_t c = 0;
                    c < std::min(search, bad_cols.size()) && located.size() < kMaxReported;
                    c++)
                {
                    int64_t i = bad_rows[a], j = bad_cols[c];
                    double  stored   = stored_at(i, j);
                    double  expected = round_to_type(double(in.exact(i, j)), p.D.type);
                    if(!(expected == stored))
                        located.push_back({b, i, j, expected, stored});
                }
        }

        FastCheckResult finish() const
        {
            FastCheckResult    result;
            std::ostringstream msg;
            auto               fail = [&](const std::string& text) {
                result.passed = false;
                msg << text << "\n";
            };
            if(over_budget)
                fail("fast_check stopped: more than " + std::to_string(slow_budget)
                     + " elements of D are outside the range its type stores exactly, and each "
                       "one costs a K-length dot product. Use a 32-bit output type, reduce K, "
                       "alpha or the scale factors, or use unit_check for this shape.");
            if(beyond_compute_range > 0)
                fail(std::to_string(beyond_compute_range.load())
                     + " elements of D exceed the range the compute type holds exactly, so their "
                       "GPU values depend on summation order. Reduce K, alpha or the scale "
                       "factors, or use unit_check for this shape.");
            if(bad_element_count > 0)
            {
                std::ostringstream s;
                s << bad_element_count << " elements of D are wrong or not finite:";
                for(auto& el : bad_elements)
                    s << "\n  " << format_offset(p, el);
                fail(s.str());
            }
            if(bad_row_count > 0 || bad_col_count > 0)
            {
                std::ostringstream s;
                s << "Probe sums disagree in " << bad_row_count << " rows and " << bad_col_count
                  << " columns of D.";
                if(!located.empty())
                {
                    s << " Mismatched elements found where they cross:";
                    for(auto& el : located)
                        s << "\n  " << format_offset(p, el);
                }
                fail(s.str());
            }
            result.message = msg.str();
            return result;
        }
    };
} // namespace

bool fast_check_supported_type(hipDataType type, std::string* why)
{
    if(element_size(type) != 0)
        return true;
    if(why)
        *why = "fast_check does not support data type " + std::to_string(int(type));
    return false;
}

uint64_t fast_check_probe(uint64_t seed, uint64_t index)
{
    // Uniform in [1, P): zero would leave a row or column out of the check.
    return 1 + splitmix64(splitmix64(seed) ^ index) % (P - 1);
}

std::string fast_check_describe_buffers(const std::vector<FastCheckBuffer>& buffers)
{
    constexpr uint64_t four_gib = uint64_t(1) << 32;
    std::ostringstream s;
    s << "Buffer placement:";
    for(const auto& b : buffers)
    {
        if(!b.base || !b.bytes)
            continue;
        uint64_t begin = uint64_t(reinterpret_cast<uintptr_t>(b.base));
        uint64_t end   = begin + b.bytes; // one past the last byte
        s << "\n  " << b.name << ": 0x" << std::hex << begin << " to 0x" << end << std::dec << " ("
          << b.bytes << " bytes)";
        uint64_t boundary = (begin / four_gib + 1) * four_gib;
        if(boundary < end)
            s << ", crosses a 4 GiB boundary at 0x" << std::hex << boundary << std::dec;
    }
    return s.str();
}

FastCheckExpected fast_check_expected(const FastCheckProblem& p)
{
    FastCheckExpected  e;
    std::ostringstream msg;
    auto               fail = [&](const std::string& text) {
        e.status.passed = false;
        msg << text << "\n";
    };
    auto finish = [&]() -> FastCheckExpected {
        e.status.message = msg.str();
        return std::move(e);
    };

    for(const auto* m : {&p.A, &p.B, &p.C, &p.D})
    {
        std::string why;
        if(!fast_check_supported_type(m->type, &why))
            fail(why);
    }
    if(exact_limit(p.compute_type) == 0)
        fail("fast_check does not support compute type " + std::to_string(int(p.compute_type)));
    if(!is_exact_integer(p.alpha) || !is_exact_integer(p.beta))
        fail("fast_check requires integer alpha and beta");
    if(!is_exact_integer(p.scale_c) || !is_exact_integer(p.scale_d))
        fail("fast_check requires integer scaleC and scaleD");
    if(!e.status.passed)
        return finish();

    // An empty output needs neither input fingerprints nor operand storage.
    if(p.M == 0 || p.N == 0 || p.batch_count == 0)
        return finish();

    const int64_t M = p.M, N = p.N, K = p.K;
    // Multiplies two integer factors, failing rather than overflowing: a combined factor of 2^61
    // or more is far outside every compute type's exact range anyway.
    auto times = [](int64_t a, int64_t b, int64_t& out) {
        if(std::fabs(double(a) * double(b)) >= 0x1p61)
            return false;
        out = a * b;
        return true;
    };
    const char* too_large = "fast_check requires alpha, beta, the scales and the bias to combine "
                            "to values below 2^61";
    if(!times(int64_t(p.alpha), int64_t(p.scale_d), e.alpha)
       || !times(int64_t(p.beta), int64_t(p.scale_c), e.beta)
       || !times(e.beta, int64_t(p.scale_d), e.beta))
    {
        fail(too_large);
        return finish();
    }

    // Loads entry i of a scale that is one value or a vector, and requires an integer.
    auto load_scale = [&](const void* v, bool vector, int64_t i, int64_t& out) {
        double x = load(v, p.scale_ab_type, vector ? size_t(i) : 0);
        if(!is_exact_integer(x))
            return false;
        out = int64_t(x);
        return true;
    };

    e.scale.assign(size_t(M), 1);
    if(p.scale_alpha_vec)
        for(int64_t i = 0; i < M; i++)
        {
            double s = load(p.scale_alpha_vec, p.scale_alpha_vec_type, size_t(i));
            if(!is_exact_integer(s))
            {
                fail("fast_check requires integer scaleAlpha_vector entries");
                return finish();
            }
            e.scale[size_t(i)] = int64_t(s);
        }
    if(p.scale_a)
        for(int64_t i = 0; i < M; i++)
        {
            int64_t a = 0;
            if(!load_scale(p.scale_a, p.scale_a_vector, i, a))
            {
                fail("fast_check requires integer scaleA entries");
                return finish();
            }
            if(!times(e.scale[size_t(i)], a, e.scale[size_t(i)]))
            {
                fail(too_large);
                return finish();
            }
        }
    e.col_scale.assign(size_t(N), 1);
    if(p.scale_b)
        for(int64_t j = 0; j < N; j++)
            if(!load_scale(p.scale_b, p.scale_b_vector, j, e.col_scale[size_t(j)]))
            {
                fail("fast_check requires integer scaleB entries");
                return finish();
            }
    double col_scale_max = 0;
    for(int64_t c : e.col_scale)
        col_scale_max = std::max(col_scale_max, std::fabs(double(c)));

    // Row probe r (length N) and column probe t (length M). rs = r .* scale_b.
    e.row_probe.resize(size_t(N));
    e.col_probe.resize(size_t(M));
    uint64_t              sum_r = 0;
    std::vector<uint64_t> rs(static_cast<size_t>(N));
    for(int64_t j = 0; j < N; j++)
    {
        e.row_probe[size_t(j)] = fast_check_probe(2 * p.seed, uint64_t(j));
        sum_r                  = mod_add(sum_r, e.row_probe[size_t(j)]);
        rs[size_t(j)]          = mod_mul(e.row_probe[size_t(j)], to_mod(e.col_scale[size_t(j)]));
    }
    std::vector<uint64_t> ts(static_cast<size_t>(M));
    for(int64_t i = 0; i < M; i++)
    {
        e.col_probe[size_t(i)] = fast_check_probe(2 * p.seed + 1, uint64_t(i));
        ts[size_t(i)]          = mod_mul(e.col_probe[size_t(i)], to_mod(e.scale[size_t(i)]));
    }

    const uint64_t    alpha_m = to_mod(e.alpha), beta_m = to_mod(e.beta);
    const bool        use_c = p.beta != 0;
    std::atomic<bool> non_integer_input{false};

    e.bias.resize(size_t(p.batch_count));
    e.row_sums.resize(size_t(p.batch_count));
    e.col_sums.resize(size_t(p.batch_count));
    for(int64_t b = 0; b < p.batch_count; b++)
    {
        auto& bias = e.bias[size_t(b)];
        bias.assign(size_t(M), 0);
        if(p.bias)
        {
            const void* base = static_cast<const char*>(p.bias)
                               + size_t(b) * size_t(p.bias_stride) * element_size(p.bias_type);
            for(int64_t i = 0; i < M; i++)
            {
                double v = load(base, p.bias_type, size_t(i));
                if(!is_exact_integer(v))
                {
                    fail("fast_check requires integer bias entries");
                    return finish();
                }
                if(!times(int64_t(v), int64_t(p.scale_d), bias[size_t(i)]))
                {
                    fail(too_large);
                    return finish();
                }
            }
        }

        const View opA = batch_view(p.A, b, p.transA); // M x K
        const View opB = batch_view(p.B, b, p.transB); // K x N
        const View C   = batch_view(p.C, b, false); // M x N

        // |any partial sum of row i| <= sum over k of |A(i, k)| * max over j of |B(k, j)|.
        {
            const std::vector<double> b_max = abs_row_max(opB, K, N);
            const std::vector<double> s     = abs_mul_right(opA, M, K, b_max);
            const std::vector<double> c_max
                = use_c ? abs_row_max(C, M, N) : std::vector<double>(size_t(M), 0);
            for(int64_t i = 0; i < M; i++)
            {
                const double scaled = std::fabs(double(e.alpha) * double(e.scale[size_t(i)]))
                                      * col_scale_max * s[size_t(i)];
                e.max_partial = std::max({e.max_partial, s[size_t(i)], scaled});
                e.max_result  = std::max(e.max_result,
                                        scaled + std::fabs(double(e.beta)) * c_max[size_t(i)]
                                            + std::fabs(double(bias[size_t(i)])));
            }
        }

        // Expected D*r = alpha * diag(scale) * opA * (opB * (r .* col_scale)) + beta * C * r
        // + bias * sum(r), and t^T*D = alpha * (((t .* scale)^T * opA) * opB) .* col_scale
        // + beta * t^T * C + (t^T * bias) * 1^T, with alpha, beta and bias already scaled.
        std::vector<uint64_t> y, z, cr, u, w, ct;
        mul_right(opB, K, N, rs, y, non_integer_input);
        mul_right(opA, M, K, y, z, non_integer_input);
        mul_right(opA.transposed(), K, M, ts, u, non_integer_input);
        mul_right(opB.transposed(), N, K, u, w, non_integer_input);
        if(use_c)
        {
            mul_right(C, M, N, e.row_probe, cr, non_integer_input);
            mul_right(C.transposed(), N, M, e.col_probe, ct, non_integer_input);
        }
        if(non_integer_input)
        {
            std::string where;
            auto        find = [&](const View& m, int64_t rows, int64_t cols, const char* name) {
                for(int64_t c = 0; c < cols && where.empty(); c++)
                    for(int64_t r = 0; r < rows && where.empty(); r++)
                        if(!is_exact_integer(m.at(r, c)))
                        {
                            std::ostringstream s;
                            s << " (first: batch " << b << ", " << name << "(" << r << ", " << c
                              << ") = " << m.at(r, c) << ")";
                            where = s.str();
                        }
            };
            find(opA, M, K, "op(A)");
            find(opB, K, N, "op(B)");
            if(use_c)
                find(C, M, N, "C");
            fail("fast_check found a non-integer value in A, B or C" + where
                 + "; it requires integer_exact initialization");
            return finish();
        }

        uint64_t sum_t_bias = 0;
        for(int64_t i = 0; i < M; i++)
            sum_t_bias
                = mod_add(sum_t_bias, mod_mul(e.col_probe[size_t(i)], to_mod(bias[size_t(i)])));

        auto& rows = e.row_sums[size_t(b)];
        rows.resize(size_t(M));
        for(int64_t i = 0; i < M; i++)
        {
            uint64_t v = mod_mul(mod_mul(alpha_m, to_mod(e.scale[size_t(i)])), z[size_t(i)]);
            if(use_c)
                v = mod_add(v, mod_mul(beta_m, cr[size_t(i)]));
            rows[size_t(i)] = mod_add(v, mod_mul(to_mod(bias[size_t(i)]), sum_r));
        }
        auto& cols = e.col_sums[size_t(b)];
        cols.resize(size_t(N));
        for(int64_t j = 0; j < N; j++)
        {
            uint64_t v
                = mod_add(mod_mul(mod_mul(alpha_m, to_mod(e.col_scale[size_t(j)])), w[size_t(j)]),
                          sum_t_bias);
            if(use_c)
                v = mod_add(v, mod_mul(beta_m, ct[size_t(j)]));
            cols[size_t(j)] = v;
        }
    }

    const double acc_limit = exact_limit(p.compute_type);
    if(e.max_partial >= acc_limit || e.max_result >= acc_limit)
    {
        std::ostringstream s;
        s << "fast_check refuses this configuration: a partial sum can reach " << e.max_partial
          << " and a result " << e.max_result << ", but the compute type holds integers exactly "
          << "only below " << acc_limit << ", so D would depend on summation order. Reduce K, "
          << "alpha or the scale factors, or use integer_exact_pattern ternary or sparse_k.";
        fail(s.str());
    }
    return finish();
}

FastCheckResult fast_check_result(const FastCheckProblem& p, const FastCheckExpected& e)
{
    if(!e.status.passed)
        return e.status;
    if(p.M == 0 || p.N == 0 || p.batch_count == 0)
        return {};

    Report        rep(p);
    const int64_t M = p.M, N = p.N;
    const double  out_limit = fast_limit(p.D.type);

    for(int64_t b = 0; b < p.batch_count && !rep.over_budget; b++)
    {
        const BatchInputs in = batch_inputs(p, e, b);
        const View        D  = batch_view(p.D, b, false);

        // D*r and t^T*D in one pass over D. Each thread owns a block of rows; column sums are
        // reduced across blocks afterwards.
        const int64_t                        block    = std::max<int64_t>(64, (M + 255) / 256);
        const int64_t                        n_blocks = (M + block - 1) / block;
        std::vector<uint64_t>                got_row(static_cast<size_t>(M));
        std::vector<uint64_t>                got_col_partial(size_t(n_blocks) * size_t(N));
        std::vector<std::vector<BadElement>> bad_per_block(static_cast<size_t>(n_blocks));
        std::vector<int64_t>                 bad_count_per_block(size_t(n_blocks), 0);
        const uint64_t*                      r = e.row_probe.data();
        const uint64_t*                      t = e.col_probe.data();

        dispatch_type(p.D.type, [&](auto typed_null) {
            using T          = std::remove_const_t<std::remove_pointer_t<decltype(typed_null)>>;
            const T* d_batch = static_cast<const T*>(D.base);

#pragma omp parallel for schedule(dynamic)
            for(int64_t blk = 0; blk < n_blocks; blk++)
            {
                const int64_t       i0 = blk * block, i1 = std::min(M, i0 + block);
                std::vector<ModAcc> row_acc(size_t(i1 - i0));
                for(int64_t j = 0; j < N; j++)
                {
                    ModAcc   col_acc;
                    const T* d_col = d_batch + size_t(j) * size_t(p.D.ld);
                    for(int64_t i = i0; i < i1; i++)
                    {
                        double  stored = to_double(d_col[i]);
                        int64_t value  = 0;
                        // A NaN fails the range test and takes the slow path.
                        bool fast = std::fabs(stored) < out_limit;
                        if(fast)
                        {
                            value = int64_t(stored);
                            fast  = double(value) == stored;
                        }
                        if(!fast)
                            value = rep.slow_element(in,
                                                     b,
                                                     i,
                                                     j,
                                                     stored,
                                                     bad_per_block[size_t(blk)],
                                                     bad_count_per_block[size_t(blk)]);
                        uint64_t vm = to_mod(value);
                        row_acc[size_t(i - i0)].add(vm, r[j]);
                        col_acc.add(vm, t[i]);
                    }
                    got_col_partial[size_t(blk) * size_t(N) + size_t(j)] = col_acc.value();
                }
                for(int64_t i = i0; i < i1; i++)
                    got_row[size_t(i)] = row_acc[size_t(i - i0)].value();
            }
        });
        if(rep.over_budget)
            break;

        for(int64_t blk = 0; blk < n_blocks; blk++)
            rep.add_bad(bad_per_block[size_t(blk)], bad_count_per_block[size_t(blk)]);

        std::vector<uint64_t> got_col(size_t(N), 0);
        for(int64_t j = 0; j < N; j++)
            for(int64_t blk = 0; blk < n_blocks; blk++)
                got_col[size_t(j)] = mod_add(got_col[size_t(j)],
                                             got_col_partial[size_t(blk) * size_t(N) + size_t(j)]);

        rep.compare(e, in, b, got_row, got_col, [&](int64_t i, int64_t j) { return D.at(i, j); });
    }
    return rep.finish();
}

FastCheckResult fast_check_gemm(const FastCheckProblem& p)
{
    return fast_check_result(p, fast_check_expected(p));
}

// ----------------------------------------------------------------------------------------------
// Device helpers
// ----------------------------------------------------------------------------------------------

namespace
{
    __device__ inline uint64_t device_mod_mul(uint64_t a, uint64_t b)
    {
        // a * b = hi * 2^64 + lo with hi below 2^58, and 2^64 == 8 (mod P).
        uint64_t lo = a * b;
        uint64_t hi = __umul64hi(a, b);
        uint64_t x  = (lo & P) + (lo >> 61) + (hi << 3);
        x           = (x & P) + (x >> 61);
        return x >= P ? x - P : x;
    }

    // Reads one element of D as an integer when it is an exact integer below limit in magnitude.
    template <typename T>
    __device__ inline bool device_exact_value(T x, double limit, int64_t& v, double& stored)
    {
        if constexpr(std::is_same_v<T, hipblasLtHalf> || std::is_same_v<T, hip_bfloat16>)
            stored = float(x);
        else
            stored = double(x);
        if(!(fabs(stored) < limit))
            return false;
        v = int64_t(stored);
        return double(v) == stored;
    }

    // partial[chunk * M + i] = sum over the chunk's columns j of D(i, j) * r[j]. Elements that
    // are not exact integers below limit add nothing and are listed for the host instead.
    template <typename T>
    __global__ void row_probe_kernel(const T*            D,
                                     int64_t             M,
                                     int64_t             N,
                                     int64_t             ld,
                                     int64_t             cols_per_chunk,
                                     double              limit,
                                     const uint64_t*     r,
                                     uint64_t*           partial,
                                     unsigned long long* special_count,
                                     uint64_t*           special_offset,
                                     double*             special_value,
                                     uint64_t            special_capacity)
    {
        int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if(i >= M)
            return;
        int64_t  j0  = int64_t(blockIdx.y) * cols_per_chunk;
        int64_t  j1  = j0 + cols_per_chunk < N ? j0 + cols_per_chunk : N;
        uint64_t acc = 0;
        for(int64_t j = j0; j < j1; j++)
        {
            int64_t v;
            double  stored;
            if(device_exact_value(D[size_t(j) * size_t(ld) + size_t(i)], limit, v, stored))
                acc = mod_add(acc, device_mod_mul(to_mod(v), r[j]));
            else
            {
                unsigned long long slot = atomicAdd(special_count, 1ull);
                if(slot < special_capacity)
                {
                    special_offset[slot] = uint64_t(j) * uint64_t(ld) + uint64_t(i);
                    special_value[slot]  = stored;
                }
            }
        }
        partial[size_t(blockIdx.y) * size_t(M) + size_t(i)] = acc;
    }

    // partial[chunk * N + j] = sum over the chunk's rows i of t[i] * D(i, j), skipping the same
    // elements row_probe_kernel lists.
    template <typename T>
    __global__ void col_probe_kernel(const T*        D,
                                     int64_t         M,
                                     int64_t         N,
                                     int64_t         ld,
                                     int64_t         rows_per_chunk,
                                     double          limit,
                                     const uint64_t* t,
                                     uint64_t*       partial)
    {
        __shared__ uint64_t sums[256];
        int64_t             j   = blockIdx.x;
        int64_t             i0  = int64_t(blockIdx.y) * rows_per_chunk;
        int64_t             i1  = i0 + rows_per_chunk < M ? i0 + rows_per_chunk : M;
        uint64_t            acc = 0;
        const T*            col = D + size_t(j) * size_t(ld);
        for(int64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x)
        {
            int64_t v;
            double  stored;
            if(device_exact_value(col[i], limit, v, stored))
                acc = mod_add(acc, device_mod_mul(to_mod(v), t[i]));
        }
        sums[threadIdx.x] = acc;
        __syncthreads();
        for(unsigned s = blockDim.x / 2; s > 0; s /= 2)
        {
            if(threadIdx.x < s)
                sums[threadIdx.x] = mod_add(sums[threadIdx.x], sums[threadIdx.x + s]);
            __syncthreads();
        }
        if(threadIdx.x == 0)
            partial[size_t(blockIdx.y) * size_t(N) + size_t(j)] = sums[0];
    }

    __device__ inline bool in_region(
        size_t idx, int64_t rows, int64_t cols, int64_t ld, int64_t stride, int64_t batch_count)
    {
        size_t b = 0, rem = idx;
        if(batch_count > 1)
        {
            b   = idx / size_t(stride);
            rem = idx - b * size_t(stride);
        }
        if(b >= size_t(batch_count))
            return false;
        size_t j = rem / size_t(ld);
        size_t i = rem - j * size_t(ld);
        return j < size_t(cols) && i < size_t(rows);
    }

    template <typename U>
    __global__ void poison_kernel(U*      p,
                                  size_t  total,
                                  int64_t rows,
                                  int64_t cols,
                                  int64_t ld,
                                  int64_t stride,
                                  int64_t batch,
                                  U       value)
    {
        for(size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total;
            idx += size_t(gridDim.x) * blockDim.x)
            if(!in_region(idx, rows, cols, ld, stride, batch))
                p[idx] = value;
    }

    // counters: [0] padding mismatches, [1] first padding mismatch offset,
    //           [2] unwritten elements,  [3] first unwritten offset
    template <typename U>
    __global__ void scan_kernel(const U*            p,
                                size_t              total,
                                int64_t             rows,
                                int64_t             cols,
                                int64_t             ld,
                                int64_t             stride,
                                int64_t             batch,
                                U                   padding_value,
                                bool                check_unwritten,
                                U                   sentinel,
                                unsigned long long* counters)
    {
        for(size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < total;
            idx += size_t(gridDim.x) * blockDim.x)
        {
            U v = p[idx];
            if(in_region(idx, rows, cols, ld, stride, batch))
            {
                if(check_unwritten && v == sentinel)
                {
                    atomicAdd(&counters[2], 1ull);
                    atomicMin(&counters[3], (unsigned long long)idx);
                }
            }
            else if(v != padding_value)
            {
                atomicAdd(&counters[0], 1ull);
                atomicMin(&counters[1], (unsigned long long)idx);
            }
        }
    }

    uint64_t poison_bits(hipDataType t)
    {
        uint64_t bits = 0;
        auto     put  = [&](auto value) { std::memcpy(&bits, &value, sizeof(value)); };
        switch(t)
        {
        case HIP_R_32F:
            put(float(kFastCheckPoisonValue));
            break;
        case HIP_R_64F:
            put(double(kFastCheckPoisonValue));
            break;
        case HIP_R_16F:
            put(hipblasLtHalf(kFastCheckPoisonValue));
            break;
        case HIP_R_16BF:
            put(hip_bfloat16(kFastCheckPoisonValue));
            break;
        case HIP_R_8F_E4M3_FNUZ:
            put(hipblaslt_f8_fnuz(kFastCheckPoisonValue));
            break;
        case HIP_R_8F_E5M2_FNUZ:
            put(hipblaslt_bf8_fnuz(kFastCheckPoisonValue));
            break;
        case HIP_R_8F_E4M3:
            put(hipblaslt_f8(kFastCheckPoisonValue));
            break;
        case HIP_R_8F_E5M2:
            put(hipblaslt_bf8(kFastCheckPoisonValue));
            break;
        case HIP_R_32I:
            put(int32_t(kFastCheckPoisonValue));
            break;
        case HIP_R_8I:
            put(int8_t(kFastCheckPoisonValue));
            break;
        default:
            break;
        }
        return bits;
    }

    uint64_t sentinel_bits(hipDataType t)
    {
        switch(t)
        {
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_8F_E5M2_FNUZ:
        case HIP_R_8I:
            return 0x80;
        case HIP_R_32I:
            return 0x80000000;
        default:
        {
            const size_t es = element_size(t);
            return es >= 8 ? ~uint64_t(0) : (uint64_t(1) << (8 * es)) - 1;
        }
        }
    }

    bool sentinel_is_nan(hipDataType t)
    {
        uint64_t bits = sentinel_bits(t);
        return std::isnan(load(&bits, t, 0));
    }

    unsigned grid_for(size_t total)
    {
        return unsigned(std::min<size_t>((total + 255) / 256, 16384));
    }

    // Writes `value` into every element outside the rows x cols x batch region.
    template <typename U>
    hipError_t launch_poison(
        const FastCheckMatrix& m, int64_t batch, size_t total, uint64_t value, hipStream_t stream)
    {
        // HIP rejects a launch with an empty grid.
        if(total == 0)
            return hipSuccess;
        hipLaunchKernelGGL(poison_kernel<U>,
                           dim3(grid_for(total)),
                           dim3(256),
                           0,
                           stream,
                           static_cast<U*>(const_cast<void*>(m.data)),
                           total,
                           m.rows,
                           m.cols,
                           m.ld,
                           m.stride,
                           batch,
                           U(value));
        return hipGetLastError();
    }

    // Leaves a buffer of a type fast_check does not support alone and says so.
    hipError_t fill_outside_region(
        const FastCheckMatrix& m, int64_t batch, size_t total, uint64_t value, hipStream_t stream)
    {
        switch(element_size(m.type))
        {
        case 1:
            return launch_poison<uint8_t>(m, batch, total, value, stream);
        case 2:
            return launch_poison<uint16_t>(m, batch, total, value, stream);
        case 4:
            return launch_poison<uint32_t>(m, batch, total, value, stream);
        case 8:
            return launch_poison<uint64_t>(m, batch, total, value, stream);
        default:
            return hipErrorInvalidValue;
        }
    }

    template <typename U>
    hipError_t launch_scan(const FastCheckMatrix& m,
                           int64_t                batch,
                           size_t                 total,
                           bool                   expect_poison,
                           unsigned long long*    d_counters,
                           hipStream_t            stream)
    {
        U sentinel      = U(sentinel_bits(m.type));
        U padding_value = expect_poison ? U(poison_bits(m.type)) : sentinel;
        if(total == 0)
            return hipSuccess;
        hipLaunchKernelGGL(scan_kernel<U>,
                           dim3(grid_for(total)),
                           dim3(256),
                           0,
                           stream,
                           static_cast<const U*>(m.data),
                           total,
                           m.rows,
                           m.cols,
                           m.ld,
                           m.stride,
                           batch,
                           padding_value,
                           !expect_poison && sentinel_is_nan(m.type),
                           sentinel,
                           d_counters);
        return hipGetLastError();
    }

    std::string describe_offset(const FastCheckMatrix& m, int64_t batch, uint64_t offset)
    {
        std::ostringstream s;
        s << "element offset " << offset << " = 0x" << std::hex << offset << std::dec;
        uint64_t b = 0, rem = offset;
        if(batch > 1)
        {
            b   = offset / uint64_t(m.stride);
            rem = offset - b * uint64_t(m.stride);
        }
        s << " (batch " << b << ", row " << rem % uint64_t(m.ld) << ", col " << rem / uint64_t(m.ld)
          << ")";
        return s.str();
    }

    // Device memory released when it goes out of scope.
    template <typename T>
    struct DeviceArray
    {
        T* ptr = nullptr;
        explicit DeviceArray(size_t n)
        {
            if(n && hipMalloc(&ptr, n * sizeof(T)) != hipSuccess)
                ptr = nullptr;
        }
        ~DeviceArray()
        {
            (void)hipFree(ptr);
        }
        DeviceArray(const DeviceArray&)            = delete;
        DeviceArray& operator=(const DeviceArray&) = delete;
    };

    // Copies D's region to the host and runs the host pass; used when the device pass cannot.
    FastCheckResult
        device_fallback(const FastCheckProblem& p, const FastCheckExpected& e, hipStream_t stream)
    {
        const size_t bytes
            = size_t(p.M) * size_t(p.N) * size_t(p.batch_count) * element_size(p.D.type);
        std::unique_ptr<char[]> host(new char[bytes]);
        hipError_t err = fast_check_copy_region_to_host(host.get(), p.D, p.batch_count, stream);
        if(err != hipSuccess)
            return {false, std::string("fast_check could not copy D: ") + hipGetErrorString(err)};
        FastCheckProblem hp = p;
        hp.D.data           = host.get();
        hp.D.ld             = p.M;
        hp.D.stride         = p.M * p.N;
        hp.device_ldd       = p.device_ldd ? p.device_ldd : p.D.ld;
        hp.device_stride_d  = p.device_stride_d ? p.device_stride_d : p.D.stride;
        return fast_check_result(hp, e);
    }
} // namespace

FastCheckResult fast_check_result_device(const FastCheckProblem&  p,
                                         const FastCheckExpected& e,
                                         hipStream_t              stream)
{
    if(!e.status.passed)
        return e.status;
    // Chunk sizing divides by M and N; empty outputs require no device work.
    if(p.M == 0 || p.N == 0 || p.batch_count == 0)
        return {};

    // The device pass handles the types the GPU converts natively; fp8 outputs use the host pass.
    switch(p.D.type)
    {
    case HIP_R_32F:
    case HIP_R_64F:
    case HIP_R_16F:
    case HIP_R_16BF:
    case HIP_R_32I:
    case HIP_R_8I:
        break;
    default:
        return device_fallback(p, e, stream);
    }

    const int64_t M = p.M, N = p.N;
    const double  out_limit = fast_limit(p.D.type);
    Report        rep(p);
    const size_t  es = element_size(p.D.type);

    // Chunk the columns of the row probe and the rows of the column probe for parallelism, with
    // the partial sums capped at about 2^25 entries (256 MiB).
    const int64_t partial_cap = int64_t(1) << 25;
    int64_t       col_chunks
        = std::clamp<int64_t>((N + 255) / 256, 1, std::max<int64_t>(1, partial_cap / M));
    col_chunks                   = std::min<int64_t>(col_chunks, 65535);
    const int64_t cols_per_chunk = (N + col_chunks - 1) / col_chunks;
    col_chunks                   = (N + cols_per_chunk - 1) / cols_per_chunk;
    int64_t row_chunks
        = std::clamp<int64_t>((M + 4095) / 4096, 1, std::max<int64_t>(1, partial_cap / N));
    row_chunks                   = std::min<int64_t>(row_chunks, 65535);
    const int64_t rows_per_chunk = (M + row_chunks - 1) / row_chunks;
    row_chunks                   = (M + rows_per_chunk - 1) / rows_per_chunk;
    const uint64_t special_capacity
        = uint64_t(std::min<int64_t>(rep.slow_budget, int64_t(1) << 22));

    DeviceArray<uint64_t>           d_r{size_t(N)}, d_t{size_t(M)};
    DeviceArray<uint64_t>           d_row_partial(size_t(col_chunks) * size_t(M));
    DeviceArray<uint64_t>           d_col_partial(size_t(row_chunks) * size_t(N));
    DeviceArray<unsigned long long> d_special_count(1);
    DeviceArray<uint64_t>           d_special_offset(special_capacity);
    DeviceArray<double>             d_special_value(special_capacity);
    if(!d_r.ptr || !d_t.ptr || !d_row_partial.ptr || !d_col_partial.ptr || !d_special_count.ptr
       || !d_special_offset.ptr || !d_special_value.ptr)
        return device_fallback(p, e, stream);

    hipError_t err = hipMemcpyAsync(
        d_r.ptr, e.row_probe.data(), size_t(N) * sizeof(uint64_t), hipMemcpyHostToDevice, stream);
    if(err == hipSuccess)
        err = hipMemcpyAsync(d_t.ptr,
                             e.col_probe.data(),
                             size_t(M) * sizeof(uint64_t),
                             hipMemcpyHostToDevice,
                             stream);
    if(err != hipSuccess)
        return {false,
                std::string("fast_check could not copy its probes: ") + hipGetErrorString(err)};

    std::vector<uint64_t> row_partial(size_t(col_chunks) * size_t(M));
    std::vector<uint64_t> col_partial(size_t(row_chunks) * size_t(N));
    std::vector<uint64_t> special_offset;
    std::vector<double>   special_value;

    for(int64_t b = 0; b < p.batch_count && !rep.over_budget; b++)
    {
        const void* d_batch
            = static_cast<const char*>(p.D.data) + size_t(b) * size_t(p.D.stride) * es;
        unsigned long long special_count = 0;
        err = hipMemsetAsync(d_special_count.ptr, 0, sizeof(unsigned long long), stream);

        dispatch_type(p.D.type, [&](auto typed_null) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(typed_null)>>;
            if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>
                         || std::is_same_v<T, hipblasLtHalf> || std::is_same_v<T, hip_bfloat16>
                         || std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>)
            {
                hipLaunchKernelGGL(row_probe_kernel<T>,
                                   dim3(unsigned((M + 255) / 256), unsigned(col_chunks)),
                                   dim3(256),
                                   0,
                                   stream,
                                   static_cast<const T*>(d_batch),
                                   M,
                                   N,
                                   p.D.ld,
                                   cols_per_chunk,
                                   out_limit,
                                   d_r.ptr,
                                   d_row_partial.ptr,
                                   d_special_count.ptr,
                                   d_special_offset.ptr,
                                   d_special_value.ptr,
                                   special_capacity);
                hipLaunchKernelGGL(col_probe_kernel<T>,
                                   dim3(unsigned(N), unsigned(row_chunks)),
                                   dim3(256),
                                   0,
                                   stream,
                                   static_cast<const T*>(d_batch),
                                   M,
                                   N,
                                   p.D.ld,
                                   rows_per_chunk,
                                   out_limit,
                                   d_t.ptr,
                                   d_col_partial.ptr);
            }
        });
        if(err == hipSuccess)
            err = hipGetLastError();
        if(err == hipSuccess)
            err = hipMemcpyAsync(row_partial.data(),
                                 d_row_partial.ptr,
                                 row_partial.size() * sizeof(uint64_t),
                                 hipMemcpyDeviceToHost,
                                 stream);
        if(err == hipSuccess)
            err = hipMemcpyAsync(col_partial.data(),
                                 d_col_partial.ptr,
                                 col_partial.size() * sizeof(uint64_t),
                                 hipMemcpyDeviceToHost,
                                 stream);
        if(err == hipSuccess)
            err = hipMemcpyAsync(&special_count,
                                 d_special_count.ptr,
                                 sizeof(special_count),
                                 hipMemcpyDeviceToHost,
                                 stream);
        if(err == hipSuccess)
            err = hipStreamSynchronize(stream);
        if(err != hipSuccess)
            return {false, std::string("fast_check device pass failed: ") + hipGetErrorString(err)};

        // Too many elements outside the exact range for the list: the host pass reports them.
        if(special_count > special_capacity)
            return device_fallback(p, e, stream);

        special_offset.resize(size_t(special_count));
        special_value.resize(size_t(special_count));
        if(special_count)
        {
            err = hipMemcpyAsync(special_offset.data(),
                                 d_special_offset.ptr,
                                 special_offset.size() * sizeof(uint64_t),
                                 hipMemcpyDeviceToHost,
                                 stream);
            if(err == hipSuccess)
                err = hipMemcpyAsync(special_value.data(),
                                     d_special_value.ptr,
                                     special_value.size() * sizeof(double),
                                     hipMemcpyDeviceToHost,
                                     stream);
            if(err == hipSuccess)
                err = hipStreamSynchronize(stream);
            if(err != hipSuccess)
                return {false,
                        std::string("fast_check could not copy its element list: ")
                            + hipGetErrorString(err)};
        }

        std::vector<uint64_t> got_row(size_t(M), 0), got_col(size_t(N), 0);
        for(int64_t c = 0; c < col_chunks; c++)
            for(int64_t i = 0; i < M; i++)
                got_row[size_t(i)]
                    = mod_add(got_row[size_t(i)], row_partial[size_t(c) * size_t(M) + size_t(i)]);
        for(int64_t c = 0; c < row_chunks; c++)
            for(int64_t j = 0; j < N; j++)
                got_col[size_t(j)]
                    = mod_add(got_col[size_t(j)], col_partial[size_t(c) * size_t(N) + size_t(j)]);

        // The listed elements entered neither probe sum on the device: check each one (a
        // K-length dot product, so in parallel) and add the exact value to both sums.
        const BatchInputs                    in        = batch_inputs(p, e, b);
        const int64_t                        n_special = int64_t(special_offset.size());
        const int64_t                        block     = 256;
        const int64_t                        n_blocks  = (n_special + block - 1) / block;
        std::vector<int64_t>                 values(special_offset.size(), 0);
        std::vector<std::vector<BadElement>> bad_per_block(static_cast<size_t>(n_blocks));
        std::vector<int64_t>                 bad_count_per_block(static_cast<size_t>(n_blocks), 0);
#pragma omp parallel for schedule(dynamic)
        for(int64_t blk = 0; blk < n_blocks; blk++)
            for(int64_t s = blk * block; s < std::min(n_special, (blk + 1) * block); s++)
            {
                if(rep.over_budget)
                    break;
                values[size_t(s)]
                    = rep.slow_element(in,
                                       b,
                                       int64_t(special_offset[size_t(s)] % uint64_t(p.D.ld)),
                                       int64_t(special_offset[size_t(s)] / uint64_t(p.D.ld)),
                                       special_value[size_t(s)],
                                       bad_per_block[size_t(blk)],
                                       bad_count_per_block[size_t(blk)]);
            }
        if(rep.over_budget)
            break;
        for(int64_t blk = 0; blk < n_blocks; blk++)
            rep.add_bad(bad_per_block[size_t(blk)], bad_count_per_block[size_t(blk)]);
        for(size_t s = 0; s < special_offset.size(); s++)
        {
            int64_t  i         = int64_t(special_offset[s] % uint64_t(p.D.ld));
            int64_t  j         = int64_t(special_offset[s] / uint64_t(p.D.ld));
            uint64_t vm        = to_mod(values[s]);
            got_row[size_t(i)] = mod_add(got_row[size_t(i)], mod_mul(vm, e.row_probe[size_t(j)]));
            got_col[size_t(j)] = mod_add(got_col[size_t(j)], mod_mul(vm, e.col_probe[size_t(i)]));
        }

        rep.compare(e, in, b, got_row, got_col, [&](int64_t i, int64_t j) {
            std::vector<char> one(es);
            (void)hipMemcpy(one.data(),
                            static_cast<const char*>(d_batch)
                                + (size_t(j) * size_t(p.D.ld) + size_t(i)) * es,
                            es,
                            hipMemcpyDeviceToHost);
            return load(one.data(), p.D.type, 0);
        });
    }
    return rep.finish();
}

uint64_t fast_check_sentinel_bits(hipDataType type)
{
    return sentinel_bits(type);
}

hipError_t fast_check_fill_sentinel_device(void*       buffer,
                                           hipDataType type,
                                           size_t      elements,
                                           hipStream_t stream)
{
    // An empty region makes every element padding.
    return fill_outside_region(
        {buffer, type, 0, 0, 1, 0}, 1, elements, sentinel_bits(type), stream);
}

hipError_t fast_check_poison_padding_device(const FastCheckMatrix& m,
                                            int64_t                batch_count,
                                            size_t                 total_elements,
                                            hipStream_t            stream)
{
    return fill_outside_region(m, batch_count, total_elements, poison_bits(m.type), stream);
}

namespace
{
    // Runs scan_kernel and returns its counters: [0] padding mismatches, [1] first padding
    // mismatch, [2] unwritten elements, [3] first unwritten element.
    hipError_t run_scan(const FastCheckMatrix& m,
                        int64_t                batch_count,
                        size_t                 total_elements,
                        bool                   expect_poison,
                        hipStream_t            stream,
                        unsigned long long (&counters)[4])
    {
        unsigned long long              init[4] = {0, ~0ull, 0, ~0ull};
        DeviceArray<unsigned long long> d_counters(4);
        if(!d_counters.ptr)
            return hipErrorOutOfMemory;
        hipError_t err
            = hipMemcpyAsync(d_counters.ptr, init, sizeof(init), hipMemcpyHostToDevice, stream);
        if(err != hipSuccess)
            return err;
        // A failed launch, or an unsupported type, would otherwise leave the counters clean.
        err = hipErrorInvalidValue;
        switch(element_size(m.type))
        {
        case 1:
            err = launch_scan<uint8_t>(
                m, batch_count, total_elements, expect_poison, d_counters.ptr, stream);
            break;
        case 2:
            err = launch_scan<uint16_t>(
                m, batch_count, total_elements, expect_poison, d_counters.ptr, stream);
            break;
        case 4:
            err = launch_scan<uint32_t>(
                m, batch_count, total_elements, expect_poison, d_counters.ptr, stream);
            break;
        case 8:
            err = launch_scan<uint64_t>(
                m, batch_count, total_elements, expect_poison, d_counters.ptr, stream);
            break;
        default:
            // An unsupported type would otherwise scan nothing and look clean.
            return hipErrorInvalidValue;
        }
        if(err != hipSuccess)
            return err;
        err = hipMemcpyAsync(
            counters, d_counters.ptr, sizeof(counters), hipMemcpyDeviceToHost, stream);
        if(err == hipSuccess)
            err = hipStreamSynchronize(stream);
        return err;
    }
} // namespace

size_t fast_check_element_size(hipDataType type)
{
    return element_size(type);
}

uint64_t fast_check_poison_bits(hipDataType type)
{
    return poison_bits(type);
}

FastCheckChanged fast_check_count_changed_device(const void* data,
                                                 hipDataType type,
                                                 size_t      elements,
                                                 hipStream_t stream)
{
    // An empty region makes every element padding, so the scan compares all of them with the
    // poison value.
    FastCheckMatrix    m{data, type, 0, 0, 1, 0};
    unsigned long long counters[4];
    FastCheckChanged   changed;
    if(run_scan(m, 1, elements, true, stream, counters) != hipSuccess)
    {
        changed.ok = false;
        return changed;
    }
    changed.count = counters[0];
    changed.first = counters[1];
    return changed;
}

FastCheckResult fast_check_scan_padding_device(const FastCheckMatrix& m,
                                               int64_t                batch_count,
                                               size_t                 total_elements,
                                               bool                   expect_poison,
                                               hipStream_t            stream)
{
    FastCheckResult    result;
    unsigned long long counters[4];
    hipError_t err = run_scan(m, batch_count, total_elements, expect_poison, stream, counters);
    if(err != hipSuccess)
        return {false, std::string("fast_check scan failed: ") + hipGetErrorString(err)};

    std::ostringstream msg;
    if(counters[0] > 0)
    {
        result.passed = false;
        msg << counters[0] << " elements outside the " << m.rows << " x " << m.cols << " x "
            << batch_count << " region were changed (out-of-bounds write); first at "
            << describe_offset(m, batch_count, counters[1]) << "\n";
    }
    if(counters[2] > 0)
    {
        result.passed = false;
        msg << counters[2] << " elements inside the region were never written; first at "
            << describe_offset(m, batch_count, counters[3]) << "\n";
    }
    result.message = msg.str();
    return result;
}

hipError_t fast_check_copy_region_to_host(void*                  host,
                                          const FastCheckMatrix& m,
                                          int64_t                batch_count,
                                          hipStream_t            stream)
{
    const size_t es    = element_size(m.type);
    const size_t width = size_t(m.rows) * es;
    for(int64_t b = 0; b < batch_count; b++)
    {
        char*       dst = static_cast<char*>(host) + size_t(b) * width * size_t(m.cols);
        const char* src = static_cast<const char*>(m.data) + size_t(b) * size_t(m.stride) * es;
        hipError_t  err = hipMemcpy2DAsync(dst,
                                          width,
                                          src,
                                          size_t(m.ld) * es,
                                          width,
                                          size_t(m.cols),
                                          hipMemcpyDeviceToHost,
                                          stream);
        if(err != hipSuccess)
        {
            // Some pitches exceed what the 2D copy accepts; fall back to one copy per column.
            (void)hipGetLastError();
            for(int64_t j = 0; j < m.cols; j++)
            {
                err = hipMemcpyAsync(dst + size_t(j) * width,
                                     src + size_t(j) * size_t(m.ld) * es,
                                     width,
                                     hipMemcpyDeviceToHost,
                                     stream);
                if(err != hipSuccess)
                    return err;
            }
        }
    }
    return hipStreamSynchronize(stream);
}

void FastCheckSolutionLog::record(
    size_t solution, int library_index, const std::string& kernel, int iteration, bool passed)
{
    auto it = std::find_if(
        entries.begin(), entries.end(), [&](const Entry& e) { return e.solution == solution; });
    if(it == entries.end())
    {
        entries.push_back({solution, library_index, kernel, {}});
        it = entries.end() - 1;
    }
    if(!kernel.empty())
        it->kernel = kernel;
    if(!passed)
        it->failed_iterations.push_back(iteration);
}

size_t FastCheckSolutionLog::failed_solutions() const
{
    return std::count_if(entries.begin(), entries.end(), [](const Entry& e) {
        return !e.failed_iterations.empty();
    });
}

std::string FastCheckSolutionLog::summary(int iterations) const
{
    const size_t failed = failed_solutions();
    if(failed == 0)
        return {};
    std::ostringstream s;
    s << "fast_check: " << failed << " of " << entries.size() << " solutions failed";
    if(iterations > 1)
        s << " (" << iterations << " iterations each)";
    s << ":";
    for(const Entry& e : entries)
    {
        if(e.failed_iterations.empty())
            continue;
        s << "\n  solution " << e.solution << " (library index " << e.library_index << ", kernel "
          << e.kernel << ")";
        if(iterations > 1)
        {
            s << ": iteration" << (e.failed_iterations.size() > 1 ? "s " : " ");
            for(size_t i = 0; i < e.failed_iterations.size(); i++)
                s << (i ? ", " : "") << e.failed_iterations[i];
        }
    }
    return s.str();
}

hipError_t fast_check_corrupt_element_device(
    const FastCheckMatrix& m, int64_t batch, int64_t row, int64_t col, hipStream_t stream)
{
    const size_t es = element_size(m.type);
    if(es == 0)
        return hipErrorInvalidValue;
    char* p = static_cast<char*>(const_cast<void*>(m.data))
              + (size_t(batch) * size_t(m.stride) + size_t(col) * size_t(m.ld) + size_t(row)) * es;
    uint64_t   current = 0;
    hipError_t err     = hipMemcpyAsync(&current, p, es, hipMemcpyDeviceToHost, stream);
    if(err == hipSuccess)
        err = hipStreamSynchronize(stream);
    if(err != hipSuccess)
        return err;
    const uint64_t mask     = es >= 8 ? ~uint64_t(0) : (uint64_t(1) << (8 * es)) - 1;
    const uint64_t sentinel = fast_check_sentinel_bits(m.type) & mask;
    uint64_t replacement = (current & mask) != sentinel ? sentinel : fast_check_poison_bits(m.type);
    err                  = hipMemcpyAsync(p, &replacement, es, hipMemcpyHostToDevice, stream);
    if(err == hipSuccess)
        err = hipStreamSynchronize(stream);
    return err;
}

double fast_check_load(const void* data, hipDataType type, size_t i)
{
    return load(data, type, i);
}

namespace
{
    // The rows x cols x batch region of a device matrix, copied to contiguous host memory.
    std::vector<char> host_region(const FastCheckMatrix& m, int64_t batch_count, hipStream_t stream)
    {
        std::vector<char> h(size_t(m.rows * m.cols * batch_count) * element_size(m.type));
        if(fast_check_copy_region_to_host(h.data(), m, batch_count, stream) != hipSuccess)
            h.clear();
        return h;
    }
}

FastCheckResult fast_check_activation_device(const FastCheckMatrix& d,
                                             const FastCheckMatrix& e,
                                             int64_t                batch_count,
                                             double                 scale_d,
                                             double                 scale_e,
                                             FastCheckActivation    act,
                                             double                 arg1,
                                             double                 arg2,
                                             hipStream_t            stream,
                                             double*                amax)
{
    FastCheckResult   result;
    if(amax)
        *amax = std::numeric_limits<double>::quiet_NaN();
    std::vector<char> hd = host_region(d, batch_count, stream);
    std::vector<char> he = host_region(e, batch_count, stream);
    if(hd.empty() || he.empty())
    {
        result.passed  = false;
        result.message = "fast_check could not copy D and E to the host\n";
        return result;
    }
    if(scale_e == 0)
    {
        result.passed  = false;
        result.message = "E is scaled by 0, so it does not give the pre-activation values and the "
                         "activation cannot be checked from it\n";
        if(amax)
            *amax = std::numeric_limits<double>::quiet_NaN();
        return result;
    }
    const int64_t      count   = d.rows * d.cols * batch_count;
    const double       e_limit = exact_limit(e.type);
    int64_t            bad = 0, inexact = 0;
    double             top = 0;
    std::ostringstream msg;
    for(int64_t idx = 0; idx < count; idx++)
    {
        const double stored_e = load(he.data(), e.type, size_t(idx));
        // E is verified against its rounded value, so at or past this limit the pre-activation it
        // gives may not be the one D was computed from.
        if(!(std::fabs(stored_e) < e_limit))
        {
            ++inexact;
            continue;
        }
        double pre = stored_e / scale_e;
        double a   = pre;
        if(act == FastCheckActivation::relu)
            a = std::max(pre, 0.0);
        else if(act == FastCheckActivation::clamp)
            a = std::max(arg1, std::min(pre, arg2));
        top             = std::max(top, std::fabs(a));
        double expected = round_to_type(a * scale_d, d.type);
        double got      = load(hd.data(), d.type, size_t(idx));
        if(!(expected == got) && bad++ < int64_t(kMaxReported))
        {
            int64_t b = idx / (d.rows * d.cols), rem = idx % (d.rows * d.cols);
            msg << "\n  batch " << b << ", row " << rem % d.rows << ", col " << rem / d.rows
                << ": E gives " << pre << ", expected D " << expected << ", got " << got;
        }
    }
    if(bad > 0)
    {
        result.passed  = false;
        result.message = std::to_string(bad)
                         + " elements of D do not match the activation applied to the verified E:"
                         + msg.str() + "\n";
    }
    if(inexact > 0)
    {
        result.passed = false;
        result.message += std::to_string(inexact)
                          + " elements of E are outside the range its type stores exactly, so the "
                            "activation cannot be checked from them. Use a wider aux_type, or "
                            "smaller inputs or scales.\n";
    }
    if(amax)
        *amax = inexact > 0 ? std::numeric_limits<double>::quiet_NaN() : top;
    return result;
}

double fast_check_amax_device(const FastCheckMatrix& d,
                              int64_t                batch_count,
                              double                 scale_d,
                              hipStream_t            stream)
{
    std::vector<char> hd    = host_region(d, batch_count, stream);
    const double      limit = exact_limit(d.type);
    double            top   = 0;
    // A failed copy would otherwise look like amaxD 0.
    if(hd.empty() && d.rows * d.cols * batch_count > 0)
        return std::numeric_limits<double>::quiet_NaN();
    for(int64_t idx = 0; idx < d.rows * d.cols * batch_count && !hd.empty(); idx++)
        top = std::max(top, std::fabs(load(hd.data(), d.type, size_t(idx))));
    // A rounded D, or one scaled by 0, no longer gives the pre-scale value amaxD is computed from.
    if(!(top < limit) || scale_d == 0)
        return std::numeric_limits<double>::quiet_NaN();
    return top / std::fabs(scale_d);
}

FastCheckResult fast_check_bias_gradient(const FastCheckProblem& p,
                                         char                    source,
                                         const void*             bias,
                                         hipDataType             bias_type)
{
    const View           opA = batch_view(p.A, 0, p.transA); // M x K
    const View           opB = batch_view(p.B, 0, p.transB); // K x N
    std::vector<int64_t> expected(size_t(source == 'a' ? p.M : p.N), 0);
    for(size_t x = 0; x < expected.size(); x++)
    {
        double bound = 0;
        for(int64_t k = 0; k < p.K; k++)
        {
            const double value = source == 'a' ? opA.at(int64_t(x), k) : opB.at(k, int64_t(x));
            bound += std::fabs(value);
            // The GEMM bound includes the other operand, which may be zero.
            // Bound this reduction separately before accumulating into int64_t.
            if(!is_exact_integer(value) || !(bound < exact_limit(p.compute_type)))
                return {false,
                        "the bias gradient cannot be checked exactly: its inputs must be "
                        "integers and every partial sum must fit the compute type\n"};
            expected[x] += int64_t(value);
        }
    }

    FastCheckResult    result;
    std::ostringstream msg;
    int64_t            bad = 0;
    for(size_t i = 0; i < expected.size(); i++)
    {
        double want = round_to_type(double(expected[i]), bias_type);
        double got  = load(bias, bias_type, i);
        if(!(want == got) && bad++ < int64_t(kMaxReported))
            msg << "\n  element " << i << ": expected " << want << ", got " << got;
    }
    if(bad > 0)
    {
        result.passed  = false;
        result.message = std::to_string(bad) + " elements of the bias gradient (source "
                         + std::string(1, source) + ") are wrong:" + msg.str() + "\n";
    }
    return result;
}

namespace
{
    // Host memory the operating system reports as available, or 0 when it cannot tell.
    size_t available_host_bytes()
    {
#ifdef _WIN32
        MEMORYSTATUSEX status = {};
        status.dwLength       = sizeof(status);
        return GlobalMemoryStatusEx(&status) ? size_t(status.ullAvailPhys) : 0;
#else
        std::ifstream meminfo("/proc/meminfo");
        std::string   key;
        size_t        kib = 0;
        while(meminfo >> key >> kib)
        {
            if(key == "MemAvailable:")
                return kib * 1024;
            meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return 0;
#endif
    }
}

std::string fast_check_memory_shortfall(size_t device_bytes, size_t host_bytes)
{
    auto gib = [](size_t bytes) {
        std::ostringstream s;
        s.precision(3);
        s << double(bytes) / double(size_t(1) << 30) << " GiB";
        return s.str();
    };
    auto shortfall = [&]() -> std::string {
        size_t free_bytes = 0, total_bytes = 0;
        if(hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess && device_bytes > free_bytes)
            return "this case needs " + gib(device_bytes) + " of device memory and "
                   + gib(free_bytes) + " is free";

        const size_t available = available_host_bytes();
        if(available > 0 && host_bytes > available)
            return "this case needs " + gib(host_bytes) + " of host memory and " + gib(available)
                   + " is available";
        return {};
    };
    std::string why = shortfall();
    if(!why.empty())
    {
        // A previous large case can retain most of the free memory in these pools. The
        // allocator could reuse or release it, but a preflight skip never reaches allocation.
        memory_pool<d_memory>::ReleaseCached();
        memory_pool<h_memory>::ReleaseCached();
        why = shortfall();
    }
    return why;
}
