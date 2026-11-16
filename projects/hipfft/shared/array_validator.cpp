// Copyright (C) 2022 - 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <algorithm> // std::sort, std::max, std::min
#include <cassert>
#include <cstddef> // ptrdiff_t, size_t
#include <iostream>
#include <limits> // std::numeric_limits
#include <numeric> // std:gcd
#include <sstream> // std::stringstream
#include <stdexcept> // runtime_error, logic_error
#include <utility> // std::swap
#include <vector>

#include "array_validator.h"

namespace
{
    /**
     * @brief structure encapsulating info about array dimensions
     *
     */
    struct array_dimension
    {
        array_dimension(size_t length, size_t stride, size_t dim_idx)
            : l(length)
            , s(stride)
            , idx(dim_idx)
        {
        }

        const size_t& length() const
        {
            return l;
        }
        const size_t& stride() const
        {
            return s;
        }
        const size_t& index() const
        {
            return idx;
        }

        // Array dimensions are sorted by increasing strides
        bool operator<(const array_dimension& other) const
        {
            return s < other.s;
        }

    private:
        size_t l;
        size_t s;
        size_t idx;
    };

    // adhoc exception type thrown by array_t constructor(s) when an
    // invalid or trivially self-aliasing array creation is attempted.
    struct array_construction_exception : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

#ifndef NDEBUG
    struct array_debug_exception : std::logic_error
    {
        using std::logic_error::logic_error;
    };
#endif

    struct array_t
    {
    public:
        array_t(const std::vector<size_t>& lengths, const std::vector<size_t>& strides)
        {
            if(lengths.size() != strides.size())
                throw array_construction_exception(
                    "Inconsistent sizes between lengths and strides");
            dimensions.clear();
            for(size_t dim = 0; dim < lengths.size(); dim++)
            {
                if(lengths[dim] == 0) // not an array...
                {
                    std::stringstream info;
                    info << "Zero length detected along dimension " << dim;
                    throw array_construction_exception(info.str());
                }
                if(lengths[dim] == 1)
                {
                    // unit lengths make the corresponding dimension irrelevant
                    // --> ignored from considerations
                    continue;
                }
                if(strides[dim] == 0) // trivial degenerate array
                {
                    std::stringstream info;
                    info << "Zero stride for non-unit length detected along dimension " << dim;
                    throw array_construction_exception(info.str());
                }

                dimensions.emplace_back(lengths[dim], strides[dim], dim);
            }
            std::sort(dimensions.begin(), dimensions.end());
            // dimensions contains only the non-trivial array dimensions (lengths > 1 and
            // non-zero strides), sorted by increasing stride values.

            // The search below uses signed arithmetic on values bounded by twice the array's
            // span (largest flattened index), i.e., reach.back(), which may not exceed max_span.
            for(const auto& dim : dimensions)
            {
                const size_t previous_reach = reach.empty() ? 0 : static_cast<size_t>(reach.back());
                if(dim.stride() > (max_span - previous_reach) / (dim.length() - 1))
                {
                    std::stringstream info;
                    info << "Array span too large (supported maximum: " << max_span << ")";
                    throw array_construction_exception(info.str());
                }
                reach.push_back(
                    static_cast<ptrdiff_t>(previous_reach + (dim.length() - 1) * dim.stride()));
                stride_gcd.push_back(std::gcd(stride_gcd.empty() ? ptrdiff_t(0) : stride_gcd.back(),
                                              static_cast<ptrdiff_t>(dim.stride())));
            }
        }

        /**
         * @brief checks if the array is self-aliasing, i.e., if there exist two different
         * multi-indices (x_0, x_1, ..., x_n) and (y_0, y_1, ..., y_n) such that
         * x_0*s_0 + x_1*s_1 + ... + x_n*s_n == y_0*s_0 + y_1*s_1 + ... + y_n*s_n
         * wherein s_i = dimensions[j].stride() for j s.t. dimensions[j].index() == i.
         *
         * Writing d_i = x_i - y_i, the array is self-aliasing iff
         *
         *        d_0*s_0 + d_1*s_1 + ... + d_n*s_n == 0                          (E)
         *
         * has a solution with -l_i < d_i < l_i for all i and d_i != 0 for some i, wherein
         * l_i is the length associated with s_i. This is decided as follows (dimensions
         * being sorted by increasing strides):
         *
         * 1. Pigeonhole principle: if the array has more elements than there are flattened
         *    indices in [0, span], at least two of its elements collide.
         *
         * 2. The coefficients d_i are enumerated by decreasing strides, except for the two
         *    smallest-stride dimensions (see solve). Each d_i is restricted to the values
         *    leaving a target reachable by the dimensions of smaller strides, and branches
         *    are cut when the target is not a multiple of the gcd of their strides. The
         *    first nonzero coefficient is kept positive, since d and -d describe the same
         *    collision. For "nested" arrays (each stride at least the previous dimension's
         *    length times its stride), only zero coefficients remain at every level, so
         *    they are decided without branching.
         *
         * 3. The two smallest-stride dimensions are not enumerated: all integer solutions
         *    of d_0*s_0 + d_1*s_1 == target follow from the extended Euclidean algorithm,
         *    so checking whether one of them is in range takes O(log) operations (see
         *    solve_two_dimensions).
         *
         * @return true if the array is self-aliasing, false if not.
         */
        bool is_self_aliasing() const
        {
            if(size() < 2)
            {
                // nontrivial lengths associated with a zero stride are excluded by construction
                // --> no conflict can exist
                return false;
            }

            // 1. pigeonhole principle
            if(has_more_elements_than_indices())
                return true;

            // 2. and 3.
            constexpr bool nontrivial_required = true;
            return solve(size() - 1, 0, nontrivial_required);
        }

        /**
         * @brief checks if an element of the array is located at a given flattened index, i.e.,
         * if there is a multi-index (x_0, x_1, ..., x_n) with 0 <= x_i < l_i such that
         * x_0*s_0 + x_1*s_1 + ... + x_n*s_n == index. Coefficients x_i are searched for as in
         * is_self_aliasing (steps 2 and 3), with non-negative coefficients and index as target.
         *
         * @param[in] index flattened index
         * @return true if an element is located at index, false otherwise
         */
        bool reaches(size_t index) const
        {
            if(size() == 0)
                return index == 0; // single element, at flattened index 0
            if(index > static_cast<size_t>(reach.back()))
                return false;
            constexpr bool signed_coefficients = false;
            constexpr bool nontrivial_required = false;
            return solve<signed_coefficients>(
                size() - 1, static_cast<ptrdiff_t>(index), nontrivial_required);
        }

        inline size_t size() const
        {
            return dimensions.size();
        }

    private:
        // largest supported span: the search computes signed values up to twice the span
        static constexpr size_t max_span = std::numeric_limits<ptrdiff_t>::max() / 2;

        std::vector<array_dimension> dimensions;
        // reach[k]: largest |d_0*s_0 + ... + d_k*s_k| with -l_i < d_i < l_i, i.e., span of the
        // sub-array made of the first k + 1 dimensions
        std::vector<ptrdiff_t> reach;
        // stride_gcd[k]: greatest common divisor of s_0, ..., s_k
        std::vector<ptrdiff_t> stride_gcd;

        // true iff the array has more elements than there are flattened indices in [0, span]
        bool has_more_elements_than_indices() const
        {
            const size_t n_indices  = static_cast<size_t>(reach.back()) + 1;
            size_t       n_elements = 1;
            for(const auto& dim : dimensions)
            {
                // n_elements * dim.length() > n_indices, without overflow
                if(n_elements > n_indices / dim.length())
                    return true;
                n_elements *= dim.length();
            }
            return false;
        }

        /**
         * @brief searches for a solution of d_0*s_0 + ... + d_k*s_k == target with
         * lo_i <= d_i <= hi_i, wherein hi_i = l_i - 1, and lo_i = -hi_i for signed coefficients
         * (lo_i = 0 otherwise), by enumerating d_k, d_{k-1}, ..., d_2 (decreasing strides).
         *
         * @tparam signed_coefficients true for signed coefficients, false for non-negative ones
         * @param[in] k index of the dimension of largest stride involved
         * @param[in] target right-hand side of the equation
         * @param[in] nontrivial_required true iff a nontrivial solution of the homogeneous
         * equation is required (target == 0, signed coefficients), i.e., when searching for a
         * nonzero solution of (E) with all coefficients of the larger-stride dimensions being 0.
         * d and -d being both solutions then, the first nonzero coefficient (by decreasing
         * strides) is kept positive
         * @return true if a solution was found, false otherwise
         */
        template <bool signed_coefficients = true>
        bool solve(size_t k, ptrdiff_t target, bool nontrivial_required) const
        {
            assert(k < size());
            assert(!nontrivial_required || (signed_coefficients && target == 0));
            // range of d_0*s_0 + ... + d_k*s_k
            const ptrdiff_t sum_min = signed_coefficients ? -reach[k] : 0;
            const ptrdiff_t sum_max = reach[k];
            if(target < sum_min || target > sum_max || target % stride_gcd[k] != 0)
                return false;
            if(k == 0)
            {
                // target being a multiple of s_0 within [sum_min, sum_max], d_0 = target/s_0 is
                // the only solution; it is nonzero iff target != 0
                return target != 0 || !nontrivial_required;
            }
            if(k == 1)
                return solve_two_dimensions<signed_coefficients>(target, nontrivial_required);

            // d_k is restricted to values leaving a target that the dimensions of smaller
            // strides can reach: rest_min <= target - d_k*s_k <= rest_max
            const auto      s_k      = static_cast<ptrdiff_t>(dimensions[k].stride());
            const ptrdiff_t hi_k     = static_cast<ptrdiff_t>(dimensions[k].length()) - 1;
            const ptrdiff_t lo_k     = signed_coefficients ? -hi_k : 0;
            const ptrdiff_t rest_min = signed_coefficients ? -reach[k - 1] : 0;
            const ptrdiff_t rest_max = reach[k - 1];
            ptrdiff_t       d_min    = std::max(lo_k, divUp(target - rest_max, s_k));
            const ptrdiff_t d_max    = std::min(hi_k, divDown(target - rest_min, s_k));
            if(nontrivial_required)
                d_min = std::max<ptrdiff_t>(d_min, 0);
            for(ptrdiff_t d_k = d_min; d_k <= d_max; d_k++)
            {
                if(solve<signed_coefficients>(
                       k - 1, target - d_k * s_k, nontrivial_required && d_k == 0))
                    return true;
            }
            return false;
        }

        /**
         * @brief decides if d_0*s_0 + d_1*s_1 == target has a solution with lo_0 <= d_0 <= hi_0
         * and lo_1 <= d_1 <= hi_1 (see solve), excluding d_0 == d_1 == 0 if nontrivial_required,
         * without enumeration.
         *
         * With g = gcd(s_0, s_1) dividing target (checked by the caller) and Bezout's
         * coefficients p, q such that p*s_0 + q*s_1 == g, the integer solutions are
         *
         *        d_0 = c + j*(s_1/g),   d_1 = (target - d_0*s_0)/s_1 = d_1(0) - j*(s_0/g)
         *
         * for any integer j, wherein c = p*(target/g) mod (s_1/g). The values of j keeping
         * either coefficient in range form two intervals, which only need to intersect.
         *
         * @tparam signed_coefficients see solve
         * @param[in] target right-hand side of the equation
         * @param[in] nontrivial_required see solve
         * @return true if a solution was found, false otherwise
         */
        template <bool signed_coefficients>
        bool solve_two_dimensions(ptrdiff_t target, bool nontrivial_required) const
        {
            const size_t    s0  = dimensions[0].stride();
            const size_t    s1  = dimensions[1].stride();
            const ptrdiff_t hi0 = static_cast<ptrdiff_t>(dimensions[0].length()) - 1;
            const ptrdiff_t hi1 = static_cast<ptrdiff_t>(dimensions[1].length()) - 1;
            const ptrdiff_t lo0 = signed_coefficients ? -hi0 : 0;
            const ptrdiff_t lo1 = signed_coefficients ? -hi1 : 0;
            ptrdiff_t       p, q; // Bezout's coefficients
            const auto      g = static_cast<ptrdiff_t>(ext_euclide_gcd(s0, s1, p, q));
            // period of d_0 (resp. d_1) along the family of solutions
            const ptrdiff_t period0 = static_cast<ptrdiff_t>(s1) / g;
            const ptrdiff_t period1 = static_cast<ptrdiff_t>(s0) / g;
            // residue of all valid d_0 modulo period0; both factors are reduced first so that
            // the product cannot overflow
            const auto c = static_cast<ptrdiff_t>(mulmod(non_negative_mod(p, period0),
                                                         non_negative_mod(target / g, period0),
                                                         static_cast<size_t>(period0)));
            // smallest d_0 >= lo0 in that residue class, and the matching d_1
            const ptrdiff_t d0_first = c + divUp(lo0 - c, period0) * period0;
            if(d0_first > hi0)
                return false;
            const ptrdiff_t d1_first
                = (target - d0_first * static_cast<ptrdiff_t>(s0)) / static_cast<ptrdiff_t>(s1);
            // d_0 = d0_first + j*period0 and d_1 = d1_first - j*period1 are both in range iff
            // j_min <= j <= j_max
            const ptrdiff_t j_min = std::max<ptrdiff_t>(0, divUp(d1_first - hi1, period1));
            const ptrdiff_t j_max
                = std::min(divDown(hi0 - d0_first, period0), divDown(d1_first - lo1, period1));
            if(j_min > j_max)
                return false;
            // target being 0 when nontrivial_required, d_0 == 0 forces d_1 == 0: the solution at
            // j_min is either nontrivial or, d_0 increasing with j, followed by nontrivial ones if
            // j_min < j_max
            return !nontrivial_required || d0_first + j_min * period0 != 0 || j_min < j_max;
        }

        /**
         * @brief computes the greatest common divisor between two (strictly positive) integers
         * via the extended Euclid algorithm, with Bezout's coefficients p and q such that
         * gcd(a, b) = p*a + q*b as a side result.
         * @param[in] a, b: strictly positive integers
         * @param[out] p, q: Bezout's coefficients
         * returns greatest common divisor between a and b as well as
         * the Bezout coefficients p and q
         */
        static size_t ext_euclide_gcd(size_t a, size_t b, ptrdiff_t& p, ptrdiff_t& q)
        {
#ifndef NDEBUG
            if(a == 0 || b == 0)
                throw array_debug_exception(
                    "array_t::ext_euclide_gcd: invalid (zero) value for a or b");
            // save inputs for consistency check below
            const ptrdiff_t user_a = static_cast<ptrdiff_t>(a);
            const ptrdiff_t user_b = static_cast<ptrdiff_t>(b);
#endif
            p = 1, q = 0;
            ptrdiff_t  p_kp1 = 0, q_kp1 = 1;
            const bool swap_needed = a < b;
            if(swap_needed)
                std::swap(a, b);
            while(b != 0)
            {
                size_t    next_b = a % b;
                ptrdiff_t ratio  = (a / b);
                ptrdiff_t p_kp2  = p - ratio * p_kp1;
                ptrdiff_t q_kp2  = q - ratio * q_kp1;
                a                = b;
                b                = next_b;
                q = q_kp1, p = p_kp1;
                q_kp1 = q_kp2, p_kp1 = p_kp2;
            }
            if(swap_needed)
            {
                std::swap(p, q);
            }
            // a := GCD(user_a, user_b) and
            // p * user_a + q * user_b == GCD(user_a, user_b) now
#ifndef NDEBUG
            // check returned value
            if(a != static_cast<size_t>(std::gcd(user_a, user_b)))
            {
                std::stringstream info;
                info << "Different value returned by array_t::ext_euclide_gcd (" << a
                     << ") than by std::gcd (" << std::gcd(user_a, user_b) << ").";
                throw array_debug_exception(info.str());
            }
            // check on Bezout's coefficients
            if(p * user_a + q * user_b != static_cast<ptrdiff_t>(a))
            {
                std::stringstream info;
                info << "Incorrect Bezout's coefficient detected in "
                        "array_t::ext_euclide_gcd: \n"
                     << "\tcomputed gcd(" << user_a << ", " << user_b << "): " << a << "\n"
                     << "\t(p, q) = (" << p << ", " << q << ")\n"
                     << "\tp*a + q*b = " << p * user_a + q * user_b << "\n";
                throw array_debug_exception(info.str());
            }
#endif
            return a;
        }
        // returns the smallest value x s.t. x*b >= a regardless of the sign of a (needs b > 0)
        // NOTE: different than DivRoundingUp for negative values of a, equivalent if a > 0
        static inline ptrdiff_t divUp(ptrdiff_t a, ptrdiff_t b)
        {
            return a / b + (a % b > 0 ? 1 : 0);
        }
        // returns the largest value x s.t. x*b <= a regardless of the sign of a (needs b > 0)
        // NOTE: different than a / b for negative values of a, equivalent if a > 0
        static inline ptrdiff_t divDown(ptrdiff_t a, ptrdiff_t b)
        {
            return a / b - (a % b < 0 ? 1 : 0);
        }
        // returns a mod m in [0, m) regardless of the sign of a (needs m > 0)
        static inline size_t non_negative_mod(ptrdiff_t a, ptrdiff_t m)
        {
            const ptrdiff_t r = a % m;
            return static_cast<size_t>(r < 0 ? r + m : r);
        }
        // returns (a * b) mod m for a, b < m <= max_span, without overflow: a * b is computed
        // directly unless it overflows (only possible for strides of several billion), in which
        // case double-and-add is used, with intermediate values below 2 * m
        static size_t mulmod(size_t a, size_t b, size_t m)
        {
            if(b == 0 || a <= std::numeric_limits<size_t>::max() / b)
                return (a * b) % m;
            size_t ret = 0;
            for(; b > 0; b >>= 1, a = (a + a) % m)
                if(b & 1)
                    ret = (ret + a) % m;
            return ret;
        }
    };

}

bool array_valid(const std::vector<size_t>& length,
                 const std::vector<size_t>& stride,
                 const int                  verbose)
{
    try
    {
        return !array_t(length, stride).is_self_aliasing();
    }
    catch(const array_construction_exception& e)
    {
        if(verbose)
            std::cout << e.what() << std::endl;
        return false;
    }
#ifndef NDEBUG
    catch(const array_debug_exception& e)
    {
        std::cerr << "Logic error detected in array_valid, details below.\n"
                  << e.what() << std::endl;
        throw e;
    }
#endif
}
