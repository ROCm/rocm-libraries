// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm> // std::permutation
#include <iostream>
#include <memory> // std::unique_ptr
#include <numeric> // std::lcm, std:gcd
#include <sstream> // std::stringstream
#include <stdexcept> // runtime_error, logic_error
#include <utility> // std::pair
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
        }

        using collision_t = std::pair<std::vector<size_t>, std::vector<size_t>>;

        /**
         * @brief checks if the array is self-aliasing, i.e., if there exist two different
         * multi-indices (x_0, x_1, ..., x_n) and (y_0, y_1, ..., y_n) such that
         * x_0*s_0 + x_1*s_1 + ... + x_n*s_n == y_0*s_0 + y_1*s_1 + ... + y_n*s_n
         * wherein s_i = dimensions[j].stride() for j s.t. dimensions[j].index() == i.
         * 
         * @tparam is_seed_call boolean template parameter distinguishing between the parent object's
         * function call (true specialization) and function calls by child objects generated herein
         * (false specialization).
         * IMPORTANT NOTE: the false-specialization is meant to be used internally only, as it may
         * produce incorrect results otherwise.
         * @param[out] eqn_counters pointer to a vector of size_t values of capacity >= size() - 1
         * (ignored if nullptr). If not ignored, (*eqn_counters)[i] contains the number of nontrivial
         * linear diophantine equations of (i + 1) unknowns that have been considered by this function
         * upon its return.
         * @param[out] collision pointer to a pair of vectors of size_t values of capacity equal to
         * those of the lengths and strides used at construction of this object (ignored if nullptr).
         * If not ignored, this pair contains an example of two multi-indices that result in a
         * collision upon return when a collision exists (its content is undefined upon return
         * otherwise).
         * @return true if the array is self-aliasing, false if not.
         */
        template <bool is_seed_call = true>
        bool is_self_aliasing(std::vector<size_t>* eqn_counters, collision_t* collision) const
        {
            if(dimensions.size() < 2)
            {
                // nontrivial lengths associated with a zero stride are excluded by construction
                // --> no conflict can exist
                return false;
            }

            bool happy_path = true;
            for(auto it = dimensions.begin(); (it + 1) != dimensions.end(); it++)
            {
                happy_path &= (it + 1)->stride() >= it->length() * it->stride();
            }
            if(happy_path)
                return false;

            // There exist two different multi-indices (x_0, x_1, ..., x_n) and (y_0, y_1, ..., y_n)
            // such that x_0*s_0 + x_1*s_1 + ... + x_n*s_n == y_0*s_0 + y_1*s_1 + ... + y_n*s_n
            // <=>
            // sum_{i : x_i >= y_i } [ (x_i - y_i) * s_i] == sum_{i : x_i < y_i} [ (y_i - x_i) * s_i]
            // <=>
            // there exists u_0, u_1, ..., u_n, s.t.
            // sum_{i : Si_lhs } u_i * s_i == sum_{i : Si_rhs} u_i * s_i
            // wherein
            // - Si_lhs : set of dimension indices i such that x_i >= y_i
            // - Si_rhs : set of dimension indices i such that x_i <  y_i
            // - u_i = |x_i - y_i|, therefore 0 <= u_i < l_i
            // Note: Si_lhs and Si_rhs are disjoint sets by definition.

            // Find all ways to split the current array's set of dimension
            // indices in two disjoint sub-sets
            std::vector<size_t>  mask(size());
            std::vector<size_t>* lhs_solution = collision ? &collision->first : nullptr;
            std::vector<size_t>* rhs_solution = collision ? &collision->second : nullptr;
            for(size_t lhs_array_sz = size() / 2; lhs_array_sz > 0; lhs_array_sz--)
            {
                std::fill(mask.begin(), mask.begin() + lhs_array_sz, 0);
                std::fill(mask.begin() + lhs_array_sz, mask.end(), 1);
                do
                {
                    if(2 * lhs_array_sz == size() && mask[0] == 1)
                    {
                        // avoid redundant splits that would be swapped
                        // copies of some other
                        continue;
                    }
                    array_t lhs_subarray, rhs_subarray;
                    for(size_t idx = 0; idx < size(); ++idx)
                    {
                        if(mask[idx])
                            rhs_subarray.push_back(dimensions[idx]);
                        else
                            lhs_subarray.push_back(dimensions[idx]);
                    }
                    // check if any sub-array is self-aliasing itself
                    // (lower-dimensional, much cheaper verification check)
                    // NOTE: seed-call only to avoid redundancy. Sub-arrays of sub-arrays of
                    // a parent object are (other) sub-arrays of that object (therefore already
                    // considered in another iteration herein, in the seed call).
                    if constexpr(is_seed_call)
                    {
                        if(lhs_subarray.is_self_aliasing<!is_seed_call>(eqn_counters, collision)
                           || rhs_subarray.is_self_aliasing<!is_seed_call>(eqn_counters, collision))
                        {
                            return true;
                        }
                    }
                    // if a conflict exists, it must be at
                    // - a common multiple of either side's stride_gcd
                    // - within non-degenerate range for either side (and not 0)
                    // Indeed, given the verification for self-aliazing sub-arrays done
                    // above, degenerate solutions (i.e., solution with x[i] = 0 for any i)
                    // may be ruled out below as they'd be found in another lower-dimensional
                    // sub-array verification. This significantly increases the minimum value
                    // of the range of relevance for possible matches, below.
                    const size_t lcm
                        = std::lcm(lhs_subarray.strides_gcd(), rhs_subarray.strides_gcd());
                    const size_t possible_match_min
                        = lcm
                          * divUp(std::max(lhs_subarray.first_nondegenerate_entry(),
                                           rhs_subarray.first_nondegenerate_entry()),
                                  lcm);
                    const size_t possible_match_max
                        = std::min(lhs_subarray.last_entry(), rhs_subarray.last_entry());
                    if(possible_match_max < possible_match_min)
                        continue;
                    if(lhs_solution && rhs_solution)
                    {
                        // reset solutions
                        std::fill(lhs_solution->begin(), lhs_solution->end(), 0);
                        std::fill(rhs_solution->begin(), rhs_solution->end(), 0);
                    }
                    for(size_t possible_match = possible_match_min;
                        possible_match <= possible_match_max;
                        possible_match += lcm)
                    {
                        if(lhs_subarray.reaches(possible_match, eqn_counters, lhs_solution)
                           && rhs_subarray.reaches(possible_match, eqn_counters, rhs_solution))
                        {
                            return true;
                        }
                    }
                } while(std::next_permutation(mask.begin(), mask.end()));
            }
            return false;
        }

        inline size_t size() const
        {
            return dimensions.size();
        }

    private:
        std::vector<array_dimension> dimensions;

        // computes the greatest common divisor of all strides
        size_t strides_gcd() const
        {
#ifndef NDEBUG
            if(size() < 1)
                throw array_debug_exception("array_t::strides_gcd() invoked by an empty array");
#endif
            return std::accumulate(
                dimensions.begin() + 1,
                dimensions.end(),
                dimensions[0].stride(),
                [](size_t ret, const array_dimension& dim) { return std::gcd(ret, dim.stride()); });
        }

        // calculates index of array element (1,1,...,1)
        size_t first_nondegenerate_entry() const
        {
            return std::accumulate(
                dimensions.begin(),
                dimensions.end(),
                0ULL,
                [](size_t acc, const array_dimension& dim) { return acc + dim.stride(); });
        }

        // calculates index of array element (l0-1,l1-1,...,ln-1)
        size_t last_entry() const
        {
            return std::accumulate(dimensions.begin(),
                                   dimensions.end(),
                                   0ULL,
                                   [](size_t acc, const array_dimension& dim) {
                                       return acc + dim.stride() * (dim.length() - 1);
                                   });
        }

        /**
         * @brief verifies if an array element is located at `address`, i.e., if
         * 
         *        sum_i (x[i] * dimensions[i].stride()) == address
         * 
         * has a solution in valid range, i.e., a solution such that
         * 0 <= x[i] < dimensions[i].length() for all 0 <= i < size().
         * 
         * @param[in] address flattened index value where the presence of an array
         * element is enquired
         * @param[inout] eqn_counters pointer to a vector of size_t counters of
         * diophantine equations being considered by this object (ignored if nullptr)
         * @param[out] solution pointer to a vector of size_t values capturing the
         * multi-index identification of the array element present at the queried
         * location upon return (if found). If not ignored, solution[dimensions[i].index()]
         * is set to the found x_i for all 0 <= i < size() if an array element is found.
         * @return true if an array element is located at `address`, false if not.
         */
        bool reaches(size_t               address,
                     std::vector<size_t>* eqn_counters,
                     std::vector<size_t>* solution) const
        {
            if(size() == 0)
            {
                // should never happen
#ifndef NDEBUG
                throw array_debug_exception("array_t::reaches() invoked by an empty array");
#endif
                return false;
            }
            if(address == 0)
            {
                // trivial homogeneous solution
                if(solution)
                    std::fill(solution->begin(), solution->end(), 0);
                return true;
            }
            if(address > last_entry())
                return false; // beyond range
            if(address % strides_gcd() != 0)
                return false;
            if(eqn_counters && size() > 1)
            {
                // nontrivial case, add it to counter if counting
                (*eqn_counters)[size() - 1]++;
            }
            switch(size())
            {
            case 1:
            {
                if(solution)
                {
                    (*solution)[dimensions[0].index()] = address / dimensions[0].stride();
                }
                return true;
            }
            case 2:
            {
                const size_t& s0 = dimensions[0].stride();
                const size_t& s1 = dimensions[1].stride();
                // we need to find solutions to p*s0 + q*s1 = address;
                ptrdiff_t  p, q; // Bezout's coefficients
                const auto gcd = ext_euclide_gcd(s0, s1, p, q);
                // --> p*s0 + q*s1 = gcd
                const ptrdiff_t scale = static_cast<ptrdiff_t>(address / gcd);
                p *= scale;
                q *= scale;
                // --> p*s0 + q*s1 = address, now (particular solution)
#ifndef NDEBUG
                // check the particular solution
                if(p * static_cast<ptrdiff_t>(s0) + q * static_cast<ptrdiff_t>(s1)
                   != static_cast<ptrdiff_t>(address))
                {
                    std::stringstream info;
                    info << "Incorrect particular solution detected in array_t::reaches\n"
                         << "\t(p, q) = (" << p << ", " << q << ")\n"
                         << "\tp*s0 + q*s1 = "
                         << p * static_cast<ptrdiff_t>(s0) + q * static_cast<ptrdiff_t>(s1)
                         << " != " << address << "\n";
                    throw array_debug_exception(info.str());
                }
#endif
                const ptrdiff_t l0     = static_cast<ptrdiff_t>(dimensions[0].length());
                const ptrdiff_t l1     = static_cast<ptrdiff_t>(dimensions[1].length());
                const size_t    s0_gcd = s0 / gcd;
                const size_t    s1_gcd = s1 / gcd;
                // check if a solution exists in range, i.e., if there is
                // an integer k such that
                // 0 <= p + k * s1_gcd <= dimensions[0].length() - 1;
                // and
                // 0 <= q - k * s0_gcd <= dimensions[1].length() - 1;
                // then a solution exists and address can be reached.
                const ptrdiff_t min_k = std::max(divUp(-p, s1_gcd), divUp(q - l1 + 1, s0_gcd));
                const ptrdiff_t max_k = std::min(divDown(l0 - 1 - p, s1_gcd), divDown(q, s0_gcd));
                auto            ret   = min_k <= max_k;
                if(ret && solution)
                {
                    (*solution)[dimensions[0].index()] = p + min_k * s1_gcd;
                    (*solution)[dimensions[1].index()] = q - min_k * s0_gcd;
                }
                return ret;
            }
            default:
            {
                // TODO: find something smarter for this (responsible
                // for a complexity explosion in case of nontrivially-valid
                // high-dimensional cases, e.g., size() > 6)

                // linear diophantine equation with n > 2 unknowns
                // reduce to (n-1) unknowns by scoping the range of valid
                // values for one of the unknowns: remove dimensions.back()
                // (being the dimension of largest stride, it results in the
                //  smallest range)
                array_t      sub_array(dimensions.begin(), dimensions.end() - 1);
                const auto&  last_dim = dimensions.back();
                const size_t max_k = std::min(last_dim.length() - 1, address / last_dim.stride());
                for(size_t k = 0; k <= max_k; k++)
                {
                    if(sub_array.reaches(address - k * last_dim.stride(), eqn_counters, solution))
                    {
                        if(solution)
                            (*solution)[last_dim.index()] = k;
                        return true;
                    }
                }
                return false;
            }
            }
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

        // private constructors (for subarrays)
        array_t(decltype(dimensions)::const_iterator first,
                decltype(dimensions)::const_iterator last)
        {
            dimensions.assign(first, last);
        }
        array_t()
        {
            dimensions.clear();
        }
        void push_back(const decltype(dimensions)::value_type& dim_to_add)
        {
            dimensions.push_back(dim_to_add);
        }
    };

    void report_equation_counts(const std::vector<size_t>& eqn_counts)
    {
        if(eqn_counts.empty() || std::all_of(eqn_counts.begin(), eqn_counts.end(), [](size_t val) {
               return val == 0;
           }))
        {
            std::cout << "No linear diophantine equation was considered." << std::endl;
            return;
        }
        for(size_t idx = 0; idx < eqn_counts.size(); idx++)
        {
            if(eqn_counts[idx] == 0)
                continue;
            std::cout << eqn_counts[idx] << " linear diophantine equation"
                      << (eqn_counts[idx] > 1 ? "s" : "") << " with " << idx + 1 << " unknown"
                      << (idx > 0 ? "s " : " ") << (eqn_counts[idx] > 1 ? "were " : "was ")
                      << "considered." << std::endl;
        }
    }
    void report_collision_example(const array_t::collision_t& collision_example)
    {
        std::cout << "Example of colliding multi-indices:\n";
        for(const auto& tmp : {collision_example.first, collision_example.second})
        {
            std::cout << "\t(" << tmp[0];
            for(size_t i = 1; i < tmp.size(); i++)
            {
                std::cout << ", " << tmp[i];
            }
            std::cout << ")\n";
        }
        std::cout << std::endl;
    }
}

bool array_valid(const std::vector<size_t>& length,
                 const std::vector<size_t>& stride,
                 const int                  verbose)
{
    try
    {
        array_t                               array(length, stride);
        std::unique_ptr<std::vector<size_t>>  eqn_counters;
        std::unique_ptr<array_t::collision_t> collision_example;
        if(verbose > 3 && array.size() > 1)
            eqn_counters = std::make_unique<std::vector<size_t>>(array.size() - 1, 0);
        if(verbose > 1 && array.size() > 1)
        {
            // note: length.size() may be larger than array.size()
            collision_example = std::make_unique<array_t::collision_t>();
            collision_example->first.resize(length.size());
            collision_example->second.resize(length.size());
        }
        const auto ret = !array.is_self_aliasing(eqn_counters.get(), collision_example.get());
        if(eqn_counters)
            report_equation_counts(*eqn_counters);

        if(!ret && collision_example)
            report_collision_example(*collision_example);
        return ret;
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
    catch(...)
    {
        throw;
    }
}
