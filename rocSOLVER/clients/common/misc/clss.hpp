/* **************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <complex>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <vector>

//
// @brief `class closest_largest_subsequences`: Functor to compute the closest
// largest subsequences of a given pair of sequences.
//
// Given a tolerance `tol` and a pair of sequences:
//
// (a_i), (b_j) with 0 <= i <= n, 0 <= j <= m;
//
// `closest_largests_subsequences` (`clss`) extracts the subsequences:
//
// (a_l) .=. (a_l1, a_l2, ..., a_lP) with i <= l1 < l2 < ... < lP <= n, and
// (b_k) .=. (b_k1, b_k2, ..., b_kP) with j <= k1 < k2 < ... < kP <= m;
//
// (where 0 <= P <= n, m) that satisfy the following properties:
//
// 1. |a_l1 - b_k1| <= tol, |a_l2 - b_k2| <= tol, ..., |a_lP - b_kP| <= tol;
//
// 2. maximizes P (the size of the subsequences); and, for this maximal P,
//
// 3. minimizes ||a_l - b_k||_1 = \sum_{1 <= q <= P} |a_l_q - b_k_q|;
//
// in O(max{n, m}^2) space and time.  For a commented example, see Usage
// section down below.
//
// \tparam T Type of elements in sequences (a_i), (b_j); expected to be an
// arithmetic type; otherwise, T must be endowed with an overload to
// `operator<` that defines a strict partial ordear.
//
// \tparam I Signed integer type to index the sequences.
//
//
// ## Usage:
//
// Functor `clss` primary use is to improve the tests of the expert
// eigensolvers' drivers, and allow extracting a sub-sequence of the computed
// eigenvalues that matches a given list of eigenvalues.
//
// For example, consider the use of the bisection driver (STEBZ) to compute the
// eigenvalues of a matrix A with two irreducible blocks.  The spectrum of A is
// given as:
//
// - eig(A) = {-2., -1., 1., 2., -3., -2., -1., 1., 2.}.
//
// Say that the eigenvalues computed by STEBZ (grouped with the "by block"
// ordering) are (to working precision `eps` = 0.015):
//
// - STEBZ::eig(A) = {-2., -0.99, 1.01, 1.99, -3.0, -2.01, -0.99, 1.01, 2.01},
//
// and those are meant to be compared with matrix eig(A) in the range (-1, 2].
// One would find that
//
// - eig(A) \intersect (-1, 2] = {1., 2., 1., 2.}; whereas
//
// - STEBZ::eig(A) \intersect (-1, 2] = {-0.99, 1.01, 1.99, -0.99, 1.01}.
//
// Even though the computation is correct to working precision, the sets
// `eig(A)` and `STEBZ::eig(A)` have different sizes (which breaks tests that
// target their equality) and unmatched eigenvalues (which breaks tests that
// compare the eigenvalues directly).
//
// One can avoid such problems by, instead, comparing the sub-sequences
// produced by using functor `clss` with inputs:
//
// i) eig(A) \intersect (-1, 2];
//
// ii) STEBZ eigenvalues in the interval (-1 - tol, 2 + tol]; and
//
// iii) tolerance `tol` (which will be arbitrarily set to 2*`eps` = 0.03 here;
// in general, `tol` is a function of `eps` and matrix A).
//
// For this example, such a call would look like:
//
// - `clss({1., 2., 1., 2.}, {-0.99, 1.01, 1.99, -0.99, 1.01, 2.01}, tol)`;
//
// which yields the subsequences (obtained with `clss::subseqs`):
//
// - {1.,   2.,   1.,   2.}, (i.e., the reference eigenvalues) and
//
// - {1.01, 1.99, 1.01, 2.01};
//
// where the latter is the maximal subsequence of STEBZ::eig(A) that satisfies
// properties (1), (2) and (3) of the definition of `clss` above.
//
// For this example, the computed `clss::distance` (i.e., the subsequences' l^1
// distance) is 0.04, and the computed `clss::inf_norm_distance` (i.e., the
// sub-sequences' l^\inf distance) is `0.01`.
//
// Moreover, method `clss::subseqs_ids` returns the indices of the elements of
// the subsequences in their original sequences.  For this example,
// `clss::subseqs_ids` would return:
//
// - {0, 1, 2, 3}, (i.e., indices in the reference eigenvalues list) and
//
// - {1, 2, 4, 5};
//
// where the latter contains the indices of the elements of the second
// subsequence ({1.01, 1.99, 1.01, 2.01}) with respect to the original sequence
// they belong to ({-0.99, 1.01, 1.99, -0.99, 1.01, 2.01}, meant to have been
// computed by STEBZ).
//
template <typename T,
          typename I = std::int64_t,
          typename = typename std::enable_if<std::is_signed<std::decay_t<I>>::value>::type>
class closest_largest_subsequences
{
public:
    using S = decltype(std::real(T{}));

    //
    // Computes the largest closest subsequences of input sequences `a` and `b`.
    //
    // \param a:      pointer to first sequence, array of const T.
    //
    // \param size_a: number of elements in first sequence.
    //
    // \param b:      pointer to second sequence, array of const T.
    //
    // \param size_b: number of elements in second sequence.
    //
    // \return size of subsequences (equals the maximal number of matching
    // elements of the original sequences)
    //

    [[maybe_unused]] auto operator()(T const* a, I size_a, T const* b, I size_b, S tol)
        -> /**! Size of subsequences */ I
    {
        std::lock_guard<std::mutex> lock(m_);

        clear();
        if((size_a > 0) && (size_b > 0) && (tol >= 0))
        {
            //
            // Initialize members
            //
            this->tol_ = tol;
            this->size_a_ = size_a;
            this->size_b_ = size_b;
            this->memo_distances_.resize(size_a * size_b, std::numeric_limits<S>::infinity());
            this->memo_sizes_.resize(size_a * size_b, S(-1));
            this->memo_next_.resize(size_a * size_b, I(-1));
            // Copy original sequences for debugging purposes
            this->seq_a_.resize(size_a, T(0));
            memcpy(seq_a_.data(), a, sizeof(T) * size_a);
            this->seq_b_.resize(size_b, T(0));
            memcpy(seq_b_.data(), b, sizeof(T) * size_b);

            //
            // Call recursive, memoized, implementation to compute subsequences
            //
            auto [distance, sseqs_size, _] = clss_implr(a, size_a - 1, b, size_b - 1);
            this->distance_ = distance;
            this->sseqs_size_ = sseqs_size;

            //
            // Extract `sseq_a_` and `sseq_b_` from `a` and `b` and set:
            // inf_norm_ = ||sseq_a_ - sseq_b_||_inf
            //
            this->inf_norm_ = extract_subsequences(a, size_a, b, size_b);
        }

        return sseqs_size_;
    }

    //
    // Computes the largest closest subsequences of input sequences `a` and `b`.
    //
    // \param a:      pointer to first sequence, array of T.
    //
    // \param size_a: number of elements in first sequence.
    //
    // \param b:      pointer to second sequence, array of T.
    //
    // \param size_b: number of elements in second sequence.
    //
    // \return size of subsequences (equals the maximal number of matching
    // elements of the original sequences)
    //
    [[maybe_unused]] auto operator()(T* a, I size_a, T* b, I size_b, S tol)
        -> /**! Size of subsequences */ I
    {
        return this->operator()(const_cast<T const*>(a), size_a, const_cast<T const*>(b), size_b,
                                tol);
    }

    //
    // Computes the largest closest subsequences of input sequences `a` and `b`.
    //
    // \param a:      pointer to first sequence, array of const T.
    //
    // \param size_a: number of elements in first sequence; type can differ from
    // template parameter I.
    //
    // \param b:      pointer to second sequence, array of const T.
    //
    // \param size_b: number of elements in second sequence; type can differ from
    // template parameter I.
    //
    // \return size of subsequences (equals the maximal number of matching
    // elements of the original sequences)
    //
    template <typename J, typename = typename std::enable_if<std::is_integral<J>::value>::type>
    [[maybe_unused]] auto operator()(T const* a, J size_a, T const* b, J size_b, S tol)
        -> /**! Size of subsequences */ I
    {
        return this->operator()(a, static_cast<I>(size_a), b, static_cast<I>(size_b), tol);
    }

    //
    // Computes the largest closest subsequences of input sequences `a` and `b`.
    //
    // \param a:      pointer to first sequence, array of T.
    //
    // \param size_a: number of elements in first sequence; type can differ from
    // template parameter I.
    //
    // \param b:      pointer to second sequence, array of T.
    //
    // \param size_b: number of elements in second sequence; type can differ from
    // template parameter I.
    //
    // \return size of subsequences (equals the maximal number of matching
    // elements of the original sequences)
    //
    template <typename J, typename = typename std::enable_if<std::is_integral<J>::value>::type>
    [[maybe_unused]] auto operator()(T* a, J size_a, T* b, J size_b, S tol)
        -> /**! Size of subsequences */ I
    {
        return this->operator()(const_cast<T const*>(a), static_cast<I>(size_a),
                                const_cast<T const*>(b), static_cast<I>(size_b), tol);
    }

    //
    // Computes the largest closest subsequences of input sequences `a` and `b`.
    //
    // \param a:      first sequence, const vector of T.
    //
    // \param b:      second sequence, const vector of T.
    //
    // \return size of subsequences (equals the maximal number of matching
    // elements of the original sequences)
    //
    [[maybe_unused]] auto operator()(const std::vector<T>& a, const std::vector<T>& b, S tol)
        -> /**! Size of subsequences */ I
    {
        return this->operator()(a.data(), a.size(), b.data(), b.size(), tol);
    }

    //
    // Returns the l^1 distance between subsequences, or Inf if at least one of
    // them is empty.
    //
    // \return l^1 distance between subsequences.
    //
    auto distance() -> S
    {
        std::lock_guard<std::mutex> lock(m_);
        return distance_;
    }

    //
    // Returns the l^\inf distance between subsequences, or Inf if at least one
    // of them is empty.
    //
    // \return l^\inf distance between subsequences.
    //
    auto inf_norm_distance() -> S
    {
        std::lock_guard<std::mutex> lock(m_);
        return inf_norm_;
    }

    //
    // Returns the indices of the elements of the subsequences in their
    // original sequences.
    //
    // Let a, b denote the original sequences, and sseq_a, sseq_b denote
    // subsequences computed by functor `clss`.  Write:
    //
    // `auto [a_ids, b_ids] = clss::subseqs_ids();`
    //
    // Then:
    //
    // a) For 0 <= i < sseq_a.size(), sseq_a[i] == a[a_ids[i]];
    //
    // b) For 0 <= j < sseq_b.size(), sseq_b[i] == b[b_ids[j]].
    //
    // \return std::pair of std::vector containing indices of subsequences'
    // elements as they appear in the original sequences.
    //
    auto subseqs_ids() -> std::pair<std::vector<S>, std::vector<S>>
    {
        std::lock_guard<std::mutex> lock(m_);
        return std::make_pair(sseq_a_ids_, sseq_b_ids_);
    }

    //
    // Returns two subsequences satisfying properties (1), (2) and (3)
    // of the functor description.
    //
    // \return std::pair of std::vector containing subsequences.
    //
    auto subseqs() -> std::pair<std::vector<S>, std::vector<S>>
    {
        std::lock_guard<std::mutex> lock(m_);
        return std::make_pair(sseq_a_, sseq_b_);
    }

    //
    // Returns the number of elements of the subsequences.
    //
    // \return number of elements of the subsequences.
    //
    auto subseqs_size() -> I
    {
        std::lock_guard<std::mutex> lock(m_);
        return sseqs_size_;
    }

    ///
    /// For debugging
    ///

    //
    // Prints internal information for debugging purposes.
    //
    // \return std::string with debug information.
    //
    auto print_debug_str() -> std::string
    {
        std::ostringstream os;
        return print_debug(os).str();
    }

    //
    // Prints internal information for debugging purposes.
    //
    // \param os: reference to a variable of a type that derives from
    // std::ostream, in which debug information is meant to be appended to.
    //
    // \return *reference* to input parameter `os`, for convenience.
    //
    // See `clss::print_debug_str` for an example of usage.
    //
    template <typename K = std::ostringstream,
              typename = typename std::enable_if<std::is_base_of_v<std::ostream, K>>::type>
    [[maybe_unused]] auto print_debug(K& os) -> K&
    {
        std::lock_guard<std::mutex> lock(m_);

        auto a = seq_a_.data();
        auto b = seq_b_.data();

        const auto default_precision{os.precision()};
        const auto digits
            = static_cast<I>(tol_ > S(0) ? std::ceil(-std::min(std::log10(tol_), S(0))) + 2
                                         : std::numeric_limits<T>::max_digits10);
        os << std::fixed << std::setprecision(digits);

        auto print_input_sequences = [&os](auto& a, auto a_size, auto& b, auto b_size) {
            os << ">>> Input: \n";

            os << ":: :: a = {";
            for(I i = 0; i < a_size; ++i)
            {
                os << a[i];
                if(i != a_size - 1)
                {
                    os << ", ";
                }
            }
            os << "}\n\n";

            os << ":: :: b = {";
            for(I i = 0; i < b_size; ++i)
            {
                os << b[i];
                if(i != b_size - 1)
                {
                    os << ", ";
                }
            }
            os << "}\n\n";
        };

        os << ">>>>>>>>>>>>\n";
        os << ":: :: closest_largest_subsequences::print_debug()\n\n" << std::flush;
        print_input_sequences(a, size_a_, b, size_b_);
        os << ":: :: tol = " << tol_ << std::endl << std::endl;

        os << "++++++++++++\n";
        os << ":: :: Subsequences sub_a, sub_b have distance: " << distance_
           << ", size: " << sseqs_size_ << ", and ||sub_a - sub_b||_inf = " << inf_norm_ << std::endl
           << std::endl;

        print_extract_subsequences(os);
        os << "<<<<<<<<<<<<\n" << std::flush;

        // Restore defaults
        os << std::setprecision(default_precision);

        return os;
    }

private:
    S tol_{};
    I sseqs_size_{};
    S distance_ = std::numeric_limits<S>::infinity();
    S inf_norm_ = std::numeric_limits<S>::infinity();
    I size_a_{};
    I size_b_{};
    std::vector<T> seq_a_{};
    std::vector<T> seq_b_{};
    std::vector<T> sseq_a_{};
    std::vector<T> sseq_b_{};
    std::vector<T> sseq_a_ids_{};
    std::vector<T> sseq_b_ids_{};
    std::vector<S> memo_distances_{};
    std::vector<I> memo_sizes_{};
    std::vector<I> memo_next_{};
    std::mutex m_;

    void clear()
    {
        tol_ = {};
        sseqs_size_ = {};
        distance_ = std::numeric_limits<T>::infinity();
        inf_norm_ = std::numeric_limits<S>::infinity();
        size_a_ = {};
        size_b_ = {};
        seq_a_ = {};
        seq_b_ = {};
        sseq_a_ = {};
        sseq_b_ = {};
        sseq_a_ids_ = {};
        sseq_b_ids_ = {};
        memo_distances_ = {};
        memo_sizes_ = {};
        memo_next_ = {};
    }

    /// Recursive implementation with memoization
    auto clss_implr(T const* a, I sa, T const* b, I sb)
        -> std::tuple</* acc distance */ S, /* size */ I, /* next */ I>
    {
        //
        // Base case: at least one of the sequences is empty
        //
        if(!in_range(sa, sb))
        {
            return std::make_tuple(std::numeric_limits<S>::infinity(), I(0), I(-1));
        }

        //
        // If `dist`, `size` and `next_index` have already been computed for this pair of `sa`, `sb` return
        //
        auto [dist, size, _] = memo(sa, sb);
        I next_index = I(-1);

        if(memo_valid(dist, size))
        {
            // Make next entry point to this one
            next_index = ij2index(sa, sb);

            return std::make_tuple(dist, size, next_index);
        }

        //
        // Otherwise, compute new `dist`, `size` and `next_index`
        //

        // Initialize local vars
        dist = std::numeric_limits<S>::infinity();
        size = I(0);
        // Compare current optimum (dist, size) with candidate optimum (d, s), and update if necessary
        auto do_update = [](S d, I s, I nindex, S& dist, I& size, I& next_index) -> bool {
            bool update = false;
            if(size < s)
            {
                dist = d;
                size = s;
                next_index = nindex;
                update = true;
            }
            else if(size == s)
            {
                if(dist > d)
                {
                    dist = d;
                    next_index = nindex;
                    update = true;
                }
            }

            return update;
        };
        [[maybe_unused]] bool update = false;

        // Case 1: a[0] .==. b[0], try to match next element of sequence `a` with next element of sequence `b`
        if(equiv(a[0], b[0]))
        {
            auto [d, s, nindex] = clss_implr(a + I(1), sa - I(1), b + I(1), sb - I(1));
            if(d == std::numeric_limits<S>::infinity())
            {
                dist = std::abs(a[0] - b[0]);
                size = I(1);
                next_index = ij2index(sa, sb);
                update = true;
            }
            else
            {
                d += std::abs(a[0] - b[0]);
                ++s;
                update = do_update(d, s, nindex, dist, size, next_index);
            }
        }

        // Case 2: try to match next element of sequence `a` with current element of sequence `b`
        {
            auto [d, s, nindex] = clss_implr(a + I(1), sa - I(1), b, sb);
            update = do_update(d, s, nindex, dist, size, next_index);
        }

        // Case 3: try to match current element of sequence `a` with next element of sequence `b`
        {
            auto [d, s, nindex] = clss_implr(a, sa, b + I(1), sb - I(1));
            update = do_update(d, s, nindex, dist, size, next_index);
        }

        // Save best results from 3 cases
        memo_dist(sa, sb) = dist;
        memo_size(sa, sb) = size;
        memo_next(sa, sb) = next_index;

        // Make next entry point to this one
        next_index = ij2index(sa, sb);

        return std::make_tuple(dist, size, next_index);
    }

    auto extract_subsequences(T const* a, I size_a, T const* b, I size_b)
        -> /* || sseq_a_ - sseq_b_ ||_inf */ S
    {
        S inf_norm = std::numeric_limits<S>::infinity();
        I sa = size_a - I(1);
        I sb = size_b - I(1);

        I index = ij2index(sa, sb);
        if(!in_range(index) || (sseqs_size_ == I(0)))
        {
            return inf_norm;
        }

        I next_index = index;
        inf_norm = static_cast<S>(0);
        do
        {
            index = next_index;
            next_index = memo_next(index);
            next_index = in_range(next_index) ? next_index : index;

            I ia, ib;
            I si = memo_size(index);
            I nsi = memo_size(next_index);
            if((nsi < si) || (index == next_index))
            {
                auto [ja, jb] = index2ij(index);

                ia = sa - ja;
                sseq_a_ids_.push_back(ia);
                sseq_a_.push_back(a[ia]);

                ib = sb - jb;
                sseq_b_ids_.push_back(ib);
                sseq_b_.push_back(b[ib]);

                S norm = std::abs(a[ia] - b[ib]);
                inf_norm = std::max(inf_norm, norm);
            }
        } while((index != next_index) && in_range(index));

        return inf_norm;
    }

    template <typename K = std::ostream>
    void print_extract_subsequences(K&& os)
    {
        os << ">>> Traversing:";
        I sa = size_a_ - I(1);
        I sb = size_b_ - I(1);
        I index = ij2index(sa, sb);
        if(!in_range(index) || (sseqs_size_ == I(0)))
        {
            os << " nothing to print\n";
            return;
        }
        os << std::endl;

        I next_index = index, i = I(0);
        do
        {
            index = next_index;
            next_index = memo_next(index);
            next_index = in_range(next_index) ? next_index : index;

            I ia, ib;
            I si = memo_size(index);
            I nsi = memo_size(next_index);
            if((nsi < si) || (index == next_index))
            {
                auto [ja, jb] = index2ij(index);

                ia = sa - ja;
                ib = sb - jb;

                os << ""
                   << ":: :: Indices: (" << ia << ", " << ib << ") :: Elements: (" << sseq_a_[i]
                   << ", " << sseq_b_[i] << ") :: (acc dist = " << memo_dist(ja, jb)
                   << ", size = " << memo_size(ja, jb) << ")\n";
                ++i;
            }
        } while((index != next_index) && in_range(index));

        return;
    }

    ///
    /// Helper functions
    ///

    /// lhs, rhs are "equivalent" (symbolyc notation: lhs .=. rhs)
    /// when |lhs - rhs| <= tol.
    ///
    /// This is not a true equivalence relation.
    bool equiv(T lhs, T rhs) const
    {
        if(std::abs(lhs - rhs) <= tol_)
        {
            return true;
        }

        return false;
    }

    bool in_range(I i, I j) const
    {
        bool in_range = false;

        if((i >= 0) && (i < size_a_) && (j >= I(0)) && (j < size_b_))
        {
            in_range = true;
        }

        return in_range;
    }

    bool in_range(I index) const
    {
        bool in_range = false;

        I upper_bound = size_a_ * size_b_;
        if((index >= I(0)) && (index < upper_bound))
        {
            in_range = true;
        }

        return in_range;
    }

    auto memo(I i, I j) -> std::tuple<S, I, I> const
    {
        auto d = memo_dist(i, j);
        auto s = memo_size(i, j);
        auto n = memo_next(i, j);

        return std::make_tuple(d, s, n);
    }

    S memo_dist(I i, I j) const&&
    {
        auto x = memo_distances_[ij2index(i, j)];
        return x;
    }

    S& memo_dist(I i, I j) &
    {
        auto& x = memo_distances_[ij2index(i, j)];
        return x;
    }

    I memo_size(I i, I j) const&&
    {
        auto x = memo_sizes_[ij2index(i, j)];
        return x;
    }

    I& memo_size(I i, I j) &
    {
        auto& x = memo_sizes_[ij2index(i, j)];
        return x;
    }

    I memo_size(I index) const&&
    {
        auto x = memo_sizes_[index];
        return x;
    }

    I& memo_size(I index) &
    {
        auto& x = memo_sizes_[index];
        return x;
    }

    I memo_next(I i, I j) const&&
    {
        auto x = memo_next_[ij2index(i, j)];
        return x;
    }

    I& memo_next(I i, I j) &
    {
        auto& x = memo_next_[ij2index(i, j)];
        return x;
    }

    I memo_next(I index) const&&
    {
        auto x = memo_next_[index];
        return x;
    }

    I& memo_next(I index) &
    {
        auto& x = memo_next_[index];
        return x;
    }

    bool memo_valid(S d, I s) const
    {
        bool valid = true;
        if((d == S(-1)) || (s == I(-1)))
        {
            valid = false;
        }

        return valid;
    }

    auto ij2index(I i, I j) -> I const
    {
        return i + size_a_ * j;
    }

    auto index2ij(I index) -> std::pair<I, I> const
    {
        I i = index % size_a_;
        I j = (index - i) / size_a_;
        return std::make_pair(i, j);
    }
};
