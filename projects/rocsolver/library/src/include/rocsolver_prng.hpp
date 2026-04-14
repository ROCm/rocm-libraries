/* **************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <type_traits>

#include "rocblas_utility.hpp"

ROCSOLVER_BEGIN_NAMESPACE

namespace detail
{
///
/// @brief Pseudorandom number generator
///
/// Generates pseudorandom numbers 0 <= X < 2^31 - 1,
/// U = X/(2^31 - 1) in [0, 1) using the sequence
///
/// X_n = Y_n - Z_n, (*)
///
/// where Y_n, Z_n are the following linear congruential generators
/// - Y_n = 48271 * Y_{n-1} mod (2^31 - 1);
/// - Z_n = 40692 * Z_{n-1} mod (2^31 - 249).
///
/// Sequence X_n (*) has the following properties:
/// - it can be computed with 32-bit signed integer arithmetic;
/// - its period length size (2^31 - 2)(2^31 - 250)/62 ~ 2^56
///   renders it useful for generating significantly more than 2^32
///   pseudorandom numbers per run;
/// - it achieves good ratings on the spectral test.
///
/// This generator appears on Eq. 3.4.4-(38) of Donald Knuth, The Art
/// of Computer Programming Volume 2 -- Seminumerical Algorithms,
/// 3rd ed. See book and references therein for facts about Xn (*),
/// (in particular, see entries 20, 21 and 24 in Table 1, pg. 106,
/// and the discussion found in Subsection 3.3.4 near Eq. 3.4.4-(38)).
///
/// \tparam I Type used for input and output of numbers Xn, Yn, Zn.
///         Must be an integer and, at least, 32-bit wide.
///
template <typename I, typename = typename std::enable_if<std::is_integral<I>::value && sizeof(I) >= 4>::type>
struct pseudorandom_number_generator
{
    static constexpr std::int32_t m_m31_mXY
        = (1LL << 31) - 1; ///< Modulus of Xn and Yn, Mersenne prime M_31 = 2^31 - 1
    static constexpr std::int32_t m_mZ = (1LL << 31) - 249; ///< Modulus of Zn sequence, another prime

    ///
    /// Range check Y0, such that 0 < Y0 < m_m31_mXY = 2^31 - 1.
    ///
    /// The number Y' = range_check_Y(Y0) yields next_Y(Y') != 0,
    /// for all Y0. Hence, this method is meant to be used only
    /// when seeding the Zn generator.
    ///
    /// \param Z0 Input: number to check, output: 0 < Y' < 2^31 - 1.
    ///
    /// \return Number Y' lying strictly between 0 and m_mZ = 2^31 - 1.
    ///
    template <typename K = I,
              typename = typename std::enable_if<std::is_same<std::decay_t<K>, I>::value>::type>
    __device__ __host__ static auto range_check_Y(K&& Y0) -> I
    {
        Y0 = std::max(Y0 % static_cast<I>(m_m31_mXY), static_cast<I>(1));
        return Y0;
    }

    ///
    /// Range check Z0, such that 0 < Z0 < m_mZ = 2^31 - 249.
    ///
    /// The number Z' = range_check_Z(Z0) yields next_Z(Z') != 0,
    /// for all Z0. Hence, this method is meant to be used only
    /// when seeding the Zn generator.
    ///
    /// \param Z0 Input: number to check, output: 0 < Z' < 2^31 - 249.
    ///
    /// \return Number Z' lying strictly between 0 and m_mZ = 2^31 - 249.
    ///
    template <typename K = I,
              typename = typename std::enable_if<std::is_same<std::decay_t<K>, I>::value>::type>
    __device__ __host__ static auto range_check_Z(K&& Z0) -> I
    {
        Z0 = std::max(Z0 % static_cast<I>(m_mZ), static_cast<I>(1));
        return Z0;
    }

    ///
    /// Iterates Y, Z and computes U = (Y - Z)/(2^31 - 1) \in [0., 1.).
    ///
    /// \tparam S Output type, typically a floating point number.
    ///
    /// \param Y Input: iterate n-1 of Yn, output: next_Y(Y).
    ///
    /// \param Z [Optional] Input: iterate n-1 of Zn, output: next_Z(Z).
    //
    /// \return Number (Y - Z)/(2^31 - 1) cast to type S.
    ///
    template <typename S,
              typename K = I,
              typename = typename std::enable_if<std::is_same<std::decay_t<K>, I>::value>::type>
    __device__ __host__ static auto uniform01(K&& Y, K&& Z = 0) -> S
    {
        constexpr double range = static_cast<double>(m_m31_mXY);
        I Xnext = next_X(Y, Z);
        double U = static_cast<double>(Xnext) / range;
        return static_cast<S>(U);
    }

    ///
    /// Computes next iterate of sequence Xn.
    ///
    /// \param Y Input: iterate n-1 of Yn, output: next_Y(Y).
    ///
    /// \param Z [Optional] Input: iterate n-1 of Zn, output: next_Z(Z).
    ///
    /// \return Next iterate of Xn, 0 <= Xnext < m_m31_mXY = 2^31 - 1.
    ///
    template <typename K = I,
              typename = typename std::enable_if<std::is_same<std::decay_t<K>, I>::value>::type>
    __device__ __host__ static auto next_X(K&& Y, K&& Z = 0) -> I
    {
        std::int32_t Xnext, Ynext, Znext, sgnX;

        Ynext = static_cast<std::int32_t>(next_Y(Y));
        Znext = static_cast<std::int32_t>(next_Z(Z));

        Xnext = Ynext - Znext;
        sgnX = Xnext < 0 ? -1 : 1;
        Xnext += std::max(-sgnX, 0) * m_m31_mXY;

        return static_cast<I>(Xnext);
    }

    ///
    /// Computes next iterate of sequence Yn.
    ///
    /// Facts:
    /// - Yn has period m_m31_mXY = 2^31 - 2;
    /// - next_Y(0) == 0;
    /// - if 0 < Y' < 2^31 - 1, next_Y(Y') != 0,
    ///   thus, for all integers k, next_Y(range_check_Y(k)) != 0.
    ///
    /// \param Y Input: iterate n-1 of Yn, output: next_Y(Y).
    ///
    /// \return Next iterate of Yn, 0 <= Ynext < m_m31_mXY = 2^31 - 1.
    ///
    template <typename K = I,
              typename = typename std::enable_if<std::is_same<std::decay_t<K>, I>::value>::type>
    __device__ __host__ static auto next_Y(K&& Y) -> I
    {
        std::int32_t Ynext = static_cast<std::int32_t>(Y), sgnY;

        constexpr std::int32_t ay = 48271;
        constexpr std::int32_t qy = 44488; // qy = floor(m_m31_mXY/ay)
        constexpr std::int32_t ry = 3399; // ry = m_m31_mXY % ay

        // Ynext = (ay * Y) % m_m31_mXY;
        Ynext = ay * (Ynext % qy) - ry * static_cast<std::int32_t>(Ynext / qy);
        sgnY = Ynext < 0 ? -1 : 1;
        Ynext += std::max(-sgnY, 0) * m_m31_mXY;

        Y = static_cast<I>(Ynext);
        return Y;
    }

    ///
    /// Computes next iterate of sequence Zn.
    ///
    /// Facts:
    /// - Zn has period m_Z = 2^31 - 250;
    /// - next_Z(0) == 0;
    /// - if 0 < Z' < 2^31 - 249, next_Z(Z') != 0,
    ///   thus, for all integers k, next_Z(range_check_Z(k)) != 0.
    ///
    /// \param Z Input: iterate n-1 of Zn, output: next_Z(Z).
    ///
    /// \return Next iterate of Zn, 0 <= Znext < m_Z = 2^31 - 249.
    ///
    template <typename K = I,
              typename = typename std::enable_if<std::is_same<std::decay_t<K>, I>::value>::type>
    __device__ __host__ static auto next_Z(K&& Z) -> I
    {
        std::int32_t Znext = static_cast<std::int32_t>(Z), sgnZ;

        constexpr std::int32_t az = 40692;
        constexpr std::int32_t qz = 52774; // qz = floor(m_mZ/az)
        constexpr std::int32_t rz = 3791; // rz = m_mZ % az

        // Znext = (az * Z) % m_mZ;
        Znext = az * (Znext % qz) - rz * static_cast<std::int32_t>(Znext / qz);
        sgnZ = Znext < 0 ? -1 : 1;
        Znext += std::max(-sgnZ, 0) * m_mZ;

        Z = static_cast<I>(Znext);
        return Z;
    }
};
} /// namespace detail

/// \class rocSOLVER wrapper class for integer, pseudo-random, number generation
///
/// \brief Produces pseudo-random integer values distributed on the interval
/// [0, rocsolver_int_prng<T>::max()].  If T is at least 32-bit wide,
/// then generated values are uniformly distributed.
///
/// \tparam T - type of generated values, usually an integral type.
///
/// \sa detail::pseudorandom_number_generator for more details.
///
template <class T = std::int32_t>
class rocsolver_int_prng
{
public:
    using rocsolver_prng_impl_t = typename detail::pseudorandom_number_generator<T>;

    /// Constructor.
    __device__ __host__ rocsolver_int_prng(T Y0, T Z0 = 0)
    {
        m_Y = rocsolver_prng_impl_t::range_check_Y(static_cast<std::int32_t>(Y0));
        if(Z0 == 0)
        {
            m_Z = m_Y;
        }
        else
        {
            m_Z = rocsolver_prng_impl_t::range_check_Z(static_cast<std::int32_t>(Z0));
        }
    }

    /// Default destructor.
    __device__ __host__ ~rocsolver_int_prng() = default;

    /// Returns the smallest possible value that can be generated.
    static constexpr __device__ __host__ auto min() -> T
    {
        return static_cast<T>(0);
    }

    /// Returns the largest possible value that can be generated.
    static constexpr __device__ __host__ auto max() -> T
    {
        return static_cast<T>(rocsolver_prng_impl_t::m_m31_mXY - 1);
    }

    /// Generates a new pseudo-random integer.
    __device__ __host__ auto operator()(void) -> T
    {
        auto output = rocsolver_prng_impl_t::next_X(m_Y, m_Z);
        return static_cast<T>(output);
    }

private:
    std::int32_t m_Y{}, m_Z{};
};

ROCSOLVER_END_NAMESPACE
