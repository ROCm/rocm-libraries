/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_OPERATOR_HPP_
#define ROCALUTION_OPERATOR_HPP_

#include "base_rocalution.hpp"
#include "rocalution/export.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace rocalution
{

    template <typename ValueType>
    class GlobalVector;
    template <typename ValueType>
    class LocalVector;

    /** \ingroup op_vec_module
  * \class Operator
  * \brief Operator class
  * \details
  * The Operator class defines the generic interface for applying an operator (e.g.
  * matrix or stencil) from/to global and local vectors.
  *
  * \tparam ValueType - can be int, float, double, std::complex<float> and
  *                     std::complex<double>
  */
    template <typename ValueType>
    class Operator : public BaseRocalution<ValueType>
    {
    public:
        ROCALUTION_EXPORT
        Operator();
        ROCALUTION_EXPORT
        virtual ~Operator();

        /** \brief Return the number of rows in the matrix/stencil */
        virtual int64_t GetM(void) const = 0;
        /** \brief Return the number of columns in the matrix/stencil */
        virtual int64_t GetN(void) const = 0;
        /** \brief Return the number of non-zeros in the matrix/stencil */
        virtual int64_t GetNnz(void) const = 0;

        /** \brief Return the number of rows in the local matrix/stencil */
        ROCALUTION_EXPORT
        virtual int64_t GetLocalM(void) const;
        /** \brief Return the number of columns in the local matrix/stencil */
        ROCALUTION_EXPORT
        virtual int64_t GetLocalN(void) const;
        /** \brief Return the number of non-zeros in the local matrix/stencil */
        ROCALUTION_EXPORT
        virtual int64_t GetLocalNnz(void) const;

        /** \brief Return the number of rows in the ghost matrix/stencil */
        ROCALUTION_EXPORT
        virtual int64_t GetGhostM(void) const;
        /** \brief Return the number of columns in the ghost matrix/stencil */
        ROCALUTION_EXPORT
        virtual int64_t GetGhostN(void) const;
        /** \brief Return the number of non-zeros in the ghost matrix/stencil */
        ROCALUTION_EXPORT
        virtual int64_t GetGhostNnz(void) const;

        /** \brief Transpose the operator */
        ROCALUTION_EXPORT
        virtual void Transpose(void);

        /** \brief Apply the operator, out = Operator(in), where in and out are local
      * vectors
      */
        ROCALUTION_EXPORT
        virtual void Apply(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;

        /** \brief Apply and add the operator, out += scalar * Operator(in), where in and out
      * are local vectors
      */
        ROCALUTION_EXPORT
        virtual void ApplyAdd(const LocalVector<ValueType>& in,
                              ValueType                     scalar,
                              LocalVector<ValueType>*       out) const;

        /** \brief Apply the operator, out = Operator(in), where in and out are global
      * vectors
      */
        ROCALUTION_EXPORT
        virtual void Apply(const GlobalVector<ValueType>& in, GlobalVector<ValueType>* out) const;

        /** \brief Apply and add the operator, out += scalar * Operator(in), where in and out
      * are global vectors
      */
        ROCALUTION_EXPORT
        virtual void ApplyAdd(const GlobalVector<ValueType>& in,
                              ValueType                      scalar,
                              GlobalVector<ValueType>*       out) const;
    };

} // namespace rocalution

#endif // ROCALUTION_OPERTOR_HPP_
