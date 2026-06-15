/* ************************************************************************
 * Copyright (C) 2018-2021 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_PRECONDITIONER_AS_HPP_
#define ROCALUTION_PRECONDITIONER_AS_HPP_

#include "preconditioner.hpp"
#include "rocalution/export.hpp"

namespace rocalution
{

    /** \ingroup precond_module
  * \class AS
  * \brief Additive Schwarz Preconditioner
  * \details
  * The Additive Schwarz preconditioner relies on a preconditioning technique, where the
  * linear system \f$Ax=b\f$ can be decomposed into small sub-problems based on
  * \f$A_{i} = R_{i}^{T}AR_{i}\f$, where \f$R_{i}\f$ are restriction operators. Those
  * restriction operators produce sub-matrices wich overlap. This leads to contributions
  * from two preconditioners on the overlapped area which are scaled by \f$1/2\f$.
  * \cite RAS
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class AS : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        AS();
        ROCALUTION_EXPORT
        virtual ~AS();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        /** \brief Set number of blocks, overlap and array of preconditioners */
        ROCALUTION_EXPORT
        void Set(int nb, int overlap, Solver<OperatorType, VectorType, ValueType>** preconds);

        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

        /** \brief Number of blocks */
        int num_blocks_; /**< Number of blocks */
        /** \brief Overlap */
        int overlap_;
        /** \brief Position */
        int* pos_;
        /** \brief Sizes including overlap */
        int* sizes_;

        /** \brief Preconditioner for each block */
        Solver<OperatorType, VectorType, ValueType>** local_precond_;

        /** \brief Local operator */
        OperatorType** local_mat_;
        /** \brief r */
        VectorType** r_;
        /** \brief z */
        VectorType** z_;
        /** \brief weights */
        VectorType weight_;
    };

    /** \ingroup precond_module
  * \class RAS
  * \brief Restricted Additive Schwarz Preconditioner
  * \details
  * The Restricted Additive Schwarz preconditioner relies on a preconditioning technique,
  * where the linear system \f$Ax=b\f$ can be decomposed into small sub-problems based on
  * \f$A_{i} = R_{i}^{T}AR_{i}\f$, where \f$R_{i}\f$ are restriction operators. The RAS
  * method is a mixture of block Jacobi and the AS scheme. In this case, the sub-matrices
  * contain overlapped areas from other blocks, too.
  * \cite RAS
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class RAS : public AS<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        RAS();
        ROCALUTION_EXPORT
        virtual ~RAS();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);
    };

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_AS_HPP_
