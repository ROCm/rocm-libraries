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

#ifndef ROCALUTION_PRECONDITIONER_SADDLEPOINT_HPP_
#define ROCALUTION_PRECONDITIONER_SADDLEPOINT_HPP_

#include "../../base/local_vector.hpp"
#include "../solver.hpp"
#include "preconditioner.hpp"
#include "rocalution/export.hpp"

#include <vector>

namespace rocalution
{

    /** \ingroup precond_module
  * \class DiagJacobiSaddlePointPrecond
  * \brief Diagonal Preconditioner for Saddle-Point Problems
  * \details
  * Consider the following saddle-point problem
  * \f[
  *   A = \begin{pmatrix} K & F \\ E & 0 \end{pmatrix}.
  * \f]
  * For such problems we can construct a diagonal Jacobi-type preconditioner of type
  * \f[
  *   P = \begin{pmatrix} K & 0 \\ 0 & S \end{pmatrix},
  * \f]
  * with \f$S=ED^{-1}F\f$, where \f$D\f$ are the diagonal elements of \f$K\f$. The matrix
  * \f$S\f$ is fully constructed (via sparse matrix-matrix multiplication). The
  * preconditioner needs to be initialized with two external solvers/preconditioners -
  * one for the matrix \f$K\f$ and one for the matrix \f$S\f$.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class DiagJacobiSaddlePointPrecond : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        DiagJacobiSaddlePointPrecond();
        ROCALUTION_EXPORT
        virtual ~DiagJacobiSaddlePointPrecond();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Initialize solver for \f$K\f$ and \f$S\f$ */
        ROCALUTION_EXPORT
        void Set(Solver<OperatorType, VectorType, ValueType>& K_Solver,
                 Solver<OperatorType, VectorType, ValueType>& S_Solver);

        ROCALUTION_EXPORT
        virtual void Build(void);

        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

    protected:
        /** \brief A_ is decomposed into \f$[K,F;E,0]\f$ */
        OperatorType A_;
        /** \brief Operator \f$K\f$ */
        OperatorType K_;
        /** \brief Operator \f$S\f$ */
        OperatorType S_;

        /** \brief The sizes of the \f$K\f$ matrix */
        int K_nrow_;
        /** \brief The sizes of the \f$K\f$ matrix */
        int K_nnz_;

        /** \brief Keep the precond matrix in CSR or not */
        bool op_mat_format_;
        /** \brief Precond matrix format */
        unsigned int precond_mat_format_;

        /** \brief Vector x_ */
        VectorType x_;
        /** \brief Vector x_1_ */
        VectorType x_1_;
        /** \brief Vector x_2_ */
        VectorType x_2_;
        /** \brief Vector x_1tmp_ */
        VectorType x_1tmp_;

        /** \brief Vector rhs_ */
        VectorType rhs_;
        /** \brief Vector rhs_1_ */
        VectorType rhs_1_;
        /** \brief Vector rhs_2_ */
        VectorType rhs_2_;

        /** \brief Solver for \f$K\f$ */
        Solver<OperatorType, VectorType, ValueType>* K_solver_;
        /** \brief Solver for \f$S\f$ */
        Solver<OperatorType, VectorType, ValueType>* S_solver_;

        /** \brief Permutation vector */
        LocalVector<int> permutation_;
        /** \brief Size */
        int size_;

        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);
    };

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_SADDLEPOINT_HPP_
