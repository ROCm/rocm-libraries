/* ************************************************************************
 * Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_PRECONDITIONER_MULTICOLORED_HPP_
#define ROCALUTION_PRECONDITIONER_MULTICOLORED_HPP_

#include "../../base/local_vector.hpp"
#include "../solver.hpp"
#include "preconditioner.hpp"

#include <vector>

namespace rocalution
{

    /** \ingroup precond_module
  * \class MultiColored
  * \brief Base class for all multi-colored preconditioners
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class MultiColored : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        MultiColored();
        virtual ~MultiColored();

        virtual void Clear(void);

        virtual void Build(void);

        /** \brief Set a specific matrix type of the decomposed block matrices */
        void SetPrecondMatrixFormat(unsigned int mat_format, int blockdim = 1);

        /** \brief Set if the preconditioner should be decomposed or not */
        void SetDecomposition(bool decomp);

        virtual void Solve(const VectorType& rhs, VectorType* x);

    protected:
        /** \brief Operator for analyzing */
        OperatorType* analyzer_op_;
        /** \brief Preconditioner */
        OperatorType* preconditioner_;

        /** \brief Preconditioner for each block */
        OperatorType*** preconditioner_block_;

        /** \brief Solution vector for each block */
        VectorType** x_block_;
        /** \brief Diagonal for each block */
        VectorType** diag_block_;
        /** \brief Solution vector */
        VectorType x_;
        /** \brief Diagonal */
        VectorType diag_;

        /** \brief Diagonal solver */
        Solver<OperatorType, VectorType, ValueType>** diag_solver_;

        /** \brief Number of blocks */
        int num_blocks_;
        /** \brief Block sizes */
        int* block_sizes_;

        /** \brief Keep the precond matrix in CSR or not */
        bool op_mat_format_;
        /** \brief Precond matrix format */
        unsigned int precond_mat_format_;
        /** \brief Matrix format block dimension */
        int format_block_dim_;

        /** \brief Decompose the preconditioner into blocks or not */
        bool decomp_;

        /** \brief Extract b into x under the permutation (see Analyse_()) and
      * decompose x into blocks (x_block_[])
      */
        void ExtractRHSinX_(const VectorType& rhs, VectorType* x);

        /** \brief Solve the lower-triangular (left) matrix */
        virtual void SolveL_(void) = 0;
        /** \brief Solve the diagonal part (only for SGS) */
        virtual void SolveD_(void) = 0;
        /** \brief Solve the upper-trianguler (right) matrix */
        virtual void SolveR_(void) = 0;

        /** \brief Solve directly without block decomposition */
        virtual void Solve_(const VectorType& rhs, VectorType* x) = 0;

        /** \brief Insert the solution with backward permutation (from x_block_[]) */
        void InsertSolution_(VectorType* x);

        /** \brief Build the analyzing matrix */
        virtual void Build_Analyser_(void);
        /** \brief Analyse the matrix (i.e. multi-coloring decomposition) */
        void Analyse_(void);
        /** \brief Permute the preconditioning matrix */
        void Permute_(void);
        /** \brief Factorize (i.e. build the preconditioner) */
        virtual void Factorize_(void);
        /** \brief Decompose the structure into blocks (preconditioner_block_[] for the
      * the preconditioning matrix; and x_block_[] for the x vector)
      */
        void Decompose_(void);
        /** \brief Post-analyzing if the preconditioner is not decomposed */
        virtual void PostAnalyse_(void);

        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);
    };

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_MULTICOLORED_HPP_
