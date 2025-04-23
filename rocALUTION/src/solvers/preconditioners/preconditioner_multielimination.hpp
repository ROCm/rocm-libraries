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

#ifndef ROCALUTION_PRECONDITIONER_MULTIELIMINATION_HPP_
#define ROCALUTION_PRECONDITIONER_MULTIELIMINATION_HPP_

#include "../../base/local_vector.hpp"
#include "../solver.hpp"
#include "preconditioner.hpp"
#include "rocalution/export.hpp"

#include <vector>

namespace rocalution
{

    /** \ingroup precond_module
  * \class MultiElimination
  * \brief Multi-Elimination Incomplete LU Factorization Preconditioner
  * \details
  * The Multi-Elimination Incomplete LU preconditioner is based on the following
  * decomposition
  * \f[
  *   A = \begin{pmatrix} D & F \\ E & C \end{pmatrix}
  *     = \begin{pmatrix} I & 0 \\ ED^{-1} & I \end{pmatrix} \times
  *       \begin{pmatrix} D & F \\ 0 & \hat{A} \end{pmatrix},
  * \f]
  * where \f$\hat{A} = C - ED^{-1} F\f$. To make the inversion of \f$D\f$ easier, we
  * permute the preconditioning before the factorization with a permutation \f$P\f$ to
  * obtain only diagonal elements in \f$D\f$. The permutation here is based on a maximal
  * independent set. This procedure can be applied to the block matrix \f$\hat{A}\f$, in
  * this way we can perform the factorization recursively. In the last level of the
  * recursion, we need to provide a solution procedure. By the design of the library,
  * this can be any kind of solver.
  * \cite SAAD
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class MultiElimination : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        MultiElimination();
        ROCALUTION_EXPORT
        virtual ~MultiElimination();

        /** \brief Returns the size of the first (diagonal) block of the preconditioner */
        inline int GetSizeDiagBlock(void) const
        {
            return this->size_;
        }

        /** \brief Return the depth of the current level */
        inline int GetLevel(void) const
        {
            return this->level_;
        }

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Initialize (recursively) ME-ILU with level (depth of recursion)
      * \details AA_Solvers - defines the last-block solver <br>
      * drop_off - defines drop-off tolerance
      */
        ROCALUTION_EXPORT
        void Set(Solver<OperatorType, VectorType, ValueType>& AA_Solver,
                 int                                          level,
                 double                                       drop_off = 0.0);

        /** \brief Set a specific matrix type of the decomposed block matrices */
        ROCALUTION_EXPORT
        void SetPrecondMatrixFormat(unsigned int mat_format, int blockdim = 1);

        ROCALUTION_EXPORT
        virtual void Build(void);

        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

    protected:
        /** \brief A_ is decomposed into \f$[D,F;E,C]\f$, where \f$AA=C-ED^{-1}F\f$ and
      * \f$E=ED^{-1}\f$
      */
        OperatorType A_;
        /** \brief Operator \$D\$ */
        OperatorType D_;
        /** \brief Operator \$E\$ */
        OperatorType E_;
        /** \brief Operator \$F\$ */
        OperatorType F_;
        /** \brief Operator \$C\$ */
        OperatorType C_;
        /** \brief \f$AA=C-ED^{-1}F\f$ */
        OperatorType AA_;

        /** \brief The sizes of the AA_ matrix */
        int64_t AA_nrow_;
        /** \brief The sizes of the AA_ matrix */
        int64_t AA_nnz_;

        /** \brief Keep the precond matrix in CSR or not */
        bool op_mat_format_;
        /** \brief Precond matrix format */
        unsigned int precond_mat_format_;
        /** \brief Matrix format block dimension */
        int format_block_dim_;

        /** \brief Vector x_ */
        VectorType x_;
        /** \brief Vector x_1_ */
        VectorType x_1_;
        /** \brief Vector x_2_ */
        VectorType x_2_;

        /** \brief Vector rhs_ */
        VectorType rhs_;
        /** \brief Vector rhs_1_ */
        VectorType rhs_1_;
        /** \brief Vector rhs_2_ */
        VectorType rhs_2_;

        /** \brief AA me preconditioner */
        MultiElimination<OperatorType, VectorType, ValueType>* AA_me_;
        /** \brief AA solver */
        Solver<OperatorType, VectorType, ValueType>* AA_solver_;

        /** \brief Diagonal solver init flag */
        bool diag_solver_init_;
        /** \brief Level */
        int level_;
        /** \brief Drop off */
        double drop_off_;

        /** \brief Inverse diagonal */
        VectorType inv_vec_D_;
        /** \brief Diagonal */
        VectorType vec_D_;
        /** \brief Permutation vector */
        LocalVector<int> permutation_;
        /** \brief Size */
        int size_;

        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);
    };

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_MULTIELIMINATION_HPP_
