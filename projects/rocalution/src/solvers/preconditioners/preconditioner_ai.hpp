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

#ifndef ROCALUTION_PRECONDITIONER_AI_HPP_
#define ROCALUTION_PRECONDITIONER_AI_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"
#include "rocalution/export.hpp"

namespace rocalution
{

    /** \ingroup precond_module
  * \class AIChebyshev
  * \brief Approximate Inverse - Chebyshev Preconditioner
  * \details
  * The Approximate Inverse - Chebyshev Preconditioner is an inverse matrix
  * preconditioner with values from a linear combination of matrix-valued
  * Chebyshev polynomials.
  * \cite chebpoly
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class AIChebyshev : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        AIChebyshev();
        ROCALUTION_EXPORT
        virtual ~AIChebyshev();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

        /** \brief Set order, min and max eigenvalues */
        ROCALUTION_EXPORT
        void Set(int p, ValueType lambda_min, ValueType lambda_max);
        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType AIChebyshev_;
        int          p_;
        ValueType    lambda_min_, lambda_max_;
    };

    /** \ingroup precond_module
  * \class FSAI
  * \brief Factorized Approximate Inverse Preconditioner
  * \details
  * The Factorized Sparse Approximate Inverse preconditioner computes a direct
  * approximation of \f$M^{-1}\f$ by minimizing the Frobenius norm \f$||I - GL||_{F}\f$,
  * where \f$L\f$ denotes the exact lower triangular part of \f$A\f$ and \f$G:=M^{-1}\f$.
  * The FSAI preconditioner is initialized by \f$q\f$, based on the sparsity pattern of
  * \f$|A^{q}|\f$. However, it is also possible to supply external sparsity patterns in form
  * of the LocalMatrix class.
  * \cite kolotilina
  *
  * \note
  * The FSAI preconditioner is only suited for symmetric positive definite matrices.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class FSAI : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        FSAI();
        ROCALUTION_EXPORT
        virtual ~FSAI();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

        /** \brief Set the power of the system matrix sparsity pattern */
        ROCALUTION_EXPORT
        void Set(int power);
        /** \brief Set an external sparsity pattern */
        ROCALUTION_EXPORT
        void Set(const OperatorType& pattern);

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Set the matrix format of the preconditioner */
        ROCALUTION_EXPORT
        void SetPrecondMatrixFormat(unsigned int mat_format, int blockdim = 1);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType FSAI_L_;
        OperatorType FSAI_LT_;
        VectorType   t_;

        int matrix_power_;

        bool                external_pattern_;
        const OperatorType* matrix_pattern_;

        // Keep the precond matrix in CSR or not
        bool op_mat_format_;
        // Precond matrix format
        unsigned int precond_mat_format_;
        // Matrix format block dimension
        int format_block_dim_;
    };

    /** \ingroup precond_module
  * \class SPAI
  * \brief SParse Approximate Inverse Preconditioner
  * \details
  * The SParse Approximate Inverse algorithm is an explicitly computed preconditioner for
  * general sparse linear systems. In its current implementation, only the sparsity
  * pattern of the system matrix is supported. The SPAI computation is based on the
  * minimization of the Frobenius norm \f$||AM - I||_{F}\f$.
  * \cite grote
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class SPAI : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        SPAI();
        ROCALUTION_EXPORT
        virtual ~SPAI();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);
        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Set the matrix format of the preconditioner */
        ROCALUTION_EXPORT
        void SetPrecondMatrixFormat(unsigned int mat_format, int blockdim = 1);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType SPAI_;

        // Keep the precond matrix in CSR or not
        bool op_mat_format_;
        // Precond matrix format
        unsigned int precond_mat_format_;
        // Matrix format block dimension
        int format_block_dim_;
    };

    /** \ingroup precond_module
  * \class TNS
  * \brief Truncated Neumann Series Preconditioner
  * \details
  * The Truncated Neumann Series (TNS) preconditioner is based on
  * \f$M^{-1} = K^{T} D^{-1} K\f$, where \f$K=(I-LD^{-1}+(LD^{-1})^{2})\f$, with the
  * diagonal \f$D\f$ of \f$A\f$ and the strictly lower triangular part \f$L\f$ of
  * \f$A\f$. The preconditioner can be computed in two forms - explicitly and implicitly.
  * In the explicit form, the full construction of \f$M\f$ is performed via matrix-matrix
  * operations, whereas in the implicit from, the application of the preconditioner is
  * based on matrix-vector operations only. The matrix format for the stored matrices can
  * be specified.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class TNS : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        TNS();
        ROCALUTION_EXPORT
        virtual ~TNS();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        /** \brief Set implicit (true) or explicit (false) computation */
        ROCALUTION_EXPORT
        void Set(bool imp);

        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);
        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Set the matrix format of the preconditioner */
        ROCALUTION_EXPORT
        void SetPrecondMatrixFormat(unsigned int mat_format, int blockdim = 1);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType L_;
        OperatorType LT_;
        OperatorType TNS_;
        VectorType   Dinv_;

        VectorType tmp1_;
        VectorType tmp2_;

        // Keep the precond matrix in CSR or not
        bool op_mat_format_;
        // Precond matrix format
        unsigned int precond_mat_format_;
        // Matrix format block dimension
        int format_block_dim_;
        // implicit (true) or explicit (false) computation
        bool impl_;
    };

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_AI_HPP_
