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
  * \brief Approximate inverse Chebyshev polynomial preconditioner.
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
  * \brief Factorized Sparse Approximate Inverse preconditioner for SPD systems.
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
  * \brief Sparse Approximate Inverse preconditioner.
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
  * \brief Truncated Neumann Series preconditioner.
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
