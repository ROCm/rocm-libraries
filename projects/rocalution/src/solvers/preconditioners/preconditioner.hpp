/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_PRECONDITIONER_HPP_
#define ROCALUTION_PRECONDITIONER_HPP_

#include "../solver.hpp"
#include "rocalution/export.hpp"

namespace rocalution
{

    /** \ingroup precond_module
  * \class Preconditioner
  * \brief Base class for all preconditioners
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class Preconditioner : public Solver<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        Preconditioner();
        ROCALUTION_EXPORT
        virtual ~Preconditioner();

        ROCALUTION_EXPORT
        virtual void SolveZeroSol(const VectorType& rhs, VectorType* x);

    protected:
        virtual void PrintStart_(void) const;
        virtual void PrintEnd_(void) const;
    };

    /** \ingroup precond_module
  * \class Jacobi
  * \brief Jacobi preconditioner for diagonally dominant linear systems.
  *
  * \tparam OperatorType - can be LocalMatrix or GlobalMatrix
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class Jacobi : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        Jacobi();
        ROCALUTION_EXPORT
        virtual ~Jacobi();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);
        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        ROCALUTION_EXPORT
        virtual void ResetOperator(const OperatorType& op);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        VectorType inv_diag_entries_;
    };

    /** \ingroup precond_module
  * \class GS
  * \brief Gauss-Seidel / Successive Over-Relaxation preconditioner.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class GS : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        GS();
        ROCALUTION_EXPORT
        virtual ~GS();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);
        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        ROCALUTION_EXPORT
        virtual void ResetOperator(const OperatorType& op);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType GS_;
    };

    /** \ingroup precond_module
  * \class SGS
  * \brief Symmetric Gauss-Seidel / Symmetric Successive Over-Relaxation preconditioner.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class SGS : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        SGS();
        ROCALUTION_EXPORT
        virtual ~SGS();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);
        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        ROCALUTION_EXPORT
        virtual void ResetOperator(const OperatorType& op);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType SGS_;

        VectorType diag_entries_;
        VectorType v_;
    };

    /** \ingroup precond_module
  * \class ILU
  * \brief Incomplete LU factorization preconditioner based on fill levels.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class ILU : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        ILU();
        ROCALUTION_EXPORT
        virtual ~ILU();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

        /** \brief Initialize ILU(p) factorization
      * \details
      * Initialize ILU(p) factorization based on power.
      * \cite SAAD
      * - level = true build the structure based on levels
      * - level = false build the structure only based on the power(p+1)
      */
        ROCALUTION_EXPORT
        virtual void Set(int p, bool level = true);
        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType ILU_;
        int          p_;
        bool         level_;
    };

    /*! \brief List of ItILU0 algorithms.
     *  \details
     *  This is a list of supported algorithm types that are used to perform the ItILU0 preconditioner.
     */
    typedef enum _itilu0_alg : unsigned int
    {
        Default      = 0, /**< ASynchronous ITILU0 algorithm with in-place storage */
        AsyncInPlace = 1, /**< ASynchronous ITILU0 algorithm with in-place storage */
        AsyncSplit   = 2, /**< ASynchronous ITILU0 algorithm with explicit storage splitting */
        SyncSplit    = 3, /**< Synchronous ITILU0 algorithm with explicit storage splitting */
        SyncSplitFusion
        = 4 /**< Semi-synchronous ITILU0 algorithm with explicit storage splitting */
    } ItILU0Algorithm;

    /*! \brief List of ItILU0 options.
     *  \details
     *  This is a list of supported options that are used to perform the ItILU0 preconditioner.
     */
    typedef enum _itilu0_option : unsigned int
    {
        Verbose              = 1,
        StoppingCriteria     = 2,
        ComputeNrmCorrection = 4,
        ComputeNrmResidual   = 8,
        ConvergenceHistory   = 16,
        COOFormat            = 32
    } ItILU0Option;

    /** \ingroup precond_module
  * \class ItILU0
  * \brief Iterative Incomplete LU factorization with zero fill-in and no pivoting.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class ItILU0 : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        ItILU0();
        ROCALUTION_EXPORT
        virtual ~ItILU0();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

        /** \brief Initialize ItILU0(p) preconditioner.
      *
      * - alg       = Iterative Incomplete LU factorization algorithm.
      * - option    = Combination of ItILU0 option enumeration values
      * - max_iter  = Maximum number of iterations.
      * - tolerance = Tolerance to use for stopping criteria.
      */
        ROCALUTION_EXPORT
        void SetAlgorithm(ItILU0Algorithm alg);
        /** \brief Set the preconditioner options */
        ROCALUTION_EXPORT
        void SetOptions(int option);
        /** \brief Set the preconditioner convergence criteria */
        ROCALUTION_EXPORT
        void SetMaxIter(int max_iter);
        /** \brief Set the preconditioner convergence criteria */
        ROCALUTION_EXPORT
        void SetTolerance(double tolerance);

        /** \brief Get the convergence history */
        ROCALUTION_EXPORT
        const double* GetConvergenceHistory(int* niter);

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType    ItILU0_;
        ItILU0Algorithm alg_;
        int             option_;
        int             maxiter_;
        double          tol_;
        int             niter_{};
        double*         history_{};
    };

    /** \ingroup precond_module
  * \class ILUT
  * \brief Incomplete LU factorization preconditioner based on a drop threshold.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class ILUT : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        ILUT();
        ROCALUTION_EXPORT
        virtual ~ILUT();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

        /** \brief Set drop-off threshold */
        ROCALUTION_EXPORT
        virtual void Set(double t);

        /** \brief Set drop-off threshold and maximum fill-ins per row */
        ROCALUTION_EXPORT
        virtual void Set(double t, int maxrow);

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType ILUT_;
        double       t_;
        int          max_row_;
    };

    /** \ingroup precond_module
  * \class IC
  * \brief Incomplete Cholesky factorization preconditioner without fill-in.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class IC : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        IC();
        ROCALUTION_EXPORT
        virtual ~IC();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);
        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        OperatorType IC_;
        VectorType   inv_diag_entries_;
    };

    /** \ingroup precond_module
  * \class VariablePreconditioner
  * \brief Preconditioner that cycles through a sequence of preconditioners.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class VariablePreconditioner : public Preconditioner<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        VariablePreconditioner();
        ROCALUTION_EXPORT
        virtual ~VariablePreconditioner();

        ROCALUTION_EXPORT
        virtual void Print(void) const;
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);
        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Set the preconditioner sequence */
        ROCALUTION_EXPORT
        virtual void SetPreconditioner(int                                           n,
                                       Solver<OperatorType, VectorType, ValueType>** precond);

    protected:
        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);

    private:
        int                                           num_precond_;
        int                                           counter_;
        Solver<OperatorType, VectorType, ValueType>** precond_;
    };

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_HPP_
