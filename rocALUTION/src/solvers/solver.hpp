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

#ifndef ROCALUTION_SOLVER_HPP_
#define ROCALUTION_SOLVER_HPP_

#include "../base/base_rocalution.hpp"
#include "../base/local_vector.hpp"
#include "iter_ctrl.hpp"
#include "rocalution/export.hpp"

// HELPER DEFINITIONS
#define DISPATCH_OPERATOR_SOLVE_STRATEGY(descr_, op_, func_, ...) \
    switch(descr_.GetTriSolverAlg())                              \
    {                                                             \
    case TriSolverAlg_Default:                                    \
    {                                                             \
        op_.func_(__VA_ARGS__);                                   \
        break;                                                    \
    }                                                             \
    case TriSolverAlg_Iterative:                                  \
    {                                                             \
        op_.It##func_(descr_.GetIterativeSolverMaxIteration(),    \
                      descr_.GetIterativeSolverTolerance(),       \
                      descr_.GetIterativeSolverUseTolerance(),    \
                      __VA_ARGS__);                               \
        break;                                                    \
    }                                                             \
    }

#define DISPATCH_OPERATOR_ANALYSE_STRATEGY(descr_, op_, func_, ...) \
    switch(descr_.GetTriSolverAlg())                                \
    {                                                               \
    case TriSolverAlg_Default:                                      \
    {                                                               \
        op_.func_(__VA_ARGS__);                                     \
        break;                                                      \
    }                                                               \
    case TriSolverAlg_Iterative:                                    \
    {                                                               \
        op_.It##func_(__VA_ARGS__);                                 \
        break;                                                      \
    }                                                               \
    }

namespace rocalution
{
    /*! \brief Triangular system solve algorithms
     *  \details
     *  This is a list of algorithms to solve triangular systems
     */
    typedef enum _tri_solver_alg : unsigned int
    {
        TriSolverAlg_Default   = 0, /**< The default direct solver. */
        TriSolverAlg_Iterative = 1, /**< Iteratively solve triangular systems. */
    } TriSolverAlg;

    /** \ingroup precond_module
  * \class SolverDescr
  * \brief Descriptor class that controls the solving strategy.
  */
    class SolverDescr
    {
    public:
        ROCALUTION_EXPORT
        SolverDescr();
        /** \brief Constructor */
        ROCALUTION_EXPORT
        SolverDescr(const SolverDescr& other);

        ROCALUTION_EXPORT
        virtual ~SolverDescr(void);

        /** \brief operator=*/
        ROCALUTION_EXPORT
        SolverDescr& operator=(const SolverDescr& rhs);

        // Setters and Getters

        /** \brief Set triangular solver algorithm */
        ROCALUTION_EXPORT
        void SetTriSolverAlg(TriSolverAlg alg);
        /** \brief Get triangular solver algorithm */
        ROCALUTION_EXPORT
        TriSolverAlg GetTriSolverAlg(void) const;

        /** \brief Set maximum solver iterations */
        ROCALUTION_EXPORT
        void SetIterativeSolverMaxIteration(int max_iter);
        /** \brief Get maximum solver iterations */
        ROCALUTION_EXPORT
        int GetIterativeSolverMaxIteration(void) const;

        /** \brief Set solver tolerance */
        ROCALUTION_EXPORT
        void SetIterativeSolverTolerance(double tol);
        /** \brief Get solver tolerance */
        ROCALUTION_EXPORT
        double GetIterativeSolverTolerance(void) const;

        /** \brief Print solver stats */
        ROCALUTION_EXPORT
        void Print(void) const;

        /** \brief Enable tolerance as stopping criteria (default) */
        ROCALUTION_EXPORT
        void EnableIterativeSolverTolerance(void);

        /** \brief Disable tolerance as stopping criteria */
        ROCALUTION_EXPORT
        void DisableIterativeSolverTolerance(void);

        /** \brief Return if tolerance is used as a stopping criteria */
        ROCALUTION_EXPORT
        bool GetIterativeSolverUseTolerance(void) const;

    protected:
        /** \brief Triangular solver algorithm */
        TriSolverAlg tri_solver_alg_ = TriSolverAlg_Default;

        /** \brief Maximum solver iterations */
        int itsolver_max_iter_ = 30;

        /** \brief Solver tolerance */
        double itsolver_tol_ = 1e-3;

        /** \brief Use tolerance as stopping criteria */
        bool itsolver_use_tol_ = true;
    };

    /** \ingroup solver_module
  * \class Solver
  * \brief Base class for all solvers and preconditioners
  * \details
  * Most of the solvers can be performed on linear operators LocalMatrix, LocalStencil
  * and GlobalMatrix - i.e. the solvers can be performed locally (on a shared memory
  * system) or in a distributed manner (on a cluster) via MPI. The only exception is the
  * AMG (Algebraic Multigrid) solver which has two versions (one for LocalMatrix and one
  * for GlobalMatrix class). The only pure local solvers (which do not support global/MPI
  * operations) are the mixed-precision defect-correction solver and all direct solvers.
  *
  * All solvers need three template parameters - Operators, Vectors and Scalar type.
  *
  * The Solver class is purely virtual and provides an interface for
  * - SetOperator() to set the operator \f$A\f$, i.e. the user can pass the matrix here.
  * - Build() to build the solver (including preconditioners, sub-solvers, etc.). The
  *   user need to specify the operator first before calling Build().
  * - Solve() to solve the system \f$Ax = b\f$. The user need to pass a right-hand-side
  *   \f$b\f$ and a vector \f$x\f$, where the solution will be obtained.
  * - Print() to show solver information.
  * - ReBuildNumeric() to only re-build the solver numerically (if possible).
  * - MoveToHost() and MoveToAccelerator() to offload the solver (including
  *   preconditioners and sub-solvers) to the host/accelerator.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class Solver : public RocalutionObj
    {
    public:
        ROCALUTION_EXPORT
        Solver();
        ROCALUTION_EXPORT
        virtual ~Solver();

        /** \brief Set the Operator of the solver */
        ROCALUTION_EXPORT
        void SetOperator(const OperatorType& op);

        /** \brief Reset the operator; see ReBuildNumeric() */
        ROCALUTION_EXPORT
        virtual void ResetOperator(const OperatorType& op);

        /** \brief Print information about the solver */
        virtual void Print(void) const = 0;

        /** \brief Solve Operator x = rhs */
        virtual void Solve(const VectorType& rhs, VectorType* x) = 0;

        /** \brief Solve Operator x = rhs, setting initial x = 0 */
        ROCALUTION_EXPORT
        virtual void SolveZeroSol(const VectorType& rhs, VectorType* x);

        /** \brief Clear (free all local data) the solver */
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Build the solver (data allocation, structure and numerical computation) */
        ROCALUTION_EXPORT
        virtual void Build(void);

        /** \brief Build the solver and move it to the accelerator asynchronously */
        ROCALUTION_EXPORT
        virtual void BuildMoveToAcceleratorAsync(void);

        /** \brief Synchronize the solver */
        ROCALUTION_EXPORT
        virtual void Sync(void);

        /** \brief Rebuild the solver only with numerical computation (no allocation or data
        * structure computation)
        */
        ROCALUTION_EXPORT
        virtual void ReBuildNumeric(void);

        /** \brief Move all data (i.e. move the solver) to the host */
        ROCALUTION_EXPORT
        virtual void MoveToHost(void);
        /** \brief Move all data (i.e. move the solver) to the accelerator */
        ROCALUTION_EXPORT
        virtual void MoveToAccelerator(void);

        /** \brief Provide verbose output of the solver
        * \details
        * - verb = 0 -> no output
        * - verb = 1 -> print info about the solver (start, end);
        * - verb = 2 -> print (iter, residual) via iteration control;
        */
        ROCALUTION_EXPORT
        virtual void Verbose(int verb = 1);

        /** \brief Set solver descriptor */
        ROCALUTION_EXPORT
        virtual void SetSolverDescriptor(const SolverDescr& descr);

        /** \brief Mark this solver as being a preconditioner */
        inline void FlagPrecond(void)
        {
            this->is_precond_ = true;
        }

        /** \brief Mark this solver as being a smoother */
        inline void FlagSmoother(void)
        {
            this->is_smoother_ = true;
        }

    protected:
        /** \brief Pointer to the operator */
        const OperatorType* op_;

        /** \brief Pointer to the defined preconditioner */
        Solver<OperatorType, VectorType, ValueType>* precond_;

        /** \brief Solver descriptor */
        SolverDescr solver_descr_;

        /** \brief Flag to store whether this solver is a preconditioner or not */
        bool is_precond_;

        /** \brief Flag to store whether this solver is a smoother or not */
        bool is_smoother_;

        /** \brief Flag == true after building the solver (e.g. Build()) */
        bool build_;

        /** \brief Permutation vector (used if the solver performs permutation/re-ordering
      * techniques)
      */
        LocalVector<int> permutation_;

        /** \brief Verbose flag */
        int verb_;

        /** \brief Print starting message of the solver */
        virtual void PrintStart_(void) const = 0;
        /** \brief Print ending message of the solver */
        virtual void PrintEnd_(void) const = 0;

        /** \brief Move all local data to the host */
        virtual void MoveToHostLocalData_(void) = 0;
        /** \brief Move all local data to the accelerator */
        virtual void MoveToAcceleratorLocalData_(void) = 0;
    };

    /** \ingroup solver_module
  * \class IterativeLinearSolver
  * \brief Base class for all linear iterative solvers
  * \details
  * The iterative solvers are controlled by an iteration control object, which monitors
  * the convergence properties of the solver, i.e. maximum number of iteration, relative
  * tolerance, absolute tolerance and divergence tolerance. The iteration control can
  * also record the residual history and store it in an ASCII file.
  * - Init(), InitMinIter(), InitMaxIter() and InitTol() initialize the solver and set the
  * stopping criteria.
  * - RecordResidualHistory() and RecordHistory() start the recording of the residual and
  * write it into a file.
  * - Verbose() sets the level of verbose output of the solver (0 - no output, 2 - detailed
  * output, including residual and iteration information).
  * - SetPreconditioner() sets the preconditioning.
  *
  * All iterative solvers are controlled based on
  * - Absolute stopping criteria, when \f$|r_{k}|_{L_{p}} < \epsilon_{abs}\f$
  * - Relative stopping criteria, when \f$|r_{k}|_{L_{p}} / |r_{1}|_{L_{p}} \leq
  *   \epsilon_{rel}\f$
  * - Divergence stopping criteria, when \f$|r_{k}|_{L_{p}} / |r_{1}|_{L_{p}} \geq
  *   \epsilon_{div}\f$
  * - Maximum number of iteration \f$N\f$, when \f$k = N\f$
  *
  * where \f$k\f$ is the current iteration, \f$r_{k}\f$ the residual for the current
  * iteration \f$k\f$ (i.e. \f$r_{k} = b - Ax_{k}\f$) and \f$r_{1}\f$ the starting
  * residual (i.e. \f$r_{1} = b - Ax_{init}\f$). In addition, the minimum number of
  * iterations \f$M\f$ can be specified. In this case, the solver will not stop to
  * iterate, before \f$k \geq M\f$.
  *
  * The \f$L_{p}\f$ norm is used for the computation, where \f$p\f$ could be 1, 2 and
  * \f$\infty\f$. The norm computation can be set with SetResidualNorm() with 1 for
  * \f$L_{1}\f$, 2 for \f$L_{2}\f$ and 3 for \f$L_{\infty}\f$. For the computation with
  * \f$L_{\infty}\f$, the index of the maximum value can be obtained with
  * GetAmaxResidualIndex(). If this function is called and \f$L_{\infty}\f$ was not
  * selected, this function will return -1.
  *
  * The reached criteria can be obtained with GetSolverStatus(), returning
  * - 0, if no criteria has been reached yet
  * - 1, if absolute tolerance has been reached
  * - 2, if relative tolerance has been reached
  * - 3, if divergence tolerance has been reached
  * - 4, if maximum number of iteration has been reached
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class IterativeLinearSolver : public Solver<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        IterativeLinearSolver();
        ROCALUTION_EXPORT
        virtual ~IterativeLinearSolver();

        /** \brief Initialize the solver with absolute/relative/divergence tolerance and
      * maximum number of iterations
      */
        ROCALUTION_EXPORT
        void Init(double abs_tol, double rel_tol, double div_tol, int max_iter);

        /** \brief Initialize the solver with absolute/relative/divergence tolerance and
      * minimum/maximum number of iterations
      */
        ROCALUTION_EXPORT
        void Init(double abs_tol, double rel_tol, double div_tol, int min_iter, int max_iter);

        /** \brief Set the minimum number of iterations */
        ROCALUTION_EXPORT
        void InitMinIter(int min_iter);

        /** \brief Set the maximum number of iterations */
        ROCALUTION_EXPORT
        void InitMaxIter(int max_iter);

        /** \brief Set the absolute/relative/divergence tolerance */
        ROCALUTION_EXPORT
        void InitTol(double abs, double rel, double div);

        /** \brief Set the residual norm to \f$L_1\f$, \f$L_2\f$ or \f$L_\infty\f$ norm
      * \details
      * - resnorm = 1 -> \f$L_1\f$ norm
      * - resnorm = 2 -> \f$L_2\f$ norm
      * - resnorm = 3 -> \f$L_\infty\f$ norm
      */
        ROCALUTION_EXPORT
        void SetResidualNorm(int resnorm);

        /** \brief Record the residual history */
        ROCALUTION_EXPORT
        void RecordResidualHistory(void);

        /** \brief Write the history to file */
        ROCALUTION_EXPORT
        void RecordHistory(const std::string& filename) const;

        /** \brief Set the solver verbosity output */
        ROCALUTION_EXPORT
        virtual void Verbose(int verb = 1);

        /** \brief Solve Operator x = rhs */
        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

        /** \brief Set a preconditioner of the linear solver */
        ROCALUTION_EXPORT
        virtual void SetPreconditioner(Solver<OperatorType, VectorType, ValueType>& precond);

        /** \brief Return the iteration count */
        ROCALUTION_EXPORT
        virtual int GetIterationCount(void);

        /** \brief Return the current residual */
        ROCALUTION_EXPORT
        virtual double GetCurrentResidual(void);

        /** \brief Return the current status */
        ROCALUTION_EXPORT
        virtual int GetSolverStatus(void);

        /** \brief Return absolute maximum index of residual vector when using
      * \f$L_\infty\f$ norm
      */
        ROCALUTION_EXPORT
        virtual int64_t GetAmaxResidualIndex(void);

    protected:
        // Iteration control (monitor)
        IterationControl iter_ctrl_; /**< \private */

        /** \brief Non-preconditioner solution procedure */
        virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x) = 0;

        /** \brief Preconditioned solution procedure */
        virtual void SolvePrecond_(const VectorType& rhs, VectorType* x) = 0;

        /** \brief Residual norm type (i.e. L1, L2, L-infinity etc) */
        int res_norm_type_;

        /** \brief Absolute maximum index of residual vector when using \f$L_\infty\f$ */
        int64_t index_;

        /** \brief Computes the vector norm */
        ValueType Norm_(const VectorType& vec);
    };

    /** \ingroup solver_module
  * \class FixedPoint
  * \brief Fixed-Point Iteration Scheme
  * \details
  * The Fixed-Point iteration scheme is based on additive splitting of the matrix
  * \f$A = M + N\f$. The scheme reads
  * \f[
  *   x_{k+1} = M^{-1} (b - N x_{k}).
  * \f]
  * It can also be reformulated as a weighted defect correction scheme
  * \f[
  *   x_{k+1} = x_{k} - \omega M^{-1} (Ax_{k} - b).
  * \f]
  * The inversion of \f$M\f$ can be performed by preconditioners (Jacobi, Gauss-Seidel,
  * ILU, etc.) or by any type of solvers.
  *
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class FixedPoint : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        FixedPoint();
        ROCALUTION_EXPORT
        virtual ~FixedPoint();

        ROCALUTION_EXPORT
        virtual void Print(void) const;

        ROCALUTION_EXPORT
        virtual void ReBuildNumeric(void);

        /** \brief Set relaxation parameter \f$\omega\f$ */
        ROCALUTION_EXPORT
        void SetRelaxation(ValueType omega);

        ROCALUTION_EXPORT
        virtual void Build(void);
        ROCALUTION_EXPORT
        virtual void Clear(void);

        /** \brief Solve Operator x = rhs, setting initial x = 0 */
        ROCALUTION_EXPORT
        virtual void SolveZeroSol(const VectorType& rhs, VectorType* x);

    protected:
        /** \brief Solve Operator x = rhs, setting initial x = 0 */
        void SolveZeroSol_(const VectorType& rhs, VectorType* x);

        /** \brief Relaxation parameter */
        ValueType  omega_;
        VectorType x_old_; /**< \private */
        VectorType x_res_; /**< \private */

        virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
        virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

        virtual void PrintStart_(void) const;
        virtual void PrintEnd_(void) const;

        virtual void MoveToHostLocalData_(void);
        virtual void MoveToAcceleratorLocalData_(void);
    };

    /** \ingroup solver_module
  * \class DirectLinearSolver
  * \brief Base class for all direct linear solvers
  * \details
  * The library provides three direct methods - LU, QR and Inversion (based on QR
  * decomposition). The user can pass a sparse matrix, internally it will be converted to
  * dense and then the selected method will be applied. These methods are not very
  * optimal and due to the fact that the matrix is converted to a dense format, these
  * methods should be used only for very small matrices.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
    template <class OperatorType, class VectorType, typename ValueType>
    class DirectLinearSolver : public Solver<OperatorType, VectorType, ValueType>
    {
    public:
        ROCALUTION_EXPORT
        DirectLinearSolver();
        ROCALUTION_EXPORT
        virtual ~DirectLinearSolver();

        ROCALUTION_EXPORT
        virtual void Verbose(int verb = 1);

        ROCALUTION_EXPORT
        virtual void Solve(const VectorType& rhs, VectorType* x);

    protected:
        /** \brief Solve Operator x = rhs */
        virtual void Solve_(const VectorType& rhs, VectorType* x) = 0;
    };

} // namespace rocalution

#endif // ROCALUTION_SOLVER_HPP_
