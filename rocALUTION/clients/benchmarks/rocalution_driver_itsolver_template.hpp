/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocalution_bench_solver_parameters.hpp"
#include "rocalution_driver_itsolver_traits.hpp"

//
// @brief This structure is responsible of defining a driver to configure an iterative linear solver.
//
template <rocalution_enum_itsolver::value_type ITSOLVER, typename T>
struct rocalution_driver_itsolver_template
{

protected:
    static constexpr bool s_verbose = true;
    using traits_t                  = rocalution_driver_itsolver_traits<ITSOLVER, T>;
    using solver_t                  = typename traits_t::solver_t;
    using preconditioner_t          = typename traits_t::preconditioner_t;
    using params_t                  = rocalution_bench_solver_parameters;

    //
    // Pure protected virtual methods.
    // ______________________________
protected:
    // @brief Get the preconditioner.
    // @return constant pointer to the preconditioner.
    virtual const preconditioner_t* GetPreconditioner() const = 0;

    // @brief Get the preconditioner.
    // @return pointer to the preconditioner.
    virtual preconditioner_t* GetPreconditioner() = 0;

    // @brief Set the preconditioner.
    // @param[in]
    // preconditioner pointer to the preconditioner.
    virtual void SetPreconditioner(preconditioner_t* preconditioner) = 0;

    // @brief Create preconditioner.
    // @param[in]
    // A           matrix of the linear system.
    // @param[in]
    // B           right-hand-side vector of the linear system.
    // @param[in]
    // X           unknown vector of the linear system.
    // @param[in]
    // parameters  parameters to configure the operation.
    // @return true if successful, false otherwise.
    virtual bool CreatePreconditioner(LocalMatrix<T>& A,
                                      LocalVector<T>& B,
                                      LocalVector<T>& X,
                                      const params_t& parameters)
        = 0;

    //
    // Protected virtual methods.
    // _________________________

protected:
    //
    // @brief Method called before the Build method from the solver being called.
    // @param[inout]
    // A           matrix of the linear system.
    // @param[inout]
    // B           right-hand-side vector of the linear system.
    // @param[inout]
    // X           unknown vector of the linear system.
    // @param[inout]
    // solver  linear solver.
    // @param[inout]
    // parameters  parameters to configure the operation.
    // @return true if successful, false otherwise.
    //
    virtual bool PreprocessSolverBuild(LocalMatrix<T>& A,
                                       LocalVector<T>& B,
                                       LocalVector<T>& X,
                                       solver_t&       solver,
                                       const params_t& parameters)
    {
        //
        // by default do nothing.
        //
        return true;
    }

    //
    // @brief Method called after the Build method from the solver being called.
    // @param[inout]
    // A           matrix of the linear system.
    // @param[inout]
    // B           right-hand-side vector of the linear system.
    // @param[inout]
    // X           unknown vector of the linear system.
    // @param[inout]
    // solver  linear solver.
    // @param[inout]
    // parameters  parameters to configure the operation.
    // @return true if successful, false otherwise.
    //
    virtual bool PostprocessSolverBuild(LocalMatrix<T>& A,
                                        LocalVector<T>& B,
                                        LocalVector<T>& X,
                                        solver_t&       solver,
                                        const params_t& parameters)
    {
        //
        // by default do nothing.
        //
        return true;
    }

    //
    // Public pure virtual methods.
    // ______________________
public:
    //
    // @brief Import the linear system Ax=b
    // @param[out]
    // A           matrix to import.
    // @param[out]
    // B           right-hand-side to import.
    // @param[out]
    // S           solution to import.
    // @param[in]
    // parameters  parameters to set up the import.
    // @return true if successful, false otherwise.
    //
    virtual bool ImportLinearSystem(LocalMatrix<T>& A,
                                    LocalVector<T>& B,
                                    LocalVector<T>& S,
                                    const params_t& parameters)
        = 0;

    //
    // Public virtual methods.
    // ______________________
public:
    //
    // @brief Method called before the execution linear solver.
    // @param[inout]
    // A           matrix of the linear system.
    // @param[inout]
    // B           right-hand-side vector of the linear system.
    // @param[inout]
    // X           unknown vector of the linear system.
    // @param[inout]
    // solver  linear solver.
    // @param[inout]
    // parameters  parameters to configure the operation.
    // @return true if successful, false otherwise.
    //
    virtual bool PreprocessLinearSolve(LocalMatrix<T>& A,
                                       LocalVector<T>& B,
                                       LocalVector<T>& X,
                                       solver_t&       solver,
                                       const params_t& parameters)
    {
        return true;
    }

    //
    // @brief Method called after the execution linear solver.
    // @param[inout]
    // A           matrix of the linear system.
    // @param[inout]
    // B           right-hand-side vector of the linear system.
    // @param[inout]
    // X           unknown vector of the linear system.
    // @param[inout]
    // solver      linear solver.
    // @param[inout]
    // parameters  parameters to configure the operation.
    // @return true if successful, false otherwise.
    //
    virtual bool PostprocessLinearSolve(LocalMatrix<T>& A,
                                        LocalVector<T>& B,
                                        LocalVector<T>& X,
                                        solver_t&       solver,
                                        const params_t& parameters)
    {
        return true;
    }

    //
    // @brief Postprocess the import, A, B and X are moved to accelerator.
    // @param[inout]
    // A           matrix of the linear system.
    // @param[inout]
    // B           right-hand-side vector of the linear system.
    // @param[inout]
    // S           solution vector of the linear system.
    // @param[inout]
    // solver  parameters to set up the import.
    // @param[inout]
    // parameters  parameters to configure the operation.
    // @return true if successful, false otherwise.
    //
    virtual bool PostprocessImportLinearSystem(LocalMatrix<T>& A,
                                               LocalVector<T>& B,
                                               LocalVector<T>& S,
                                               const params_t& parameters)
    {
        S.Allocate("s", A.GetN());
        B.Allocate("b", A.GetM());

        //
        // Get a temp vector and set it to one.
        //
        S.Ones();

        //
        // Matrix vector product to build B.
        //
        A.Apply(S, &B);
        return true;
    }

    //
    // @brief Configure the linear solver.
    //
    // @param[inout]
    // A           matrix of the linear system.
    // @param[inout]
    // B           right-hand-side vector of the linear system.
    // @param[inout]
    // X           unknown vector of the linear system.
    // @param[inout]
    // solver      linear solver
    // @param[inout]
    // parameters  parameters to configure the operation.
    // @return true if successful, false otherwise.
    //
    bool ConfigureLinearSolver(LocalMatrix<T>& A,
                               LocalVector<T>& B,
                               LocalVector<T>& X,
                               solver_t&       solver,
                               const params_t& parameters)
    {
        //
        // Get parameters.
        //
        const auto abs_tol  = parameters.Get(params_t::abs_tol);
        const auto rel_tol  = parameters.Get(params_t::rel_tol);
        const auto div_tol  = parameters.Get(params_t::div_tol);
        const auto max_iter = parameters.Get(params_t::max_iter);

        //
        // Create preconditioner.
        //
        if(s_verbose)
        {
            std::cout << "ConfigureLinearSolver CreatePreconditioner ..." << std::endl;
        }
        bool success = this->CreatePreconditioner(A, B, X, parameters);
        if(!success)
        {
            rocalution_bench_errmsg << "create preconditioner failed.." << std::endl;
            return false;
        }
        if(s_verbose)
        {
            std::cout << "ConfigureLinearSolver CreatePreconditioner done." << std::endl;
        }

        //
        // Initialize the solver.
        //
        solver.Verbose(0);
        solver.SetOperator(A);
        solver.Init(abs_tol, rel_tol, div_tol, max_iter);

        //
        // Get preconditioner.
        //
        auto* preconditioner = this->GetPreconditioner();
        if(preconditioner != nullptr)
        {
            solver.SetPreconditioner(*preconditioner);
        }

        //
        // Preprocess solver build.
        //
        if(s_verbose)
        {
            std::cout << "ConfigureLinearSolver PreprocessSolverBuild ..." << std::endl;
        }
        success = this->PreprocessSolverBuild(A, B, X, solver, parameters);
        if(!success)
        {
            rocalution_bench_errmsg << "preprocess solver build failed.." << std::endl;
            return false;
        }
        if(s_verbose)
        {
            std::cout << "ConfigureLinearSolver PreprocessSolverBuild done." << std::endl;
        }

        //
        // solver build.
        //
        solver.Build();

        //
        // Postprocess solver build.
        //
        if(s_verbose)
        {
            std::cout << "ConfigureLinearSolver PostprocessSolverBuild ..." << std::endl;
        }
        success = this->PostprocessSolverBuild(A, B, X, solver, parameters);
        if(!success)
        {
            rocalution_bench_errmsg << "postprocess solver build failed.." << std::endl;
            return false;
        }
        if(s_verbose)
        {
            std::cout << "ConfigureLinearSolver PostprocessSolverBuild ..." << std::endl;
        }

        return success;
    }
};
