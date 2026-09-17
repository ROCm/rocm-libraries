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

#include <rocalution/rocalution.hpp>
using namespace rocalution;

#include "rocalution_bench_itsolver.hpp"
#include "rocalution_driver_itsolver_fgmres.hpp"
#include "rocalution_driver_itsolver_gmres.hpp"
#include "rocalution_driver_itsolver_uaamg.hpp"

static constexpr bool s_verbose = true;

bool rocalution_bench_record_results(const rocalution_bench_solver_parameters&,
                                     const rocalution_bench_solver_results&);

//
// @brief This structure is an implementation of \ref rocalution_bench_itsolver.
//
template <rocalution_enum_itsolver::value_type ITSOLVER, typename T>
struct rocalution_bench_itsolver_impl : public rocalution_bench_itsolver<T>
{
    using driver_traits_t = rocalution_driver_itsolver_traits<ITSOLVER, T>;
    using solver_t        = typename driver_traits_t::solver_t;
    using results_t       = rocalution_bench_solver_results;
    using params_t        = rocalution_bench_solver_parameters;

private:
    //
    // @brief Input parameters.
    //
    const rocalution_bench_solver_parameters* m_params{};

    //
    // @brief Output results.
    //
    results_t* m_results{};

    //
    // @brief Solver.
    //
    solver_t m_solver{};

    //
    // @brief Iterative Driver.
    //
    rocalution_driver_itsolver<ITSOLVER, T> m_driver{};

public:
    rocalution_bench_itsolver_impl(const params_t* parameters, results_t* results)
        : m_params(parameters)
        , m_results(results){};

    virtual ~rocalution_bench_itsolver_impl()
    {
        this->m_solver.Clear();
    };

    //
    // @copydoc rocalution_bench_itsolver<T>::ImportLinearSystem
    //
    virtual bool
        ImportLinearSystem(LocalMatrix<T>& A, LocalVector<T>& B, LocalVector<T>& S) override
    {
        //
        // Import linear system.
        //
        bool success = this->m_driver.ImportLinearSystem(A, B, S, *this->m_params);
        if(!success)
        {
            rocalution_bench_errmsg << "import linear system failed." << std::endl;
            return false;
        }

        //
        // Move to accelerator.
        //
        A.MoveToAccelerator();
        S.MoveToAccelerator();
        B.MoveToAccelerator();

        //
        // Finalize the import.
        //
        success = this->m_driver.PostprocessImportLinearSystem(A, B, S, *this->m_params);
        if(!success)
        {
            rocalution_bench_errmsg << "post process Import linear system failed." << std::endl;
            return false;
        }

        return true;
    }

    //
    // @copydoc rocalution_bench_itsolver<T>::LogBenchResults
    //
    virtual bool LogBenchResults(LocalMatrix<T>& A,
                                 LocalVector<T>& B,
                                 LocalVector<T>& X,
                                 LocalVector<T>& S,
                                 results_t&      results) override
    {
        return rocalution_bench_record_results(*this->m_params, results);
    }

    //
    // @copydoc rocalution_bench_itsolver<T>::AnalyzeLinearSystem
    //

    virtual bool
        AnalyzeLinearSystem(LocalMatrix<T>& A, LocalVector<T>& B, LocalVector<T>& X) override
    {
        //
        // Configure linear solver.
        //
        {
            if(s_verbose)
            {
                std::cout << "AnalyzeLinearSystem ConfigureLinearSolver ..." << std::endl;
                std::cout << "A.M   = " << A.GetM() << std::endl;
                std::cout << "A.N   = " << A.GetN() << std::endl;
                std::cout << "A.NNZ = " << A.GetNnz() << std::endl;
            }
            {
                bool success = this->m_driver.ConfigureLinearSolver(
                    A, B, X, this->m_solver, *this->m_params);
                if(!success)
                {
                    rocalution_bench_errmsg << "configure linear solver failed." << std::endl;
                    return false;
                }
            }
            if(s_verbose)
            {
                std::cout << "AnalyzeLinearSystem ConfigureLinearSolver done." << std::endl;
            }
        }

        //
        // Convert the matrix to the required format.
        //
        {
            if(s_verbose)
            {
                std::cout << "AnalyzeLinearSystem ConvertTo..." << std::endl;
            }

            {
                auto format = this->m_params->Get(params_t::format);
                if(s_verbose)
                {
                    std::cout << "AnalyzeLinearSystem ConvertTo ... format = " << format << " ... "
                              << std::endl;
                }
                const auto blockdim = this->m_params->Get(params_t::blockdim);
                A.ConvertTo(format, format == BCSR ? blockdim : 1);
            }

            if(s_verbose)
            {
                std::cout << "AnalyzeLinearSystem ConvertTo done." << std::endl;
            }
        }

        return true;
    }

    //
    // @copydoc rocalution_bench_itsolver<T>::SolveLinearSystem
    //
    virtual bool SolveLinearSystem(LocalMatrix<T>& A,
                                   LocalVector<T>& B,
                                   LocalVector<T>& X,
                                   results_t&      results) override
    {
        //
        // Pre-solve.
        //
        if(!this->m_driver.PreprocessLinearSolve(A, B, X, this->m_solver, *this->m_params))
        {
            rocalution_bench_errmsg << "preprocess linear solve failed." << std::endl;
            return false;
        }

        //
        // Solve
        //
        this->m_solver.Solve(B, &X);

        //
        // Post solve.
        //
        if(!this->m_driver.PostprocessLinearSolve(A, B, X, this->m_solver, *this->m_params))
        {
            rocalution_bench_errmsg << "postprocess linear solve failed." << std::endl;
            return false;
        }

        //
        // Record results.
        //
        results.Set(results_t::iter, this->m_solver.GetIterationCount());

        results.Set(results_t::norm_residual, this->m_solver.GetCurrentResidual());

        {
            auto status = this->m_solver.GetSolverStatus();
            results.Set(results_t::convergence, (status == 1) ? true : false);
        }

        return true;
    }
};
