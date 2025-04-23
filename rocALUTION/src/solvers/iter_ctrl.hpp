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

#ifndef ROCALUTION_ITER_CTRL_HPP_
#define ROCALUTION_ITER_CTRL_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace rocalution
{

    // Iteration control for iterative solvers, monitor the
    // residual (L2 norm) behavior
    /** \private */
    class IterationControl
    {
    public:
        IterationControl();
        ~IterationControl();

        // Initialize with absolute/relative/divergence
        // tolerance and maximum number of iterations
        void Init(double abs, double rel, double div, int max);

        // Initialize with absolute/relative/divergence
        // tolerance and minimum/maximum number of iterations
        void Init(double abs, double rel, double div, int min, int max);

        // Initialize with absolute/relative/divergence tolerance
        void InitTolerance(double abs, double rel, double div);

        // Set the minimum number of iterations
        void InitMinimumIterations(int min);

        // Set the maximal number of iterations
        void InitMaximumIterations(int max);

        // Get the minimum number of iterations
        int GetMinimumIterations(void) const;

        // Get the maximal number of iterations
        int GetMaximumIterations(void) const;

        // Initialize the initial residual
        bool InitResidual(double res);

        // Clear (reset)
        void Clear(void);

        // Check the residual (this count also the number of iterations)
        bool CheckResidual(double res);

        // Check the residual and index value for the inf norm
        // (this count also the number of iterations)
        bool CheckResidual(double res, int64_t index);

        // Check the residual (without counting the number of iterations)
        bool CheckResidualNoCount(double res);

        // Check for maximum iterations (without counting the number of iterations)
        bool CheckMaximumIterNoCount(void);

        // Record the history of the residual
        void RecordHistory(void);

        // Write the history of the residual to an ASCII file
        void WriteHistoryToFile(const std::string& filename) const;

        // Provide verbose output of the solver (iter, residual)
        void Verbose(int verb = 1);

        // Print the initialized information of the iteration control
        void PrintInit(void) const;

        // Print the current status (is the any criteria reached or not)
        void PrintStatus(void);

        // Return the iteration count
        int GetIterationCount(void) const;

        // Return the current residual
        double GetCurrentResidual(void) const;

        // Return the current status
        int GetSolverStatus(void) const;

        // Return absolute maximum index of residual vector when using Linf norm
        int64_t GetAmaxResidualIndex(void) const;

    private:
        // Verbose flag
        // verb == 0 no output
        // verb == 1 print info about the solver (start,end);
        // verb == 2 print (iter, residual) via iteration control;
        int verb_;

        // Iteration count
        int iteration_;

        // Flag == true after calling InitResidual()
        bool init_res_;

        // Initial residual
        double initial_residual_;

        // Absolute tolerance
        double absolute_tol_;
        // Relative tolerance
        double relative_tol_;
        // Divergence tolerance
        double divergence_tol_;
        // Minimum number of iteration
        int minimum_iter_;
        // Maximum number of iteration
        int maximum_iter_;

        // Indicator for the reached criteria:
        // 0 - not yet;
        // 1 - abs tol is reached;
        // 2 - rel tol is reached;
        // 3 - div tol is reached;
        // 4 - max iter is reached
        int reached_;

        // STL vector keeping the residual history
        std::vector<double> residual_history_;

        // Current residual (obtained via CheckResidual())
        double current_res_;

        // Current residual, index for the inf norm (obtained via CheckResidual())
        int64_t current_index_;

        // Flag == true then the residual is recorded in the residual_history_ vector
        bool rec_;
    };

} // namespace rocalution

#endif // ROCALUTION_ITER_CTRL_HPP_
