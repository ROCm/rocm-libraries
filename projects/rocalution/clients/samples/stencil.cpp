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

#include <cstdlib>
#include <iostream>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[])
{

    // Initialize rocALUTION
    init_rocalution();

    // Print rocALUTION info
    info_rocalution();

    // rocALUTION objects
    LocalVector<double>  x;
    LocalVector<double>  rhs;
    LocalVector<double>  e;
    LocalStencil<double> stencil(Laplace2D);

    // Set up stencil grid
    stencil.SetGrid(100); // 100x100

    // Allocate vectors
    x.Allocate("x", stencil.GetN());
    rhs.Allocate("rhs", stencil.GetM());
    e.Allocate("e", stencil.GetN());

    // Linear Solver
    CG<LocalStencil<double>, LocalVector<double>, double> ls;

    // Initialize rhs such that A 1 = rhs
    e.Ones();
    stencil.Apply(e, &rhs);

    // Initial zero guess
    x.Zeros();

    // Set solver operator
    ls.SetOperator(stencil);

    // Build solver
    ls.Build();

    // Print stencil info
    stencil.Info();

    // Start time measurement
    double tick, tack;
    tick = rocalution_time();

    // Solve A x = rhs
    ls.Solve(rhs, &x);

    // Stop time measurement
    tack = rocalution_time();
    std::cout << "Solver execution:" << (tack - tick) / 1e6 << " sec" << std::endl;

    // Clear solver
    ls.Clear();

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    double error = e.Norm();
    std::cout << "||e - x||_2 = " << error << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();

    return 0;
}
