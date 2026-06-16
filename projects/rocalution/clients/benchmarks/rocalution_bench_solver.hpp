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

#include "rocalution_bench_itsolver_impl.hpp"

//
// @brief Core structure responsible of configuring the benchmark from parsing command line arguments
// and execute it.
//
template <typename T>
struct rocalution_bench_solver
{

private:
    rocalution_bench_itsolver<T>*             m_itsolver{};
    rocalution_bench_solver_results           m_output_results{};
    const rocalution_bench_solver_parameters* m_input_parameters{};

public:
    ~rocalution_bench_solver()
    {
        if(this->m_itsolver)
        {
            delete this->m_itsolver;
            this->m_itsolver = nullptr;
        }
    }

    rocalution_bench_solver(const rocalution_bench_solver_parameters* config)
        : m_input_parameters(config)
    {
        //
        // Try to config iterative solver.
        //
        const rocalution_enum_itsolver enum_itsolver
            = this->m_input_parameters->GetEnumIterativeSolver();
        if(enum_itsolver.is_invalid())
        {
            rocalution_bench_errmsg << "invalid iterative solver." << std::endl;
            throw false;
        }

        this->m_itsolver = nullptr;
        switch(enum_itsolver.value)
        {

            //
            // Define cases.
            //
#define ROCALUTION_ENUM_ITSOLVER_TRANSFORM(x_)                                                  \
    case rocalution_enum_itsolver::x_:                                                          \
    {                                                                                           \
        this->m_itsolver = new rocalution_bench_itsolver_impl<rocalution_enum_itsolver::x_, T>( \
            this->m_input_parameters, &this->m_output_results);                                 \
        break;                                                                                  \
    }

            //
            // Generate cases.
            //
            ROCALUTION_ENUM_ITSOLVER_TRANSFORM_EACH;

            //
            // Undefine cases.
            //
#undef ROCALUTION_ENUM_ITSOLVER_TRANSFORM
        }

        if(this->m_itsolver == nullptr)
        {
            rocalution_bench_errmsg << "iterative solver instantiation failed." << std::endl;
            throw false;
        }
    }

    bool Run()
    {
        if(this->m_itsolver != nullptr)
        {
            return this->m_itsolver->Run(this->m_output_results);
        }
        return false;
    }
};
