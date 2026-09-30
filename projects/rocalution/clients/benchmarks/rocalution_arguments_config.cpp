/*! \file */
/* ************************************************************************
* Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocalution_arguments_config.hpp"
#include "rocalution_bench_solver.hpp"

//
//
//
rocalution_arguments_config::rocalution_arguments_config() {}

std::string make_option(const char* name)
{
    std::string s(name);
    for(size_t i = 0; i < s.length(); ++i)
    {
        if(s[i] == '_')
            s[i] = '-';
    }
    return s;
}

//
//
//
void rocalution_arguments_config::set_description(options_description& desc)
{
    desc.add_options()("help,h", "produces this help message");

#define ADD_OPTION(type_, e_, x_default_, x_desc_)                                    \
    desc.add_options()(make_option(rocalution_bench_solver_parameters::GetName(e_)),  \
                       value<type_>(this->GetPointer(e_))->default_value(x_default_), \
                       x_desc_)

    for(auto e : rocalution_bench_solver_parameters::e_double_all)
    {
        switch(e)
        {
        case rocalution_bench_solver_parameters::abs_tol:
        {
            ADD_OPTION(double, e, 1.0e-6, "absolute tolerance");
            break;
        }
        case rocalution_bench_solver_parameters::rel_tol:
        {
            ADD_OPTION(double, e, 0.0, "relative tolerance");
            break;
        }
        case rocalution_bench_solver_parameters::div_tol:
        {
            ADD_OPTION(double, e, 1e+8, "divide tolerance");
            break;
        }
        case rocalution_bench_solver_parameters::residual_tol:
        {
            ADD_OPTION(double, e, 1.0e-8, "residual tolerance");
            break;
        }
        case rocalution_bench_solver_parameters::ilut_tol:
        {
            ADD_OPTION(double, e, 0.05, "ilut tolerance");
            break;
        }
        case rocalution_bench_solver_parameters::itsolve_tol:
        {
            ADD_OPTION(double, e, 1e-03, "iterative solve tolerance (see --iterative_solve).");
            break;
        }
        case rocalution_bench_solver_parameters::itilu0_tol:
        {
            ADD_OPTION(double, e, 2e-07, "itilu0 tolerance");
            break;
        }
        case rocalution_bench_solver_parameters::mcgs_relax:
        {
            ADD_OPTION(double, e, 1.0, "relaxation coefficient");
            break;
        }
        case rocalution_bench_solver_parameters::solver_over_interp:
        {
            ADD_OPTION(double, e, 1.2, "over interp coefficient for multigrid");
            break;
        }
        case rocalution_bench_solver_parameters::solver_coupling_strength:
        {
            ADD_OPTION(double, e, 0.005, "coupling strength coefficient for multigrid");
            break;
        }
        }
    }

    for(auto e : rocalution_bench_solver_parameters::e_int_all)
    {
        switch(e)
        {
        case rocalution_bench_solver_parameters::krylov_basis:
        {
            ADD_OPTION(int, e, 30, "dimension of the Krylov basis.");
            break;
        }
        case rocalution_bench_solver_parameters::max_iter:
        {
            ADD_OPTION(int, e, 2000, "default maximum number of iterations.");
            break;
        }
        case rocalution_bench_solver_parameters::ndim:
        {
            ADD_OPTION(int, e, 200, "dimension");
            break;
        }

        case rocalution_bench_solver_parameters::solver_pre_smooth:
        {
            ADD_OPTION(int, e, 2, "number of iteration of pre-smoother");
            break;
        }
        case rocalution_bench_solver_parameters::solver_post_smooth:
        {
            ADD_OPTION(int, e, 2, "number of iteration of post-smoother");
            break;
        }

        case rocalution_bench_solver_parameters::solver_ordering:
        {
            ADD_OPTION(int, e, 1, "ordering type");
            break;
        }

        case rocalution_bench_solver_parameters::rebuild_numeric:
        {
            ADD_OPTION(int, e, 0, "rebuild numeric");
            break;
        }

        case rocalution_bench_solver_parameters::cycle:
        {
            ADD_OPTION(int, e, 0, "Number of cycle");
            break;
        }

        case rocalution_bench_solver_parameters::ilut_n:
        {
            ADD_OPTION(int, e, 4, "number of elements per row kept");
            break;
        }

        case rocalution_bench_solver_parameters::itsolve_max_iter:
        {
            ADD_OPTION(int,
                       e,
                       30,
                       "iterative solve maximum number of iterations (see "
                       "--iterative_solve).");
            break;
        }
        case rocalution_bench_solver_parameters::itilu0_max_iter:
        {
            ADD_OPTION(int, e, 1000, "itilu0 maximum number of iterations.");
            break;
        }

        case rocalution_bench_solver_parameters::itilu0_options:
        {
            ADD_OPTION(int, e, 1000, "itilu0 combination of options.");
            break;
        }

        case rocalution_bench_solver_parameters::mcilu_p:
        {
            ADD_OPTION(int, e, 0, "multicolored ilu parameter p.");
            break;
        }

        case rocalution_bench_solver_parameters::mcilu_q:
        {
            ADD_OPTION(int, e, 0, "multicolored ilu parameter q.");
            break;
        }

        case rocalution_bench_solver_parameters::solver_coarsest_level:
        {
            ADD_OPTION(int, e, 200, "multigrid coarsest_level.");
            break;
        }

        case rocalution_bench_solver_parameters::blockdim:
        {
            ADD_OPTION(int, e, 3, "block dimension.");
            break;
        }
        }
    }

    for(auto e : rocalution_bench_solver_parameters::e_string_all)
    {
        switch(e)
        {
        case rocalution_bench_solver_parameters::matrix_filename:
        {
            ADD_OPTION(std::string,
                       e,
                       "",
                       "read from matrix "
                       "market (.mtx) format. This will override parameters -m, -n, and -z.");
            break;
        }
        case rocalution_bench_solver_parameters::smoother:
        {
            ADD_OPTION(std::string, e, "", "solver smoother");
            break;
        }
        case rocalution_bench_solver_parameters::iterative_solver:
        {
            ADD_OPTION(std::string, e, "", "iterative solver");
            break;
        }
        case rocalution_bench_solver_parameters::direct_solver:
        {
            ADD_OPTION(std::string, e, "", "direct solver");
            break;
        }
        case rocalution_bench_solver_parameters::preconditioner:
        {
            ADD_OPTION(std::string, e, "", "preconditioner");
            break;
        }
        case rocalution_bench_solver_parameters::coarsening_strategy:
        {
            ADD_OPTION(std::string, e, "", "coarsening strategy");
            break;
        }
        case rocalution_bench_solver_parameters::itilu0_alg:
        {
            ADD_OPTION(std::string, e, "Default", "ItILU0 algorithm");
            break;
        }
        case rocalution_bench_solver_parameters::matrix:
        {
            ADD_OPTION(std::string, e, "", "matrix initialization");
            break;
        }
        }
    }

    for(auto e : rocalution_bench_solver_parameters::e_uint_all)
    {
        switch(e)
        {
        case rocalution_bench_solver_parameters::format:
        {
            ADD_OPTION(unsigned int, e, CSR, "matrix format");
            break;
        }
        }
    }

    for(auto e : rocalution_bench_solver_parameters::e_bool_all)
    {
        switch(e)
        {
        case rocalution_bench_solver_parameters::iterative_solve:
        {
            ADD_OPTION(bool, e, false, "perform triangular iterative solve during solving step.");
            break;
        }
        case rocalution_bench_solver_parameters::verbose:
        {
            ADD_OPTION(bool, e, false, "verbose");
            break;
        }
        case rocalution_bench_solver_parameters::mcilu_use_level:
        {
            ADD_OPTION(bool, e, false, "use level in mcilu");
            break;
        }
        }
    }
}

//
//
//
int rocalution_arguments_config::parse(int& argc, char**& argv, options_description& desc)
{
    variables_map vm;
    store(parse_command_line(argc, argv, desc, sizeof(rocalution_arguments_config)), vm);
    notify(vm);
    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }
    //
    // Init enums.
    //
    for(auto e : rocalution_bench_solver_parameters::e_string_all)
    {
        switch(e)
        {
        case rocalution_bench_solver_parameters::matrix_filename:
        {
            break;
        }
        case rocalution_bench_solver_parameters::smoother:
        {

            auto smoother_string = this->Get(rocalution_bench_solver_parameters::smoother);
            if(smoother_string != "")
            {
                this->m_enum_smoother(smoother_string.c_str());
            }

            break;
        }
        case rocalution_bench_solver_parameters::iterative_solver:
        {

            auto iterative_solver_string
                = this->Get(rocalution_bench_solver_parameters::iterative_solver);
            if(iterative_solver_string != "")
            {
                this->m_enum_itsolver(iterative_solver_string.c_str());
            }

            break;
        }
        case rocalution_bench_solver_parameters::direct_solver:
        {
            auto direct_solver_string
                = this->Get(rocalution_bench_solver_parameters::direct_solver);
            if(direct_solver_string != "")
            {
                this->m_enum_directsolver(direct_solver_string.c_str());
            }
            break;
        }
        case rocalution_bench_solver_parameters::preconditioner:
        {
            auto preconditioner_string
                = this->Get(rocalution_bench_solver_parameters::preconditioner);
            if(preconditioner_string != "")
            {
                this->m_enum_preconditioner(preconditioner_string.c_str());
            }
            break;
        }
        case rocalution_bench_solver_parameters::coarsening_strategy:
        {
            auto coarsening_strategy_string
                = this->Get(rocalution_bench_solver_parameters::coarsening_strategy);
            if(coarsening_strategy_string != "")
            {
                this->m_enum_coarsening_strategy(coarsening_strategy_string.c_str());
            }

            break;
        }
        case rocalution_bench_solver_parameters::itilu0_alg:
        {

            auto itilu0_alg_string = this->Get(rocalution_bench_solver_parameters::itilu0_alg);
            if(itilu0_alg_string != "")
            {
                this->m_enum_itilu0_alg(itilu0_alg_string.c_str());
            }

            break;
        }

        case rocalution_bench_solver_parameters::matrix:
        {
            auto matrix_init_string = this->Get(rocalution_bench_solver_parameters::matrix);
            if(matrix_init_string != "")
            {
                this->m_enum_matrix_init(matrix_init_string.c_str());
            }
            break;
        }
        }
    }

    return 0;
}

//
//
//
int rocalution_arguments_config::parse_no_default(int&                 argc,
                                                  char**&              argv,
                                                  options_description& desc)
{
    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);
    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

    return 0;
}
