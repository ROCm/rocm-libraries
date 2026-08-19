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
#include "rocalution_bench_solver_parameters.hpp"

rocalution_enum_smoother rocalution_bench_solver_parameters::GetEnumSmoother() const
{
    return this->m_enum_smoother;
};
rocalution_enum_coarsening_strategy
    rocalution_bench_solver_parameters::GetEnumCoarseningStrategy() const
{
    return this->m_enum_coarsening_strategy;
};
rocalution_enum_preconditioner rocalution_bench_solver_parameters::GetEnumPreconditioner() const
{
    return this->m_enum_preconditioner;
};
rocalution_enum_itilu0_alg rocalution_bench_solver_parameters::GetEnumItILU0Algorithm() const
{
    return this->m_enum_itilu0_alg;
};
rocalution_enum_itsolver rocalution_bench_solver_parameters::GetEnumIterativeSolver() const
{
    return this->m_enum_itsolver;
};
rocalution_enum_directsolver rocalution_bench_solver_parameters::GetEnumDirectSolver() const
{
    return this->m_enum_directsolver;
};
rocalution_enum_matrix_init rocalution_bench_solver_parameters::GetEnumMatrixInit() const
{
    return this->m_enum_matrix_init;
};

constexpr rocalution_bench_solver_parameters::e_bool
    rocalution_bench_solver_parameters::e_bool_all[];
constexpr rocalution_bench_solver_parameters::e_int rocalution_bench_solver_parameters::e_int_all[];
constexpr rocalution_bench_solver_parameters::e_string
    rocalution_bench_solver_parameters::e_string_all[];
constexpr rocalution_bench_solver_parameters::e_uint
    rocalution_bench_solver_parameters::e_uint_all[];
constexpr rocalution_bench_solver_parameters::e_double
    rocalution_bench_solver_parameters::e_double_all[];

constexpr const char* rocalution_bench_solver_parameters::e_bool_names[];
constexpr const char* rocalution_bench_solver_parameters::e_int_names[];
constexpr const char* rocalution_bench_solver_parameters::e_string_names[];
constexpr const char* rocalution_bench_solver_parameters::e_uint_names[];
constexpr const char* rocalution_bench_solver_parameters::e_double_names[];

bool rocalution_bench_solver_parameters::Get(e_bool v) const
{
    return this->bool_values[v];
}
void rocalution_bench_solver_parameters::Set(e_bool v, bool s)
{
    this->bool_values[v] = s;
}

std::string rocalution_bench_solver_parameters::Get(e_string v) const
{
    return this->string_values[v];
}
void rocalution_bench_solver_parameters::Set(e_string v, const std::string& s)
{
    this->string_values[v] = s;
}

unsigned int rocalution_bench_solver_parameters::Get(e_uint v) const
{
    return this->uint_values[v];
}
void rocalution_bench_solver_parameters::Set(e_uint v, unsigned int s)
{
    this->uint_values[v] = s;
}

int rocalution_bench_solver_parameters::Get(e_int v) const
{
    return this->int_values[v];
}
void rocalution_bench_solver_parameters::Set(e_int v, int s)
{
    this->int_values[v] = s;
}

double rocalution_bench_solver_parameters::Get(e_double v) const
{
    return this->double_values[v];
}
void rocalution_bench_solver_parameters::Set(e_double v, double s)
{
    this->double_values[v] = s;
}

bool* rocalution_bench_solver_parameters::GetPointer(e_bool v)
{
    return &this->bool_values[v];
}
std::string* rocalution_bench_solver_parameters::GetPointer(e_string v)
{
    return &this->string_values[v];
}
unsigned int* rocalution_bench_solver_parameters::GetPointer(e_uint v)
{
    return &this->uint_values[v];
}
int* rocalution_bench_solver_parameters::GetPointer(e_int v)
{
    return &this->int_values[v];
}
double* rocalution_bench_solver_parameters::GetPointer(e_double v)
{
    return &this->double_values[v];
}

void rocalution_bench_solver_parameters::WriteNicely(std::ostream& out) const
{
    out.precision(2);
    out.setf(std::ios::fixed);
    out.setf(std::ios::left);
    out << std::scientific;

    for(auto e : e_bool_all)
    {
        out << std::setw(20) << e_bool_names[e] << " " << std::setw(20) << bool_values[e];
        out << std::endl;
    }
    for(auto e : e_int_all)
    {
        out << std::setw(20) << e_int_names[e] << " " << std::setw(20) << int_values[e];
        out << std::endl;
    }
    for(auto e : e_double_all)
    {
        out << std::setw(20) << e_double_names[e] << " " << std::setw(20) << double_values[e];
        out << std::endl;
    }
    for(auto e : e_uint_all)
    {
        out << std::setw(20) << e_uint_names[e] << " " << std::setw(20) << uint_values[e];
        out << std::endl;
    }
    for(auto e : e_string_all)
    {
        out << std::setw(20) << e_string_names[e] << " " << std::setw(20) << string_values[e];
        out << std::endl;
    }
}

void rocalution_bench_solver_parameters::WriteJson(std::ostream& out) const
{
    bool first = false;
    for(auto e : e_bool_all)
    {
        if(!first)
            first = true;
        else
            out << "," << std::endl;
        out << "\"" << e_bool_names[e] << "\" : "
            << "\"" << bool_values[e] << "\"";
    }
    for(auto e : e_int_all)
    {
        out << "," << std::endl;
        out << "\"" << e_int_names[e] << "\" : "
            << "\"" << int_values[e] << "\"";
    }
    for(auto e : e_double_all)
    {
        out << "," << std::endl;
        out << "\"" << e_double_names[e] << "\" : "
            << "\"" << double_values[e] << "\"";
    }
    for(auto e : e_uint_all)
    {
        out << "," << std::endl;
        out << "\"" << e_uint_names[e] << "\" : "
            << "\"" << uint_values[e] << "\"";
    }
    for(auto e : e_string_all)
    {
        out << "," << std::endl;
        out << "\"" << e_string_names[e] << "\" : "
            << "\"" << string_values[e] << "\"";
    }
}
