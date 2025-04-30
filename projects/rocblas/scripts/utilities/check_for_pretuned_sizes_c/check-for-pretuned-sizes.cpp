/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include <rocblas/rocblas.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// rocBLAS clients contains a substitute for Boost's program_options
#include "../../../clients/benchmarks/program_options.hpp"
using namespace roc;

void init_scalar_value(rocblas_union_t* scalar, rocblas_datatype type, double value)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:
        scalar->h = rocblas_half(value);
        break;
    case rocblas_datatype_f32_r:
        scalar->s = float(value);
        break;
    case rocblas_datatype_f64_r:
        scalar->d = double(value);
        break;
    case rocblas_datatype_i32_r:
        scalar->i = int32_t(value);
        break;
    case rocblas_datatype_f32_c:
        scalar->c = rocblas_float_complex(value);
        break;
    case rocblas_datatype_f64_c:
        scalar->z = rocblas_double_complex(value);
        break;
    default:
        throw std::runtime_error("Unhandled type in init_scalar_value");
    }
}

std::vector<char*> separateIntoTokens(std::string line, char delim = ' ')
{
    std::vector<char*> rv;
    std::string        token;
    for(auto c : line)
    {
        if(c != delim)
            token.push_back(c);
        else if(token.size() > 0)
        {
            rv.push_back(strdup(token.c_str()));
            token.clear();
        }
    }

    if(token.size() > 0)
        rv.push_back(strdup(token.c_str()));

    return rv;
}

//Copied from rocblas_datatype2string.hpp
// clang-format off
inline rocblas_datatype string2rocblas_datatype(const std::string& value)
{
    return
        value == "f16_r" || value == "h" ? rocblas_datatype_f16_r  :
        value == "f32_r" || value == "s" ? rocblas_datatype_f32_r  :
        value == "f64_r" || value == "d" ? rocblas_datatype_f64_r  :
        value == "bf16_r"                ? rocblas_datatype_bf16_r :
        value == "f16_c"                 ? rocblas_datatype_f16_c  :
        value == "f32_c" || value == "c" ? rocblas_datatype_f32_c  :
        value == "f64_c" || value == "z" ? rocblas_datatype_f64_c  :
        value == "bf16_c"                ? rocblas_datatype_bf16_c :
        value == "i8_r"                  ? rocblas_datatype_i8_r   :
        value == "i32_r"                 ? rocblas_datatype_i32_r  :
        value == "i8_c"                  ? rocblas_datatype_i8_c   :
        value == "i32_c"                 ? rocblas_datatype_i32_c  :
        value == "u8_r"                  ? rocblas_datatype_u8_r   :
        value == "u32_r"                 ? rocblas_datatype_u32_r  :
        value == "u8_c"                  ? rocblas_datatype_u8_c   :
        value == "u32_c"                 ? rocblas_datatype_u32_c  :
        static_cast<rocblas_datatype>(-1);
}
// clang-format on

struct Arguments
{
    rocblas_int M;
    rocblas_int N;
    rocblas_int K;

    rocblas_datatype a_type;
    rocblas_datatype b_type;
    rocblas_datatype c_type;
    rocblas_datatype d_type;
    rocblas_datatype compute_type;

    double alpha;
    double beta;

    rocblas_operation transA;
    rocblas_operation transB;

    rocblas_int batch_count;

    char function[64];
};

Arguments parseArguments(char** args, int argCount)
{
    Arguments   arg;
    std::string function;
    std::string precision;
    std::string a_type;
    std::string b_type;
    std::string c_type;
    std::string d_type;
    std::string compute_type;

    char transA;
    char transB;

    options_description desc("rocblas-bench command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m",
         value<rocblas_int>(&arg.M)->default_value(128),
         "Specific matrix size: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
         "rows or columns in matrix.")

        ("sizen,n",
         value<rocblas_int>(&arg.N)->default_value(128),
         "Specific matrix/vector size: BLAS-1: the length of the vector. BLAS-2 & "
         "BLAS-3: the number of rows or columns in matrix")

        ("sizek,k",
         value<rocblas_int>(&arg.K)->default_value(128),
         "Specific matrix size: BLAS-2: the number of sub or super-diagonals of A. BLAS-3: "
         "the number of columns in A and rows in B.")

        ("alpha",
          value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
         value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("function,f",
         value<std::string>(&function),
         "BLAS function to test.")

        ("precision,r",
         value<std::string>(&precision)->default_value("f32_r"), "Precision. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("a_type",
         value<std::string>(&a_type), "Precision of matrix A. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("b_type",
         value<std::string>(&b_type), "Precision of matrix B. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("c_type",
         value<std::string>(&c_type), "Precision of matrix C. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("d_type",
         value<std::string>(&d_type), "Precision of matrix D. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("compute_type",
         value<std::string>(&compute_type), "Precision of computation. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("transposeA",
         value<char>(&transA)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
         value<char>(&transB)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        //Not currently needed, but may be in the future
        ("batch_count",
         value<rocblas_int>(&arg.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched and strided_batched routines");

    variables_map vm;
    store(parse_command_line(argCount, args, desc, true), vm);
    notify(vm);

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string2rocblas_datatype(precision);
    if(prec == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --precision " + precision);

    arg.a_type = a_type == "" ? prec : string2rocblas_datatype(a_type);
    if(arg.a_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --a_type " + a_type);

    arg.b_type = b_type == "" ? prec : string2rocblas_datatype(b_type);
    if(arg.b_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --b_type " + b_type);

    arg.c_type = c_type == "" ? prec : string2rocblas_datatype(c_type);
    if(arg.c_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --c_type " + c_type);

    arg.d_type = d_type == "" ? prec : string2rocblas_datatype(d_type);
    if(arg.d_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --d_type " + d_type);

    arg.compute_type = compute_type == "" ? prec : string2rocblas_datatype(compute_type);
    if(arg.compute_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --compute_type " + compute_type);

    snprintf(arg.function, sizeof(arg.function), "%s", function.c_str());

    arg.transA = transA == 'N' ? rocblas_operation_none : rocblas_operation_transpose;
    arg.transB = transB == 'N' ? rocblas_operation_none : rocblas_operation_transpose;

    return arg;
}

int main(int argc, char* argv[])
{
    if(argc == 1 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0){
        std::cout << "Parses rocblas-bench log file and determines which gemm problem sizes have been pretuned for in Tensile" << std::endl;
        std::cout << "Usage:\n\tcheck-for-pretuned-sizes <rocblas-bench logfile>" << std::endl;
        return 1;
    }

    rocblas_initialize();
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    std::fstream f{argv[1]};      //Load log file

    char buffer [2048];
    while(f.getline(buffer, 2048)){
        auto tokens = separateIntoTokens(std::string(buffer));
        if(tokens.size() && strstr(tokens[0], "rocblas-bench")){
            auto arg = parseArguments(tokens.data(), tokens.size());
            if(strstr(arg.function, "gemm")){
                rocblas_union_t alpha, beta;
                init_scalar_value(&alpha, arg.compute_type, 1.0);
                init_scalar_value(&beta, arg.compute_type, 1.0);

                // The solution fitness query is initialized to std::numeric_limits<double>::lowest()
                double fitness;
                rocblas_set_solution_fitness_query(handle, &fitness);
                std::cout << "m - " << arg.M << " n - " << arg.N << " k - " << arg.K << std::endl;
                rocblas_gemm_ex(
                    handle,
                    arg.transA,
                    arg.transB,
                    arg.M,
                    arg.N,
                    arg.K,
                    &alpha,
                    nullptr, // A
                    arg.a_type,
                    1, // lda
                    nullptr, // B
                    arg.b_type,
                    1, // ldb
                    &beta,
                    nullptr, // C
                    arg.c_type,
                    1, // ldc
                    nullptr, // D
                    arg.d_type,
                    1, // ldd
                    arg.compute_type,
                    rocblas_gemm_algo_standard,
                    0, // solution_index
                    0 // flags
                );

               if(!fitness)
               {
                   std::cout << "\033[0;32m[Pretuned]\033[0m " << buffer << std::endl;
               }
               else
               {
                   std::cout << "\033[0;31m[Not pretuned]\033[0m " << buffer << std::endl;
               }

               // We reset the solution fitness query to nullptr to avoid a dangling pointer
               rocblas_set_solution_fitness_query(handle, nullptr);
            }
        }

        for (auto* t: tokens)
            free(t);
    }

    f.close();
}
