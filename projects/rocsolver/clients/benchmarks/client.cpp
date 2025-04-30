/* **************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <fmt/core.h>
#include <fmt/ostream.h>

#include "common/misc/program_options.hpp"
#include "common/misc/rocsolver_dispatcher.hpp"

using namespace roc;

// clang-format off
const char* help_str = R"HELP_STR(
rocSOLVER benchmark client help.

Usage: ./rocsolver-bench <options>

In addition to some common general options, the following list of options corresponds to all the parameters
that might be needed to test a given rocSOLVER function. The parameters are named as in the API user guide.
The arrays are initialized internally by the program with random values.

Note: When a required parameter/option is not provided, it will take the default value as listed below.
If no default value is defined, the program will try to calculate a suitable value depending on the context
of the problem and the tested function; if this is not possible, the program will abort with an error.
Functions that accept multiple size parameters can generally be provided a single size parameter (typically,
m) and a square-size matrix will be assumed.

Example: ./rocsolver-bench -f getf2_batched -m 30 --lda 75 --batch_count 350
This will test getf2_batched with a set of 350 random 30x30 matrices. strideP will be set to be equal to 30.

Options:
)HELP_STR";
// clang-format on

static std::string rocblas_version()
{
    size_t size;
    rocblas_get_version_string_size(&size);
    std::string str(size - 1, '\0');
    rocblas_get_version_string(str.data(), size);
    return str;
}

static std::string rocsolver_version()
{
    size_t size;
    rocsolver_get_version_string_size(&size);
    std::string str(size - 1, '\0');
    rocsolver_get_version_string(str.data(), size);
    return str;
}

static void print_version_info()
{
    fmt::print("rocSOLVER version {} (with rocBLAS {})\n", rocsolver_version(), rocblas_version());
    std::fflush(stdout);
}

int main(int argc, char* argv[])
try
{
    Arguments argus;

    // disable unit_check in client benchmark, it is only
    // used in gtest unit test
    argus.unit_check = 0;

    // enable timing check,otherwise no performance data collected
    argus.timing = 1;

    std::string function;
    char precision = 's';
    rocblas_int device_id = 0;

    // take arguments and set default values
    // clang-format off
    options_description desc("rocsolver client command line options");
    desc.add_options()("help,h", "Produces this help message.")

        // test options
        ("batch_count",
         value<rocblas_int>(&argus.batch_count)->default_value(1),
            "Number of matrices or problem instances in the batch.\n"
            "                           Only applicable to batch routines.\n"
            "                           ")

        ("device",
         value<rocblas_int>(&device_id)->default_value(0),
            "Set the default device to be used for subsequent program runs.\n"
            "                           ")

        ("function,f",
         value<std::string>(&function)->default_value("potf2"),
            "The LAPACK function to test.\n"
            "                           Options are: getf2, getrf, gesvd_batched, etc.\n"
            "                           ")

        ("iters,i",
         value<rocblas_int>(&argus.iters)->default_value(10),
            "Iterations to run inside the GPU timing loop.\n"
            "                           Reported time will be the average.\n"
            "                           ")

        ("alg_mode",
         value<rocblas_int>(&argus.alg_mode)->default_value(0),
            "0 = GPU-only, 1 = Hybrid\n"
            "                           This will change how the algorithm operates.\n"
            "                           Only applicable to functions with hybrid support.\n"
            "                           ")

        ("mem_query",
         value<rocblas_int>(&argus.mem_query)->default_value(0),
            "Calculate the required amount of device workspace memory? 0 = No, 1 = Yes.\n"
            "                           This forces the client to print only the amount of device memory required by\n"
            "                           the function, in bytes.\n"
            "                           ")

        ("perf",
         value<rocblas_int>(&argus.perf)->default_value(0),
            "Ignore CPU timing results? 0 = No, 1 = Yes.\n"
            "                           This forces the client to print only the GPU time and the error if requested.\n"
            "                           ")

        ("precision,r",
         value<char>(&precision)->default_value('s'),
            "Precision to be used in the tests.\n"
            "                           Options are: s, d, c, z.\n"
            "                           ")

        ("profile",
         value<rocblas_int>(&argus.profile)->default_value(0),
            "Print profile logging results for the tested function.\n"
            "                           The argument specifies the max depth of the nested output.\n"
            "                           If the argument is unset or <= 0, profile logging is disabled.\n"
            "                           ")

        ("profile_kernels",
         value<rocblas_int>(&argus.profile_kernels)->default_value(0),
            "Include kernels in profile logging results? 0 = No, 1 = Yes.\n"
            "                           Used in conjunction with --profile to include kernels in the profile log.\n"
            "                           ")

        ("singular",
         value<rocblas_int>(&argus.singular)->default_value(0),
            "Test with degenerate matrices? 0 = No, 1 = Yes\n"
            "                           This will produce matrices that are singular, non positive-definite, etc.\n"
            "                           ")

        ("verify,v",
         value<rocblas_int>(&argus.norm_check)->default_value(0),
            "Validate GPU results with CPU? 0 = No, 1 = Yes.\n"
            "                           This will additionally print the relative error of the computations.\n"
            "                           ")

        ("hash",
         value<rocblas_int>(&argus.hash_check)->default_value(0),
            "Print hash of GPU results? 0 = No, 1 = Yes.\n"
            "                           Meant for checking reproducibility of computations.\n"
            "                           ")

        // size options
        ("k",
         value<rocblas_int>(),
            "Matrix/vector size parameter.\n"
            "                           Represents a sub-dimension of a problem.\n"
            "                           For example, the number of Householder reflections in a transformation.\n"
            "                           ")

        ("m",
         value<rocblas_int>(),
            "Matrix/vector size parameter.\n"
            "                           Typically, the number of rows of a matrix.\n"
            "                           ")

        ("n",
         value<rocblas_int>(),
            "Matrix/vector size parameter.\n"
            "                           Typically, the number of columns of a matrix,\n"
            "                           or the order of a system or transformation.\n"
            "                           ")

        ("nrhs",
         value<rocblas_int>(),
            "Matrix/vector size parameter.\n"
            "                           Typically, the number of columns of a matrix on the right-hand side of a problem.\n"
            "                           ")

        // increment options
        ("inca",
         value<rocblas_int>()->default_value(1),
            "Matrix/vector increment parameter.\n"
            "                           Increment between values in matrices A.\n"
            "                           ")

        ("incb",
         value<rocblas_int>()->default_value(1),
            "Matrix/vector increment parameter.\n"
            "                           Increment between values in matrices B.\n"
            "                           ")

        ("incc",
         value<rocblas_int>()->default_value(1),
            "Matrix/vector increment parameter.\n"
            "                           Increment between values in matrices C.\n"
            "                           ")

        ("incx",
         value<rocblas_int>()->default_value(1),
            "Matrix/vector increment parameter.\n"
            "                           Increment between values in matrices/vectors X.\n"
            "                           ")

        // leading dimension options
        ("lda",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices A.\n"
            "                           ")

        ("ldb",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices B.\n"
            "                           ")

        ("ldc",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices C.\n"
            "                           ")

        ("ldt",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices T.\n"
            "                           ")

        ("ldu",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices U.\n"
            "                           ")

        ("ldv",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices V.\n"
            "                           ")

        ("ldw",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices W.\n"
            "                           ")

        ("ldx",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices X.\n"
            "                           ")

        ("ldy",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices Y.\n"
            "                           ")

        ("ldz",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices Z.\n"
            "                           ")

        // stride options
        ("strideA",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors A.\n"
            "                           ")

        ("strideB",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors B.\n"
            "                           ")

        ("strideD",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors D.\n"
            "                           ")

        ("strideE",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors E.\n"
            "                           ")

        ("strideF",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for vectors ifail.\n"
            "                           ")


        ("strideQ",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for vectors tauq.\n"
            "                           ")

        ("strideP",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for vectors tau, taup, and ipiv.\n"
            "                           ")

        ("strideS",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors S.\n"
            "                           ")

        ("strideU",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors U.\n"
            "                           ")

        ("strideV",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors V.\n"
            "                           ")

        ("strideW",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors W.\n"
            "                           ")

        ("strideX",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors X.\n"
            "                           ")

        ("strideZ",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors Z.\n"
            "                           ")

        // refactorization options
        ("nnzM",
         value<rocblas_int>(),
            "The number of non-zero elements in sparse matrix M.\n"
            "                           Currently only a few test cases can be generated.\n"
            "                           The benchmark client will use the available case closest to the input value.\n"
            "                           ")

        ("nnzA",
         value<rocblas_int>(),
            "The number of non-zero elements in sparse matrix A.\n"
            "                           Currently only a few test cases can be generated.\n"
            "                           The benchmark client will use the available case closest to the input value.\n"
            "                           ")

        ("nnzL",
         value<rocblas_int>(),
            "The number of non-zero elements in sparse matrix L.\n"
            "                           Currently only a few test cases can be generated.\n"
            "                           The benchmark client will use the available case closest to the input value.\n"
            "                           ")

        ("nnzU",
         value<rocblas_int>(),
            "The number of non-zero elements in sparse matrix U.\n"
            "                           Currently only a few test cases can be generated.\n"
            "                           The benchmark client will use the available case closest to the input value.\n"
            "                           ")

        ("nnzT",
         value<rocblas_int>(),
            "The number of non-zero elements in sparse matrix T.\n"
            "                           Currently only a few test cases can be generated.\n"
            "                           The benchmark client will use the available case closest to the input value.\n"
            "                           ")

        ("rfinfo_mode",
         value<char>(),
            "Specifies the desired re-factorization algorithm.\n"
            "                           1 = LU, 2 = Cholesky.\n"
            "                           ")

        // bdsqr options
        ("nc",
         value<rocblas_int>()->default_value(0),
            "The number of columns of matrix C.\n"
            "                           Only applicable to bdsqr.\n"
            "                           ")

        ("nu",
         value<rocblas_int>(),
            "The number of columns of matrix U.\n"
            "                           Only applicable to bdsqr.\n"
            "                           ")

        ("nv",
         value<rocblas_int>()->default_value(0),
            "The number of columns of matrix V.\n"
            "                           Only applicable to bdsqr.\n"
            "                           ")

        // bdsvdx options
        ("svect",
         value<char>()->default_value('N'),
            "N = none, S or V = the singular vectors are computed.\n"
            "                           Indicates how the left singular vectors are to be calculated and stored.\n"
            "                           Only applicable to bdsvdx.\n"
            "                           ")

        // laswp options
        ("k1",
         value<rocblas_int>(),
            "First index for row interchange.\n"
            "                           Only applicable to laswp.\n"
            "                           ")

        ("k2",
         value<rocblas_int>(),
            "Last index for row interchange.\n"
            "                           Only applicable to laswp.\n"
            "                           ")

        // gesvd options
        ("left_svect",
         value<char>()->default_value('N'),
            "N = none, A = the entire orthogonal matrix is computed,\n"
            "                           S or V = the singular vectors are computed,\n"
            "                           O = the singular vectors overwrite the original matrix.\n"
            "                           Indicates how the left singular vectors are to be calculated and stored.\n"
            "                           ")

        ("right_svect",
         value<char>()->default_value('N'),
            "N = none, A = the entire orthogonal matrix is computed,\n"
            "                           S or V = the singular vectors are computed,\n"
            "                           O = the singular vectors overwrite the original matrix.\n"
            "                           Indicates how the right singular vectors are to be calculated and stored.\n"
            "                           ")

        // stein options
         ("nev",
         value<rocblas_int>(),
            "Number of eigenvectors to compute in a partial decomposition.\n"
            "                           Only applicable to stein.\n"
            "                           ")

        // trtri options
        ("diag",
         value<char>()->default_value('N'),
            "N = non-unit triangular, U = unit triangular.\n"
            "                           Indicates whether the diagonal elements of a triangular matrix are assumed to be one.\n"
            "                           Only applicable to trtri.\n"
            "                           ")

        // stebz options
        ("eorder",
         value<char>()->default_value('E'),
            "E = entire matrix, B = by blocks.\n"
            "                           Indicates whether the computed eigenvalues are ordered by blocks or for the entire matrix.\n"
            "                           Only applicable to stebz.\n"
            "                           ")

        // geblttrf/geblttrs options
        ("nb",
         value<rocblas_int>(),
            "Number of rows and columns in each block.\n"
            "                           Only applicable to block tridiagonal matrix APIs.\n"
            "                           ")

        ("nblocks",
         value<rocblas_int>(),
            "Number of blocks along the diagonal.\n"
            "                           Only applicable to block tridiagonal matrix APIs.\n"
            "                           ")

        // partial eigenvalue/singular value decomposition options
        ("il",
         value<rocblas_int>(),
            "Lower index in ordered subset of eigenvalues.\n"
            "                           Used in partial eigenvalue decomposition functions.\n"
            "                           ")

        ("iu",
         value<rocblas_int>(),
            "Upper index in ordered subset of eigenvalues.\n"
            "                           Used in partial eigenvalue decomposition functions.\n"
            "                           ")

        ("erange",
         value<char>()->default_value('A'),
            "A = all eigenvalues, V = in (vl, vu], I = from the il-th to the iu-th.\n"
            "                           For partial eigenvalue decompositions, it indicates the type of interval in which\n"
            "                           the eigenvalues will be found.\n"
            "                           ")

        ("srange",
         value<char>()->default_value('A'),
            "A = all singular values, V = in (vl, vu], I = from the il-th to the iu-th.\n"
            "                           For partial singular value decompositions, it indicates the type of interval in which\n"
            "                           the singular values will be found.\n"
            "                           ")

        ("vl",
         value<double>(),
            "Lower bound of half-open interval (vl, vu].\n"
            "                           Used in partial eigenvalue decomposition functions.\n"
            "                           Note: the used random input matrices have all eigenvalues in [-20, 20].\n"
            "                           ")

        ("vu",
         value<double>(),
            "Upper bound of half-open interval (vl, vu].\n"
            "                           Used in partial eigenvalue decomposition functions.\n"
            "                           Note: the used random input matrices have all eigenvalues in [-20, 20].\n"
            "                           ")

        // iterative Jacobi options
        ("max_sweeps",
         value<rocblas_int>()->default_value(100),
            "Maximum number of sweeps/iterations.\n"
            "                           Used in iterative Jacobi functions.\n"
            "                           ")

        ("esort",
         value<char>()->default_value('A'),
            "N = no sorting, A = ascending order.\n"
            "                           Indicates whether the computed eigenvalues are sorted in ascending order.\n"
            "                           Used in iterative Jacobi functions.\n"
            "                           ")

        // other options
        ("abstol",
         value<double>()->default_value(0),
            "Absolute tolerance at which convergence is accepted.\n"
            "                           Used in iterative Jacobi and partial eigenvalue decomposition functions.\n"
            "                           ")

        ("direct",
         value<char>()->default_value('F'),
            "F = forward, B = backward.\n"
            "                           The order in which a series of transformations are applied.\n"
            "                           ")

        ("pivot",
         value<char>()->default_value('V'),
            "V = variable, T = top, B = bottom.\n"
            "                           Defines the planes on which a sequence of rotations is applied.\n"
            "                           ")

        ("evect",
         value<char>()->default_value('N'),
            "N = none, V = compute eigenvectors of the matrix,\n"
            "                           I = compute eigenvectors of the tridiagonal matrix.\n"
            "                           Indicates how the eigenvectors are to be calculated and stored.\n"
            "                           ")

        ("fast_alg",
         value<char>()->default_value('O'),
            "O = out-of-place, I = in-place.\n"
            "                           Enables out-of-place computations.\n"
            "                           ")

        ("itype",
         value<char>()->default_value('1'),
            "1 = Ax, 2 = ABx, 3 = BAx.\n"
            "                           Problem type for generalized eigenproblems.\n"
            "                           ")

        ("side",
         value<char>(),
            "L = left, R = right.\n"
            "                           The side from which a matrix should be multiplied.\n"
            "                           ")

        ("storev",
         value<char>(),
            "C = column-wise, R = row-wise.\n"
            "                           Indicates whether data is stored column-wise or row-wise.\n"
            "                           ")

        ("trans",
         value<char>()->default_value('N'),
            "N = no transpose, T = transpose, C = conjugate transpose.\n"
            "                           Indicates if a matrix should be transposed.\n"
            "                           ")

        ("uplo",
         value<char>()->default_value('U'),
            "U = upper, L = lower.\n"
            "                           Indicates where the data for a triangular or symmetric/hermitian matrix is stored.\n"
            "                           ");

    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    // print help message
    if(vm.count("help"))
    {
        std::stringstream desc_ss{};
        desc_ss << desc;
        fmt::print("{}{}\n", help_str, desc_ss.str());
        return 0;
    }

    argus.populate(vm);

    if(!argus.perf)
    {
        print_version_info();

        rocblas_int device_count = query_device_property();
        if(device_count <= 0)
            throw std::runtime_error("No devices found");
        if(device_count <= device_id)
            throw std::invalid_argument("Invalid Device ID");
    }
    set_device(device_id);

    // catch invalid arguments
    argus.validate_precision("precision");
    argus.validate_operation("trans");
    argus.validate_side("side");
    argus.validate_fill("uplo");
    argus.validate_diag("diag");
    argus.validate_direct("direct");
    argus.validate_pivot("pivot");
    argus.validate_storev("storev");
    argus.validate_svect("svect");
    argus.validate_svect("left_svect");
    argus.validate_svect("right_svect");
    argus.validate_erange("srange");
    argus.validate_workmode("fast_alg");
    argus.validate_evect("evect");
    argus.validate_erange("erange");
    argus.validate_eorder("eorder");
    argus.validate_esort("esort");
    argus.validate_itype("itype");
    argus.validate_rfinfo_mode("rfinfo_mode");

    // prepare logging infrastructure and ignore environment variables
    rocsolver_log_begin();
    rocsolver_log_set_layer_mode(rocblas_layer_mode_none);

    // select and dispatch function test/benchmark
    rocsolver_dispatcher::invoke(function, precision, argus);

    // terminate logging
    rocsolver_log_end();

    return 0;
}
catch(const std::exception& exp)
{
    fmt::print(stderr, "{}\n", exp.what());
    return -1;
}
