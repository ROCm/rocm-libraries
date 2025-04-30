/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * ************************************************************************ */

#include "../include/hipsolver_dispatcher.hpp"
#include "../rocblascommon/program_options.hpp"

using rocblas_int    = int;
using rocblas_stride = ptrdiff_t;
using namespace roc;

// clang-format off
const char* help_str = R"HELP_STR(
hipSOLVER benchmark client help.

Usage: ./hipsolver-bench <options>

In addition to some common general options, the following list of options corresponds to all the parameters
that might be needed to test a given hipSOLVER function. The parameters are named as in the API user guide.
The arrays are initialized internally by the program with random values.

Note: When a required parameter/option is not provided, it will take the default value as listed below.
If no default value is defined, the program will try to calculate a suitable value depending on the context
of the problem and the tested function; if this is not possible, the program will abort with an error.
Functions that accept multiple size parameters can generally be provided a single size parameter (typically,
m) and a square-size matrix will be assumed.

Example: ./hipsolver-bench -f getrf -m 30 --lda 75
This will test getrf with a random 30x30 matrix.

Options:
)HELP_STR";
// clang-format on

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
    char        precision;
    rocblas_int device_id;

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
         value<std::string>(&function)->default_value("getrf"),
            "The LAPACK function to test.\n"
            "                           Options are: getrf, getrs, potrf, potrf_batched, etc.\n"
            "                           ")

        ("iters,i",
         value<rocblas_int>(&argus.iters)->default_value(10),
            "Iterations to run inside the GPU timing loop.\n"
            "                           Reported time will be the average.\n"
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

        // ("singular",
        //  value<rocblas_int>(&argus.singular)->default_value(0),
        //     "Test with degenerate matrices? 0 = No, 1 = Yes\n"
        //     "                           This will produce matrices that are singular, non positive-definite, etc.\n"
        //     "                           ")

        ("verify,v",
         value<rocblas_int>(&argus.norm_check)->default_value(0),
            "Validate GPU results with CPU? 0 = No, 1 = Yes.\n"
            "                           This will additionally print the relative error of the computations.\n"
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

        // ("ldt",
        //  value<rocblas_int>(),
        //     "Matrix size parameter.\n"
        //     "                           Leading dimension of matrices T.\n"
        //     "                           ")

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

        // ("ldw",
        //  value<rocblas_int>(),
        //     "Matrix size parameter.\n"
        //     "                           Leading dimension of matrices W.\n"
        //     "                           ")

        ("ldx",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Leading dimension of matrices X.\n"
            "                           ")

        // ("ldy",
        //  value<rocblas_int>(),
        //     "Matrix size parameter.\n"
        //     "                           Leading dimension of matrices Y.\n"
        //     "                           ")

        // stride options
        ("strideA",
         value<rocblas_stride>(),
            "Matrix/vector stride parameter.\n"
            "                           Stride for matrices/vectors A.\n"
            "                           ")

        // ("strideB",
        //  value<rocblas_stride>(),
        //     "Matrix/vector stride parameter.\n"
        //     "                           Stride for matrices/vectors B.\n"
        //     "                           ")

        // ("strideD",
        //  value<rocblas_stride>(),
        //     "Matrix/vector stride parameter.\n"
        //     "                           Stride for matrices/vectors D.\n"
        //     "                           ")

        // ("strideE",
        //  value<rocblas_stride>(),
        //     "Matrix/vector stride parameter.\n"
        //     "                           Stride for matrices/vectors E.\n"
        //     "                           ")

        // ("strideQ",
        //  value<rocblas_stride>(),
        //     "Matrix/vector stride parameter.\n"
        //     "                           Stride for vectors tauq.\n"
        //     "                           ")

        // ("strideP",
        //  value<rocblas_stride>(),
        //     "Matrix/vector stride parameter.\n"
        //     "                           Stride for vectors tau, taup, and ipiv.\n"
        //     "                           ")

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

        // bdsqr options
        // ("nc",
        //  value<rocblas_int>()->default_value(0),
        //     "The number of columns of matrix C.\n"
        //     "                           Only applicable to bdsqr.\n"
        //     "                           ")

        // ("nu",
        //  value<rocblas_int>(),
        //     "The number of columns of matrix U.\n"
        //     "                           Only applicable to bdsqr.\n"
        //     "                           ")

        // ("nv",
        //  value<rocblas_int>()->default_value(0),
        //     "The number of columns of matrix V.\n"
        //     "                           Only applicable to bdsqr.\n"
        //     "                           ")

        // laswp options
        // ("k1",
        //  value<rocblas_int>(),
        //     "First index for row interchange.\n"
        //     "                           Only applicable to laswp.\n"
        //     "                           ")

        // ("k2",
        //  value<rocblas_int>(),
        //     "Last index for row interchange.\n"
        //     "                           Only applicable to laswp.\n"
        //     "                           ")

        // gesvd options
        ("jobu",
         value<char>()->default_value('N'),
            "N = none, A = the entire orthogonal matrix is computed,\n"
            "                           S = the singular vectors are computed,\n"
            "                           O = the singular vectors overwrite the original matrix.\n"
            "                           Indicates how the left singular vectors are to be calculated and stored.\n"
            "                           ")

        ("jobv",
         value<char>()->default_value('N'),
            "N = none, A = the entire orthogonal matrix is computed,\n"
            "                           S = the singular vectors are computed,\n"
            "                           O = the singular vectors overwrite the original matrix.\n"
            "                           Indicates how the right singular vectors are to be calculated and stored.\n"
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

        ("range",
         value<char>()->default_value('A'),
            "A = all eigenvalues, V = in (vl, vu], I = from the il-th to the iu-th.\n"
            "                           For partial eigenvalue decompositions, it indicates the type of interval in which\n"
            "                           the eigenvalues will be found.\n"
            "                           ")

        ("rank",
         value<rocblas_int>(),
            "The number of singular values to be computed.\n"
            "                           Used in partial SVD functions.\n"
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
        ("econ",
         value<rocblas_int>()->default_value(0),
            "Enable economy size for singular vector matrices? 0 = No, 1 = Yes.\n"
            "                           Only applicable to gesvdj.\n"
            "                           ")

        ("max_sweeps",
         value<rocblas_int>()->default_value(100),
            "Maximum number of sweeps/iterations.\n"
            "                           Used in iterative Jacobi functions.\n"
            "                           ")

        ("tolerance",
         value<double>(),
            "Absolute tolerance at which convergence is accepted.\n"
            "                           Used in iterative Jacobi functions.\n"
            "                           ")

        ("sort_eig",
         value<rocblas_int>()->default_value(1),
            "0 = no sorting, 1 = ascending order.\n"
            "                           Indicates whether the computed eigenvalues are sorted in ascending order.\n"
            "                           Used in iterative Jacobi functions.\n"
            "                           ")

        // sparse routine options
        ("base1",
         value<rocblas_int>(),
            "0 = use base zero indices, 1 = use base one indices.\n"
            "                           ")

        ("nnzA",
         value<rocblas_int>(),
            "Matrix size parameter.\n"
            "                           Maximum number of non-zero entries of matrices A.\n"
            "                           ")

        ("reorder",
         value<rocblas_int>(),
            "0 = no reordering, 1 = RCM reordering,\n"
            "                           2 = AMD reordering, 3 = METIS reordering.\n"
            "                           ")

        // other options
        // ("direct",
        //  value<char>()->default_value('F'),
        //     "F = forward, B = backward.\n"
        //     "                           The order in which a series of transformations are applied.\n"
        //     "                           ")

        // ("fast_alg",
        //  value<char>()->default_value('O'),
        //     "O = out-of-place, I = in-place.\n"
        //     "                           Enables out-of-place computations.\n"
        //     "                           ")

        // ("incx",
        //  value<rocblas_int>()->default_value(1),
        //     "Increment between values in vector x.\n"
        //     "                           ")

        ("itype",
         value<char>()->default_value('1'),
            "1 = Ax, 2 = ABx, 3 = BAx.\n"
            "                           Problem type for generalized eigenproblems.\n"
            "                           ")

        ("jobz",
         value<char>()->default_value('N'),
            "N = none, V = compute eigenvectors/singular vectors of the matrix,\n"
            "                           Indicates how the eigenvectors/singular vectors are to be calculated and stored.\n"
            "                           ")

        ("side",
         value<char>(),
            "L = left, R = right.\n"
            "                           The side from which a matrix should be multiplied.\n"
            "                           ")

        // ("storev",
        //  value<char>(),
        //     "C = column-wise, R = row-wise.\n"
        //     "                           Indicates whether data is stored column-wise or row-wise.\n"
        //     "                           ")

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
        std::cout << help_str << desc << std::endl;
        return 0;
    }

    argus.populate(vm);

    // set device ID
    if(!argus.perf)
    {
        rocblas_int device_count = query_device_property();
        if(device_count <= device_id)
            throw std::invalid_argument("Invalid Device ID");
    }
    set_device(device_id);

    // catch invalid arguments
    argus.validate_precision("precision");
    argus.validate_operation("trans");
    argus.validate_side("side");
    argus.validate_fill("uplo");
    // argus.validate_direct("direct");
    // argus.validate_storev("storev");
    argus.validate_svect("jobu");
    argus.validate_svect("jobv");
    // argus.validate_workmode("fast_alg");
    argus.validate_itype("itype");
    argus.validate_evect("jobz");
    argus.validate_erange("range");

    // select and dispatch function test/benchmark
    hipsolver_dispatcher::invoke(function, precision, argus);

    return 0;
}

catch(const std::invalid_argument& exp)
{
    std::cerr << exp.what() << std::endl;
    return -1;
}
