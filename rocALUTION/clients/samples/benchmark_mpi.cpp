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

#include "common.hpp"

#include <iostream>
#include <mpi.h>
#include <rocalution/rocalution.hpp>

#define ValueType double

using namespace rocalution;

int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    int num_procs;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    // Check command line parameters
    if(num_procs < 2)
    {
        std::cerr << "Expecting at least 2 MPI processes" << std::endl;
        return -1;
    }

    if(argc < 2)
    {
        std::cerr << argv[0] << " <global_matrix>" << std::endl;
        return -1;
    }

    // Disable OpenMP thread affinity
    set_omp_affinity_rocalution(false);

    // Initialize platform with rank and # of accelerator devices in the node
    init_rocalution(rank, 2);

    // Disable OpenMP
    set_omp_threads_rocalution(1);

    // Print platform
    info_rocalution();

    // Load undistributed matrix
    LocalMatrix<ValueType> lmat;
    lmat.ReadFileMTX(argv[1]);

    // Global structures
    ParallelManager         manager;
    GlobalMatrix<ValueType> mat;

    // Distribute matrix - lmat will be destroyed
    distribute_matrix(&comm, &lmat, &mat, &manager);

    // rocALUTION vectors
    GlobalVector<ValueType> v1(manager);
    GlobalVector<ValueType> v2(manager);

    // Move structures to accelerator, if available
    mat.MoveToAccelerator();
    v1.MoveToAccelerator();
    v2.MoveToAccelerator();

    // Allocate memory
    v1.Allocate("v1", mat.GetN());
    v2.Allocate("v2", mat.GetM());

    size_t size = mat.GetM();
    size_t nnz;

    // Initialize objects
    v1.Ones();
    v2.Zeros();
    mat.Apply(v1, &v2);

    // Print object info
    mat.Info();
    v1.Info();
    v2.Info();

    // Number of tests
    int tests = 200;

    double time;

    if(rank == 0)
    {
        std::cout << "--------------------------------------- BENCHMARKS "
                     "--------------------------------------"
                  << std::endl;
    }

    // Dot product
    // Size = 2*size
    // Flop = 2 per element
    v1.Dot(v2);

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        v1.Dot(v2);
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "Dot execution: " << time / 1e3 / tests << " msec"
                  << "; " << tests * double(sizeof(ValueType) * (2 * size)) / time / 1e3
                  << " GByte/sec; " << tests * double(2 * size) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    // L2 Norm
    // Size = size
    // Flop = 2 per element
    v1.Norm();

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        v1.Norm();
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "Norm2 execution: " << time / 1e3 / tests << " msec"
                  << "; " << tests * double(sizeof(ValueType) * (size)) / time / 1e3
                  << " GByte/sec; " << tests * double(2 * size) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    // Reduction
    // Size = size
    // Flop = 1 per element
    v1.Reduce();

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        v1.Reduce();
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "Reduce execution: " << time / 1e3 / tests << " msec"
                  << "; " << tests * double(sizeof(ValueType) * (size)) / time / 1e3
                  << " GByte/sec; " << tests * double(size) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    if(rank == 0)
    {
        std::cout << "----------------------------------------------------"
                     "-------------------------------------"
                  << std::endl;
    }

    // Vector Update 1
    // Size = 3*size
    // Flop = 2 per element
    v1.ScaleAdd((ValueType)3.1, v2);

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        v1.ScaleAdd((ValueType)3.1, v2);
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "Vector update (ScaleAdd) execution: " << time / 1e3 / tests << " msec"
                  << "; " << tests * double(sizeof(ValueType) * (3 * size)) / time / 1e3
                  << " GByte/sec; " << tests * double(2 * size) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    // Vector Update 2
    // Size = 3*size
    // Flop = 2 per element
    v1.AddScale(v2, (ValueType)3.1);
    _rocalution_sync();

    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        v1.AddScale(v2, (ValueType)3.1);
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "Vector update (AddScale) execution: " << time / 1e3 / tests << " msec"
                  << "; " << tests * double(sizeof(ValueType) * (3 * size)) / time / 1e3
                  << " GByte/sec; " << tests * double(2 * size) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    if(rank == 0)
    {
        std::cout << "----------------------------------------------------"
                     "-------------------------------------"
                  << std::endl;
    }

    // Matrix vector multiplication CSR
    // Size = int(size+1+nnz) [row_offset + col] + ValueType(2*size+nnz) [in + out + nnz]
    // Flop = 2 per entry (nnz)

    mat.ConvertToCSR();
    nnz = mat.GetNnz();

    mat.Info();

    mat.Apply(v1, &v2);

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        mat.Apply(v1, &v2);
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "CSR SpMV execution: " << time / 1e3 / tests << " msec"
                  << "; "
                  << tests
                         * double((sizeof(ValueType) * (2 * size + nnz)
                                   + sizeof(int) * (size + 1 + nnz)))
                         / time / 1e3
                  << " GByte/sec; " << tests * double(2 * nnz) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    if(rank == 0)
    {
        std::cout << "----------------------------------------------------"
                     "-------------------------------------"
                  << std::endl;
    }

    // Matrix vector multiplication MCSR
    // Size = int(size+(nnz-size)) [row_offset + col] + valuetype(2*size+nnz) [in + out + nnz]
    // Flop = 2 per entry (nnz)

    mat.ConvertToMCSR();
    nnz = mat.GetNnz();

    mat.Info();

    mat.Apply(v1, &v2);

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        mat.Apply(v1, &v2);
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "MCSR SpMV execution: " << time / 1e3 / tests << " msec"
                  << "; "
                  << tests
                         * double((sizeof(ValueType) * (2 * size + nnz - size)
                                   + sizeof(int) * (size + 1 + nnz)))
                         / time / 1e3
                  << " GByte/sec; " << tests * double(2 * nnz) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    if(rank == 0)
    {
        std::cout << "----------------------------------------------------"
                     "-------------------------------------"
                  << std::endl;
    }

    // Matrix vector multiplication ELL
    // Size = int(nnz) [col] + ValueType(2*size+nnz) [in + out + nnz]
    // Flop = 2 per entry (nnz)

    mat.ConvertToELL();
    nnz = mat.GetNnz();

    mat.Info();

    mat.Apply(v1, &v2);

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        mat.Apply(v1, &v2);
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "ELL SpMV execution: " << time / 1e3 / tests << " msec"
                  << "; "
                  << tests * double((sizeof(ValueType) * (2 * size + nnz) + sizeof(int) * (nnz)))
                         / time / 1e3
                  << " GByte/sec; " << tests * double(2 * nnz) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    if(rank == 0)
    {
        std::cout << "----------------------------------------------------"
                     "-------------------------------------"
                  << std::endl;
    }

    // Matrix vector multiplication COO
    // Size = int(2*nnz) [col+row] + ValueType(2*size+nnz) [in + out + nnz]
    // Flop = 2 per entry (nnz)

    mat.ConvertToCOO();
    nnz = mat.GetNnz();

    mat.Info();

    mat.Apply(v1, &v2);

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        mat.Apply(v1, &v2);
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "COO SpMV execution: " << time / 1e3 / tests << " msec"
                  << "; "
                  << tests
                         * double((sizeof(ValueType) * (2 * size + nnz) + sizeof(int) * (2 * nnz)))
                         / time / 1e3
                  << " GByte/sec; " << tests * double(2 * nnz) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    if(rank == 0)
    {
        std::cout << "----------------------------------------------------"
                     "-------------------------------------"
                  << std::endl;
    }

    // Matrix vector multiplication HYB
    // Size = int(nnz) [col] + valuetype(2*size+nnz) [in + out + nnz]
    // Flop = 2 per entry (nnz)

    mat.ConvertToHYB();
    nnz = mat.GetNnz();

    mat.Info();

    mat.Apply(v1, &v2);

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        mat.Apply(v1, &v2);
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "HYB SpMV execution: " << time / 1e3 / tests << " msec"
                  << "; "
                  << tests * double((sizeof(ValueType) * (2 * size + nnz) + sizeof(int) * (nnz)))
                         / time / 1e3
                  << " GByte/sec; " << tests * double(2 * nnz) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    if(rank == 0)
    {
        std::cout << "----------------------------------------------------"
                     "-------------------------------------"
                  << std::endl;
    }

    // Matrix vector multiplication DIA
    // Size = int(size+nnz) + valuetype(2*size+nnz)
    // Flop = 2 per entry (nnz)

    mat.ConvertToDIA();
    nnz = mat.GetNnz();

    mat.Info();

    mat.Apply(v1, &v2);

    _rocalution_sync();
    time = rocalution_time();

    for(int i = 0; i < tests; ++i)
    {
        mat.Apply(v1, &v2);
        _rocalution_sync();
    }

    _rocalution_sync();
    time = rocalution_time() - time;

    if(rank == 0)
    {
        std::cout << "DIA SpMV execution: " << time / 1e3 / tests << " msec"
                  << "; " << tests * double((sizeof(ValueType) * (nnz))) / time / 1e3
                  << " GByte/sec; " << tests * double(2 * nnz) / time / 1e3 << " GFlop/sec"
                  << std::endl;
    }

    if(rank == 0)
    {
        std::cout << "----------------------------------------------------"
                     "-------------------------------------"
                  << std::endl;
    }

    mat.ConvertToCSR();

    if(rank == 0)
    {
        std::cout << "----------------------------------------------------" << std::endl;
        std::cout << "Combined micro benchmarks" << std::endl;
    }

    double dot_tick = 0, dot_tack = 0;
    double norm_tick = 0, norm_tack = 0;
    double red_tick = 0, red_tack = 0;
    double updatev1_tick = 0, updatev1_tack = 0;
    double updatev2_tick = 0, updatev2_tack = 0;
    double spmv_tick = 0, spmv_tack = 0;

    for(int i = 0; i < tests; ++i)
    {
        v1.Ones();
        v2.Zeros();
        mat.Apply(v1, &v2);

        // Dot product
        // Size = 2*size
        // Flop = 2 per element
        v1.Dot(v2);

        dot_tick += rocalution_time();

        v1.Dot(v2);

        dot_tack += rocalution_time();

        v1.Ones();
        v2.Zeros();
        mat.Apply(v1, &v2);

        // Norm
        // Size = size
        // Flop = 2 per element
        v1.Norm();

        norm_tick += rocalution_time();

        v1.Norm();

        norm_tack += rocalution_time();

        v1.Ones();
        v2.Zeros();
        mat.Apply(v1, &v2);

        // Reduce
        // Size = size
        // Flop = 1 per element
        v1.Reduce();

        red_tick += rocalution_time();

        v1.Reduce();

        red_tack += rocalution_time();

        v1.Ones();
        v2.Zeros();
        mat.Apply(v1, &v2);

        // Vector Update 1
        // Size = 3xsize
        // Flop = 2 per element
        v1.ScaleAdd(double(5.5), v2);

        updatev1_tick += rocalution_time();

        v1.ScaleAdd(double(5.5), v2);

        updatev1_tack += rocalution_time();

        v1.Ones();
        v2.Zeros();
        mat.Apply(v1, &v2);

        // Vector Update 2
        // Size = 3*size
        // Flop = 2 per element
        v1.AddScale(v2, double(5.5));

        updatev2_tick += rocalution_time();

        v1.AddScale(v2, double(5.5));

        updatev2_tack += rocalution_time();

        v1.Ones();
        v2.Zeros();
        mat.Apply(v1, &v2);

        // Matrix-Vector Multiplication
        // Size = int(size+nnz) + valuetype(2*size+nnz)
        // Flop = 2 per entry (nnz)
        mat.Apply(v1, &v2);

        spmv_tick += rocalution_time();

        mat.Apply(v1, &v2);

        spmv_tack += rocalution_time();
    }

    if(rank == 0)
    {
        std::cout << "Dot execution: " << (dot_tack - dot_tick) / tests / 1e3 << " msec"
                  << "; "
                  << tests * double(sizeof(double) * (size + size)) / (dot_tack - dot_tick) / 1e3
                  << " Gbyte/sec; " << tests * double(2 * size) / (dot_tack - dot_tick) / 1e3
                  << " GFlop/sec" << std::endl;

        std::cout << "Norm execution: " << (norm_tack - norm_tick) / tests / 1e3 << " msec"
                  << "; " << tests * double(sizeof(double) * (size)) / (norm_tack - norm_tick) / 1e3
                  << " Gbyte/sec; " << tests * double(2 * size) / (norm_tack - norm_tick) / 1e3
                  << " GFlop/sec" << std::endl;

        std::cout << "Reduce execution: " << (red_tack - red_tick) / tests / 1e3 << " msec"
                  << "; " << tests * double(sizeof(double) * (size)) / (red_tack - red_tick) / 1e3
                  << " Gbyte/sec; " << tests * double(size) / (red_tack - red_tick) / 1e3
                  << " GFlop/sec" << std::endl;

        std::cout << "Vector update (scaleadd) execution: "
                  << (updatev1_tack - updatev1_tick) / tests / 1e3 << " msec"
                  << "; "
                  << tests * double(sizeof(double) * (size + size + size))
                         / (updatev1_tack - updatev1_tick) / 1e3
                  << " Gbyte/sec; "
                  << tests * double(2 * size) / (updatev1_tack - updatev1_tick) / 1e3
                  << " GFlop/sec" << std::endl;

        std::cout << "Vector update (addscale) execution: "
                  << (updatev2_tack - updatev2_tick) / tests / 1e3 << " msec"
                  << "; "
                  << tests * double(sizeof(double) * (size + size + size))
                         / (updatev2_tack - updatev2_tick) / 1e3
                  << " Gbyte/sec; "
                  << tests * double(2 * size) / (updatev2_tack - updatev2_tick) / 1e3
                  << " GFlop/sec" << std::endl;

        std::cout << "SpMV execution: " << (spmv_tack - spmv_tick) / tests / 1e3 << " msec"
                  << "; "
                  << tests
                         * double(
                             (sizeof(double) * (size + size + nnz) + sizeof(int) * (size + nnz)))
                         / (spmv_tack - spmv_tick) / 1e3
                  << " Gbyte/sec; " << tests * double((2 * nnz) / (spmv_tack - spmv_tick)) / 1e3
                  << " GFlop/sec" << std::endl;
    }

    stop_rocalution();

    MPI_Finalize();

    return 0;
}
