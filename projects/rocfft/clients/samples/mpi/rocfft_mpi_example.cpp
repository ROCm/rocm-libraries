
/******************************************************************************
* Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <numeric>
#include <type_traits>
#include <vector>

#include "../../../shared/CLI11.hpp"
#include "rocfft.h"

// Check all ranks for an rocFFT non-success status.
auto rocfft_status_sync(const rocfft_status fftrc, const MPI_Comm comm)
{
    // Since hipSuccess is the lowest enum value, we can find if there are any errors
    // by getting the maximum value of the return code over all procs.

    // Guarantee that the enum is an unsigned int so that we can send this via MPI:
    static_assert(std::is_same_v<std::underlying_type_t<decltype(fftrc)>, unsigned int>);

    auto       global_fftrc = rocfft_status_success;
    const auto mpirc        = MPI_Allreduce(&fftrc, &global_fftrc, 1, MPI_UNSIGNED, MPI_MAX, comm);

    if(mpirc != MPI_SUCCESS)
    {
        return rocfft_status_failure;
    }
    return global_fftrc;
}

// Check all ranks for an hip runtime non-success status.
auto hip_status_sync(const hipError_t hiprc, const MPI_Comm comm)
{
    // Since rocfft_status_success is the lowest enum value, we can find if there are any errors
    // by getting the maximum value of the return code over all procs.

    // Guarantee that the enum is an unsigned int so that we can send this via MPI:
    static_assert(std::is_same_v<std::underlying_type_t<decltype(hiprc)>, unsigned int>);

    auto       global_hiprc = hipSuccess;
    const auto mpirc        = MPI_Allreduce(&hiprc, &global_hiprc, 1, MPI_UNSIGNED, MPI_MAX, comm);

    if(mpirc != MPI_SUCCESS)
    {
        return hipErrorUnknown;
    }
    return global_hiprc;
}

int main(int argc, char** argv)
{

    MPI_Init(&argc, &argv);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int mpi_size = 0;
    MPI_Comm_size(mpi_comm, &mpi_size);

    int mpi_rank = 0;
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    // General FFT parameters:
    std::vector<size_t> length = {8, 8};

    // Number of independent FFTs of size `length` per execution
    // (override with `--nbatch N`; 1 reproduces the unbatched case)
    // Decomposition is purely spatial, so each rank holds its slab
    // of all `nbatch` transforms
    size_t nbatch = 4;

    // MPI launchers replicate argv to every rank, so parsing
    // identically on each rank gives a consistent value
    CLI::App app{"rocFFT batched MPI example"};
    app.add_option("-b,--nbatch", nbatch, "Number of batched 2-D transforms")
        ->default_val(nbatch)
        ->check(CLI::PositiveNumber);
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    // Slab decomposition splits the slowest spatial dimension across
    // ranks; more ranks than slabs would leave some bricks empty
    if(static_cast<size_t>(mpi_size) > length.back())
    {
        if(mpi_rank == 0)
        {
            std::cerr << "Error: mpi_size (" << mpi_size
                      << ") exceeds the slowest FFT dimension length.back() (" << length.back()
                      << ").  Re-run with at most " << length.back() << " ranks.\n";
        }
        MPI_Finalize();
        return 1;
    }

    const rocfft_transform_type   direction = rocfft_transform_type_complex_forward;
    const rocfft_result_placement place     = rocfft_placement_notinplace;

    if(mpi_rank == 0)
    {
        std::cout << "rocFFT batched MPI example\n";
        std::cout << "MPI size: " << mpi_size << "\n";
        std::cout << "FFT length: " << length[0] << " x " << length[1] << ", nbatch: " << nbatch
                  << "\n";
    }

    auto fftrc = rocfft_status_success;
    auto hiprc = hipSuccess;

    fftrc = rocfft_setup();
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("failed to set up rocFFT");

    rocfft_plan_description description = nullptr;
    rocfft_plan_description_create(&description);

    fftrc = rocfft_plan_description_set_comm(description, rocfft_comm_mpi, &mpi_comm);
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("failed add communicator to description");

    // Do not set stride information via the descriptor, they are to be defined during field
    // creation below
    fftrc = rocfft_plan_description_set_data_layout(description,
                                                    rocfft_array_type_complex_interleaved,
                                                    rocfft_array_type_complex_interleaved,
                                                    nullptr,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    nullptr,
                                                    0);
    if(fftrc != rocfft_status_success)
        throw std::runtime_error("failed to create description");

    if(mpi_rank == 0)
    {
        std::cout << "input data decomposition:\n";
    }
    std::vector<void*> gpu_in = {nullptr};
    {
        rocfft_field infield = nullptr;
        rocfft_field_create(&infield);

        // This rank's slab in the slowest spatial dimension
        const size_t inbrick_length1 = length[1] / (size_t)mpi_size
                                       + ((size_t)mpi_rank < length[1] % (size_t)mpi_size ? 1 : 0);
        const size_t inbrick_lower1
            = mpi_rank * (length[1] / mpi_size) + std::min((size_t)mpi_rank, length[1] % mpi_size);
        const size_t inbrick_upper1 = inbrick_lower1 + inbrick_length1;

        // Brick coordinates and strides are column-major (fastest-moving
        // dimension first) and include the batch dimension as the last
        // (slowest) entry, so the arrays have N+1 entries for an N-dim FFT.
        // For a packed layout the batch stride is the local plane size
        const size_t        inbrick_dist   = length[0] * inbrick_length1;
        std::vector<size_t> inbrick_stride = {1, length[0], inbrick_dist};
        std::vector<size_t> inbrick_lower  = {0, inbrick_lower1, 0};
        std::vector<size_t> inbrick_upper  = {length[0], inbrick_upper1, nbatch};

        rocfft_brick inbrick = nullptr;
        rocfft_brick_create(&inbrick,
                            inbrick_lower.data(),
                            inbrick_upper.data(),
                            inbrick_stride.data(),
                            inbrick_lower.size(),
                            0);
        rocfft_field_add_brick(infield, inbrick);
        rocfft_brick_destroy(inbrick);
        inbrick = nullptr;

        const size_t memSize = nbatch * inbrick_dist * sizeof(std::complex<double>);
        std::vector<std::complex<double>> host_in(nbatch * inbrick_dist);
        // idx0 (unit stride) innermost for contiguous access
        for(size_t batch = 0; batch < nbatch; ++batch)
        {
            for(auto idx1 = inbrick_lower[1]; idx1 < inbrick_upper[1]; ++idx1)
            {
                for(auto idx0 = inbrick_lower[0]; idx0 < inbrick_upper[0]; ++idx0)
                {
                    const auto pos = batch * inbrick_stride[2]
                                     + (idx0 - inbrick_lower[0]) * inbrick_stride[0]
                                     + (idx1 - inbrick_lower[1]) * inbrick_stride[1];
                    // Offset by batch so each transform's output is distinct,
                    // confirming batches were processed independently
                    const double real = static_cast<double>(idx0 + batch);
                    const double imag = static_cast<double>(idx1);
                    host_in[pos]      = std::complex<double>(real, imag);
                }
            }
        }

        // Serialize output:
        for(int irank = 0; irank < mpi_size; ++irank)
        {
            if(mpi_rank == irank)
            {
                std::cout << "in-brick rank " << irank;
                std::cout << "\n\tlower indices:";
                for(const auto val : inbrick_lower)
                    std::cout << " " << val;
                std::cout << "\n\tupper indices:";
                for(const auto val : inbrick_upper)
                    std::cout << " " << val;
                std::cout << "\n\tstrides:";
                for(const auto val : inbrick_stride)
                    std::cout << " " << val;
                std::cout << "\n";
                std::cout << "\tnbatch: " << nbatch << "\n";
                std::cout << "\tbuffer size: " << memSize << "\n";
                std::cout << "\tinput batch 0:\n";
                const size_t batch0 = 0;
                for(auto idx1 = inbrick_lower[1]; idx1 < inbrick_upper[1]; ++idx1)
                {
                    for(auto idx0 = inbrick_lower[0]; idx0 < inbrick_upper[0]; ++idx0)
                    {
                        const auto pos = batch0 * inbrick_stride[2]
                                         + (idx0 - inbrick_lower[0]) * inbrick_stride[0]
                                         + (idx1 - inbrick_lower[1]) * inbrick_stride[1];
                        std::cout << host_in[pos] << " ";
                    }
                    std::cout << "\n";
                }
            }
            MPI_Barrier(mpi_comm);
        }

        hiprc = hipMalloc(&gpu_in[0], memSize);
        if(hiprc != hipSuccess)
            throw std::runtime_error("inbrick hipMalloc failed");
        hiprc = hipMemcpy(gpu_in[0], host_in.data(), memSize, hipMemcpyHostToDevice);
        if(hiprc != hipSuccess)
            throw std::runtime_error("inbrick hipMemcpy failed");

        rocfft_plan_description_add_infield(description, infield);

        fftrc = rocfft_field_destroy(infield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("failed destroy infield");
    }

    if(mpi_rank == 0)
    {
        std::cout << "output data decomposition:\n";
    }
    std::vector<void*>  gpu_out = {nullptr};
    std::vector<size_t> outbrick_lower;
    std::vector<size_t> outbrick_upper;
    std::vector<size_t> outbrick_stride;
    size_t              outbrick_dist = 0;
    {
        const size_t outbrick_length1 = length[1] / (size_t)mpi_size
                                        + ((size_t)mpi_rank < length[1] % (size_t)mpi_size ? 1 : 0);
        const size_t outbrick_lower1
            = mpi_rank * (length[1] / mpi_size) + std::min((size_t)mpi_rank, length[1] % mpi_size);
        const size_t outbrick_upper1 = outbrick_lower1 + outbrick_length1;

        // Same column-major + batched-last layout as the input brick
        outbrick_dist   = length[0] * outbrick_length1;
        outbrick_stride = {1, length[0], outbrick_dist};
        outbrick_lower  = {0, outbrick_lower1, 0};
        outbrick_upper  = {length[0], outbrick_upper1, nbatch};

        const size_t memSize = nbatch * outbrick_dist * sizeof(std::complex<double>);
        for(int irank = 0; irank < mpi_size; ++irank)
        {
            if(mpi_rank == irank)
            {
                std::cout << "out-brick rank " << irank;
                std::cout << "\n\tlower indices:";
                for(const auto val : outbrick_lower)
                    std::cout << " " << val;
                std::cout << "\n\tupper indices:";
                for(const auto val : outbrick_upper)
                    std::cout << " " << val;
                std::cout << "\n\tstrides:";
                for(const auto val : outbrick_stride)
                    std::cout << " " << val;
                std::cout << "\n";
                std::cout << "\tnbatch: " << nbatch << "\n";
                std::cout << "\tbuffer size: " << memSize << "\n";
            }
            MPI_Barrier(mpi_comm);
        }

        rocfft_field outfield = nullptr;
        rocfft_field_create(&outfield);

        rocfft_brick outbrick = nullptr;
        rocfft_brick_create(&outbrick,
                            outbrick_lower.data(),
                            outbrick_upper.data(),
                            outbrick_stride.data(),
                            outbrick_lower.size(),
                            0);
        rocfft_field_add_brick(outfield, outbrick);
        rocfft_brick_destroy(outbrick);
        outbrick = nullptr;

        hiprc = hipMalloc(&gpu_out[0], memSize);
        if(hiprc != hipSuccess)
            throw std::runtime_error("outbrick hipMalloc failed");

        rocfft_plan_description_add_outfield(description, outfield);

        fftrc = rocfft_field_destroy(outfield);
        if(fftrc != rocfft_status_success)
            throw std::runtime_error("failed destroy outfield");
    }

    // In order still handle non-success return codes without killing all of the MPI processes, we
    // put object creation in a try/catch block and destroy non-nullptr objects.

    // Serialize output:
    for(int irank = 0; irank < mpi_size; ++irank)
    {
        if(mpi_rank == irank)
        {
            std::cout << "rank " << irank << "\n";
            std::cout << "input ";
            for(const auto& b : gpu_in)
                std::cout << " " << b;
            std::cout << "\n";
            std::cout << "output ";
            for(const auto& b : gpu_out)
                std::cout << " " << b;
            std::cout << "\n";
        }
        MPI_Barrier(mpi_comm);
    }

    fftrc = rocfft_status_sync(fftrc, mpi_comm);
    hiprc = hip_status_sync(hiprc, mpi_comm);

    if(mpi_rank == 0)
    {
        if(fftrc == rocfft_status_success && hiprc == hipSuccess)
        {
            std::cout << "so far so good, trying to make a plan....\n";
        }
        else
        {
            std::cout << "failure: will not make a plan....\n";
        }
    }

    // Create a multi-process plan:
    rocfft_plan gpu_plan = nullptr;
    if(fftrc == rocfft_status_success && hiprc == hipSuccess)
    {
        fftrc = rocfft_plan_create(&gpu_plan,
                                   place,
                                   direction,
                                   rocfft_precision_double,
                                   length.size(), // Dimension
                                   length.data(), // lengths
                                   nbatch, // Number of transforms
                                   description); // Description
    }

    fftrc = rocfft_status_sync(fftrc, mpi_comm);
    if(mpi_rank == 0)
    {
        if(fftrc == rocfft_status_success)
        {
            std::cout << "so far so good, we have a plan....\n";
        }
        else
        {
            std::cout << "failure: we do not have a plan....\n";
        }
    }

    // Execute plan:
    if(fftrc == rocfft_status_success)
    {
        fftrc = rocfft_execute(gpu_plan, (void**)gpu_in.data(), (void**)gpu_out.data(), nullptr);
    }

    fftrc = rocfft_status_sync(fftrc, mpi_comm);
    if(mpi_rank == 0)
    {
        if(fftrc == rocfft_status_success)
        {
            std::cout << "The FFT was successful....\n";
        }
        else
        {
            std::cout << "The FFT execution failed....\n";
        }
    }

    // Copy back all batches; print only batch 0 for compactness
    for(int irank = 0; irank < mpi_size; ++irank)
    {
        if(mpi_rank == irank)
        {
            std::cout << "out brick rank " << irank << " (showing batch 0 of " << nbatch << ")\n";
            const size_t                      outcount = nbatch * outbrick_dist;
            std::vector<std::complex<double>> host_out(outcount);
            hiprc = hipMemcpy(host_out.data(),
                              gpu_out[0],
                              outcount * sizeof(std::complex<double>),
                              hipMemcpyDeviceToHost);
            if(hiprc != hipSuccess)
                throw std::runtime_error("hipMemcpy failed");
            const size_t batch0 = 0;
            for(auto idx1 = outbrick_lower[1]; idx1 < outbrick_upper[1]; ++idx1)
            {
                for(auto idx0 = outbrick_lower[0]; idx0 < outbrick_upper[0]; ++idx0)
                {
                    const auto pos = batch0 * outbrick_stride[2]
                                     + (idx0 - outbrick_lower[0]) * outbrick_stride[0]
                                     + (idx1 - outbrick_lower[1]) * outbrick_stride[1];
                    std::cout << host_out[pos] << " ";
                }
                std::cout << "\n";
            }
        }
        MPI_Barrier(mpi_comm);
    }

    // Cleanup anything plan-generation structs (that aren't null pointers):
    if(description != nullptr)
    {
        if(rocfft_plan_description_destroy(description) != rocfft_status_success)
        {
            std::cerr << "description destruction failed\n";
        }
        else
        {
            description = nullptr;
        }
    }

    // Clean up the plan and rocfft:
    try
    {
        if(gpu_plan != nullptr)
        {
            if(rocfft_plan_destroy(gpu_plan) != rocfft_status_success)
                throw std::runtime_error("rocfft_plan_destroy failed.");
            gpu_plan = nullptr;
        }
    }
    catch(const std::exception&)
    {
        std::cerr << "rank " << mpi_rank << " plan destroy failed\n";
    }

    for(auto& buf : gpu_in)
    {
        if(buf != nullptr)
        {
            hiprc = hipFree(buf);
            if(hiprc != hipSuccess)
                std::cerr << "hipFree failed\n";
            buf = nullptr;
        }
    }

    for(auto& buf : gpu_out)
    {
        if(buf != nullptr)
        {
            hiprc = hipFree(buf);
            if(hiprc != hipSuccess)
                std::cerr << "hipFree failed\n";
            buf = nullptr;
        }
    }

    fftrc = rocfft_cleanup();

    MPI_Finalize();

    return 0;
}
