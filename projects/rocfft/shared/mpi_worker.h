/******************************************************************************
* Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "fft_params.h"

#include "CLI11.hpp"
#include "gpubuf.h"
#include "hostbuf.h"
#include "ptrdiff.h"
#include "rocfft_against_fftw.h"
#include "rocfft_hip.h"
#include <chrono>
#include <mpi.h>

static MPI_Datatype get_mpi_type(size_t elem_size)
{
    MPI_Datatype mpi_type;
    // pick an MPI data type that matches the element size we need.
    // we're just copying data (no reductions or anything), so MPI
    // doesn't need to understand the data, just needs to know how
    // big each element is.
    //
    // elements could be either {real,complex} x {FP16,FP32,FP64}
    //
    // Real FP16
    if(elem_size == 2)
        mpi_type = MPI_UINT16_T;
    // Complex FP16 or real FP32
    else if(elem_size == 4)
        mpi_type = MPI_FLOAT;
    // Complex FP32 or real FP64
    else if(elem_size == 8)
        mpi_type = MPI_DOUBLE;
    // Complex FP64
    else if(elem_size == 16)
        mpi_type = MPI_C_DOUBLE_COMPLEX;
    else
        throw std::runtime_error("invalid element size");
    return mpi_type;
}

static size_t add_brick_elems(size_t val, const fft_params::fft_brick& b)
{
    return val + compute_ptrdiff(b.length(), b.stride, 0, 0);
}

// Test if any rank uses multiple devices.
static bool multiple_devices_on_rank(const std::vector<fft_params::fft_brick>& bricks)
{
    // Go over each rank's bricks
    for(auto range
        = std::equal_range(bricks.begin(), bricks.end(), bricks.front().rank, match_rank());
        range.first != range.second;
        range = std::equal_range(range.second, bricks.end(), range.second->rank, match_rank()))
    {
        // If we find a device on this rank that has a different device
        // from the first, then that means this rank is using multiple
        // devices.
        int first_device = range.first->device;
        if(std::any_of(
               range.first, range.second, [first_device](const fft_params::fft_brick& brick) {
                   return brick.device != first_device;
               }))
            return true;
    }
    // No rank had multiple devices
    return false;
}

// Gather a whole field to a host buffer on rank 0, using MPI_Gatherv.
// This is only possible if each rank's bricks are on a single device.
// local_bricks is the contiguous buffer allocated by
// alloc_local_bricks with all of the current rank's bricks.
static void gather_field_v(MPI_Comm                                  mpi_comm,
                           const std::vector<fft_params::fft_brick>& bricks,
                           const std::vector<size_t>&                field_stride,
                           size_t                                    field_dist,
                           const fft_precision                       precision,
                           const fft_array_type                      array_type,
                           std::map<int, gpubuf>&                    local_bricks,
                           hostbuf&                                  output)
{
    int mpi_rank = 0;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    int mpi_size = 0;
    MPI_Comm_size(mpi_comm, &mpi_size);

    auto elem_size = var_size<size_t>(precision, array_type);

    hostbuf recvbuf;

    // allocate buffer for rank 0 to run fftw
    if(mpi_rank == 0)
    {
        size_t field_elems = std::accumulate(
            bricks.begin(), bricks.end(), static_cast<size_t>(0), add_brick_elems);
        recvbuf.alloc(field_elems * elem_size);
    }

    // work out how much to receive from each rank and where
    std::vector<int> recvcounts(static_cast<size_t>(mpi_size));
    std::vector<int> displs(static_cast<size_t>(mpi_size));
    // loop over each rank's bricks
    size_t elem_total = 0;
    for(auto range
        = std::equal_range(bricks.begin(), bricks.end(), bricks.front().rank, match_rank());
        range.first != range.second;
        range = std::equal_range(range.second, bricks.end(), range.second->rank, match_rank()))
    {
        size_t current_rank = range.first->rank;
        size_t rank_elems
            = std::accumulate(range.first, range.second, static_cast<size_t>(0), add_brick_elems);
        recvcounts[current_rank] = rank_elems;
        displs[current_rank]     = elem_total;
        elem_total += rank_elems;
    }

    // gather brick(s) to rank 0 (to host memory)
    auto mpi_type = get_mpi_type(elem_size);

    MPI_Gatherv(local_bricks.empty() ? nullptr : local_bricks.begin()->second.data(),
                local_bricks.empty() ? 0 : local_bricks.begin()->second.size() / elem_size,
                mpi_type,
                recvbuf.data(),
                recvcounts.data(),
                displs.data(),
                mpi_type,
                0,
                mpi_comm);

    // data is gathered, but bricks need to be transposed to be in the right order
    if(mpi_rank == 0)
    {
        // go over each rank's bricks again
        size_t recv_idx = 0;
        for(auto range
            = std::equal_range(bricks.begin(), bricks.end(), bricks.front().rank, match_rank());
            range.first != range.second;
            range = std::equal_range(range.second, bricks.end(), range.second->rank, match_rank()),
            ++recv_idx)
        {
            auto rank_base
                = hostbuf::make_nonowned(recvbuf.data_offset(displs[recv_idx] * elem_size));
            size_t cur_brick_offset_bytes = 0;
            for(; range.first != range.second; ++range.first)
            {
                auto& brick           = *range.first;
                void* brick_read_ptr  = rank_base.data_offset(cur_brick_offset_bytes);
                void* brick_write_ptr = output.data_offset(
                    brick.lower_field_offset(field_stride, field_dist) * elem_size);

                std::vector<hostbuf> copy_in(1);
                std::vector<hostbuf> copy_out(1);
                copy_in.front()  = hostbuf::make_nonowned(brick_read_ptr);
                copy_out.front() = hostbuf::make_nonowned(brick_write_ptr);

                // separate batch length + stride for the sake of copy_buffers
                std::vector<size_t> brick_len_nobatch = brick.length();
                auto                brick_batch       = brick_len_nobatch.front();
                brick_len_nobatch.erase(brick_len_nobatch.begin());
                std::vector<size_t> brick_stride_nobatch = brick.stride;
                auto                brick_dist           = brick_stride_nobatch.front();
                brick_stride_nobatch.erase(brick_stride_nobatch.begin());

                copy_buffers(copy_in,
                             copy_out,
                             brick_len_nobatch,
                             brick_batch,
                             precision,
                             array_type,
                             brick_stride_nobatch,
                             brick_dist,
                             array_type,
                             field_stride,
                             field_dist,
                             {0},
                             {0});

                size_t brick_bytes
                    = compute_ptrdiff(brick.length(), brick.stride, 0, 0) * elem_size;
                cur_brick_offset_bytes += brick_bytes;
            }
        }
    }
}

// Gather a whole field to a host buffer on rank 0, using MPI
// point-to-point operations.  local_bricks is the contiguous buffer
// allocated by alloc_local_bricks with all of the current rank's
// bricks.
static void gather_field_p2p(MPI_Comm                                  mpi_comm,
                             const std::vector<fft_params::fft_brick>& bricks,
                             const std::vector<size_t>&                field_stride,
                             size_t                                    field_dist,
                             const fft_precision                       precision,
                             const fft_array_type                      array_type,
                             std::map<int, gpubuf>&                    local_bricks,
                             hostbuf&                                  output)
{
    int mpi_rank = 0;
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    auto elem_size = var_size<size_t>(precision, array_type);
    auto mpi_type  = get_mpi_type(elem_size);

    // map device -> offset, to keep track of the offset of each
    // brick in the per-device buffers
    std::map<int, size_t> offsets;

    for(unsigned int i = 0; i < bricks.size(); ++i)
    {
        const auto& brick       = bricks[i];
        size_t      brick_elems = compute_ptrdiff(brick.length(), brick.stride, 0, 0);

        // The rank that this brick is on needs to send to rank 0,
        // and rank 0 needs to receive all bricks
        if(brick.rank != mpi_rank && mpi_rank != 0)
            continue;

        void* brick_ptr   = nullptr;
        auto  local_brick = local_bricks.find(brick.device);
        if(local_brick != local_bricks.end())
        {
            // get pointer to this brick in the per-device buffer
            auto& cur_offset = offsets.emplace(brick.device, static_cast<size_t>(0)).first->second;
            brick_ptr        = local_brick->second.data_offset(cur_offset);
            cur_offset += brick_elems * elem_size;
        }

        // rank 0 needs to receive the data
        hostbuf recvbuf;
        if(mpi_rank == 0)
            recvbuf.alloc(brick_elems * elem_size);

        if(brick.rank == 0)
        {
            if(mpi_rank == 0)
            {
                rocfft_scoped_device dev(brick.device);
                // Data is already on a rank-0 local device, just memcpy it
                //
                // Ignore error as we don't want to hang collective
                // operations, but we will notice accuracy test
                // problems if this fails.
                (void)hipMemcpy(
                    recvbuf.data(), brick_ptr, brick_elems * elem_size, hipMemcpyDeviceToHost);
            }
        }
        else
        {
            // otherwise, brick is on another rank and needs to be
            // communicated via Send/Recv

            if(mpi_rank == 0)
            {
                // Receive this brick to rank 0
                MPI_Recv(recvbuf.data(),
                         static_cast<int>(brick_elems),
                         mpi_type,
                         brick.rank,
                         i,
                         mpi_comm,
                         MPI_STATUS_IGNORE);
            }
            else if(mpi_rank == brick.rank)
            {
                // Send this brick to rank 0
                rocfft_scoped_device dev(brick.device);
                MPI_Send(brick_ptr, static_cast<int>(brick_elems), mpi_type, 0, i, mpi_comm);
            }
        }

        if(mpi_rank == 0)
        {
            // Brick is now local, transpose to the output buf
            void* brick_write_ptr = output.data_offset(
                brick.lower_field_offset(field_stride, field_dist) * elem_size);

            std::vector<hostbuf> copy_in(1);
            std::vector<hostbuf> copy_out(1);
            copy_in.front()  = hostbuf::make_nonowned(recvbuf.data());
            copy_out.front() = hostbuf::make_nonowned(brick_write_ptr);

            // separate batch length + stride for the sake of copy_buffers
            std::vector<size_t> brick_len_nobatch = brick.length();
            auto                brick_batch       = brick_len_nobatch.front();
            brick_len_nobatch.erase(brick_len_nobatch.begin());
            std::vector<size_t> brick_stride_nobatch = brick.stride;
            auto                brick_dist           = brick_stride_nobatch.front();
            brick_stride_nobatch.erase(brick_stride_nobatch.begin());

            copy_buffers(copy_in,
                         copy_out,
                         brick_len_nobatch,
                         brick_batch,
                         precision,
                         array_type,
                         brick_stride_nobatch,
                         brick_dist,
                         array_type,
                         field_stride,
                         field_dist,
                         {0},
                         {0});
        }
    }
}

// Gather a whole field to a host buffer on rank 0.
static void gather_field(MPI_Comm                                  mpi_comm,
                         const std::vector<fft_params::fft_brick>& bricks,
                         const std::vector<size_t>&                field_stride,
                         size_t                                    field_dist,
                         const fft_precision                       precision,
                         const fft_array_type                      array_type,
                         std::map<int, gpubuf>&                    local_bricks,
                         hostbuf&                                  output)
{
    if(multiple_devices_on_rank(bricks))
    {
        // Can't do MPI_Gather, as MPI assumes we only have one pointer per rank
        gather_field_p2p(mpi_comm,
                         bricks,
                         field_stride,
                         field_dist,
                         precision,
                         array_type,
                         local_bricks,
                         output);
    }
    else
    {
        // Do gather, which is more efficient
        gather_field_v(mpi_comm,
                       bricks,
                       field_stride,
                       field_dist,
                       precision,
                       array_type,
                       local_bricks,
                       output);
    }
}

// Allocate device buffer(s) to hold all of the bricks for this rank.
// A rank can have N bricks on it but this will allocate one
// contiguous buffer per device and return pointers to each of the N bricks.
static void alloc_local_bricks(int                                       mpi_rank,
                               const std::vector<fft_params::fft_brick>& bricks,
                               size_t                                    elem_size,
                               std::map<int, gpubuf>&                    buffers,
                               std::vector<void*>&                       buffer_ptrs)
{
    // Get bricks that are local to this rank
    auto local_range = std::equal_range(bricks.begin(), bricks.end(), mpi_rank, match_rank());

    // Do one pass over these bricks to work out how big of a buffer
    // we need to allocate on each device
    std::map<int, size_t> buffer_sizes;
    for(auto brick = local_range.first; brick != local_range.second; ++brick)
    {
        buffer_sizes.insert({brick->device, static_cast<size_t>(0)}).first->second
            += compute_ptrdiff(brick->length(), brick->stride, 0, 0);
    }

    // Alloc buffers for each device
    for(const auto buffer_size : buffer_sizes)
    {
        rocfft_scoped_device dev(buffer_size.first);
        if(buffers.emplace(buffer_size.first, gpubuf{})
               .first->second.alloc(buffer_size.second * elem_size)
           != hipSuccess)
        {
            throw std::runtime_error("Failed to allocate buffer on device "
                                     + std::to_string(buffer_size.first));
        }
    }

    // Return pointers for each brick
    for(auto brick = local_range.first; brick != local_range.second; ++brick)
    {
        auto& buf = buffers[brick->device];

        // Use buffer_sizes to count down bricks for each device
        auto& remaining_size = buffer_sizes[brick->device];
        auto  offset_elems   = (buf.size() / elem_size) - remaining_size;
        remaining_size -= compute_ptrdiff(brick->length(), brick->stride, 0, 0);

        buffer_ptrs.push_back(buf.data_offset(offset_elems * elem_size));
    }
}

template <typename Tfloat>
void execute_reference_fft(const fft_params& params, std::vector<hostbuf>& input)
{
    auto cpu_plan = fftw_plan_via_rocfft<Tfloat>(params.length,
                                                 params.istride,
                                                 params.ostride,
                                                 params.nbatch,
                                                 params.idist,
                                                 params.odist,
                                                 params.transform_type,
                                                 input,
                                                 input);

    fftw_run<Tfloat>(params.transform_type, cpu_plan, input, input);

    fftw_destroy_plan_type(cpu_plan);
}

bool   use_fftw_wisdom = false;
double half_epsilon    = default_half_epsilon();
double single_epsilon  = default_single_epsilon();
double double_epsilon  = default_double_epsilon();

// execute the specific number of trials on a vec of libraries
template <typename AllParams>
void exec_testcases(std::function<AllParams(const std::vector<std::string>&)> make_params,
                    MPI_Comm                                                  mpi_comm,
                    int                                                       mpi_rank,
                    bool                                                      run_bench,
                    bool                                                      run_fftw,
                    int                                                       test_sequence,
                    const std::string&                                        token,
                    const std::vector<std::string>&                           lib_strings,
                    std::vector<std::vector<double>>&                         gpu_time,
                    std::map<int, gpubuf>&                                    local_inputs,
                    std::vector<void*>&                                       local_input_ptrs,
                    std::map<int, gpubuf>&                                    local_outputs,
                    std::vector<void*>&                                       local_output_ptrs,
                    size_t                                                    ntrial)
{
    auto all_params = make_params(lib_strings);

    for(auto& p : all_params)
    {
        p.setup();

        p.from_token(token);
        p.validate();
        p.ifields.front().stable_sort_by_rank();
        p.ofields.front().stable_sort_by_rank();

        p.mp_lib  = fft_params::fft_mp_lib_mpi;
        p.mp_comm = &mpi_comm;
    }

    // allocate library test order in advance so we can communicate
    // it to all ranks
    std::vector<size_t> testcases;
    testcases.reserve(ntrial * lib_strings.size());

    if(mpi_rank == 0)
    {
        switch(test_sequence)
        {
        case 0:
        case 2:
        {
            // Random and sequential order:
            for(size_t ilib = 0; ilib < lib_strings.size(); ++ilib)
                std::fill_n(std::back_inserter(testcases), ntrial, ilib);
            if(test_sequence == 0)
            {
                std::random_device rd;
                std::mt19937       g(rd());
                std::shuffle(testcases.begin(), testcases.end(), g);
            }
            break;
        }
        case 1:
            // Alternating order:
            for(size_t itrial = 0; itrial < ntrial; ++itrial)
            {
                for(size_t ilib = 0; ilib < lib_strings.size(); ++ilib)
                {
                    testcases.push_back(ilib);
                }
            }
            break;
        default:
            throw std::runtime_error("Invalid test sequence choice.");
        }
    }

    // for all other ranks, resize to what rank 0 has, in preparation
    // for receiving the test case order
    testcases.resize(ntrial * lib_strings.size());

    // send test case order from rank 0 to all ranks
    MPI_Bcast(testcases.data(), testcases.size(), MPI_UINT64_T, 0, mpi_comm);

    // use first params to know things like precision, type that
    // won't change between libraries
    const auto& params        = all_params.front();
    const auto  in_elem_size  = var_size<size_t>(params.precision, params.itype);
    const auto  out_elem_size = var_size<size_t>(params.precision, params.otype);

    // allocate and initialize input buffers
    alloc_local_bricks(
        mpi_rank, params.ifields.back().bricks, in_elem_size, local_inputs, local_input_ptrs);

    init_local_input<decltype(params), gpubuf>(
        mpi_rank, params, params.ifields.back().bricks, in_elem_size, local_input_ptrs);

    // gather input for FFTW before we transform, in case we're doing an in-place FFT
    std::vector<hostbuf> cpu_data(1);
    if(run_fftw)
    {
        if(mpi_rank == 0)
            cpu_data.front().alloc(std::max(params.isize.front() * in_elem_size,
                                            params.osize.front() * out_elem_size));

        gather_field(mpi_comm,
                     params.ifields.front().bricks,
                     params.istride,
                     params.idist,
                     params.precision,
                     params.itype,
                     local_inputs,
                     cpu_data.front());
    }

    // if this is not an in-place transform, then allocate output buffers
    if(params.placement == fft_placement_inplace)
    {
        local_output_ptrs = local_input_ptrs;
    }
    else
    {
        alloc_local_bricks(mpi_rank,
                           params.ofields.back().bricks,
                           out_elem_size,
                           local_outputs,
                           local_output_ptrs);
    }

    // execute FFTs
    std::chrono::time_point<std::chrono::steady_clock> start, stop;

    // call rocfft_plan_create
    for(auto& p : all_params)
        p.create_plan();

    for(size_t i = 0; i < testcases.size(); ++i)
    {
        size_t testcase = testcases[i];

        if(run_bench)
        {
            if(i > 0)
            {
                init_local_input<decltype(params), gpubuf>(
                    mpi_rank, params, params.ifields.back().bricks, in_elem_size, local_input_ptrs);
            }

            (void)hipDeviceSynchronize();
            MPI_Barrier(mpi_comm);

            start = std::chrono::steady_clock::now();
        }

        all_params[testcase].execute(reinterpret_cast<void**>(local_input_ptrs.data()),
                                     reinterpret_cast<void**>(local_output_ptrs.data()));

        if(run_bench)
        {
            (void)hipDeviceSynchronize();
            stop                                              = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> diff    = stop - start;
            double                                    diff_ms = diff.count();

            double max_diff_ms = 0.0;
            MPI_Reduce(&diff_ms, &max_diff_ms, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);

            if(mpi_rank == 0)
                gpu_time[testcase].push_back(max_diff_ms);
        }
    }

    // FFTW Validation
    if(run_fftw)
    {
        std::vector<hostbuf> gpu_output(1);
        VectorNorms          cpu_output_norm;

        if(mpi_rank == 0)
        {
            fft_params params_inplace = params;
            params_inplace.placement  = fft_placement_inplace;

            switch(params_inplace.precision)
            {
            case fft_precision_half:
                execute_reference_fft<rocfft_fp16>(params_inplace, cpu_data);
                break;
            case fft_precision_single:
                execute_reference_fft<float>(params_inplace, cpu_data);
                break;
            case fft_precision_double:
                execute_reference_fft<double>(params_inplace, cpu_data);
                break;
            }

            cpu_output_norm = norm(cpu_data,
                                   params_inplace.ilength(),
                                   params_inplace.nbatch,
                                   params_inplace.precision,
                                   params_inplace.itype,
                                   params_inplace.istride,
                                   params_inplace.idist,
                                   params_inplace.ioffset);
        }

        if(mpi_rank == 0)
            gpu_output.front().alloc(params.osize.front() * out_elem_size);

        gather_field(mpi_comm,
                     params.ofields.front().bricks,
                     params.ostride,
                     params.odist,
                     params.precision,
                     params.otype,
                     params.placement == fft_placement_inplace ? local_inputs : local_outputs,
                     gpu_output.front());

        if(mpi_rank == 0)
        {
            // Compare data to reference implementation
            const double linf_cutoff = type_epsilon(params.precision) * cpu_output_norm.l_inf
                                       * log(product(params.length.begin(), params.length.end()));

            std::vector<std::pair<size_t, size_t>> linf_failures;

            auto diff = distance(cpu_data,
                                 gpu_output,
                                 params.olength(),
                                 params.nbatch,
                                 params.precision,
                                 params.otype,
                                 params.ostride,
                                 params.odist,
                                 params.otype,
                                 params.ostride,
                                 params.odist,
                                 &linf_failures,
                                 linf_cutoff,
                                 params.ooffset,
                                 params.ooffset);

            if(diff.l_inf > linf_cutoff)
            {
                std::stringstream msg;
                msg << "linf diff " << diff.l_inf << " exceeds cutoff " << linf_cutoff;
                throw std::runtime_error(msg.str());
            }
            if(diff.l_2 > cpu_output_norm.l_2)
            {
                std::stringstream msg;
                msg << "l_2 diff " << diff.l_2 << " exceeds input norm l_2 " << cpu_output_norm.l_2;
                throw std::runtime_error(msg.str());
            }
        }
    }

    for(auto& p : all_params)
    {
        p.free();
        p.cleanup();
    }
}

// returns final grid integrating inter-process and intra-process grids
std::vector<unsigned int> compute_final_grid(const std::vector<unsigned int>& mpi_grid,
                                             const std::vector<unsigned int>& intra_grid)
{
    std::vector<unsigned int> final_grid(mpi_grid.size());
    for(size_t i = 0; i < mpi_grid.size(); ++i)
    {
        final_grid[i] = mpi_grid[i] * intra_grid[i];
    }
    return final_grid;
}

// AllParams is a callable that returns a container of fft_params
// structs to test.  It accepts a vector of library strings, which
// "dyna" workers will turn into params that load the specified
// libraries.  Non-"dyna" workers return fft_params for the library
// that's linked in.
template <typename AllParams, bool dyna_load_libs>
int mpi_worker_main(const char*                                               description,
                    int                                                       argc,
                    char*                                                     argv[],
                    std::function<AllParams(const std::vector<std::string>&)> make_params)
{
    MPI_Init(&argc, &argv);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_set_errhandler(mpi_comm, MPI_ERRORS_ARE_FATAL);

    int mpi_rank = 0;
    int mp_size;

    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mp_size);

    CLI::App    app{description};
    size_t      ntrial = 1;
    std::string token;

    // Test sequence choice:
    int test_sequence = 0;

    // Vector of test target libraries
    std::vector<std::string> lib_strings;

    // Bool to specify whether the libs are loaded in forward or forward+reverse order.
    int reverse{};

    bool run_fftw  = false;
    bool run_bench = false;
    auto bench_flag
        = app.add_flag("--benchmark", run_bench, "Benchmark a specified number of MPI transforms");
    app.add_option("-N, --ntrial", ntrial, "Number of trials to benchmark")
        ->default_val(1)
        ->check(CLI::PositiveNumber);
    app.add_flag("--accuracy", run_fftw, "Check accuracy of an MPI transform")
        ->excludes(bench_flag);

    CLI::Option* opt_token
        = app.add_option("--token", token, "Problem token to test/benchmark")->default_val("");

    app.add_option(
           "--sequence", test_sequence, "Test sequence:\n0) random\n1) alternating\n2) sequential")
        ->check(CLI::Range(0, 2))
        ->default_val(0);
    if(dyna_load_libs)
    {
        app.add_option("--lib", lib_strings, "Set test target library full path (appendable)")
            ->required();
        app.add_flag("--reverse", reverse, "Load libs in forward and reverse order")
            ->default_val(1);
    }
    else
    {
        // add a library string to represent the rocFFT that we are
        // linked to
        lib_strings.push_back("");
    }

    // read benchmark parameters
    fft_params params;

    params.mp_lib = fft_params::fft_mp_lib_mpi;

    // Control output verbosity:
    int verbose{};

    // input/output FFT grids
    std::vector<unsigned int> imgrid;
    std::vector<unsigned int> omgrid;

    // input/output GPU grids per process
    std::vector<unsigned int> ingrid;
    std::vector<unsigned int> outgrid;

    // number of GPUs to use per rank
    int ngpus{};

    auto* non_token = app.add_option_group("Token Conflict", "Options excluded by --token");
    non_token
        ->add_flag("--double", "Double precision transform (deprecated: use --precision double)")
        ->each([&](const std::string&) { params.precision = fft_precision_double; });
    non_token->excludes(opt_token);
    non_token
        ->add_option("-t, --transformType",
                     params.transform_type,
                     "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
                     "forward\n3) real inverse")
        ->default_val(fft_transform_type_complex_forward)
        ->check(CLI::Range(0, 4));

    non_token
        ->add_option(
            "--precision", params.precision, "Transform precision: single (default), double, half")
        ->excludes("--double");

    CLI::Option* opt_not_in_place
        = non_token->add_flag("-o, --notInPlace", "Not in-place FFT transform (default: in-place)")
              ->each([&](const std::string&) { params.placement = fft_placement_notinplace; });
    non_token
        ->add_option("--itype",
                     params.itype,
                     "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
                     "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset)
        ->check(CLI::Range(0, 4));

    non_token
        ->add_option("--otype",
                     params.otype,
                     "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
                     "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset)
        ->check(CLI::Range(0, 4));

    CLI::Option* opt_length
        = non_token->add_option("--length", params.length, "Lengths")->required()->expected(1, 3);

    non_token->add_option("--imgrid", imgrid, "Input multi-processes grid")->expected(1, 3);

    non_token->add_option("--omgrid", omgrid, "Output multi-processes grid")->expected(1, 3);

    non_token
        ->add_option("-b, --batchSize",
                     params.nbatch,
                     "If this value is greater than one, arrays will be used")
        ->default_val(1);

    // set number of GPUs to user per MPI rank
    non_token->add_option("--ngpus", ngpus, "Number of GPUs to use per rank")
        ->default_val(1)
        ->check(CLI::NonNegativeNumber);

    // define multi-GPU grids per process
    non_token->add_option("--ingrid", ingrid, "Single-process grid of GPUs at input")
        ->expected(1, 3)
        ->needs("--ngpus");

    non_token->add_option("--outgrid", outgrid, "Single-process grid of GPUs at output")
        ->expected(1, 3)
        ->needs("--ngpus");

    CLI::Option* opt_istride = non_token->add_option("--istride", params.istride, "Input strides");
    CLI::Option* opt_ostride = non_token->add_option("--ostride", params.ostride, "Output strides");

    non_token->add_option("--idist", params.idist, "Logical distance between input batches")
        ->default_val(0)
        ->each([&](const std::string& val) { std::cout << "idist: " << val << "\n"; });
    non_token->add_option("--odist", params.odist, "Logical distance between output batches")
        ->default_val(0)
        ->each([&](const std::string& val) { std::cout << "odist: " << val << "\n"; });

    CLI::Option* opt_ioffset = non_token->add_option("--ioffset", params.ioffset, "Input offset");
    CLI::Option* opt_ooffset = non_token->add_option("--ooffset", params.ooffset, "Output offset");

    app.add_option("-g, --inputGen",
                   params.igen,
                   "Input data generation:\n0) PRNG sequence (device)\n"
                   "1) PRNG sequence (host)\n"
                   "2) linearly-spaced sequence (device)\n"
                   "3) linearly-spaced sequence (host)")
        ->default_val(fft_input_random_generator_device)
        ->check(CLI::Range(0, 3));

    app.add_option("--isize", params.isize, "Logical size of input buffer");
    app.add_option("--osize", params.osize, "Logical size of output buffer");
    app.add_option("--scalefactor", params.scale_factor, "Scale factor to apply to output");

    app.add_flag("--verbose", verbose, "Control output verbosity")->default_val(0);

    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    if(token.empty())
    {
        // set default multi-process grids in case none were given
        params.set_default_grid(mp_size, imgrid, omgrid);

        // set default GPU grids per process
        params.set_default_grid(ngpus, ingrid, outgrid);

        // start with all-ones in grids
        std::vector<unsigned int> input_grid(params.length.size() + 1, 1);
        std::vector<unsigned int> output_grid(params.length.size() + 1, 1);

        // sanity checks
        int imgrid_size = product(imgrid.begin(), imgrid.end());
        int omgrid_size = product(omgrid.begin(), omgrid.end());

        int ingrid_size  = product(ingrid.begin(), ingrid.end());
        int outgrid_size = product(outgrid.begin(), outgrid.end());

        if((imgrid.size() != params.length.size()) || (omgrid.size() != params.length.size())
           || (ingrid.size() != params.length.size()) || (outgrid.size() != params.length.size()))
        {
            throw std::runtime_error(
                "grid of processors and GPUs must be of the same size as the problem dimension!");
        }

        if((imgrid_size != mp_size) || (omgrid_size != mp_size))
        {
            throw std::runtime_error(
                "size of grid of processors must be equal to the number of MPI resources!");
        }

        if((ingrid_size != ngpus) || (outgrid_size != ngpus))
        {
            throw std::runtime_error("size of grid of GPUs per process must be equal to ngpus!");
        }

        // create input and output grids and distribute it according to user requirements
        std::vector<unsigned int> final_input_grid, final_output_grid;
        final_input_grid  = compute_final_grid(imgrid, ingrid);
        final_output_grid = compute_final_grid(omgrid, outgrid);

        std::copy(final_input_grid.begin(), final_input_grid.end(), input_grid.begin() + 1);
        std::copy(final_output_grid.begin(), final_output_grid.end(), output_grid.begin() + 1);

        // get number of nodes to asign local GPU indexing, since within
        // each node, GPUs are indexed 0,1,...,N

        // distribute input and output among the available number of ranks and GPUs per rank
        params.distribute_input(ngpus, input_grid, mp_size);
        params.distribute_output(ngpus, output_grid, mp_size);

        params.validate();
        token = params.token();

        if(verbose && mpi_rank == 0)
        {
            if(*opt_not_in_place)
            {
                std::cout << "out-of-place\n";
            }
            else
            {
                std::cout << "in-place\n";
            }

            if(*opt_length)
            {
                std::cout << "length:";
                for(auto& i : params.length)
                    std::cout << " " << i;
                std::cout << "\n";
            }

            if(*opt_istride)
            {
                std::cout << "istride:";
                for(auto& i : params.istride)
                    std::cout << " " << i;
                std::cout << "\n";
            }
            if(*opt_ostride)
            {
                std::cout << "ostride:";
                for(auto& i : params.ostride)
                    std::cout << " " << i;
                std::cout << "\n";
            }

            if(*opt_ioffset)
            {
                std::cout << "ioffset:";
                for(auto& i : params.ioffset)
                    std::cout << " " << i;
                std::cout << "\n";
            }
            if(*opt_ooffset)
            {
                std::cout << "ooffset:";
                for(auto& i : params.ooffset)
                    std::cout << " " << i;
                std::cout << "\n";
            }

            // print multi-processes grids
            std::cout << "input grid :";
            for(auto& i : imgrid)
                std::cout << " " << i;
            std::cout << "\n";

            if(!ingrid.empty())
            {
                std::cout << "\tGPU grid:";
                for(auto& i : ingrid)
                    std::cout << " " << i;
                std::cout << "\n";
            }

            std::cout << "output grid:";
            for(auto& i : omgrid)
                std::cout << " " << i;
            std::cout << "\n";

            if(!outgrid.empty())
            {
                std::cout << "\tGPU grid:";
                for(auto& i : outgrid)
                    std::cout << " " << i;
                std::cout << "\n";
            }

            std::cout << "\n";
        }

        if(mpi_rank == 0)
        {
            std::cout << "Token: " << token << std::endl;
            std::cout << "\n";

            std::cout << std::flush;
        }
    }

    // Resize buffers based on the number of GPUs assigned
    std::map<int, gpubuf> local_inputs;
    std::vector<void*>    local_input_ptrs;
    std::map<int, gpubuf> local_outputs;
    std::vector<void*>    local_output_ptrs;

    // timing results - for each lib, store a vector of measured
    // times
    std::vector<std::vector<double>> gpu_time;
    gpu_time.resize(lib_strings.size());

    // if reversing, cut runs in half for first pass
    size_t ntrial_pass1 = reverse ? (ntrial + 1) / 2 : ntrial;
    exec_testcases(make_params,
                   mpi_comm,
                   mpi_rank,
                   run_bench,
                   run_fftw,
                   test_sequence,
                   token,
                   lib_strings,
                   gpu_time,
                   local_inputs,
                   local_input_ptrs,
                   local_outputs,
                   local_output_ptrs,
                   ntrial_pass1);
    if(reverse)
    {
        size_t ntrial_pass2 = ntrial / 2;

        // do libraries in reverse order
        std::reverse(lib_strings.begin(), lib_strings.end());
        std::reverse(gpu_time.begin(), gpu_time.end());
        exec_testcases(make_params,
                       mpi_comm,
                       mpi_rank,
                       run_bench,
                       run_fftw,
                       test_sequence,
                       token,
                       lib_strings,
                       gpu_time,
                       local_inputs,
                       local_input_ptrs,
                       local_outputs,
                       local_output_ptrs,
                       ntrial_pass2);
        // put back to normal order
        std::reverse(lib_strings.begin(), lib_strings.end());
        std::reverse(gpu_time.begin(), gpu_time.end());
    }

    if(run_bench && mpi_rank == 0)
    {
        for(auto& times : gpu_time)
        {
            std::cout << "Max rank time:";
            for(auto i : times)
                std::cout << " " << i;
            std::cout << " ms" << std::endl;

            std::sort(times.begin(), times.end());
            // print median
            double median;
            // average the two times around the middle
            if(ntrial % 2 && ntrial > 1)
                median = (times[ntrial / 2] + times[(ntrial + 1) / 2]) / 2;
            else
                median = times[ntrial / 2];
            std::cout << "Median: " << median << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
