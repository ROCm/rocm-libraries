/******************************************************************************
* Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

// functor to search for bricks on a rank, in a container of bricks
// sorted by rank
struct match_rank
{
    bool operator()(const fft_params::fft_brick& b, int rank) const
    {
        return b.rank < rank;
    }
    bool operator()(int rank, const fft_params::fft_brick& b) const
    {
        return rank < b.rank;
    }
};

// Initialize input for the bricks on the current rank.
template <typename Tparams>
void init_local_input(MPI_Comm                  mpi_comm,
                      const Tparams&            params,
                      size_t                    elem_size,
                      const std::vector<void*>& input_ptrs)
{
    int mpi_rank = 0;
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    // get bricks for this rank
    auto range = std::equal_range(params.ifields.front().bricks.begin(),
                                  params.ifields.front().bricks.end(),
                                  mpi_rank,
                                  match_rank());

    size_t ptr_idx = 0;
    for(auto brick = range.first; brick != range.second; ++brick, ++ptr_idx)
    {
        // some utility code below needs batch separated from brick lengths
        std::vector<size_t> brick_len_nobatch = brick->length();
        auto                brick_batch       = brick_len_nobatch.front();
        brick_len_nobatch.erase(brick_len_nobatch.begin());
        std::vector<size_t> brick_stride_nobatch = brick->stride;
        auto                brick_dist           = brick_stride_nobatch.front();
        brick_stride_nobatch.erase(brick_stride_nobatch.begin());
        std::vector<size_t> brick_lower_nobatch = brick->lower;
        auto                brick_lower_batch   = brick_lower_nobatch.front();
        brick_lower_nobatch.erase(brick_lower_nobatch.begin());

        auto contiguous_stride = params.compute_stride(params.ilength());
        auto contiguous_dist   = params.compute_idist();

        std::vector<gpubuf> bufvec(1);
        bufvec.back() = gpubuf::make_nonowned(
            input_ptrs[ptr_idx], compute_ptrdiff(brick->length(), brick->stride, 0, 0) * elem_size);

        // generate data (in device mem)
        switch(params.precision)
        {
        case fft_precision_half:
            set_input<gpubuf, rocfft_fp16>(bufvec,
                                           fft_input_random_generator_device,
                                           params.itype,
                                           brick_len_nobatch,
                                           brick_len_nobatch,
                                           brick_stride_nobatch,
                                           brick_dist,
                                           brick_batch,
                                           get_curr_device_prop(),
                                           brick_lower_nobatch,
                                           brick_lower_batch,
                                           contiguous_stride,
                                           contiguous_dist);
            break;
        case fft_precision_single:
            set_input<gpubuf, float>(bufvec,
                                     fft_input_random_generator_device,
                                     params.itype,
                                     brick_len_nobatch,
                                     brick_len_nobatch,
                                     brick_stride_nobatch,
                                     brick_dist,
                                     brick_batch,
                                     get_curr_device_prop(),
                                     brick_lower_nobatch,
                                     brick_lower_batch,
                                     contiguous_stride,
                                     contiguous_dist);
            break;
        case fft_precision_double:
            set_input<gpubuf, double>(bufvec,
                                      fft_input_random_generator_device,
                                      params.itype,
                                      brick_len_nobatch,
                                      brick_len_nobatch,
                                      brick_stride_nobatch,
                                      brick_dist,
                                      brick_batch,
                                      get_curr_device_prop(),
                                      brick_lower_nobatch,
                                      brick_lower_batch,
                                      contiguous_stride,
                                      contiguous_dist);
            break;
        }
    }
}

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

// Gather a whole field to a host buffer on rank 0.  local_bricks is
// the contiguous buffer allocated by alloc_local_bricks with all of
// the current rank's bricks.
static void gather_field(MPI_Comm                                  mpi_comm,
                         const std::vector<fft_params::fft_brick>& bricks,
                         const std::vector<size_t>&                field_stride,
                         size_t                                    field_dist,
                         const fft_precision                       precision,
                         const fft_array_type                      array_type,
                         gpubuf&                                   local_bricks,
                         hostbuf&                                  output)
{
    int mpi_rank = 0;
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    auto elem_size = var_size<size_t>(precision, array_type);

    hostbuf recvbuf;

    // allocate buffer for rank 0 to run fftw
    if(mpi_rank == 0)
    {
        size_t field_elems = std::accumulate(bricks.begin(), bricks.end(), 0UL, add_brick_elems);
        recvbuf.alloc(field_elems * elem_size);
    }

    // work out how much to receive from each rank and where
    std::vector<int> recvcounts;
    std::vector<int> displs;
    // loop over each rank's bricks
    size_t elem_total = 0;
    for(auto range
        = std::equal_range(bricks.begin(), bricks.end(), bricks.front().rank, match_rank());
        range.first != range.second;
        range = std::equal_range(range.second, bricks.end(), range.second->rank, match_rank()))
    {
        size_t rank_elems = std::accumulate(range.first, range.second, 0UL, add_brick_elems);
        recvcounts.push_back(rank_elems);
        displs.push_back(elem_total);
        elem_total += rank_elems;
    }

    // gather brick(s) to rank 0 (to host memory)
    auto mpi_type = get_mpi_type(elem_size);

    MPI_Gatherv(local_bricks.data(),
                local_bricks.size() / elem_size,
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

// Allocate a device buffer to hold all of the bricks for this rank.
// A rank can have N bricks on it but this will allocate one
// contiguous buffer and return pointers to each of the N bricks.
static void alloc_local_bricks(MPI_Comm                                  mpi_comm,
                               const std::vector<fft_params::fft_brick>& bricks,
                               size_t                                    elem_size,
                               gpubuf&                                   buffer,
                               std::vector<void*>&                       buffer_ptrs)
{
    int mpi_rank = 0;
    MPI_Comm_rank(mpi_comm, &mpi_rank);

    auto range = std::equal_range(bricks.begin(), bricks.end(), mpi_rank, match_rank());

    // get ptrdiff (i.e. length to alloc) of all bricks on this rank
    std::vector<size_t> brick_ptrdiffs_bytes;
    for(auto b = range.first; b != range.second; ++b)
        brick_ptrdiffs_bytes.push_back(compute_ptrdiff(b->length(), b->stride, 0, 0) * elem_size);

    size_t alloc_length_bytes
        = std::accumulate(brick_ptrdiffs_bytes.begin(), brick_ptrdiffs_bytes.end(), 0);

    if(buffer.alloc(alloc_length_bytes) != hipSuccess)
        throw std::runtime_error("failed to alloc brick");

    // return pointers to the bricks
    size_t cur_offset_bytes = 0;
    for(auto len : brick_ptrdiffs_bytes)
    {
        buffer_ptrs.push_back(buffer.data_offset(cur_offset_bytes));
        cur_offset_bytes += len;
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

// get the device that the field's bricks are on, for this rank.
// throws std::runtime_error if bricks for this rank are on multiple
// devices since that's not something we currently handle.
static int get_field_device(int mpi_rank, const fft_params::fft_field& field)
{
    // get first brick on this rank
    auto first
        = std::find_if(field.bricks.begin(),
                       field.bricks.end(),
                       [mpi_rank](const fft_params::fft_brick& b) { return b.rank == mpi_rank; });

    if(first == field.bricks.end())
        return true;

    int first_device = first->device;

    // check if remaining bricks are either not on this rank or on
    // the same device
    if(std::all_of(
           first, field.bricks.end(), [mpi_rank, first_device](const fft_params::fft_brick& b) {
               return b.rank != mpi_rank || b.device == first_device;
           }))
        return first_device;
    throw std::runtime_error("field spans multiple devices");
}

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
                    gpubuf&                                                   local_input,
                    std::vector<void*>&                                       local_input_ptrs,
                    gpubuf&                                                   local_output,
                    std::vector<void*>&                                       local_output_ptrs,
                    size_t                                                    ntrial)
{
    auto all_params = make_params(lib_strings);

    for(auto& p : all_params)
    {
        p.setup();

        p.from_token(token);
        p.validate();
        p.ifields.front().sort_by_rank();
        p.ofields.front().sort_by_rank();

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

    // currently, MPI worker requires that any rank only uses a
    // single device
    int input_device  = get_field_device(mpi_rank, params.ifields.front());
    int output_device = get_field_device(mpi_rank, params.ifields.front());
    if(input_device != output_device)
        throw std::runtime_error("input field uses different device from output field");

    rocfft_scoped_device dev(input_device);

    // check accuracy vs. FFTW - we only do this if FFTW actually ran
    // on this iteration
    bool                 check_fftw = false;
    std::vector<hostbuf> cpu_data(1);
    VectorNorms          cpu_output_norm;

    // allocate input/output if it hasn't been allocated already
    if(local_input_ptrs.empty())
    {
        alloc_local_bricks(
            mpi_comm, params.ifields.back().bricks, in_elem_size, local_input, local_input_ptrs);
        init_local_input(mpi_comm, params, in_elem_size, local_input_ptrs);

        // allocate local output bricks
        if(params.placement == fft_placement_inplace)
        {
            local_output_ptrs = local_input_ptrs;
        }
        else
        {
            alloc_local_bricks(mpi_comm,
                               params.ofields.back().bricks,
                               out_elem_size,
                               local_output,
                               local_output_ptrs);
        }

        if(run_fftw)
        {
            check_fftw = true;
            if(mpi_rank == 0)
                cpu_data.front().alloc(std::max(params.isize.front() * in_elem_size,
                                                params.osize.front() * out_elem_size));

            gather_field(mpi_comm,
                         params.ifields.front().bricks,
                         params.istride,
                         params.idist,
                         params.precision,
                         params.itype,
                         local_input,
                         cpu_data.front());

            if(mpi_rank == 0)
            {
                fft_params params_inplace = params;
                params_inplace.placement  = fft_placement_inplace;

                // create fftw plan and run it
                switch(params_inplace.precision)
                {
                case fft_precision_half:
                {
                    execute_reference_fft<rocfft_fp16>(params_inplace, cpu_data);
                    break;
                }
                case fft_precision_single:
                {
                    execute_reference_fft<float>(params_inplace, cpu_data);
                    break;
                }
                case fft_precision_double:
                {
                    execute_reference_fft<double>(params_inplace, cpu_data);
                    break;
                }
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
        }
    }

    // now all ranks are ready to start FFT

    // call rocfft_plan_create
    for(auto& p : all_params)
        p.create_plan();

    std::chrono::time_point<std::chrono::steady_clock> start, stop;
    for(size_t i = 0; i < testcases.size(); ++i)
    {
        size_t testcase = testcases[i];
        if(run_bench)
        {
            // reinit input for tests after the first one
            if(i > 0)
                init_local_input(mpi_comm, params, in_elem_size, local_input_ptrs);

            // ensure plan is finished building, synchronize all devices
            // in the input bricks
            (void)hipDeviceSynchronize();

            MPI_Barrier(mpi_comm);

            // start timer
            start = std::chrono::steady_clock::now();
        }

        all_params[testcase].execute(local_input_ptrs.data(), local_output_ptrs.data());

        if(run_bench)
        {
            // ensure FFT is finished executing - synchronize all devices
            // on output bricks
            {
                rocfft_scoped_device dev(output_device);
                (void)hipDeviceSynchronize();
            }

            stop = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::milli> diff = stop - start;

            double diff_ms = diff.count();

            double max_diff_ms = 0.0;
            // reduce max runtime to root
            MPI_Reduce(&diff_ms, &max_diff_ms, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);

            if(mpi_rank == 0)
                gpu_time[testcase].push_back(max_diff_ms);
        }
    }

    if(check_fftw)
    {
        // gather output
        std::vector<hostbuf> gpu_output(1);
        if(mpi_rank == 0)
            gpu_output.front().alloc(params.osize.front() * out_elem_size);
        gather_field(mpi_comm,
                     params.ofields.front().bricks,
                     params.ostride,
                     params.odist,
                     params.precision,
                     params.otype,
                     params.placement == fft_placement_inplace ? local_input : local_output,
                     gpu_output.front());

        if(mpi_rank == 0)
        {
            // compare data to reference implementation
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

        int localDeviceCount = 0;
        (void)hipGetDeviceCount(&localDeviceCount);

        // set default multi-process grids in case none were given
        params.set_default_grid(mp_size, imgrid, omgrid);

        // start with all-ones in grids
        std::vector<unsigned int> input_grid(params.length.size() + 1, 1);
        std::vector<unsigned int> output_grid(params.length.size() + 1, 1);

        // sanity checks
        int imgrid_size = std::accumulate(imgrid.begin(), imgrid.end(), 1, std::multiplies<int>());
        int omgrid_size = std::accumulate(omgrid.begin(), omgrid.end(), 1, std::multiplies<int>());

        if((imgrid.size() != params.length.size()) || (omgrid.size() != params.length.size()))
        {
            throw std::runtime_error(
                "grid of processors must be of the same size as the problem dimension!");
        }

        if((imgrid_size != mp_size) || (omgrid_size != mp_size))
        {
            throw std::runtime_error(
                "size of grid of processors must be equal to the number of MPI resources!");
        }

        // create input and output grids and distribute it according to user requirements
        std::copy(imgrid.begin(), imgrid.end(), input_grid.begin() + 1);
        std::copy(omgrid.begin(), omgrid.end(), output_grid.begin() + 1);

        params.distribute_input(localDeviceCount, input_grid);
        params.distribute_output(localDeviceCount, output_grid);

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

            std::cout << "output grid:";
            for(auto& i : omgrid)
                std::cout << " " << i;
            std::cout << "\n";

            std::cout << "Token: " << token << std::endl;
            std::cout << "\n";

            std::cout << std::flush;
        }
    }

    gpubuf             local_input;
    std::vector<void*> local_input_ptrs;
    gpubuf             local_output;
    std::vector<void*> local_output_ptrs;

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
                   local_input,
                   local_input_ptrs,
                   local_output,
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
                       local_input,
                       local_input_ptrs,
                       local_output,
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
            if(ntrial % 2)
                median = (times[ntrial / 2] + times[(ntrial + 1) / 2]) / 2;
            else
                median = times[ntrial / 2];
            std::cout << "Median: " << median << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
