// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../shared/accuracy_test.h"
#include "../../shared/fft_enums.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_params.h"
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

extern size_t                 random_seed;
extern fft_params::fft_mp_lib mp_lib;
extern int                    mp_ranks;

static const std::vector<std::vector<size_t>> multi_gpu_sizes = {
    {256},
    {256, 256},
    {256, 1},
    {1, 256},
    {256, 256, 256},
    {256, 256, 1},
    {256, 1, 256},
    {1, 256, 256},
    {256, 1, 1},
    {1, 1, 256},
    {1, 256, 1},
};
static const std::vector<size_t>        multi_gpu_batch_range = {4, 1};
static std::vector<std::vector<size_t>> ioffset_range_zero    = {{0, 0}};
static std::vector<std::vector<size_t>> ooffset_range_zero    = {{0, 0}};

enum SplitType
{
    // split both input and output on slow FFT dimension
    SLOW_INOUT,
    // split only input on slow FFT dimension, output is not split
    SLOW_IN,
    // split only output on slow FFT dimension, input is not split
    SLOW_OUT,
    // split input on slow FFT dimension, and output on fast FFT dimension
    SLOW_IN_FAST_OUT,
    // 3D pencil decomposition - one dimension is contiguous on input
    // and another dimension contiguous on output, remaining dims are
    // both split
    PENCIL_3D,
    // split I/O as (single-process) hipfft would implicitly do for the
    // considered number of devices
    IMPLICIT_HIPFFT,
    // different I/O devices but no decomposition
    DIFFERENT_IO_DEVICES
};

std::vector<fft_params> param_generator_multi_gpu(const SplitType type, const int ngpus)
{
    int gpusperrank = 0;
    if(ngpus <= 0)
    {
        // Use the command-line option as a priority
        if(hipGetDeviceCount(&gpusperrank) != hipSuccess)
        {
            throw std::runtime_error("hipGetDeviceCount failed");
        }

        // Limit local device testing to 16 GPUs, as we have some
        // bottlenecks with larger device counts that unreasonably slow
        // down plan creation
        gpusperrank = std::min<int>(16, gpusperrank);
    }
    else
    {
        gpusperrank = ngpus;
    }

    if(mp_lib == fft_params::fft_mp_lib_none && mp_ranks != 1)
        throw std::runtime_error("Unexpected value of mp_ranks (" + std::to_string(mp_ranks)
                                 + ") without a multi-process library.");

    // need more than one device overall, of course
    const auto total_num_devices = mp_ranks * gpusperrank;
    if(total_num_devices < 2)
        return {};

    static const std::vector<std::vector<size_t>> stride_range = {{1}};

    // gather cases to test as single-device params, then distribute
    // to multiple GPUs
    std::vector<fft_params> params_single;

    for(auto run_callbacks : {false, true})
    {
        auto params = param_generator_base(test_prob,
                                           trans_type_range_full,
                                           multi_gpu_sizes,
                                           precision_range_sp_dp,
                                           multi_gpu_batch_range,
                                           generate_types,
                                           stride_generator(stride_range),
                                           stride_generator(stride_range),
                                           ioffset_range_zero,
                                           ooffset_range_zero,
                                           place_range,
                                           false,
                                           run_callbacks);
        std::copy(params.begin(), params.end(), std::back_inserter(params_single));
    }

    std::vector<fft_params> all_params;

    auto distribute_params = [=, &all_params](const std::vector<fft_params>& params) {
        for(auto& p : params)
        {
            // start with all-ones in grids
            std::vector<unsigned int>          input_grid(p.length.size() + 1, 1);
            std::vector<unsigned int>          output_grid(p.length.size() + 1, 1);
            int                                start_global_dev_id_input  = 0;
            int                                start_global_dev_id_output = 0;
            static std::ranlux24_base          gen(random_seed);
            std::uniform_int_distribution<int> dev_rng(0, total_num_devices);

            auto p_dist = p;
            switch(type)
            {
            case SLOW_INOUT:
                input_grid[1]  = total_num_devices;
                output_grid[1] = total_num_devices;
                break;
            case SLOW_IN:
                // this type only specifies input field and no output
                // field, but multi-process transforms require both
                // fields.
                if(mp_ranks > 1)
                    continue;
                if(p_dist.placement == fft_placement_inplace)
                    continue;
                input_grid[1] = total_num_devices;
                break;
            case SLOW_OUT:
                // this type only specifies output field and no input
                // field, but multi-process transforms require both
                // fields.
                if(mp_ranks > 1)
                    continue;
                if(p_dist.placement == fft_placement_inplace)
                    continue;
                output_grid[1] = total_num_devices;
                break;
            case SLOW_IN_FAST_OUT:
                // requires at least rank-2 FFT
                if(p.length.size() < 2)
                    continue;
                input_grid[1]      = total_num_devices;
                output_grid.back() = total_num_devices;
                break;
            case PENCIL_3D:
                // need at least 2 bricks per split dimension, or 4 devices.
                // also needs to be a 3D problem.
                if(total_num_devices < 4 || p.length.size() != 3)
                    continue;

                // make fast dimension contiguous on input
                input_grid[1] = static_cast<unsigned int>(sqrt(total_num_devices));
                input_grid[2] = total_num_devices / input_grid[1];
                // make middle dimension contiguous on output
                output_grid[1] = input_grid[1];
                output_grid[3] = input_grid[2];
                break;
            case IMPLICIT_HIPFFT:
                // unbatched 1D cases are irrelevant at the moment
                if(p.length.size() < 2 && p.nbatch <= 1)
                    continue;
                if(p.nbatch == 1 && p.placement != fft_placement_inplace)
                    continue; // only in-place is relevant for unbatched cases
                if(p.nbatch > 1)
                {
                    // only the batch dimension is split
                    input_grid[0] = output_grid[0]
                        = std::min(p.nbatch, static_cast<size_t>(total_num_devices));
                }
                else
                {
                    // Slowest nonbatch dimension is split on input (resp. output)
                    // of forward (resp. inverse) transforms. Second-slowest nonbatch
                    // dimension is split on output (resp. input) of forward (resp.
                    // inverse) transforms
                    if(p.is_forward())
                    {
                        input_grid[1]  = total_num_devices;
                        output_grid[2] = total_num_devices;
                    }
                    else
                    {
                        input_grid[2]  = total_num_devices;
                        output_grid[1] = total_num_devices;
                    }
                }
                break;
            case DIFFERENT_IO_DEVICES:
                if(mp_lib != fft_params::fft_mp_lib_none)
                    continue; // FIXME, fails even with only 2 ranks
                if(p.placement == fft_placement_inplace)
                    continue; // only out-of-place
                if(p.run_callbacks)
                    continue; // known issue to fix w/ callbacks
                start_global_dev_id_input  = dev_rng(gen);
                start_global_dev_id_output = dev_rng(gen);
                while(start_global_dev_id_input == start_global_dev_id_output)
                    start_global_dev_id_output = dev_rng(gen);
                break;
            default:
                throw std::invalid_argument("param_generator_multi_gpu: unkonwn split type");
            }

            bool too_short_lengths = false;
            for(size_t dim = 0; !too_short_lengths && dim < p.length.size(); dim++)
                too_short_lengths
                    |= p.length[dim] < input_grid[dim + 1] || p.length[dim] < output_grid[dim + 1];
            if(too_short_lengths)
                continue;
            if(mp_lib == fft_params::fft_mp_lib_mpi && p_dist.is_real() && p.length.back() == 1)
                continue; // known issues to fix in mpi_worker.h

            p_dist.mp_lib = mp_lib;
            p_dist.distribute_field<fft_io::fft_io_in>(
                gpusperrank, input_grid, mp_ranks, start_global_dev_id_input);
            p_dist.distribute_field<fft_io::fft_io_out>(
                gpusperrank, output_grid, mp_ranks, start_global_dev_id_output);

            if(mp_ranks > 1)
            {
                std::set<int> used_ranks;
                for(const auto& io_fields : {p_dist.ifields, p_dist.ofields})
                {
                    if(io_fields.empty())
                        used_ranks.insert(0); // implicit "current" rank
                    for(const auto& io_field : io_fields)
                    {
                        for(const auto& b : io_field.bricks)
                            used_ranks.insert(b.rank);
                    }
                }
                if(used_ranks.size() < static_cast<size_t>(mp_ranks))
                {
                    // some ranks have nothing to do...
                    continue;
                }
            }

            all_params.push_back(p_dist);
            // also test result scaling for multi-GPU plans
            p_dist.scale_factor = 4.23;
            all_params.push_back(std::move(p_dist));
        }
    };

    distribute_params(params_single);

    return all_params;
}

// split both input and output on slowest FFT dim
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_INOUT, ngpus)),
                         accuracy_test::TestName);

// split slowest FFT dim only on input, or only on output
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_input_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_IN, ngpus)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_output_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_OUT, ngpus)),
                         accuracy_test::TestName);

// split input on slowest FFT and output on fastest, to minimize data
// movement (only makes sense for rank-2 and higher FFTs)
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowin_fastout,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_IN_FAST_OUT, ngpus)),
                         accuracy_test::TestName);

// 3D pencil decompositions
INSTANTIATE_TEST_SUITE_P(multi_gpu_3d_pencils,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(PENCIL_3D, ngpus)),
                         accuracy_test::TestName);

// decompositions as hipFFT would define under the hood
INSTANTIATE_TEST_SUITE_P(multi_gpu_implicit_hipfft,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(IMPLICIT_HIPFFT, ngpus)),
                         accuracy_test::TestName);

// input data on one device, output data on another
INSTANTIATE_TEST_SUITE_P(multi_gpu_different_io_devices,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(DIFFERENT_IO_DEVICES,
                                                                       ngpus)),
                         accuracy_test::TestName);

TEST(multi_gpu_validate, catch_validation_errors)
{
    const auto all_split_types
        = {SLOW_INOUT, SLOW_IN, SLOW_OUT, SLOW_IN_FAST_OUT, PENCIL_3D, IMPLICIT_HIPFFT};

    for(auto type : all_split_types)
    {
        // gather all of the multi-GPU test cases
        auto params = param_generator_multi_gpu(type, ngpus);

        for(size_t i = 0; i < params.size(); ++i)
        {
            auto& param = params[i];

            // this validation runs in rocfft-test itself and
            // multi-process libs are not initialized.
            if(param.mp_lib != fft_params::fft_mp_lib_none)
                continue;

            std::vector<fft_params::fft_field*> available_fields;
            if(!param.ifields.empty())
                available_fields.push_back(&param.ifields.front());
            if(!param.ofields.empty())
                available_fields.push_back(&param.ofields.front());

            // get iterator to the brick we will modify
            auto field      = available_fields[i % available_fields.size()];
            auto brick_iter = field->bricks.begin() + i % field->bricks.size();

            // iterate through the 5 cases we want to test:
            switch(i % 5)
            {
            case 0:
            {
                // missing brick
                field->bricks.erase(brick_iter);
                break;
            }
            case 1:
            {
                // a brick's lower index too small by one
                size_t& index = brick_iter->lower[i % brick_iter->lower.size()];
                // don't worry about underflow since that should also
                // produce an invalid brick layout
                --index;
                break;
            }
            case 2:
            {
                // a brick's lower index too large by one
                size_t& index = brick_iter->lower[i % brick_iter->lower.size()];
                ++index;
                break;
            }
            case 3:
            {
                // a brick's upper index too small by one
                size_t& index = brick_iter->upper[i % brick_iter->lower.size()];
                // don't worry about underflow since that should also
                // produce an invalid brick layout
                --index;
                break;
            }
            case 4:
            {
                // a brick's upper index too large by one
                size_t& index = brick_iter->upper[i % brick_iter->lower.size()];
                ++index;
                break;
            }
            }

            rocfft_params rparam{param};
            // brick layout is invalid, so this should fail
            try
            {
                rparam.setup_structs();
            }
            catch(std::runtime_error&)
            {
                continue;
            }
            // didn't get an exception, fail the test
            GTEST_FAIL() << "invalid brick layout " << rparam.token()
                         << " should have failed, but plan was created successfully";
        }
    }
}

static const auto multi_gpu_tokens = {
    // clang-format off

    // input bricks are not contiguous
    "real_forward_len_160_160_160_single_op_batch_1_ifield_brick_lower_0_0_0_0_upper_1_80_160_160_stride_0_25920_162_1_dev_0_brick_lower_0_80_0_0_upper_1_160_160_160_stride_0_25920_162_1_rank_1_dev_1_ofield_brick_lower_0_0_0_0_upper_1_160_80_81_stride_0_6480_81_1_dev_0_brick_lower_0_0_80_0_upper_1_160_160_81_stride_0_6480_81_1_rank_1_dev_1",
    // output bricks are not contiguous
    "real_forward_len_160_160_160_single_op_batch_1_ifield_brick_lower_0_0_0_0_upper_1_80_160_160_stride_0_25600_160_1_dev_0_brick_lower_0_80_0_0_upper_1_160_160_160_stride_0_25600_160_1_rank_1_dev_1_ofield_brick_lower_0_0_0_0_upper_1_160_80_81_stride_0_6560_82_1_dev_0_brick_lower_0_0_80_0_upper_1_160_160_81_stride_0_6560_82_1_rank_1_dev_1",
    // neither input nor output bricks are contiguous
    "real_forward_len_160_160_160_single_op_batch_1_ifield_brick_lower_0_0_0_0_upper_1_80_160_160_stride_0_25920_162_1_dev_0_brick_lower_0_80_0_0_upper_1_160_160_160_stride_0_25920_162_1_rank_1_dev_1_ofield_brick_lower_0_0_0_0_upper_1_160_80_81_stride_0_6560_82_1_dev_0_brick_lower_0_0_80_0_upper_1_160_160_81_stride_0_6560_82_1_rank_1_dev_1",
    // 1D multi-process batched in-place transform using 1 device per rank
    "complex_forward_len_256_double_ip_batch_4_ifield_brick_lower_0_0_upper_4_128_stride_128_1_dev_0_brick_lower_0_128_upper_4_256_stride_128_1_rank_1_dev_1_ofield_brick_lower_0_0_upper_4_128_stride_128_1_dev_0_brick_lower_0_128_upper_4_256_stride_128_1_rank_1_dev_1",
    // 2D multi-process out-of-place transform using 2 MPI ranks each with 2 GPUs
    "complex_forward_len_128_256_single_op_batch_1_ifield_brick_lower_0_0_0_upper_1_128_64_stride_8192_64_1_dev_0_brick_lower_0_0_64_upper_1_128_128_stride_8192_64_1_rank_1_dev_1_brick_lower_0_0_128_upper_1_128_192_stride_8192_64_1_rank_0_dev_2_brick_lower_0_0_192_upper_1_128_256_stride_8192_64_1_rank_1_dev_3_ofield_brick_lower_0_0_0_upper_1_128_64_stride_8192_64_1_dev_0_brick_lower_0_0_64_upper_1_128_128_stride_8192_64_1_rank_1_dev_1_brick_lower_0_0_128_upper_1_128_192_stride_8192_64_1_rank_0_dev_2_brick_lower_0_0_192_upper_1_128_256_stride_8192_64_1_rank_1_dev_3",    
    // 3D multi-process out-of-place transform using 2 MPI ranks each with 2 GPUs
    "complex_forward_len_256_256_256_double_op_batch_1_ifield_brick_lower_0_0_0_0_upper_1_64_256_256_stride_4194304_65536_256_1_dev_0_brick_lower_0_64_0_0_upper_1_128_256_256_stride_4194304_65536_256_1_rank_0_dev_1_brick_lower_0_128_0_0_upper_1_192_256_256_stride_4194304_65536_256_1_rank_1_dev_2_brick_lower_0_192_0_0_upper_1_256_256_256_stride_4194304_65536_256_1_rank_1_dev_3_ofield_brick_lower_0_0_0_0_upper_1_256_256_64_stride_4194304_16384_64_1_dev_0_brick_lower_0_0_0_64_upper_1_256_256_128_stride_4194304_16384_64_1_rank_0_dev_1_brick_lower_0_0_0_128_upper_1_256_256_192_stride_4194304_16384_64_1_rank_1_dev_2_brick_lower_0_0_0_192_upper_1_256_256_256_stride_4194304_16384_64_1_rank_1_dev_3",
    // 3D multi-process batched in-place transform using 2 MPI ranks each with 2 GPUs
    "complex_forward_len_128_300_256_single_op_batch_4_ifield_brick_lower_0_0_0_0_upper_4_32_300_256_stride_2457600_76800_256_1_dev_0_brick_lower_0_32_0_0_upper_4_64_300_256_stride_2457600_76800_256_1_rank_1_dev_1_brick_lower_0_64_0_0_upper_4_96_300_256_stride_2457600_76800_256_1_rank_0_dev_2_brick_lower_0_96_0_0_upper_4_128_300_256_stride_2457600_76800_256_1_rank_1_dev_3_ofield_brick_lower_0_0_0_0_upper_4_128_300_64_stride_2457600_19200_64_1_dev_0_brick_lower_0_0_0_64_upper_4_128_300_128_stride_2457600_19200_64_1_rank_1_dev_1_brick_lower_0_0_0_128_upper_4_128_300_192_stride_2457600_19200_64_1_rank_0_dev_2_brick_lower_0_0_0_192_upper_4_128_300_256_stride_2457600_19200_64_1_rank_1_dev_3 ",

    // clang-format on
};

std::vector<fft_params> param_generator_multi_gpu_adhoc()
{
    int localDeviceCount = 0;
    if(ngpus <= 0)
    {
        // Use the command-line option as a priority
        if(hipGetDeviceCount(&localDeviceCount) != hipSuccess)
        {
            throw std::runtime_error("hipGetDeviceCount failed");
        }

        // Limit local device testing to 16 GPUs, as we have some
        // bottlenecks with larger device counts that unreasonably slow
        // down plan creation
        localDeviceCount = std::min<int>(16, localDeviceCount);
    }
    else
    {
        localDeviceCount = ngpus;
    }

    auto all_params = param_generator_token(test_prob, multi_gpu_tokens);

    // check if fields use more bricks than we can support
    auto too_many_bricks = [=](const std::vector<fft_params::fft_field>& fields, size_t maxBricks) {
        for(const auto& f : fields)
        {
            if(f.bricks.size() > maxBricks)
                return true;

            // also remove a test case if it uses a numbered device
            // that isn't available
            if(std::any_of(f.bricks.begin(), f.bricks.end(), [=](const fft_params::fft_brick& b) {
                   return b.device >= localDeviceCount;
               }))
                return true;
        }
        return false;
    };

    // remove test cases where we don't have enough ranks/devices for
    // the number of bricks
    all_params.erase(std::remove_if(all_params.begin(),
                                    all_params.end(),
                                    [=](const fft_params& params) {
                                        size_t maxBricks = mp_lib == fft_params::fft_mp_lib_mpi
                                                               ? mp_ranks
                                                               : localDeviceCount;
                                        return too_many_bricks(params.ifields, maxBricks)
                                               || too_many_bricks(params.ofields, maxBricks);
                                    }),
                     all_params.end());

    // set all bricks in a field to rank-0, to change an MPI test
    // case to single-proc
    auto set_rank_0 = [](std::vector<fft_params::fft_field>& fields) {
        for(auto& f : fields)
        {
            for(auto& b : f.bricks)
                b.rank = 0;
        }
    };

    // modify the remaining test cases to use the current multi-GPU lib
    for(auto& params : all_params)
    {
        params.mp_lib = mp_lib;
        if(mp_lib == fft_params::fft_mp_lib_none)
        {
            set_rank_0(params.ifields);
            set_rank_0(params.ofields);
        }
    }
    return all_params;
}

INSTANTIATE_TEST_SUITE_P(multi_gpu_adhoc_token,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu_adhoc()),
                         accuracy_test::TestName);
