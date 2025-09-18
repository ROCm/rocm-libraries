// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../../shared/params_gen.h"
#include "../hipfft_params.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <optional>

extern fft_params::fft_mp_lib mp_lib;
extern int                    mp_ranks;

static const std::vector<std::vector<size_t>> multi_gpu_sizes = {
    {128, 256},
    {192, 768},
    {64, 128, 256},
    {96, 160, 192},
};
static const std::vector<size_t>        multi_gpu_batch_range = {10, 1};
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
};

std::vector<fft_params> param_generator_multi_gpu(const std::optional<SplitType> type,
                                                  fft_auto_allocation            auto_alloc_setting
                                                  = fft_auto_allocation_default)
{
    int localDeviceCount = 0;
    (void)hipGetDeviceCount(&localDeviceCount);

    // if we have an explicit split of data on the user side, we need
    // to use the multiprocessing API
    if(type)
    {
        if(mp_lib == fft_params::fft_mp_lib_none)
            return {};
    }
    // data is not explicitly split up, that means the library is
    // asked to do the split.  We need multiple GPUs to do this.
    else if(localDeviceCount < 2)
        return {};

    static const std::vector<std::vector<size_t>> stride_range = {{1}};
    auto params_complex                                        = param_generator_complex(test_prob,
                                                  multi_gpu_sizes,
                                                  precision_range_sp_dp,
                                                  multi_gpu_batch_range,
                                                  stride_generator(stride_range),
                                                  stride_generator(stride_range),
                                                  ioffset_range_zero,
                                                  ooffset_range_zero,
                                                  place_range,
                                                  false,
                                                  false,
                                                  auto_alloc_setting);

    auto params_real = param_generator_real(test_prob,
                                            multi_gpu_sizes,
                                            precision_range_sp_dp,
                                            multi_gpu_batch_range,
                                            stride_generator(stride_range),
                                            stride_generator(stride_range),
                                            ioffset_range_zero,
                                            ooffset_range_zero,
                                            {fft_placement_notinplace},
                                            false,
                                            false,
                                            auto_alloc_setting);

    std::vector<fft_params> all_params;

    auto distribute_params = [=, &all_params](const std::vector<fft_params>& params) {
        for(auto& p : params)
        {
            // test library splitting
            if(!type)
            {
                auto param_multi = p;

                // for single-batch, cuFFT only allows in-place
                if(p.nbatch == 1 && p.placement == fft_placement_notinplace)
                    continue;

                param_multi.multiGPU = std::min(static_cast<int>(p.nbatch), localDeviceCount);
                all_params.emplace_back(std::move(param_multi));
            }
            else
            {
                // the API only allows for batch-1 multi-process FFTs
                if(p.nbatch > 1)
                    continue;

                // user-specified split
                int brickCount = mp_ranks;

                // start with all-ones in grids
                std::vector<unsigned int> input_grid(p.length.size() + 1, 1);
                std::vector<unsigned int> output_grid(p.length.size() + 1, 1);

                auto p_dist = p;
                switch(*type)
                {
                case SLOW_INOUT:
                    input_grid[1]  = brickCount;
                    output_grid[1] = brickCount;
                    break;
                case SLOW_IN:
                    // this type only specifies input field and no output
                    // field, but multi-process transforms require both
                    // fields.
                    if(mp_lib != fft_params::fft_mp_lib_none)
                        continue;
                    input_grid[1] = brickCount;
                    break;
                case SLOW_OUT:
                    // this type only specifies output field and no input
                    // field, but multi-process transforms require both
                    // fields.
                    if(mp_lib != fft_params::fft_mp_lib_none)
                        continue;
                    output_grid[1] = brickCount;
                    break;
                case SLOW_IN_FAST_OUT:
                    // requires at least rank-2 FFT
                    if(p.length.size() < 2)
                        continue;
                    input_grid[1]      = brickCount;
                    output_grid.back() = brickCount;
                    break;
                case PENCIL_3D:
                    // need at least 2 bricks per split dimension, or 4 devices.
                    // also needs to be a 3D problem.
                    if(brickCount < 4 || p.length.size() != 3)
                        continue;

                    // make fast dimension contiguous on input
                    input_grid[1] = static_cast<unsigned int>(sqrt(brickCount));
                    input_grid[2] = brickCount / input_grid[1];
                    // make middle dimension contiguous on output
                    output_grid[1] = input_grid[1];
                    output_grid[3] = input_grid[2];
                    break;
                }

                p_dist.mp_lib = mp_lib;
                p_dist.distribute_input(localDeviceCount, input_grid);
                p_dist.distribute_output(localDeviceCount, output_grid);

                // "placement" flag is meaningless if exactly one of
                // input+output is a field.  So just add those cases if
                // the flag is "out-of-place", since "in-place" is
                // exactly the same test case.
                if(p_dist.placement == fft_placement_inplace
                   && p_dist.ifields.empty() != p_dist.ofields.empty())
                    continue;

                // in-place transforms require identical input/output layouts
                if(p.placement == fft_placement_inplace && input_grid != output_grid)
                    continue;
                all_params.push_back(std::move(p_dist));
            }
        }
    };

    distribute_params(params_complex);
    distribute_params(params_real);

    return all_params;
}

// split both input and output on slowest FFT dim
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_INOUT)),
                         accuracy_test::TestName);

// split slowest FFT dim only on input, or only on output
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_input_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_IN)),
                         accuracy_test::TestName);
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowest_output_dim,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_OUT)),
                         accuracy_test::TestName);

// split input on slowest FFT and output on fastest, to minimize data
// movement (only makes sense for rank-2 and higher FFTs)
INSTANTIATE_TEST_SUITE_P(multi_gpu_slowin_fastout,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(SLOW_IN_FAST_OUT)),
                         accuracy_test::TestName);

// 3D pencil decompositions
INSTANTIATE_TEST_SUITE_P(multi_gpu_3d_pencils,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu(PENCIL_3D)),
                         accuracy_test::TestName);

// library-decided splits
INSTANTIATE_TEST_SUITE_P(multi_gpu,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu({})),
                         accuracy_test::TestName);

// Note: disabled for now due to implementation issues and
// unimplemented features in hipFFT (to fix first)
INSTANTIATE_TEST_SUITE_P(DISABLED_various_multi_gpu,
                         accuracy_test,
                         ::testing::ValuesIn(param_generator_multi_gpu({},
                                                                       fft_auto_allocation_off)),
                         accuracy_test::TestName);
