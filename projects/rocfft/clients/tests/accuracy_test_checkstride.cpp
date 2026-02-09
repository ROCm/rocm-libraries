// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

extern bool fftw_compare;

inline auto param_checkstride()
{
    // checkstride requires us to copy data back to the host for
    // checking, which we only do when comparing against FFTW.
    if(!fftw_compare)
        return std::vector<fft_params>{};

    // tuples of length,stride,nbatch,dist to test.  strides are arranged so
    // there's space either between elements on the fastest dim, or
    // between dims, or both.
    std::vector<std::tuple<std::vector<size_t>, std::vector<size_t>, size_t, size_t>> sizes = {
        // 1D single kernel non-unit stride
        {{64}, {2}, 2, 140},
        // 1D single kernel unit stride but non-contiguous batch
        {{64}, {1}, 2, 80},
        // 1D odd length (to test odd-length R2C/C2R)
        {{15}, {2}, 2, 40},
        // 1D SBCC+SBRC
        {{8192}, {2}, 2, 17000},
        // 1D TRTRT
        {{24000}, {2}, 2, 50000},
        // 2D_RTRT
        {{20, 30}, {80, 2}, 2, 1700},
        {{40, 30}, {80, 2}, 2, 3600},
        // 2D_RTRT unit stride along fast dim
        {{20, 30}, {40, 1}, 2, 1000},
        {{40, 30}, {40, 1}, 2, 2000},
        // 2D_RC
        {{64, 64}, {130, 2}, 2, 8400},
        // 3D_RC
        {{64, 64, 64}, {8400, 130, 2}, 2, 540000},
        // 3D_RTRTRT
        {{2, 3, 4}, {40, 10, 2}, 2, 100},
        // bigger 3D_RTRTRT
        {{30, 40, 50}, {3000, 60, 1}, 2, 100000},
    };

    std::vector<fft_params> params;
    for(const auto trans_type : trans_type_range)
    {
        for(const auto& s : sizes)
        {
            for(const auto precision : precision_range_sp_dp)
            {
                for(const auto& types :
                    generate_types(trans_type, {fft_placement_notinplace}, true))
                {
#ifdef WIN32
                    for(bool callback : {false})
#else
                    for(bool callback : {true, false})
#endif
                    {
                        // callbacks don't work for planar
                        bool is_planar = std::get<2>(types) == fft_array_type_complex_planar
                                         || std::get<2>(types) == fft_array_type_hermitian_planar
                                         || std::get<3>(types) == fft_array_type_complex_planar
                                         || std::get<3>(types) == fft_array_type_hermitian_planar;
                        if(callback && is_planar)
                            continue;

                        fft_params param;

                        param.length               = std::get<0>(s);
                        param.istride              = std::get<1>(s);
                        param.ostride              = std::get<1>(s);
                        param.nbatch               = std::get<2>(s);
                        param.precision            = precision;
                        param.idist                = std::get<3>(s);
                        param.odist                = std::get<3>(s);
                        param.transform_type       = std::get<0>(types);
                        param.placement            = std::get<1>(types);
                        param.itype                = std::get<2>(types);
                        param.otype                = std::get<3>(types);
                        param.run_callbacks        = callback;
                        param.check_output_strides = true;

                        param.validate();

                        const double roll = hash_prob(random_seed, param.token());
                        const double run_prob
                            = test_prob * (param.is_planar() ? complex_planar_prob_factor : 1.0)
                              * (param.is_interleaved() ? complex_interleaved_prob_factor : 1.0)
                              * (param.is_real() ? real_prob_factor : 1.0);

                        if(roll > run_prob)
                        {
                            if(verbose > 4)
                            {
                                std::cout << "Test skipped (probability " << run_prob << " > "
                                          << roll << ")\n";
                            }
                            continue;
                        }
                        if(param.valid(0))
                        {
                            params.push_back(param);
                        }
                    }
                }
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(checkstride,
                         accuracy_test,
                         ::testing::ValuesIn(param_checkstride()),
                         accuracy_test::TestName);
