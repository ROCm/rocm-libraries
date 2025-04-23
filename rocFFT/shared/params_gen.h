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

#pragma once
#ifndef TEST_PARAMS_GEN_H
#define TEST_PARAMS_GEN_H

#include <vector>

#include "fft_params.h"
#include "test_params.h"

const static std::vector<size_t> batch_range = {2, 1};

const static std::vector<fft_precision> precision_range_full
    = {fft_precision_double, fft_precision_single, fft_precision_half};
const static std::vector<fft_precision> precision_range_sp_dp
    = {fft_precision_double, fft_precision_single};

const static std::vector<fft_result_placement> place_range
    = {fft_placement_inplace, fft_placement_notinplace};
const static std::vector<fft_transform_type> trans_type_range
    = {fft_transform_type_complex_forward, fft_transform_type_real_forward};
const static std::vector<fft_transform_type> trans_type_range_complex
    = {fft_transform_type_complex_forward};
const static std::vector<fft_transform_type> trans_type_range_real
    = {fft_transform_type_real_forward};

// Take a string (in particular the token from a test) and return a uniform random variable in [0,1]
// using the seed and hash of the string.
inline double hash_prob(const int seed, const std::string& token)
{
    // Keeping the random number generator here
    // allows one to run the same tests for a given
    // random seed; ie the test suite is repeatable.
    std::hash<std::string>           hasher;
    std::ranlux24_base               gen(random_seed + hasher(token));
    std::uniform_real_distribution<> dis(0.0, 1.0);

    const double roll = dis(gen);
    return roll;
}

// Given a vector of vector of lengths, generate all unique permutations.
// Add an optional vector of ad-hoc lengths to the result.
inline std::vector<std::vector<size_t>>
    generate_lengths(const std::vector<std::vector<size_t>>& inlengths)
{
    std::vector<std::vector<size_t>> output;
    if(inlengths.size() == 0)
    {
        return output;
    }
    const size_t        dim = inlengths.size();
    std::vector<size_t> looplength(dim);
    for(unsigned int i = 0; i < dim; ++i)
    {
        looplength[i] = inlengths[i].size();
    }
    for(unsigned int idx = 0; idx < inlengths.size(); ++idx)
    {
        std::vector<size_t> index(dim);
        do
        {
            std::vector<size_t> length(dim);
            for(unsigned int i = 0; i < dim; ++i)
            {
                length[i] = inlengths[i][index[i]];
            }
            output.push_back(length);
        } while(increment_rowmajor(index, looplength));
    }
    // uniquify the result
    std::sort(output.begin(), output.end());
    output.erase(std::unique(output.begin(), output.end()), output.end());
    return output;
}

typedef std::tuple<fft_transform_type, fft_result_placement, fft_array_type, fft_array_type>
    type_place_io_t;

// Return the valid rocFFT input and output types for a given transform type.
inline std::vector<std::pair<fft_array_type, fft_array_type>>
    iotypes(const fft_transform_type   transformType,
            const fft_result_placement place,
            const bool                 planar = true)
{
    std::vector<std::pair<fft_array_type, fft_array_type>> iotypes;
    switch(transformType)
    {
    case fft_transform_type_complex_forward:
    case fft_transform_type_complex_inverse:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_complex_interleaved, fft_array_type_complex_interleaved));
        if(planar)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_complex_planar, fft_array_type_complex_planar));
            if(place == fft_placement_notinplace)
            {
                iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                    fft_array_type_complex_planar, fft_array_type_complex_interleaved));
                iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                    fft_array_type_complex_interleaved, fft_array_type_complex_planar));
            }
        }
        break;
    case fft_transform_type_real_forward:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_real, fft_array_type_hermitian_interleaved));
        if(planar && place == fft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_real, fft_array_type_hermitian_planar));
        }
        break;
    case fft_transform_type_real_inverse:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_hermitian_interleaved, fft_array_type_real));
        if(planar && place == fft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_hermitian_planar, fft_array_type_real));
        }
        break;
    default:
        throw std::runtime_error("Invalid transform type");
    }
    return iotypes;
}

// Generate all combinations of input/output types, from combinations of transform and placement
// types.
static std::vector<type_place_io_t>
    generate_types(fft_transform_type                       transform_type,
                   const std::vector<fft_result_placement>& place_range,
                   const bool                               planar)
{
    std::vector<type_place_io_t> ret;
    for(auto place : place_range)
    {
        for(auto iotype : iotypes(transform_type, place, planar))
        {
            ret.push_back(std::make_tuple(transform_type, place, iotype.first, iotype.second));
        }
    }
    return ret;
}

struct stride_generator
{
    struct stride_dist
    {
        stride_dist(const std::vector<size_t>& s, size_t d)
            : stride(s)
            , dist(d)
        {
        }
        std::vector<size_t> stride;
        size_t              dist;
    };

    // NOTE: allow for this ctor to be implicit, so it's less typing for a test writer
    //
    // cppcheck-suppress noExplicitConstructor
    stride_generator(const std::vector<std::vector<size_t>>& stride_list_in)
        : stride_list(stride_list_in)
    {
    }
    virtual std::vector<stride_dist> generate(const std::vector<size_t>& lengths,
                                              size_t                     batch) const
    {
        std::vector<stride_dist> ret;
        for(const auto& s : stride_list)
            ret.emplace_back(s, 0);
        return ret;
    }
    std::vector<std::vector<size_t>> stride_list;
};

// Generate strides such that batch is essentially the innermost dimension
// e.g. given a batch-2 4x3x2 transform which logically looks like:
//
// batch0:
// A B A B
// A B A B
// A B A B
//
// A B A B
// A B A B
// A B A B
//
// batch1:
// A B A B
// A B A B
// A B A B
//
// A B A B
// A B A B
// A B A B
//
// we instead do stride-2 4x3x2 transform where first batch is the
// A's and second batch is the B's.
struct stride_generator_3D_inner_batch : public stride_generator
{
    explicit stride_generator_3D_inner_batch(const std::vector<std::vector<size_t>>& stride_list_in)
        : stride_generator(stride_list_in)
    {
    }
    std::vector<stride_dist> generate(const std::vector<size_t>& lengths,
                                      size_t                     batch) const override
    {
        std::vector<stride_dist> ret = stride_generator::generate(lengths, batch);
        std::vector<size_t> strides{lengths[1] * lengths[2] * batch, lengths[2] * batch, batch};
        ret.emplace_back(strides, 1);
        return ret;
    }
};

// Create an array of parameters to pass to gtest.  Base generator
// that allows choosing transform type.
inline auto param_generator_base(const double                             base_prob,
                                 const std::vector<fft_transform_type>&   type_range,
                                 const std::vector<std::vector<size_t>>&  v_lengths,
                                 const std::vector<fft_precision>&        precisions,
                                 const std::vector<size_t>&               batch_range,
                                 decltype(generate_types)                 types_generator,
                                 const stride_generator&                  istride,
                                 const stride_generator&                  ostride,
                                 const std::vector<std::vector<size_t>>&  ioffset_range,
                                 const std::vector<std::vector<size_t>>&  ooffset_range,
                                 const std::vector<fft_result_placement>& place_range,
                                 const bool                               planar        = true,
                                 const bool                               run_callbacks = false)
{
    std::vector<fft_params> params;

    if(base_prob == 0)
        return params;

    // For any length, we compute double-precision CPU reference
    // for largest batch size first and reuse for smaller batch
    // sizes, then convert to single-precision.

    for(auto& transform_type : type_range)
    {
        for(const auto& lengths : v_lengths)
        {
            // try to ensure that we are given literal lengths, not
            // something to be passed to generate_lengths
            if(lengths.empty() || lengths.size() > 3)
            {
                continue;
            }
            {
                for(const auto precision : precisions)
                {
                    for(const auto batch : batch_range)
                    {
                        for(const auto& types :
                            types_generator(transform_type, place_range, planar))
                        {
                            for(const auto& istride_dist : istride.generate(lengths, batch))
                            {
                                for(const auto& ostride_dist : ostride.generate(lengths, batch))
                                {
                                    for(const auto& ioffset : ioffset_range)
                                    {
                                        for(const auto& ooffset : ooffset_range)
                                        {
                                            fft_params param;

                                            param.length         = lengths;
                                            param.istride        = istride_dist.stride;
                                            param.ostride        = ostride_dist.stride;
                                            param.nbatch         = batch;
                                            param.precision      = precision;
                                            param.transform_type = std::get<0>(types);
                                            param.placement      = std::get<1>(types);
                                            param.idist          = istride_dist.dist;
                                            param.odist          = ostride_dist.dist;
                                            param.itype          = std::get<2>(types);
                                            param.otype          = std::get<3>(types);
                                            param.ioffset        = ioffset;
                                            param.ooffset        = ooffset;

                                            if(run_callbacks)
                                            {
                                                // add a test if both input and output support callbacks
                                                if(param.itype != fft_array_type_complex_planar
                                                   && param.itype != fft_array_type_hermitian_planar
                                                   && param.otype != fft_array_type_complex_planar
                                                   && param.otype
                                                          != fft_array_type_hermitian_planar)
                                                {
                                                    param.run_callbacks = true;
                                                }
                                                else
                                                {
                                                    continue;
                                                }
                                            }
                                            param.validate();

                                            const double roll
                                                = hash_prob(random_seed, param.token());
                                            const double run_prob
                                                = base_prob
                                                  * (param.is_planar() ? complex_planar_prob_factor
                                                                       : 1.0)
                                                  * (param.is_interleaved()
                                                         ? complex_interleaved_prob_factor
                                                         : 1.0)
                                                  * (param.is_real() ? real_prob_factor : 1.0)
                                                  * (run_callbacks ? callback_prob_factor : 1.0);

                                            if(roll > run_prob)
                                            {
                                                if(verbose > 4)
                                                {
                                                    std::cout << "Test skipped: (roll=" << roll
                                                              << " > " << run_prob << ")\n";
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
                    }
                }
            }
        }
    }
    return params;
}

// Create an array of parameters to pass to gtest.  Default generator
// that picks all transform types.
inline auto param_generator(const double                             base_prob,
                            const std::vector<std::vector<size_t>>&  v_lengths,
                            const std::vector<fft_precision>&        precision_range,
                            const std::vector<size_t>&               batch_range,
                            const stride_generator&                  istride,
                            const stride_generator&                  ostride,
                            const std::vector<std::vector<size_t>>&  ioffset_range,
                            const std::vector<std::vector<size_t>>&  ooffset_range,
                            const std::vector<fft_result_placement>& place_range,
                            const bool                               planar,

                            const bool run_callbacks = false)
{
    return param_generator_base(base_prob,
                                trans_type_range,
                                v_lengths,
                                precision_range,
                                batch_range,
                                generate_types,
                                istride,
                                ostride,
                                ioffset_range,
                                ooffset_range,
                                place_range,

                                planar,
                                run_callbacks);
}

// Create an array of parameters to pass to gtest.  Only tests complex-type transforms
inline auto param_generator_complex(const double                             base_prob,
                                    const std::vector<std::vector<size_t>>&  v_lengths,
                                    const std::vector<fft_precision>&        precision_range,
                                    const std::vector<size_t>&               batch_range,
                                    const stride_generator&                  istride,
                                    const stride_generator&                  ostride,
                                    const std::vector<std::vector<size_t>>&  ioffset_range,
                                    const std::vector<std::vector<size_t>>&  ooffset_range,
                                    const std::vector<fft_result_placement>& place_range,

                                    const bool planar,
                                    const bool run_callbacks = false)
{
    return param_generator_base(base_prob,
                                trans_type_range_complex,
                                v_lengths,
                                precision_range,
                                batch_range,
                                generate_types,
                                istride,
                                ostride,
                                ioffset_range,
                                ooffset_range,
                                place_range,

                                planar,
                                run_callbacks);
}

// Create an array of parameters to pass to gtest.
inline auto param_generator_real(const double                             base_prob,
                                 const std::vector<std::vector<size_t>>&  v_lengths,
                                 const std::vector<fft_precision>&        precision_range,
                                 const std::vector<size_t>&               batch_range,
                                 const stride_generator&                  istride,
                                 const stride_generator&                  ostride,
                                 const std::vector<std::vector<size_t>>&  ioffset_range,
                                 const std::vector<std::vector<size_t>>&  ooffset_range,
                                 const std::vector<fft_result_placement>& place_range,

                                 const bool planar,
                                 const bool run_callbacks = false)
{
    return param_generator_base(test_prob,
                                trans_type_range_real,
                                v_lengths,
                                precision_range,
                                batch_range,
                                generate_types,
                                istride,
                                ostride,
                                ioffset_range,
                                ooffset_range,
                                place_range,

                                planar,
                                run_callbacks);
}

template <class Tcontainer>
auto param_generator_token(const double base_prob, const Tcontainer& tokens)
{
    std::vector<fft_params> params;

    if(base_prob == 0)
        return params;

    for(const auto& token : tokens)
    {
        fft_params param;
        param.from_token(token);
        param.validate();
        if(param.valid())
        {
            const double roll = hash_prob(random_seed, param.token());
            const double run_prob
                = base_prob * (param.is_planar() ? complex_planar_prob_factor : 1.0)
                  * (param.is_interleaved() ? complex_interleaved_prob_factor : 1.0)
                  * (param.is_real() ? real_prob_factor : 1.0)
                  * (param.is_callback() ? callback_prob_factor : 1.0);

            if(roll > run_prob)
            {
                if(verbose > 4)
                {
                    std::cout << "Test skipped: (roll=" << roll << " > " << run_prob << ")\n";
                }
                continue;
            }

            params.push_back(param);
        }
        else
        {
            std::cout << "invalid token\n";
        }
    }
    return params;
}

#endif
