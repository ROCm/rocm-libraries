// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "reference_fft_types.h"
#include <algorithm>
#include <aoclfftz.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace reference_fft_backend
{
    inline constexpr const char* name            = "FFTZ";
    inline constexpr bool        supports_wisdom = false;
    inline int                   thread_count    = 1;

    // Precision belongs to the C++ ownership type, independently of the native
    // library's opaque handle representation.
    template <typename Real>
    struct plan_deleter
    {
        void operator()(void* handle) const
        {
            if(handle)
                aoclfftz_destroy(handle);
        }
    };

    template <typename Real>
    using plan = std::unique_ptr<void, plan_deleter<Real>>;

    template <typename Real>
    plan<Real> empty_plan()
    {
        return plan<Real>{};
    }

    template <typename Real>
    plan<Real> make_plan(fft_transform_type                           type,
                         const std::vector<reference_fft::dimension>& dimensions,
                         reference_fft::dimension                     batch,
                         void*                                        input,
                         void*                                        output,
                         bool                                         measure)
    {
        if(measure)
            throw std::invalid_argument("FFTZ does not support FFTW wisdom/measure planning");
        // FFTZ orders dimensions from fastest to slowest, opposite to fft_params.
        std::vector<aoclfftz_dim_t_64_> dims;
        for(auto it = dimensions.rbegin(); it != dimensions.rend(); ++it)
            dims.push_back({it->length, it->input_stride, it->output_stride});
        // FFTZ requires positive vector strides even for a single batch.
        aoclfftz_dim_t_64_ vec{
            batch.length,
            batch.length == 1 && batch.input_stride == 0 ? 1 : batch.input_stride,
            batch.length == 1 && batch.output_stride == 0 ? 1 : batch.output_stride};
        using descriptor = std::conditional_t<std::is_same_v<Real, double>,
                                              aoclfftz_prob_desc_d_64_,
                                              aoclfftz_prob_desc_f_64_>;
        descriptor problem{};
        problem.in       = static_cast<Real*>(input);
        problem.out      = static_cast<Real*>(output);
        problem.dim_rank = static_cast<FFTZ_INT32>(dims.size());
        problem.vec_rank = 1;
        problem.dims     = dims.data();
        problem.vecs     = &vec;
        switch(type)
        {
        case fft_transform_type_complex_forward:
        case fft_transform_type_complex_inverse:
            problem.flags.fft_type = 0;
            break;
        case fft_transform_type_real_forward:
        case fft_transform_type_real_inverse:
            problem.flags.fft_type = 1;
            break;
        default:
            throw std::invalid_argument("Invalid reference transform type");
        }
        problem.flags.fft_direction
            = type == fft_transform_type_complex_inverse || type == fft_transform_type_real_inverse;
        problem.flags.fft_placement         = input != output;
        problem.pthr_fft.num_threads        = thread_count;
        problem.pthr_fft.dynamic_load_model = 1;
        // Use FFTZ's automatic runtime ISA selection, as its FFTW wrapper does.
        problem.cntrl_params.opt_level = 4;
        void* handle;
        if constexpr(std::is_same_v<Real, double>)
            handle = aoclfftz_setup_d_64_(&problem);
        else
            handle = aoclfftz_setup_f_64_(&problem);
        if(!handle)
            throw std::runtime_error("FFTZ reference planning failed");
        return plan<Real>{handle};
    }

    template <typename Real>
    void execute(fft_transform_type, const plan<Real>& p, void* input, void* output)
    {
        if(aoclfftz_execute_io(p.get(), input, output) != AOCLFFTZ_SUCCESS)
            throw std::runtime_error("FFTZ reference execution failed");
    }

    template <typename Real>
    void cleanup()
    {
        // Each native solution owns its resources; FFTZ has no global plan cache.
    }

    inline void initialize(int threads)
    {
        thread_count = std::max(1, threads);
    }

    inline void import_wisdom(const std::string&)
    {
        throw std::invalid_argument("FFTZ does not support FFTW wisdom");
    }

    inline void export_wisdom(const std::string&)
    {
        throw std::invalid_argument("FFTZ does not support FFTW wisdom");
    }
}
