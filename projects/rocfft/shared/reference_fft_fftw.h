// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "fftw_transform.h"
#include "reference_fft_types.h"
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

namespace reference_fft_backend
{
    inline constexpr const char* name            = "FFTW";
    inline constexpr bool        supports_wisdom = true;

    template <typename Real>
    using plan = fftw_plan_wrapper_t<Real>;

    template <typename Real>
    plan<Real> empty_plan()
    {
        return fftw_trait<Real>::make_wrapper(nullptr);
    }

    template <typename Real>
    plan<Real> make_plan(fft_transform_type                           type,
                         const std::vector<reference_fft::dimension>& dimensions,
                         reference_fft::dimension                     batch,
                         void*                                        input,
                         void*                                        output,
                         bool                                         measure)
    {
        std::vector<fftw_iodim64> dims;
        for(const auto& d : dimensions)
            dims.push_back({d.length, d.input_stride, d.output_stride});
        fftw_iodim64   batches{batch.length, batch.input_stride, batch.output_stride};
        const unsigned flags  = measure ? FFTW_MEASURE : FFTW_ESTIMATE;
        plan<Real>     result = empty_plan<Real>();
        switch(type)
        {
        case fft_transform_type_complex_forward:
        case fft_transform_type_complex_inverse:
            result = fftw_plan_guru64_dft<Real>(dims.size(),
                                                dims.data(),
                                                1,
                                                &batches,
                                                reinterpret_cast<fftw_complex_t<Real>*>(input),
                                                reinterpret_cast<fftw_complex_t<Real>*>(output),
                                                type == fft_transform_type_complex_forward ? -1 : 1,
                                                flags);
            break;
        case fft_transform_type_real_forward:
            result = fftw_plan_guru64_r2c<Real>(dims.size(),
                                                dims.data(),
                                                1,
                                                &batches,
                                                static_cast<Real*>(input),
                                                reinterpret_cast<fftw_complex_t<Real>*>(output),
                                                flags);
            break;
        case fft_transform_type_real_inverse:
            result = fftw_plan_guru64_c2r<Real>(dims.size(),
                                                dims.data(),
                                                1,
                                                &batches,
                                                reinterpret_cast<fftw_complex_t<Real>*>(input),
                                                static_cast<Real*>(output),
                                                flags);
            break;
        default:
            throw std::invalid_argument("Invalid reference transform type");
        }
        if(!result)
            throw std::runtime_error("FFTW reference planning failed");
        return result;
    }

    template <typename Real>
    void execute(fft_transform_type type, const plan<Real>& p, void* input, void* output)
    {
        if(type == fft_transform_type_real_forward)
        {
            if constexpr(std::is_same_v<Real, double>)
                fftw_execute_dft_r2c(
                    p.get(), static_cast<double*>(input), static_cast<fftw_complex*>(output));
            else
                fftwf_execute_dft_r2c(
                    p.get(), static_cast<float*>(input), static_cast<fftwf_complex*>(output));
        }
        else if(type == fft_transform_type_real_inverse)
        {
            if constexpr(std::is_same_v<Real, double>)
                fftw_execute_dft_c2r(
                    p.get(), static_cast<fftw_complex*>(input), static_cast<double*>(output));
            else
                fftwf_execute_dft_c2r(
                    p.get(), static_cast<fftwf_complex*>(input), static_cast<float*>(output));
        }
        else
        {
            if constexpr(std::is_same_v<Real, double>)
                fftw_execute_dft(
                    p.get(), static_cast<fftw_complex*>(input), static_cast<fftw_complex*>(output));
            else
                fftwf_execute_dft(p.get(),
                                  static_cast<fftwf_complex*>(input),
                                  static_cast<fftwf_complex*>(output));
        }
    }

    template <typename Real>
    void cleanup()
    {
        if constexpr(std::is_same_v<Real, double>)
            fftw_cleanup();
        else
            fftwf_cleanup();
    }

    inline void initialize(int threads)
    {
#ifdef FFTW_MULTITHREAD
        if(!fftw_init_threads() || !fftwf_init_threads())
            throw std::runtime_error("FFTW thread initialization failed");
        fftw_plan_with_nthreads(threads);
        fftwf_plan_with_nthreads(threads);
#endif
    }

    inline void import_wisdom(const std::string& filename)
    {
        std::ifstream input(filename);
        std::string   double_wisdom, float_wisdom, line;
        std::string*  current = nullptr;
        while(std::getline(input, line))
        {
            if(line.rfind("(fftw", 0) == 0)
            {
                if(line.find("fftw_wisdom") != std::string::npos)
                    current = &double_wisdom;
                else if(line.find("fftwf_wisdom") != std::string::npos)
                    current = &float_wisdom;
            }
            if(current)
                *current += line + "\n";
            if(line.rfind(")", 0) == 0)
                current = nullptr;
        }
        fftw_import_wisdom_from_string(double_wisdom.c_str());
        fftwf_import_wisdom_from_string(float_wisdom.c_str());
    }

    inline void export_wisdom(const std::string& filename)
    {
        std::ofstream output(filename);
        if(!output)
            throw std::runtime_error("Cannot write reference wisdom file: " + filename);
        char* d = fftw_export_wisdom_to_string();
        char* f = fftwf_export_wisdom_to_string();
        if(d)
            output << d;
        if(f)
            output << f;
        fftw_free(d);
        fftwf_free(f);
    }
}
