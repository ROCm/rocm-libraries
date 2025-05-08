/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "argument_model.hpp"
#include "frequency_monitor.hpp"

// FLOPS/CLOCK/CU values for gfx942
auto rocblas_get_flops_per_clock_per_cu_gfx942(rocblas_datatype type)
{
    if(type == rocblas_datatype_f32_r || type == rocblas_datatype_f64_r)
        return 256;
    else if(type == rocblas_datatype_f16_r || type == rocblas_datatype_bf16_r)
        return 2048;
    else if(type == rocblas_datatype_i8_r)
        return 4096;
    else
        return 0;
}

// this should have been a member variable but due to the complex variadic template this singleton allows global control
static bool log_function_name = false;

void ArgumentModel_set_log_function_name(bool f)
{
    log_function_name = f;
}

bool ArgumentModel_get_log_function_name()
{
    return log_function_name;
}

static bool log_datatype = false;

void ArgumentModel_set_log_datatype(bool d)
{
    log_datatype = d;
}

bool ArgumentModel_get_log_datatype()
{
    return log_datatype;
}

void ArgumentModel_log_efficiency(rocblas_internal_ostream& name_line,
                                  rocblas_internal_ostream& val_line,
                                  const Arguments&          arg,
                                  double                    rocblas_gflops)
{
    FrequencyMonitor& frequency_monitor = getFrequencyMonitor();
    if(!frequency_monitor.enabled())
        return;

    if(rocblas_internal_get_arch_name() == "gfx942"
       && rocblas_get_flops_per_clock_per_cu_gfx942(arg.a_type) != 0)
    {
        double theoretical_gflops = rocblas_get_flops_per_clock_per_cu_gfx942(arg.a_type)
                                    * frequency_monitor.getCuCount()
                                    * frequency_monitor.getLowestAverageSYSCLK() * 0.001;
        name_line << ",efficiency";
        val_line << "," << (rocblas_gflops / theoretical_gflops) * 100;
    }
}

void ArgumentModel_log_frequencies(rocblas_internal_ostream& name_line,
                                   rocblas_internal_ostream& val_line)
{

    FrequencyMonitor& frequency_monitor = getFrequencyMonitor();
    if(!frequency_monitor.enabled())
        return;
    if(!frequency_monitor.detailedReport())
    {
        name_line << ",lowest-avg-freq";
        val_line << "," << frequency_monitor.getLowestAverageSYSCLK();

        name_line << ",lowest-median-freq";
        val_line << "," << frequency_monitor.getLowestMedianSYSCLK();
    }
    else
    {
        auto allAvgSYSCLK = frequency_monitor.getAllAverageSYSCLK();
        for(int i = 0; i < allAvgSYSCLK.size(); i++)
        {
            name_line << ",avg-freq_" << i;
            val_line << "," << allAvgSYSCLK[i];
        }

        auto allMedianSYSCLK = frequency_monitor.getAllMedianSYSCLK();
        for(int i = 0; i < allMedianSYSCLK.size(); i++)
        {
            name_line << ",median-freq_" << i;
            val_line << "," << allMedianSYSCLK[i];
        }
    }

    name_line << ",avg-MCLK";
    val_line << "," << frequency_monitor.getAverageMEMCLK();

    name_line << ",median-MCLK";
    val_line << "," << frequency_monitor.getMedianMEMCLK();
}
