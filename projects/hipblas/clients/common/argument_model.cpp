/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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

void ArgumentModel_log_frequencies(std::stringstream& name_line, std::stringstream& val_line)
{
    FrequencyMonitor& frequency_monitor = getFrequencyMonitor();
    if(!frequency_monitor.enabled())
        return;

    if(!frequency_monitor.detailedReport())
    {
        name_line << "lowest-avg-freq,lowest-median-freq,";
        val_line << frequency_monitor.getLowestAverageSYSCLK() << ", "
                 << frequency_monitor.getLowestMedianSYSCLK() << ", ";
    }
    else
    {
        auto all_avg = frequency_monitor.getAllAverageSYSCLK();
        for(size_t i = 0; i < all_avg.size(); i++)
        {
            name_line << "avg-freq_" << i << ",";
            val_line << all_avg[i] << ", ";
        }

        auto all_median = frequency_monitor.getAllMedianSYSCLK();
        for(size_t i = 0; i < all_median.size(); i++)
        {
            name_line << "median-freq_" << i << ",";
            val_line << all_median[i] << ", ";
        }
    }

    name_line << "avg-MCLK,median-MCLK,";
    val_line << frequency_monitor.getAverageMEMCLK() << ", " << frequency_monitor.getMedianMEMCLK()
             << ", ";
}
