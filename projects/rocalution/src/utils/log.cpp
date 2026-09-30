/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "log.hpp"
#include "../base/backend_manager.hpp"
#include "def.hpp"
#include "time_functions.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>

namespace rocalution
{

    void _rocalution_open_log_file(void)
    {
        char* str_layer_mode;
        if((str_layer_mode = getenv("ROCALUTION_LAYER")) != NULL)
        {
            // LCOV_EXCL_START
            if(atoi(str_layer_mode) == 1)
            {
                if(_get_backend_descriptor()->log_file != NULL)
                {
                    _rocalution_close_log_file();
                }

                assert(_get_backend_descriptor()->log_file == NULL);

                _get_backend_descriptor()->log_file = new std::ofstream;

                std::ostringstream str_double;
                str_double.precision(20);
                str_double << rocalution_time();
                std::string mid_name = str_double.str();

                std::ostringstream rank;
                rank << _get_backend_descriptor()->rank;
                std::string rank_name = rank.str();

                std::string str_name;
                str_name = "rocalution-rank-" + rank_name + "-" + mid_name + ".log";

                _get_backend_descriptor()->log_file->open(str_name.c_str(),
                                                          std::ios::out | std::ios::trunc);
            }
            // LCOV_EXCL_STOP
        }
    }

    void _rocalution_close_log_file(void)
    {
        if(_get_backend_descriptor()->log_file != NULL)
        {
            // LCOV_EXCL_START
            if(_get_backend_descriptor()->log_file->is_open())
            {
                _get_backend_descriptor()->log_file->close();
                delete _get_backend_descriptor()->log_file;
                _get_backend_descriptor()->log_file = NULL;
            }
            // LCOV_EXCL_STOP
        }
    }

} // namespace rocalution
