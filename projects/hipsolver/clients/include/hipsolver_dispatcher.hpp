/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * ************************************************************************ */

#pragma once

#include "../rocsolvercommon/rocsolver_arguments.hpp"
#include <map>
#include <string>

#include "testing_gebrd.hpp"
#include "testing_gels.hpp"
#include "testing_geqrf.hpp"
#include "testing_gesv.hpp"
#include "testing_gesvd.hpp"
#include "testing_gesvda.hpp"
#include "testing_gesvdj.hpp"
#include "testing_getrf.hpp"
#include "testing_getrs.hpp"
#include "testing_orgbr_ungbr.hpp"
#include "testing_orgqr_ungqr.hpp"
#include "testing_orgtr_ungtr.hpp"
#include "testing_ormqr_unmqr.hpp"
#include "testing_ormtr_unmtr.hpp"
#include "testing_potrf.hpp"
#include "testing_potri.hpp"
#include "testing_potrs.hpp"
#include "testing_syevd_heevd.hpp"
#include "testing_syevdx_heevdx.hpp"
#include "testing_syevj_heevj.hpp"
#include "testing_sygvd_hegvd.hpp"
#include "testing_sygvdx_hegvdx.hpp"
#include "testing_sygvj_hegvj.hpp"
#include "testing_sytrd_hetrd.hpp"
#include "testing_sytrf.hpp"

#ifdef HAVE_HIPSPARSE
#include "testing_csrlsvchol.hpp"
#include "testing_csrlsvqr.hpp"
#endif

struct str_less
{
    bool operator()(const char* a, const char* b) const
    {
        return strcmp(a, b) < 0;
    }
};

// Map from const char* to function taking Arguments& using lexicographical comparison
using func_map = std::map<const char*, void (*)(Arguments&), str_less>;

// Function dispatcher for hipSOLVER tests
class hipsolver_dispatcher
{
    template <typename T>
    static hipsolverStatus_t run_function(const char* name, Arguments& argus)
    {
        // Map for functions that support all precisions
        static const func_map map = {
            {"gebrd", testing_gebrd<API_NORMAL, false, false, T>},
            {"gels", testing_gels<API_NORMAL, false, false, false, T>},
            {"geqrf", testing_geqrf<API_NORMAL, false, false, T, int, int>},
            {"geqrf_64", testing_geqrf<API_COMPAT, false, false, T, int64_t, size_t>},
            {"gesv", testing_gesv<API_NORMAL, false, false, false, T>},
            {"gesvd", testing_gesvd<API_NORMAL, false, false, false, T>},
            {"gesvda_strided_batched", testing_gesvda<API_COMPAT, false, true, T>},
            {"gesvdj", testing_gesvdj<API_NORMAL, false, false, T>},
            {"gesvdj_batched", testing_gesvdj<API_NORMAL, false, true, T>},
            {"getrf", testing_getrf<API_NORMAL, false, false, false, T, int, int>},
            {"getrf_64", testing_getrf<API_COMPAT, false, false, false, T, int64_t, size_t>},
            {"getrs", testing_getrs<API_NORMAL, false, false, T, int, int>},
            {"getrs_64", testing_getrs<API_COMPAT, false, false, T, int64_t, size_t>},
            {"potrf", testing_potrf<API_NORMAL, false, false, T>},
            {"potrf_batched", testing_potrf<API_NORMAL, true, false, T>},
            {"potri", testing_potri<API_NORMAL, false, false, T>},
            {"potrs", testing_potrs<API_NORMAL, false, false, T>},
            {"potrs_batched", testing_potrs<API_NORMAL, true, false, T>},
            {"sytrf", testing_sytrf<API_NORMAL, false, false, T>},
        };

        // Grab function from the map and execute
        auto match = map.find(name);
        if(match != map.end())
        {
            match->second(argus);
            return HIPSOLVER_STATUS_SUCCESS;
        }
        else
            return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
    static hipsolverStatus_t run_function_limited_precision(const char* name, Arguments& argus)
    {
        // Map for functions that support single and double precisions
        static const func_map map_real = {
            {"orgbr", testing_orgbr_ungbr<API_NORMAL, T>},
            {"orgqr", testing_orgqr_ungqr<API_NORMAL, T>},
            {"orgtr", testing_orgtr_ungtr<API_NORMAL, T>},
            {"ormqr", testing_ormqr_unmqr<API_NORMAL, T>},
            {"ormtr", testing_ormtr_unmtr<API_NORMAL, T>},
            {"syevd", testing_syevd_heevd<API_NORMAL, false, false, T>},
            {"syevdx", testing_syevdx_heevdx<API_NORMAL, false, false, T>},
            {"syevj", testing_syevj_heevj<API_NORMAL, false, false, T>},
            {"syevj_batched", testing_syevj_heevj<API_NORMAL, false, true, T>},
            {"sygvd", testing_sygvd_hegvd<API_NORMAL, false, false, T>},
            {"sygvdx", testing_sygvdx_hegvdx<API_NORMAL, false, false, T>},
            {"sygvj", testing_sygvj_hegvj<API_NORMAL, false, false, T>},
            {"sytrd", testing_sytrd_hetrd<API_NORMAL, false, false, T>},

#ifdef HAVE_HIPSPARSE
            {"csrlsvchol", testing_csrlsvchol<false, T>},
            {"csrlsvcholHost", testing_csrlsvchol<true, T>},
            {"csrlsvqr", testing_csrlsvqr<false, T>},
#endif
        };

        // Grab function from the map and execute
        auto match = map_real.find(name);
        if(match != map_real.end())
        {
            match->second(argus);
            return HIPSOLVER_STATUS_SUCCESS;
        }
        else
            return HIPSOLVER_STATUS_INVALID_VALUE;
    }

    template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
    static hipsolverStatus_t run_function_limited_precision(const char* name, Arguments& argus)
    {
        // Map for functions that support single complex and double complex precisions
        static const func_map map_complex = {
            {"ungbr", testing_orgbr_ungbr<API_NORMAL, T>},
            {"ungqr", testing_orgqr_ungqr<API_NORMAL, T>},
            {"ungtr", testing_orgtr_ungtr<API_NORMAL, T>},
            {"unmqr", testing_ormqr_unmqr<API_NORMAL, T>},
            {"unmtr", testing_ormtr_unmtr<API_NORMAL, T>},
            {"heevd", testing_syevd_heevd<API_NORMAL, false, false, T>},
            {"heevdx", testing_syevdx_heevdx<API_NORMAL, false, false, T>},
            {"heevj", testing_syevj_heevj<API_NORMAL, false, false, T>},
            {"heevj_batched", testing_syevj_heevj<API_NORMAL, false, true, T>},
            {"hegvd", testing_sygvd_hegvd<API_NORMAL, false, false, T>},
            {"hegvdx", testing_sygvdx_hegvdx<API_NORMAL, false, false, T>},
            {"hegvj", testing_sygvj_hegvj<API_NORMAL, false, false, T>},
            {"hetrd", testing_sytrd_hetrd<API_NORMAL, false, false, T>},
        };

        // Grab function from the map and execute
        auto match = map_complex.find(name);
        if(match != map_complex.end())
        {
            match->second(argus);
            return HIPSOLVER_STATUS_SUCCESS;
        }
        else
            return HIPSOLVER_STATUS_INVALID_VALUE;
    }

public:
    static void invoke(const std::string& name, char precision, Arguments& argus)
    {
        hipsolverStatus_t status;

        if(precision == 's')
            status = run_function<float>(name.c_str(), argus);
        else if(precision == 'd')
            status = run_function<double>(name.c_str(), argus);
        else if(precision == 'c')
            status = run_function<hipsolverComplex>(name.c_str(), argus);
        else if(precision == 'z')
            status = run_function<hipsolverDoubleComplex>(name.c_str(), argus);
        else
            throw std::invalid_argument("Invalid value for --precision");

        if(status == HIPSOLVER_STATUS_INVALID_VALUE)
        {
            if(precision == 's')
                status = run_function_limited_precision<float>(name.c_str(), argus);
            else if(precision == 'd')
                status = run_function_limited_precision<double>(name.c_str(), argus);
            else if(precision == 'c')
                status = run_function_limited_precision<hipsolverComplex>(name.c_str(), argus);
            else if(precision == 'z')
                status
                    = run_function_limited_precision<hipsolverDoubleComplex>(name.c_str(), argus);
        }

        if(status == HIPSOLVER_STATUS_INVALID_VALUE)
        {
            std::string msg = "Invalid combination --function ";
            msg += name;
            msg += " --precision ";
            msg += precision;
            throw std::invalid_argument(msg);
        }
    }
};
