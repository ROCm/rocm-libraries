/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <set>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <rocblas/rocblas.h>

#include "program_options.hpp"

using variables_map = roc::variables_map;
using variable_value = roc::variable_value;

class Arguments : private std::map<std::string, variable_value>
{
    using base = std::map<std::string, variable_value>;

    // names of arguments that have not yet been used by tests
    std::set<std::string> to_consume;

public:
    // test options
    rocblas_int norm_check = 0;
    rocblas_int unit_check = 1;
    rocblas_int hash_check = 0;
    rocblas_int timing = 0;
    rocblas_int perf = 0;
    rocblas_int singular = 0;
    rocblas_int iters = 5;
    rocblas_int alg_mode = 0;
    rocblas_int mem_query = 0;
    rocblas_int profile = 0;
    rocblas_int profile_kernels = 0;
    rocblas_int batch_count = 1;

    // get and set function arguments
    template <typename T>
    const T& peek(const std::string& name) const
    {
        return at(name).as<T>();
    }

    template <typename T>
    const T& get(const std::string& name)
    {
        to_consume.erase(name);
        auto val = find(name);
        if(val != end() && !val->second.empty())
            return val->second.as<T>();
        else
            throw std::invalid_argument("No value provided for " + name);
    }

    template <typename T>
    const T get(const std::string& name, const T& default_value)
    {
        to_consume.erase(name);
        auto val = find(name);
        if(val != end() && !val->second.empty() && !val->second.defaulted())
            return val->second.as<T>();
        else
            return default_value;
    }

    template <typename T>
    void set(const std::string& name, const T& val)
    {
        to_consume.insert(name);
        base::operator[](name) = variable_value(val, false);
    }

    void populate(const variables_map& vm)
    {
        for(auto& pair : vm)
        {
            base::operator[](pair.first) = pair.second;

            if(!pair.second.empty() && !pair.second.defaulted())
                to_consume.insert(pair.first);
        }

        // remove test arguments
        to_consume.erase("help");
        to_consume.erase("function");
        to_consume.erase("precision");
        to_consume.erase("batch_count");
        to_consume.erase("verify");
        to_consume.erase("iters");
        to_consume.erase("alg_mode");
        to_consume.erase("mem_query");
        to_consume.erase("profile");
        to_consume.erase("profile_kernels");
        to_consume.erase("perf");
        to_consume.erase("singular");
        to_consume.erase("device");
        to_consume.erase("hash");
    }

    void clear()
    {
        to_consume.clear();
        base::clear();
    }

    // validate function arguments
    void validate_precision(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char precision = val->second.as<char>();
        if(precision != 's' && precision != 'd' && precision != 'c' && precision != 'z')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_operation(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char trans = val->second.as<char>();
        if(trans != 'N' && trans != 'T' && trans != 'C')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_side(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char side = val->second.as<char>();
        if(side != 'L' && side != 'R' && side != 'B')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_fill(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char uplo = val->second.as<char>();
        if(uplo != 'U' && uplo != 'L' && uplo != 'F')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_diag(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char diag = val->second.as<char>();
        if(diag != 'N' && diag != 'U')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_direct(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char direct = val->second.as<char>();
        if(direct != 'F' && direct != 'B')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_pivot(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char pivot = val->second.as<char>();
        if(pivot != 'V' && pivot != 'T' && pivot != 'B')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_storev(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char storev = val->second.as<char>();
        if(storev != 'R' && storev != 'C')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_svect(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char svect = val->second.as<char>();
        if(svect != 'A' && svect != 'S' && svect != 'V' && svect != 'O' && svect != 'N')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_srange(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char range = val->second.as<char>();
        if(range != 'A' && range != 'V' && range != 'I')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_workmode(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char workmode = val->second.as<char>();
        if(workmode != 'O' && workmode != 'I')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_evect(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char evect = val->second.as<char>();
        if(evect != 'V' && evect != 'I' && evect != 'N')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_erange(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char range = val->second.as<char>();
        if(range != 'A' && range != 'V' && range != 'I')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_eorder(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char order = val->second.as<char>();
        if(order != 'B' && order != 'E')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_esort(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char sort = val->second.as<char>();
        if(sort != 'A' && sort != 'N')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_itype(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char itype = val->second.as<char>();
        if(itype != '1' && itype != '2' && itype != '3')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_rfinfo_mode(const std::string name) const
    {
        auto val = find(name);
        if(val == end())
            return;

        char mode = val->second.as<char>();
        if(mode != '1' && mode != '2')
            throw std::invalid_argument("Invalid value for " + name);
    }

    void validate_consumed() const
    {
        if(!to_consume.empty())
            throw std::invalid_argument(
                fmt::format("Not all arguments were consumed: {}", fmt::join(to_consume, " ")));
    }
};
