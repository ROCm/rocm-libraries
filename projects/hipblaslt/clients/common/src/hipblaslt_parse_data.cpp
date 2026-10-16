/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "hipblaslt_parse_data.hpp"
#include "hipblaslt_arguments.hpp"
#include "hipblaslt_data.hpp"
#include "utility.hpp"
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>

namespace
{
    struct ValidationOverrides
    {
        std::optional<bool>        freivalds, verify, bounds;
        std::optional<int32_t>     iterations, injection;
        std::optional<std::string> placement;
        std::optional<size_t>      placementOffset;
    } validationOverrides;

    template <typename T>
    T parseInteger(std::string_view value, std::string_view option)
    {
        T result{};
        const auto [end, error]
            = std::from_chars(value.data(), value.data() + value.size(), result);
        if(error != std::errc{} || end != value.data() + value.size())
            throw std::invalid_argument(std::string(option) + " requires an integer");
        return result;
    }

    bool parseValidationOption(int& i, int argc, char** argv)
    {
        const std::string_view token(argv[i]);
        const auto             equals = token.find('=');
        const auto             option = token.substr(0, equals);
        auto                   value  = [&]() -> std::string_view {
            if(equals != std::string_view::npos)
                return token.substr(equals + 1);
            if(i + 1 >= argc)
                throw std::invalid_argument(std::string(option) + " requires a value");
            return argv[++i];
        };
        auto boolean = [&]() {
            std::string_view text = "true";
            if(equals != std::string_view::npos)
                text = value();
            else if(i + 1 < argc)
            {
                const std::string_view next(argv[i + 1]);
                if(next == "0" || next == "1" || next == "true" || next == "false")
                    text = argv[++i];
            }
            if(text == "true" || text == "1")
                return true;
            if(text == "false" || text == "0")
                return false;
            throw std::invalid_argument(std::string(option) + " requires true, false, 1 or 0");
        };
        if(option == "--freivalds")
            validationOverrides.freivalds = boolean();
        else if(option == "--verify" || option == "-v")
            validationOverrides.verify = boolean();
        else if(option == "--bounds-check")
            validationOverrides.bounds = boolean();
        else if(option == "--validation-iters")
            validationOverrides.iterations = parseInteger<int32_t>(value(), option);
        else if(option == "--validation-inject")
            validationOverrides.injection = parseInteger<int32_t>(value(), option);
        else if(option == "--placement")
            validationOverrides.placement = std::string(value());
        else if(option == "--placement-offset")
            validationOverrides.placementOffset = parseInteger<size_t>(value(), option);
        else
            return false;
        return true;
    }
}

void hipblaslt_apply_validation_options(Arguments& arg)
{
    const auto& options = validationOverrides;
    if(options.freivalds)
    {
        arg.freivalds = *options.freivalds;
        if(arg.freivalds)
            arg.unit_check = arg.norm_check = arg.allclose_check = 0;
    }
    if(options.verify)
    {
        // Use the ordinary norm-based reference policy in both clients. The
        // unit-test ULP policy remains available through YAML unit_check.
        arg.unit_check = 0;
        arg.norm_check = arg.allclose_check = *options.verify;
        if(*options.verify)
            arg.norm_check_assert = true;
        else
            arg.ulp_check = 0;
    }
    if(arg.ulp_check)
        arg.norm_check = 1;
    if(options.bounds)
        arg.bounds_check = *options.bounds;
    if(options.iterations)
        arg.validation_iters = *options.iterations;
    if(options.injection)
        arg.validation_inject = *options.injection;
    if(options.placement)
    {
        if(options.placement->size() >= sizeof(arg.placement))
            throw std::invalid_argument("--placement operand name is too long");
        std::strcpy(arg.placement, options.placement->c_str());
    }
    if(options.placementOffset)
        arg.placement_offset = *options.placementOffset;
}

// Parse YAML data
static std::string hipblaslt_parse_yaml(const std::string& yaml)
{
#ifdef _WIN32
    // Explicitly run via `python.exe`, without relying on the .py file being
    // treated as an executable that should be run via the python interpreter.
    std::string python_command_launcher = "python ";
#else
    // Rely on the shebang in the file, e.g. `#!/usr/bin/env python3`.
    std::string python_command_launcher = "";
#endif

    // TODO: This function is inherently unsafe because it returns a string vs an open
    // file handle which will block further colliding creates. See comments in
    // hipblaslt_tempname() and under no circumstances copy this to new code.
    std::string tmp     = hipblaslt_tempname();
    auto        exepath = hipblaslt_exepath();
    auto cmd = python_command_launcher + exepath + "hipblaslt_gentest.py --template " + exepath
               + "hipblaslt_template.yaml -o " + tmp + " " + yaml;
    hipblaslt_cerr << cmd << std::endl;

#ifdef _WIN32
    int status = std::system(cmd.c_str());
    if(status != 0)
        exit(EXIT_FAILURE);
#else
    int status = system(cmd.c_str());
    if(status == -1 || !WIFEXITED(status) || WEXITSTATUS(status))
        exit(EXIT_FAILURE);
#endif

    return tmp;
}

// Shared options apply to direct GEMMs and every record loaded from YAML or binary data.
bool hipblaslt_parse_data(int& argc, char** argv, const std::string& default_file)
{
    std::string filename;
    char**      argv_p = argv + 1;
    bool        help = false, yaml = false;

    // Scan, process and remove shared options before the client-specific parser runs.
    for(int i = 1; argv[i]; ++i)
    {
        if(parseValidationOption(i, argc, argv))
            continue;
        else if(!strcmp(argv[i], "--data") || !strcmp(argv[i], "--yaml"))
        {
            if(!strcmp(argv[i], "--yaml"))
            {
                yaml = true;
            }

            if(filename != "")
            {
                hipblaslt_cerr << "Only one of the --yaml and --data options may be specified"
                               << std::endl;
                exit(EXIT_FAILURE);
            }

            if(!argv[i + 1] || !argv[i + 1][0])
            {
                hipblaslt_cerr << "The " << argv[i] << " option requires an argument" << std::endl;
                exit(EXIT_FAILURE);
            }
            filename = argv[++i];
        }
        else
        {
            *argv_p++ = argv[i];
            if(!help && (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")))
            {
                help = true;
                hipblaslt_cout << "\n"
                               << argv[0] << " [ --data <path> | --yaml <path> ] <options> ...\n"
                               << "  --freivalds [0|1]        Select HostNumerics Freivalds; "
                                  "requires certifiable integral inputs.\n"
                               << "  --verify, -v [0|1]       Compare with a dense CPU reference "
                                  "(also with --freivalds).\n"
                               << "  --bounds-check [0|1]     Poison input padding and check "
                                  "output padding/stores.\n"
                               << "  --validation-iters <n>   Checked launches per solution, "
                                  "outside benchmark timing (default 1).\n"
                               << "  --validation-inject <i>  Corrupt output on checked iteration "
                                  "i; validation must fail. -1 disables.\n"
                               << "  --placement <operand>    Place a buffer across a 4 GiB "
                                  "address boundary; requires --bounds-check.\n"
                               << "  --placement-offset <n>   Bytes before that boundary (0 "
                                  "selects the middle).\n"
                               << std::endl;
            }
        }
    }

    // argc and argv contain remaining options and non-option arguments
    *argv_p = nullptr;
    argc    = argv_p - argv;

    if(filename == "-")
        filename = "/dev/stdin";
    else if(filename == "")
        filename = default_file;

    if(yaml)
        filename = hipblaslt_parse_yaml(filename);

    if(filename != "")
    {
        HipBlasLt_TestData::set_filename(filename, yaml);
        return true;
    }

    return false;
}
