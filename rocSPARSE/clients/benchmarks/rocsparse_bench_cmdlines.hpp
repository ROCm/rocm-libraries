/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#pragma once

#include "rocsparse/rocsparse-auxiliary.h"
#include "rocsparse_clients_envariables.hpp"
#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>

//
// @brief The role of this class is to expand a command line into multiple command lines.
// @details
//
// What is expanding a command line into multiple command lines?
// Let's consider the following command line './foo -m 10' where option '-m' of ./foo
// takes only one argument, here 10.
//
// An expansion mechanism is implemented in this class to provide the set of command lines
// './foo -m 10',
// './foo -m 2',
// './foo -m 7',
// './foo -m -4'
//
// from: ./foo -m 10 2 7 -4
//
// It allows to generate a set of command lines to be useful in a benchmarking context.
//
// Rules:
// - any keyword starting with '-' is considered as an option.
// - each option having exactly one argument is subject to a possible expansion, there is no limit on the number of options to expand.
//
//
// Number of command lines generated : product of all the options' number (>=1) of arguments
// examples:
//  cmd: './foo -m 10 2 7 -k 32 -l f -v' gives
//       './foo -m 10 -k 32 -l f'
//       './foo -m  2 -k 32 -l f'
//       './foo -m  7 -k 32 -l f'
//  num cmds: max(1,3) * max(1,1) * max(1,0) = 3

//  cmd: './foo -m 10 2 7 -k 32 64 -l f g' gives
//       './foo -m 10 -k 32 -l f'
//       './foo -m  2 -k 32 -l f'
//       './foo -m  7 -k 32 -l f'
//       './foo -m 10 -k 64 -l f'
//       './foo -m  2 -k 64 -l f'
//       './foo -m  7 -k 64 -l f'
//       './foo -m 10 -k 32 -l g'
//       './foo -m  2 -k 32 -l g'
//       './foo -m  7 -k 32 -l g'
//       './foo -m 10 -k 64 -l g'
//       './foo -m  2 -k 64 -l g'
//       './foo -m  7 -k 64 -l g'
//  num cmds: max(1,3) * max(1,2) * max(1,2) = 12
//
// Specific options:
//
// option: --bench-x, to precede the option the user want to be the first one.
// example
//  cmd: './foo -m 10 2 7 --bench-x -k 32 64 -l f g' gives
//       './foo -m 32 -k 10 -l f'
//       './foo -m 64 -k 10 -l f'
//       './foo -m 32 -k 2 -l f'
//       './foo -m 64 -k 2 -l f'
//       './foo -m 32 -k 7 -l f'
//       './foo -m 64 -k 7 -l f'
//       './foo -m 32 -k 10 -l g'
//       './foo -m 64 -k 10 -l g'
//       './foo -m 32 -k 2 -l g'
//       './foo -m 64 -k 2 -l g'
//       './foo -m 32 -k 7 -l g'
//       './foo -m 64 -k 7 -l g'
//
// option: --bench-o, output filename.
// option: --bench-n, number of runs.
// option: --bench-std, prevent from standard output to be disabled.
//

class rocsparse_bench_cmdlines
{
private:
    struct val
    {
        //
        // Everything is public.
        //
    public:
        int    argc{};
        char** argv{};
        val&   operator=(const val&) = delete;
        ~val()
        {
            if(this->argv != nullptr)
            {
                delete[] this->argv;
                this->argv = nullptr;
            }
        }

        val(){};
        val(const val& v) = delete;
        explicit val(int n)
            : argc(n)
        {
            this->argv = new char*[this->argc];
        }

        val& operator()(int n)
        {
            this->argc = n;
            if(this->argv)
            {
                delete[] this->argv;
            }
            this->argv = new char*[this->argc];
            return *this;
        }
    };

    struct cmdline
    {

    public:
        //
        // @brief Return the output filename.
        //
        const char* get_ofilename() const
        {
            return this->m_ofilename;
        };

        //
        // @brief Return the number of plots.
        //
        int get_nplots() const
        {
            return this->get_nsamples() / this->m_options[this->m_option_index_x].args.size();
        };

        int get_noptions_x() const
        {
            return this->m_options[this->m_option_index_x].args.size();
        };
        int get_noptions() const
        {
            return this->m_options.size();
        };
        int get_option_nargs(int i)
        {
            return this->m_options[i].args.size();
        }
        const char* get_option_arg(int i, int j)
        {
            return this->m_options[i].args[j].name;
        }
        const char* get_option_name(int i)
        {
            return this->m_options[i].name;
        }
        int get_nsamples() const
        {
            return this->m_nsamples;
        }

        int get_option_index_x() const
        {
            return this->m_option_index_x;
        }

        int get_nruns() const
        {
            return this->m_bench_nruns;
        }

        bool is_stdout_disabled() const
        {
            return this->m_is_stdout_disabled;
        }

        bool no_rawdata() const
        {
            return this->m_no_rawdata;
        }

        //
        // Constructor.
        //
        cmdline(int argc, char** argv)
        {
            if(detect_flag(argc, argv, "--rocsparse-clients-enable-test-debug-arguments"))
            {
                rocsparse_clients_envariables::set(
                    rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS, true);
            }

            if(detect_flag(argc, argv, "--rocsparse-clients-disable-test-debug-arguments"))
            {
                rocsparse_clients_envariables::set(
                    rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS, false);
            }

            const bool enable_rocsparse_debug = detect_flag(argc, argv, "--rocsparse-enable-debug");
            if(enable_rocsparse_debug)
            {
                rocsparse_enable_debug();
            }
            const bool disable_rocsparse_debug
                = detect_flag(argc, argv, "--rocsparse-disable-debug");
            if(disable_rocsparse_debug)
            {
                rocsparse_disable_debug();
            }

            const bool enable_rocsparse_debug_verbose
                = detect_flag(argc, argv, "--rocsparse-enable-debug-verbose");
            if(enable_rocsparse_debug_verbose)
            {
                rocsparse_enable_debug_verbose();
            }
            const bool disable_rocsparse_debug_verbose
                = detect_flag(argc, argv, "--rocsparse-disable-debug-verbose");
            if(disable_rocsparse_debug_verbose)
            {
                rocsparse_disable_debug_verbose();
            }

            const bool enable_rocsparse_debug_arguments
                = detect_flag(argc, argv, "--rocsparse-enable-debug-arguments");
            if(enable_rocsparse_debug_arguments)
            {
                rocsparse_enable_debug_arguments();
            }
            const bool disable_rocsparse_debug_arguments
                = detect_flag(argc, argv, "--rocsparse-disable-debug-arguments");
            if(disable_rocsparse_debug_arguments)
            {
                rocsparse_disable_debug_arguments();
            }

            const bool enable_rocsparse_debug_arguments_verbose
                = detect_flag(argc, argv, "--rocsparse-enable-debug-arguments-verbose");
            if(enable_rocsparse_debug_arguments_verbose)
            {
                rocsparse_enable_debug_arguments_verbose();
            }
            const bool disable_rocsparse_debug_arguments_verbose
                = detect_flag(argc, argv, "--rocsparse-disable-debug-arguments-verbose");
            if(disable_rocsparse_debug_arguments_verbose)
            {
                rocsparse_disable_debug_arguments_verbose();
            }

            //
            // Any option --bench-?
            //

            //
            // Try to get the option --bench-n.
            //
            int detected_option_bench_n
                = detect_option(argc, argv, "--bench-n", this->m_bench_nruns);

            if(detected_option_bench_n == -1)
            {
                std::cerr << "missing parameter ?" << std::endl;
                exit(1);
            }

            //
            // Try to get the option --bench-o.
            //
            int detected_option_bench_o
                = detect_option_string(argc, argv, "--bench-o", this->m_ofilename);
            if(detected_option_bench_o == -1)
            {
                std::cerr << "missing parameter ?" << std::endl;
                exit(1);
            }

            //
            // Try to get the option --bench-x.
            //
            const char* option_x        = nullptr;
            int detected_option_bench_x = detect_option_string(argc, argv, "--bench-x", option_x);
            if(detected_option_bench_x == -1 || false == is_option(option_x))
            {
                std::cerr << "wrong position of option --bench-x  ?" << std::endl;
                exit(1);
            }

            this->m_name = argv[0];
            this->m_has_bench_option
                = (detected_option_bench_x || detected_option_bench_o || detected_option_bench_n);

            this->m_no_rawdata = detect_flag(argc, argv, "--bench-no-rawdata");

            this->m_is_stdout_disabled = (false == detect_flag(argc, argv, "--bench-std"));

            int jarg = -1;
            for(int iarg = 1; iarg < argc; ++iarg)
            {
                if(argv[iarg] == option_x)
                {
                    jarg = iarg;
                    break;
                }
            }

            int iarg = 1;
            while(iarg < argc)
            {
                //
                // Any argument starting with the character '-' is considered as an option.
                //
                if(is_option(argv[iarg]))
                {
                    if(!strcmp(argv[iarg], "--bench-std"))
                    {
                        ++iarg;
                    }
                    else if(!strcmp(argv[iarg], "--bench-o"))
                    {
                        iarg += 2;
                    }
                    else if(!strcmp(argv[iarg], "--bench-x"))
                    {
                        ++iarg;
                    }
                    else if(!strcmp(argv[iarg], "--bench-n"))
                    {
                        iarg += 2;
                    }
                    else
                    {
                        //
                        // Create the option.
                        //
                        cmdline_option option(argv[iarg]);

                        //
                        // Calculate the number of arguments based on the position of the next option, if any.
                        //
                        const int option_nargs      = count_option_nargs(iarg, argc, argv);
                        const int next_option_index = iarg + 1 + option_nargs;
                        for(int k = iarg + 1; k < next_option_index; ++k)
                        {
                            option.args.push_back(cmdline_arg(argv[k]));
                        }

                        //
                        // If this option has been flagged being the 'X' field.
                        // otherwise, other ('Y') options will be classified from the order of their appearances as Y1, Y2, Y3.
                        //

                        if(jarg == iarg) //
                        {
                            this->m_option_index_x = this->m_options.size();
                        }

                        //
                        // Insert the option created.
                        //
                        this->m_options.push_back(option);
                        iarg = next_option_index;
                    }
                }
                else
                {
                    //
                    // Regular argument.
                    //
                    this->m_args.push_back(cmdline_arg(argv[iarg]));
                    ++iarg;
                }
            }

            this->m_nsamples = 1;
            for(size_t ioption = 0; ioption < this->m_options.size(); ++ioption)
            {
                size_t n = this->m_options[ioption].args.size();
                this->m_nsamples *= std::max(n, static_cast<size_t>(1));
            }
        }

        void expand(val* p)
        {
            const auto num_options = this->m_options.size();
            const auto num_samples = this->m_nsamples;
            for(int i = 0; i < num_samples; ++i)
            {
                p[i](1 + this->m_args.size() + num_options * 2);
                p[i].argc = 0;
            }

            //
            // Program name.
            //
            for(int i = 0; i < num_samples; ++i)
            {
                p[i].argv[p[i].argc++] = this->m_name;
            }

            //
            // Arguments without options
            //
            for(auto& arg : this->m_args)
            {
                for(int i = 0; i < num_samples; ++i)
                    p[i].argv[p[i].argc++] = arg.name;
            }

            const int option_x_nargs = this->m_options[this->m_option_index_x].args.size();
            int       N              = option_x_nargs;
            for(int iopt = 0; iopt < num_options; ++iopt)
            {
                cmdline_option& option = this->m_options[iopt];

                //
                //
                //
                for(int isample = 0; isample < num_samples; ++isample)
                {
                    p[isample].argv[p[isample].argc++] = option.name;
                }

                if(iopt == this->m_option_index_x)
                {
                    //
                    //
                    //
                    {
                        const int ngroups = num_samples / option_x_nargs;
                        for(int jgroup = 0; jgroup < ngroups; ++jgroup)
                        {
                            for(int ix = 0; ix < option_x_nargs; ++ix)
                            {
                                const int flat_index = jgroup * option_x_nargs + ix;
                                p[flat_index].argv[p[flat_index].argc++] = option.args[ix].name;
                            }
                        }
                    }

                    //
                    //
                    //
                    for(int isample = 0; isample < num_samples; ++isample)
                    {
                        if(p[isample].argc != p[0].argc)
                        {
                            std::cerr << "invalid struct line " << __LINE__ << std::endl;
                        }
                    }
                }
                else
                {
                    const int option_narg = option.args.size();
                    if(option_narg > 1)
                    {
                        const int ngroups = num_samples / (N * option_narg);
                        for(int jgroup = 0; jgroup < ngroups; ++jgroup)
                        {
                            for(int option_iarg = 0; option_iarg < option_narg; ++option_iarg)
                            {
                                for(int i = 0; i < N; ++i)
                                {
                                    const int flat_index
                                        = N * (jgroup * option_narg + option_iarg) + i;
                                    p[flat_index].argv[p[flat_index].argc++]
                                        = option.args[option_iarg].name;
                                }
                            }
                        }
                        N *= std::max(option_narg, 1);
                    }
                    else
                    {
                        if(option_narg == 1)
                        {
                            for(int isample = 0; isample < num_samples; ++isample)
                            {
                                p[isample].argv[p[isample].argc++] = option.args[0].name;
                            }
                        }
                    }
                }
            }
        }

    private:
        static inline int count_option_nargs(int iarg, int argc, char** argv)
        {
            int c = 0;
            for(int j = iarg + 1; j < argc; ++j)
            {
                if(is_option(argv[j]))
                {
                    return c;
                }
                ++c;
            }
            return c;
        }

        static bool detect_flag(int argc, char** argv, const char* option_name)
        {
            for(int iarg = 1; iarg < argc; ++iarg)
            {
                if(!strcmp(argv[iarg], option_name))
                {
                    return true;
                }
            }
            return false;
        }
        template <typename T>
        static int detect_option(int argc, char** argv, const char* option_name, T& value)
        {
            for(int iarg = 1; iarg < argc; ++iarg)
            {
                if(!strcmp(argv[iarg], option_name))
                {
                    ++iarg;
                    if(iarg < argc)
                    {
                        std::istringstream iss(argv[iarg]);
                        iss >> value;
                        return 1;
                    }
                    else
                    {
                        std::cerr << "missing value for option --bench-n " << std::endl;
                        return -1;
                    }
                }
            }
            return 0;
        }

        static int
            detect_option_string(int argc, char** argv, const char* option_name, const char*& value)
        {
            for(int iarg = 1; iarg < argc; ++iarg)
            {
                if(!strcmp(argv[iarg], option_name))
                {
                    ++iarg;
                    if(iarg < argc)
                    {
                        value = argv[iarg];
                        return 1;
                    }
                    else
                    {
                        std::cerr << "missing value for option " << option_name << std::endl;
                        return -1;
                    }
                }
            }
            return 0;
        }

        //
        // argument name.
        //
        struct cmdline_arg
        {
            char* name{};
            explicit cmdline_arg(char* name_)
                : name(name_){};
        };

        //
        // argument option.
        //
        struct cmdline_option
        {
            char*                    name{};
            std::vector<cmdline_arg> args{};
            explicit cmdline_option(char* name_)
                : name(name_){};
        };

        static inline bool is_option(const char* arg)
        {
            return arg[0] == '-';
        }

        //
        // Name.
        //
        char* m_name;

        //
        // set of options.
        //
        std::vector<cmdline_option> m_options;

        //
        // set of arguments.
        //
        std::vector<cmdline_arg> m_args;
        bool                     m_has_bench_option{};
        int                      m_bench_nruns{1};
        int                      m_option_index_x;
        int                      m_nsamples;
        bool                     m_is_stdout_disabled{true};
        bool                     m_no_rawdata{};
        const char*              m_ofilename{};
    };

private:
    cmdline m_cmd;
    val*    m_cmdset{};

public:
    static void help(std::ostream& out)
    {
        out << "" << std::endl;
        out << "Specific environment variables:" << std::endl;
        for(const auto v : rocsparse_clients_envariables::s_var_bool_all)
        {
            out << rocsparse_clients_envariables::get_name(v) << " "
                << rocsparse_clients_envariables::get_description(v) << std::endl;
        }
        for(const auto v : rocsparse_clients_envariables::s_var_string_all)
        {
            out << rocsparse_clients_envariables::get_name(v) << " "
                << rocsparse_clients_envariables::get_description(v) << std::endl;
        }
        out << "" << std::endl;
        out << "Rocsparse clients debug options:" << std::endl;
        out << "--rocsparse-clients-enable-test-debug-arguments   enable rocsparse clients test "
               "debug arguments, it discards any environment variable definition of "
               "ROCSPARSE_CLIENTS_TEST_DEBUG_ARGUMENTS."
            << std::endl;
        out << "--rocsparse-clients-disable-test-debug-arguments  disable rocsparse clients test "
               "debug arguments, it discards any environment variable definition of "
               "ROCSPARSE_CLIENTS_TEST_DEBUG_ARGUMENTS."
            << std::endl;
        out << "" << std::endl;
        out << "Rocsparse debug options:" << std::endl;
        out << "--rocsparse-enable-debug                     enable rocsparse debug, it discards "
               "any environment variable definition of ROCSPARSE_DEBUG."
            << std::endl;
        out << "--rocsparse-disable-debug                    disable rocsparse debug, it discards "
               "any environment variable definition of ROCSPARSE_DEBUG."
            << std::endl;
        out << "--rocsparse-enable-debug-verbose             enable rocsparse debug verbose, it "
               "discards any environment variable definition of ROCSPARSE_DEBUG_VERBOSE."
            << std::endl;
        out << "--rocsparse-disable-debug-verbose            disable rocsparse debug verbose, it "
               "discards any environment variable definition of ROCSPARSE_DEBUG_VERBOSE"
            << std::endl;
        out << "--rocsparse-enable-debug-arguments           enable rocsparse debug arguments, it "
               "discards any environment variable definition of ROCSPARSE_DEBUG_ARGUMENTS."
            << std::endl;
        out << "--rocsparse-disable-debug-arguments          disable rocsparse debug arguments, it "
               "discards any environment variable definition of ROCSPARSE_DEBUG_ARGUMENTS."
            << std::endl;
        out << "--rocsparse-enable-debug-arguments-verbose   enable rocsparse debug arguments "
               "verbose, it discards any environment variable definition of "
               "ROCSPARSE_DEBUG_ARGUMENTS_VERBOSE"
            << std::endl;
        out << "--rocsparse-disable-debug-arguments-verbose  disable rocsparse debug arguments "
               "verbose, it discards any environment variable definition of "
               "ROCSPARSE_DEBUG_ARGUMENTS_VERBOSE"
            << std::endl;
        out << "" << std::endl;
        out << "Benchmarks options:" << std::endl;
        out << "--bench-x                                         flag to preceed the main option "
            << std::endl;
        out << "--bench-o                                         output JSON file, (default = "
               "a.json)"
            << std::endl;
        out << "--bench-n                                         number of runs, (default = 1)"
            << std::endl;
        out << "--bench-no-rawdata                                do not export raw data."
            << std::endl;
        out << "" << std::endl;
        out << "Example:" << std::endl;
        out << "rocsparse-bench -f csrmv --bench-x -M 10 20 30 40" << std::endl;
    }

    //
    // @brief Get the output filename.
    //
    const char* get_ofilename() const;

    //
    // @brief Get the number of samples..
    //
    int         get_nsamples() const;
    int         get_option_index_x() const;
    int         get_option_nargs(int i);
    const char* get_option_arg(int i, int j);
    const char* get_option_name(int i);
    int         get_noptions_x() const;
    int         get_noptions() const;
    bool        is_stdout_disabled() const;
    bool        no_rawdata() const;

    //
    // @brief Get the number of runs per sample.
    //
    int  get_nruns() const;
    void get(int isample, int& argc, char** argv) const;

    void                      get_argc(int isample, int& argc_) const;
    rocsparse_bench_cmdlines& operator=(const rocsparse_bench_cmdlines&) = delete;
    //
    // @brief Constructor.
    //
    rocsparse_bench_cmdlines(int argc, char** argv);
    rocsparse_bench_cmdlines(const rocsparse_bench_cmdlines&) = delete;

    virtual ~rocsparse_bench_cmdlines();
    static bool applies(int argc, char** argv);

    //
    // @brief Some info.
    //
    void info() const;
};
