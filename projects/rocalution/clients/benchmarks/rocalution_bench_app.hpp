/*! \file */
/* ************************************************************************
* Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocalution_arguments_config.hpp"
#include "rocalution_bench_cmdlines.hpp"

//
// only for status
//
#include "rocalution_bench_solver_results.hpp"

struct rocalution_benchfile_format
{
    typedef enum value_type_ : int
    {
        json = 0,
        yaml
    } value_type;

protected:
    value_type value{json};

public:
    inline constexpr operator value_type() const
    {
        return this->value;
    };
    inline constexpr rocalution_benchfile_format(){};
    inline constexpr rocalution_benchfile_format(int ival)
        : value((value_type)ival)
    {
    }

    static constexpr value_type all[2]
        = {rocalution_benchfile_format::json, rocalution_benchfile_format::yaml};

    inline bool is_invalid() const
    {
        switch(this->value)
        {
        case json:
        case yaml:
        {
            return false;
        }
        }
        return true;
    };

    inline rocalution_benchfile_format(const char* ext)
    {
        if(!strcmp(ext, ".json"))
        {
            value = json;
        }
        else if(!strcmp(ext, ".JSON"))
        {
            value = json;
        }
        else if(!strcmp(ext, ".yaml"))
        {
            value = yaml;
        }
        else if(!strcmp(ext, ".YAML"))
        {
            value = yaml;
        }
        else
            value = (value_type)-1;
    };

    inline const char* to_string() const
    {
        switch(this->value)
        {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
            CASE(json);
            CASE(yaml);
#undef CASE
        }
        return "unknown";
    }
};

//
// Struct collecting benchmark timing results.
//
struct rocalution_bench_timing_t
{
    //
    // Local item.
    //
    struct item_t
    {
        int                                          m_nruns{};
        std::vector<rocalution_bench_solver_results> m_results;
        rocalution_bench_solver_parameters           m_parameters{};

        item_t(){};

        item_t(int nruns_)
            : m_nruns(nruns_)
            , m_results(nruns_){};

        item_t& operator()(int nruns_)
        {
            this->m_nruns = nruns_;
            this->m_results.resize(nruns_);
            return *this;
        };
        bool record(const rocalution_bench_solver_parameters& parameters)
        {
            m_parameters = parameters;
            return true;
        }
        bool record(int irun, const rocalution_bench_solver_results& results)
        {

            if(irun >= 0 && irun < m_nruns)
            {
                this->m_results[irun] = results;
                return true;
            }
            else
            {
                rocalution_bench_errmsg << "out of bounds from item_t::record " << std::endl;
                return false;
            }
        }
    };

    size_t size() const
    {
        return this->m_items.size();
    };
    item_t& operator[](size_t i)
    {
        return this->m_items[i];
    }
    const item_t& operator[](size_t i) const
    {
        return this->m_items[i];
    }

    rocalution_bench_timing_t(int nsamples, int nruns_per_sample)
        : m_items(nsamples)
    {
        for(int i = 0; i < nsamples; ++i)
        {
            m_items[i](nruns_per_sample);
        }
    }

private:
    std::vector<item_t> m_items;
};

class rocalution_bench_app_base
{
protected:
    //
    // Record initial command line.
    //
    int    m_initial_argc;
    char** m_initial_argv;
    //
    // Set of command lines.
    //
    rocalution_bench_cmdlines m_bench_cmdlines;
    //
    //
    //
    rocalution_bench_timing_t m_bench_timing;

    bool m_stdout_skip_legend{};
    bool m_stdout_disabled{true};

    static int save_initial_cmdline(int argc, char** argv, char*** argv_)
    {
        argv_[0] = new char*[argc];
        for(int i = 0; i < argc; ++i)
        {
            argv_[0][i] = argv[i];
        }
        return argc;
    }
    //
    // @brief Constructor.
    //
    rocalution_bench_app_base(int argc, char** argv);

    //
    // @brief Run case.
    //
    bool run_case(int isample, int irun, int argc, char** argv);

    //
    // For internal use, to get the current isample and irun.
    //
    int m_isample;
    int m_irun;
    int get_isample() const
    {
        return this->m_isample;
    };
    int get_irun() const
    {
        return this->m_irun;
    };

public:
    bool is_stdout_disabled() const
    {
        return m_bench_cmdlines.is_stdout_disabled();
    }

    bool stdout_skip_legend() const
    {
        return this->m_stdout_skip_legend;
    }

    //
    // @brief Run cases.
    //
    bool run_cases();
};

class rocalution_bench_app : public rocalution_bench_app_base
{
private:
    static rocalution_bench_app* s_instance;

public:
    static rocalution_bench_app* instance(int argc, char** argv)
    {
        s_instance = new rocalution_bench_app(argc, argv);
        return s_instance;
    }

    static rocalution_bench_app* instance()
    {
        return s_instance;
    }

    rocalution_bench_app(const rocalution_bench_app&) = delete;
    rocalution_bench_app& operator=(const rocalution_bench_app&) = delete;

    static bool applies(int argc, char** argv)
    {
        return rocalution_bench_cmdlines::applies(argc, argv);
    }

    rocalution_bench_app(int argc, char** argv);
    ~rocalution_bench_app();
    bool export_file();
    bool record_results(const rocalution_bench_solver_parameters& parameters,
                        const rocalution_bench_solver_results&    results)
    {
        if(this->m_irun == 0)
        {
            this->m_bench_timing[this->m_isample].record(parameters);
        }
        return this->m_bench_timing[this->m_isample].record(this->m_irun, results);
    }

protected:
    void        export_item(std::ostream& out, rocalution_bench_timing_t::item_t& item);
    bool        define_case_json(std::ostream& out, int isample, int argc, char** argv);
    bool        close_case_json(std::ostream& out, int isample, int argc, char** argv);
    bool        define_results_json(std::ostream& out);
    bool        close_results_json(std::ostream& out);
    static void confidence_interval(const double               alpha,
                                    const int                  resize,
                                    const int                  nboots,
                                    const std::vector<double>& v,
                                    double                     interval[2]);
    static void series_lower_upper(int                                                 N,
                                   const std::vector<rocalution_bench_solver_results>& r,
                                   rocalution_bench_solver_results::e_double           e,
                                   double&                                             median,
                                   double&                                             lower,
                                   double&                                             upper);
};
