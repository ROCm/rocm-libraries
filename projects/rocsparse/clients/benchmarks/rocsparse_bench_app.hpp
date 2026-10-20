/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-types.h"
#include "rocsparse_bench_cmdlines.hpp"
#include <iostream>
#include <vector>

struct rocsparse_benchfile_format
{
    typedef enum value_type_ : rocsparse_int
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
    inline constexpr rocsparse_benchfile_format(){};
    inline constexpr explicit rocsparse_benchfile_format(rocsparse_int ival)
        : value((value_type)ival)
    {
    }

    static constexpr value_type all[2]
        = {rocsparse_benchfile_format::json, rocsparse_benchfile_format::yaml};

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

    inline explicit rocsparse_benchfile_format(const char* ext)
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
struct rocsparse_bench_timing_t
{
    //
    // Local item
    //
    struct item_t
    {
        int                      m_nruns{};
        std::vector<double>      msec{};
        std::vector<double>      gflops{};
        std::vector<double>      gbs{};
        std::vector<std::string> outputs{};
        std::string              outputs_legend{};
        item_t(){};

        explicit item_t(int nruns_)
            : m_nruns(nruns_)
            , msec(nruns_)
            , gflops(nruns_)
            , gbs(nruns_)
            , outputs(nruns_){};

        item_t& operator()(int nruns_)
        {
            this->m_nruns = nruns_;
            this->msec.resize(nruns_);
            this->gflops.resize(nruns_);
            this->gbs.resize(nruns_);
            this->outputs.resize(nruns_);
            return *this;
        };

        rocsparse_status record(int irun, double msec_, double gflops_, double gbs_)
        {
            if(irun >= 0 && irun < m_nruns)
            {
                this->msec[irun]   = msec_;
                this->gflops[irun] = gflops_;
                this->gbs[irun]    = gbs_;
                return rocsparse_status_success;
            }
            else
            {
                return rocsparse_status_internal_error;
            }
        }

        rocsparse_status record(int irun, const std::string& s)
        {
            if(irun >= 0 && irun < m_nruns)
            {
                this->outputs[irun] = s;
                return rocsparse_status_success;
            }
            else
            {
                return rocsparse_status_internal_error;
            }
        }
        rocsparse_status record_output_legend(const std::string& s)
        {
            this->outputs_legend = s;
            return rocsparse_status_success;
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

    rocsparse_bench_timing_t(int nsamples, int nruns_per_sample)
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

class rocsparse_bench_app_base
{
protected:
    //
    // Record initial command line.
    //
    int    m_initial_argc{};
    char** m_initial_argv;
    //
    // Set of command lines.
    //
    rocsparse_bench_cmdlines m_bench_cmdlines;
    //
    //
    //
    rocsparse_bench_timing_t m_bench_timing;

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
    rocsparse_bench_app_base(int argc, char** argv);

    //
    // @brief Run case.
    //
    rocsparse_status run_case(int isample, int irun, int argc, char** argv);

    //
    // For internal use, to get the current isample and irun.
    //
    int m_isample{};
    int m_irun{};
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
    bool no_rawdata() const
    {
        return m_bench_cmdlines.no_rawdata();
    }

    //
    // @brief Run cases.
    //
    rocsparse_status run_cases();
};

class rocsparse_bench_app : public rocsparse_bench_app_base
{
private:
    static rocsparse_bench_app* s_instance;

public:
    static rocsparse_bench_app* instance(int argc, char** argv)
    {
        s_instance = new rocsparse_bench_app(argc, argv);
        return s_instance;
    }

    static rocsparse_bench_app* instance()
    {
        return s_instance;
    }

    rocsparse_bench_app(const rocsparse_bench_app&) = delete;
    rocsparse_bench_app& operator=(const rocsparse_bench_app&) = delete;

    static bool applies(int argc, char** argv)
    {
        return rocsparse_bench_cmdlines::applies(argc, argv);
    }

    rocsparse_bench_app(int argc, char** argv);
    ~rocsparse_bench_app();
    rocsparse_status export_file();
    rocsparse_status record_timing(double msec, double gflops, double bandwidth)
    {
        return this->m_bench_timing[this->m_isample].record(this->m_irun, msec, gflops, bandwidth);
    }
    rocsparse_status record_output(const std::string& s)
    {
        return this->m_bench_timing[this->m_isample].record(this->m_irun, s);
    }
    rocsparse_status record_output_legend(const std::string& s)
    {
        return this->m_bench_timing[this->m_isample].record_output_legend(s);
    }

protected:
    void             export_item(std::ostream& out, rocsparse_bench_timing_t::item_t& item);
    rocsparse_status define_case_json(std::ostream& out, int isample, int argc, char** argv);
    rocsparse_status close_case_json(std::ostream& out, int isample, int argc, char** argv);
    rocsparse_status define_results_json(std::ostream& out);
    rocsparse_status close_results_json(std::ostream& out);
    void             confidence_interval(const double               alpha,
                                         const int                  resize,
                                         const int                  nboots,
                                         const std::vector<double>& v,
                                         double                     interval[2]);
};
