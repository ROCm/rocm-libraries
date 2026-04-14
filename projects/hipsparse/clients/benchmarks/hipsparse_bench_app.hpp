/*! \file */
/* ************************************************************************
* Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse.h"
#include "hipsparse_bench_cmdlines.hpp"
#include <iostream>
#include <vector>

//
// Struct collecting benchmark timing results.
//
struct hipsparse_bench_timing_t
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

        hipsparseStatus_t record(int irun, double msec_, double gflops_, double gbs_)
        {
            if(irun >= 0 && irun < m_nruns)
            {
                this->msec[irun]   = msec_;
                this->gflops[irun] = gflops_;
                this->gbs[irun]    = gbs_;
                return HIPSPARSE_STATUS_SUCCESS;
            }
            else
            {
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }

        hipsparseStatus_t record(int irun, const std::string& s)
        {
            if(irun >= 0 && irun < m_nruns)
            {
                this->outputs[irun] = s;
                return HIPSPARSE_STATUS_SUCCESS;
            }
            else
            {
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        hipsparseStatus_t record_output_legend(const std::string& s)
        {
            this->outputs_legend = s;
            return HIPSPARSE_STATUS_SUCCESS;
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

    hipsparse_bench_timing_t(int nsamples, int nruns_per_sample)
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

class hipsparse_bench_app_base
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
    hipsparse_bench_cmdlines m_bench_cmdlines;
    //
    //
    //
    hipsparse_bench_timing_t m_bench_timing;

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
    hipsparse_bench_app_base(int argc, char** argv);

    //
    // @brief Run case.
    //
    hipsparseStatus_t run_case(int isample, int irun, int argc, char** argv);

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
    hipsparseStatus_t run_cases();
};

class hipsparse_bench_app : public hipsparse_bench_app_base
{
private:
    static hipsparse_bench_app* s_instance;

public:
    static hipsparse_bench_app* instance(int argc, char** argv)
    {
        s_instance = new hipsparse_bench_app(argc, argv);
        return s_instance;
    }

    static hipsparse_bench_app* instance()
    {
        return s_instance;
    }

    hipsparse_bench_app(const hipsparse_bench_app&) = delete;
    hipsparse_bench_app& operator=(const hipsparse_bench_app&) = delete;

    static bool applies(int argc, char** argv)
    {
        return hipsparse_bench_cmdlines::applies(argc, argv);
    }

    hipsparse_bench_app(int argc, char** argv);
    ~hipsparse_bench_app();
    hipsparseStatus_t export_file();
    hipsparseStatus_t record_timing(double msec, double gflops, double bandwidth)
    {
        return this->m_bench_timing[this->m_isample].record(this->m_irun, msec, gflops, bandwidth);
    }
    hipsparseStatus_t record_output(const std::string& s)
    {
        return this->m_bench_timing[this->m_isample].record(this->m_irun, s);
    }
    hipsparseStatus_t record_output_legend(const std::string& s)
    {
        return this->m_bench_timing[this->m_isample].record_output_legend(s);
    }

protected:
    void              export_item(std::ostream& out, hipsparse_bench_timing_t::item_t& item);
    hipsparseStatus_t define_case_json(std::ostream& out, int isample, int argc, char** argv);
    hipsparseStatus_t close_case_json(std::ostream& out, int isample, int argc, char** argv);
    hipsparseStatus_t define_results_json(std::ostream& out);
    hipsparseStatus_t close_results_json(std::ostream& out);
    void              confidence_interval(const double               alpha,
                                          const int                  resize,
                                          const int                  nboots,
                                          const std::vector<double>& v,
                                          double                     interval[2]);
};
