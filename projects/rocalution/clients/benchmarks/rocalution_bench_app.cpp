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

#include "rocalution_bench_app.hpp"
#include "rocalution_bench.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
rocalution_bench_app* rocalution_bench_app::s_instance = nullptr;

rocalution_bench_app_base::rocalution_bench_app_base(int argc, char** argv)
    : m_initial_argc(rocalution_bench_app_base::save_initial_cmdline(argc, argv, &m_initial_argv))
    , m_bench_cmdlines(argc, argv)
    , m_bench_timing(m_bench_cmdlines.get_nsamples(), m_bench_cmdlines.get_nruns())

          {};

bool rocalution_bench_app_base::run_case(int isample, int irun, int argc, char** argv)
{
    rocalution_bench bench(argc, argv);
    return bench.run();
}

bool rocalution_bench_app_base::run_cases()
{
    int    sample_argc;
    char** sample_argv = nullptr;
    //
    // Loop over cases.
    //
    int nruns                  = this->m_bench_cmdlines.get_nruns();
    int nsamples               = this->m_bench_cmdlines.get_nsamples();
    this->m_stdout_skip_legend = false;

    if(is_stdout_disabled())
    {
        printf("// start benchmarking ... (nsamples = %d, nruns = %d)\n", nsamples, nruns);
    }
    bool status = true;
    for(int isample = 0; isample < nsamples; ++isample)
    {
        this->m_isample = isample;
        //
        // Add an item to collect data through rocalution_record_timing
        //
        for(int irun = 0; irun < nruns; ++irun)
        {
            this->m_irun = irun;

            if(false == this->m_stdout_skip_legend)
            {
                this->m_stdout_skip_legend = (irun > 0 && isample == 0);
            }

            //
            // Get command line arguments, copy each time since it is mutable afterwards.
            //
            if(sample_argv == nullptr)
            {
                this->m_bench_cmdlines.get_argc(this->m_isample, sample_argc);
                sample_argv = new char*[sample_argc];
            }

            this->m_bench_cmdlines.get(this->m_isample, sample_argc, sample_argv);

            //
            // Run the case.
            //
            status &= this->run_case(this->m_isample, this->m_irun, sample_argc, sample_argv);
            if(!status)
            {
                rocalution_bench_errmsg << "run_cases::run_case failed at line " << __LINE__
                                        << std::endl;
                break;
            }
            if(is_stdout_disabled())
            {
                if((isample * nruns + irun) % 10 == 0)
                {
                    fprintf(stdout,
                            "\r// %2.0f%%",
                            (double(isample * nruns + irun + 1) / double(nsamples * nruns)) * 100);
                    fflush(stdout);
                }
            }
        }
    }
    if(is_stdout_disabled())
    {
        printf("\r// benchmarking done.\n");
    }

    if(sample_argv != nullptr)
    {
        delete[] sample_argv;
    }
    return status;
};

rocalution_bench_app::rocalution_bench_app(int argc, char** argv)
    : rocalution_bench_app_base(argc, argv)
{
}

rocalution_bench_app::~rocalution_bench_app() {}

void rocalution_bench_app::confidence_interval(const double               alpha,
                                               const int                  resize,
                                               const int                  nboots,
                                               const std::vector<double>& v,
                                               double                     interval[2])
{
    const size_t        size = v.size();
    std::vector<double> medians(nboots);
    std::vector<double> resample(resize);
#define median_value(n__, s__) \
    ((n__ % 2 == 0) ? (s__[n__ / 2 - 1] + s__[n__ / 2]) * 0.5 : s__[n__ / 2])
    std::srand(0);
    for(int iboot = 0; iboot < nboots; ++iboot)
    {
        for(int i = 0; i < resize; ++i)
        {
            const int j = (std::rand() % size);
            resample[i] = v[j];
        }
        std::sort(resample.begin(), resample.end());
        medians[iboot] = median_value(resize, resample);
    }

    std::sort(medians.begin(), medians.end());
    interval[0] = medians[int(floor(nboots * 0.5 * (1.0 - alpha)))];
    interval[1] = medians[int(ceil(nboots * (1.0 - 0.5 * (1.0 - alpha))))];
#undef median_value
}

void rocalution_bench_app::series_lower_upper(int                                                 N,
                                              const std::vector<rocalution_bench_solver_results>& r,
                                              rocalution_bench_solver_results::e_double           e,
                                              double& median,
                                              double& lower,
                                              double& upper)
{
    static constexpr double alpha = 0.95;
    std::vector<double>     s(N);
    for(int i = 0; i < N; ++i)
    {
        s[i] = r[i].Get(e);
    }
    std::sort(s.begin(), s.end());
    double interval[2];

    static constexpr int nboots = 200;
    confidence_interval(alpha, 10, nboots, s, interval);

    median = (N % 2 == 0) ? (s[N / 2 - 1] + s[N / 2]) * 0.5 : s[N / 2];
    lower  = interval[0];
    upper  = interval[1];
}

void rocalution_bench_app::export_item(std::ostream& out, rocalution_bench_timing_t::item_t& item)
{
    out << " \"setup\"  : { ";
    item.m_parameters.WriteJson(out);
    out << " }," << std::endl;

    //
    //
    //
    auto N = item.m_nruns;
    if(N > 1)
    {
        rocalution_bench_solver_results res;
        rocalution_bench_solver_results res_low;
        rocalution_bench_solver_results res_up;

        for(auto e : rocalution_bench_solver_results::e_bool_all)
        {
            for(int i = 0; i < N; ++i)
            {
                if(item.m_results[i].Get(e) != item.m_results[0].Get(e))
                {
                    std::cerr << "WARNING/ERROR Boolean result '"
                              << rocalution_bench_solver_results::Name(e)
                              << "' is not constant over runs." << std::endl;
                }
            }
            res.Set(e, item.m_results[0].Get(e));
            res_low.Set(e, item.m_results[0].Get(e));
            res_up.Set(e, item.m_results[0].Get(e));
        }

        for(auto e : rocalution_bench_solver_results::e_int_all)
        {
            for(int i = 0; i < N; ++i)
            {
                if(item.m_results[i].Get(e) != item.m_results[0].Get(e))
                {
                    std::cerr << "WARNING/ERROR Integer result '"
                              << rocalution_bench_solver_results::Name(e)
                              << "' is not constant over runs." << std::endl;
                }
            }
            res.Set(e, item.m_results[0].Get(e));
            res_low.Set(e, item.m_results[0].Get(e));
            res_up.Set(e, item.m_results[0].Get(e));
        }

        for(auto e : rocalution_bench_solver_results::e_double_all)
        {

            double median;
            double lower;
            double upper;
            series_lower_upper(N, item.m_results, e, median, lower, upper);

            res.Set(e, median);
            res_low.Set(e, lower);
            res_up.Set(e, upper);
        }

        out << " \"nsamples\": \"" << N << "\"," << std::endl;
        out << " \"median\"  : { ";
        res.WriteJson(out);
        out << " }," << std::endl;
        out << " \"low\"     : { ";
        res_low.WriteJson(out);
        out << " }," << std::endl;
        out << " \"up\"      : { ";
        res_up.WriteJson(out);
        out << " }" << std::endl;
    }
    else
    {
        out << " \"nsamples\": \"" << N << "\"," << std::endl;
        out << " \"median\"  : { ";
        item.m_results[0].WriteJson(out);
        out << " }," << std::endl;
        out << " \"low\"  : { ";
        item.m_results[0].WriteJson(out);
        out << " }," << std::endl;
        out << " \"up\"  : { ";
        item.m_results[0].WriteJson(out);
        out << " }" << std::endl;
    }
}

bool rocalution_bench_app::export_file()
{
    const char* ofilename = this->m_bench_cmdlines.get_ofilename();
    if(ofilename == nullptr)
    {
        std::cerr << "//" << std::endl;
        std::cerr << "// rocalution_bench_app warning: no output filename has been specified,"
                  << std::endl;
        std::cerr << "// default output filename is 'a.json'." << std::endl;
        std::cerr << "//" << std::endl;
        ofilename = "a.json";
    }

    std::ofstream out(ofilename);

    int   sample_argc;
    char* sample_argv[64];

    bool status;

    //
    // Write header.
    //
    status = define_results_json(out);
    if(!status)
    {
        rocalution_bench_errmsg << "run_cases failed at line " << __LINE__ << std::endl;
        return status;
    }

    //
    // Loop over cases.
    //
    const int nsamples          = m_bench_cmdlines.get_nsamples();
    const int bench_timing_size = m_bench_timing.size();
    if(nsamples != bench_timing_size)
    {
        rocalution_bench_errmsg << "incompatible sizes at line " << __LINE__ << " " << nsamples
                                << " " << bench_timing_size << std::endl;
        if(bench_timing_size == 0)
        {
            rocalution_bench_errmsg << "No data has been harvested from running case" << std::endl;
        }
        exit(1);
    }

    for(int isample = 0; isample < nsamples; ++isample)
    {
        this->m_bench_cmdlines.get(isample, sample_argc, sample_argv);

        this->define_case_json(out, isample, sample_argc, sample_argv);
        out << "{ ";
        {
            this->export_item(out, this->m_bench_timing[isample]);
        }
        out << " }";
        this->close_case_json(out, isample, sample_argc, sample_argv);
    }

    //
    // Write footer.
    //
    status = this->close_results_json(out);
    if(!status)
    {
        rocalution_bench_errmsg << "run_cases failed at line " << __LINE__ << std::endl;
        return status;
    }
    out.close();
    return true;
}

bool rocalution_bench_app::define_case_json(std::ostream& out, int isample, int argc, char** argv)
{
    if(isample > 0)
        out << "," << std::endl;
    out << std::endl;
    out << "{ \"cmdline\": \"";
    out << argv[0];
    for(int i = 1; i < argc; ++i)
        out << " " << argv[i];
    out << " \"," << std::endl;
    out << "  \"timing\": ";
    return true;
}

bool rocalution_bench_app::close_case_json(std::ostream& out, int isample, int argc, char** argv)
{
    out << " }";
    return true;
}

bool rocalution_bench_app::define_results_json(std::ostream& out)
{
    out << "{" << std::endl;
    auto        end      = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    char*       str      = std::ctime(&end_time);
    for(int i = 0; i >= 0; ++i)
        if(str[i] == '\n')
        {
            str[i] = '\0';
            break;
        }
    out << "\"date\": \"" << str << "\"," << std::endl;

    out << std::endl << "\"cmdline\": \"" << this->m_initial_argv[0];

    for(int i = 1; i < this->m_initial_argc; ++i)
    {
        out << " " << this->m_initial_argv[i];
    }
    out << "\"," << std::endl;

    int option_index_x = this->m_bench_cmdlines.get_option_index_x();
    out << std::endl << "\"xargs\": \[";
    for(int j = 0; j < this->m_bench_cmdlines.get_option_nargs(option_index_x); ++j)
    {
        auto arg = this->m_bench_cmdlines.get_option_arg(option_index_x, j);
        if(j > 0)
            out << ", ";
        out << "\"" << arg << "\"";
    }
    out << "]," << std::endl;
    out << std::endl << "\"yargs\":";

    //
    // Harvest expanded options.
    //
    std::vector<int> y_options_size;
    std::vector<int> y_options_index;
    for(int k = 0; k < this->m_bench_cmdlines.get_noptions(); ++k)
    {
        if(k != option_index_x)
        {
            if(this->m_bench_cmdlines.get_option_nargs(k) > 1)
            {
                y_options_index.push_back(k);
                y_options_size.push_back(this->m_bench_cmdlines.get_option_nargs(k));
            }
        }
    }

    const int num_y_options = y_options_index.size();
    if(num_y_options > 0)
    {
        std::vector<std::vector<int>> indices(num_y_options);
        for(int k = 0; k < num_y_options; ++k)
        {
            indices[k].resize(y_options_size[k], 0);
        }
    }

    int nplots = this->m_bench_cmdlines.get_nsamples()
                 / this->m_bench_cmdlines.get_option_nargs(option_index_x);
    std::vector<std::string> plot_titles(nplots);
    if(plot_titles.size() == 1)
    {
        plot_titles.push_back("");
    }
    else
    {
        int  n        = y_options_size[0];
        auto argname0 = this->m_bench_cmdlines.get_option_name(y_options_index[0]);
        for(int iplot = 0; iplot < nplots; ++iplot)
        {
            std::string title("");
            int         p = n;

            {
                int  jref = iplot % p;
                auto arg0 = this->m_bench_cmdlines.get_option_arg(y_options_index[0], jref);
                title += std::string(argname0 + ((argname0[1] == '-') ? 2 : 1)) + std::string("=")
                         + arg0;
            }

            for(int k = 1; k < num_y_options; ++k)
            {
                int kref = iplot / p;
                p *= this->m_bench_cmdlines.get_option_nargs(y_options_index[k]);
                auto arg     = this->m_bench_cmdlines.get_option_arg(y_options_index[k], kref);
                auto argname = this->m_bench_cmdlines.get_option_name(y_options_index[k]);

                title += std::string(",") + std::string(argname + ((argname[1] == '-') ? 2 : 1))
                         + std::string("=") + arg;
            }
            plot_titles[iplot] = title;
        }
    }
    out << "[";
    {
        out << "\"" << plot_titles[0] << "\"";
        for(int iplot = 1; iplot < nplots; ++iplot)
            out << ", \"" << plot_titles[iplot] << "\"";
    }
    out << "]," << std::endl << std::endl;
    ;
    out << "\""
        << "results"
        << "\": [";

    return true;
}

bool rocalution_bench_app::close_results_json(std::ostream& out)
{
    out << "]" << std::endl;
    out << "}" << std::endl;
    return true;
}
