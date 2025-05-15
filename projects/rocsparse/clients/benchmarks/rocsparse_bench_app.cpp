/*! \file */
/* ************************************************************************
* Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_bench_app.hpp"
#include "rocsparse_bench.hpp"
#include "rocsparse_random.hpp"
#include <chrono>
#include <fstream>

rocsparse_bench_app* rocsparse_bench_app::s_instance = nullptr;

rocsparse_bench_app_base::rocsparse_bench_app_base(int argc, char** argv)
    : m_initial_argc(rocsparse_bench_app_base::save_initial_cmdline(argc, argv, &m_initial_argv))
    , m_bench_cmdlines(argc, argv)
    , m_bench_timing(m_bench_cmdlines.get_nsamples(), m_bench_cmdlines.get_nruns())

          {};

rocsparse_status rocsparse_bench_app_base::run_case(int isample, int irun, int argc, char** argv)
{
    rocsparse_bench bench(argc, argv);
    return bench.run();
}

rocsparse_status rocsparse_bench_app_base::run_cases()
{
    int    sample_argc;
    char** sample_argv = nullptr;
    //
    // Loop over cases.
    //
    int nruns    = this->m_bench_cmdlines.get_nruns();
    int nsamples = this->m_bench_cmdlines.get_nsamples();
    if(is_stdout_disabled())
    {
        printf("// start benchmarking ... (nsamples = %d, nruns = %d)\n", nsamples, nruns);
    }

    for(int isample = 0; isample < nsamples; ++isample)
    {
        this->m_isample = isample;
        //
        // Add an item to collect data through rocsparse_record_timing
        //
        for(int irun = 0; irun < nruns; ++irun)
        {
            this->m_irun = irun;

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
            rocsparse_status status
                = this->run_case(this->m_isample, this->m_irun, sample_argc, sample_argv);
            if(status != rocsparse_status_success)
            {
                std::cerr << "run_cases::run_case failed at line " << __LINE__ << std::endl;
                return status;
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
    return rocsparse_status_success;
};

rocsparse_bench_app::rocsparse_bench_app(int argc, char** argv)
    : rocsparse_bench_app_base(argc, argv)
{
}

rocsparse_bench_app::~rocsparse_bench_app() {}

void rocsparse_bench_app::confidence_interval(const double               alpha,
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

    for(int iboot = 0; iboot < nboots; ++iboot)
    {
        for(int i = 0; i < resize; ++i)
        {
            const int j = random_generator_exact<int>(0, size - 1);
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

void rocsparse_bench_app::export_item(std::ostream& out, rocsparse_bench_timing_t::item_t& item)
{
    //
    //
    //
    auto N = item.m_nruns;
    if(N > 1)
    {
        const double alpha = 0.95;
        std::sort(item.msec.begin(), item.msec.end());
        std::sort(item.gflops.begin(), item.gflops.end());
        std::sort(item.gbs.begin(), item.gbs.end());
        double msec
            = (N % 2 == 0) ? (item.msec[N / 2 - 1] + item.msec[N / 2]) * 0.5 : item.msec[N / 2];
        double gflops = (N % 2 == 0) ? (item.gflops[N / 2 - 1] + item.gflops[N / 2]) * 0.5
                                     : item.gflops[N / 2];
        double gbs = (N % 2 == 0) ? (item.gbs[N / 2 - 1] + item.gbs[N / 2]) * 0.5 : item.gbs[N / 2];

        double interval_msec[2], interval_gflops[2], interval_gbs[2];
        int    nboots = 200;
        confidence_interval(alpha, 10, nboots, item.msec, interval_msec);
        confidence_interval(alpha, 10, nboots, item.gflops, interval_gflops);
        confidence_interval(alpha, 10, nboots, item.gbs, interval_gbs);

        out << std::endl
            << "    \"time\": [\"" << msec << "\", \"" << interval_msec[0] << "\", \""
            << interval_msec[1] << "\"]," << std::endl;
        out << "    \"flops\": [\"" << gflops << "\", \"" << interval_gflops[0] << "\", \""
            << interval_gflops[1] << "\"]," << std::endl;
        out << "    \"bandwidth\": [\"" << gbs << "\", \"" << interval_gbs[0] << "\", \""
            << interval_gbs[1] << "\"]";

        if(!no_rawdata())
        {
            out << ",";
            out << std::endl << "    \"raw_legend\": \"" << item.outputs_legend << "\"";
            out << ",";
            out << std::endl << "    \"raw_data\": \"" << item.outputs[0] << "\"";
        }
    }
    else
    {
        out << std::endl
            << "\"time\": [\"" << item.msec[0] << "\", \"" << item.msec[0] << "\", \""
            << item.msec[0] << "\"]," << std::endl;
        out << "\"flops\": [\"" << item.gflops[0] << "\", \"" << item.gflops[0] << "\", \""
            << item.gflops[0] << "\"]," << std::endl;
        out << "\"bandwidth\": [\"" << item.gbs[0] << "\", \"" << item.gbs[0] << "\", \""
            << item.gbs[0] << "\"]";
        if(!no_rawdata())
        {
            out << ",";
            out << std::endl << "    \"raw_legend\": \"" << item.outputs_legend << "\"";
            out << ",";
            out << std::endl << "    \"raw_data\": ";
            out << "\"" << item.outputs[0] << "\"";
            out << "" << std::endl;
        }
    }
}

rocsparse_status rocsparse_bench_app::export_file()
{
    const char* ofilename = this->m_bench_cmdlines.get_ofilename();
    if(ofilename == nullptr)
    {
        std::cerr << "//" << std::endl;
        std::cerr << "// rocsparse_bench_app warning: no output filename has been specified,"
                  << std::endl;
        std::cerr << "// default output filename is 'a.json'." << std::endl;
        std::cerr << "//" << std::endl;
        ofilename = "a.json";
    }

    std::ofstream out(ofilename);

    int   sample_argc;
    char* sample_argv[64];

    rocsparse_status status;

    //
    // Write header.
    //
    status = define_results_json(out);
    if(status != rocsparse_status_success)
    {
        std::cerr << "run_cases failed at line " << __LINE__ << std::endl;
        return status;
    }

    //
    // Loop over cases.
    //
    const int nsamples = m_bench_cmdlines.get_nsamples();
    if(nsamples != m_bench_timing.size())
    {
        std::cerr << "incompatible sizes at line " << __LINE__ << " "
                  << m_bench_cmdlines.get_nsamples() << " " << m_bench_timing.size() << std::endl;
        if(m_bench_timing.size() == 0)
        {
            std::cerr << "No data has been harvested from running case" << std::endl;
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
    if(status != rocsparse_status_success)
    {
        std::cerr << "run_cases failed at line " << __LINE__ << std::endl;
        return status;
    }
    out.close();
    return rocsparse_status_success;
}

rocsparse_status
    rocsparse_bench_app::define_case_json(std::ostream& out, int isample, int argc, char** argv)
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
    return rocsparse_status_success;
}

rocsparse_status
    rocsparse_bench_app::close_case_json(std::ostream& out, int isample, int argc, char** argv)
{
    out << " }";
    return rocsparse_status_success;
}

rocsparse_status rocsparse_bench_app::define_results_json(std::ostream& out)
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
    out << "\"rocSPARSE version\": \"" << rocsparse_get_version() << "\"," << std::endl;

    //
    // !!! To fix, not necessarily the gpu used from rocsparse_bench.
    //
    hipDeviceProp_t prop;
    if(hipGetDeviceProperties(&prop, 0) != hipSuccess)
    {
        std::cerr << "rocsparse-bench error: cannot get device properties" << std::endl;
        return rocsparse_status_internal_error;
    }
    gpu_config g(prop);
    g.print_json(out);

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
                if(argname0[1] == '-')
                {
                    title += std::string(argname0 + 2) + std::string("=") + arg0;
                }
                else
                {
                    title += std::string(argname0 + 1) + std::string("=") + arg0;
                }
            }

            for(int k = 1; k < num_y_options; ++k)
            {
                int kref = iplot / p;
                p *= this->m_bench_cmdlines.get_option_nargs(y_options_index[k]);
                auto arg     = this->m_bench_cmdlines.get_option_arg(y_options_index[k], kref);
                auto argname = this->m_bench_cmdlines.get_option_name(y_options_index[k]);
                if(argname[1] == '-')
                {
                    title += std::string(",") + std::string(argname + 2) + std::string("=") + arg;
                }
                else
                {
                    title += std::string(",") + std::string(argname + 1) + std::string("=") + arg;
                }
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

    return rocsparse_status_success;
}

rocsparse_status rocsparse_bench_app::close_results_json(std::ostream& out)
{
    out << "]" << std::endl;
    out << "}" << std::endl;
    return rocsparse_status_success;
}
