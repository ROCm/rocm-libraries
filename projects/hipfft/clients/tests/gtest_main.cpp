// Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/// @file
/// @brief googletest based unit tester for hipfft
///

#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <streambuf>
#include <string>

#include "../../shared/CLI11.hpp"
#include "../../shared/concurrency.h"
#include "../../shared/device_properties.h"
#include "../../shared/environment.h"
#include "../../shared/hostbuf.h"
#include "../../shared/sys_mem.h"
#include "../../shared/work_queue.h"
#include "../hipfft_params.h"
#include "hipfft/hipfft.h"
#include "hipfft_accuracy_test.h"

// initialize static class member of hipfft_params
std::vector<gpubuf> hipfft_params::externally_managed_workareas = std::vector<gpubuf>();

// Control output verbosity:
int verbose;

// User-defined random seed
size_t             random_seed;
std::random_device default_seed_dev;
// Overall probability of running conventional tests
double test_prob;
// Modifier for probability of running tests with complex interleaved data
double complex_interleaved_prob_factor;
// Modifier for probability of running tests with real data
double real_prob_factor;
// Modifier for probability of running tests with complex planar data
double complex_planar_prob_factor;
// Modifier for probability of running tests with callbacks
double callback_prob_factor;
// Constraints for the hipfftw tests
size_t max_length_for_hipfftw_test;
size_t max_io_gb_for_hipfftw_test;

// Transform parameters for manual test:
hipfft_params manual_params;

// Host memory limitation for tests (GiB):
size_t ramgb;

// Device memory limitation for tests (GiB):
size_t vramgb;

// Allow skipping tests if there is a runtime error
bool skip_runtime_fails;
// But count the number of failures
int n_hip_failures = 0;

// Manually specified precision cutoffs:
double single_epsilon;
double double_epsilon;
double half_epsilon;

// Measured precision cutoffs:
double max_linf_eps_double = 0.0;
double max_l2_eps_double   = 0.0;
double max_linf_eps_single = 0.0;
double max_l2_eps_single   = 0.0;
double max_linf_eps_half   = 0.0;
double max_l2_eps_half     = 0.0;

// Control whether we use FFTW's wisdom (which we use to imply FFTW_MEASURE).
bool use_fftw_wisdom = false;

// Compare results against FFTW in accuracy tests
bool fftw_compare = true;

// Cache the last cpu fft that was requested
last_cpu_fft_cache last_cpu_fft_data;

// Multi-process library to use
fft_params::fft_mp_lib mp_lib = fft_params::fft_mp_lib_none;
// Number of multi-process ranks to launch
int mp_ranks = 1;
// Multi-process launch command (e.g. mpirun --np 4 /path/to/hipfft_mpi_worker)
std::string mp_launch;

void init_gtest_flags()
{
    // HACK: gtest maintains a "should run" flag on each test case,
    // but only sets it during RUN_ALL_TESTS.  Precompiling should
    // ideally only happen for the test cases that would actually
    // run.
    //
    // So call RUN_ALL_TESTS once with the "list tests" temporarily set
    // to true, to initialize all of that.
    //
    // gtest will then print all of the test cases to stdout.
    // Temporarily redirect stdout to /dev/null as well.
    bool temp_list_tests = true;

    std::swap(temp_list_tests, testing::GTEST_FLAG(list_tests));

    // move stdout to devnull
#ifdef WIN32
    int stdout_fd   = _fileno(stdout);
    int devnull     = _open("NUL", _O_WRONLY);
    int stdout_copy = _dup(stdout_fd);
    _dup2(devnull, stdout_fd);
#else
    int stdout_fd   = STDOUT_FILENO;
    int devnull     = open("/dev/null", O_WRONLY);
    int stdout_copy = dup(stdout_fd);
    dup2(devnull, stdout_fd);
#endif

    (void)RUN_ALL_TESTS();

    // put stdout back
#ifdef WIN32
    _dup2(stdout_copy, stdout_fd);
    _close(stdout_copy);
    _close(devnull);
#else
    dup2(stdout_copy, stdout_fd);
    close(stdout_copy);
    close(devnull);
#endif

    std::swap(temp_list_tests, testing::GTEST_FLAG(list_tests));
}

void precompile_test_kernels(const std::string& precompile_file)
{
    std::cout << "precompiling test kernels...\n";
    WorkQueue<std::string> tokenQueue;

    init_gtest_flags();

    std::vector<std::string> tokens;
    auto                     ut = testing::UnitTest::GetInstance();
    for(int ts_index = 0; ts_index < ut->total_test_suite_count(); ++ts_index)
    {
        const auto ts = ut->GetTestSuite(ts_index);
        for(int ti_index = 0; ti_index < ts->total_test_count(); ++ti_index)
        {
            const auto  ti   = ts->GetTestInfo(ti_index);
            std::string name = ti->name();

            // only precompile test cases that will run
            if(!ti->should_run())
                continue;

            // only care about accuracy tests
            const auto pos = name.find("vs_fftw/");
            if(pos != std::string::npos)
            {
                name.erase(0, pos + 8);

                // change batch to 1, so we don't waste time creating
                // multiple plans that differ only by batch
                auto idx = name.find("_batch_");
                if(idx == std::string::npos)
                    continue;
                // advance idx to batch number
                idx += 7;
                auto end = name.find('_', idx);
                if(end == std::string::npos)
                    continue;
                name.replace(idx, end - idx, "1");

                tokens.emplace_back(std::move(name));
            }
        }
    }

    std::random_device dev;
    std::mt19937       dist(dev());
    std::shuffle(tokens.begin(), tokens.end(), dist);
    auto precompile_begin = std::chrono::steady_clock::now();
    std::cout << "precompiling kernels for " << tokens.size() << " tokens...\n";

    for(auto&& t : tokens)
        tokenQueue.push(std::move(t));

    EnvironmentSetTemp       env_compile_only{"ROCFFT_INTERNAL_COMPILE_ONLY", "1"};
    const size_t             NUM_THREADS = rocfft_concurrency();
    std::vector<std::thread> threads;
    for(size_t i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&tokenQueue]() {
            for(;;)
            {
                std::string token{tokenQueue.pop()};
                if(token.empty())
                    break;

                try
                {
                    hipfft_params params;
                    params.from_token(token);
                    params.validate();
                    params.create_plan();
                    if(params.is_forward())
                    {
                        hipfft_params inverse_params;
                        inverse_params.inverse_from_forward(params);
                        inverse_params.validate();
                        inverse_params.create_plan();
                    }
                }
                catch(fft_params::work_buffer_alloc_failure&)
                {
                    continue;
                }
                catch(std::exception& e)
                {
                    // failed to create a plan, abort
                    //
                    // we could continue on, but the test should just
                    // fail later anyway in the same way.  so report
                    // which token failed early and get out
                    throw std::runtime_error(token + " plan creation failure: " + e.what());
                }
            }
        });
        // insert empty tokens to tell threads to stop
        tokenQueue.push({});
    }
    for(auto& t : threads)
        t.join();

    auto                                      precompile_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> precompile_ms  = precompile_end - precompile_begin;
    std::cout << "done precompiling FFT plans in " << static_cast<size_t>(precompile_ms.count())
              << " ms\n";
}

int main(int argc, char* argv[])
{
    CLI::App app{
        "\n"
        "hipFFT Runtime Test command line options\n"
        "NB: input parameters are row-major.\n"
        "\n"
        "FFTW accuracy test cases are named using these identifiers:\n"
        "\n"
        "  len_<n>: problem dimensions, row-major\n"
        "  single,double: precision\n"
        "  ip,op: in-place or out-of-place\n"
        "  batch_<n>: batch size\n"
        "  istride_<n>_<format>: input stride (ostride for output stride), format may be:\n"
        "      CI - complex interleaved\n"
        "      CP - complex planar\n"
        "      R  - real\n"
        "      HI - hermitian interleaved\n"
        "      HP - hermitian planar\n"
        "\n"
        "Usage"};
    // Override CLI11 help to print it along gtest's help
    app.set_help_flag("");
    const auto opt_help = app.add_flag("-h, --help", "Produces this help message");
    app.add_option("-v, --verbose", verbose, "Print out detailed information for the tests")
        ->default_val(0);
    app.add_option("--test_prob", test_prob, "Probability of running individual tests")
        ->default_val(1.0)
        ->check(CLI::Range(0.0, 1.0));
    app.add_option("--real_prob",
                   real_prob_factor,
                   "Probability multiplier for running individual real/complex transforms")
        ->default_val(1.0)
        ->check(CLI::PositiveNumber);
    app.add_option("--planar_prob",
                   complex_planar_prob_factor,
                   "Probability multiplier for running individual planar transforms")
        ->default_val(0.1)
        ->check(CLI::PositiveNumber);
    app.add_option(
           "--complex_interleaved_prob_factor",
           complex_interleaved_prob_factor,
           "Probability multiplier for running individual transforms with complex interleaved data")
        ->default_val(1)
        ->check(CLI::PositiveNumber);
    app.add_option("--callback_prob",
                   callback_prob_factor,
                   "Probability multiplier for running individual callback transforms")
        ->default_val(0.1)
        ->check(CLI::NonNegativeNumber);
    app.add_option("--max_hipfftw_test_len",
                   max_length_for_hipfftw_test,
                   "Maximum length to be considered in hipfftw tests")
        ->default_val(8192)
        ->check(CLI::PositiveNumber);
    app.add_option("--max_io_gb_for_hipfftw_test",
                   max_io_gb_for_hipfftw_test,
                   "Maximum size of I/O to be considered in hipfftw tests in GiB")
        ->default_val(1) /* 1 GiB */
        ->check(CLI::PositiveNumber);

    app.add_option("--fftw_compare", fftw_compare, "Compare to FFTW in accuracy tests")
        ->default_val(true);
    app.add_option("--mp_lib", mp_lib, "Multi-process library type: none (default), mpi")
        ->default_val("none");
    app.add_option("--mp_ranks", mp_ranks, "Number of multi-process ranks to launch")
        ->default_val(1)
        ->check(CLI::NonNegativeNumber);
    app.add_option("--mp_launch",
                   mp_launch,
                   "Command line prefix to launch multi-process transforms, e.g. \"mpirun --np 4 "
                   "/path/to/hipfft_mpi_worker\"")
        ->default_val("")
        ->each([&](const std::string&) {
            if(mp_lib == fft_params::fft_mp_lib_none)
            {
                throw CLI::ValidationError(
                    "--mp_launch requires an mp library (see mp_lib in --help)");
            }
        })
        ->needs("--mp_lib");
    app.add_option("--seed", random_seed, "Random seed; if unset, use an actual random seed")
        ->default_val(default_seed_dev());
    app.add_flag("--smoketest", "Run a short (approx 5 minute) randomized selection of tests")
        ->each([&](const std::string&) {
            // The objective is to have an test that takes about 5 minutes, so just set the
            // probability per test to a small value to achieve this result.
            test_prob = 0.002;
        });
    // Token string to fully specify fft params for the manual test.
    std::string test_token;
    auto*       opt_token
        = app.add_option("--token", test_token, "Test token name for manual test")->default_val("");
    // Group together options that conflict with --token
    auto* non_token = app.add_option_group("Token Conflict", "Options excluded by --token");
    non_token->excludes(opt_token);
    // Declare the supported options. Some option pointers are declared to track passed opts.
    non_token->add_flag("--callback", "Inject load/store callbacks")->each([&](const std::string&) {
        manual_params.run_callbacks = true;
    });
    non_token
        ->add_option("--auto_allocation",
                     manual_params.auto_allocate,
                     "hipFFT's auto-allocation behavior: \"on\", \"off\", or \"default\"")
        ->default_val("default");
    non_token
        ->add_flag("--double", "Double precision transform (deprecated: use --precision double)")
        ->each([&](const std::string&) { manual_params.precision = fft_precision_double; });
    non_token
        ->add_option("-t, --transformType",
                     manual_params.transform_type,
                     "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
                     "forward\n3) real inverse")
        ->default_val(fft_transform_type_complex_forward);
    non_token
        ->add_option("--precision",
                     manual_params.precision,
                     "Transform precision: single (default), double, half")
        ->excludes("--double");
    non_token->add_flag("-o, --notInPlace", "Not in-place FFT transform (default: in-place)")
        ->each([&](const std::string&) { manual_params.placement = fft_placement_notinplace; });
    non_token
        ->add_option("--itype",
                     manual_params.itype,
                     "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
                     "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset);
    non_token
        ->add_option("--otype",
                     manual_params.otype,
                     "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
                     "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset);
    non_token->add_option("--length", manual_params.length, "Lengths")->expected(1, 3);
    non_token
        ->add_option("-b, --batchSize",
                     manual_params.nbatch,
                     "If this value is greater than one, arrays will be used")
        ->default_val(1);
    non_token->add_option("--istride", manual_params.istride, "Input stride");
    non_token->add_option("--ostride", manual_params.ostride, "Output stride");
    non_token->add_option("--idist", manual_params.idist, "Logical distance between input batches")
        ->default_val(0);
    non_token->add_option("--odist", manual_params.odist, "Logical distance between output batches")
        ->default_val(0);
    non_token->add_option("--ioffset", manual_params.ioffset, "Input offset");
    non_token->add_option("--ooffset", manual_params.ooffset, "Output offset");
    non_token->add_option("--isize", manual_params.isize, "Logical size of input buffer");
    non_token->add_option("--osize", manual_params.osize, "Logical size of output buffer");
    non_token->add_option(
        "--scalefactor", manual_params.scale_factor, "Scale factor to apply to output");
    // Default value is set in fft_params.h based on if device-side PRNG was enabled.
    non_token->add_option("-g, --inputGen",
                          manual_params.igen,
                          "Input data generation:\n0) PRNG sequence (device)\n"
                          "1) PRNG sequence (host)\n"
                          "2) linearly-spaced sequence (device)\n"
                          "3) linearly-spaced sequence (host)");
    const auto* opt_version = app.add_flag(
        "--version",
        "Print queryable version information from the hipfft library's backend (and return)");
    app.add_option("--R", ramgb, "RAM limit in GiB for tests")
        ->default_val(host_memory::singleton().get_total_gbytes());
    app.add_option("--V", vramgb, "VRAM limit in GiB for tests")->default_val(0);
    app.add_option("--half_epsilon", half_epsilon)->default_val(9.77e-4);
    app.add_option("--single_epsilon", single_epsilon)->default_val(3.75e-5);
    app.add_option("--double_epsilon", double_epsilon)->default_val(1e-15);
    app.add_option("--skip_runtime_fails",
                   skip_runtime_fails,
                   "Skip the test if there is a runtime failure")
        ->default_val(true);
    app.add_option("-w, --wise", use_fftw_wisdom, "Use FFTW wisdom");
    // Filename for fftw and fftwf wisdom.
    std::string fftw_wisdom_filename;
    app.add_option("-W, --wisdomfile", fftw_wisdom_filename, "FFTW3 wisdom filename")
        ->default_val("wisdom3.txt");
    // Filename for precompiled kernels to be written to
    std::string precompile_file;
    app.add_option("--precompile",
                   precompile_file,
                   "Precompile kernels to a file for all test cases before running tests")
        ->default_val("");
    // Try parsing initial args that will be used to configure tests
    // Allow extras to pass on gtest arguments without error
    app.allow_extras();
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    if(!test_token.empty())
    {
        std::cout << "Reading fft params from token:\n" << test_token << std::endl;

        try
        {
            manual_params.from_token(test_token);
            std::cout << "manual_params.token() = " << manual_params.token() << std::endl;
        }
        catch(...)
        {
            std::cout << "Unable to parse token." << std::endl;
            return 1;
        }
    }
    if(manual_params.length.empty())
    {
        manual_params.length.push_back(8);
        // TODO: add random size?
    }

    if(manual_params.istride.empty())
    {
        manual_params.istride.push_back(1);
        // TODO: add random size?
    }

    if(manual_params.ostride.empty())
    {
        manual_params.ostride.push_back(1);
        // TODO: add random size?
    }

    // User-settable options defining the values of all the actual test parameters
    // (e.g., probability factors and value of manual_params) must be handled
    // before invoking ::testing::InitGoogleTest as it triggers evaluation of said
    // parameters (e.g., args of "::testing::Values{In}" in instantiations of test
    // suites).
    // set any "unset" parameters of manual_params before initiating gtests
    // (makes the token reported by gtest less ambiguous)
    manual_params.validate();

    // extract remaining arguments for subsequent gtest initialization
    std::vector<std::string> remaining_args = app.remaining();
    std::string              gtest_help_opt = "--help";
    // NB: If we initialize gtest first, then it removes all of its own command-line
    // arguments and sets argc and argv correctly;
    std::vector<char*> gtest_argv;
    gtest_argv.insert(gtest_argv.begin(), argv[0]);
    for(std::string& s : remaining_args)
    {
        gtest_argv.push_back(&s[0]);
    }
    if(*opt_help)
    {
        // make sure gtest prints its help as well
        gtest_argv.push_back(&gtest_help_opt[0]);
    }
    gtest_argv.push_back(NULL);
    decltype(argc) gtest_argc = gtest_argv.size() - 1;
    ::testing::InitGoogleTest(&gtest_argc, gtest_argv.data()); // gtest-relevant args are removed

    if(*opt_help)
    {
        std::cout << app.help() << "\n";
        return EXIT_SUCCESS;
    }
    // no help was used, gtest_argc is expected to be 1 at this point. If not, some of the
    // used options were not recognized at all
    if(gtest_argc > 1)
    {
        std::cout << "Unrecognised option(s) found:\n  ";
        for(auto i = 1; i < gtest_argc; i++)
            std::cout << gtest_argv[i] << " ";
        std::cout << "\nRun with --help for more information.\n";
        return EXIT_FAILURE;
    }

    if(*opt_version || verbose > 0)
    {
        int hipfft_version;
        hipfftGetVersion(&hipfft_version);
        std::cout << "hipFFT version: " << hipfft_version << std::endl;
        if(*opt_version)
        {
            return EXIT_SUCCESS;
        }
    }

    std::cout << "Using random_seed = " << random_seed << std::endl;
    std::cout << "half epsilon: " << half_epsilon << "\tsingle epsilon: " << single_epsilon
              << "\tdouble epsilon: " << double_epsilon << std::endl;

    // if precompiling, tell rocFFT to use the specified cache file
    // to write kernels to
    //
    // but if our environment already has a cache file for RTC, then
    // we should just use that
    std::unique_ptr<EnvironmentSetTemp> env_precompile;
    if(!precompile_file.empty() && rocfft_getenv("ROCFFT_RTC_CACHE_PATH").empty())
    {
        env_precompile = std::make_unique<EnvironmentSetTemp>("ROCFFT_RTC_CACHE_PATH",
                                                              precompile_file.c_str());
    }

#ifdef FFTW_MULTITHREAD
    fftw_init_threads();
    fftwf_init_threads();
    fftw_plan_with_nthreads(rocfft_concurrency());
    fftwf_plan_with_nthreads(rocfft_concurrency());
#endif

    // Set host memory limit from command-line options
    host_memory::singleton().set_limit_gbytes(ramgb);
    std::cout << "Host memory limit: " << ramgb << " GiB" << std::endl;

    if(use_fftw_wisdom)
    {
        if(verbose)
        {
            std::cout << "Using " << fftw_wisdom_filename << " wisdom file\n";
        }
        std::ifstream fftw_wisdom_file(fftw_wisdom_filename);
        std::string   allwisdom = std::string(std::istreambuf_iterator<char>(fftw_wisdom_file),
                                            std::istreambuf_iterator<char>());

        std::string fftw_wisdom;
        std::string fftwf_wisdom;

        bool               load_wisdom  = false;
        bool               load_fwisdom = false;
        std::istringstream input;
        input.str(allwisdom);
        // Separate the single-precision and double-precision wisdom:
        for(std::string line; std::getline(input, line);)
        {
            if(line.rfind("(fftw", 0) == 0 && line.find("fftw_wisdom") != std::string::npos)
            {
                load_wisdom = true;
            }
            if(line.rfind("(fftw", 0) == 0 && line.find("fftwf_wisdom") != std::string::npos)
            {
                load_fwisdom = true;
            }
            if(load_wisdom)
            {
                fftw_wisdom.append(line + "\n");
            }
            if(load_fwisdom)
            {
                fftwf_wisdom.append(line + "\n");
            }
            if(line.rfind(")", 0) == 0)
            {
                load_wisdom  = false;
                load_fwisdom = false;
            }
        }
        fftw_import_wisdom_from_string(fftw_wisdom.c_str());
        fftwf_import_wisdom_from_string(fftwf_wisdom.c_str());
    }

    if(!precompile_file.empty())
        precompile_test_kernels(precompile_file);

    auto retval = RUN_ALL_TESTS();

    if(use_fftw_wisdom)
    {
        std::string fftw_wisdom  = std::string(fftw_export_wisdom_to_string());
        std::string fftwf_wisdom = std::string(fftwf_export_wisdom_to_string());
        fftw_wisdom.append(std::string(fftwf_export_wisdom_to_string()));
        std::ofstream fftw_wisdom_file(fftw_wisdom_filename);
        fftw_wisdom_file << fftw_wisdom;
        fftw_wisdom_file << fftwf_wisdom;
        fftw_wisdom_file.close();
    }

    std::cout << "half precision max l-inf epsilon: " << max_linf_eps_half << std::endl;
    std::cout << "half precision max l2 epsilon:     " << max_l2_eps_half << std::endl;
    std::cout << "single precision max l-inf epsilon: " << max_linf_eps_single << std::endl;
    std::cout << "single precision max l2 epsilon:     " << max_l2_eps_single << std::endl;
    std::cout << "double precision max l-inf epsilon: " << max_linf_eps_double << std::endl;
    std::cout << "double precision max l2 epsilon:     " << max_l2_eps_double << std::endl;

    hipfft_params::externally_managed_workareas.clear();

    return retval;
}

// instantiation of the paramameterized accuracy_test for the
// configuration set manually:
INSTANTIATE_TEST_SUITE_P(manual,
                         accuracy_test,
                         ::testing::Values(static_cast<const fft_params&>(manual_params)),
                         accuracy_test::TestName);
