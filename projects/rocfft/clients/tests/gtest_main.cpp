// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
/// @brief googletest based unit tester for rocfft
///

#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <streambuf>
#include <string>
#include <thread>

#include <initializer_list>
#include <list>

#include "../../shared/CLI11.hpp"
#include "../../shared/concurrency.h"
#include "../../shared/device_properties.h"
#include "../../shared/environment.h"
#include "../../shared/hostbuf.h"
#include "../../shared/rocfft_accuracy_test.h"
#include "../../shared/sys_mem.h"
#include "../../shared/test_params.h"
#include "../../shared/work_queue.h"
#include "bitwise_repro/bitwise_repro_db.h"
#include "bitwise_repro/bitwise_repro_test.h"
#include "rocfft/rocfft.h"

// Control output verbosity:
int verbose;

// User-defined random seed
size_t             random_seed;
std::random_device default_seed_dev;
// Overall probability of running conventional tests
double test_prob;
// Probability of running tests from the emulation suite
double emulation_prob;
// Probability of running unit tests
double unittest_prob;
// Modifier for probability of running tests with complex interleaved data
double complex_interleaved_prob_factor;
// Modifier for probability of running tests with real data
double real_prob_factor;
// Modifier for probability of running tests with complex planar data
double complex_planar_prob_factor;
// Modifier for probability of running tests with callbacks
double callback_prob_factor;

// Number of random tests per suite
size_t n_random_tests = 0;

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(bitwise_repro_test);

// Transform parameters for manual test:
fft_params manual_params;

// Host memory limitation for tests (GiB):
size_t ramgb;

// Device memory limitation for tests (GiB):
size_t vramgb;

// Number of hip devices to use.
int ngpus{};

// Allow skipping tests if there is a runtime error
bool skip_runtime_fails;
// But count the number of failures
int n_hip_failures = 0;

// Pointer to a bitwise repro-db file
std::unique_ptr<fft_hash_db> repro_db;

// Manually specified precision cutoffs:
double half_epsilon;
double single_epsilon;
double double_epsilon;

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

// Number of devices to distribute the FFT to for manual tests
int manual_devices = 1;

// Multi-process library to use
fft_params::fft_mp_lib mp_lib = fft_params::fft_mp_lib_none;
// Number of multi-process ranks to launch
int mp_ranks = 1;
// Multi-process launch command (e.g. mpirun --np 4 /path/to/rocfft_mpi_worker)
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

std::vector<std::string> tokens_to_run()
{
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
            if(name.find("vs_fftw/") != std::string::npos)
            {
                name.erase(0, 8);

                // Run any problem that uses brick decomposition
                // without touching batch.  Bricks are specified with
                // batch indexes, so arbitrarily changing batch to 1
                // can break those cases.
                if(name.find("_brick_") != std::string::npos)
                {
                    tokens.emplace_back(std::move(name));
                    continue;
                }

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
    return tokens;
}

void precompile_test_kernels(const std::string& precompile_file)
{
    std::cout << "precompiling test kernels...\n";
    WorkQueue<std::string> tokenQueue;

    auto tokens = tokens_to_run();

    std::random_device dev;
    std::mt19937       dist(dev());
    std::shuffle(tokens.begin(), tokens.end(), dist);
    auto precompile_begin = std::chrono::steady_clock::now();
    std::cout << "precompiling " << tokens.size() << " FFT plans...\n";

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
                    rocfft_params params_forward;
                    params_forward.from_token(token);
                    params_forward.validate();
                    params_forward.setup_structs();

                    params_forward.free();

                    rocfft_params params_inverse;
                    params_inverse.inverse_from_forward(params_forward);
                    params_inverse.validate();
                    params_inverse.setup_structs();
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
    const auto test_begin = std::chrono::system_clock::now();

    // We would like to parse a few arguments before initiating gtest.

    // Save argv[0] because CLI doesn't include this in the remaining args, and it's expected when
    // we re-parse the arguments with gtest and CLI.
    std::string argv0 = argv[0];

    CLI::App app{
        "\n"
        "rocFFT Runtime Test command line options\n"
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
    app.add_option("--nrand", n_random_tests, "Number of extra randomized tests")->default_val(0);

    app.add_option("--ngpus", ngpus, "Number of GPUs to use per rank")
        ->default_val(-1)
        ->check(CLI::NonNegativeNumber);
    app.add_option("--gpus", n_random_tests, "Number of extra randomized tests")->default_val(0);
    app.add_option("--test_prob", test_prob, "Probability of running individual tests")
        ->default_val(1.0)
        ->check(CLI::Range(0.0, 1.0));
    app.add_option("--unittest_prob", unittest_prob, "Probability of running individual unit tests")
        ->default_val(1.0)
        ->check(CLI::Range(0.0, 1.0));
    app.add_option(
           "--emulation_prob", emulation_prob, "Probability of running individual emulation tests")
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
        ->check(CLI::PositiveNumber);

    constexpr std::array<std::string_view, 4> emulation_types
        = {"none", "smoke", "regression", "extended"};
    app.add_option("--emulation", "Run emulation tests")
        ->check(CLI::IsMember(emulation_types))
        ->each([&](const std::string& emulationtype) {
            constexpr auto nidx = [emulation_types](const auto name) {
                return std::find(emulation_types.begin(), emulation_types.end(), name)
                       - emulation_types.begin();
            };

            // Emulation test suites focus on well-established software paths; we are looking for
            // information about the hardware, which means that we aren't trying to find out a lot
            // of information about the software.  Thus, no randomly-generated tests.
            n_random_tests = 0;

            // Run all of the emulation tests:
            emulation_prob = 1.0;

            // Callbacks are not an emulation test target.
            callback_prob_factor = 0;

            // We can do a switch on nidx(emulationtype) when we have C++20
            // switch(nidx(emulationtype))
            // {
            // case nidx("smoke"):
            // etc.

            if(nidx(emulationtype) == nidx("smoke"))
            {
                // 2GB vram limit, approx 1 minute GPU time with short tests.
                vramgb         = 2;
                test_prob      = 0;
                emulation_prob = 0.005;
            }
            if(nidx(emulationtype) == nidx("regression"))
            {
                vramgb         = 16;
                emulation_prob = 1;
                test_prob      = 0.01;
            }
            if(nidx(emulationtype) == nidx("extended"))
            {
                emulation_prob = 1;
                test_prob      = 0.02;
            }
        });

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
                   "/path/to/rocfft_mpi_worker\"")
        ->default_val("")
        ->each([&](const std::string&) {
            if(mp_lib == fft_params::fft_mp_lib_none)
            {
                std::cout << "--mp_launch requires an mp library (see mp_lib in --help).\n";
                std::exit(EXIT_FAILURE);
            }
        })
        ->needs("--mp_lib");

    app.add_flag("--smoketest", "Run a short (approx 5 minute) randomized selection of tests")
        ->each([&](const std::string&) {
            // The objective is to have an test that takes about 5 minutes, so just set the
            // probability per test to a small value to achieve this result.
            test_prob      = 0.0005;
            emulation_prob = 0.005;
            unittest_prob  = 0.2;
            n_random_tests = 10;
        });

    app.add_flag("--callback", "Inject load/store callbacks")->each([&](const std::string&) {
        manual_params.run_callbacks = true;
    });

    app.add_option("--seed", random_seed, "Random seed; if unset, use an actual random seed")
        ->default_val(default_seed_dev());
    // Filename for fftw and fftwf wisdom.
    std::string fftw_wisdom_filename;

    // Token string to fully specify fft params for the manual test.
    std::string test_token;

    // Filename for precompiled kernels to be written to
    std::string precompile_file;

    // Full path to bitwise repro database file
    std::string repro_db_path;

    // Bool option to just print tokens and exit
    bool printtokens{false};

    // Declare the supported options. Some option pointers are declared to track passed opts.
    app.add_flag("--version", "Print queryable version information from the rocfft library")
        ->each([](const std::string&) {
            char v[256];
            rocfft_get_version_string(v, 256);
            std::cout << "version " << v << std::endl;
            return EXIT_SUCCESS;
        });

    app.add_flag("--checkstride", "Check that data is not written outside of output strides")
        ->each([&](const std::string&) { manual_params.check_output_strides = true; });

    auto opt_token
        = app.add_option("--token", test_token, "Test token name for manual test")->default_val("");
    // Group together options that conflict with --token
    auto* non_token = app.add_option_group("Token Conflict", "Options excluded by --token");
    non_token
        ->add_flag("--double", "Double precision transform (deprecated: use --precision double)")
        ->each([&](const std::string&) { manual_params.precision = fft_precision_double; });
    non_token->excludes(opt_token);
    non_token
        ->add_option("-t, --transformType",
                     manual_params.transform_type,
                     "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
                     "forward\n3) real inverse")
        ->default_val(fft_transform_type_complex_forward);
    non_token
        ->add_option("--auto_allocation",
                     manual_params.auto_allocate,
                     "rocFFT's auto-allocation behavior: \"on\", \"off\", or \"default\"")
        ->default_val("default");
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
    app.add_option("--isize", manual_params.isize, "Logical size of input buffer");
    app.add_option("--osize", manual_params.osize, "Logical size of output buffer");
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
    app.add_option("-W, --wisdomfile", fftw_wisdom_filename, "FFTW3 wisdom filename")
        ->default_val("wisdom3.txt");
    app.add_option("--manual_devices",
                   manual_devices,
                   "Distribute manual test case among this many devices")
        ->default_val(1)
        ->check(CLI::PositiveNumber);
    app.add_option("--scalefactor", manual_params.scale_factor, "Scale factor to apply to output");

    app.add_option("--repro-db",
                   repro_db_path,
                   "Database file full path name for bitwise reproducibility tests");
    app.add_option("--precompile",
                   precompile_file,
                   "Precompile kernels to a file for all test cases before running tests")
        ->default_val("");
    app.add_flag("--printtokens", printtokens, "Print test tokens to scheduled to be run and exit");
    // Default value is set in fft_params.h based on if device-side PRNG was enabled.
    app.add_option("-g, --inputGen",
                   manual_params.igen,
                   "Input data generation:\n0) PRNG sequence (device)\n"
                   "1) PRNG sequence (host)\n"
                   "2) linearly-spaced sequence (device)\n"
                   "3) linearly-spaced sequence (host)");

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

    std::cout << "half epsilon: " << half_epsilon << "\tsingle epsilon: " << single_epsilon
              << "\tdouble epsilon: " << double_epsilon << std::endl;
    std::cout << "Random seed: " << random_seed << std::endl;

    // If precompiling, tell rocFFT to use the specified cache file
    // to write kernels to
    //
    // But if our environment already has a cache file for RTC, then
    // we should just use that
    std::unique_ptr<EnvironmentSetTemp> env_precompile;
    if(!precompile_file.empty() && rocfft_getenv("ROCFFT_RTC_CACHE_PATH").empty())
    {
        env_precompile = std::make_unique<EnvironmentSetTemp>("ROCFFT_RTC_CACHE_PATH",
                                                              precompile_file.c_str());
    }

    rocfft_setup();
    {
        char v[256];
        rocfft_get_version_string(v, 256);
        std::cout << "rocFFT version: " << v << std::endl;
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

    if(!repro_db_path.empty())
        repro_db.reset(new fft_hash_db(repro_db_path));

    if(!test_token.empty())
    {
        std::cout << "Reading fft params from token:\n" << test_token << std::endl;

        try
        {
            manual_params.from_token(test_token);
        }
        catch(...)
        {
            std::cout << "Unable to parse token." << std::endl;
            return EXIT_FAILURE;
        }
    }
    else
    {
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
    }

    if(!precompile_file.empty())
        precompile_test_kernels(precompile_file);

    if(printtokens)
    {
        std::cout << "Tokens:" << std::endl;
        const auto tokens = tokens_to_run();
        for(const auto& token : tokens)
        {
            std::cout << token << std::endl;
        }
        return EXIT_SUCCESS;
    }

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

    rocfft_cleanup();

    const auto test_duration = std::chrono::system_clock::now() - test_begin;
    const auto test_hours    = std::chrono::duration_cast<std::chrono::hours>(test_duration);
    const auto test_minutes
        = std::chrono::duration_cast<std::chrono::minutes>(test_duration - test_hours);
    std::cout << "Test suite took " << test_hours.count() << " hours " << test_minutes.count()
              << " minutes\n"
              << std::endl;

    std::cout << "half precision max l-inf epsilon: " << max_linf_eps_half << "\n";
    std::cout << "half precision max l2 epsilon:     " << max_l2_eps_half << "\n";
    std::cout << "single precision max l-inf epsilon: " << max_linf_eps_single << "\n";
    std::cout << "single precision max l2 epsilon:     " << max_l2_eps_single << "\n";
    std::cout << "double precision max l-inf epsilon: " << max_linf_eps_double << "\n";
    std::cout << "double precision max l2 epsilon:     " << max_l2_eps_double << "\n";
    std::cout << "Number of runtime issues: " << n_hip_failures << "\n";
    std::cout << "Number of successful tests: "
              << ::testing::UnitTest::GetInstance()->successful_test_count() << "\n";
    std::cout << "Number of skipped tests: "
              << ::testing::UnitTest::GetInstance()->skipped_test_count() << "\n";

    std::cout << "\nRandom seed: " << random_seed << std::endl;

    return retval;
}

TEST(manual, vs_fftw) // MANUAL TESTS HERE
{
    rocfft_params params(manual_params);

    if(manual_devices > 1)
    {
        // just distribute along the slowest FFT dimension
        std::vector<unsigned int> deviceGrid(params.length.size() + 1, 1);
        deviceGrid[1] = manual_devices;

        params.distribute_input(manual_devices, deviceGrid);
        params.distribute_output(manual_devices, deviceGrid);
    }

    // Run an individual test using the provided command-line parameters.
    params.validate();

    std::cout << "Manual test:"
              << "\n\t" << params.str("\n\t") << "\n";

    std::cout << "Token: " << params.token() << "\n";

    if(!params.valid(verbose + 2))
    {
        std::cout << "manual params are not valid\n";
    }

    try
    {
        fft_vs_reference(params);
    }
    catch(std::bad_alloc&)
    {
        GTEST_SKIP() << "host memory allocation failure";
    }
    catch(HOSTBUF_MEM_USAGE& e)
    {
        // explicitly clear test cache
        last_cpu_fft_data = last_cpu_fft_cache();
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_SKIP& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_FAIL& e)
    {
        GTEST_FAIL() << e.msg;
    }
}

TEST(manual, bitwise_reproducibility) // MANUAL TESTS HERE
{
    if(repro_db == nullptr)
        GTEST_SKIP() << "A database file is required for this test." << std::endl;

    rocfft_params params(manual_params);

    // Run an individual test using the provided command-line parameters.
    params.validate();

    std::cout << "Manual test:"
              << "\n\t" << params.str("\n\t") << "\n";

    std::cout << "Token: " << params.token() << "\n";

    if(!params.valid(verbose + 2))
    {
        std::cout << "manual params are not valid\n";
    }

    try
    {
        bitwise_repro(params);
    }
    catch(std::bad_alloc&)
    {
        GTEST_SKIP() << "host memory allocation failure";
    }
    catch(ROCFFT_SKIP& e)
    {
        GTEST_SKIP() << e.msg;
    }
    catch(ROCFFT_FAIL& e)
    {
        GTEST_FAIL() << e.msg;
    }
    SUCCEED();
}
