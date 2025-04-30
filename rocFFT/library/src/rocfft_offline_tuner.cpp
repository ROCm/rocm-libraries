// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../shared/CLI11.hpp"
#include "../../shared/environment.h"
#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/rocfft_params.h"
#include "rocfft/rocfft.h"
#include "tuning_helper.h"

inline void
    hip_V_Throw(hipError_t res, const std::string& msg, size_t lineno, const std::string& fileName)
{
    if(res != hipSuccess)
    {
        std::stringstream tmp;
        tmp << "HIP_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
}

inline void
    lib_V_Throw(fft_status res, const std::string& msg, size_t lineno, const std::string& fileName)
{
    if(res != fft_status_success)
    {
        std::stringstream tmp;
        tmp << "LIB_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
}

#define HIP_V_THROW(_status, _message) hip_V_Throw(_status, _message, __LINE__, __FILE__)
#define LIB_V_THROW(_status, _message) lib_V_Throw(_status, _message, __LINE__, __FILE__)

int merge_solutions(const std::string& base_filename,
                    const std::string& new_filename,
                    const std::string& probKey,
                    const std::string& out_filename)
{
    // don't use anything from solutions.cpp
    rocfft_setenv("ROCFFT_USE_EMPTY_SOL_MAP", "1");

    rocfft_setup();

    // create tuning parameters
    TuningBenchmarker* offline_tuner = nullptr;
    rocfft_get_offline_tuner_handle((void**)(&offline_tuner));

    // Manupulating the solution map from tuner...
    bool merge_result
        = offline_tuner->MergingSolutionsMaps(base_filename, new_filename, probKey, out_filename);

    rocfft_cleanup();

    if(!merge_result)
    {
        std::cout << "Merge Solutions Failed" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int offline_tune_problems(rocfft_params& params, int verbose, int ntrial)
{
    // don't use anything from solutions.cpp
    rocfft_setenv("ROCFFT_USE_EMPTY_SOL_MAP", "1");

    rocfft_setup();

    params.validate();

    if(!params.valid(verbose))
        throw std::runtime_error("Invalid parameters, add --verbose=1 for detail");
    if(verbose)
        std::cout << params.str(" ") << std::endl;

    std::cout << "Token: " << params.token() << std::endl;

    // create tuning parameters
    TuningBenchmarker* offline_tuner = nullptr;
    rocfft_get_offline_tuner_handle((void**)(&offline_tuner));

    // first time call create_plan is actually generating a bunch of combination of configs
    offline_tuner->SetInitStep(0);

    // Check free and total available memory:
    size_t free  = 0;
    size_t total = 0;
    HIP_V_THROW(hipMemGetInfo(&free, &total), "hipMemGetInfo failed");
    const auto raw_vram_footprint
        = params.fft_params_vram_footprint() + twiddle_table_vram_footprint(params);
    if(!vram_fits_problem(raw_vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << raw_vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    const auto vram_footprint = params.vram_footprint();
    if(!vram_fits_problem(vram_footprint, free))
    {
        std::cout << "SKIPPED: Problem size (" << vram_footprint
                  << ") raw data too large for device.\n";
        return EXIT_SUCCESS;
    }

    LIB_V_THROW(params.create_plan(), "Plan creation failed");

    // GPU input buffer:
    auto                ibuffer_sizes = params.ibuffer_sizes();
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        HIP_V_THROW(ibuffer[i].alloc(ibuffer_sizes[i]), "Creating input Buffer failed");
        pibuffer[i] = ibuffer[i].data();
    }

// Input data:
#ifdef USE_HIPRAND
    params.compute_input(ibuffer);
#else
    // Use host-side input generation
    std::vector<hostbuf> gpu_input_data
        = allocate_host_buffer(params.precision, params.itype, ibuffer_sizes);
    params.compute_input(gpu_input_data);

    // Copy input to GPU
    for(unsigned int idx = 0; idx < gpu_input_data.size(); ++idx)
    {
        HIP_V_THROW(hipMemcpy(ibuffer[idx].data(),
                              gpu_input_data.at(idx).data(),
                              ibuffer_sizes[idx],
                              hipMemcpyHostToDevice),
                    "hipMemcpy failed");
    }
#endif

    // GPU output buffer:
    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    if(params.placement == fft_placement_inplace)
    {
        obuffer = &ibuffer;
    }
    else
    {
        auto obuffer_sizes = params.obuffer_sizes();
        obuffer_data.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer_data.size(); ++i)
        {
            HIP_V_THROW(obuffer_data[i].alloc(obuffer_sizes[i]), "Creating output Buffer failed");
        }
    }
    std::vector<void*> pobuffer(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    // finish initialization, solution map now contains all the candidates
    // start doing real benchmark with different configurations
    int num_nodes = offline_tuner->UpdateNumOfTuningNodes();
    if(num_nodes == 0)
    {
        std::cout << "[Result]: This fft problem hasn't been supported yet. (Prime number or "
                     "2D-Single)"
                  << std::endl;
        rocfft_cleanup();
        return EXIT_FAILURE;
    }

    static const double max_double = std::numeric_limits<double>().max();

    bool                     csv_is_created    = false;
    double                   overall_best_time = max_double;
    std::vector<int>         winner_phases     = std::vector<int>(num_nodes, 0);
    std::vector<int>         winner_ids        = std::vector<int>(num_nodes, 0);
    std::vector<std::string> kernels           = std::vector<std::string>(num_nodes, "");
    std::vector<double>      node_best_times   = std::vector<double>(num_nodes, max_double);

    // calculate this once only
    const double totsize = product(params.length.begin(), params.length.end());
    const double k
        = ((params.itype == fft_array_type_real) || (params.otype == fft_array_type_real)) ? 2.5
                                                                                           : 5.0;
    const double opscount = (double)params.nbatch * k * totsize * log(totsize) / log(2.0);

    static const int TUNING_PHASE = 2;
    for(int curr_phase = 0; curr_phase < TUNING_PHASE; ++curr_phase)
    {
        if(curr_phase > 0)
        {
            // SET TARGET_FACTOR and current PHASE
            offline_tuner->SetInitStep(curr_phase);

            // make sure we can re-create the plan
            params.free();

            LIB_V_THROW(params.create_plan(), "Plan creation failed");
        }

        // keeping creating plan
        for(int node_id = 0; node_id < num_nodes; ++node_id)
        {
            std::string winner_name;
            int         winner_phase;
            int         winner_id;
            int         num_benchmarks = offline_tuner->GetNumOfKernelCandidates(node_id);

            offline_tuner->SetCurrentTuningNodeId(node_id);
            for(int ssn = 0; ssn < num_benchmarks; ++ssn)
            {
                offline_tuner->SetCurrentKernelCandidateId(ssn);
                std::cout << "\nTuning for node " << node_id << "/" << (num_nodes - 1)
                          << ", tuning phase :" << curr_phase << "/" << (TUNING_PHASE - 1)
                          << ", config :" << ssn << "/" << (num_benchmarks - 1) << std::endl;

                // make sure we can re-create the plan
                params.free();

                LIB_V_THROW(params.create_plan(), "Plan creation failed");

                // skip low occupancy test...simple output gflops 0, and a max double as ms
                BenchmarkInfo info = offline_tuner->GetCurrBenchmarkInfo();
                // we allow 2D_SINGLE kernels with occupancy 1
                if(info.threads_per_trans[1] != 0)
                {
                    if(info.occupancy < 0)
                    {
                        std::cout << "\nOccupancy -1 (unable to gen kernel), Skipped" << std::endl;
                        offline_tuner->UpdateCurrBenchResult(max_double, 0);
                        continue;
                    }
                }
                else
                {
                    if(info.occupancy == 1 || info.occupancy < 0)
                    {
                        std::cout << "\nOccupancy 1 or -1, Skipped" << std::endl;
                        offline_tuner->UpdateCurrBenchResult(max_double, 0);
                        continue;
                    }
                }

                params.execute(pibuffer.data(), pobuffer.data());

                // Run the transform several times and record the execution time:
                std::vector<double> gpu_time(ntrial);

                hipEvent_wrapper_t start, stop;
                start.alloc();
                stop.alloc();
                for(unsigned int itrial = 0; itrial < gpu_time.size(); ++itrial)
                {
                    HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

                    params.execute(pibuffer.data(), pobuffer.data());

                    HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
                    HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

                    float time;
                    HIP_V_THROW(hipEventElapsedTime(&time, start, stop),
                                "hipEventElapsedTime failed");
                    gpu_time[itrial] = time;
                }

                std::cout << "Execution gpu time:";
                for(const auto& i : gpu_time)
                {
                    std::cout << " " << i;
                }
                std::cout << " ms" << std::endl;

                std::cout << "Execution gflops:  ";
                for(const auto& i : gpu_time)
                {
                    double gflops = opscount / (1e6 * i);
                    std::cout << " " << gflops;
                }
                std::cout << std::endl;

                // get median, if odd, get middle one, else get avg(middle twos)
                std::sort(gpu_time.begin(), gpu_time.end());
                double ms_median
                    = (gpu_time.size() % 2 == 1)
                          ? gpu_time[gpu_time.size() / 2]
                          : (gpu_time[gpu_time.size() / 2] + gpu_time[gpu_time.size() / 2 - 1]) / 2;
                double gflops_median = opscount / (1e6 * ms_median);

                offline_tuner->UpdateCurrBenchResult(ms_median, gflops_median);
                overall_best_time = std::min(overall_best_time, ms_median);
            }

            offline_tuner->FindWinnerForCurrNode(
                node_best_times[node_id], winner_phase, winner_id, winner_name);
            std::cout << "\n[UP_TO_PHASE_" << curr_phase << "_RESULT]:" << std::endl;
            std::cout << "\n[BEST_KERNEL]: In Phase: " << winner_phase
                      << ", Config ID: " << winner_id << std::endl;

            // update the latest winner info
            winner_phases[node_id] = winner_phase;
            winner_ids[node_id]    = winner_id;
            kernels[node_id]       = winner_name;

            bool is_last_phase = (curr_phase == TUNING_PHASE - 1);
            bool is_last_node  = (node_id == num_nodes - 1);

            // output data of this turn to csv
            csv_is_created = offline_tuner->ExportCSV(csv_is_created) || csv_is_created;
            if(!csv_is_created)
                std::cout << "CSV is not created or is written failed." << std::endl;

            // pass the target factors to next phase with permutation
            if(!is_last_phase)
                offline_tuner->PropagateBestFactorsToNextPhase();

            // in last phase and last node: finished tuning
            // export to file (output the winner solutions to solution map)
            if(is_last_phase && is_last_node)
                offline_tuner->ExportWinnerToSolutions();
        }
    }

    std::string out_path;
    offline_tuner->GetOutputSolutionMapPath(out_path);

    std::cout << "\n[OUTPUT_FILE]: " << out_path << std::endl;
    std::cout << "\n[BEST_SOLUTION]: " << params.token() << std::endl;
    for(int node_id = 0; node_id < num_nodes; ++node_id)
    {
        std::cout << "[Result]: Node " << node_id << ":" << std::endl;
        std::cout << "[Result]:     in phase   : " << winner_phases[node_id] << std::endl;
        std::cout << "[Result]:     best config: " << winner_ids[node_id] << std::endl;
        std::cout << "[Result]:     kernel name: " << kernels[node_id] << std::endl;
    }
    double best_gflops = opscount / (1e6 * overall_best_time);
    std::cout << "[Result]: GPU Time: " << overall_best_time << std::endl;
    std::cout << "[Result]: GFLOPS: " << best_gflops << std::endl;

    rocfft_cleanup();

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    // FFT parameters:
    rocfft_params params;
    // Token string to fully specify fft params.
    std::string token;

    bool notInPlace{false};
    int  verbose;
    int  deviceId;
    int  ntrial;

    std::string base_sol_filename   = "";
    std::string adding_sol_filename = "";
    std::string adding_problemkey   = "";
    std::string output_sol_filename = "";

    CLI::App app{"rocFFT offline tuner"};

    // Declare the supported options. Some option pointers are declared to track passed opts.
    app.add_flag("-v, --version", "Print queryable version information from the rocfft library")
        ->each([](const std::string& val) {
            char v[256];
            rocfft_get_version_string(v, 256);
            std::cout << "version " << v << std::endl;
            return EXIT_SUCCESS;
        });

    auto tuning = app.add_subcommand("tune", "Tuning solution subcommand");

    tuning->add_option("-d, --device", deviceId, "Select a specific device id")
        ->default_val(0)
        ->check(CLI::NonNegativeNumber);
    tuning->add_option("--verbose", verbose, "Control output verbosity")->default_val(0);
    tuning->add_option("-N, --ntrial", ntrial, "Trial size for the problem")
        ->default_val(1)
        ->check(CLI::NonNegativeNumber);
    tuning
        ->add_option("-t, --transformType",
                     params.transform_type,
                     "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
                     "forward\n3) real inverse")
        ->default_val(fft_transform_type_complex_forward);
    CLI::Option* opt_token = tuning->add_option("--token", token, "rocFFT token string");

    auto* non_token = tuning->add_option_group("Token Conflict", "Options excluded by --token");
    non_token->excludes(opt_token);
    non_token->add_flag(
        "-o, --notInPlace", notInPlace, "Not in-place FFT transform (default: in-place)");
    non_token->add_option("--length", params.length, "Lengths")->required()->expected(1, 3);
    non_token->add_option(
        "--precision", params.precision, "Transform precision: single (default), double, half");
    non_token
        ->add_option("-b, --batchSize",
                     params.nbatch,
                     "If this value is greater than one, arrays will be used")
        ->default_val(1);
    non_token
        ->add_option("--itype",
                     params.itype,
                     "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
                     "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset);
    non_token
        ->add_option("--otype",
                     params.otype,
                     "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
                     "hermitian interleaved\n4) hermitian planar")
        ->default_val(fft_array_type_unset);

    tuning->callback([&]() {
        if(*opt_token)
        {
            std::cout << "Reading fft params from token:\n" << token << std::endl;
            try
            {
                params.from_token(token);
            }
            catch(...)
            {
                std::cout << "Unable to parse token." << std::endl;
                std::exit(-1);
            }
            return;
        }

        if(!notInPlace)
        {
            params.placement = fft_placement_inplace;
            std::cout << "in-place\n";
        }
        else
        {
            params.placement = fft_placement_notinplace;
            std::cout << "out-of-place\n";
        }

        std::cout << "length(s): ";
        for(auto& len : params.length)
            std::cout << len << " ";
        std::cout << "\n";
    });

    auto merging = app.add_subcommand("merge", "Merging solution map subcommand");

    merging->add_option("--base_sol_file", base_sol_filename, "Filename of base-solution-map")
        ->required()
        ->check(CLI::ExistingFile);
    merging->add_option("--new_sol_file", adding_sol_filename, "Filename of new-solution-map")
        ->required()
        ->check(CLI::ExistingFile);
    merging
        ->add_option("--new_probkey",
                     adding_problemkey,
                     "Problemkey (\"arch:token\") of the solution to be added, (looking up the "
                     "new-solution-map)")
        ->required();
    merging->add_option("--output_sol_file", output_sol_filename, "Filename of merged-solution-map")
        ->required()
        ->check(CLI::ExistingFile);

    app.require_subcommand(0, 1);

    CLI11_PARSE(app, argc, argv);

    if(tuning->parsed())
    {
        std::cout << std::flush;
        return offline_tune_problems(params, verbose, ntrial);
    }

    if(merging->parsed())
    {
        return merge_solutions(
            base_sol_filename, adding_sol_filename, adding_problemkey, output_sol_filename);
    }

    if(!tuning->parsed() && !merging->parsed())
        std::cout << app.help() << std::endl;
}
