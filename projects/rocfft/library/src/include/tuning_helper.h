
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

#ifndef TUNING_HELPER_H
#define TUNING_HELPER_H

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "solution_map.h"

struct BenchmarkInfo
{
    std::string        prob_token;
    std::string        kernel_name;
    std::string        factors_str; // factors string as "[a, b, c]" as a feild in CSV
    std::string        util_rate; // utilization is define as "how many butterflies per thread does"
    int                tuning_phase; // phase-0/1, see comments of PropagateBestFactorsToNextPhase()
    int                SSN; // the serial number of the kernel-candidate
    int                num_blocks; // following is information of current kernel execution
    int                workgroup_size;
    std::array<int, 2> threads_per_trans;
    int                trans_per_block;
    int                LDS_bytes;
    int                globalRW_per_thread;
    int                occupancy;
    int                numCUs;
    double             milli_seconds;
    double             gflops;
    double             granularity;
    double             bw_eff;
};

struct rocfft_tuning_packet
{
    // tuning result, vector size is num_nodes
    std::string              tuning_arch_name;
    std::string              tuning_problem_name;
    std::string              output_solution_map_path;
    std::vector<std::string> tuning_kernel_tokens;
    std::vector<std::string> kernel_names;
    std::vector<std::string> factors_str;
    std::vector<std::string> util_rate;
    std::vector<bool>        is_builtin_kernel;
    std::vector<double>      bw_effs;
    std::vector<size_t>      num_of_blocks;
    std::vector<size_t>      wgs;
    std::vector<size_t>      tpt0;
    std::vector<size_t>      tpt1;
    std::vector<size_t>      tpb;
    std::vector<size_t>      lds_bytes;
    std::vector<int>         globalRW_per_thread; // 2D-kernel
    std::vector<int>         occupancy; // we allow -1, indicating the RTC kernel compiled failed
    int                      numCUs;

    // setting
    bool dump_candidates   = false;
    bool export_full_token = false;
    // reserved, indicating if we dump a full token,
    // making the solution exclusively use by that exact problem

    // tuning status
    bool             init_step      = false;
    bool             is_tuning      = false;
    int              total_nodes    = 0;
    int              tuning_node_id = -1;
    int              current_ssn    = -1;
    int              tuning_phase   = 0; // 0: no factor permutation; 1: permute
    std::vector<int> total_candidates;

    // size is #-nodes, each elem indicates in which phase, which id, what name is the winner
    std::vector<int>         winner_phases;
    std::vector<int>         winner_ids;
    std::vector<std::string> winner_kernel_names;

    // size is #-nodes, each elem is the target_factors of this node
    std::vector<std::set<std::string>> target_factors;

    rocfft_tuning_packet() = default;
};

class TuningBenchmarker
{
private:
    solution_map*                         binding_solution_map = nullptr;
    std::unique_ptr<rocfft_tuning_packet> packet               = nullptr;
    // outter vector size is #-nodes, each elem is a vector with size = #- kernel-candidates
    std::vector<std::vector<BenchmarkInfo>> benchmark_infos_of_node;

    void ResetKernelInfo();

    TuningBenchmarker() = default;

public:
    TuningBenchmarker(const TuningBenchmarker&) = delete;

    TuningBenchmarker& operator=(const TuningBenchmarker&) = delete;

    static TuningBenchmarker& GetSingleton()
    {
        static TuningBenchmarker singleton;
        return singleton;
    }

    ~TuningBenchmarker();

    // create packet
    void Setup();

    // release packet
    void Clean();

    rocfft_tuning_packet* GetPacket();

    void SetBindingSolutionMap(solution_map* sol_map);

    solution_map* GetBindingSolutionMap();

    // the status query
    bool IsInitializingTuning();
    bool IsProcessingTuning();

    bool SetInitStep(int tuning_phase);

    int UpdateNumOfTuningNodes();

    int GetNumOfKernelCandidates(size_t node_id);

    bool SetCurrentTuningNodeId(size_t node_id);

    bool SetCurrentKernelCandidateId(size_t kernel_config_id);

    BenchmarkInfo GetCurrBenchmarkInfo();

    void UpdateCurrBenchResult(double ms, double gflops);

    void FindWinnerForCurrNode(double&      curr_best_msec,
                               int&         winner_phase,
                               int&         winner_config_id,
                               std::string& winner_kernel_name);

    void ExportWinnerToSolutions();

    // We do a 2-phase tuning:
    //  phase 0: tuning without permuting the factors, and get best 3 factors sets
    //  phase 1: propagate the best 3 factors to phase 1, and do permutation.
    void PropagateBestFactorsToNextPhase();

    void GetOutputSolutionMapPath(std::string& out_path);

    bool ExportCSV(bool append_data = false);

    bool MergingSolutionsMaps(const std::string& base_map_path,
                              const std::string& new_map_path,
                              const std::string& probKeyStr,
                              const std::string& out_map_path);
};

#endif // TUNING_PACKET_H