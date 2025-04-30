// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tuning_helper.h"
#include "../../shared/environment.h"
#include "function_pool.h"
#include "rocfft/rocfft.h"
#include "solution_map.h"
#include "twiddles.h"

#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <unordered_set>

static const char* results_folder = "ResultSolutions";
static const char* csv_out_folder = "TuningData";

TuningBenchmarker::~TuningBenchmarker()
{
    if(binding_solution_map != nullptr || packet != nullptr)
        Clean();
}

void TuningBenchmarker::Setup()
{
    packet = std::unique_ptr<rocfft_tuning_packet>(new rocfft_tuning_packet());

    std::string dump_str = rocfft_getenv("DUMP_TUNING");
    if(!dump_str.empty())
        packet->dump_candidates = true;

    std::string exact_str = rocfft_getenv("TUNE_EXACT_PROB");
    if(!exact_str.empty())
        packet->export_full_token = true;
}

void TuningBenchmarker::Clean()
{
    binding_solution_map = nullptr;
    if(packet != nullptr)
        packet.reset();
    packet = nullptr;
}

rocfft_tuning_packet* TuningBenchmarker::GetPacket()
{
    return packet.get();
}

void TuningBenchmarker::SetBindingSolutionMap(solution_map* sol_map)
{
    binding_solution_map = sol_map;
}

solution_map* TuningBenchmarker::GetBindingSolutionMap()
{
    return binding_solution_map;
}

bool TuningBenchmarker::SetInitStep(int tuning_phase)
{
    packet->init_step      = true;
    packet->is_tuning      = false;
    packet->tuning_phase   = tuning_phase;
    packet->total_nodes    = 0;
    packet->current_ssn    = -1;
    packet->tuning_node_id = -1;

    packet->total_candidates.clear();

    for(auto& infos_for_candidates : benchmark_infos_of_node)
        infos_for_candidates.clear();

    return true;
}

bool TuningBenchmarker::IsInitializingTuning()
{
    return (packet && packet->init_step);
}

// true when during tuning, guarentee a valid GetPacket()
bool TuningBenchmarker::IsProcessingTuning()
{
    return (packet && packet->is_tuning);
}

// Called between each candidate
void TuningBenchmarker::ResetKernelInfo()
{
    int num_nodes = packet->total_nodes;

    packet->kernel_names.clear();
    packet->factors_str.clear();
    packet->util_rate.clear();
    packet->bw_effs.clear();
    packet->num_of_blocks.clear();
    packet->wgs.clear();
    packet->tpt0.clear();
    packet->tpt1.clear();
    packet->tpb.clear();
    packet->lds_bytes.clear();
    packet->globalRW_per_thread.clear();
    packet->occupancy.clear();

    packet->kernel_names.resize(num_nodes);
    packet->factors_str.resize(num_nodes);
    packet->util_rate.resize(num_nodes);
    packet->bw_effs.resize(num_nodes);
    packet->num_of_blocks.resize(num_nodes);
    packet->wgs.resize(num_nodes);
    packet->tpt0.resize(num_nodes);
    packet->tpt1.resize(num_nodes);
    packet->tpb.resize(num_nodes);
    packet->lds_bytes.resize(num_nodes);
    packet->globalRW_per_thread.resize(num_nodes);
    packet->occupancy.resize(num_nodes);
}

int TuningBenchmarker::UpdateNumOfTuningNodes()
{
    if(!packet)
        return 0;

    int total_nodes = packet->total_nodes;

    benchmark_infos_of_node.resize(total_nodes);
    packet->winner_phases.resize(total_nodes);
    packet->winner_ids.resize(total_nodes);
    packet->winner_kernel_names.resize(total_nodes);
    packet->target_factors.resize(total_nodes);

    return total_nodes;
}

int TuningBenchmarker::GetNumOfKernelCandidates(size_t node_id)
{
    if(packet && packet->total_candidates.size() > node_id)
        return packet->total_candidates[node_id];

    return 0;
}

bool TuningBenchmarker::SetCurrentTuningNodeId(size_t node_id)
{
    if(node_id >= (size_t)packet->total_nodes)
        return false;

    packet->tuning_node_id = node_id;
    return true;
}

bool TuningBenchmarker::SetCurrentKernelCandidateId(size_t kernel_config_id)
{
    size_t curr_node_id = packet->tuning_node_id;
    if(kernel_config_id >= (size_t)packet->total_candidates[curr_node_id])
        return false;

    packet->current_ssn = kernel_config_id;
    ResetKernelInfo();
    return true;
}

BenchmarkInfo TuningBenchmarker::GetCurrBenchmarkInfo()
{
    int   curr_phase       = packet->tuning_phase;
    int   tuning_node_id   = packet->tuning_node_id;
    int   kernel_config_id = packet->current_ssn;
    auto& bench_infos_vec  = benchmark_infos_of_node[tuning_node_id];

    BenchmarkInfo info;
    info.tuning_phase         = curr_phase;
    info.SSN                  = kernel_config_id;
    info.prob_token           = packet->tuning_problem_name;
    info.kernel_name          = packet->kernel_names[tuning_node_id];
    info.factors_str          = packet->factors_str[tuning_node_id];
    info.util_rate            = packet->util_rate[tuning_node_id];
    info.num_blocks           = packet->num_of_blocks[tuning_node_id];
    info.workgroup_size       = packet->wgs[tuning_node_id];
    info.threads_per_trans[0] = packet->tpt0[tuning_node_id];
    info.threads_per_trans[1] = packet->tpt1[tuning_node_id];
    info.trans_per_block      = packet->tpb[tuning_node_id];
    info.LDS_bytes            = packet->lds_bytes[tuning_node_id];
    info.globalRW_per_thread  = packet->globalRW_per_thread[tuning_node_id];
    info.occupancy            = packet->occupancy[tuning_node_id];
    info.numCUs               = packet->numCUs;
    info.granularity          = (double)info.num_blocks / info.numCUs;

    bench_infos_vec.push_back(info);

    return info;
}

void TuningBenchmarker::UpdateCurrBenchResult(double ms, double gflops)
{
    int    curr_tuning_node_id   = packet->tuning_node_id;
    int    curr_kernel_config_id = packet->current_ssn;
    double curr_node_bw_eff      = packet->bw_effs[curr_tuning_node_id];

    // un-sorted
    auto& bench_infos_vec = benchmark_infos_of_node[curr_tuning_node_id];
    auto& info            = bench_infos_vec[curr_kernel_config_id];
    info.bw_eff           = curr_node_bw_eff;
    info.milli_seconds    = ms;
    info.gflops           = gflops;
}

void TuningBenchmarker::FindWinnerForCurrNode(double&      curr_best_msec,
                                              int&         winner_phase,
                                              int&         winner_config_id,
                                              std::string& winner_kernel_name)
{
    int   curr_tuning_node_id = packet->tuning_node_id;
    auto& bench_infos_vec     = benchmark_infos_of_node[curr_tuning_node_id];

    // set 0 to build-in kernels
    if(packet->is_builtin_kernel[curr_tuning_node_id])
    {
        winner_phase       = 0;
        winner_config_id   = 0;
        winner_kernel_name = "built_in_kernel";
    }
    else
    {
        // set to previous result in case there is no new candidate in this phase
        // for example, len-125 can only be 5x5x5 so it does nothing in phase-2
        winner_phase       = packet->winner_phases[curr_tuning_node_id];
        winner_config_id   = packet->winner_ids[curr_tuning_node_id];
        winner_kernel_name = packet->winner_kernel_names[curr_tuning_node_id];
    }

    // if not empty, then sort and update IDs
    if(!bench_infos_vec.empty())
    {
        std::sort(
            bench_infos_vec.begin(), bench_infos_vec.end(), [](BenchmarkInfo& a, BenchmarkInfo& b) {
                return a.milli_seconds < b.milli_seconds;
            });

        // check if the best of this phase is better than previous winner
        auto& winner_of_this_phase = bench_infos_vec.front();
        if(winner_of_this_phase.milli_seconds < curr_best_msec)
        {
            winner_phase       = winner_of_this_phase.tuning_phase;
            winner_config_id   = winner_of_this_phase.SSN;
            winner_kernel_name = winner_of_this_phase.kernel_name;

            // update the best ms up to now
            curr_best_msec = winner_of_this_phase.milli_seconds;
        }
    }

    packet->winner_phases[curr_tuning_node_id]       = winner_phase;
    packet->winner_ids[curr_tuning_node_id]          = winner_config_id;
    packet->winner_kernel_names[curr_tuning_node_id] = winner_kernel_name;
}

void TuningBenchmarker::PropagateBestFactorsToNextPhase()
{
    std::vector<std::string> best_factors;

    // the infos are already sorted by gflops,
    // we need to return the factors in the order of their best one
    int   curr_tuning_node_id = packet->tuning_node_id;
    auto& bench_infos_vec     = benchmark_infos_of_node[curr_tuning_node_id];

    // clear previous data
    packet->target_factors[curr_tuning_node_id].clear();

    std::set<std::string> seen_factors;
    for(auto& info : bench_infos_vec)
    {
        if(seen_factors.count(info.factors_str) == 0)
        {
            seen_factors.insert(info.factors_str);
            best_factors.push_back(info.factors_str);
        }
    }

    // we will focus on the best 3 factors (at most) in the next phase tuning (permuting)
    size_t num_target_factors = 3;
    if(best_factors.size() > num_target_factors)
        best_factors.resize(num_target_factors);

    for(auto& factor : best_factors)
        packet->target_factors[curr_tuning_node_id].insert(factor);
}

// Export the winner solution
void TuningBenchmarker::ExportWinnerToSolutions()
{
    // Informations of Winner kernels
    std::vector<int>&         winners           = packet->winner_ids;
    std::vector<int>&         winners_phase     = packet->winner_phases;
    std::vector<bool>&        is_builtin_kernel = packet->is_builtin_kernel;
    std::vector<std::string>& kernelTokens      = packet->tuning_kernel_tokens;

    // root solution node for the tuning problem
    std::string rootToken    = packet->tuning_problem_name;
    std::string archName     = packet->tuning_arch_name;
    ProblemKey  rootKey      = ProblemKey(archName, rootToken);
    auto&       rootSolution = binding_solution_map->get_solution_node(rootKey);

    auto InsertWinnerKernelSolution = [&](size_t kernel_node_id) -> size_t {
        // get the default kernel token (without extra phase, bench ...etc)
        int         kernel_winner = winners[kernel_node_id];
        int         phase_idx     = winners_phase[kernel_node_id];
        bool        is_builtin    = is_builtin_kernel[kernel_node_id];
        std::string kernelToken   = kernelTokens[kernel_node_id];
        std::string srcToken      = kernelToken;

        // for a tuning external kernel, we fetch the winner/phase from the candidate and get the token
        // a built-in kernel has fixed (not elaborated one) token string
        if(is_builtin == false)
        {
            srcToken += std::string("_leafnode_") + std::to_string(kernel_node_id);
            srcToken += std::string("_phase_") + std::to_string(phase_idx);
        }

        // src = elaborated tuningToken (with extra phase, bench...etc),
        // dst = token without them.
        std::string dstToken = kernelToken;
        ProblemKey  srcKey(archName, srcToken);
        ProblemKey  dstKey(archName, dstToken);

        // get the winner of the node from current solution map (primary map)
        SolutionNode kernel_solution
            = binding_solution_map->get_solution_node(srcKey, kernel_winner);

        // and insert to temp_working_map (isRoot=false, check_dup=true, primary=false)
        return binding_solution_map->add_solution(dstKey, kernel_solution, false, true, false);
    };

    size_t current_adding_kernel_node = 0;

    // This recursion will add solution from bottom to up:
    // recursively add all the solution-node to from current map to tuning_map, and need to check ducplication.
    auto RecursivelyAddSolution
        = [&](ProblemKey& key, SolutionNode& solution, auto&& RecursivelyAddSolution) -> size_t {
        // if adding a leaf node (together with a kernel node)
        // then we add the kernel-solution from winners in tuning packet,
        // and get the child_option of that kernel, update this value to the leaf node child_option
        if(solution.sol_node_type == SOL_LEAF_NODE)
        {
            // insert kernel to tuning_map, return the position(index) in the vector
            size_t child_option = InsertWinnerKernelSolution(current_adding_kernel_node);
            ++current_adding_kernel_node;

            // update the child_option value for tuning_map
            solution.solution_childnodes[0].child_option = child_option;
        }
        // Adding an internal node, simply recursively adding its children
        else
        {
            for(auto& child : solution.solution_childnodes)
            {
                ProblemKey childKey(archName, child.child_token);

                // get the child solution object from current solution map (using existing child_option)
                auto& childSol
                    = binding_solution_map->get_solution_node(childKey, child.child_option);

                // since we are add solution to another solution map , so we have to update the child option
                child.child_option
                    = RecursivelyAddSolution(childKey, childSol, RecursivelyAddSolution);
            }
        }

        // add itself with new child_option value.
        // the add_solution(s) here is to build a final solution tree with proper option_id
        // so we need to check_dup.
        // insert to temp_working_map (check_dup=true, primary=false)
        bool isSolutionRoot = (key == rootKey);
        return binding_solution_map->add_solution(key, solution, isSolutionRoot, true, false);
    };
    // Call funtion !!
    RecursivelyAddSolution(rootKey, rootSolution, RecursivelyAddSolution);

    // Then output to a solution map file from the temp_working_map;
    // export to solution map dat file
    std::string filename         = archName + "_" + rootToken + ".dat";
    std::string workspace_folder = "";
    workspace_folder             = rocfft_getenv("TUNING_WORKSPACE");

    fs::path result_path(workspace_folder.c_str());
    result_path /= results_folder;
    result_path /= filename.c_str();

    // sort=true, primary_map=false
    binding_solution_map->write_solution_map_data(result_path, true, false);

    packet->output_solution_map_path = result_path.string();
}

void TuningBenchmarker::GetOutputSolutionMapPath(std::string& out_path)
{
    out_path = packet->output_solution_map_path;
}

// return bool: if csv is created/written or not
bool TuningBenchmarker::ExportCSV(bool append_data)
{
    int curr_tuning_node_id = packet->tuning_node_id;

    // skip the kernel-node with #-candidates = 0
    if(packet->total_candidates[curr_tuning_node_id] == 0)
        return false;

    auto&       bench_infos_vec  = benchmark_infos_of_node[curr_tuning_node_id];
    std::string filename         = bench_infos_vec[0].prob_token + ".csv";
    std::string workspace_folder = "";

    workspace_folder = rocfft_getenv("TUNING_WORKSPACE");

    fs::path csv_path(workspace_folder.c_str());
    csv_path /= csv_out_folder;
    csv_path /= filename.c_str();

    std::ofstream outfile;
    outfile.open(csv_path.c_str(),
                 (append_data) ? (std::ios::out | std::ios::app | std::ios::ate)
                               : (std::ios::out | std::ios::trunc));
    if(!outfile.is_open())
        return false;

    // if appending to file, add a few new lines
    if(append_data)
        outfile << std::endl << std::endl;

    outfile << "SSN, Problem, MS, GFLOPS, NumBlocks, WGS, TPT_0, TPT_1, TPB, LDS_Bytes, GRW_PT, "
               "Util_Rate, Factors, Occupancy, NumCUs, Granularity, BW_EFF, KernelName"
            << std::endl;

    for(auto& info : bench_infos_vec)
    {
        outfile << info.SSN << "," << info.prob_token << "," << info.milli_seconds << ","
                << info.gflops << "," << info.num_blocks << "," << info.workgroup_size << ","
                << info.threads_per_trans[0] << "," << info.threads_per_trans[1] << ","
                << info.trans_per_block << "," << info.LDS_bytes << "," << info.globalRW_per_thread
                << ","
                << "\"" << info.util_rate << "\""
                << ","
                << "\"" << info.factors_str << "\""
                << "," << info.occupancy << "," << info.numCUs << "," << info.granularity << ","
                << info.bw_eff << "," << info.kernel_name << std::endl;
    }

    outfile.close();

    return true;
}

bool TuningBenchmarker::MergingSolutionsMaps(const std::string& base_map_path,
                                             const std::string& new_map_path,
                                             const std::string& probKeyStr,
                                             const std::string& out_map_path)
{
    static const std::string sep = ":";

    try
    {
        // read the existing solutions to primary map, primary = true
        if(binding_solution_map->read_solution_map_data(base_map_path) == false)
            std::cout << "\t\tBase solution map file is not existing, will create a new one\n";

        std::string token = probKeyStr;
        size_t      pos   = token.find(sep);
        if(pos == std::string::npos)
            throw std::runtime_error(probKeyStr
                                     + " is an inccorect probKeyStr format. Please pass probKeyStr "
                                       "as \"arch:probToken\"");

        std::string arch = token.substr(0, pos);
        token.erase(0, pos + 1);
        std::vector<ProblemKey> merging_problems = {ProblemKey(arch, token)};

        // read mering-solutions from new file and merge to primary map
        if(binding_solution_map->merge_solutions_from_file(new_map_path, merging_problems))
            // output to the merged map, sort = true, output primary = true
            return binding_solution_map->write_solution_map_data(out_map_path);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    return false;
}