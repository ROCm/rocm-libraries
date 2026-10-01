// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "a2a_bench.hpp"
#include "flops.hpp"
#include "program_options.hpp"

#include <functional>

using namespace hipblaslt_bench;
using namespace roc; // For emulated program_options

int main(int argc, char* argv[])
try
{
    const hipblaslt_bench::LauncherEnv env = hipblaslt_bench::read_launcher_env();

    Arguments arg;
    arg.init();

    bool verify = false;
    bool timing = false;
    // 0 marks "not given on the command line".
    int a2a_world = 0;

    options_description desc("hipblaslt-bench-a2a command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m",
         value<int64_t>(&arg.M[0])->default_value(18432),
         "Feature extent (free0)")

        ("sizen,n",
         value<int64_t>(&arg.N[0])->default_value(2048),
         "Token extent (free1)")

        ("sizek,k",
         value<int64_t>(&arg.K[0])->default_value(8192),
         "Bound extent")

        ("a2a_extent",
         value<int64_t>(&arg.a2a_extent)->default_value(10240),
         "Features taking the all-to-all path")

        ("a2a_world",
         value<int>(&a2a_world)->default_value(0),
         "Checked against WORLD_SIZE, which wins")

        ("timing",
         value<bool>(&timing)->default_value(false),
         "Measure latency; otherwise only the configuration is reported")

        ("iters,i",
         value<int32_t>(&arg.iters)->default_value(10),
         "Enqueues per sample; also sizes the --verify pass")

        ("cold_iters,j",
         value<int32_t>(&arg.cold_iters)->default_value(2),
         "Cold iterations to run before entering the timing loop")

        ("adaptive",
         value<bool>(&arg.adaptive)->default_value(false),
         "Self-size the sample count; runs no --cold_iters warmup")

        ("verify,v",
         value<bool>(&verify)->default_value(false),
         "Check every launch of a pass run before the timed one")

        ("help,h", "produces this help message");
    // clang-format on

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if(vm.count("help"))
    {
        hipblaslt_cout << desc << "\n"
                       << "Rank identity comes from RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR"
                          " / MASTER_PORT. With none set the run is single-rank.\n";
        return 0;
    }

    arg.timing = timing ? 1 : 0;
    if(verify)
        arg.norm_check = arg.allclose_check = 1;

    if(a2a_world != 0 && a2a_world != int(env.world))
    {
        hipblaslt_cerr << "error: --a2a_world " << a2a_world << " disagrees with WORLD_SIZE "
                       << env.world << "\n";
        return 1;
    }
    arg.a2a_world = uint8_t(env.world);

    hipblaslt_bench::TcpRendezvous rendezvous(env, kRendezvousTimeoutSec);
    if(!join_group(env, rendezvous, HIPBLASLT_DEVICE_COMM_MAX_WORLD))
        return 0;

    const auto agreement = make_agreement(rendezvous, env.world);

    RankResources res;
    res.rendezvous = &rendezvous;
    if(!setup_rank(env, arg, res))
        return 1;

    hipblasLtMatmulHeuristicResult_t heur{};
    int                              algoCount  = 0;
    const hipblasStatus_t            algoStatus = select_algo(res, heur, algoCount);
    if(algoStatus != HIPBLAS_STATUS_SUCCESS)
    {
        hipblaslt_cerr << "error: hipblasLtMatmulAlgoGetHeuristic -> " << int(algoStatus) << "\n";
        return 1;
    }
    if(!agreement.agree(algoCount > 0, std::logical_and<>{}))
    {
        hipblaslt_cout << "skipped: no fused GEMM+A2A solution in the loaded library\n";
        return 0;
    }

    uint32_t                       launchCount = 0;
    hipblasStatus_t                lastStatus  = HIPBLAS_STATUS_SUCCESS;
    std::vector<hipblasLtBfloat16> hostGold, hostLanded;
    auto                           launch = make_launch(res, heur, launchCount, lastStatus);

    if(arg.timing)
    {
        hipblaslt_bench::TimingConfig cfg;
        cfg.adaptive      = arg.adaptive;
        cfg.iters         = arg.iters;
        cfg.use_gpu_timer = false;
        if(arg.adaptive)
        {
            cfg.warmup_time         = arg.warmup_time;
            cfg.sample_time         = arg.sample_time;
            cfg.measure_time        = arg.measure_time;
            cfg.max_measure_time    = arg.max_measure_time;
            cfg.min_iters           = arg.min_iters;
            cfg.max_iters           = arg.max_iters;
            cfg.noise_threshold     = arg.noise_threshold;
            cfg.stability_threshold = arg.stability_threshold;
            cfg.stability_window    = arg.stability_window;
            cfg.stability_interval  = arg.stability_interval;
        }

        bool verified = true;
        if(arg.norm_check || arg.allclose_check)
            for(int32_t i = 0; i < arg.iters; ++i)
            {
                launch(0);
                if(hipStreamSynchronize(res.stream) != hipSuccess
                   || !check_recv(env, arg, res, hostGold, hostLanded))
                    verified = false;
            }

        if(!agreement.agree(verified, std::logical_and<>{}))
        {
            if(lastStatus != HIPBLAS_STATUS_SUCCESS)
                hipblaslt_cerr << "error: matmul -> " << int(lastStatus) << "\n";
            else
                hipblaslt_cerr << "error: verification failed\n";
            return 1;
        }

        if(!arg.adaptive)
            for(int32_t i = 0; i < arg.cold_iters; ++i)
                launch(0);

        hipblaslt_bench::TimingResult result;
        hipblaslt_bench::run_measurement(
            launch, cfg, nullptr, nullptr, res.stream, result, {}, agreement);

        const bool ok = lastStatus == HIPBLAS_STATUS_SUCCESS;
        if(!agreement.agree(ok, std::logical_and<>{}))
        {
            if(ok)
                hipblaslt_cerr << "error: peer rank failed\n";
            else
                hipblaslt_cerr << "error: matmul -> " << int(lastStatus) << "\n";
            return 1;
        }

        const double slowest = agreement.agree(result.median_us, MaxOp{});
        if(env.rank == 0)
        {
            const double gflops = gemm_gflop_count<hipblasLtBfloat16>(arg.M[0], arg.N[0], arg.K[0]);

            hipblaslt_cout << "a2a_world,a2a_extent,M,N,K,hipblaslt-Gflops,us\n"
                           << unsigned(arg.a2a_world) << "," << arg.a2a_extent << "," << arg.M[0]
                           << "," << arg.N[0] << "," << arg.K[0] << ","
                           << hipblaslt_bench::rate_per_second(gflops, slowest, -1.0) << ","
                           << slowest << "\n";
        }
    }
    else
    {
        if(env.rank == 0)
            hipblaslt_cout << "a2a_world,a2a_extent,M,N,K\n"
                           << unsigned(arg.a2a_world) << "," << arg.a2a_extent << "," << arg.M[0]
                           << "," << arg.N[0] << "," << arg.K[0] << "\n";

        launch(0);
        bool ok = hipStreamSynchronize(res.stream) == hipSuccess
                  && lastStatus == HIPBLAS_STATUS_SUCCESS;
        if(ok && (arg.norm_check || arg.allclose_check)
           && !check_recv(env, arg, res, hostGold, hostLanded))
            ok = false;

        if(!agreement.agree(ok, std::logical_and<>{}))
        {
            if(ok)
                hipblaslt_cerr << "error: peer rank failed\n";
            else if(lastStatus != HIPBLAS_STATUS_SUCCESS)
                hipblaslt_cerr << "error: matmul -> " << int(lastStatus) << "\n";
            else
                hipblaslt_cerr << "error: verification failed\n";
            return 1;
        }
    }
    return 0;
}
catch(const std::exception& e)
{
    hipblaslt_cerr << "error: " << e.what() << "\n";
    return 1;
}
