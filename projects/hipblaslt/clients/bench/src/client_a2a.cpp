// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "a2a_bench.hpp"
#include "flops.hpp"

#include <algorithm>

using namespace hipblaslt_bench;

int main(int argc, char* argv[])
try
{
    const hipblaslt_bench::LauncherEnv env = hipblaslt_bench::read_launcher_env();

    Arguments arg;
    arg.init();
    arg.M[0]       = 18432;
    arg.N[0]       = 2048;
    arg.K[0]       = 8192;
    arg.a2a_extent = 10240;

    // 0 marks "not given on the command line".
    arg.a2a_world = 0;

    std::string error;
    if(!parse_a2a_args(argc, argv, arg, error))
    {
        if(error != "help")
            hipblaslt_cerr << "error: " << error << "\n";
        print_usage(argv[0]);
        return error == "help" ? 0 : 1;
    }
    if(arg.a2a_world != 0 && arg.a2a_world != uint8_t(env.world))
    {
        hipblaslt_cerr << "error: --a2a_world " << unsigned(arg.a2a_world)
                       << " disagrees with WORLD_SIZE " << env.world << "\n";
        return 1;
    }
    arg.a2a_world = uint8_t(env.world);

    if(env.world > HIPBLASLT_DEVICE_COMM_MAX_WORLD)
    {
        hipblaslt_cout << "skipped: WORLD_SIZE " << env.world << " exceeds "
                       << HIPBLASLT_DEVICE_COMM_MAX_WORLD << "\n";
        return 0;
    }

    hipblaslt_bench::TcpRendezvous rendezvous(env, kRendezvousTimeoutSec);
    if(!rendezvous.same_host_group())
    {
        hipblaslt_cout << "skipped: ranks span hosts\n";
        return 0;
    }

    const uint8_t reachable = peers_reachable(env, arg) ? 1 : 0;

    std::vector<uint8_t> allReachable(env.world);
    if(rendezvous.allgather(&reachable, allReachable.data(), sizeof(reachable))
       != HIPBLAS_STATUS_SUCCESS)
    {
        hipblaslt_cerr << "error: allgather for peer pre-check failed\n";
        return 1;
    }
    bool groupReachable = true;
    for(uint32_t j = 0; j < env.world; ++j)
        groupReachable = groupReachable && allReachable[j] != 0;
    if(!groupReachable)
    {
        hipblaslt_cout << "skipped: peer access unavailable on at least one rank\n";
        return 0;
    }

    RankResources res;
    res.rendezvous = &rendezvous;
    if(!setup_rank(env, arg, res))
        return 1;

    hipblasLtMatmulHeuristicResult_t heur{};
    const uint8_t                    found = select_algo(res, heur) ? 1 : 0;

    std::vector<uint8_t> allFound(env.world);
    if(rendezvous.allgather(&found, allFound.data(), sizeof(found)) != HIPBLAS_STATUS_SUCCESS)
    {
        hipblaslt_cerr << "error: allgather for algo selection failed\n";
        return 1;
    }
    bool groupFound = true;
    for(uint32_t j = 0; j < env.world; ++j)
        groupFound = groupFound && allFound[j] != 0;
    if(!groupFound)
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

        uint8_t verified = 1;
        if(arg.norm_check || arg.allclose_check)
            for(int32_t i = 0; i < arg.iters; ++i)
            {
                launch(0);
                if(hipStreamSynchronize(res.stream) != hipSuccess
                   || !check_recv(env, arg, res, hostGold, hostLanded))
                    verified = 0;
            }

        std::vector<uint8_t> allVerified(env.world);
        if(rendezvous.allgather(&verified, allVerified.data(), sizeof(verified))
           != HIPBLAS_STATUS_SUCCESS)
        {
            hipblaslt_cerr << "error: allgather for verification failed\n";
            return 1;
        }
        bool groupVerified = true;
        for(uint32_t j = 0; j < env.world; ++j)
            groupVerified = groupVerified && allVerified[j] != 0;
        if(!groupVerified)
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
        const auto                    agreement = make_agreement(rendezvous, env.world);
        hipblaslt_bench::run_measurement(
            launch, cfg, nullptr, nullptr, res.stream, result, {}, agreement);

        struct LatencyContribution
        {
            uint8_t ok;
            double  median_us;
        };
        LatencyContribution mine{lastStatus == HIPBLAS_STATUS_SUCCESS ? uint8_t(1) : uint8_t(0),
                                 result.median_us};
        std::vector<LatencyContribution> perRank(env.world);
        if(rendezvous.allgather(&mine, perRank.data(), sizeof(mine)) != HIPBLAS_STATUS_SUCCESS)
            return 1;

        bool groupOk = true;
        for(uint32_t j = 0; j < env.world; ++j)
            groupOk = groupOk && perRank[j].ok != 0;
        if(!groupOk)
        {
            if(mine.ok != 0)
                hipblaslt_cerr << "error: peer rank failed\n";
            else
                hipblaslt_cerr << "error: matmul -> " << int(lastStatus) << "\n";
            return 1;
        }

        if(env.rank == 0)
        {
            double slowest = perRank[0].median_us;
            for(uint32_t j = 1; j < env.world; ++j)
                slowest = std::max(slowest, perRank[j].median_us);
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
        uint8_t ok = (hipStreamSynchronize(res.stream) == hipSuccess
                      && lastStatus == HIPBLAS_STATUS_SUCCESS)
                         ? 1
                         : 0;
        if(ok != 0 && (arg.norm_check || arg.allclose_check)
           && !check_recv(env, arg, res, hostGold, hostLanded))
            ok = 0;

        std::vector<uint8_t> allOk(env.world);
        if(rendezvous.allgather(&ok, allOk.data(), sizeof(ok)) != HIPBLAS_STATUS_SUCCESS)
            return 1;

        bool groupOk = true;
        for(uint32_t j = 0; j < env.world; ++j)
            groupOk = groupOk && allOk[j] != 0;
        if(!groupOk)
        {
            if(ok != 0)
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
