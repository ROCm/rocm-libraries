/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <origami/simulator/tensilelite/formocast_simulator.hpp>
#include <origami/hardware.hpp>
#include <queue>

using Catch::Approx;
using namespace origami;

// Helper function to create a ProblemInfo
Formocast::ProblemInfo make_problem_info(double M, double N, double K,
                                         uint32_t bpeA = 2, uint32_t bpeB = 2,
                                         uint32_t bpeD = 2, uint32_t bpeCompute = 2,
                                         bool transA = true, bool transB = false,
                                         double numBatches = 1,
                                         data_type_t dataType = data_type_t::BFloat16)
{
    Formocast::ProblemInfo problem;
    problem.M = M;
    problem.N = N;
    problem.K = K;
    problem.NumBatches = numBatches;
    problem.bpeA = bpeA;
    problem.bpeB = bpeB;
    problem.bpeD = bpeD;
    problem.bpeCompute = bpeCompute;
    problem.transA = transA;
    problem.transB = transB;
    problem.swizzleTensorA = false;
    problem.swizzleTensorB = false;
    problem.dataType = dataType;
    return problem;
}

// Helper function to create a SizeMapping
Formocast::SizeMapping make_size_mapping(int MT0 = 128, int MT1 = 128, int depthU = 32,
                                         int miM = 32, int miN = 32, int miK = 8,
                                         int waveNum = 4, int globalSplitU = 1)
{
    Formocast::SizeMapping mapping;
    mapping.macroTile[0] = MT0;
    mapping.macroTile[1] = MT1;
    mapping.macroTile[2] = 1;
    mapping.matrixInstruction[0] = miM;
    mapping.matrixInstruction[1] = miN;
    mapping.matrixInstruction[2] = miK;
    mapping.matrixInstruction[3] = 1;
    mapping.depthU = depthU;
    mapping.waveNum = waveNum;
    mapping.globalSplitU = globalSplitU;
    mapping.workGroupMapping = 8;
    mapping.CUOccupancy = 2;
    mapping.PrefetchGlobalRead = 2;
    mapping.MathClocksUnrolledLoop = 2048;
    mapping.grvwA = 4;
    mapping.grvwB = 4;
    mapping.gwvwD = 4;
    mapping.VectorWidthA = 4;
    mapping.VectorWidthB = 4;
    mapping.LocalSplitU = 1;
    mapping.DirectToVgprA = false;
    mapping.DirectToVgprB = false;
    mapping.DirectToLdsA = false;
    mapping.DirectToLdsB = false;
    mapping.NumLoadsCoalescedA = 1;
    mapping.NumLoadsCoalescedB = 1;
    mapping.waveGroup[0] = 2;
    mapping.waveGroup[1] = 2;
    mapping.globalAccumulation = 0;
    mapping.globalSplitUCoalesced = false;
    mapping.globalSplitUWorkGroupMappingRoundRobin = false;
    return mapping;
}

TEST_CASE("Formocast: Hardware constants retrieval", "[formocast]") {
    Formocast simulator;
    
    SECTION("gfx950 hardware constants") {
        auto hw = simulator.getHardwareConstants(hardware_t::architecture_t::gfx950);
        
        REQUIRE(hw.architecture == hardware_t::architecture_t::gfx950);
        REQUIRE(hw.NumCUs > 0);
        REQUIRE(hw.wavefrontSize > 0);
        REQUIRE(hw.L1CacheCapacity > 0);
        REQUIRE(hw.L2CacheCapacity > 0);
        REQUIRE(hw.L3CacheCapacity > 0);
        REQUIRE(hw.math_frequency > 0);
        REQUIRE(hw.mem_frequency > 0);
        REQUIRE(hw.hbmBandWidth > 0);
    }
    
    SECTION("gfx942 hardware constants") {
        auto hw = simulator.getHardwareConstants(hardware_t::architecture_t::gfx942);
        
        REQUIRE(hw.architecture == hardware_t::architecture_t::gfx942);
        REQUIRE(hw.NumCUs > 0);
        REQUIRE(hw.wavefrontSize > 0);
        REQUIRE(hw.L1CacheCapacity > 0);
        REQUIRE(hw.L2CacheCapacity > 0);
        REQUIRE(hw.L3CacheCapacity > 0);
        REQUIRE(hw.math_frequency > 0);
        REQUIRE(hw.mem_frequency > 0);
        REQUIRE(hw.hbmBandWidth > 0);
    }

    SECTION("gfx1201 hardware constants") {
        auto hw = simulator.getHardwareConstants(hardware_t::architecture_t::gfx1201);
        
        REQUIRE(hw.architecture == hardware_t::architecture_t::gfx1201);
        REQUIRE(hw.NumCUs > 0);
        REQUIRE(hw.wavefrontSize > 0);
        REQUIRE(hw.L1CacheCapacity > 0);
    }

    SECTION("Unsupported architecture throws exception") {
        REQUIRE_THROWS_AS(
            simulator.getHardwareConstants(static_cast<hardware_t::architecture_t>(9999)),
            std::runtime_error
        );
    }
}

TEST_CASE("Formocast: Basic problem and solution setup", "[formocast]") {
    Formocast simulator;
    
    SECTION("Set valid problem") {
        auto problem = make_problem_info(1024, 1024, 1024);
        REQUIRE_NOTHROW(simulator.setProblem(problem));
        REQUIRE(simulator.problem.M == 1024);
        REQUIRE(simulator.problem.N == 1024);
        REQUIRE(simulator.problem.K == 1024);
    }
    
    SECTION("Set problem with zero dimension throws exception") {
        auto problem = make_problem_info(0, 1024, 1024);
        REQUIRE_THROWS_AS(simulator.setProblem(problem), std::runtime_error);
        
        problem = make_problem_info(1024, 0, 1024);
        REQUIRE_THROWS_AS(simulator.setProblem(problem), std::runtime_error);
        
        problem = make_problem_info(1024, 1024, 0);
        REQUIRE_THROWS_AS(simulator.setProblem(problem), std::runtime_error);
    }
    
    SECTION("Set solution mapping") {
        auto mapping = make_size_mapping();
        REQUIRE_NOTHROW(simulator.setSolution(mapping));
        REQUIRE(simulator.sizeMapping.macroTile[0] == 128);
        REQUIRE(simulator.sizeMapping.macroTile[1] == 128);
        REQUIRE(simulator.sizeMapping.depthU == 32);
    }
    
    SECTION("Set hardware architecture") {
        REQUIRE_NOTHROW(simulator.setHardware(hardware_t::architecture_t::gfx950));
        REQUIRE(simulator.hw_consts.architecture == hardware_t::architecture_t::gfx950);
    }
}

TEST_CASE("Formocast: Performance prediction", "[formocast]") {
    Formocast simulator;
    
    SECTION("Basic performance prediction for square problem") {
        auto problem = make_problem_info(1024, 1024, 1024);
        auto mapping = make_size_mapping(128, 128, 32);
        
        simulator.setProblem(problem);
        simulator.setSolution(mapping);
        simulator.setHardware(hardware_t::architecture_t::gfx950);
        
        auto perf = simulator.predictedPerformance();
        
        REQUIRE(perf.microSeconds > 0);
        REQUIRE(perf.microSeconds < 1000000); // Reasonable upper bound
        REQUIRE(perf.hitRate >= 0);
        REQUIRE(perf.hitRate <= 100);
    }
    
    SECTION("Performance prediction for rectangular problem") {
        auto problem = make_problem_info(2048, 512, 1024);
        auto mapping = make_size_mapping(256, 64, 32);
        
        simulator.setProblem(problem);
        simulator.setSolution(mapping);
        simulator.setHardware(hardware_t::architecture_t::gfx942);
        
        auto perf = simulator.predictedPerformance();
        
        REQUIRE(perf.microSeconds > 0);
        REQUIRE(perf.hitRate >= 0);
        REQUIRE(perf.hitRate <= 100);
    }
    
    SECTION("Performance prediction with batched problem") {
        auto problem = make_problem_info(512, 512, 512, 2, 2, 2, 2, true, false, 4);
        auto mapping = make_size_mapping(128, 128, 32);
        
        simulator.setProblem(problem);
        simulator.setSolution(mapping);
        simulator.setHardware(hardware_t::architecture_t::gfx950);
        
        auto perf = simulator.predictedPerformance();
        
        REQUIRE(perf.microSeconds > 0);
    }
    
    SECTION("Performance prediction with GlobalSplitU") {
        auto problem = make_problem_info(1024, 1024, 2048);
        auto mapping = make_size_mapping(128, 128, 32, 32, 32, 8, 4, 2);
        
        simulator.setProblem(problem);
        simulator.setSolution(mapping);
        simulator.setHardware(hardware_t::architecture_t::gfx950);
        
        auto perf = simulator.predictedPerformance();
        
        REQUIRE(perf.microSeconds > 0);
    }
}

TEST_CASE("Formocast: Cache hit rate computation", "[formocast]") {
    Formocast simulator;
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    auto hw = simulator.hw_consts;
    
    SECTION("L1 cache hit rate computation") {
        auto l1_hit = simulator.computeL1CacheHitRate(
            hw, 128, 128, 2, 2, 0, 0, 4, 4,
            false, false, false, false,
            4, 4, true, false,
            1024, 1024, 1, 1,
            256, 2, 2
        );
        
        REQUIRE(l1_hit.tile0HitRate >= 0.0);
        REQUIRE(l1_hit.tile0HitRate <= 1.0);
        REQUIRE(l1_hit.tile1HitRate >= 0.0);
        REQUIRE(l1_hit.tile1HitRate <= 1.0);
    }
    
    SECTION("L2 cache hit rate computation") {
        auto problem = make_problem_info(1024, 1024, 1024);
        auto mapping = make_size_mapping();
        
        simulator.setProblem(problem);
        simulator.setSolution(mapping);
        
        auto l2_hit = simulator.computeL2CacheHitRate(
            1024, 1024, 1024, hw, 1, 8, 1, 2, 2, 0, 0, false
        );
        
        REQUIRE(l2_hit.totalHitRate >= 0.0);
        REQUIRE(l2_hit.totalHitRate <= 1.0);
        REQUIRE(l2_hit.tile0HitRate >= 0.0);
        REQUIRE(l2_hit.tile0HitRate <= 1.0);
        REQUIRE(l2_hit.tile1HitRate >= 0.0);
        REQUIRE(l2_hit.tile1HitRate <= 1.0);
    }
    
    SECTION("L3 cache hit rate computation") {
        auto l3_hit = simulator.computeL3CacheHitRate(
            1024, 1024, 1024, hw, 2, 2, 0, 0, 8, 8, 8, 8
        );
        
        REQUIRE(l3_hit.totalHitRate >= 0.0);
        REQUIRE(l3_hit.totalHitRate <= 1.0);
        REQUIRE(l3_hit.tile0HitRate >= 0.0);
        REQUIRE(l3_hit.tile0HitRate <= 1.0);
        REQUIRE(l3_hit.tile1HitRate >= 0.0);
        REQUIRE(l3_hit.tile1HitRate <= 1.0);
    }
}

TEST_CASE("Formocast: Store performance calculation", "[formocast]") {
    Formocast simulator;
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    auto hw = simulator.hw_consts;
    double store, store_edge;
    
    SECTION("Calculate store performance for square problem") {
        simulator.calculateStorePerformance(
            1024, 1024, 1, 128, 128, 4, 2, hw, 304, 38, store, store_edge
        );
        
        REQUIRE(store > 0);
        REQUIRE(store_edge >= 0);
    }
    
    SECTION("Calculate store performance with different GWVWD") {
        double store1, store_edge1, store2, store_edge2;
        
        simulator.calculateStorePerformance(
            1024, 1024, 1, 128, 128, 1, 2, hw, 304, 38, store1, store_edge1
        );
        
        simulator.calculateStorePerformance(
            1024, 1024, 1, 128, 128, 4, 2, hw, 304, 38, store2, store_edge2
        );
        
        // GWVWD=1 should have higher cost (multiplied by 2)
        REQUIRE(store1 > store2);
    }
}

TEST_CASE("Formocast: Tie-breaker comparison", "[formocast]") {
    Formocast simulator;
    auto problem = make_problem_info(1024, 1024, 1024);
    simulator.setProblem(problem);
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    SECTION("Standalone tie-breaker function for skinny N") {
        // Skinny N case: N <= 32 && M >= 1024 && K >= 1024
        bool result = compareConfigTieBreaker(
            1024, 32, 1024, 1,  // problem dimensions
            128, 32, 64, 4,     // config A
            64, 32, 64, 4       // config B
        );
        
        // Config A has larger skinny ratio when mt1 == N
        REQUIRE(result == true);
    }
    
    SECTION("Standalone tie-breaker function for skinny M") {
        // Skinny M case: M <= 32 && N >= 1024 && K >= 1024
        bool result = compareConfigTieBreaker(
            32, 1024, 1024, 1,  // problem dimensions
            32, 128, 64, 4,     // config A
            32, 64, 64, 4       // config B
        );
        
        // Config A has larger skinny ratio when mt0 == M
        REQUIRE(result == true);
    }
    
    SECTION("Standalone tie-breaker function for wide & short K") {
        // Wide & short K case: batch == 1 && K <= 512 && M >= 1024 && N >= 1024
        bool result = compareConfigTieBreaker(
            1024, 1024, 512, 1, // problem dimensions
            128, 128, 32, 8,    // config A (svw=8)
            128, 128, 32, 4     // config B (svw=4)
        );
        
        // Config A has larger store vector width
        REQUIRE(result == true);
    }
    
    SECTION("Tie-breaker returns false for equal configs") {
        bool result = compareConfigTieBreaker(
            1024, 1024, 1024, 1,
            128, 128, 32, 4,
            128, 128, 32, 4
        );
        
        REQUIRE(result == false);
    }
}

TEST_CASE("Formocast: GSU overhead calculation", "[formocast]") {
    Formocast simulator;
    auto problem = make_problem_info(1024, 1024, 1024);
    auto mapping = make_size_mapping();
    
    simulator.setProblem(problem);
    simulator.setSolution(mapping);
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    auto hw = simulator.hw_consts;
    
    SECTION("GSU overhead with MultipleBuffer method") {
        double gsu_overhead = simulator.calculateGSUOverhead(
            1024, 1024, 1024, 1, 2, 2, // gsuMethod=2 (MultipleBuffer)
            problem, hw, 304, 38, 128, 128, 32, 1.0, 1.0
        );
        
        REQUIRE(gsu_overhead >= 0);
    }
    
    SECTION("GSU overhead with MBSK method") {
        double gsu_overhead = simulator.calculateGSUOverhead(
            1024, 1024, 1024, 1, 2, 3, // gsuMethod=3 (MBSK)
            problem, hw, 304, 38, 128, 128, 32, 1.0, 1.0
        );
        
        REQUIRE(gsu_overhead >= 0);
    }
    
    SECTION("No GSU overhead when GlobalSplitU=1") {
        double gsu_overhead = simulator.calculateGSUOverhead(
            1024, 1024, 1024, 1, 1, 2, // GlobalSplitU=1
            problem, hw, 304, 38, 128, 128, 32, 1.0, 1.0
        );
        
        REQUIRE(gsu_overhead == 0);
    }
}

TEST_CASE("Formocast: LSU overhead calculation", "[formocast]") {
    Formocast simulator;
    auto problem = make_problem_info(1024, 1024, 1024);
    
    simulator.setProblem(problem);
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    auto hw = simulator.hw_consts;
    
    SECTION("LSU overhead calculation") {
        double lsu_overhead = simulator.calculateLSUOverhead(
            128, 128, 2, 4, 256, problem, hw
        );
        
        REQUIRE(lsu_overhead >= 0);
    }
    
    SECTION("LSU overhead increases with larger LSU value") {
        double lsu_overhead1 = simulator.calculateLSUOverhead(
            128, 128, 2, 4, 256, problem, hw
        );
        
        double lsu_overhead2 = simulator.calculateLSUOverhead(
            128, 128, 4, 4, 256, problem, hw
        );
        
        REQUIRE(lsu_overhead2 > lsu_overhead1);
    }
}

TEST_CASE("Formocast: Intermediate metrics calculation", "[formocast]") {
    Formocast simulator;
    
    auto problem = make_problem_info(1024, 1024, 1024);
    auto mapping = make_size_mapping();
    
    simulator.setProblem(problem);
    simulator.setSolution(mapping);
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    SECTION("Calculate intermediate metrics") {
        auto metrics = simulator.calculateIntermediateMetrics();
        
        REQUIRE(metrics.compute_cycles > 0);
        REQUIRE(metrics.prefetch_cost >= 0);
        REQUIRE(metrics.startup_cost >= 0);
        REQUIRE(metrics.output_write_cost >= 0);
        REQUIRE(metrics.output_write_cost_edge >= 0);
        REQUIRE(metrics.split_accumulation_overhead >= 0);
        REQUIRE(metrics.local_split_overhead >= 0);
        REQUIRE(metrics.cache_hits.A_L1_hit >= 0.0);
        REQUIRE(metrics.cache_hits.A_L1_hit <= 1.0);
    }
    
    SECTION("Calculate final performance from intermediate metrics") {
        auto metrics = simulator.calculateIntermediateMetrics();
        auto perf = simulator.calculateFinalPerformance(metrics);
        
        REQUIRE(perf.microSeconds > 0);
        REQUIRE(perf.hitRate >= 0);
        REQUIRE(perf.hitRate <= 100);
    }
    
    SECTION("Consistency between direct and two-stage prediction") {
        auto perf_direct = simulator.predictedPerformance();
        
        auto metrics = simulator.calculateIntermediateMetrics();
        auto perf_twostage = simulator.calculateFinalPerformance(metrics);
        
        // Both methods should produce the same result
        REQUIRE(perf_direct.microSeconds == Approx(perf_twostage.microSeconds).epsilon(0.01));
        REQUIRE(perf_direct.hitRate == Approx(perf_twostage.hitRate).epsilon(0.01));
    }
}

TEST_CASE("Formocast: FIFO queue operations", "[formocast]") {
    Formocast simulator;
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    SECTION("Check global read FIFO full") {
        std::queue<int> fifo;
        int result = simulator.checkGlobalReadFIFOFull(100, fifo, 8, 4, false);
        REQUIRE(result >= 0);
    }
    
    SECTION("Check local read FIFO full") {
        std::queue<int> fifo;
        int result = simulator.checkLocalReadFIFOFull(100, fifo, 8, 4, false, 1.0);
        REQUIRE(result >= 0);
    }
    
    SECTION("Check local read finished") {
        std::queue<int> fifo;
        fifo.push(50);
        fifo.push(60);
        fifo.push(70);
        
        int result = simulator.checkLocalReadFinished(100, fifo, 2);
        REQUIRE(result >= 0);
    }
    
    SECTION("Push local read for gfx950") {
        std::queue<int> fifo;
        simulator.pushLocalRead(100, fifo, 8, true);
        
        REQUIRE(fifo.size() > 0);
    }
    
    SECTION("Push local read write with bank conflict") {
        std::queue<int> fifo;
        simulator.pushLocalReadWrite(100, fifo, 8, 1.5);
        
        REQUIRE(fifo.size() > 0);
        REQUIRE(fifo.front() > 100); // Should have added latency
    }
}

TEST_CASE("Formocast: Bank conflict analysis", "[formocast]") {
    Formocast simulator;
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    SECTION("Analyze bank conflicts from VGPR state") {
        // Create mock VGPR state for 64 threads
        std::vector<std::unordered_map<std::string, int64_t>> vgprState(64);
        
        // Initialize with some test addresses
        for (int i = 0; i < 64; i++) {
            vgprState[i]["vgprLocalReadAddrA"] = i * 4;
            vgprState[i]["vgprLocalReadAddrB"] = i * 8;
        }
        
        auto result = simulator.analyzeBankConflictsFromVGPR(
            vgprState, "vgprLocalReadAddrA", "vgprLocalReadAddrB", 4, 8
        );
        
        REQUIRE(result.ratioA >= 1.0);
        REQUIRE(result.ratioB >= 1.0);
    }
}

TEST_CASE("Formocast: Memory access costs calculation", "[formocast]") {
    Formocast simulator;
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    auto hw = simulator.hw_consts;
    
    SECTION("Calculate memory access costs") {
        Formocast::CacheHitRates hr;
        hr.A_L1_hit = 0.9;
        hr.B_L1_hit = 0.9;
        hr.A_L2_hit = 0.7;
        hr.B_L2_hit = 0.7;
        hr.A_L3_hit = 0.5;
        hr.B_L3_hit = 0.5;
        hr.totalL2HitRate = 0.8;
        hr.totalL3HitRate = 0.6;
        
        auto mem_costs = simulator.calculateMemoryAccessCosts(
            128, 128, hw, hr,
            1000.0, 800.0, 600.0, // Bandwidths
            false, false,
            100.0, 100.0,  // L1 requests
            50.0, 25.0, 10.0,  // A L2/L3/HBM
            50.0, 25.0, 10.0   // B L2/L3/HBM
        );
        
        REQUIRE(mem_costs.mem_overall > 0);
        REQUIRE(mem_costs.mem_l1 >= 0);
        REQUIRE(mem_costs.mem_l2 >= 0);
        REQUIRE(mem_costs.mem_l3 >= 0);
        REQUIRE(mem_costs.mem_hbm >= 0);
        REQUIRE(mem_costs.l1_hit >= 0);
        REQUIRE(mem_costs.l1_hit <= 1.0);
    }
}

TEST_CASE("Formocast: Loop overall calculation", "[formocast]") {
    Formocast simulator;
    
    SECTION("Loop overall with prefetch") {
        Formocast::MemoryAccessCosts mem;
        mem.mem_overall = 10.0;
        
        double loop = simulator.getLoopOverall(mem, 8.0, 10, 2.0);
        
        REQUIRE(loop > 0);
    }
    
    SECTION("Loop overall without prefetch") {
        Formocast::MemoryAccessCosts mem;
        mem.mem_overall = 10.0;
        
        double loop = simulator.getLoopOverall(mem, 8.0, 10, 1.0);
        
        REQUIRE(loop > 0);
    }
}

TEST_CASE("Formocast: Occupancy resolution", "[formocast]") {
    Formocast simulator;
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    auto hw = simulator.hw_consts;
    
    SECTION("Resolve occupancy with multiple tiles") {
        double perf = simulator.resolveOccupancy(hw, 100.0, 10.0, 50.0, 20.0, 2, 2);
        REQUIRE(perf > 0);
    }
    
    SECTION("Resolve occupancy with single tile") {
        double perf = simulator.resolveOccupancy(hw, 100.0, 10.0, 50.0, 20.0, 1, 2);
        REQUIRE(perf > 0);
    }
}

TEST_CASE("Formocast: Tie-breaker info", "[formocast]") {
    Formocast simulator;
    
    auto problem = make_problem_info(1024, 1024, 1024);
    auto mapping = make_size_mapping();
    
    simulator.setProblem(problem);
    simulator.setSolution(mapping);
    simulator.setHardware(hardware_t::architecture_t::gfx950);
    
    // Run prediction to populate perfInfo
    simulator.predictedPerformance();
    
    SECTION("Get tie-breaker info") {
        auto tbInfo = simulator.getTieBreakerInfo();
        
        REQUIRE(tbInfo.mt0 > 0);
        REQUIRE(tbInfo.mt1 > 0);
        REQUIRE(tbInfo.du > 0);
        REQUIRE(tbInfo.perf > 0);
    }
    
    SECTION("Get min tie-breaker info") {
        auto minTbInfo = simulator.getMinTieBreakerInfo();
        
        REQUIRE(minTbInfo.mt0 > 0);
        REQUIRE(minTbInfo.mt1 > 0);
        REQUIRE(minTbInfo.du > 0);
    }
    
    SECTION("Compare tie-breaker configs") {
        Formocast::MinTieBreakerInfo config1;
        config1.mt0 = 128;
        config1.mt1 = 128;
        config1.du = 32;
        config1.svw = 4;
        
        Formocast::MinTieBreakerInfo config2;
        config2.mt0 = 64;
        config2.mt1 = 64;
        config2.du = 32;
        config2.svw = 4;
        
        // Equal configs
        REQUIRE(simulator.isBetter(config1, config1) == false);
    }
}

TEST_CASE("Formocast: Edge cases and error handling", "[formocast]") {
    Formocast simulator;
    
    SECTION("Very small problem size") {
        auto problem = make_problem_info(32, 32, 32);
        auto mapping = make_size_mapping(128, 128, 32);
        
        simulator.setProblem(problem);
        simulator.setSolution(mapping);
        simulator.setHardware(hardware_t::architecture_t::gfx950);
        
        auto perf = simulator.predictedPerformance();
        
        // Should return a large penalty for inefficient configuration
        REQUIRE(perf.microSeconds > 0);
    }
    
    SECTION("Very large problem size") {
        auto problem = make_problem_info(8192, 8192, 8192);
        auto mapping = make_size_mapping(256, 256, 64);
        
        simulator.setProblem(problem);
        simulator.setSolution(mapping);
        simulator.setHardware(hardware_t::architecture_t::gfx950);
        
        auto perf = simulator.predictedPerformance();
        
        REQUIRE(perf.microSeconds > 0);
    }
    
    SECTION("Problem with GlobalSplitU=0 returns penalty") {
        auto problem = make_problem_info(1024, 1024, 1024);
        auto mapping = make_size_mapping(128, 128, 32, 32, 32, 8, 4, 0);
        
        simulator.setProblem(problem);
        simulator.setSolution(mapping);
        simulator.setHardware(hardware_t::architecture_t::gfx950);
        
        auto perf = simulator.predictedPerformance();
        
        // Should return a very large penalty
        REQUIRE(perf.microSeconds > 1000000);
    }
}

