// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TilewrightLaunch over the kernels of real libraries: the plans plan()
// evaluates from its per-kernel records are those resolvePersistentSettings()
// gives, for every kernel, CU budget and schedule. Runs only when
// TILEWRIGHT_LAUNCH_RULE_LIBRARIES lists library files (.dat or .dat.zlib),
// separated by ':'.

#include <gtest/gtest.h>

#if defined(TENSILELITE_HAS_TILEWRIGHT)

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <msgpack.hpp>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TilewrightRanker.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/msgpack/Loading.hpp>
#include <origami/hardware.hpp>

using namespace TensileLite;

namespace
{
    using Launch = TilewrightLaunch<ContractionSolution>;
    using Master = MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>;

    std::shared_ptr<hip::HipAMDGPU> makeGfx950(size_t cuCount)
    {
        auto device              = std::make_shared<hip::HipAMDGPU>();
        device->processor        = AMDGPU::Processor::gfx950;
        device->computeUnitCount = static_cast<int>(cuCount);
        device->deviceName       = "tilewright-launch-library-test";
        device->analyticalHardware
            = std::make_shared<origami::hardware_t>(origami::hardware_t::architecture_t::gfx950,
                                                    cuCount,
                                                    163840,
                                                    262144,
                                                    8,
                                                    1.0,
                                                    1.0,
                                                    1.0,
                                                    4000000,
                                                    1.2,
                                                    1,
                                                    std::make_tuple(0.0, 0.008, 0.0));
        return device;
    }

    bool isXccCheck(msgpack::object const& object)
    {
        if(object.type != msgpack::type::MAP)
            return false;
        for(uint32_t i = 0; i < object.via.map.size; ++i)
        {
            auto const& entry = object.via.map.ptr[i];
            if(entry.key.type == msgpack::type::STR && entry.key.as<std::string>() == "type")
                return entry.val.type == msgpack::type::STR
                       && entry.val.as<std::string>() == "WorkgroupMappingXCCCheck";
        }
        return false;
    }

    // WorkgroupMappingXCCCheck asks the HIP runtime for the current device when
    // it is made, and only selection reads it.
    void packWithoutXccChecks(msgpack::packer<msgpack::sbuffer>& packer,
                              msgpack::object const&             object)
    {
        if(isXccCheck(object))
        {
            packer.pack_map(1);
            packer.pack(std::string("type"));
            packer.pack(std::string("TruePred"));
            return;
        }
        switch(object.type)
        {
        case msgpack::type::MAP:
            packer.pack_map(object.via.map.size);
            for(uint32_t i = 0; i < object.via.map.size; ++i)
            {
                packWithoutXccChecks(packer, object.via.map.ptr[i].key);
                packWithoutXccChecks(packer, object.via.map.ptr[i].val);
            }
            break;
        case msgpack::type::ARRAY:
            packer.pack_array(object.via.array.size);
            for(uint32_t i = 0; i < object.via.array.size; ++i)
                packWithoutXccChecks(packer, object.via.array.ptr[i]);
            break;
        default:
            packer.pack(object);
            break;
        }
    }

    std::vector<std::shared_ptr<ContractionSolution>> loadSolutions(std::string const& path)
    {
        msgpack::object_handle handle;
        if(!fileToMsgObject(path, handle))
            return {};
        msgpack::sbuffer                  buffer;
        msgpack::packer<msgpack::sbuffer> packer(&buffer);
        packWithoutXccChecks(packer, handle.get());

        std::random_device    seed;
        std::filesystem::path dir = std::filesystem::temp_directory_path()
                                    / ("tensilelite_tilewright_launch_" + std::to_string(seed()));
        std::filesystem::create_directories(dir);
        auto const copy = dir / "library.dat";
        {
            std::ofstream out(copy, std::ios::binary);
            out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        }
        auto const library = LoadLibraryFile<ContractionProblemGemm>(copy.string());
        std::filesystem::remove_all(dir);
        auto const master = std::dynamic_pointer_cast<Master>(library);
        if(!master)
            return {};
        std::vector<std::shared_ptr<ContractionSolution>> solutions;
        for(auto const& [index, solution] : master->solutions)
            solutions.push_back(solution);
        return solutions;
    }

    std::vector<std::string> split(std::string const& list)
    {
        std::vector<std::string> items;
        std::istringstream       in(list);
        for(std::string item; std::getline(in, item, ':');)
            if(!item.empty())
                items.push_back(item);
        return items;
    }

    char const* reductionName(tilewright::PartialReduction reduction)
    {
        switch(reduction)
        {
        case tilewright::PartialReduction::Tree:
            return "tree";
        case tilewright::PartialReduction::Parallel:
            return "parallel";
        case tilewright::PartialReduction::Atomic:
            return "atomic";
        default:
            return "none";
        }
    }
}

TEST(TilewrightLaunchLibraryTest, RuleMatchesResolvePersistentSettings)
{
    char const* listed = std::getenv("TILEWRIGHT_LAUNCH_RULE_LIBRARIES");
    if(listed == nullptr || listed[0] == '\0')
        GTEST_SKIP() << "set TILEWRIGHT_LAUNCH_RULE_LIBRARIES";

    std::vector<std::shared_ptr<ContractionSolution>> pool;
    for(auto const& path : split(listed))
    {
        auto const solutions = loadSolutions(path);
        ASSERT_FALSE(solutions.empty()) << path;
        pool.insert(pool.end(), solutions.begin(), solutions.end());
    }

    size_t const cuCount = 256;
    auto const   device  = makeGfx950(cuCount);
    Launch const launch("gfx950", pool, device);

    size_t ruled = 0;
    for(size_t i = 0; i < pool.size(); ++i)
        ruled += launch.followsRule(i, *device);
    std::cout << "[ INFO     ] " << pool.size() << " kernels, " << ruled
              << " planned from their records" << std::endl;
    ASSERT_GT(ruled, 0u);

    auto const& type    = pool.front()->problemType;
    auto const  toTw    = datatypeToTilewrightDatatype;
    auto const  problem = [&](size_t m, size_t n, size_t k, size_t batch) {
        return tilewright::Problem{
             .size        = {m, n, k},
             .batch       = batch,
             .a_transpose = type.transA ? tilewright::Transpose::T : tilewright::Transpose::N,
             .b_transpose = type.transB ? tilewright::Transpose::T : tilewright::Transpose::N,
             .a_dtype     = toTw(type.aType),
             .b_dtype     = toTw(type.bType),
             .c_dtype     = toTw(type.cType),
             .d_dtype     = toTw(type.dType),
             .mi_dtype    = type.f32XdlMathOp == rocisa::DataType::XFloat32
                                ? tilewright::DataType::XFloat32
                                : toTw(type.computeInputTypeA)};
    };

    std::vector<tilewright::Problem> problems{
        problem(1, 1, 1, 1),
        problem(16, 16, 1, 1),
        problem(256, 256, 1 << 22, 1),
        problem(1 << 16, 1 << 16, 64, 1),
        problem(1 << 15, 1 << 16, 1 << 17, 1),
        problem(4096, 4096, 512, 16),
        problem(8192, 128, 8192, 1),
        problem(128, 8192, 65536, 1),
        problem(256 * 31, 256 * 17, 4096, 1),
        problem(61440, 256, 4096, 1),
        problem(2048, 2048, 16384, 3),
        problem(7, 50000, 3000, 64),
    };
    std::mt19937_64 rng(0);
    auto const      logUniform = [&](double hi) {
        return static_cast<size_t>(
            std::exp(std::uniform_real_distribution<double>(0.0, std::log(hi))(rng)));
    };
    for(int i = 0; i < 40; ++i)
    {
        size_t const batch = std::uniform_int_distribution<int>(0, 9)(rng) == 0
                                 ? size_t(1) << std::uniform_int_distribution<int>(1, 6)(rng)
                                 : 1;
        problems.push_back(
            problem(logUniform(65536.0), logUniform(65536.0), logUniform(131072.0), batch));
    }

    size_t compared = 0;
    size_t differ   = 0;
    for(auto const& p : problems)
    {
        for(size_t const budget : {0, 32, 64, 96, 100, 128, 160, 192, 224, 240, 255})
        {
            for(auto const schedule : {tilewright::Schedule::Default,
                                       tilewright::Schedule::Dynamic,
                                       tilewright::Schedule::Auto})
            {
                tilewright::ExecutionContext const context{.cu_budget = budget,
                                                           .schedule  = schedule};
                for(size_t i = 0; i < pool.size(); ++i)
                {
                    tilewright::Config config{};
                    config.index        = i;
                    auto const got      = launch.plan(config, p, context);
                    auto const expected = launch.resolvedPlan(config, p, context);
                    ++compared;
                    if(got.dynamic_queue == expected.dynamic_queue
                       && got.grid_wgs == expected.grid_wgs
                       && got.partial_reduction == expected.partial_reduction)
                        continue;
                    if(++differ <= 20)
                        ADD_FAILURE()
                            << "solution " << pool[i]->index << " m=" << p.size.m
                            << " n=" << p.size.n << " k=" << p.size.k << " batch=" << p.batch
                            << " cu_budget=" << budget << " schedule=" << static_cast<int>(schedule)
                            << ": plan " << got.dynamic_queue << '/' << got.grid_wgs << '/'
                            << reductionName(got.partial_reduction) << ", resolved "
                            << expected.dynamic_queue << '/' << expected.grid_wgs << '/'
                            << reductionName(expected.partial_reduction);
                }
            }
        }
    }
    std::cout << "[ INFO     ] " << compared << " plans compared" << std::endl;
    EXPECT_EQ(differ, 0u);
}

#endif
