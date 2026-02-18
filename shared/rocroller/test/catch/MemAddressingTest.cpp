/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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

#include <algorithm>
#include <catch2/catch_test_macros.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/CommandSolution_detail.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/TensorDescriptor.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <common/CommonGraphs.hpp>
#include <common/Scheduling.hpp>
#include <common/Utilities.hpp>
#include <common/mxDataGen.hpp>

#include "CustomAssertions.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"

namespace MemAddressingTest
{
    using namespace rocRoller;
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;
    using namespace rocRoller::KernelGraph::CoordinateGraph;

    TEST_CASE("LDS Address Modelling", "[mem-addressing]")
    {
        struct Variants
        {
            std::string                              name;
            std::function<void(std::vector<size_t>)> validate;
            decltype(GEMMProblem::padA)              padA = GEMMProblem{}.padA;
            decltype(GEMMProblem::padB)              padB = GEMMProblem{}.padB;

            static void NoPadding(std::vector<size_t> addresses)
            {
                INFO(addresses);
                // Load addresses from rocgdb (ds_read)
                std::vector<size_t> load_pattern
                    = {0,    256,  512,  768,  1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072,
                       3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400,
                       6656, 6912, 7168, 7424, 7680, 7936, 4,    260,  516,  772,  1028, 1284, 1540,
                       1796, 2052, 2308, 2564, 2820, 3076, 3332, 3588, 3844, 4100, 4356, 4612, 4868,
                       5124, 5380, 5636, 5892, 6148, 6404, 6660, 6916, 7172, 7428, 7684, 7940};
                // Store addresses from rocgdb (ds_write)
                std::vector<size_t> store_pattern
                    = {0,    16,   32,   48,   64,   80,   96,   112,  128,  144,  160,  176,  192,
                       208,  224,  240,  1024, 1040, 1056, 1072, 1088, 1104, 1120, 1136, 1152, 1168,
                       1184, 1200, 1216, 1232, 1248, 1264, 2048, 2064, 2080, 2096, 2112, 2128, 2144,
                       2160, 2176, 2192, 2208, 2224, 2240, 2256, 2272, 2288, 3072, 3088, 3104, 3120,
                       3136, 3152, 3168, 3184, 3200, 3216, 3232, 3248, 3264, 3280, 3296, 3312};
                CHECK((addresses == load_pattern || addresses == store_pattern));
            }

            static void YesPadding(std::vector<size_t> addresses)
            {
                INFO(addresses);
                // Load addresses from rocgdb (ds_read)
                std::vector<size_t> load_A_pattern
                    = {0,    256,  512,  768,  1024, 1280, 1536, 1792, 2052, 2308, 2564, 2820, 3076,
                       3332, 3588, 3844, 4104, 4360, 4616, 4872, 5128, 5384, 5640, 5896, 6156, 6412,
                       6668, 6924, 7180, 7436, 7692, 7948, 4,    260,  516,  772,  1028, 1284, 1540,
                       1796, 2056, 2312, 2568, 2824, 3080, 3336, 3592, 3848, 4108, 4364, 4620, 4876,
                       5132, 5388, 5644, 5900, 6160, 6416, 6672, 6928, 7184, 7440, 7696, 7952};
                std::vector<size_t> load_B_pattern
                    = {0,    256,  512,  768,  1028, 1284, 1540, 1796, 2056, 2312, 2568, 2824, 3084,
                       3340, 3596, 3852, 4112, 4368, 4624, 4880, 5140, 5396, 5652, 5908, 6168, 6424,
                       6680, 6936, 7196, 7452, 7708, 7964, 4,    260,  516,  772,  1032, 1288, 1544,
                       1800, 2060, 2316, 2572, 2828, 3088, 3344, 3600, 3856, 4116, 4372, 4628, 4884,
                       5144, 5400, 5656, 5912, 6172, 6428, 6684, 6940, 7200, 7456, 7712, 7968};
                // Store addresses from rocgdb (ds_write)
                std::vector<size_t> store_A_pattern
                    = {0,    16,   32,   48,   64,   80,   96,   112,  128,  144,  160,  176,  192,
                       208,  224,  240,  1024, 1040, 1056, 1072, 1088, 1104, 1120, 1136, 1152, 1168,
                       1184, 1200, 1216, 1232, 1248, 1264, 2052, 2068, 2084, 2100, 2116, 2132, 2148,
                       2164, 2180, 2196, 2212, 2228, 2244, 2260, 2276, 2292, 3076, 3092, 3108, 3124,
                       3140, 3156, 3172, 3188, 3204, 3220, 3236, 3252, 3268, 3284, 3300, 3316};
                std::vector<size_t> store_B_pattern
                    = {0,    16,   32,   48,   64,   80,   96,   112,  128,  144,  160,  176,  192,
                       208,  224,  240,  1028, 1044, 1060, 1076, 1092, 1108, 1124, 1140, 1156, 1172,
                       1188, 1204, 1220, 1236, 1252, 1268, 2056, 2072, 2088, 2104, 2120, 2136, 2152,
                       2168, 2184, 2200, 2216, 2232, 2248, 2264, 2280, 2296, 3084, 3100, 3116, 3132,
                       3148, 3164, 3180, 3196, 3212, 3228, 3244, 3260, 3276, 3292, 3308, 3324};
                // TODO: split between load and store
                CHECK((addresses == load_A_pattern || addresses == load_B_pattern
                       || addresses == store_A_pattern || addresses == store_B_pattern));
            }
        };

        const auto variants
            = GENERATE(Variants{"No padding", Variants::NoPadding},
                       Variants("Yes padding", Variants::YesPadding, {32 * 64, 4}, {16 * 64, 4}));

        SECTION(variants.name)
        {
            auto context = TestContext::ForTestDevice(
                {{.dsObserver = DSObserverType::WeightlessDSMemObserver}}, variants.name);
            auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

            example.setUseLDS(true, true, false);
            example.setTranspose("T", "N");
            example.setPad(variants.padA, variants.padB);

            auto command = example.getCommand();
            auto params  = example.getCommandParameters();

            CommandKernel commandKernel(command, context.KernelName());
            commandKernel.setContext(context.get());
            commandKernel.setCommandParameters(params);

            commandKernel.generateKernelGraph();
            auto graph = commandKernel.getKernelGraph();

            for(auto inst : kernelInstructions(context.get(), command, graph))
            {
                context.get()->schedule(inst);
                if(inst.getModelledAddresses().has_value())
                {
                    auto addresses = inst.getModelledAddresses().value();
                    variants.validate(addresses);
                }
            }

            auto [commandArgs, deviceA, deviceB, deviceC, deviceD]
                = example.getCommandArguments<float>();
            commandKernel.launchKernel(commandArgs.runtimeArguments());
        }
    }
}
