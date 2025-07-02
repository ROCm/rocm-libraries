/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Transforms/MemoryTracer.hpp>

#include <common/CommonGraphs.hpp>

#include "TestContext.hpp"

namespace MemoryTracerTest
{
    TEST_CASE("LDS bank conflicts", "[kernel-graph]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph::ControlGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        auto context = TestContext::ForDefaultTarget();
        auto params  = std::make_shared<CommandParameters>();
        params->setManualKernelDimension(2);
        params->setManualWorkgroupSize({256, 1, 1});
        context.get()->kernel()->setWorkgroupSize({256, 1, 1});

        auto graph = KernelGraph::KernelGraph();

        auto wgTile = graph.coordinates.addElement(
            MacroTile({256, 128}, LayoutType::MATRIX_A, {16, 16}, MemoryType::VGPR));
        auto lds      = graph.coordinates.addElement(LDS());
        auto waveTile = graph.coordinates.addElement(
            MacroTile({256, 128}, LayoutType::MATRIX_A, {16, 16, 4, 1}));

        auto kernel   = graph.control.addElement(Kernel());
        auto storeLDS = graph.control.addElement(StoreLDSTile());
        auto loadLDS  = graph.control.addElement(LoadLDSTile());

        graph.control.addElement(Body(), {kernel}, {storeLDS});
        graph.control.addElement(Sequence(), {storeLDS}, {loadLDS});

        graph.mapper.connect<LDS>(storeLDS, lds);
        graph.mapper.connect<MacroTile>(storeLDS, wgTile);

        graph.mapper.connect<LDS>(loadLDS, lds);
        graph.mapper.connect<MacroTile>(loadLDS, waveTile);

        auto lowerTile = std::make_shared<KernelGraph::LowerTile>(params, context.get());
        graph          = graph.transform(lowerTile);

        // XXX REMOVE THIS
        {
            std::ofstream dfile;
            dfile.open("foo.yaml", std::ofstream::out | std::ofstream::trunc);
            dfile << toYAML(graph);
            dfile.close();
        }
    }
}
