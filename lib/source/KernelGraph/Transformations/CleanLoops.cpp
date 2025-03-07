/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/CleanLoops.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Utilities/Logging.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        KernelGraph CleanLoops::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::cleanLoops");

            auto k = original;
            for(auto const& loop : k.control.getNodes<ForLoopOp>().to<std::vector>())
            {
                auto [lhs, rhs] = getForLoopIncrement(k, loop);
                auto forLoopDim
                    = getSize(k.coordinates.getNode(k.mapper.get(loop, NaryArgument::DEST)));

                //Ensure forLoopDim is translate time evaluatable.
                if(!(evaluationTimes(forLoopDim)[EvaluationTime::Translate]))
                    continue;

                //Ensure RHS is translate time evaluatable.
                if(!(evaluationTimes(rhs)[EvaluationTime::Translate]))
                    continue;

                //Only remove single iteration loops!
                if(evaluate(rhs) != evaluate(forLoopDim))
                    continue;

                // Replace ForLoop with Scope; ideally would reconnect but OK for now
                auto scope = replaceWith(k, loop, k.control.addElement(Scope()));

                purgeFor(k, loop);
            }

            return k;
        }
    }
}
