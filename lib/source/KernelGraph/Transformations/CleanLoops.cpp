
#include <rocRoller/KernelGraph/KernelGraph.hpp>
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

        KernelGraph cleanLoops(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::cleanLoops");

            rocRoller::Log::getLogger()->debug("KernelGraph::cleanLoops()");

            auto k = original;
            for(auto const& loop : k.control.getNodes<ForLoopOp>().to<std::vector>())
            {
                auto [lhs, rhs] = getForLoopIncrement(k, loop);
                auto forLoopDim = getSize(
                    std::get<Dimension>(k.coordinates.getElement(k.mapper.get<Dimension>(loop))));

                //Ensure forLoopDim is translate time evaluatable.
                if(!(evaluationTimes(forLoopDim)[EvaluationTime::Translate]))
                    continue;

                //Only remove single iteration loops!
                if(evaluate(rhs) != evaluate(forLoopDim))
                    continue;

                // Replace ForLoop with Scope; ideally would reconnect but OK for now
                auto scope = replaceWithScope(k, loop);

                purgeFor(k, loop);
            }

            return k;
        }
    }
}
