#include <string>
#include <variant>

#include <rocRoller/KernelGraph/ControlGraph/ControlEdge.hpp>

namespace rocRoller
{
    namespace KernelGraph::ControlGraph
    {
        std::string toString(ControlEdge const& e)
        {
            struct
            {
                std::string operator()(Sequence const&)
                {
                    return "Sequence";
                }
                std::string operator()(Initialize const&)
                {
                    return "Initialize";
                }
                std::string operator()(ForLoopIncrement const&)
                {
                    return "ForLoopIncrement";
                }
                std::string operator()(Body const&)
                {
                    return "Body";
                }
            } visitor;

            return std::visit(visitor, e);
        }
    }
}
