
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

TEST_CASE("Colour by unroll", "[kernel-graph]")
{
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;
    using namespace rocRoller::KernelGraph::CoordinateGraph;

    KernelGraph graph0;

    auto kernel = graph0.control.addElement(Kernel());
    auto loadA  = graph0.control.addElement(LoadTiled());
    auto loadB  = graph0.control.addElement(LoadTiled());
    auto assign = graph0.control.addElement(Assign());

    // ZZZ
}
