#pragma once

#include <memory>

namespace rocRoller
{
    namespace KernelGraph
    {
        class KernelGraph;
        struct KernelUnrollVisitor;
        struct LoopDistributeVisitor;

        using KernelGraphPtr = std::shared_ptr<KernelGraph>;
    }
}
