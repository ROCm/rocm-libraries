#include <AssertOpKinds.hpp>

namespace rocRoller
{

    std::string toString(const AssertOpKind& assertOpKind)
    {
        switch(assertOpKind)
        {
        case AssertOpKind::NoOp:
            return "NoOp";
            break;
        case AssertOpKind::MemoryViolation:
            return "MemoryViolation";
            break;
        case AssertOpKind::STrap:
            return "STrap";
            break;
        default:
            return "Invalid";
            break;
        }
    }
}
