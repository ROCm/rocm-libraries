#include <rocRoller/AssertOpKinds.hpp>

namespace rocRoller
{

    std::string toString(const AssertOpKind& assertOpKind)
    {
        switch(assertOpKind)
        {
        case AssertOpKind::NoOp:
            return "NoOp";
        case AssertOpKind::MemoryViolation:
            return "MemoryViolation";
        case AssertOpKind::STrap:
            return "STrap";
        default:
            return "Invalid";
        }
    }

    std::ostream& operator<<(std::ostream& stream, AssertOpKind const k)
    {
        return stream << toString(k);
    }
}
