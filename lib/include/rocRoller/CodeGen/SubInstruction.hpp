#include <rocRoller/CodeGen/Instruction.hpp>

namespace rocRoller
{
    Instruction ScalarSubInt32(ContextPtr         ctx,
                               Register::ValuePtr dest,
                               Register::ValuePtr lhs,
                               Register::ValuePtr rhs,
                               std::string        comment = "");

    Instruction ScalarSubUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment = "");

    Instruction ScalarSubUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment = "");

    Instruction VectorSubUInt32(ContextPtr         ctx,
                                Register::ValuePtr dest,
                                Register::ValuePtr lhs,
                                Register::ValuePtr rhs,
                                std::string        comment = "");

    Instruction VectorSubUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr carryOut,
                                          Register::ValuePtr carryIn,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment = "");

    Instruction VectorSubUInt32CarryInOut(ContextPtr         ctx,
                                          Register::ValuePtr dest,
                                          Register::ValuePtr lhs,
                                          Register::ValuePtr rhs,
                                          std::string        comment = "");

    Instruction VectorSubUInt32CarryOut(ContextPtr         ctx,
                                        Register::ValuePtr dest,
                                        Register::ValuePtr carry,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        std::string        comment = "");

    Instruction VectorSubUInt32CarryOut(ContextPtr         ctx,
                                        Register::ValuePtr dest,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        std::string        comment = "");

    Instruction VectorSubRevUInt32(ContextPtr         ctx,
                                   Register::ValuePtr dest,
                                   Register::ValuePtr lhs,
                                   Register::ValuePtr rhs,
                                   std::string        comment = "");

    Instruction VectorSubRevUInt32CarryOut(ContextPtr         ctx,
                                           Register::ValuePtr dest,
                                           Register::ValuePtr carryOut,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs,
                                           std::string        comment = "");

    Instruction VectorSubRevUInt32CarryOut(ContextPtr         ctx,
                                           Register::ValuePtr dest,
                                           Register::ValuePtr lhs,
                                           Register::ValuePtr rhs,
                                           std::string        comment = "");

    Instruction VectorSubRevUInt32CarryInOut(ContextPtr         ctx,
                                             Register::ValuePtr dest,
                                             Register::ValuePtr carryOut,
                                             Register::ValuePtr carryIn,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs,
                                             std::string        comment = "");

    Instruction VectorSubRevUInt32CarryInOut(ContextPtr         ctx,
                                             Register::ValuePtr dest,
                                             Register::ValuePtr lhs,
                                             Register::ValuePtr rhs,
                                             std::string        comment = "");
}
