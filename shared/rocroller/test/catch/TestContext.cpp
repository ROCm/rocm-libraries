// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestContext.hpp"

#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

TestContext::TestContext(rocRoller::ContextPtr context)
    : m_context(context)
{
    auto settings = rocRoller::Settings::getInstance();
    settings->set(rocRoller::Settings::EnforceGraphConstraints, true);
    settings->set(rocRoller::Settings::AuditControlTracers, true);
}

TestContext::~TestContext()
{
    m_context.reset();
    rocRoller::Settings::reset();
}

std::vector<rocRoller::Register::ValuePtr>
    TestContext::createRegisters(rocRoller::Register::Type const        regType,
                                 rocRoller::DataType const              dataType,
                                 size_t const                           amount,
                                 int const                              regCount,
                                 rocRoller::Register::AllocationOptions allocOptions)
{
    std::vector<rocRoller::Register::ValuePtr> regs;
    for(size_t i = 0; i < amount; i++)
    {
        auto reg = std::make_shared<rocRoller::Register::Value>(
            m_context, regType, dataType, regCount, allocOptions);
        try
        {
            reg->allocateNow();
        }
        catch(...)
        {
            std::cout << i << std::endl;
            throw;
        }
        regs.push_back(reg);
    }
    return regs;
}
