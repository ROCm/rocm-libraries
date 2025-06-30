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

#include "ContextFixture.hpp"

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUCapability.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Scheduling/MetaObserver.hpp>
#include <rocRoller/Scheduling/Observers/AllocatingObserver.hpp>
#include <rocRoller/Scheduling/Observers/WaitcntObserver.hpp>
#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Utils.hpp>

void ContextFixture::SetUp()
{
    using namespace rocRoller;

    m_context = createContext();

    RecordProperty("local_device", isLocalDevice());

    auto settings = Settings::getInstance();
    settings->set(Settings::EnforceGraphConstraints, true);
    settings->set(Settings::AuditControlTracers, true);
}

void ContextFixture::TearDown()
{
    m_context.reset();
    rocRoller::Settings::reset();
    rocRoller::Component::ComponentFactoryBase::ClearAllCaches();
}

std::string ContextFixture::output()
{
    return m_context->instructions()->toString();
}

void ContextFixture::clearOutput()
{
    m_context->instructions()->clear();
}

void ContextFixture::writeOutputToFile(std::string const& filename)
{
    std::ofstream file(filename);
    file << m_context->instructions()->toString();
}

std::string ContextFixture::testKernelName() const
{
    return testKernelName("");
}

std::string ContextFixture::testKernelName(std::string const& suffix) const
{
    auto const* info = testing::UnitTest::GetInstance()->current_test_info();

    std::string rv = info->test_suite_name();

    rv += info->name();

    char const* type = info->type_param();
    if(type)
        rv += type;

    rv += "_kernel";
    rv += suffix;

    // Replace 'bad' characters with '_'.
    for(auto& c : rv)
        if(!isalnum(c) && c != '_')
            c = '_';

    return rv;
}

void ContextFixture::setKernelOptions(rocRoller::KernelOptions const& kernelOption)
{
    m_context->m_kernelOptions = kernelOption;
    m_kernelOptions            = kernelOption;
}

bool ContextFixture::isLocalDevice() const
{
    return m_context && m_context->hipDeviceIndex() >= 0;
}

std::vector<rocRoller::Register::ValuePtr>
    ContextFixture::createRegisters(rocRoller::Register::Type const        regType,
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
