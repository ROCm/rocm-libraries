
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
}

void ContextFixture::TearDown()
{
    m_context.reset();
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

    char const* value = info->type_param();
    if(value)
        rv += value;

    rv += "_kernel";
    rv += suffix;

    // Replace 'bad' characters with '_'.
    for(auto& c : rv)
        if(!isalnum(c) && c != '_')
            c = '_';

    return rv;
}

bool ContextFixture::isLocalDevice() const
{
    return m_context && m_context->hipDeviceIndex() >= 0;
}
