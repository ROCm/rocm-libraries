
#pragma once

#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Settings.hpp>

class SimpleTest
{
public:
    SimpleTest() = default;
    ~SimpleTest()
    {
        rocRoller::Settings::reset();
        rocRoller::Component::ComponentFactoryBase::ClearAllCaches();
    }
};
