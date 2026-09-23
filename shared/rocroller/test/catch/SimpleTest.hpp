// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Settings.hpp>

class SimpleTest
{
public:
    SimpleTest() = default;
    virtual ~SimpleTest()
    {
        rocRoller::Settings::reset();
        rocRoller::Component::ComponentFactoryBase::ClearAllCaches();
    }
};
