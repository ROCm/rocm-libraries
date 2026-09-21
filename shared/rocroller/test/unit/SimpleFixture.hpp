// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Utilities.hpp"

#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Settings.hpp>

class SimpleFixture : public ::testing::Test
{
protected:
    void TearDown() override
    {
        rocRoller::Settings::reset();
        rocRoller::Component::ComponentFactoryBase::ClearAllCaches();
    }
};
