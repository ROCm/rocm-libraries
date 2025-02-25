#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Utilities.hpp"

class SimpleFixture : public ::testing::Test
{
protected:
    void TearDown() override
    {
        rocRoller::Settings::reset();
        rocRoller::Component::ComponentFactoryBase::ClearAllCaches();
    }
};
