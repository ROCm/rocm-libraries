#pragma once

#include "ContextFixture.hpp"

class GenericContextFixture : public ContextFixture
{
protected:
    void SetUp() override;

    rocRoller::ContextPtr createContext() override;

    virtual rocRoller::GPUArchitectureTarget targetArchitecture();
};
