
#pragma once

#include "ContextFixture.hpp"

class GenericContextFixture : public ContextFixture
{
protected:
    void SetUp() override;

    virtual rocRoller::ContextPtr createContext() override;

    virtual std::string targetArchitecture();
};
