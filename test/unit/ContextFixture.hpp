#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sstream>

#include <rocRoller/Context_fwd.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include "Utilities.hpp"

class ContextFixture : public ::testing::Test
{
protected:
    rocRoller::ContextPtr m_context;

    void SetUp() override;
    void TearDown() override;

    std::string output();
    void        clearOutput();
    void        writeOutputToFile(std::string const& filename);

    std::string testKernelName() const;
    std::string testKernelName(std::string const& suffix) const;

    void setKernelOptions(rocRoller::KernelOptions const& kernelOption);

    virtual rocRoller::ContextPtr createContext() = 0;

    bool isLocalDevice() const;

    std::vector<rocRoller::Register::ValuePtr>
        createRegisters(rocRoller::Register::Type const        regType,
                        rocRoller::DataType const              dataType,
                        size_t const                           amount,
                        int const                              regCount     = 1,
                        rocRoller::Register::AllocationOptions allocOptions = {});

    rocRoller::KernelOptions m_kernelOptions;
};
