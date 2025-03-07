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
