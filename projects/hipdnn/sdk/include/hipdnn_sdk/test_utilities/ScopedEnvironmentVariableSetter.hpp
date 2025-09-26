// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdlib>
#include <hipdnn_sdk/utilities/PlatformUtils.hpp>
#include <string>

namespace hipdnn_sdk::test_utilities
{

class ScopedEnvironmentVariableSetter
{
public:
    ScopedEnvironmentVariableSetter(const std::string& varName, const std::string& value = "")
        : _varName(varName)
    {
        _originalValue = hipdnn_sdk::utilities::getEnv(varName.c_str());
        _hadOriginalValue = !_originalValue.empty();
        hipdnn_sdk::utilities::setEnv(varName.c_str(), value.c_str());
    }

    ~ScopedEnvironmentVariableSetter()
    {
        if(_hadOriginalValue)
        {
            hipdnn_sdk::utilities::setEnv(_varName.c_str(), _originalValue.c_str());
        }
        else
        {
            hipdnn_sdk::utilities::unsetEnv(_varName.c_str());
        }
    }

    void setValue(const std::string& value)
    {
        hipdnn_sdk::utilities::setEnv(_varName.c_str(), value.c_str());
    }

    ScopedEnvironmentVariableSetter(const ScopedEnvironmentVariableSetter&) = delete;
    ScopedEnvironmentVariableSetter& operator=(const ScopedEnvironmentVariableSetter&) = delete;

    ScopedEnvironmentVariableSetter(ScopedEnvironmentVariableSetter&&) = default;
    ScopedEnvironmentVariableSetter& operator=(ScopedEnvironmentVariableSetter&&) = default;

private:
    std::string _varName;
    std::string _originalValue;
    bool _hadOriginalValue;
};

} // namespace hipdnn_sdk::test_utilities
