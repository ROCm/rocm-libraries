// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdlib>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <string>

namespace hipdnn_test_sdk::utilities
{

class ScopedEnvironmentVariableSetter
{
public:
    ScopedEnvironmentVariableSetter(const std::string& varName, const std::string& value = "")
        : _varName(varName)
    {
        _originalValue = hipdnn_data_sdk::utilities::getEnv(varName.c_str());
        _hadOriginalValue = !_originalValue.empty();
        hipdnn_data_sdk::utilities::setEnv(varName.c_str(), value.c_str());
    }

    ~ScopedEnvironmentVariableSetter()
    {
        if(_hadOriginalValue)
        {
            hipdnn_data_sdk::utilities::setEnv(_varName.c_str(), _originalValue.c_str());
        }
        else
        {
            hipdnn_data_sdk::utilities::unsetEnv(_varName.c_str());
        }
    }

    void setValue(const std::string& value)
    {
        hipdnn_data_sdk::utilities::setEnv(_varName.c_str(), value.c_str());
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

} // namespace hipdnn_test_sdk::utilities
