/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <string>
#include <iostream>
#include <sstream>

namespace unittest
{

class UnitTestException
{
    public:
    std::string message;

    UnitTestException() {}
    UnitTestException(const std::string& msg) : message(msg) {}

    friend std::ostream& operator<<(std::ostream& os, const UnitTestException& e)
    {
        return os << e.message;
    }

    template <typename T>
    UnitTestException& operator<<(const T& t)
    {
        std::ostringstream oss;
        oss << t;
        message += oss.str();
        return *this;
    }
};


class UnitTestError   : public UnitTestException
{
    public:
    UnitTestError() {}
    UnitTestError(const std::string& msg) : UnitTestException(msg) {}
};

class UnitTestFailure : public UnitTestException
{
    public:
    UnitTestFailure() {}
    UnitTestFailure(const std::string& msg) : UnitTestException(msg) {}
};

class UnitTestKnownFailure : public UnitTestException
{
    public:
    UnitTestKnownFailure() {}
    UnitTestKnownFailure(const std::string& msg) : UnitTestException(msg) {}
};


}; //end namespace unittest
