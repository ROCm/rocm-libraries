// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "client/GEMMParameters.hpp"
#include <rocRoller/Parameters/Solution/LDSBankSwizzleMode.hpp>
#include <rocRoller/Parameters/Solution/LoadOption.hpp>
#include <rocRoller/Parameters/Solution/StoreOption.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace CLI
{
    namespace detail
    {
        inline bool lexical_cast(const std::string& s, rocRoller::Parameters::Solution::LoadPath& v)
        {
            v = rocRoller::fromString<rocRoller::Parameters::Solution::LoadPath>(s);
            return true;
        }

        inline bool lexical_cast(const std::string&                          s,
                                 rocRoller::Parameters::Solution::StorePath& v)
        {
            v = rocRoller::fromString<rocRoller::Parameters::Solution::StorePath>(s);
            return true;
        }

        inline bool lexical_cast(const std::string& s, rocRoller::StreamKMode& v)
        {
            v = rocRoller::fromString<rocRoller::StreamKMode>(s);
            return true;
        }

        inline bool lexical_cast(const std::string& s, rocRoller::LDSBankSwizzleMode& v)
        {
            v = rocRoller::fromString<rocRoller::LDSBankSwizzleMode>(s);
            return true;
        }

        inline bool lexical_cast(const std::string& s, rocRoller::ScaleSkipPermlaneMode& v)
        {
            v = rocRoller::fromString<rocRoller::ScaleSkipPermlaneMode>(s);
            return true;
        }

        inline bool lexical_cast(const std::string& s, rocRoller::Client::GEMMClient::XYTuple& v)
        {
            return rocRoller::Client::GEMMClient::CLI::ParseXY(s, v);
        }

        inline bool lexical_cast(const std::string& s, rocRoller::Client::GEMMClient::MNKTuple& v)
        {
            return rocRoller::Client::GEMMClient::CLI::ParseMNK(s, v);
        }

        inline bool lexical_cast(const std::string& s, rocRoller::Client::GEMMClient::MNKBTuple& v)
        {
            return rocRoller::Client::GEMMClient::CLI::ParseMNKB(s, v);
        }

        inline bool lexical_cast(const std::string& s, rocRoller::Client::GEMMClient::MKNLTuple& v)
        {
            return rocRoller::Client::GEMMClient::CLI::ParseMKNL(s, v);
        }

        inline bool lexical_cast(const std::string& s, std::pair<int, int>& v)
        {
            return rocRoller::Client::GEMMClient::CLI::ParseIntPair(s, v);
        }
    }
}
