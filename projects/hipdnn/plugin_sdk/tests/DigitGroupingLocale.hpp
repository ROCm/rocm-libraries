// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <locale>

namespace hipdnn_plugin_sdk::test
{

/// A locale that groups every three digits with ',' -- the shape `std::locale("")` takes on
/// most hosts. Any stream imbued with it groups the numbers it writes, in hexadecimal as well
/// as decimal. Defined in DigitGroupingLocale.cpp, the one test source built with RTTI: the
/// standard library finds a facet through its type information, which a class compiled
/// without RTTI does not have.
std::locale digitGroupingLocale();

/// Installs digitGroupingLocale() as the global C++ locale for its lifetime and restores the
/// previous one on destruction. The locale is unnamed, so the C library's locale is untouched.
class ScopedDigitGroupingLocale
{
public:
    ScopedDigitGroupingLocale()
        : _previous(std::locale::global(digitGroupingLocale()))
    {
    }

    ~ScopedDigitGroupingLocale()
    {
        std::locale::global(_previous);
    }

    ScopedDigitGroupingLocale(const ScopedDigitGroupingLocale&) = delete;
    ScopedDigitGroupingLocale& operator=(const ScopedDigitGroupingLocale&) = delete;
    ScopedDigitGroupingLocale(ScopedDigitGroupingLocale&&) = delete;
    ScopedDigitGroupingLocale& operator=(ScopedDigitGroupingLocale&&) = delete;

private:
    std::locale _previous;
};

} // namespace hipdnn_plugin_sdk::test
