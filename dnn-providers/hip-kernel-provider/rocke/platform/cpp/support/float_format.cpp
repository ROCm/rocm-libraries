// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Python's repr(float). CPython finds the shortest decimal digit string that
 * reads back as the same double (David Gay's dtoa, mode 0) and then lays it
 * out with PyOS_double_to_string's 'r' rules. std::to_chars without a
 * precision finds the same shortest digit string -- when two strings of that
 * length read back, both pick the one nearer the value -- and, unlike the
 * printf family, it ignores the C locale. This file only re-lays-out its
 * scientific form.
 */
#include "rocke/float_format.h"

#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdlib>

void rocke_append_python_float_repr(rocke_strbuf_t* out, double value)
{
    if(std::isnan(value))
    {
        rocke_strbuf_append(out, "nan"); /* repr never signs a NaN */
        return;
    }
    if(std::isinf(value))
    {
        rocke_strbuf_append(out, value < 0 ? "-inf" : "inf");
        return;
    }

    /* "[-]d[.ddd]e[+-]x": the shortest digits that read back as `value`. A
     * double needs at most 17 digits and a three-digit exponent, so the 31
     * usable bytes always fit and to_chars cannot report value_too_large. */
    char scientific[32];
    const std::to_chars_result conversion = std::to_chars(
        scientific, scientific + sizeof(scientific) - 1, value, std::chars_format::scientific);
    assert(conversion.ec == std::errc{});
    *conversion.ptr = '\0';

    const char* cursor = scientific;
    if(*cursor == '-')
    {
        rocke_strbuf_append_char(out, '-');
        ++cursor;
    }
    char digits[20];
    int digit_count = 0;
    for(; *cursor != 'e'; ++cursor)
    {
        if(*cursor != '.')
        {
            digits[digit_count++] = *cursor;
        }
    }
    digits[digit_count] = '\0';
    const int exp10 = std::atoi(cursor + 1); /* value = d.ddd x 10**exp10 */

    if(exp10 < -4 || exp10 >= 16)
    {
        /* Exponential: d[.ddd]e+XX. */
        rocke_strbuf_append_char(out, digits[0]);
        if(digit_count > 1)
        {
            rocke_strbuf_append_char(out, '.');
            rocke_strbuf_append(out, digits + 1);
        }
        rocke_strbuf_appendf(out, "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
    }
    else if(exp10 < 0)
    {
        /* 0.000ddd */
        rocke_strbuf_append(out, "0.");
        for(int zero = 1; zero < -exp10; ++zero)
        {
            rocke_strbuf_append_char(out, '0');
        }
        rocke_strbuf_append(out, digits);
    }
    else if(exp10 + 1 >= digit_count)
    {
        /* ddd000.0 */
        rocke_strbuf_append(out, digits);
        for(int zero = digit_count; zero < exp10 + 1; ++zero)
        {
            rocke_strbuf_append_char(out, '0');
        }
        rocke_strbuf_append(out, ".0");
    }
    else
    {
        /* ddd.ddd */
        rocke_strbuf_append_n(out, digits, (size_t)(exp10 + 1));
        rocke_strbuf_append_char(out, '.');
        rocke_strbuf_append(out, digits + exp10 + 1);
    }
}
