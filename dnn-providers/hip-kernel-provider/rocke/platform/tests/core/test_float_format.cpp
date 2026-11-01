// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* rocke_append_python_float_repr, checked against Python's repr(float).
 *
 * Without arguments it checks a table of values whose repr is known: zeros,
 * integral values, both sides of the fixed/exponential switches at 1e-4 and
 * 1e16, 2**-24 and its neighbours, subnormals, the finite extremes, nan and the
 * infinities. It checks the table again under a locale whose decimal point is
 * a comma, when one is installed, because repr does not depend on the locale.
 *
 *   --repr
 *       read one IEEE-754 double per stdin line as 16 hex digits of its bit
 *       pattern and print the formatter's text for it, one per line.
 *       tests/core/test_float_format.py (found through ROCKE_FLOAT_FORMAT_TEST)
 *       compares that text with repr() over many values.
 */
#include "rocke/float_format.h"

#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "rocke/strbuf.h"

namespace
{
struct repr_case
{
    const char* name;
    double value;
    const char* repr; /* Python: repr(value) */
};

const double k_inf = std::numeric_limits<double>::infinity();
const double k_nan = std::numeric_limits<double>::quiet_NaN();

const repr_case k_cases[] = {
    {"0.0", 0x0.0p+0, "0.0"},
    {"-0.0", -0x0.0p+0, "-0.0"},
    {"1.0", 0x1.0000000000000p+0, "1.0"},
    {"-1.0", -0x1.0000000000000p+0, "-1.0"},
    {"100.0", 0x1.9000000000000p+6, "100.0"},
    {"2**53", 0x1.0000000000000p+53, "9007199254740992.0"},
    {"0.5", 0x1.0000000000000p-1, "0.5"},
    {"0.1", 0x1.999999999999ap-4, "0.1"},
    {"1/3", 0x1.5555555555555p-2, "0.3333333333333333"},
    {"1e-4", 0x1.a36e2eb1c432dp-14, "0.0001"},
    {"below 1e-4", 0x1.a36e2eb1c432cp-14, "9.999999999999999e-05"},
    {"above 1e-4", 0x1.a36e2eb1c432ep-14, "0.00010000000000000002"},
    {"1e-5", 0x1.4f8b588e368f1p-17, "1e-05"},
    {"0.00012", 0x1.f75104d551d69p-14, "0.00012"},
    {"1e15", 0x1.c6bf526340000p+49, "1000000000000000.0"},
    {"below 1e16", 0x1.1c37937e07fffp+53, "9999999999999998.0"},
    {"1e16", 0x1.1c37937e08000p+53, "1e+16"},
    {"above 1e16", 0x1.1c37937e08001p+53, "1.0000000000000002e+16"},
    {"1e17", 0x1.6345785d8a000p+56, "1e+17"},
    {"1.5e300", 0x1.1eb2d66005835p+997, "1.5e+300"},
    {"below 2**-24", 0x1.fffffffffffffp-25, "5.960464477539062e-08"},
    {"2**-24", 0x1.0000000000000p-24, "5.960464477539063e-08"},
    {"above 2**-24", 0x1.0000000000001p-24, "5.960464477539064e-08"},
    {"2**-25", 0x1.0000000000000p-25, "2.9802322387695312e-08"},
    {"2**-23", 0x1.0000000000000p-23, "1.1920928955078125e-07"},
    {"smallest subnormal", 0x0.0000000000001p-1022, "5e-324"},
    {"largest subnormal", 0x0.fffffffffffffp-1022, "2.225073858507201e-308"},
    {"smallest normal", 0x1.0000000000000p-1022, "2.2250738585072014e-308"},
    {"largest finite", 0x1.fffffffffffffp+1023, "1.7976931348623157e+308"},
    {"-largest finite", -0x1.fffffffffffffp+1023, "-1.7976931348623157e+308"},
    {"float32 largest", 0x1.fffffe0000000p+127, "3.4028234663852886e+38"},
    {"float32 smallest normal", 0x1.0000000000000p-126, "1.1754943508222875e-38"},
    {"inf", k_inf, "inf"},
    {"-inf", -k_inf, "-inf"},
    {"nan", k_nan, "nan"},
    {"-nan", -k_nan, "nan"},
};

/* Locales that spell the decimal point as a comma; the first one installed is
 * used for the locale check. */
const char* const k_comma_locales[]
    = {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "fr_FR.UTF-8", "fr_FR.utf8", "fr_FR"};

int check_table(const char* locale_name)
{
    int failures = 0;
    for(const repr_case& c : k_cases)
    {
        rocke_strbuf_t text;
        if(rocke_strbuf_init(&text, 32) != 0)
        {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        rocke_append_python_float_repr(&text, c.value);
        if(text.oom || strcmp(rocke_strbuf_cstr(&text), c.repr) != 0)
        {
            fprintf(stderr,
                    "%s (locale %s): got \"%s\", repr is \"%s\"\n",
                    c.name,
                    locale_name,
                    text.oom ? "<out of memory>" : rocke_strbuf_cstr(&text),
                    c.repr);
            ++failures;
        }
        rocke_strbuf_free(&text);
    }
    return failures;
}

int print_repr_lines()
{
    char line[64];
    while(fgets(line, sizeof(line), stdin))
    {
        uint64_t bits = 0;
        if(sscanf(line, "%16" SCNx64, &bits) != 1)
        {
            fprintf(stderr, "expected 16 hex digits, got: %s", line);
            return 1;
        }
        double value;
        memcpy(&value, &bits, sizeof(value));
        rocke_strbuf_t text;
        if(rocke_strbuf_init(&text, 32) != 0)
        {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        rocke_append_python_float_repr(&text, value);
        puts(text.oom ? "<out of memory>" : rocke_strbuf_cstr(&text));
        rocke_strbuf_free(&text);
    }
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    if(argc == 2 && strcmp(argv[1], "--repr") == 0)
    {
        return print_repr_lines();
    }
    if(argc != 1)
    {
        fprintf(stderr, "usage: %s [--repr]\n", argv[0]);
        return 2;
    }

    int failures = check_table("C");
    const char* comma_locale = nullptr;
    for(const char* name : k_comma_locales)
    {
        if(setlocale(LC_ALL, name) && strcmp(localeconv()->decimal_point, ",") == 0)
        {
            comma_locale = name;
            break;
        }
    }
    if(comma_locale)
    {
        failures += check_table(comma_locale);
        setlocale(LC_ALL, "C");
    }
    else
    {
        printf("no comma-decimal locale is installed; checked the C locale only\n");
    }
    if(failures)
    {
        fprintf(stderr, "%d float repr mismatches\n", failures);
        return 1;
    }
    printf("ok\n");
    return 0;
}
