/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// A YAML host and a msgpack host share one SONAME. LoadLibraryFile names the
// parser mismatch from the filename suffix before it opens the file.

#include <gtest/gtest.h>

#include <string>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Tensile.hpp>

#if defined(TENSILE_YAML)
#define TENSILELITE_FOREIGN_LIBRARY "library-load-error-mismatch.dat"
#define TENSILELITE_FOREIGN_DETAIL \
    "is msgpack. This libtensilelite-host was built with the YAML parser only."
#elif defined(TENSILE_MSGPACK)
#define TENSILELITE_FOREIGN_LIBRARY "library-load-error-mismatch.yaml"
#define TENSILELITE_FOREIGN_DETAIL \
    "is YAML. This libtensilelite-host was built with the msgpack parser only."
#else
#error "LibraryLoadError_test requires TENSILE_YAML or TENSILE_MSGPACK"
#endif

TEST(LibraryLoadError, ForeignSuffixNamesTheParser)
{
    auto library = TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm,
                                                 TensileLite::ContractionSolution>(
        TENSILELITE_FOREIGN_LIBRARY);

    EXPECT_EQ(library, nullptr);
    std::string const detail = TensileLite::LastLibraryLoadError();
    EXPECT_NE(detail.find(TENSILELITE_FOREIGN_LIBRARY), std::string::npos) << detail;
    EXPECT_NE(detail.find(TENSILELITE_FOREIGN_DETAIL), std::string::npos) << detail;
}
