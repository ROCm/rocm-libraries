/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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

// Catch2 adapter for the comm byte-identity suite.
//
// The comm tests were authored against a zero-dependency harness exposing
// TEST/CHECK/CHECK_NEAR/ORIGAMI_TEST_MAIN. Upstream Origami runs Catch2, so
// this shim re-expresses that tiny surface on top of Catch2 — letting every
// test body compile unchanged inside the upstream `origami-comm-tests` runner:
//
//   TEST(name) { ... }    ->  TEST_CASE("comm: name")
//   CHECK(cond)           ->  Catch2 CHECK (non-fatal; same as old semantics)
//   CHECK_NEAR(a, b, tol) ->  absolute-tolerance CHECK
//   ORIGAMI_TEST_MAIN()   ->  no-op (Catch2WithMain supplies main())
#pragma once

#include <catch2/catch_test_macros.hpp>

#include <cmath>

// Catch2 already provides CHECK(cond) with the matching non-fatal semantics
// (records the failure, continues the test), so we only add the rest.
#define TEST(name) TEST_CASE("comm: " #name)

#define CHECK_NEAR(a, b, tol)                                              \
  CHECK(std::fabs(static_cast<double>(a) - static_cast<double>(b)) <=      \
        static_cast<double>(tol))

#define ORIGAMI_TEST_MAIN()
