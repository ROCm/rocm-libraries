// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once
#ifndef TESTCONSTANTS_H
#define TESTCONSTANTS_H

#include <stdexcept>

extern int    verbose;
extern size_t ramgb;
extern size_t vramgb;

extern size_t n_random_tests;

extern size_t random_seed;
extern double test_prob;
extern double emulation_prob;
extern double complex_interleaved_prob_factor;
extern double real_prob_factor;
extern double complex_planar_prob_factor;
extern double callback_prob_factor;

extern double half_epsilon;
extern double single_epsilon;
extern double double_epsilon;
extern bool   skip_runtime_fails;

extern double max_linf_eps_double;
extern double max_l2_eps_double;
extern double max_linf_eps_single;
extern double max_l2_eps_single;
extern double max_linf_eps_half;
extern double max_l2_eps_half;

extern int n_hip_failures;

#endif
