// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_RNG_GENERATOR_TYPES_H_
#define ROCRAND_RNG_GENERATOR_TYPES_H_

#include "generator_type.hpp"

#include "lfsr113.hpp"
#include "mrg.hpp"
#include "mt19937.hpp"
#include "mtgp32.hpp"
#include "philox4x32_10.hpp"
#include "sobol.hpp"
#include "threefry.hpp"
#include "xorwow.hpp"

namespace rocrand_impl::host
{

extern template struct generator_type<lfsr113_generator>;
extern template struct generator_type<lfsr113_generator_host<false>>;
extern template struct generator_type<lfsr113_generator_host<true>>;
extern template struct generator_type<mrg31k3p_generator>;
extern template struct generator_type<mrg31k3p_generator_host<false>>;
extern template struct generator_type<mrg31k3p_generator_host<true>>;
extern template struct generator_type<mrg32k3a_generator>;
extern template struct generator_type<mrg32k3a_generator_host<false>>;
extern template struct generator_type<mrg32k3a_generator_host<true>>;
extern template struct generator_type<mt19937_generator>;
extern template struct generator_type<mt19937_generator_host<true>>;
extern template struct generator_type<mt19937_generator_host<false>>;
extern template struct generator_type<mtgp32_generator>;
extern template struct generator_type<mtgp32_generator_host<false>>;
extern template struct generator_type<mtgp32_generator_host<true>>;
extern template struct generator_type<philox4x32_10_generator>;
extern template struct generator_type<philox4x32_10_generator_host<false>>;
extern template struct generator_type<philox4x32_10_generator_host<true>>;
extern template struct generator_type<scrambled_sobol32_generator_host<false>>;
extern template struct generator_type<scrambled_sobol32_generator_host<true>>;
extern template struct generator_type<scrambled_sobol32_generator>;
extern template struct generator_type<scrambled_sobol64_generator_host<false>>;
extern template struct generator_type<scrambled_sobol64_generator_host<true>>;
extern template struct generator_type<scrambled_sobol64_generator>;
extern template struct generator_type<sobol32_generator_host<false>>;
extern template struct generator_type<sobol32_generator_host<true>>;
extern template struct generator_type<sobol32_generator>;
extern template struct generator_type<sobol64_generator_host<false>>;
extern template struct generator_type<sobol64_generator_host<true>>;
extern template struct generator_type<sobol64_generator>;
extern template struct generator_type<threefry2x32_20_generator>;
extern template struct generator_type<threefry2x32_20_generator_host<false>>;
extern template struct generator_type<threefry2x32_20_generator_host<true>>;
extern template struct generator_type<threefry2x64_20_generator>;
extern template struct generator_type<threefry2x64_20_generator_host<false>>;
extern template struct generator_type<threefry2x64_20_generator_host<true>>;
extern template struct generator_type<threefry4x32_20_generator>;
extern template struct generator_type<threefry4x32_20_generator_host<false>>;
extern template struct generator_type<threefry4x32_20_generator_host<true>>;
extern template struct generator_type<threefry4x64_20_generator>;
extern template struct generator_type<threefry4x64_20_generator_host<false>>;
extern template struct generator_type<threefry4x64_20_generator_host<true>>;
extern template struct generator_type<xorwow_generator>;
extern template struct generator_type<xorwow_generator_host<false>>;
extern template struct generator_type<xorwow_generator_host<true>>;

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_GENERATOR_TYPES_H_
