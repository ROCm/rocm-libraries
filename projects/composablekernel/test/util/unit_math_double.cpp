// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/math_v2.hpp"

// The device double specializations of ck::math's exp, log, pow, expm1 and cos used to call
// themselves (https://github.com/ROCm/rocm-libraries/issues/15813). From -O1 up the compiler
// removes the infinite recursion, so a kernel using them stored nothing.

namespace {

struct Exp
{
    __device__ double operator()(double x) const { return ck::math::exp(x); }
    static double reference(double x) { return std::exp(x); }
};

struct Log
{
    __device__ double operator()(double x) const { return ck::math::log(x); }
    static double reference(double x) { return std::log(x); }
};

struct Pow
{
    __device__ double operator()(double x) const { return ck::math::pow(x, 1.5); }
    static double reference(double x) { return std::pow(x, 1.5); }
};

struct Expm1
{
    __device__ double operator()(double x) const { return ck::math::expm1(x); }
    static double reference(double x) { return std::expm1(x); }
};

struct Cos
{
    __device__ double operator()(double x) const { return ck::math::cos(x); }
    static double reference(double x) { return std::cos(x); }
};

template <typename F>
__global__ void apply(const double* x, double* y, int n)
{
    const int i = threadIdx.x;
    if(i < n)
    {
        y[i] = F{}(x[i]);
    }
}

// Runs F on the device and compares it with the host reference in double precision. The
// outputs start as NaN, so an output the kernel never stores fails the comparison.
template <typename F>
void CheckOnDevice(const std::vector<double>& x)
{
    const int n = static_cast<int>(x.size());
    std::vector<double> y(x.size(), std::numeric_limits<double>::quiet_NaN());

    ck::DeviceMem x_buf(x.size() * sizeof(double));
    ck::DeviceMem y_buf(y.size() * sizeof(double));
    x_buf.ToDevice(x.data());
    y_buf.ToDevice(y.data());

    apply<F><<<1, n>>>(static_cast<const double*>(x_buf.GetDeviceBuffer()),
                       static_cast<double*>(y_buf.GetDeviceBuffer()),
                       n);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    y_buf.FromDevice(y.data());

    for(int i = 0; i < n; ++i)
    {
        const double expected = F::reference(x[i]);
        // A float computation would be off by about 1e-7 relative, far outside this bound.
        const double tolerance = 1e-13 * std::max(1.0, std::abs(expected));
        EXPECT_NEAR(y[i], expected, tolerance) << "x = " << x[i];
    }
}

const std::vector<double> kAnyInputs      = {-2.5, -1.0, -1e-3, 0.0, 1e-3, 0.5, 1.0, 3.75, 10.0};
const std::vector<double> kPositiveInputs = {1e-3, 0.5, 1.0, 2.0, 3.75, 10.0, 123.5};

} // namespace

TEST(CkMathDouble, Exp) { CheckOnDevice<Exp>(kAnyInputs); }

TEST(CkMathDouble, Log) { CheckOnDevice<Log>(kPositiveInputs); }

TEST(CkMathDouble, Pow) { CheckOnDevice<Pow>(kPositiveInputs); }

TEST(CkMathDouble, Expm1) { CheckOnDevice<Expm1>(kAnyInputs); }

TEST(CkMathDouble, Cos) { CheckOnDevice<Cos>(kAnyInputs); }
