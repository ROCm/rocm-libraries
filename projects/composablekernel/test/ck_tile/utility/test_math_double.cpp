// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

// The device double specializations of ck_tile's exp, log, pow and expm1 used to call
// themselves (https://github.com/ROCm/rocm-libraries/issues/15813). From -O1 up the compiler
// removes the infinite recursion, so a kernel using them stored nothing.

namespace {

struct Exp
{
    __device__ double operator()(double x) const { return ck_tile::exp(x); }
    static double reference(double x) { return std::exp(x); }
};

struct Log
{
    __device__ double operator()(double x) const { return ck_tile::log(x); }
    static double reference(double x) { return std::log(x); }
};

struct Pow
{
    __device__ double operator()(double x) const { return ck_tile::pow(x, 1.5); }
    static double reference(double x) { return std::pow(x, 1.5); }
};

struct Expm1
{
    __device__ double operator()(double x) const { return ck_tile::expm1(x); }
    static double reference(double x) { return std::expm1(x); }
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

    ck_tile::DeviceMem x_buf(x.size() * sizeof(double));
    ck_tile::DeviceMem y_buf(y.size() * sizeof(double));
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

TEST(CkTileMathDouble, Exp) { CheckOnDevice<Exp>(kAnyInputs); }

TEST(CkTileMathDouble, Log) { CheckOnDevice<Log>(kPositiveInputs); }

TEST(CkTileMathDouble, Pow) { CheckOnDevice<Pow>(kPositiveInputs); }

TEST(CkTileMathDouble, Expm1) { CheckOnDevice<Expm1>(kAnyInputs); }
