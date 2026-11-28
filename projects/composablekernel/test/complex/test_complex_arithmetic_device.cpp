// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "ck/utility/dtype_complex.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "complex_test_data.hpp"

namespace {

using ck::complex_t;
using ck::index_t;
using ck_complex_test::same_part;
using ck_complex_test::special_values;

struct Add
{
    template <typename C>
    __host__ __device__ C operator()(const C& a, const C& b) const
    {
        return a + b;
    }
};

struct Subtract
{
    template <typename C>
    __host__ __device__ C operator()(const C& a, const C& b) const
    {
        return a - b;
    }
};

struct Multiply
{
    template <typename C>
    __host__ __device__ C operator()(const C& a, const C& b) const
    {
        return a * b;
    }
};

// The second operand is not used.
struct Negate
{
    template <typename C>
    __host__ __device__ C operator()(const C& a, const C&) const
    {
        return -a;
    }
};

// The second operand is not used.
struct Conjugate
{
    template <typename C>
    __host__ __device__ C operator()(const C& a, const C&) const
    {
        return ck::conj(a);
    }
};

template <typename T, typename Op>
__global__ void
apply_kernel(const complex_t<T>* a, const complex_t<T>* b, complex_t<T>* c, index_t n, Op op)
{
    const index_t i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if(i < n)
    {
        c[i] = op(a[i], b[i]);
    }
}

// c[i] = op(a[i], b[i]) computed on the device.
template <typename T, typename Op>
std::vector<complex_t<T>>
apply_on_device(const std::vector<complex_t<T>>& a, const std::vector<complex_t<T>>& b, Op op)
{
    using Elem              = complex_t<T>;
    const index_t n         = static_cast<index_t>(a.size());
    const std::size_t bytes = sizeof(Elem) * a.size();
    ck::DeviceMem a_dev(bytes), b_dev(bytes), c_dev(bytes);
    a_dev.ToDevice(a.data());
    b_dev.ToDevice(b.data());
    constexpr index_t kBlock = 256;
    hipLaunchKernelGGL((apply_kernel<T, Op>),
                       dim3((n + kBlock - 1) / kBlock),
                       dim3(kBlock),
                       0,
                       nullptr,
                       static_cast<const Elem*>(a_dev.GetDeviceBuffer()),
                       static_cast<const Elem*>(b_dev.GetDeviceBuffer()),
                       static_cast<Elem*>(c_dev.GetDeviceBuffer()),
                       n,
                       op);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
    std::vector<Elem> c(a.size());
    c_dev.FromDevice(c.data());
    return c;
}

template <typename T>
class ComplexArithmeticDevice : public ::testing::Test
{
};

using ComponentTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ComplexArithmeticDevice, ComponentTypes);

// Every complex number with special parts, times every other one: the device gives the host's
// result in every part, including the sign of zero, overflow and the infinities of Annex G.
template <typename T, typename Op>
void check_special_pairs(Op op, const char* name)
{
    using Elem    = complex_t<T>;
    const auto v  = special_values<T>();
    const auto m  = v.size();
    const auto mm = m * m;
    std::vector<Elem> a(mm * mm), b(mm * mm);
    for(std::size_t i = 0; i < a.size(); ++i)
    {
        const std::size_t x = i % mm, y = i / mm;
        a[i] = Elem{v[x % m], v[x / m]};
        b[i] = Elem{v[y % m], v[y / m]};
    }
    const auto c = apply_on_device<T>(a, b, op);
    long bad     = 0;
    for(std::size_t i = 0; i < a.size(); ++i)
    {
        const Elem h = op(a[i], b[i]);
        if(!(same_part(c[i].real, h.real) && same_part(c[i].imag, h.imag)))
        {
            if(bad < 5)
            {
                ADD_FAILURE() << name << " (" << a[i].real << ", " << a[i].imag << "), ("
                              << b[i].real << ", " << b[i].imag << "): device (" << c[i].real
                              << ", " << c[i].imag << "), host (" << h.real << ", " << h.imag
                              << ")";
            }
            ++bad;
        }
    }
    EXPECT_EQ(bad, 0) << name;
}

TYPED_TEST(ComplexArithmeticDevice, SpecialValuesMatchHost)
{
    check_special_pairs<TypeParam>(Add{}, "a + b");
    check_special_pairs<TypeParam>(Subtract{}, "a - b");
    check_special_pairs<TypeParam>(Multiply{}, "a * b");
    check_special_pairs<TypeParam>(Negate{}, "-a");
    check_special_pairs<TypeParam>(Conjugate{}, "conj(a)");
}

// Random integers with |m| in [2^(p-1), 2^p), p the precision of T: full mantissas.
template <typename T>
T random_integer(std::uint64_t& s)
{
    constexpr int p = std::numeric_limits<T>::digits;
    s += 0x9e3779b97f4a7c15u; // splitmix64
    std::uint64_t r = s;
    r               = (r ^ (r >> 30)) * 0xbf58476d1ce4e5b9u;
    r               = (r ^ (r >> 27)) * 0x94d049bb133111ebu;
    r ^= r >> 31;
    const auto m = static_cast<std::int64_t>((r >> (65 - p)) | (std::uint64_t{1} << (p - 1)));
    return static_cast<T>((r & 1u) != 0u ? -m : m);
}

// The integer of T nearest to v; any integer T holds is exact.
template <typename T>
T nearest(long double v)
{
    return static_cast<T>(std::nearbyint(v));
}

// |r - d| <= 2u |d| with u = 2^-p, as |r - d| * 2^(p-1) <= |d| in integers. r is an integer
// here: Kahan's algorithm rounds integers only, and a rounded integer is an integer.
template <typename T>
bool within_two_u(T r, __int128 d)
{
    constexpr int p   = std::numeric_limits<T>::digits;
    const __int128 ri = static_cast<__int128>(r);
    const __int128 e  = ri > d ? ri - d : d - ri;
    return e <= (d < 0 ? -d : d) >> (p - 1);
}

// Each part of a product is within 2u of the exact part when nothing underflows (Jeannerod,
// Louvet and Muller, Math. Comp. 82, 2013). The operands are integers, so the exact parts are
// computed in __int128 without error; floating point is scale invariant away from overflow and
// underflow, so integers stand for the whole normal range. Three products in four cancel in one
// part, where the textbook formula has no correct digit.
TYPED_TEST(ComplexArithmeticDevice, MultiplyWithinTwoUnitRoundoffOfExact)
{
    using T                  = TypeParam;
    using Elem               = complex_t<T>;
    using LD                 = long double;
    constexpr std::size_t kN = 4096;
    std::vector<Elem> a(kN), b(kN);
    std::uint64_t s = 2013u;
    for(std::size_t i = 0; i < kN; ++i)
    {
        const T ar = random_integer<T>(s);
        const T ai = random_integer<T>(s);
        T br       = random_integer<T>(s);
        T bi       = random_integer<T>(s);
        if(i % 4 == 1)
        {
            br = nearest<T>(LD{ai} * bi / ar); // ar br ~ ai bi
        }
        else if(i % 4 == 2)
        {
            bi = nearest<T>(-LD{ai} * br / ar); // ar bi ~ -ai br
        }
        else if(i % 4 == 3)
        {
            bi = nearest<T>(LD{ar} * br / ai); // ai bi ~ ar br
        }
        a[i] = Elem{ar, ai};
        b[i] = Elem{br, bi};
    }
    const auto c = apply_on_device<T>(a, b, Multiply{});
    long bad     = 0;
    for(std::size_t i = 0; i < kN; ++i)
    {
        const auto ar = static_cast<__int128>(a[i].real), ai = static_cast<__int128>(a[i].imag);
        const auto br = static_cast<__int128>(b[i].real), bi = static_cast<__int128>(b[i].imag);
        const __int128 re = ar * br - ai * bi;
        const __int128 im = ar * bi + ai * br;
        for(const Elem& r : {c[i], a[i] * b[i]}) // device and host
        {
            bad += within_two_u(r.real, re) && within_two_u(r.imag, im) ? 0 : 1;
        }
    }
    EXPECT_EQ(bad, 0);
}

} // namespace
