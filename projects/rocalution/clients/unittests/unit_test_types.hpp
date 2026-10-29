/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

// Typed tests over (value type x backend), tolerances and inputs per value
// type, and comparisons for real and complex values

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "unit_test_common.hpp"
#include "unit_test_reference.hpp"

namespace rocalution_unit
{
    using cfloat  = std::complex<float>;
    using cdouble = std::complex<double>;

    template <typename T>
    struct is_complex : std::false_type
    {
    };

    template <typename T>
    struct is_complex<std::complex<T>> : std::true_type
    {
    };

    template <typename T>
    constexpr bool is_complex_v = is_complex<T>::value;

    template <typename T>
    constexpr bool is_value_type_v = std::is_arithmetic_v<T> || is_complex_v<T>;

    // float for std::complex<float>, T for real T
    template <typename T>
    struct real_type
    {
        using type = T;
    };

    template <typename T>
    struct real_type<std::complex<T>>
    {
        using type = T;
    };

    template <typename T>
    using real_t = typename real_type<T>::type;

    template <typename T>
    std::string type_name()
    {
        if constexpr(std::is_same_v<T, float>)
            return "float";
        else if constexpr(std::is_same_v<T, double>)
            return "double";
        else if constexpr(std::is_same_v<T, cfloat>)
            return "cfloat";
        else if constexpr(std::is_same_v<T, cdouble>)
            return "cdouble";
        else if constexpr(std::is_same_v<T, int>)
            return "int";
        else if constexpr(std::is_same_v<T, int64_t>)
            return "int64";
        else
        {
            static_assert(std::is_same_v<T, bool>, "no name for this value type");
            return "bool";
        }
    }

    // Relative tolerance of a computation in T, see is_near. Integral types are exact.
    template <typename T>
    constexpr double tol()
    {
        if constexpr(is_complex_v<T>)
            return tol<real_t<T>>();
        else if constexpr(std::is_same_v<T, float>)
            return 1e-5;
        else if constexpr(std::is_same_v<T, double>)
            return 1e-12;
        else
            return 0.0;
    }

    // A tolerance for double, scaled to the same number of roundings in T
    template <typename T>
    constexpr double scaled_tol(double for_double)
    {
        return for_double * static_cast<double>(std::numeric_limits<real_t<T>>::epsilon())
               / std::numeric_limits<double>::epsilon();
    }

    // A tolerance for double with slack above its rounding, such as 1e-10: scaled_tol would
    // make it meaningless for float, which gets tol<T>() instead
    template <typename T>
    constexpr double loose_tol(double for_double)
    {
        return std::max(tol<T>(), for_double);
    }

    template <typename T>
    T conj_of(const T& v)
    {
        if constexpr(is_complex_v<T>)
            return std::conj(v);
        else
            return v;
    }

    // The small integer k as a T: k + i m with m in {1, 2, -1, -2} by k mod 4 for
    // complex T, so that a missing or extra conjugation changes results; whether k is
    // odd for bool. Equal integers give equal values, and complex values keep the
    // order of their real parts.
    template <typename T>
    T from_int(int k)
    {
        if constexpr(is_complex_v<T>)
        {
            const int r = (k % 4 + 4) % 4;
            const int m = r < 2 ? r + 1 : 1 - r;
            return T(static_cast<real_t<T>>(k), static_cast<real_t<T>>(m));
        }
        else if constexpr(std::is_same_v<T, bool>)
            return k % 2 != 0;
        else
            return static_cast<T>(k);
    }

    template <typename T>
    std::vector<T> from_ints(const std::vector<int>& k)
    {
        std::vector<T> v(k.size());
        for(size_t i = 0; i < k.size(); ++i)
        {
            v[i] = from_int<T>(k[i]);
        }
        return v;
    }

    // Within tolerance * max(1, |ref|) of ref, measured by the modulus of the
    // difference for complex values; exact for integral types and bool
    template <typename T>
    testing::AssertionResult is_near(const T& actual, const T& ref, double tolerance = tol<T>())
    {
        if constexpr(std::is_integral_v<T>)
        {
            if(actual == ref)
            {
                return testing::AssertionSuccess();
            }
            return testing::AssertionFailure() << actual << " differs from " << ref;
        }
        else
        {
            const std::complex<double> a(actual), r(ref);
            const double               bound = tolerance * std::max(1.0, std::abs(r));
            if(std::abs(a - r) <= bound)
            {
                return testing::AssertionSuccess();
            }
            return testing::AssertionFailure() << actual << " differs from " << ref << " by "
                                               << std::abs(a - r) << ", more than " << bound;
        }
    }

    template <typename T, typename = std::enable_if_t<is_value_type_v<T>>>
    void expect_near(const T& actual, const T& ref, double tolerance = tol<T>())
    {
        EXPECT_TRUE(is_near(actual, ref, tolerance));
    }

    template <typename T>
    void expect_all_near(const std::vector<T>& actual, const std::vector<T>& ref, double tolerance)
    {
        ASSERT_EQ(actual.size(), ref.size());
        for(size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_TRUE(is_near<T>(actual[i], ref[i], tolerance)) << "at index " << i;
        }
    }

    template <typename T>
    void expect_near(const std::vector<T>& actual, const std::vector<T>& ref)
    {
        expect_all_near(actual, ref, tol<T>());
    }

    // The real version is in unit_test_reference.hpp
    template <typename T>
    void expect_near(const std::vector<std::complex<T>>& actual,
                     const std::vector<std::complex<T>>& ref,
                     double                              tolerance)
    {
        expect_all_near(actual, ref, tolerance);
    }

    // The make_vector and values_of of unit_test_common.hpp for bool, whose
    // std::vector has no data()
    inline std::unique_ptr<BaseVector<bool>> make_vector(Backend b, const std::vector<bool>& values)
    {
        std::unique_ptr<bool[]> data(new bool[values.size()]);
        std::copy(values.begin(), values.end(), data.get());

        auto host = std::make_unique<HostVector<bool>>(descriptor());
        host->Allocate(values.size());
        if(!values.empty())
        {
            host->CopyFromData(data.get());
        }
        if(b == Backend::Host)
        {
            return host;
        }

        std::unique_ptr<rocalution::AcceleratorVector<bool>> accel(
            rocalution::_rocalution_init_base_backend_vector<bool>(descriptor()));
        accel->Allocate(values.size());
        accel->CopyFromHost(*host);
        return accel;
    }

    inline std::vector<bool> values_of(const BaseVector<bool>& v)
    {
        HostVector<bool> host(descriptor());
        host.Allocate(v.GetSize());
        if(auto accel = dynamic_cast<const rocalution::AcceleratorVector<bool>*>(&v))
        {
            accel->CopyToHost(&host);
        }
        else
        {
            host.CopyFrom(v);
        }

        std::unique_ptr<bool[]> data(new bool[v.GetSize()]);
        if(v.GetSize() > 0)
        {
            host.CopyToData(data.get());
        }
        return std::vector<bool>(data.get(), data.get() + v.GetSize());
    }

    // The parameter of a typed test: a value type and a backend
    template <typename T, Backend B>
    struct Typed
    {
        using value_type                 = T;
        static constexpr Backend backend = B;
    };

    template <typename... Lists>
    struct concat_types;

    template <typename... A>
    struct concat_types<testing::Types<A...>>
    {
        using type = testing::Types<A...>;
    };

    template <typename... A, typename... B, typename... Rest>
    struct concat_types<testing::Types<A...>, testing::Types<B...>, Rest...>
        : concat_types<testing::Types<A..., B...>, Rest...>
    {
    };

    template <typename Ts>
    struct with_backends;

    template <typename... Ts>
    struct with_backends<testing::Types<Ts...>>
        : concat_types<testing::Types<Typed<Ts, Backend::Host>, Typed<Ts, Backend::Accelerator>>...>
    {
    };

    template <typename Ts>
    struct host_only;

    template <typename... Ts>
    struct host_only<testing::Types<Ts...>>
    {
        using type = testing::Types<Typed<Ts, Backend::Host>...>;
    };

    template <typename Ts>
    struct accelerator_only;

    template <typename... Ts>
    struct accelerator_only<testing::Types<Ts...>>
    {
        using type = testing::Types<Typed<Ts, Backend::Accelerator>...>;
    };

    // Typed<T, Host> and Typed<T, Accelerator> for each T of testing::Types<T...>
    template <typename Ts>
    using WithBackends = typename with_backends<Ts>::type;

    template <typename Ts>
    using HostOnly = typename host_only<Ts>::type;

    // For the tests that compare the backends
    template <typename Ts>
    using AcceleratorOnly = typename accelerator_only<Ts>::type;

    // The value types of the backend vectors. The library instantiates the
    // complex ones only with SUPPORT_COMPLEX.
#ifdef SUPPORT_COMPLEX
    using AllValueTypes = testing::Types<float, double, cfloat, cdouble, int, int64_t, bool>;
    using NumericTypes  = testing::Types<float, double, cfloat, cdouble, int, int64_t>;
    using FloatingTypes = testing::Types<float, double, cfloat, cdouble>;
#else
    using AllValueTypes = testing::Types<float, double, int, int64_t, bool>;
    using NumericTypes  = testing::Types<float, double, int, int64_t>;
    using FloatingTypes = testing::Types<float, double>;
#endif
    using RealTypes    = testing::Types<float, double>;
    using IntegerTypes = testing::Types<int, int64_t>;

    // Test names such as cfloat_Accelerator
    struct TypedName
    {
        template <typename P>
        static std::string GetName(int)
        {
            return type_name<typename P::value_type>() + "_" + backend_name(P::backend);
        }
    };

    template <typename T>
    struct TypeTag
    {
        using type = T;
    };

    template <typename T, typename Body>
    void run_as(Body& body)
    {
        SCOPED_TRACE(type_name<T>());
        body(TypeTag<T>());
    }

    template <typename Types>
    struct each_type;

    template <typename... Ts>
    struct each_type<testing::Types<Ts...>>
    {
        template <typename Body>
        static void run(Body& body)
        {
            (run_as<Ts>(body), ...);
        }
    };

    // body(TypeTag<T>()) for each T of Types. gtest has no typed value-parameterized
    // tests, so those run their body once per value type.
    template <typename Types, typename Body>
    void for_each_type(Body&& body)
    {
        each_type<Types>::run(body);
    }

    // Fixture of the typed tests over Typed<T, B>. In the tests, T is
    // typename TypeParam::value_type and the backend is TypeParam::backend.
    template <typename P>
    class TypedBackendTest : public testing::Test
    {
    protected:
        void SetUp() override
        {
            REQUIRE_BACKEND(P::backend);
        }
    };
}
