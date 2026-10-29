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

// The backend vectors for every value type the library instantiates them with.
// The inputs are small integers, so that most results are exact.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <type_traits>

#include "unit_test_types.hpp"
#include "unit_test_vector_data.hpp"

using namespace rocalution_unit;

namespace
{
    template <typename T>
    std::vector<T> sequence(int n, int start, int step)
    {
        std::vector<int> k(n);
        for(int i = 0; i < n; ++i)
        {
            k[i] = start + step * i;
        }
        return from_ints<T>(k);
    }

    template <typename T>
    std::vector<T> vector_of(const T* a, size_t n)
    {
        return std::vector<T>(a, a + n);
    }

    template <typename T>
    std::unique_ptr<HostVector<T>> host_vector_of(const std::vector<T>& values)
    {
        auto v = make_vector(Backend::Host, values);
        return std::unique_ptr<HostVector<T>>(dynamic_cast<HostVector<T>*>(v.release()));
    }

    // The operations that end the program with a FATAL_ERROR for some value types of
    // the numeric suite, by backend. See the backend_vector_unsupported tests.
    template <typename T>
    bool defines(Backend b, const std::string& operation)
    {
        struct Undefined
        {
            std::vector<std::string> host;
            std::vector<std::string> accelerator;
        };

        const std::vector<std::string> integers = {"int", "int64"};

        static const std::map<std::string, Undefined> table = {
            {"Scale", {{}, integers}},
            {"Dot", {{}, integers}},
            {"DotNonConj", {{}, integers}},
            {"Norm", {integers, integers}},
            {"Asum", {{}, integers}},
            {"Amax", {{}, integers}},
            {"Power", {{}, {"cfloat", "cdouble", "int"}}},
        };

        auto entry = table.find(operation);
        if(entry == table.end())
        {
            ADD_FAILURE() << "no value type entry for " << operation;
            return false;
        }

        const std::vector<std::string>& undefined
            = b == Backend::Host ? entry->second.host : entry->second.accelerator;
        return std::find(undefined.begin(), undefined.end(), type_name<T>()) == undefined.end();
    }
}

#define REQUIRE_DEFINED(operation)                                                              \
    if(!defines<typename TypeParam::value_type>(TypeParam::backend, operation))                 \
    {                                                                                           \
        GTEST_SKIP() << operation << " is not defined for "                                     \
                     << type_name<typename TypeParam::value_type>() << " on "                   \
                     << backend_name(TypeParam::backend) << ", see backend_vector_unsupported"; \
    }

template <typename P>
class TypedVectorTest : public TypedBackendTest<P>
{
protected:
    template <typename V>
    static std::unique_ptr<BaseVector<V>> vec(const std::vector<V>& values)
    {
        return make_vector(P::backend, values);
    }
};

// Arithmetic, for the value types other than bool

template <typename P>
class backend_vector : public TypedVectorTest<P>
{
};

TYPED_TEST_SUITE(backend_vector, WithBackends<NumericTypes>, TypedName);

TYPED_TEST(backend_vector, dot_sums_the_products_of_sequences)
{
    using T = typename TypeParam::value_type;
    REQUIRE_DEFINED("Dot");

    const int            n = 100;
    const std::vector<T> a = sequence<T>(n, 1, 1);
    const std::vector<T> b(n, from_int<T>(2));

    T dot = T(0);
    for(int i = 0; i < n; ++i)
    {
        dot += conj_of(a[i]) * b[i];
    }

    EXPECT_EQ(this->vec(a)->Dot(*this->vec(b)), dot);
}

TYPED_TEST(backend_vector, dot_conjugates_this_vector_and_dot_non_conj_does_not)
{
    using T = typename TypeParam::value_type;
    REQUIRE_DEFINED("Dot");

    const std::vector<T> a = x<T>(), b = y<T>();
    T                    dot = T(0), dot_non_conj = T(0);
    for(size_t i = 0; i < a.size(); ++i)
    {
        dot += conj_of(a[i]) * b[i];
        dot_non_conj += a[i] * b[i];
    }

    auto v = this->vec(a);
    EXPECT_EQ(v->Dot(*this->vec(b)), dot);
    EXPECT_EQ(v->DotNonConj(*this->vec(b)), dot_non_conj);
}

TYPED_TEST(backend_vector, reductions_of_empty_vectors_are_zero)
{
    using T = typename TypeParam::value_type;

    auto    e       = this->vec(std::vector<T>());
    auto    f       = this->vec(std::vector<T>());
    Backend backend = TypeParam::backend;

    if(defines<T>(backend, "Dot"))
    {
        EXPECT_EQ(e->Dot(*f), T(0));
        EXPECT_EQ(e->DotNonConj(*f), T(0));
    }
    EXPECT_EQ(e->Reduce(), T(0));
    if(defines<T>(backend, "Norm"))
    {
        EXPECT_EQ(e->Norm(), T(0));
    }
    if(defines<T>(backend, "Asum"))
    {
        EXPECT_EQ(e->Asum(), T(0));
    }
    if(defines<T>(backend, "Amax"))
    {
        T value = from_int<T>(5);
        EXPECT_EQ(e->Amax(value), 0);
        EXPECT_EQ(value, T(0));
    }
}

TYPED_TEST(backend_vector, scale_multiplies_every_entry)
{
    using T = typename TypeParam::value_type;
    REQUIRE_DEFINED("Scale");

    const std::vector<T> a     = sequence<T>(100, 0, 1);
    const T              alpha = from_int<T>(3);

    auto v = this->vec(a);
    v->Scale(alpha);

    EXPECT_EQ(values_of(*v), elementwise(a, a, [&](T p, T) { return p * alpha; }));
}

TYPED_TEST(backend_vector, updates_combine_the_vectors_entrywise)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();
    const std::vector<T> c     = elementwise(a, b, [](T p, T q) { return p - q; });
    const T              alpha = from_int<T>(3), beta = from_int<T>(-2), gamma = from_int<T>(-1);

    auto v = this->vec(a);
    v->AddScale(*this->vec(b), alpha);
    EXPECT_EQ(values_of(*v), elementwise(a, b, [&](T p, T q) { return p + alpha * q; }));

    v = this->vec(a);
    v->ScaleAdd(alpha, *this->vec(b));
    EXPECT_EQ(values_of(*v), elementwise(a, b, [&](T p, T q) { return alpha * p + q; }));

    v = this->vec(a);
    v->ScaleAddScale(alpha, *this->vec(b), beta);
    EXPECT_EQ(values_of(*v), elementwise(a, b, [&](T p, T q) { return alpha * p + beta * q; }));

    v = this->vec(a);
    v->ScaleAdd2(alpha, *this->vec(b), beta, *this->vec(c), gamma);
    std::vector<T> expected(a.size());
    for(size_t i = 0; i < a.size(); ++i)
    {
        expected[i] = alpha * a[i] + beta * b[i] + gamma * c[i];
    }
    EXPECT_EQ(values_of(*v), expected);

    v = this->vec(a);
    v->PointWiseMult(*this->vec(b));
    EXPECT_EQ(values_of(*v), elementwise(a, b, [](T p, T q) { return p * q; }));

    v = this->vec(a);
    v->PointWiseMult(*this->vec(b), *this->vec(c));
    EXPECT_EQ(values_of(*v), elementwise(b, c, [](T p, T q) { return p * q; }));
}

// Elements [2, 7) of this from elements [4, 9) of the source
TYPED_TEST(backend_vector, scale_add_scale_with_offsets_updates_only_the_range)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();
    const T              alpha = from_int<T>(3), beta = from_int<T>(-2);

    auto v = this->vec(a);
    v->ScaleAddScale(alpha, *this->vec(b), beta, 4, 2, 5);
    std::vector<T> expected = a;
    for(int i = 0; i < 5; ++i)
    {
        expected[2 + i] = alpha * a[2 + i] + beta * b[4 + i];
    }
    EXPECT_EQ(values_of(*v), expected);
}

TYPED_TEST(backend_vector, reduce_sums_the_entries)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>();
    EXPECT_EQ(this->vec(a)->Reduce(), std::accumulate(a.begin(), a.end(), T(0)));
}

TYPED_TEST(backend_vector, norm_is_the_euclidean_norm)
{
    using T = typename TypeParam::value_type;
    REQUIRE_DEFINED("Norm");

    const std::vector<T> a     = x<T>();
    double               norm2 = 0;
    for(const T& e : a)
    {
        norm2 += std::norm(std::complex<double>(e));
    }

    EXPECT_TRUE(is_near(this->vec(a)->Norm(), static_cast<T>(std::sqrt(norm2))));
}

TYPED_TEST(backend_vector, asum_sums_the_magnitudes)
{
    using T = typename TypeParam::value_type;
    REQUIRE_DEFINED("Asum");

    auto checks = [this] {
        const std::vector<T> a    = x<T>();
        const T              asum = this->vec(a)->Asum();

        if constexpr(is_complex_v<T>)
        {
            // As BLAS scasum: the real sum of |re| + |im|
            real_t<T> expected = 0;
            for(const T& e : a)
            {
                expected += std::abs(e.real()) + std::abs(e.imag());
            }
            EXPECT_EQ(asum.real() + asum.imag(), expected);
            KNOWN_BUG("the host returns (sum |re|, sum |im|) instead of the real sum");
            EXPECT_EQ(asum, T(expected, 0));
        }
        else
        {
            T expected = T(0);
            for(const T& e : a)
            {
                expected += std::abs(e);
            }
            EXPECT_EQ(asum, expected);
        }
    };

    if(is_complex_v<T> && TypeParam::backend == Backend::Accelerator)
    {
        KNOWN_BUG("the complex rocblasTasum is a FATAL_ERROR stub");
        expect_in_child(checks);
        return;
    }
    checks();
}

// The index is zero based. The largest magnitude may be negative.
TYPED_TEST(backend_vector, amax_returns_the_zero_based_index_and_the_largest_magnitude)
{
    using T = typename TypeParam::value_type;
    REQUIRE_DEFINED("Amax");

    auto equal_magnitude_checks = [this] {
        std::vector<T> equal_magnitudes(8);
        for(size_t i = 0; i < equal_magnitudes.size(); ++i)
        {
            equal_magnitudes[i] = i % 2 == 0 ? from_int<T>(7) : T(0) - from_int<T>(7);
        }
        T       value = T(0);
        int64_t index = this->vec(equal_magnitudes)->Amax(value);
        EXPECT_GE(index, 0);
        EXPECT_LT(index, 8);
        EXPECT_EQ(value, magnitude(from_int<T>(7)));
    };

    auto position_checks = [this] {
        for(int position : {0, 5, 11})
        {
            for(int sign : {1, -1})
            {
                SCOPED_TRACE("at " + std::to_string(position) + " with sign "
                             + std::to_string(sign));

                std::vector<T> a = x<T>();
                a[position]      = from_int<T>(sign * 20);

                T       value = T(0);
                int64_t index = this->vec(a)->Amax(value);
                EXPECT_EQ(index, position);
                EXPECT_EQ(value, magnitude(a[position]));
            }
        }
    };

    if(TypeParam::backend == Backend::Host)
    {
        equal_magnitude_checks();
        position_checks();
        return;
    }

    // Behind the complex stub, the position checks also have the bug of the real types below
    if(is_complex_v<T>)
    {
        KNOWN_BUG("the complex rocblasTamax is a FATAL_ERROR stub");
        expect_in_child([&] {
            equal_magnitude_checks();
            position_checks();
        });
        return;
    }

    equal_magnitude_checks();
    KNOWN_BUG("it returns the one-based index of rocBLAS and the value after the maximum; with "
              "the maximum last it reads past the end and exits the process");
    expect_in_child(position_checks);
}

// Both return the last element of the result. In place the source is the vector itself.
TYPED_TEST(backend_vector, inclusive_and_exclusive_sums_return_the_last_entry)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>();
    std::vector<T>       inclusive(a.size()), exclusive(a.size(), T(0));
    std::partial_sum(a.begin(), a.end(), inclusive.begin());
    std::partial_sum(a.begin(), a.end() - 1, exclusive.begin() + 1);

    auto v = this->vec(std::vector<T>(a.size(), T(0)));
    EXPECT_EQ(v->InclusiveSum(*this->vec(a)), inclusive.back());
    EXPECT_EQ(values_of(*v), inclusive);
    EXPECT_EQ(v->ExclusiveSum(*this->vec(a)), exclusive.back());
    EXPECT_EQ(values_of(*v), exclusive);

    v = this->vec(a);
    EXPECT_EQ(v->InclusiveSum(*v), inclusive.back());
    EXPECT_EQ(values_of(*v), inclusive);

    v = this->vec(a);
    EXPECT_EQ(v->ExclusiveSum(*v), exclusive.back());
    EXPECT_EQ(values_of(*v), exclusive);

    auto e = this->vec(std::vector<T>());
    EXPECT_EQ(e->InclusiveSum(*this->vec(std::vector<T>())), T(0));
    EXPECT_EQ(e->ExclusiveSum(*this->vec(std::vector<T>())), T(0));
}

// Restriction sums the fine entries of each coarse entry, prolongation copies
// them back. Fine entries mapped to -1 are skipped, and prolong to zero.
TYPED_TEST(backend_vector, restriction_sums_and_prolongation_copies_back)
{
    using T = typename TypeParam::value_type;

    const std::vector<T>   fine = x<T>();
    const std::vector<int> map  = {0, 2, -1, 1, 0, 2, 3, -1, 1, 0, 3, 2};

    std::vector<T> coarse(4, T(0));
    for(size_t i = 0; i < fine.size(); ++i)
    {
        if(map[i] >= 0)
        {
            coarse[map[i]] += fine[i];
        }
    }

    if(TypeParam::backend == Backend::Accelerator)
    {
        GTEST_SKIP() << "Restriction and Prolongation are host only, see the fallback test";
    }

    auto c = this->vec(std::vector<T>(4, from_int<T>(7)));
    ASSERT_TRUE(c->Restriction(*this->vec(fine), *this->vec(map)));
    EXPECT_EQ(values_of(*c), coarse);

    std::vector<T> prolonged(fine.size(), T(0));
    for(size_t i = 0; i < fine.size(); ++i)
    {
        prolonged[i] = map[i] >= 0 ? coarse[map[i]] : T(0);
    }

    auto f = this->vec(std::vector<T>(fine.size(), from_int<T>(7)));
    ASSERT_TRUE(f->Prolongation(*c, *this->vec(map)));
    EXPECT_EQ(values_of(*f), prolonged);
}

TYPED_TEST(backend_vector, add_index_values_adds_at_the_indices)
{
    using T = typename TypeParam::value_type;

    const std::vector<T>   a     = x<T>();
    const std::vector<int> index = {9, 2, 0, 11, 5};
    const std::vector<T>   b     = from_ints<T>({10, 20, 30, 40, 50});

    std::vector<T> add = a;
    for(size_t k = 0; k < index.size(); ++k)
    {
        add[index[k]] += b[k];
    }

    auto v = this->vec(a);
    v->AddIndexValues(*this->vec(index), *this->vec(b));
    EXPECT_EQ(values_of(*v), add);
}

TYPED_TEST(backend_vector, power_raises_every_entry)
{
    using T = typename TypeParam::value_type;
    REQUIRE_DEFINED("Power");

    if constexpr(std::is_integral_v<T>)
    {
        auto v = this->vec(from_ints<T>({1, -2, 3, 0, 5}));
        v->Power(2);
        EXPECT_EQ(values_of(*v), from_ints<T>({1, 4, 9, 0, 25}));
        v->Power(3);
        EXPECT_EQ(values_of(*v), from_ints<T>({1, 64, 729, 0, 15625}));
    }
    else if constexpr(is_complex_v<T>)
    {
        // Principal roots
        auto v = this->vec(std::vector<T>{T(-7, 24), T(3, 4), T(0, 2), T(-4, 0)});
        v->Power(0.5);
        expect_near(values_of(*v), {T(3, 4), T(2, 1), T(1, 1), T(0, 2)});
        v->Power(3);
        expect_near(values_of(*v), {T(-117, 44), T(2, 11), T(-2, 2), T(0, -8)});
    }
    else
    {
        auto v = this->vec(std::vector<T>{1, 4, 9, 16, 0.25});
        v->Power(0.5);
        expect_near(values_of(*v), {1, 2, 3, 4, 0.5});
        v->Power(3);
        expect_near(values_of(*v), {1, 8, 27, 64, 0.125});
    }
}

// The accelerator ends the program in Power for complex and int vectors only if
// they have entries
TYPED_TEST(backend_vector, updates_of_empty_vectors_leave_them_empty)
{
    using T       = typename TypeParam::value_type;
    const T alpha = from_int<T>(3), beta = from_int<T>(-2);

    auto e = this->vec(std::vector<T>());
    auto f = this->vec(std::vector<T>());
    auto g = this->vec(std::vector<T>());

    e->AddScale(*f, alpha);
    e->ScaleAdd(alpha, *f);
    e->ScaleAddScale(alpha, *f, beta);
    e->ScaleAdd2(alpha, *f, beta, *g, alpha);
    e->PointWiseMult(*f);
    e->PointWiseMult(*f, *g);
    if(defines<T>(TypeParam::backend, "Scale"))
    {
        e->Scale(alpha);
    }
    e->Power(2.0);

    EXPECT_EQ(e->GetSize(), 0);
    EXPECT_EQ(f->GetSize(), 0);
    EXPECT_EQ(g->GetSize(), 0);
}

// Storage and data movement, for every value type

template <typename P>
class backend_vector_data : public TypedVectorTest<P>
{
};

TYPED_TEST_SUITE(backend_vector_data, WithBackends<AllValueTypes>, TypedName);

TYPED_TEST(backend_vector_data, copies_of_a_vector_onto_itself_keep_the_values)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>();
    auto                 v = this->vec(a);

    v->CopyFrom(*v);
    EXPECT_EQ(values_of(*v), a);
    v->CopyFromAsync(*v);
    rocalution::_rocalution_sync();
    EXPECT_EQ(values_of(*v), a);
    v->CopyTo(v.get());
    EXPECT_EQ(values_of(*v), a);
    v->CopyToAsync(v.get());
    rocalution::_rocalution_sync();
    EXPECT_EQ(values_of(*v), a);
}

// The accelerator has no AddIndexValues for bool
TYPED_TEST(backend_vector_data, index_operations_with_empty_vectors_change_nothing)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>();
    auto                 e = this->vec(std::vector<T>());
    auto                 f = this->vec(std::vector<T>());
    auto                 p = this->vec(std::vector<int>());

    e->Permute(*p);
    e->PermuteBackward(*p);
    e->CopyFromPermute(*f, *p);
    e->CopyFromPermuteBackward(*f, *p);
    EXPECT_EQ(e->GetSize(), 0);

    auto v = this->vec(a);
    v->GetIndexValues(*p, f.get());
    v->SetIndexValues(*p, *f);
    if constexpr(!std::is_same_v<T, bool>)
    {
        v->AddIndexValues(*p, *f);
    }
    EXPECT_EQ(values_of(*v), a);
    EXPECT_EQ(f->GetSize(), 0);
}

TYPED_TEST(backend_vector_data, set_values_fills_every_entry)
{
    using T = typename TypeParam::value_type;

    auto v = this->vec(x<T>());

    v->Zeros();
    EXPECT_EQ(values_of(*v), std::vector<T>(12, T(0)));
    v->Ones();
    EXPECT_EQ(values_of(*v), std::vector<T>(12, T(1)));
    v->SetValues(from_int<T>(-2));
    EXPECT_EQ(values_of(*v), std::vector<T>(12, from_int<T>(-2)));
}

// Into an empty vector and into one of the same size
TYPED_TEST(backend_vector_data, copies_fill_empty_and_same_size_vectors)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>();
    auto                 v = this->vec(a);

    using Copy = std::function<void(BaseVector<T>&)>;
    const std::vector<std::pair<std::string, Copy>> copies
        = {{"CopyFrom", [&](BaseVector<T>& w) { w.CopyFrom(*v); }},
           {"CopyFromAsync", [&](BaseVector<T>& w) { w.CopyFromAsync(*v); }},
           {"CopyTo", [&](BaseVector<T>& w) { v->CopyTo(&w); }},
           {"CopyToAsync", [&](BaseVector<T>& w) { v->CopyToAsync(&w); }}};

    for(const auto& copy : copies)
    {
        SCOPED_TRACE(copy.first);

        auto w = this->vec(std::vector<T>());
        copy.second(*w);
        rocalution::_rocalution_sync();
        EXPECT_EQ(values_of(*w), a);

        w = this->vec(y<T>());
        copy.second(*w);
        rocalution::_rocalution_sync();
        EXPECT_EQ(values_of(*w), a);
    }
}

// From a vector of this backend into an empty one of the other backend, and back
// into the vector of this backend
TYPED_TEST(backend_vector_data, copies_between_backends_keep_the_values)
{
    using T = typename TypeParam::value_type;
    REQUIRE_BACKEND(Backend::Accelerator);

    const Backend other
        = TypeParam::backend == Backend::Host ? Backend::Accelerator : Backend::Host;
    const std::vector<T> a = x<T>(), b = y<T>();

    using Copy = std::function<void(const BaseVector<T>&, BaseVector<T>&)>;
    const std::vector<std::pair<std::string, Copy>> copies
        = {{"CopyFrom", [](const BaseVector<T>& s, BaseVector<T>& d) { d.CopyFrom(s); }},
           {"CopyFromAsync", [](const BaseVector<T>& s, BaseVector<T>& d) { d.CopyFromAsync(s); }},
           {"CopyTo", [](const BaseVector<T>& s, BaseVector<T>& d) { s.CopyTo(&d); }},
           {"CopyToAsync", [](const BaseVector<T>& s, BaseVector<T>& d) { s.CopyToAsync(&d); }}};

    for(const auto& copy : copies)
    {
        SCOPED_TRACE(copy.first);

        auto v = this->vec(a);
        auto w = make_vector(other, std::vector<T>());
        copy.second(*v, *w);
        rocalution::_rocalution_sync();
        EXPECT_EQ(values_of(*w), a);

        w = make_vector(other, b);
        copy.second(*w, *v);
        rocalution::_rocalution_sync();
        EXPECT_EQ(values_of(*v), b);
    }
}

// Elements [2, 7) of this from elements [4, 9) of the source
TYPED_TEST(backend_vector_data, copy_with_offsets_copies_only_the_range)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();

    auto v = this->vec(a);
    v->CopyFrom(*this->vec(b), 4, 2, 5);
    std::vector<T> expected = a;
    std::copy(b.begin() + 4, b.begin() + 9, expected.begin() + 2);
    EXPECT_EQ(values_of(*v), expected);
}

// Forward: element i moves to perm[i]. Backward: element i comes from perm[i].
TYPED_TEST(backend_vector_data, permutations_move_entries_forward_and_backward)
{
    using T = typename TypeParam::value_type;

    const std::vector<T>   a    = x<T>();
    const std::vector<int> perm = {4, 0, 11, 7, 1, 3, 10, 2, 6, 9, 5, 8};

    std::vector<T> forward(a.size()), backward(a.size());
    for(size_t i = 0; i < a.size(); ++i)
    {
        forward[perm[i]] = a[i];
        backward[i]      = a[perm[i]];
    }

    auto p = this->vec(perm);

    auto v = this->vec(a);
    v->Permute(*p);
    EXPECT_EQ(values_of(*v), forward);

    v = this->vec(a);
    v->PermuteBackward(*p);
    EXPECT_EQ(values_of(*v), backward);

    v = this->vec(std::vector<T>(a.size(), T(0)));
    v->CopyFromPermute(*this->vec(a), *p);
    EXPECT_EQ(values_of(*v), forward);

    v = this->vec(std::vector<T>(a.size(), T(0)));
    v->CopyFromPermuteBackward(*this->vec(a), *p);
    EXPECT_EQ(values_of(*v), backward);
}

TYPED_TEST(backend_vector_data, get_and_set_index_values_access_the_indices)
{
    using T = typename TypeParam::value_type;

    const std::vector<T>   a     = x<T>();
    const std::vector<int> index = {9, 2, 0, 11, 5};
    const std::vector<T>   b     = from_ints<T>({10, 21, 30, 41, 50});

    auto v      = this->vec(a);
    auto values = this->vec(std::vector<T>(index.size(), T(0)));
    v->GetIndexValues(*this->vec(index), values.get());
    EXPECT_EQ(values_of(*values), (std::vector<T>{a[9], a[2], a[0], a[11], a[5]}));

    std::vector<T> set = a;
    for(size_t k = 0; k < index.size(); ++k)
    {
        set[index[k]] = b[k];
    }

    v->SetIndexValues(*this->vec(index), *this->vec(b));
    EXPECT_EQ(values_of(*v), set);
}

// The values are host arrays on every backend. The accelerator copies them asynchronously.
TYPED_TEST(backend_vector_data, continuous_values_are_host_arrays)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>();
    auto                 v = this->vec(a);

    std::unique_ptr<T[]> part(new T[5]);
    v->GetContinuousValues(3, 8, part.get());
    rocalution::_rocalution_sync();
    EXPECT_EQ(vector_of(part.get(), 5), std::vector<T>(a.begin() + 3, a.begin() + 8));

    const std::vector<T> b = from_ints<T>({10, 21, 30, 41});
    v->SetContinuousValues(6, 10, array_of(b).get());
    rocalution::_rocalution_sync();
    std::vector<T> expected = a;
    std::copy(b.begin(), b.end(), expected.begin() + 6);
    EXPECT_EQ(values_of(*v), expected);
}

// The data pointers belong to the backend; the host data pointers are host arrays
TYPED_TEST(backend_vector_data, data_pointers_belong_to_the_backend)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();

    auto v    = this->vec(a);
    T*   data = nullptr;
    v->LeaveDataPtr(&data);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(v->GetSize(), 0);

    auto w = this->vec(std::vector<T>(a.size(), T(0)));
    w->CopyFromData(data);
    EXPECT_EQ(values_of(*w), a);

    this->vec(b)->CopyToData(data);
    auto u = this->vec(std::vector<T>());
    u->SetDataPtr(&data, static_cast<int64_t>(b.size()));
    EXPECT_EQ(values_of(*u), b);

    std::unique_ptr<T[]> host(new T[a.size()]);
    u->CopyFromHostData(array_of(a).get());
    u->CopyToHostData(host.get());
    EXPECT_EQ(vector_of(host.get(), a.size()), a);
}

TYPED_TEST(backend_vector_data, info_names_the_backend)
{
    using T = typename TypeParam::value_type;

    auto v = this->vec(x<T>());
    testing::internal::CaptureStdout();
    v->Info();
    const std::string info = testing::internal::GetCapturedStdout();
    EXPECT_NE(
        info.find(TypeParam::backend == Backend::Host ? "HostVector" : "HIPAcceleratorVector"),
        std::string::npos)
        << info;
}

// The value types with an order

template <typename P>
class backend_vector_sort : public TypedVectorTest<P>
{
};

using SortTypes = testing::Types<float, double, int, int64_t, bool>;
TYPED_TEST_SUITE(backend_vector_sort, WithBackends<SortTypes>, TypedName);

TYPED_TEST(backend_vector_sort, sort_orders_the_values_and_returns_the_permutation)
{
    using T = typename TypeParam::value_type;

    std::vector<T> a;
    for(int k : {6, -2, 8, 3, -10, 18, 4, -12, 11, 0})
    {
        a.push_back(std::is_floating_point_v<T> ? static_cast<T>(k / 2.0) : from_int<T>(k));
    }

    std::vector<int> order(a.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) { return a[i] < a[j]; });
    std::vector<T> sorted_values(a.size());
    for(size_t i = 0; i < a.size(); ++i)
    {
        sorted_values[i] = a[order[i]];
    }

    auto v      = this->vec(a);
    auto sorted = this->vec(std::vector<T>(a.size(), T(0)));
    v->Sort(sorted.get(), nullptr);
    EXPECT_EQ(values_of(*sorted), sorted_values);

    auto perm = this->vec(std::vector<int>(a.size(), 0));
    sorted    = this->vec(std::vector<T>(a.size(), T(0)));
    v->Sort(sorted.get(), perm.get());
    EXPECT_EQ(values_of(*sorted), sorted_values);

    std::vector<int> p = values_of(*perm);
    for(size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_EQ(a[p[i]], sorted_values[i]) << "at " << i;
    }
    std::sort(p.begin(), p.end());
    std::iota(order.begin(), order.end(), 0);
    EXPECT_EQ(p, order);
}

TYPED_TEST(backend_vector_sort, sort_of_an_empty_vector_leaves_the_outputs_empty)
{
    using T = typename TypeParam::value_type;

    auto e      = this->vec(std::vector<T>());
    auto sorted = this->vec(std::vector<T>());
    auto perm   = this->vec(std::vector<int>());
    e->Sort(sorted.get(), perm.get());
    EXPECT_EQ(sorted->GetSize(), 0);
    EXPECT_EQ(perm->GetSize(), 0);
}

// Random values, for the floating point value types. Only the real parts are
// checked: the accelerator also draws the imaginary parts, the host leaves them zero.

template <typename P>
class backend_vector_random : public TypedVectorTest<P>
{
};

TYPED_TEST_SUITE(backend_vector_random, WithBackends<FloatingTypes>, TypedName);

TYPED_TEST(backend_vector_random, uniform_values_lie_in_the_range_and_repeat_with_the_seed)
{
    using T     = typename TypeParam::value_type;
    const int n = 20000;

    auto u = this->vec(std::vector<T>(n, T(0)));
    u->SetRandomUniform(1234, T(-2), T(3));
    const std::vector<T> values = values_of(*u);

    double min = 3, max = -2, sum = 0;
    for(const T& e : values)
    {
        min = std::min(min, static_cast<double>(std::real(e)));
        max = std::max(max, static_cast<double>(std::real(e)));
        sum += std::real(e);
    }
    EXPECT_GE(min, -2.0);
    EXPECT_LE(max, 3.0);
    EXPECT_NEAR(sum / n, 0.5, 0.05);

    auto again = this->vec(std::vector<T>(n, T(0)));
    again->SetRandomUniform(1234, T(-2), T(3));
    EXPECT_EQ(values_of(*again), values);

    auto other = this->vec(std::vector<T>(n, T(0)));
    other->SetRandomUniform(4321, T(-2), T(3));
    EXPECT_NE(values_of(*other), values);
}

// The third argument is the standard deviation
TYPED_TEST(backend_vector_random, normal_values_have_the_mean_and_standard_deviation)
{
    using T     = typename TypeParam::value_type;
    const int n = 20000;

    auto g = this->vec(std::vector<T>(n, T(0)));
    g->SetRandomNormal(4321, T(3), T(2));
    const std::vector<T> values = values_of(*g);

    double sum = 0;
    for(const T& e : values)
    {
        sum += std::real(e);
    }
    const double mean  = sum / n;
    double       sumsq = 0;
    for(const T& e : values)
    {
        sumsq += (std::real(e) - mean) * (std::real(e) - mean);
    }
    EXPECT_NEAR(mean, 3.0, 0.05);
    EXPECT_NEAR(std::sqrt(sumsq / (n - 1)), 2.0, 0.05);

    auto again = this->vec(std::vector<T>(n, T(0)));
    again->SetRandomNormal(4321, T(3), T(2));
    EXPECT_EQ(values_of(*again), values);
}

TYPED_TEST(backend_vector_random, uniform_values_of_an_empty_vector_leave_it_empty)
{
    using T = typename TypeParam::value_type;

    auto e = this->vec(std::vector<T>());
    e->SetRandomUniform(1234, T(-2), T(3));
    EXPECT_EQ(e->GetSize(), 0);
}

// Copies between float and double

template <typename P>
class backend_vector_precision : public TypedVectorTest<P>
{
};

TYPED_TEST_SUITE(backend_vector_precision, WithBackends<RealTypes>, TypedName);

TYPED_TEST(backend_vector_precision, copies_from_the_other_precision_round_to_this_one)
{
    using T     = typename TypeParam::value_type;
    using Other = std::conditional_t<std::is_same_v<T, float>, double, float>;

    const std::vector<double> a = {1.5, -2.25, 3.125, 1e-3};
    const std::vector<Other>  source(a.begin(), a.end());
    const std::vector<T>      expected(source.begin(), source.end());

    auto copy = [&](BaseVector<T>& dst) {
        if constexpr(std::is_same_v<T, double>)
            dst.CopyFromFloat(*this->vec(source));
        else
            dst.CopyFromDouble(*this->vec(source));
    };

    auto v = this->vec(std::vector<T>());
    copy(*v);
    EXPECT_EQ(values_of(*v), expected);

    v = this->vec(std::vector<T>(a.size(), T(7)));
    copy(*v);
    EXPECT_EQ(values_of(*v), expected);
}

TYPED_TEST(backend_vector_precision, copies_from_an_empty_vector_of_the_other_precision)
{
    using T     = typename TypeParam::value_type;
    using Other = std::conditional_t<std::is_same_v<T, float>, double, float>;

    auto v = this->vec(std::vector<T>());
    if constexpr(std::is_same_v<T, double>)
        v->CopyFromFloat(*this->vec(std::vector<Other>()));
    else
        v->CopyFromDouble(*this->vec(std::vector<Other>()));
    EXPECT_EQ(v->GetSize(), 0);
}

// Operations for int only

template <typename P>
class backend_int_vector : public TypedVectorTest<P>
{
};

TYPED_TEST_SUITE(backend_int_vector, WithBackends<testing::Types<int>>, TypedName);

// Zero entries of values reset the CF map at their index, the others take the
// value of the CF map
TYPED_TEST(backend_int_vector, rspmis_update_cf_map_resets_or_packs_by_value)
{
    const std::vector<int> cf    = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int> index = {6, 1, 3, 0};
    const std::vector<int> value = {0, 9, 0, 9};

    auto map    = this->vec(cf);
    auto values = this->vec(value);
    map->RSPMISUpdateCFmap(*this->vec(index), values.get());

    EXPECT_EQ(values_of(*map), (std::vector<int>{1, 2, 3, 0, 5, 6, 0, 8}));
    EXPECT_EQ(values_of(*values), (std::vector<int>{0, 2, 0, 1}));
}

// The accelerator returns false and leaves the output untouched, so that
// LocalVector can move to the host and retry
TEST(backend_vector_fallback, unimplemented_operations_return_false_and_change_nothing)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    const std::vector<int> map = {0, 2, -1, 1, 0, 2};
    auto                   c   = make_vector(Backend::Accelerator, std::vector<double>(3, 7.0));
    auto f = make_vector(Backend::Accelerator, std::vector<double>(map.size(), 7.0));
    auto m = make_vector(Backend::Accelerator, map);

    EXPECT_FALSE(c->Restriction(*f, *m));
    EXPECT_EQ(values_of(*c), std::vector<double>(3, 7.0));
    EXPECT_FALSE(f->Prolongation(*c, *m));
    EXPECT_EQ(values_of(*f), std::vector<double>(map.size(), 7.0));
}

// The accelerator has no implementation of these operations for these value
// types and ends the program. One case per operation family.

#ifdef SUPPORT_COMPLEX
TEST(backend_vector_unsupported, complex_power_is_a_fatal_error_on_the_accelerator)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    auto v = make_vector(Backend::Accelerator, x<cfloat>());
    EXPECT_FATAL_ERROR(v->Power(2.0), "hip_vector\\.cpp");
}

TEST(backend_vector_unsupported, complex_sort_is_a_fatal_error)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    auto v      = make_vector(Backend::Accelerator, x<cfloat>());
    auto sorted = make_vector(Backend::Accelerator, std::vector<cfloat>(12));
    EXPECT_FATAL_ERROR(v->Sort(sorted.get(), nullptr), "hip_vector\\.cpp");
}
#endif

// Scale, Dot, DotNonConj, Norm, Asum and Amax
TEST(backend_vector_unsupported, integer_blas_operations_are_a_fatal_error_on_the_accelerator)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    auto v = make_vector(Backend::Accelerator, x<int>());
    EXPECT_FATAL_ERROR(static_cast<void>(v->Dot(*v)), "hip_vector\\.cpp");
}

TEST(backend_vector_unsupported, integer_random_values_are_a_fatal_error_on_the_accelerator)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    auto v = make_vector(Backend::Accelerator, std::vector<int>(12, 0));
    EXPECT_FATAL_ERROR(v->SetRandomUniform(1234, 0, 10), "hip_vector\\.cpp");
}

// Host only operations

template <typename P>
class host_vector_check : public TypedVectorTest<P>
{
};

TYPED_TEST_SUITE(host_vector_check, HostOnly<AllValueTypes>, TypedName);

TYPED_TEST(host_vector_check, accepts_finite_values)
{
    using T = typename TypeParam::value_type;

    EXPECT_TRUE(host_vector_of(from_ints<T>({1, 2, 3}))->Check());
    EXPECT_TRUE(host_vector_of(std::vector<T>())->Check());

    if constexpr(is_complex_v<T> || std::is_same_v<T, int> || std::is_same_v<T, int64_t>)
    {
        KNOWN_BUG("it compares |v| with numeric_limits<T>::infinity(), which is zero for "
                  "integers and complex numbers");
    }
    EXPECT_TRUE(host_vector_of(std::vector<T>{from_int<T>(1), T(0), from_int<T>(3)})->Check());
}

TYPED_TEST(host_vector_check, rejects_nan_and_infinity)
{
    using T = typename TypeParam::value_type;
    using R = real_t<T>;

    if constexpr(!std::numeric_limits<R>::has_quiet_NaN)
    {
        GTEST_SKIP() << type_name<T>() << " has no NaN and no infinity";
    }
    else
    {
        const R nan = std::numeric_limits<R>::quiet_NaN();
        const R inf = std::numeric_limits<R>::infinity();

        EXPECT_FALSE(
            host_vector_of(std::vector<T>{from_int<T>(1), T(nan), from_int<T>(3)})->Check());

        if constexpr(is_complex_v<T>)
        {
            KNOWN_BUG("numeric_limits<complex>::infinity() is zero, so no entry is infinite");
        }
        EXPECT_FALSE(
            host_vector_of(std::vector<T>{from_int<T>(1), T(-inf), from_int<T>(3)})->Check());
    }
}

// The accelerator has no random integers, see backend_vector_unsupported

template <typename P>
class host_vector_random : public TypedVectorTest<P>
{
};

TYPED_TEST_SUITE(host_vector_random, HostOnly<IntegerTypes>, TypedName);

TYPED_TEST(host_vector_random, uniform_integers_cover_the_range)
{
    using T     = typename TypeParam::value_type;
    const int n = 1000;

    auto v = this->vec(std::vector<T>(n, T(0)));
    v->SetRandomUniform(1234, T(0), T(10));
    std::vector<T> values = values_of(*v);

    EXPECT_GE(*std::min_element(values.begin(), values.end()), 0);
    EXPECT_LE(*std::max_element(values.begin(), values.end()), 10);

    KNOWN_BUG("it divides rand() by RAND_MAX in integer arithmetic, so almost every value is "
              "the lower bound");
    std::sort(values.begin(), values.end());
    EXPECT_GE(std::unique(values.begin(), values.end()) - values.begin(), 10);
}

template <typename T>
std::vector<T> file_values()
{
    if constexpr(std::is_same_v<T, float>)
        return {3, -1, 4.5f, 1e-7f, -5.25e12f};
    else if constexpr(std::is_same_v<T, double>)
        return {3, -1, 4.5, 1e-7, -5.25e12};
    else if constexpr(std::is_same_v<T, cfloat>)
        return {T(3, -1), T(4.5f, 1e-7f), T(-5.25e12f, 0)};
    else if constexpr(std::is_same_v<T, cdouble>)
        return {T(3, -1), T(4.5, 1e-7), T(-5.25e12, 0)};
    else if constexpr(std::is_same_v<T, bool>)
        return {true, false, true, true};
    else
        return {3, -1, 4, 0, 123456};
}

template <typename P>
class host_vector_files : public TypedVectorTest<P>
{
};

TYPED_TEST_SUITE(host_vector_files, HostOnly<AllValueTypes>, TypedName);

TYPED_TEST(host_vector_files, ascii_files_round_trip_with_six_significant_digits)
{
    using T = typename TypeParam::value_type;

    const std::string    file   = temp_file("txt");
    const std::vector<T> values = file_values<T>();

    host_vector_of(values)->WriteFileASCII(file);
    HostVector<T> ascii(descriptor());
    ascii.ReadFileASCII(file);
    EXPECT_EQ(values_of(ascii), values);
    std::remove(file.c_str());
}

template <typename P>
class host_vector_binary_files : public TypedVectorTest<P>
{
};

using BinaryFileTypes = testing::Types<float, double, int, int64_t>;
TYPED_TEST_SUITE(host_vector_binary_files, HostOnly<BinaryFileTypes>, TypedName);

// A header line, the version as int, the size as int64_t and the entries, as
// double for float vectors
TYPED_TEST(host_vector_binary_files, binary_files_round_trip_and_store_float_as_double)
{
    using T      = typename TypeParam::value_type;
    using Stored = std::conditional_t<std::is_floating_point_v<T>, double, T>;

    const std::string    file   = temp_file("bin");
    const std::string    header = "#rocALUTION binary vector file";
    const std::vector<T> values = file_values<T>();

    host_vector_of(values)->WriteFileBinary(file);
    {
        std::ifstream in(file, std::ios::binary | std::ios::ate);
        EXPECT_EQ(static_cast<size_t>(in.tellg()),
                  header.size() + 1 + sizeof(int) + sizeof(int64_t)
                      + values.size() * sizeof(Stored));
    }

    HostVector<T> binary(descriptor());
    binary.ReadFileBinary(file);
    EXPECT_EQ(values_of(binary), values);
    std::remove(file.c_str());
}

// Before version 3.0.0 the size was stored as int
TEST(host_vector, reads_binary_files_of_versions_before_3_with_32_bit_sizes)
{
    const std::string         file    = temp_file("bin");
    const std::vector<double> values  = {3, -1, 4.5, 1e-7};
    const int                 version = 20000;
    const int                 size    = static_cast<int>(values.size());
    {
        std::ofstream out(file, std::ios::binary);
        out << "#rocALUTION binary vector file\n";
        out.write(reinterpret_cast<const char*>(&version), sizeof(int));
        out.write(reinterpret_cast<const char*>(&size), sizeof(int));
        out.write(reinterpret_cast<const char*>(values.data()), sizeof(double) * values.size());
    }

    HostVector<double> v(descriptor());
    v.ReadFileBinary(file);
    EXPECT_EQ(values_of(v), values);
    std::remove(file.c_str());
}

// The aggregate of each fine index in index[start, end), numbered by first appearance
TEST(host_vector, extract_coarse_mapping)
{
    auto                   aggregates = host_vector(std::vector<int>{0, 2, -1, 1, 0, 2, 3, -1});
    const std::vector<int> index      = {5, 0, 2, 4, 6, 1, 3};

    std::vector<int> map(5, -7);
    int              size = 0;
    aggregates->ExtractCoarseMapping(1, 6, index.data(), 4, &size, map.data());
    EXPECT_EQ(size, 5);
    EXPECT_EQ(map, (std::vector<int>{0, -1, 0, 1, 2}));
}

// Appends the aggregates of index[start, end), each once, skipping -1
TEST(host_vector, extract_coarse_boundary)
{
    auto                   aggregates = host_vector(std::vector<int>{0, 2, -1, 1, 0, 2, 3, -1});
    const std::vector<int> index      = {5, 0, 2, 4, 6, 1, 3};

    std::vector<int> boundary = {7, 7, -7, -7, -7};
    int              size     = 2;
    aggregates->ExtractCoarseBoundary(1, 6, index.data(), 4, &size, boundary.data());
    EXPECT_EQ(size, 5);
    EXPECT_EQ(boundary, (std::vector<int>{7, 7, 0, 3, 2}));
}
