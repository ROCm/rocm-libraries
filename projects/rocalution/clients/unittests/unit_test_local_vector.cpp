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

// LocalVector on both backends and for every value type it is instantiated
// with: the dispatch to the backend vectors, the moves between the backends,
// the host fallbacks and the argument checks. The inputs are small integers,
// so that most results are exact.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "unit_test_local.hpp"
#include "unit_test_types.hpp"
#include "unit_test_vector_data.hpp"

#include "base/local_vector.hpp"

using namespace rocalution_unit;
using rocalution::LocalVector;

namespace
{

    std::vector<Backend> available_backends()
    {
        std::vector<Backend> backends = {Backend::Host};
        if(available(Backend::Accelerator))
        {
            backends.push_back(Backend::Accelerator);
        }
        return backends;
    }

    // LocalVector warns when it performs an operation on a host copy, and only then
    void expect_host_fallback(const std::string& log, const std::string& operation, Backend b)
    {
        const bool warned
            = log.find("*** warning: LocalVector::" + operation + "() is performed on the host")
              != std::string::npos;
        EXPECT_EQ(warned, b == Backend::Accelerator) << operation << " logged:\n" << log;
    }

    template <typename T>
    bool logged_check(const LocalVector<T>& v)
    {
        bool check = false;
        logged([&] { check = v.Check(); });
        return check;
    }

    template <typename... Lists>
    using Concat = typename concat_types<Lists...>::type;

#ifdef SUPPORT_COMPLEX
    using ComplexTypes = testing::Types<cfloat, cdouble>;
#else
    using ComplexTypes = testing::Types<>;
#endif

    // The value types and backends for which the backend vectors define an
    // operation; the others end the program
    using BlasTypes = Concat<WithBackends<FloatingTypes>, HostOnly<IntegerTypes>>;
    using AbsTypes  = Concat<WithBackends<RealTypes>, HostOnly<Concat<ComplexTypes, IntegerTypes>>>;
    using PowerTypes      = Concat<WithBackends<testing::Types<float, double, int64_t>>,
                                   HostOnly<Concat<testing::Types<int>, ComplexTypes>>>;
    using SortTypes       = WithBackends<testing::Types<float, double, int, int64_t, bool>>;
    using BinaryFileTypes = WithBackends<testing::Types<float, double, int, int64_t>>;
}

template <typename P>
class LocalVectorTest : public TypedBackendTest<P>
{
protected:
    template <typename V>
    static std::unique_ptr<LocalVector<V>> vec(const std::vector<V>& values)
    {
        return make_local_vector(P::backend, values);
    }

    static Backend backend()
    {
        return P::backend;
    }

    static std::string here()
    {
        return backend_name(P::backend);
    }
};

// Storage, data movement and host access, for every value type

template <typename P>
class local_vector_data : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_data, WithBackends<AllValueTypes>, TypedName);

TYPED_TEST(local_vector_data, allocate_gives_zeros_and_replaces_the_entries)
{
    using T = typename TypeParam::value_type;

    LocalVector<T> v;
    if(this->backend() == Backend::Accelerator)
    {
        v.MoveToAccelerator();
    }
    EXPECT_EQ(v.GetSize(), 0);

    v.Allocate("v", 5);
    EXPECT_EQ(backend_of(v), this->here());
    EXPECT_EQ(values_of(v), std::vector<T>(5, T(0)));

    v.Ones();
    v.Allocate("w", 3);
    EXPECT_EQ(backend_of(v), this->here());
    EXPECT_EQ(values_of(v), std::vector<T>(3, T(0)));

    v.Clear();
    EXPECT_EQ(v.GetSize(), 0);
    EXPECT_EQ(backend_of(v), this->here());

    const LocalVector<T>& c = v;
    EXPECT_EQ(&v.GetInterior(), &v);
    EXPECT_EQ(&c.GetInterior(), &v);
}

TYPED_TEST(local_vector_data, info_names_the_vector_its_size_precision_and_backend)
{
    using T = typename TypeParam::value_type;

    LocalVector<T> v;
    if(this->backend() == Backend::Accelerator)
    {
        v.MoveToAccelerator();
    }
    v.Allocate("my vector", 7);

    const std::string        info = logged([&] { v.Info(); });
    std::vector<std::string> parts
        = {"name=my vector;",
           "size=7;",
           this->backend() == Backend::Host ? "current=CPU" : "current=HIP"};
    if(!is_complex_v<T>)
    {
        parts.push_back("prec=" + std::to_string(8 * sizeof(T)) + "bit;");
    }
    for(const std::string& part : parts)
    {
        EXPECT_NE(info.find(part), std::string::npos) << part << " is not in " << info;
    }
}

TYPED_TEST(local_vector_data, fills_set_every_entry)
{
    using T = typename TypeParam::value_type;

    auto v = this->vec(x<T>());
    v->Zeros();
    EXPECT_EQ(values_of(*v), std::vector<T>(12, T(0)));
    v->Ones();
    EXPECT_EQ(values_of(*v), std::vector<T>(12, T(1)));
    v->SetValues(from_int<T>(-2));
    EXPECT_EQ(values_of(*v), std::vector<T>(12, from_int<T>(-2)));

    auto e = this->vec(std::vector<T>());
    e->Zeros();
    e->Ones();
    e->SetValues(from_int<T>(-2));
    EXPECT_EQ(e->GetSize(), 0);
}

// From a vector on either backend, into an empty vector and into one of the same size
TYPED_TEST(local_vector_data, copies_from_either_backend_keep_the_backend_of_the_copy)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();

    using Copy = std::function<void(LocalVector<T>&, const LocalVector<T>&)>;
    const std::vector<std::pair<std::string, Copy>> copies
        = {{"CopyFrom", [](LocalVector<T>& d, const LocalVector<T>& s) { d.CopyFrom(s); }},
           {"CopyFromAsync", [](LocalVector<T>& d, const LocalVector<T>& s) {
                d.CopyFromAsync(s);
                d.Sync();
            }}};

    for(Backend source : available_backends())
    {
        for(const auto& copy : copies)
        {
            SCOPED_TRACE(copy.first + " from " + backend_name(source));
            auto src = make_local_vector(source, a);

            auto empty = this->vec(std::vector<T>());
            copy.second(*empty, *src);
            EXPECT_EQ(backend_of(*empty), this->here());
            EXPECT_EQ(values_of(*empty), a);

            auto same_size = this->vec(b);
            copy.second(*same_size, *src);
            EXPECT_EQ(backend_of(*same_size), this->here());
            EXPECT_EQ(values_of(*same_size), a);
        }
    }
}

// Elements [2, 7) of this from elements [4, 9) of the source
TYPED_TEST(local_vector_data, copy_with_offsets_copies_only_the_range)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();

    auto v = this->vec(a);
    v->CopyFrom(*this->vec(b), 4, 2, 5);
    std::vector<T> expected = a;
    std::copy(b.begin() + 4, b.begin() + 9, expected.begin() + 2);
    EXPECT_EQ(values_of(*v), expected);
}

// CloneFrom takes the backend and the values, CloneBackend only the backend
TYPED_TEST(local_vector_data, clones_take_the_backend_of_the_source)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();

    for(Backend source : available_backends())
    {
        SCOPED_TRACE("from " + backend_name(source));
        auto src = make_local_vector(source, a);

        auto clone = this->vec(std::vector<T>());
        clone->CloneFrom(*src);
        EXPECT_EQ(backend_of(*clone), backend_name(source));
        EXPECT_EQ(values_of(*clone), a);

        auto moved = this->vec(b);
        moved->CloneBackend(*src);
        EXPECT_EQ(backend_of(*moved), backend_name(source));
        EXPECT_EQ(values_of(*moved), b);
    }
}

// A move to the backend the vector is on does nothing. An asynchronous move
// completes on Sync.
TYPED_TEST(local_vector_data, moves_keep_the_values)
{
    using T = typename TypeParam::value_type;
    REQUIRE_BACKEND(Backend::Accelerator);

    const std::vector<T> a = x<T>();
    auto                 v = this->vec(a);
    v->Sync();

    v->MoveToHost();
    v->MoveToHost();
    EXPECT_EQ(backend_of(*v), backend_name(Backend::Host));
    EXPECT_EQ(values_of(*v), a);

    v->MoveToAccelerator();
    v->MoveToAccelerator();
    EXPECT_EQ(backend_of(*v), backend_name(Backend::Accelerator));
    EXPECT_EQ(values_of(*v), a);

    v->MoveToHost();
    v->MoveToAcceleratorAsync();
    v->Sync();
    EXPECT_EQ(backend_of(*v), backend_name(Backend::Accelerator));
    EXPECT_EQ(values_of(*v), a);

    v->MoveToAcceleratorAsync();
    v->Sync();
    EXPECT_EQ(backend_of(*v), backend_name(Backend::Accelerator));
    EXPECT_EQ(values_of(*v), a);

    LocalVector<T> e;
    e.MoveToAccelerator();
    EXPECT_EQ(backend_of(e), backend_name(Backend::Accelerator));
    e.MoveToHost();
    EXPECT_EQ(backend_of(e), backend_name(Backend::Host));
    EXPECT_EQ(e.GetSize(), 0);
}

// The data pointers belong to the backend. SetDataPtr takes the data and
// LeaveDataPtr gives it away.
TYPED_TEST(local_vector_data, data_pointers_move_the_data_between_vectors)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();
    const int64_t        n = static_cast<int64_t>(a.size());

    auto u    = this->vec(a);
    T*   data = nullptr;
    u->LeaveDataPtr(&data);
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(u->GetSize(), 0);

    auto w = this->vec(std::vector<T>(a.size(), T(0)));
    w->CopyFromData(data);
    EXPECT_EQ(values_of(*w), a);

    this->vec(b)->CopyToData(data);
    auto s = this->vec(std::vector<T>());
    s->SetDataPtr(&data, "s", n);
    EXPECT_EQ(data, nullptr);
    EXPECT_EQ(backend_of(*s), this->here());
    EXPECT_EQ(values_of(*s), b);

    T* none = nullptr;
    s->SetDataPtr(&none, "none", 0);
    EXPECT_EQ(s->GetSize(), 0);

    T unused = from_int<T>(5);
    s->CopyFromData(&unused);
    s->CopyToData(&unused);
    s->CopyToHostData(nullptr);
    EXPECT_EQ(s->GetSize(), 0);
    EXPECT_EQ(unused, from_int<T>(5));
}

// The values are host arrays on both backends. The accelerator copies them
// asynchronously, and LocalVector::Sync does not wait for the copies, so the
// test waits for the whole device.
TYPED_TEST(local_vector_data, continuous_values_are_host_arrays)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>();
    auto                 v = this->vec(a);

    std::unique_ptr<T[]> part(new T[5]);
    v->GetContinuousValues(3, 8, part.get());
    rocalution::_rocalution_sync();
    EXPECT_EQ(std::vector<T>(part.get(), part.get() + 5),
              std::vector<T>(a.begin() + 3, a.begin() + 8));

    const std::vector<T> b      = from_ints<T>({10, 21, 30, 41});
    auto                 values = array_of(b);
    v->SetContinuousValues(6, 10, values.get());
    rocalution::_rocalution_sync();
    std::vector<T> expected = a;
    std::copy(b.begin(), b.end(), expected.begin() + 6);
    EXPECT_EQ(values_of(*v), expected);

    v->GetContinuousValues(4, 4, nullptr);
    v->SetContinuousValues(4, 4, nullptr);
    EXPECT_EQ(values_of(*v), expected);
}

TYPED_TEST(local_vector_data, index_values_are_read_and_written_at_the_indices)
{
    using T = typename TypeParam::value_type;

    const std::vector<T>   a     = x<T>();
    const std::vector<int> index = {9, 2, 0, 11, 5};
    const std::vector<T>   b     = from_ints<T>({10, 21, 30, 41, 50});

    auto v      = this->vec(a);
    auto idx    = this->vec(index);
    auto values = this->vec(std::vector<T>(index.size(), T(0)));
    v->GetIndexValues(*idx, values.get());
    EXPECT_EQ(values_of(*values), (std::vector<T>{a[9], a[2], a[0], a[11], a[5]}));

    std::vector<T> set = a;
    for(size_t k = 0; k < index.size(); ++k)
    {
        set[index[k]] = b[k];
    }
    v->SetIndexValues(*idx, *this->vec(b));
    EXPECT_EQ(values_of(*v), set);
}

// Forward: element i moves to perm[i]. Backward: element i comes from perm[i].
TYPED_TEST(local_vector_data, permutations_move_entries_forward_and_backward)
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

    auto e     = this->vec(std::vector<T>());
    auto none  = this->vec(std::vector<int>());
    auto empty = this->vec(std::vector<T>());
    e->Permute(*none);
    e->PermuteBackward(*none);
    e->CopyFromPermute(*empty, *none);
    e->CopyFromPermuteBackward(*empty, *none);
    EXPECT_EQ(e->GetSize(), 0);
}

// The file is read on the host; the vector stays on its backend and takes the
// file name as its name
TYPED_TEST(local_vector_data, ascii_files_round_trip_and_keep_the_backend)
{
    using T = typename TypeParam::value_type;

    const std::string    file = temp_file("txt");
    const std::vector<T> a    = x<T>();

    logged([&] { this->vec(a)->WriteFileASCII(file); });
    auto v = this->vec(from_ints<T>({1, 2, 3}));
    logged([&] { v->ReadFileASCII(file); });

    EXPECT_EQ(backend_of(*v), this->here());
    EXPECT_EQ(values_of(*v), a);
    const std::string info = logged([&] { v->Info(); });
    EXPECT_NE(info.find("name=" + file + ";"), std::string::npos) << info;
    std::remove(file.c_str());
}

// The accelerator checks a host copy
TYPED_TEST(local_vector_data, check_accepts_finite_values_and_rejects_nan_and_infinity)
{
    using T = typename TypeParam::value_type;
    using R = real_t<T>;

    auto        v     = this->vec(from_ints<T>({1, 2, 3}));
    bool        check = false;
    std::string log   = logged([&] { check = v->Check(); });
    EXPECT_TRUE(check);
    expect_host_fallback(log, "Check", this->backend());
    EXPECT_TRUE(logged_check(*this->vec(std::vector<T>())));

    if constexpr(std::numeric_limits<R>::has_quiet_NaN)
    {
        const R nan = std::numeric_limits<R>::quiet_NaN();
        const R inf = std::numeric_limits<R>::infinity();
        EXPECT_FALSE(
            logged_check(*this->vec(std::vector<T>{from_int<T>(1), T(nan), from_int<T>(3)})));
        if constexpr(!is_complex_v<T>)
        {
            EXPECT_FALSE(
                logged_check(*this->vec(std::vector<T>{from_int<T>(1), -inf, from_int<T>(3)})));
        }
    }

    if constexpr(is_complex_v<T> || std::is_same_v<T, int> || std::is_same_v<T, int64_t>)
    {
        KNOWN_BUG("HostVector::Check compares |v| with numeric_limits<T>::infinity(), which is "
                  "zero for integers and complex numbers");
    }
    EXPECT_TRUE(logged_check(*this->vec(std::vector<T>{from_int<T>(1), T(0), from_int<T>(3)})));
    if constexpr(is_complex_v<T>)
    {
        const R inf = std::numeric_limits<R>::infinity();
        EXPECT_FALSE(
            logged_check(*this->vec(std::vector<T>{from_int<T>(1), T(-inf), from_int<T>(3)})));
    }
}

// operator[] accesses the host data; on the accelerator it is an assert, see
// local_vector_preconditions

template <typename P>
class local_vector_host_access : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_host_access, HostOnly<AllValueTypes>, TypedName);

TYPED_TEST(local_vector_host_access, subscripts_read_and_write_the_entries)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();
    auto                 v = this->vec(a);

    const LocalVector<T>& c = *v;
    std::vector<T>        read(a.size());
    for(size_t i = 0; i < a.size(); ++i)
    {
        read[i] = c[static_cast<int64_t>(i)];
    }
    EXPECT_EQ(read, a);

    for(size_t i = 0; i < b.size(); ++i)
    {
        (*v)[static_cast<int64_t>(i)] = b[i];
    }
    EXPECT_EQ(values_of(*v), b);
}

// Arithmetic, for the value types other than bool

template <typename P>
class local_vector : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector, WithBackends<NumericTypes>, TypedName);

TYPED_TEST(local_vector, updates_combine_the_vectors_entrywise)
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

    auto e = this->vec(std::vector<T>()), f = this->vec(std::vector<T>());
    e->AddScale(*f, alpha);
    e->ScaleAdd(alpha, *f);
    e->ScaleAddScale(alpha, *f, beta);
    e->ScaleAdd2(alpha, *f, beta, *f, gamma);
    e->PointWiseMult(*f);
    e->PointWiseMult(*f, *f);
    EXPECT_EQ(e->GetSize(), 0);
}

// Elements [2, 7) of this from elements [4, 9) of the source
TYPED_TEST(local_vector, scale_add_scale_with_offsets_updates_only_the_range)
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

// The scans return the last entry of the result. Without an argument they scan in place.
TYPED_TEST(local_vector, reduce_and_scans_sum_the_entries)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>();
    std::vector<T>       inclusive(a.size()), exclusive(a.size(), T(0));
    std::partial_sum(a.begin(), a.end(), inclusive.begin());
    std::partial_sum(a.begin(), a.end() - 1, exclusive.begin() + 1);

    EXPECT_EQ(this->vec(a)->Reduce(), std::accumulate(a.begin(), a.end(), T(0)));
    EXPECT_EQ(this->vec(std::vector<T>())->Reduce(), T(0));

    auto v = this->vec(std::vector<T>(a.size(), T(0)));
    EXPECT_EQ(v->InclusiveSum(*this->vec(a)), inclusive.back());
    EXPECT_EQ(values_of(*v), inclusive);
    EXPECT_EQ(v->ExclusiveSum(*this->vec(a)), exclusive.back());
    EXPECT_EQ(values_of(*v), exclusive);

    v = this->vec(a);
    EXPECT_EQ(v->InclusiveSum(), inclusive.back());
    EXPECT_EQ(values_of(*v), inclusive);

    v = this->vec(a);
    EXPECT_EQ(v->ExclusiveSum(), exclusive.back());
    EXPECT_EQ(values_of(*v), exclusive);
}

// Restriction sums the fine entries of each coarse entry, prolongation copies
// them back; fine entries mapped to -1 are skipped and prolong to zero. The
// accelerator has neither and retries on the host.
TYPED_TEST(local_vector, restriction_sums_and_prolongation_copies_back)
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
    std::vector<T> prolonged(fine.size(), T(0));
    for(size_t i = 0; i < fine.size(); ++i)
    {
        prolonged[i] = map[i] >= 0 ? coarse[map[i]] : T(0);
    }

    auto m = this->vec(map);
    auto c = this->vec(std::vector<T>(4, from_int<T>(7)));
    auto f = this->vec(fine);

    std::string log = logged([&] { c->Restriction(*f, *m); });
    EXPECT_EQ(backend_of(*c), this->here());
    EXPECT_EQ(values_of(*c), coarse);
    expect_host_fallback(log, "Restriction", this->backend());

    auto p = this->vec(std::vector<T>(fine.size(), from_int<T>(7)));
    log    = logged([&] { p->Prolongation(*c, *m); });
    EXPECT_EQ(backend_of(*p), this->here());
    EXPECT_EQ(values_of(*p), prolonged);
    expect_host_fallback(log, "Prolongation", this->backend());

    auto e    = this->vec(std::vector<T>());
    auto none = this->vec(std::vector<int>());
    e->Restriction(*this->vec(std::vector<T>()), *none);
    e->Prolongation(*this->vec(std::vector<T>()), *none);
    EXPECT_EQ(e->GetSize(), 0);
}

TYPED_TEST(local_vector, add_index_values_adds_at_the_indices)
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

// Scale, Dot and DotNonConj, which the accelerator does not define for integers

template <typename P>
class local_vector_blas : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_blas, BlasTypes, TypedName);

TYPED_TEST(local_vector_blas, scale_multiplies_every_entry)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a     = x<T>();
    const T              alpha = from_int<T>(3);

    auto v = this->vec(a);
    v->Scale(alpha);
    EXPECT_EQ(values_of(*v), elementwise(a, a, [&](T p, T) { return p * alpha; }));

    auto e = this->vec(std::vector<T>());
    e->Scale(alpha);
    EXPECT_EQ(e->GetSize(), 0);
}

TYPED_TEST(local_vector_blas, dot_conjugates_this_vector_and_dot_non_conj_does_not)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a = x<T>(), b = y<T>();
    T                    dot = T(0), dot_non_conj = T(0);
    for(size_t i = 0; i < a.size(); ++i)
    {
        dot += conj_of(a[i]) * b[i];
        dot_non_conj += a[i] * b[i];
    }

    auto v = this->vec(a), w = this->vec(b);
    EXPECT_EQ(v->Dot(*w), dot);
    EXPECT_EQ(v->DotNonConj(*w), dot_non_conj);

    auto e = this->vec(std::vector<T>()), f = this->vec(std::vector<T>());
    EXPECT_EQ(e->Dot(*f), T(0));
    EXPECT_EQ(e->DotNonConj(*f), T(0));
}

// Norm, for the floating point value types

template <typename P>
class local_vector_norm : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_norm, WithBackends<FloatingTypes>, TypedName);

TYPED_TEST(local_vector_norm, norm_is_the_euclidean_norm)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a     = x<T>();
    double               norm2 = 0;
    for(const T& e : a)
    {
        norm2 += std::norm(std::complex<double>(e));
    }

    expect_near(this->vec(a)->Norm(), static_cast<T>(std::sqrt(norm2)));
    EXPECT_EQ(this->vec(std::vector<T>())->Norm(), T(0));
}

// Asum and Amax, which the accelerator defines only for real floating point types

template <typename P>
class local_vector_abs : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_abs, AbsTypes, TypedName);

TYPED_TEST(local_vector_abs, asum_sums_the_magnitudes)
{
    using T = typename TypeParam::value_type;

    const std::vector<T> a    = x<T>();
    const T              asum = this->vec(a)->Asum();

    EXPECT_EQ(this->vec(std::vector<T>())->Asum(), T(0));

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
}

// The index is zero based; -1 for an empty vector
TYPED_TEST(local_vector_abs, amax_returns_the_index_and_the_largest_magnitude)
{
    using T = typename TypeParam::value_type;

    T value = from_int<T>(5);
    EXPECT_EQ(this->vec(std::vector<T>())->Amax(value), -1);
    EXPECT_EQ(value, T(0));

    for(int position : {0, 5, 11})
    {
        for(int sign : {1, -1})
        {
            SCOPED_TRACE("at " + std::to_string(position) + " with sign " + std::to_string(sign));

            std::vector<T> a = x<T>();
            a[position]      = from_int<T>(sign * 20);

            T found = T(0);
            EXPECT_EQ(this->vec(a)->Amax(found), position);
            expect_near(found, magnitude(a[position]));
        }
    }
}

// Power: the accelerator has no complex and no int version

template <typename P>
class local_vector_power : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_power, PowerTypes, TypedName);

TYPED_TEST(local_vector_power, power_raises_every_entry)
{
    using T = typename TypeParam::value_type;

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

    auto e = this->vec(std::vector<T>());
    e->Power(2);
    EXPECT_EQ(e->GetSize(), 0);
}

// Sort, for the value types with an order

template <typename P>
class local_vector_sort : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_sort, SortTypes, TypedName);

// The sorted vector and the permutation may be longer than the vector; their
// remaining entries stay
TYPED_TEST(local_vector_sort, sort_orders_the_values_and_returns_the_permutation)
{
    using T = typename TypeParam::value_type;

    std::vector<T> a;
    for(int k : {6, -2, 8, 3, -10, 18, 4, -12, 11, 0})
    {
        a.push_back(std::is_floating_point_v<T> ? static_cast<T>(k / 2.0) : from_int<T>(k));
    }
    const size_t n    = a.size();
    const T      tail = from_int<T>(7);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int i, int j) { return a[i] < a[j]; });
    std::vector<T> sorted_values(n);
    for(size_t i = 0; i < n; ++i)
    {
        sorted_values[i] = a[order[i]];
    }
    std::vector<T> expected = sorted_values;
    expected.push_back(tail);

    auto v      = this->vec(a);
    auto sorted = this->vec(std::vector<T>(n + 1, tail));
    v->Sort(sorted.get());
    EXPECT_EQ(values_of(*sorted), expected);

    auto perm = this->vec(std::vector<int>(n + 1, -1));
    sorted    = this->vec(std::vector<T>(n + 1, tail));
    v->Sort(sorted.get(), perm.get());
    EXPECT_EQ(values_of(*sorted), expected);

    std::vector<int> p = values_of(*perm);
    EXPECT_EQ(p.back(), -1);
    p.pop_back();
    for(size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(a[p[i]], sorted_values[i]) << "at " << i;
    }
    std::sort(p.begin(), p.end());
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 0);
    EXPECT_EQ(p, identity);

    auto e = this->vec(std::vector<T>());
    e->Sort(sorted.get(), perm.get());
    EXPECT_EQ(values_of(*sorted), expected);
}

// Random values, for the floating point value types. Only the real parts are
// checked: the accelerator also draws the imaginary parts, the host leaves them zero.

template <typename P>
class local_vector_random : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_random, WithBackends<FloatingTypes>, TypedName);

TYPED_TEST(local_vector_random, uniform_values_cover_the_range_and_repeat_with_the_seed)
{
    using T     = typename TypeParam::value_type;
    const int n = 2000;

    auto u = this->vec(std::vector<T>(n, T(0)));
    u->SetRandomUniform(1234, T(-2), T(3));
    const std::vector<T> values = values_of(*u);

    double min = 3, max = -2;
    for(const T& e : values)
    {
        min = std::min(min, static_cast<double>(std::real(e)));
        max = std::max(max, static_cast<double>(std::real(e)));
    }
    EXPECT_GE(min, -2.0);
    EXPECT_LT(min, -1.9);
    EXPECT_LE(max, 3.0);
    EXPECT_GT(max, 2.9);

    auto again = this->vec(std::vector<T>(n, T(0)));
    again->SetRandomUniform(1234, T(-2), T(3));
    EXPECT_EQ(values_of(*again), values);

    auto e = this->vec(std::vector<T>());
    e->SetRandomUniform(1234, T(-2), T(3));
    EXPECT_EQ(e->GetSize(), 0);
}

// The third argument is the standard deviation. The bounds are more than four
// standard errors: sigma / sqrt(n) = 0.045 for the mean, about sigma / sqrt(2 n)
// = 0.032 for the sample standard deviation. A variance of 2 would give a
// standard deviation of 1.41.
TYPED_TEST(local_vector_random, normal_values_have_the_mean_and_deviation_and_repeat_with_the_seed)
{
    using T     = typename TypeParam::value_type;
    const int n = 2000;

    auto g = this->vec(std::vector<T>(n, T(0)));
    g->SetRandomNormal(4321, T(3), T(2));
    const std::vector<T> values = values_of(*g);

    double sum = 0;
    for(const T& e : values)
    {
        sum += std::real(e);
    }
    const double mean = sum / n;
    EXPECT_NEAR(mean, 3.0, 0.2);

    double squares = 0;
    for(const T& e : values)
    {
        squares += (std::real(e) - mean) * (std::real(e) - mean);
    }
    EXPECT_NEAR(std::sqrt(squares / (n - 1)), 2.0, 0.15);

    auto again = this->vec(std::vector<T>(n, T(0)));
    again->SetRandomNormal(4321, T(3), T(2));
    EXPECT_EQ(values_of(*again), values);

    auto e = this->vec(std::vector<T>());
    e->SetRandomNormal(4321, T(3), T(2));
    EXPECT_EQ(e->GetSize(), 0);
}

// The accelerator has no random integers

template <typename P>
class local_vector_random_integers : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_random_integers, HostOnly<IntegerTypes>, TypedName);

TYPED_TEST(local_vector_random_integers, uniform_integers_cover_the_range)
{
    using T     = typename TypeParam::value_type;
    const int n = 1000;

    auto v = this->vec(std::vector<T>(n, T(0)));
    v->SetRandomUniform(1234, T(0), T(10));
    std::vector<T> values = values_of(*v);

    EXPECT_GE(*std::min_element(values.begin(), values.end()), 0);
    EXPECT_LE(*std::max_element(values.begin(), values.end()), 10);

    KNOWN_BUG("HostVector::SetRandomUniform divides rand() by RAND_MAX in integer arithmetic, so "
              "almost every value is the lower bound");
    std::sort(values.begin(), values.end());
    EXPECT_GE(std::unique(values.begin(), values.end()) - values.begin(), 10);
}

// Binary files, which store float as double and have no complex and bool version

template <typename P>
class local_vector_binary_files : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_binary_files, BinaryFileTypes, TypedName);

TYPED_TEST(local_vector_binary_files, binary_files_round_trip_and_keep_the_backend)
{
    using T = typename TypeParam::value_type;

    const std::string    file = temp_file("bin");
    const std::vector<T> a    = x<T>();

    logged([&] { this->vec(a)->WriteFileBinary(file); });
    auto v = this->vec(from_ints<T>({1, 2, 3}));
    logged([&] { v->ReadFileBinary(file); });

    EXPECT_EQ(backend_of(*v), this->here());
    EXPECT_EQ(values_of(*v), a);
    const std::string info = logged([&] { v->Info(); });
    EXPECT_NE(info.find("name=" + file + ";"), std::string::npos) << info;
    std::remove(file.c_str());
}

// Copies between float and double

template <typename P>
class local_vector_precision : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_precision, WithBackends<RealTypes>, TypedName);

TYPED_TEST(local_vector_precision, copies_from_the_other_precision_round_to_this_one)
{
    using T     = typename TypeParam::value_type;
    using Other = std::conditional_t<std::is_same_v<T, float>, double, float>;

    const std::vector<double> a = {1.5, -2.25, 3.125, 1e-3};
    const std::vector<Other>  source(a.begin(), a.end());
    const std::vector<T>      expected(source.begin(), source.end());

    auto copy = [&](LocalVector<T>& dst) {
        if constexpr(std::is_same_v<T, double>)
            dst.CopyFromFloat(*this->vec(source));
        else
            dst.CopyFromDouble(*this->vec(source));
    };

    auto v = this->vec(std::vector<T>());
    copy(*v);
    EXPECT_EQ(backend_of(*v), this->here());
    EXPECT_EQ(values_of(*v), expected);

    v = this->vec(std::vector<T>(a.size(), T(7)));
    copy(*v);
    EXPECT_EQ(values_of(*v), expected);
}

// Coarse mappings of aggregates, for int vectors. The accelerator extracts them
// from a host copy.

template <typename P>
class local_vector_coarse : public LocalVectorTest<P>
{
};

TYPED_TEST_SUITE(local_vector_coarse, WithBackends<testing::Types<int>>, TypedName);

// The aggregate of each fine index in index[start, end), numbered by first appearance
TYPED_TEST(local_vector_coarse, extract_coarse_mapping_numbers_the_aggregates)
{
    auto                   aggregates = this->vec(std::vector<int>{0, 2, -1, 1, 0, 2, 3, -1});
    const std::vector<int> index      = {5, 0, 2, 4, 6, 1, 3};

    std::vector<int>  map(5, -7);
    int               size = 0;
    const std::string log  = logged(
        [&] { aggregates->ExtractCoarseMapping(1, 6, index.data(), 4, &size, map.data()); });
    EXPECT_EQ(size, 5);
    EXPECT_EQ(map, (std::vector<int>{0, -1, 0, 1, 2}));
    EXPECT_EQ(backend_of(*aggregates), this->here());
    expect_host_fallback(log, "ExtractCoarseMapping", this->backend());
}

// Appends the aggregates of index[start, end), each once, skipping -1
TYPED_TEST(local_vector_coarse, extract_coarse_boundary_appends_each_aggregate_once)
{
    auto                   aggregates = this->vec(std::vector<int>{0, 2, -1, 1, 0, 2, 3, -1});
    const std::vector<int> index      = {5, 0, 2, 4, 6, 1, 3};

    std::vector<int>  boundary = {7, 7, -7, -7, -7};
    int               size     = 2;
    const std::string log      = logged(
        [&] { aggregates->ExtractCoarseBoundary(1, 6, index.data(), 4, &size, boundary.data()); });
    EXPECT_EQ(size, 5);
    EXPECT_EQ(boundary, (std::vector<int>{7, 7, 0, 3, 2}));
    EXPECT_EQ(backend_of(*aggregates), this->here());
    expect_host_fallback(log, "ExtractCoarseBoundary", this->backend());
}

// Moves and allocation, independent of the value type

TEST(local_vector_moves, without_an_accelerator_moves_stay_on_the_host)
{
    DescriptorGuard guard;
    rocalution::_get_backend_descriptor()->accelerator = false;

    const std::vector<double> a = x<double>();
    auto                      v = make_local_vector(Backend::Host, a);

    v->MoveToAccelerator();
    EXPECT_EQ(backend_of(*v), backend_name(Backend::Host));
    v->MoveToAcceleratorAsync();
    v->Sync();
    EXPECT_EQ(backend_of(*v), backend_name(Backend::Host));
    v->MoveToHostAsync();
    v->Sync();
    v->MoveToHost();
    EXPECT_EQ(backend_of(*v), backend_name(Backend::Host));
    EXPECT_EQ(values_of(*v), a);
}

TEST(local_vector_moves, async_move_of_a_host_vector_to_the_host_changes_nothing)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    const std::vector<double> a = x<double>();
    auto                      v = make_local_vector(Backend::Host, a);
    v->MoveToHostAsync();
    v->Sync();
    EXPECT_EQ(backend_of(*v), backend_name(Backend::Host));
    EXPECT_EQ(values_of(*v), a);
}

TEST(local_vector_moves, sync_after_move_to_host_async_ends_on_the_host)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    KNOWN_BUG("Sync after MoveToHostAsync also runs the MoveToAcceleratorAsync branch, which "
              "deletes the host copy and leaves the vector without data, as for LocalMatrix");
    expect_in_child([] {
        const std::vector<double> a = x<double>();
        auto                      v = make_local_vector(Backend::Accelerator, a);
        v->MoveToHostAsync();
        v->Sync();
        EXPECT_EQ(backend_of(*v), backend_name(Backend::Host));
        EXPECT_EQ(values_of(*v), a);
    });
}

TEST(local_vector_allocation, allocate_with_size_zero_empties_the_vector)
{
    auto v = make_local_vector(Backend::Host, x<double>());
    v->Allocate("empty", 0);
    const std::string info = logged([&] { v->Info(); });
    EXPECT_NE(info.find("name=empty;"), std::string::npos) << info;

    KNOWN_BUG("Allocate with size zero keeps the entries of the vector");
    EXPECT_EQ(v->GetSize(), 0);
}

TEST(local_vector_preconditions, violations_end_the_program)
{
    LocalVector<double> e;
    EXPECT_ASSERT_DEATH(e.Allocate("v", -1), "size >= 0");
    double* null = nullptr;
    EXPECT_ASSERT_DEATH(e.SetDataPtr(&null, "v", 3), "\\*ptr != NULL");

    auto a = make_local_vector(Backend::Host, std::vector<double>(4, 1.0));
    auto b = make_local_vector(Backend::Host, std::vector<double>(3, 1.0));
    auto c = make_local_vector(Backend::Host, std::vector<double>(4, 2.0));
    EXPECT_ASSERT_DEATH(static_cast<void>(a->Dot(*b)), "x.GetSize");
    EXPECT_ASSERT_DEATH(static_cast<void>((*a)[4]), "i < vector_host_->size_");
    EXPECT_ASSERT_DEATH(
        {
            a->CopyFromAsync(*c);
            a->CopyFromAsync(*c);
        },
        "asyncf_ == false");

    if(available(Backend::Accelerator))
    {
        auto d = make_local_vector(Backend::Accelerator, std::vector<double>(4, 1.0));
        EXPECT_ASSERT_DEATH(a->AddScale(*d, 1.0), "x.vector_ == x.vector_host_");
        EXPECT_ASSERT_DEATH(static_cast<void>((*d)[0]), "vector_host_ != NULL");
    }
}
