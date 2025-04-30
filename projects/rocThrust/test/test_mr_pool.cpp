/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/mr/pool.h>
#include <thrust/mr/sync_pool.h>
#include <thrust/mr/new.h>


#include "test_header.hpp"

template<typename T>
struct reference
{
    using type = T &;
};

template<>
struct reference<void>
{
    using type = void;
};

struct unit {};

template<typename T>
struct tracked_pointer : thrust::iterator_facade<
                            tracked_pointer<T>,
                            T,
                            thrust::host_system_tag,
                            thrust::random_access_traversal_tag,
                            typename reference<T>::type,
                            std::ptrdiff_t
                         >
{
    using raw_pointer = T *;

    std::size_t id;
    std::size_t size;
    std::size_t alignment;
    std::size_t offset;
    void * ptr;

    __host__ __device__
    explicit tracked_pointer(T * ptr = nullptr) : id(), size(), alignment(), offset(), ptr(ptr)
    {
    }

    __host__ __device__
    ~tracked_pointer()
    {
    }

    template<typename U>
    operator tracked_pointer<U>() const
    {
        tracked_pointer<U> ret;
        ret.id = id;
        ret.size = size;
        ret.alignment = alignment;
        ret.offset = offset;
        ret.ptr = ptr;
        return ret;
    }

    __host__ __device__
    std::ptrdiff_t distance_to(const tracked_pointer & other) const
    {
        return static_cast<T *>(other.ptr) - static_cast<T *>(ptr);
    }

    __host__ __device__
    T * get() const
    {
        return static_cast<T *>(ptr);
    }

    // globally qualified, because MSVC somehow prefers the name from the dependent base
    // of this class over the `reference` template that's visible in the global namespace of this file...
    __host__ __device__
    typename ::reference<T>::type dereference() const
    {
        return *get();
    }

    __host__ __device__
    void increment()
    {
        advance(1);
    }

    __host__ __device__
    void decrement()
    {
        advance(-1);
    }

    __host__ __device__
    void advance(std::ptrdiff_t diff)
    {
        ptr = get() + diff;
        offset += diff * sizeof(T);
    }

    __host__ __device__
    bool equal(const tracked_pointer & other) const
    {
        return id == other.id && size == other.size && alignment == other.alignment && offset == other.offset && ptr == other.ptr;
    }
};

class tracked_resource final : public thrust::mr::memory_resource<tracked_pointer<void> >
{
public:
    tracked_resource() : id_to_allocate(0), id_to_deallocate(0)
    {
    }

    ~tracked_resource()
    {
        EXPECT_EQ(id_to_allocate, 0u);
        EXPECT_EQ(id_to_deallocate, 0u);
    }

    virtual tracked_pointer<void> do_allocate(std::size_t n, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        EXPECT_EQ(static_cast<bool>(id_to_allocate), true);

        void * raw = upstream.do_allocate(n, alignment);
        tracked_pointer<void> ret(raw);
        ret.id = id_to_allocate;
        ret.size = n;
        ret.alignment = alignment;

        id_to_allocate = 0;

        return ret;
    }

    virtual void do_deallocate(tracked_pointer<void> p, std::size_t n, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        ASSERT_EQ(p.size, n);
        ASSERT_EQ(p.alignment, alignment);

        if (id_to_deallocate != 0)
        {
            ASSERT_EQ(p.id, id_to_deallocate);
            id_to_deallocate = 0;
        }

        upstream.do_deallocate(p.ptr, n, alignment);
    }

    std::size_t id_to_allocate;
    std::size_t id_to_deallocate;

private:
    thrust::mr::new_delete_resource upstream;
};

template<template<typename> class PoolTemplate>
void TestPool()
{
    tracked_resource upstream;

    upstream.id_to_allocate = -1u;

    using Pool = PoolTemplate<tracked_resource>;

    thrust::mr::pool_options opts = Pool::get_default_options();
    opts.cache_oversized = false;

    // avoid having the destructor run when an assertion failure is raised
    // (the destructor will try to release, which in turn calls do_deallocate,
    // which may fail with an assertion failure exception...)
    Pool * pool = new Pool(&upstream, opts);

    upstream.id_to_allocate = 1;

    // first allocation
    tracked_pointer<void> a1 = pool->do_allocate(12, THRUST_MR_DEFAULT_ALIGNMENT);
    ASSERT_EQ(a1.id, 1u);

    // due to chunking, the above allocation should be enough for the next one too
    tracked_pointer<void> a2 = pool->do_allocate(16, THRUST_MR_DEFAULT_ALIGNMENT);
    ASSERT_EQ(a2.id, 1u);

    // deallocating and allocating back should give the same resource back
    pool->do_deallocate(a1, 12, THRUST_MR_DEFAULT_ALIGNMENT);
    tracked_pointer<void> a3 = pool->do_allocate(12, THRUST_MR_DEFAULT_ALIGNMENT);
    ASSERT_EQ(a1.id, a3.id);
    ASSERT_EQ(a1.size, a3.size);
    ASSERT_EQ(a1.alignment, a3.alignment);
    ASSERT_EQ(a1.offset, a3.offset);

    // allocating over-aligned memory should give non-cached results
    // unlike with the disjoint version, nothing sensible can be said about the chunk size
    upstream.id_to_allocate = 2;
    tracked_pointer<void> a4 = pool->do_allocate(32, THRUST_MR_DEFAULT_ALIGNMENT * 2);
    ASSERT_EQ(a4.id, 2u);
    ASSERT_EQ(a4.alignment, (std::size_t)THRUST_MR_DEFAULT_ALIGNMENT * 2);

    // and deallocating it should return it back to upstream
    upstream.id_to_deallocate = 2;
    pool->do_deallocate(a4, 32u, THRUST_MR_DEFAULT_ALIGNMENT * 2);
    ASSERT_EQ(upstream.id_to_deallocate, 0u);

    // release actually returns properly sized memory to upstream
    upstream.id_to_deallocate = 1;
    pool->release();
    ASSERT_EQ(upstream.id_to_deallocate, 0u);

    // and does the same for oversized/overaligned memory
    upstream.id_to_allocate = 3;
    tracked_pointer<void> a5 = pool->do_allocate(1024, THRUST_MR_DEFAULT_ALIGNMENT * 2);
    ASSERT_EQ(upstream.id_to_allocate, 0u);
    ASSERT_EQ(a5.id, 3u);

    upstream.id_to_deallocate = 3;
    pool->release();
    ASSERT_EQ(upstream.id_to_deallocate, 0u);

    // and after that, the formerly cached memory isn't used anymore,
    // so new memory from upstream is returned back
    upstream.id_to_allocate = 4;
    tracked_pointer<void> a6 = pool->do_allocate(16, THRUST_MR_DEFAULT_ALIGNMENT);
    ASSERT_EQ(upstream.id_to_allocate, 0u);
    ASSERT_EQ(a6.id, 4u);

    // destruction also returns memory
    upstream.id_to_deallocate = 4;

    // actually destroy the pool; reasons why RAII is not used outlined at the beginning
    // of this function
    delete pool;
    ASSERT_EQ(upstream.id_to_deallocate, 0u);
}

TEST(MrPoolTests, TestUnsynchronizedPool)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestPool<thrust::mr::unsynchronized_pool_resource>();
}

TEST(MrPoolTests, TestSynchronizedPool)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestPool<thrust::mr::synchronized_pool_resource>();
}

template<template<typename> class PoolTemplate>
void TestPoolCachingOversized()
{
    tracked_resource upstream;

    upstream.id_to_allocate = -1u;

    using Pool = PoolTemplate<tracked_resource>;

    thrust::mr::pool_options opts = Pool::get_default_options();
    opts.cache_oversized = true;
    opts.largest_block_size = 1024;

    Pool pool(&upstream, opts);

    upstream.id_to_allocate = 1;
    tracked_pointer<void> a1 = pool.do_allocate(2048, 32);
    ASSERT_EQ(a1.id, 1u);

    upstream.id_to_allocate = 2;
    tracked_pointer<void> a2 = pool.do_allocate(64, 32);
    ASSERT_EQ(a2.id, 2u);

    pool.do_deallocate(a2, 64, 32);
    pool.do_deallocate(a1, 2048, 32);

    // make sure a good fit is used from the cache
    tracked_pointer<void> a3 = pool.do_allocate(32, 32);
    ASSERT_EQ(a3.id, 2u);

    tracked_pointer<void> a4 = pool.do_allocate(1024, 32);
    ASSERT_EQ(a4.id, 1u);

    pool.do_deallocate(a4, 1024, 32);

    // make sure that a new block is allocated when there's nothing cached with
    // the required alignment
    upstream.id_to_allocate = 3;
    tracked_pointer<void> a5 = pool.do_allocate(32, 64);
    ASSERT_EQ(a5.id, 3u);

    pool.release();

    // make sure that release actually clears caches
    upstream.id_to_allocate = 4;
    tracked_pointer<void> a6 = pool.do_allocate(32, 64);
    ASSERT_EQ(a6.id, 4u);

    upstream.id_to_allocate = 5;
    tracked_pointer<void> a7 = pool.do_allocate(2048, 1024);
    ASSERT_EQ(a7.id, 5u);

    pool.do_deallocate(a7, 2048, 1024);

    // make sure that the 'ridiculousness' factor for size (options.cached_size_cutoff_factor)
    // is respected
    upstream.id_to_allocate = 6;
    tracked_pointer<void> a8 = pool.do_allocate(24, 1024);
    ASSERT_EQ(a8.id, 6u);

    // make sure that the 'ridiculousness' factor for alignment (options.cached_alignment_cutoff_factor)
    // is respected
    upstream.id_to_allocate = 7;
    tracked_pointer<void> a9 = pool.do_allocate(2048, 32);
    ASSERT_EQ(a9.id, 7u);
}

TEST(MrPoolTests, TestUnsynchronizedPoolCachingOversized)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestPoolCachingOversized<thrust::mr::unsynchronized_pool_resource>();
}

TEST(MrPoolTests, TestSynchronizedPoolCachingOversized)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestPoolCachingOversized<thrust::mr::synchronized_pool_resource>();
}

template<template<typename> class PoolTemplate>
void TestGlobalPool()
{
    using Pool = PoolTemplate<thrust::mr::new_delete_resource>;

    ASSERT_EQ(thrust::mr::get_global_resource<Pool>() != nullptr, true);
}

TEST(MrPoolTests, TestUnsynchronizedGlobalPool)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestGlobalPool<thrust::mr::unsynchronized_pool_resource>();
}

TEST(MrPoolTests, TestSynchronizedGlobalPool)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    TestGlobalPool<thrust::mr::synchronized_pool_resource>();
}
