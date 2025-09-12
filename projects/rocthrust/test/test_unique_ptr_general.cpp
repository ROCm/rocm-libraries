
#include <thrust/unique_ptr.h>
#include "hip/hip_runtime.h"
#include "test_param_fixtures.hpp"
#include "test_utils.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_delete.h>


class A {
public:    
    __host__ __device__ A() : finished(nullptr), arr(nullptr), n(0) {}
    __host__ __device__ A(int num): finished(nullptr), n(num) {
        arr = thrust::device_new<int>(10);
    }
    __host__ __device__ A(thrust::device_ptr<bool> dev_p): finished(dev_p), n(0) {
        arr = thrust::device_new<int>(10);
    }
    __host__ __device__ A(thrust::device_ptr<bool> dev_p, int num): finished(dev_p), arr(nullptr), n(num) {}

    __host__ __device__ int number() { return n; }

    __host__ __device__ ~A() {
    #if defined(__HIP_DEVICE_COMPILE__)
        if(finished)
            *finished = true; 
    #endif
    }
 
    thrust::device_ptr<bool> finished;
    thrust::device_ptr<int> arr; 
    int n;
}; 

template <class T>
struct A_deleter {

    __host__ void operator()(thrust::device_ptr<T> p) const {
        A host_copy = *p; 
        if (host_copy.arr != nullptr) {
            thrust::device_delete(host_copy.arr, 10);
        }

        thrust::for_each_n(p, 1, [] __device__(T& x) { x.~T(); });
        thrust::device_free(p);
    }
};

struct GetDeleterTestDeleter
{
    void operator()(thrust::device_ptr<int>) const {}
    void operator()(thrust::device_ptr<int[]>) const {}

    int test() { return 5; }
    int test() const { return 6; }
};

TESTS_DEFINE(UniquePtrGeneralTests, NumericalTestsParams);

TEST(UniquePtrGeneralTests, TestUniquePtrSwap)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Test swap for single objects
    {
        thrust::unique_ptr<int> p1 = thrust::make_unique<int>(11);
        thrust::unique_ptr<int> p2 = thrust::make_unique<int>(22);

        int* raw_p1 = p1.get_raw();
        int* raw_p2 = p2.get_raw();

        p1.swap(p2);
        ASSERT_EQ(p1.get_raw(), raw_p2);
        ASSERT_EQ(p2.get_raw(), raw_p1);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrMoveAsgn)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Move assignment for single objects
    {
        thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
        *finished = false;

        thrust::unique_ptr<A> p1 = thrust::make_unique<A>(finished, 1);
        A* raw_p1 = p1.get_raw();
        {
            thrust::unique_ptr<A> p2 = thrust::make_unique<A>(nullptr, 2);
            ASSERT_NE(p1.get_raw(), nullptr);
            ASSERT_NE(p2.get_raw(), nullptr);

            p2 = std::move(p1);

            ASSERT_EQ(p2.get_raw(), raw_p1);
            ASSERT_EQ(p1.get_raw(), nullptr);
        }
        ASSERT_EQ(*finished, true);
        thrust::device_delete(finished);
    }  
}

TEST(UniquePtrGeneralTests, TestUnqiuePtrSelfMoveAsgn)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Self-move for single object
    {
        thrust::unique_ptr<int> p = thrust::make_unique<int>(1);
        int* raw_p = p.get_raw();
        p = std::move(p);
        ASSERT_EQ(p.get_raw(), raw_p);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrNullAsgn)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // NULL assignment for single object
    thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
    *finished = false;
    {
        thrust::unique_ptr<A> p = thrust::make_unique<A>(finished, 1);
        ASSERT_NE(p, nullptr);

        p = nullptr;

        ASSERT_EQ(p, nullptr);

        ASSERT_EQ(*finished, true);
        thrust::device_delete(finished);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrDefaultCtor)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Default constructed unique_ptr
    {
        thrust::unique_ptr<int> p;
        ASSERT_EQ(p, nullptr);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrMoveCtor)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Move constructor for single object
    {
        thrust::unique_ptr<int> p1 = thrust::make_unique<int>(42);
        int* raw_p1 = p1.get_raw();
        thrust::unique_ptr<int> p2(std::move(p1));
        ASSERT_EQ(p2.get_raw(), raw_p1);
        ASSERT_EQ(p1, nullptr);
    }   
}

TEST(UniquePtrGeneralTests, TestUniquePtrNullptrCtor)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Single object
    {
        thrust::unique_ptr<int> p(nullptr);
        ASSERT_EQ(p, nullptr);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrPointerCtor)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Single object, default deleter
    {
        thrust::device_ptr<int> dev_p = thrust::device_malloc<int>(1);

        thrust::unique_ptr<int> s(dev_p);
        ASSERT_EQ(s.get(), dev_p);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrDtorNullptr)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    // Single object
    {
        thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
        *finished = false;
        {
            thrust::unique_ptr<A> p(nullptr);
            ASSERT_EQ(p, nullptr);
        }
        ASSERT_EQ(*finished, false);
        thrust::device_delete(finished);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrModifierRelease)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Single object
    {
        thrust::unique_ptr<int> p = thrust::make_unique<int>(1);
        int* raw_p = p.get_raw();
        ASSERT_NE(raw_p, nullptr);

        thrust::device_ptr<int> released_p = p.release();

        ASSERT_EQ(p, nullptr);
        ASSERT_EQ(thrust::raw_pointer_cast(released_p), raw_p);

        thrust::device_delete(released_p);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrModifierReset)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // reset(new pointer) for single object
    thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
    *finished = false;
    {
        thrust::unique_ptr<A> p = thrust::make_unique<A>(finished, 1);

        thrust::device_ptr<A> new_p = thrust::device_malloc<A>(1);
        thrust::device_new<A>(new_p, A(finished, 2), 1);

        p.reset(new_p);
        ASSERT_EQ(p.get(), new_p);
        ASSERT_EQ(*finished, true);

        *finished = false;
    }
    ASSERT_EQ(*finished, true);
    
    // reset(nullptr) for single object
    *finished = false;
    {
        thrust::unique_ptr<A> p = thrust::make_unique<A>(finished, 1);
        
        p.reset(nullptr);
        ASSERT_EQ(p, nullptr);
            
        ASSERT_EQ(*finished, true);
    }

    // reset() for single object
    *finished = false;
    {
        thrust::unique_ptr<A> p = thrust::make_unique<A>(finished, 1);

        p.reset();
        ASSERT_EQ(p, nullptr);
            
        ASSERT_EQ(*finished, true);    
    }

    thrust::device_delete(finished);
}

TEST(UniquePtrGeneralTests, TestUniquePtrObserversDereference)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::unique_ptr<int> p = thrust::make_unique<int>(3);
    ASSERT_EQ(*p, 3);
}

TEST(UniquePtrGeneralTests, TestUniquePtrObserversExplicitBool)
{
    // Single non-null object
    {
        thrust::unique_ptr<int> p = thrust::make_unique<int>(1);
        const thrust::unique_ptr<int>& const_p = p;
        if (p) 
        {
            SUCCEED();
        } 
        else
        {
            FAIL() << "Non-NULL unique_ptr evaluated to false";
        }

        if (const_p)
        {
            SUCCEED();
        }
        else
        {
            FAIL() << "Const non-NULL unique_ptr evaluated to false";
        }
    }

    // Single null object
    {
        thrust::unique_ptr<int> p;
        const thrust::unique_ptr<int>& const_p = p;
        if (!p)
        {
            SUCCEED();
        } 
        else
        {
            FAIL() << "NULL unique_ptr evaluated to true";
        }

        if (!const_p)
        {
            SUCCEED();
        }
        else
        {
            FAIL() << "Const NULL unique_ptr evaluated to true";
        }
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrObserversGet)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // get() for single object
    {
        thrust::device_ptr<int> dev_p = thrust::device_malloc<int>(1);
        thrust::unique_ptr<int> p(dev_p);
        const thrust::unique_ptr<int>& const_p = p;

        ASSERT_EQ(p.get(), dev_p);
        ASSERT_EQ(const_p.get(), dev_p);
    }

    // // get() for single const object
    // {
    //     thrust::device_ptr<const int> dev_p = thrust::device_malloc<const int>(1);
    //     thrust::unique_ptr<const int> p(dev_p);
    //     const thrust::unique_ptr<const int>& const_p = p;

    //     ASSERT_EQ(p.get(), dev_p);
    //     ASSERT_EQ(const_p.get(), dev_p);
    // }
}

TEST(UniquePtrGeneralTests, TestUniquePtrUserTypeDefaultDeleter)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
    *finished = false;
    {
        thrust::device_ptr<A> dev_ptr = thrust::device_malloc<A>(1);
        thrust::device_new<A>(dev_ptr, A(finished, 1), 1);

        thrust::unique_ptr<A> p(dev_ptr);
        A host_p = *p;

        ASSERT_EQ(host_p.n, 1);
        ASSERT_EQ(host_p.arr, nullptr);
    }
    ASSERT_EQ(*finished, true);
    thrust::device_delete(finished);
}

TEST(UniquePtrGeneralTests, TestUniquePtrUserTypeCustomDeleter)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
    *finished = false;
    {
        thrust::device_ptr<A> dev_ptr = thrust::device_malloc<A>(1);
        thrust::device_new<A>(dev_ptr, A(finished), 1);

        thrust::unique_ptr<A, A_deleter<A>> p(dev_ptr);
        A host_p = *p;

        ASSERT_EQ(host_p.n, 0);
        ASSERT_NE(host_p.arr, nullptr);
    }
    ASSERT_EQ(*finished, true);
    thrust::device_delete(finished);
}

TEST(UniquePtrGeneralTests, TestUniquePtrGetDeleter)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Deleter stored by value
    {
        thrust::unique_ptr<int, GetDeleterTestDeleter> p;
        ASSERT_EQ(p.get_deleter().test(), 5);

        const thrust::unique_ptr<int, GetDeleterTestDeleter> const_p;
        ASSERT_EQ(const_p.get_deleter().test(), 6);
    }

    // Deleter stored by const reference
    {
        const GetDeleterTestDeleter d;
        thrust::unique_ptr<int, const GetDeleterTestDeleter&> p(nullptr, d);
        const thrust::unique_ptr<int, const GetDeleterTestDeleter&>& const_p = p;

        ASSERT_EQ(p.get_deleter().test(), 6);
        ASSERT_EQ(const_p.get_deleter().test(), 6);
    }

    // Deleter stored by non-const reference
    {
        GetDeleterTestDeleter d;
        thrust::device_ptr<int> dev_p;
        thrust::unique_ptr<int, GetDeleterTestDeleter&> p(nullptr, d);
        const thrust::unique_ptr<int, GetDeleterTestDeleter&>& const_p = p;

        ASSERT_EQ(p.get_deleter().test(), 5);
        ASSERT_EQ(const_p.get_deleter().test(), 5);
    }
}
