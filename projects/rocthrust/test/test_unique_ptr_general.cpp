// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  Apache-2.0

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
    __host__ __device__ A(thrust::device_ptr<bool> dev_p, int num): finished(dev_p), arr(nullptr), n(num) {}
    
    __host__ __device__ A(int num): finished(nullptr), n(num) {
        arr = thrust::device_new<int>(10); // This ctor can only be used with custom deleter, otherwise memory will be leaked
    }
    __host__ __device__ A(thrust::device_ptr<bool> dev_p): finished(dev_p), n(0) {
        arr = thrust::device_new<int>(10); // This ctor can only be used with custom deleter, otherwise memory will be leaked
    }

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

template <class T>
struct A_deleter<T[]> {

    A_deleter(int size = 0): m_size(size) {}

    __host__ void operator()(thrust::device_ptr<T> p) const { 
        for (size_t i = 0; i < m_size; ++i) {
            A host_copy = p[i];
            if (host_copy.arr != nullptr) {
                thrust::device_delete(host_copy.arr, 10);
            }
        }

        thrust::for_each_n(p, m_size, [] __device__(T& x) { x.~T(); });
        thrust::device_free(p);
    }

    size_t m_size;
};

struct GetDeleterTestDeleter
{
    void operator()(thrust::device_ptr<int>) const {}
    void operator()(thrust::device_ptr<int[]>) const {}

    int test() { return 5; }
    int test() const { return 6; }
};

TESTS_DEFINE(UniquePtrGeneralTests, NumericalTestsParams);

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.special/swap.pass.cpp
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

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.special/swap.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrSwapArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Test swap for arrays
    {
        thrust::unique_ptr<int[]> p1 = thrust::make_unique<int[]>(5);
        thrust::unique_ptr<int[]> p2 = thrust::make_unique<int[]>(10);

        int* raw_p1 = p1.get_raw();
        int* raw_p2 = p2.get_raw();

        p1.swap(p2);
        ASSERT_EQ(p1.get_raw(), raw_p2);
        ASSERT_EQ(p2.get_raw(), raw_p1);
    }
}

// Thrust-specific test for assignment operator.
TEST(UniquePtrGeneralTests, TestUniquePtrAsgnMove)
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

// Thrust-specific test for array assignment operator.
TEST(UniquePtrGeneralTests, TestUniquePtrAsgnMoveArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Move assignment for arrays
    {
        thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
        *finished = false;

        thrust::unique_ptr<A[]> p1 = thrust::make_unique<A[]>(5);
        for (int i = 0; i < 5; ++i)
            p1[i] = A(finished, i);
        A* raw_p1 = p1.get_raw();;
        {
            thrust::unique_ptr<A[]> p2 = thrust::make_unique<A[]>(10);
            for (int i = 0; i < 10; ++i)
                p2[i] = A(finished, i);

            ASSERT_NE(p1.get_raw(), nullptr);
            ASSERT_NE(p2.get_raw(), nullptr);

            p2 = std::move(p1);

            ASSERT_EQ(p2.get_raw(), raw_p1);
            ASSERT_EQ(p1.get_raw(), nullptr);

            ASSERT_EQ(*finished, true);
            *finished = false;
        }
        ASSERT_EQ(*finished, true);
        thrust::device_delete(finished);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.asgn/move.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrAsgnSelfMove)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Self-move for single object
    {
        thrust::unique_ptr<int> p = thrust::make_unique<int>(1);
        int* raw_p = p.get_raw();
        THRUST_DIAG_PUSH
        THRUST_DIAG_SUPPRESS_CLANG("-Wself-move")
        p = std::move(p);
        ASSERT_EQ(p.get_raw(), raw_p);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.asgn/move.pass.cpp
TEST(UniquePtrGeneralTests, TestUnqiuePtrAsgnSelfMoveArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Self-move for array
    {   
        thrust::unique_ptr<int[]> p = thrust::make_unique<int[]>(5);
        int* raw_p = p.get_raw();
        THRUST_DIAG_PUSH
        THRUST_DIAG_SUPPRESS_CLANG("-Wself-move")
        p = std::move(p);
        ASSERT_EQ(p.get_raw(), raw_p);
    }
}

// Thrust-specific test for assigning nullptr to unique_ptr.
TEST(UniquePtrGeneralTests, TestUniquePtrAsgnNull)
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

// Thrust-specific test for assigning nullptr to unique_ptr array.
TEST(UniquePtrGeneralTests, TestUniquePtrAsgnNullArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    // NULL assignment for array
    thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
    *finished = false;
    {
        thrust::unique_ptr<A[]> p = thrust::make_unique<A[]>(3);
        ASSERT_NE(p, nullptr);

        p[0] = A(finished, 1);
        p[1] = A(finished, 2);
        p[2] = A(finished, 3);

        p = nullptr;

        ASSERT_EQ(p, nullptr);
        
        ASSERT_EQ(*finished, true);
        thrust::device_delete(finished);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.ctor/default.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrCtorDefault)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Default constructed unique_ptr
    {
        thrust::unique_ptr<int> p;
        ASSERT_EQ(p, nullptr);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.ctor/move.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrCtorMove)
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

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.ctor/nullptr.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrCtorNullptr)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Single object
    {
        thrust::unique_ptr<int> p(nullptr);
        ASSERT_EQ(p, nullptr);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.ctor/pointer.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrCtorPointer)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Single object, default deleter
    {
        thrust::device_ptr<int> dev_p = thrust::device_malloc<int>(1);

        thrust::unique_ptr<int> s(dev_p);
        ASSERT_EQ(s.get(), dev_p);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.ctor/default.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrCtorDefaultArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Default constructed array unique_ptr
    {
        thrust::unique_ptr<int[]> p;
        ASSERT_EQ(p, nullptr);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.ctor/move.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrCtorMoveArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Move constructor for array
    {
        thrust::unique_ptr<int[]> p1 = thrust::make_unique<int[]>(3);
        int* raw_p1 = p1.get_raw();
        thrust::unique_ptr<int[]> p2(std::move(p1));
        ASSERT_EQ(p2.get_raw(), raw_p1);
        ASSERT_EQ(p1, nullptr); 
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.ctor/nullptr.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrCtorNullptrArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Array
    {
        thrust::unique_ptr<int[]> p(nullptr);
        ASSERT_EQ(p, nullptr);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.ctor/pointer.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrCtorPointerArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Array, default deleter
    {
        thrust::device_ptr<int> dev_p = thrust::device_malloc<int>(5);

        thrust::unique_ptr<int[]> s(dev_p);
        ASSERT_EQ(s.get(), dev_p);
    }
}

// Thrust-specific test for raw pointer constructor.
TEST(UniquePtrGeneralTests, TestUniquePtrCtorRawPointer)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Single object, default deleter
    {
        thrust::device_ptr<int> dev_p = thrust::device_malloc<int>(1);
        *dev_p = 33;

        int* raw_p = thrust::raw_pointer_cast(dev_p);
        thrust::unique_ptr<int> p(raw_p);
        ASSERT_EQ(p.get_raw(), raw_p);
        ASSERT_EQ(*p, 33);
    }

    // Array, default deleter
    {
        thrust::device_ptr<int> dev_p = thrust::device_malloc<int>(10);
        for(int i = 0; i < 5; ++i)
        {
            dev_p[i] = i * 10;
        }

        int* raw_p = thrust::raw_pointer_cast(dev_p);

        thrust::unique_ptr<int[]> p(raw_p);
        ASSERT_EQ(p.get_raw(), raw_p);
        for(int i = 0; i < 5; ++i)
        {
            ASSERT_EQ(p[i], i * 10);
        }
    }

    // Array with size
    {
        thrust::device_ptr<int> dev_p = thrust::device_malloc<int>(7);
        for(int i = 0; i < 7; ++i)
        {
            dev_p[i] = i + 100;
        }

        int* raw_p = thrust::raw_pointer_cast(dev_p);

        thrust::unique_ptr<int[]> p(raw_p, 7);
        ASSERT_EQ(p.get_raw(), raw_p);
        for(int i = 0; i < 7; ++i)
        {
            ASSERT_EQ(p[i], i + 100);
        }
    }
}

// Thrust-specific test for checking if the object destruction is correct.
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

// Thrust-specific test for checking if the array destruction is correct.
TEST(UniquePtrGeneralTests, TestUniquePtrDtorNullptrArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    // Array
    {
        thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
        *finished = false;
        {
            thrust::unique_ptr<int[]> p(nullptr);
            ASSERT_EQ(p, nullptr);
        }
        ASSERT_EQ(*finished, false);
        thrust::device_delete(finished);    
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.modifiers/release.pass.cpp
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

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.modifiers/release.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrModifierReleaseArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Array
    {
        thrust::unique_ptr<int[]> p = thrust::make_unique<int[]>(7);
        int* raw_p = p.get_raw();
        ASSERT_NE(raw_p, nullptr);

        thrust::device_ptr<int> released_p = p.release();

        ASSERT_EQ(p, nullptr);
        ASSERT_EQ(thrust::raw_pointer_cast(released_p), raw_p);

        thrust::device_delete(released_p);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.modifiers/reset.pass.cpp
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

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.modifiers/reset.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrModifierResetArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // reset(new pointer) for array
    thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
    *finished = false;
    {
        thrust::unique_ptr<A[]> p = thrust::make_unique<A[]>(10);
        for (int i = 0; i < 10; ++i)
            p[i] = A(finished, i);

        thrust::device_ptr<A> new_p = thrust::device_new<A>(15);
        for (int i = 0; i < 15; ++i)
            new_p[i] = A(finished, i);

        p.reset(new_p);
        ASSERT_EQ(p.get(), new_p);
        ASSERT_EQ(*finished, true);

        *finished = false;
    }
    ASSERT_EQ(*finished, true);

    // reset(nullptr) for array
    *finished = false;
    {
        thrust::unique_ptr<A[]> p = thrust::make_unique<A[]>(10);
        for (int i = 0; i < 10; ++i)
            p[i] = A(finished, i);

        p.reset(nullptr);
        ASSERT_EQ(p, nullptr);
            
        ASSERT_EQ(*finished, true);
    }

    // reset() for array
    *finished = false;
    {
        thrust::unique_ptr<A[]> p = thrust::make_unique<A[]>(10);
        for (int i = 0; i < 10; ++i)
            p[i] = A(finished, i);

        p.reset();
        ASSERT_EQ(p, nullptr);
            
        ASSERT_EQ(*finished, true);  
    }
    thrust::device_delete(finished);
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.observers/dereference.single.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrObserversDereference)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::unique_ptr<int> p = thrust::make_unique<int>(3);
    ASSERT_EQ(*p, 3);
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.observers/explicit_bool.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrObserversExplicitBool)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

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

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.observers/explicit_bool.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrObserversExplicitBoolArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
        // Array non-null
    {
        thrust::unique_ptr<int[]> p = thrust::make_unique<int[]>(10);
        const thrust::unique_ptr<int[]>& const_p = p;
        if (p) 
        {
            SUCCEED();
        } 
        else
        {
            FAIL() << "Non-NULL array unique_ptr evaluated to false";
        }

        if (const_p)
        {
            SUCCEED();
        }
        else
        {
            FAIL() << "Const non-NULL array unique_ptr evaluated to false";
        }
    }

    // Array null
    {
        thrust::unique_ptr<int[]> p;
        const thrust::unique_ptr<int[]>& const_p = p;
        if (!p) 
        {
            SUCCEED();
        } 
        else
        {
            FAIL() << "NULL array unique_ptr evaluated to true";
        }

        if (!const_p)
        {
            SUCCEED();
        }
        else
        {
            FAIL() << "Const NULL array unique_ptr evaluated to true";
        }
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.observers/get.pass.cpp
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

    // TODO: Uncomment this part of the test when the underlying issue with thrust::device_free is resolved.
    // This test is currently commented out because it fails to compile. The
    // thrust::unique_ptr<const int> destructor calls thrust::device_free, which
    // expects a thrust::device_ptr<void>. The implicit conversion from
    // thrust::device_ptr<const int> to thrust::device_ptr<void> is not allowed,
    // causing a compilation error.
    // // get() for single const object
    // {
    //     thrust::device_ptr<const int> dev_p = thrust::device_malloc<const int>(1);
    //     thrust::unique_ptr<const int> p(dev_p);
    //     const thrust::unique_ptr<const int>& const_p = p;

    //     ASSERT_EQ(p.get(), dev_p);
    //     ASSERT_EQ(const_p.get(), dev_p);
    // }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.observers/get.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrObserversGetArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // get() for array
    {
        thrust::device_ptr<int> dev_p = thrust::device_malloc<int>(10);
        thrust::unique_ptr<int[]> p(dev_p);
        const thrust::unique_ptr<int[]>& const_p = p;

        ASSERT_EQ(p.get(), dev_p);
        ASSERT_EQ(const_p.get(), dev_p);
    }

    // TODO: Uncomment this part of the test when the underlying issue with thrust::device_free is resolved.
    // This test is currently commented out because it fails to compile. The
    // thrust::unique_ptr<const int> destructor calls thrust::device_free, which
    // expects a thrust::device_ptr<void>. The implicit conversion from
    // thrust::device_ptr<const int> to thrust::device_ptr<void> is not allowed,
    // causing a compilation error.
    // // get() for const array
    // {
    //     thrust::device_ptr<const int> dev_p = thrust::device_new<const int>(10);
    //     thrust::unique_ptr<const int[]> p(dev_p);
    //     const thrust::unique_ptr<const int[]>& const_p = p;

    //     ASSERT_EQ(p.get(), dev_p);
    //     ASSERT_EQ(const_p.get(), dev_p);
    // }
}

// Thrust-specific test for testing if thrust::default_delete is correct.
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

// Thrust-specific test for testing if thrust::unique_ptr is correct with custom, user provided deleters.
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

// Thrust-specific test for testing if thrust::unique_ptr array is correct with custom, user provided deleters.
TEST(UniquePtrGeneralTests, TestUniquePtrUserTypeDefaultDeleterArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
    *finished = false;
    {
        thrust::device_ptr<A> dev_ptr = thrust::device_malloc<A>(3);
        dev_ptr[0] = A(finished, 1);
        dev_ptr[1] = A(finished, 2);
        dev_ptr[2] = A(finished, 3);

        thrust::unique_ptr<A[]> p(dev_ptr, 3);

        A host_p0 = p[0];
        A host_p1 = p[1];
        A host_p2 = p[2];

        ASSERT_EQ(host_p0.n, 1);
        ASSERT_EQ(host_p1.n, 2);
        ASSERT_EQ(host_p2.n, 3);
        
        ASSERT_EQ(host_p0.arr, nullptr);
        ASSERT_EQ(host_p1.arr, nullptr);
        ASSERT_EQ(host_p2.arr, nullptr);
    }
    ASSERT_EQ(*finished, true);
    thrust::device_delete(finished);
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.observers/get_deleter.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrUserTypeCustomDeleterArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
    *finished = false;
    {
        thrust::device_ptr<A> dev_ptr = thrust::device_malloc<A>(3);
        dev_ptr[0] = A(finished);
        dev_ptr[1] = A(finished);
        dev_ptr[2] = A(finished);

        thrust::unique_ptr<A[], A_deleter<A[]>> p(dev_ptr, 3);

        A host_p0 = p[0];
        A host_p1 = p[1];
        A host_p2 = p[2];

        ASSERT_EQ(host_p0.n, 0);
        ASSERT_EQ(host_p1.n, 0);
        ASSERT_EQ(host_p2.n, 0);

        ASSERT_NE(host_p0.arr, nullptr);
        ASSERT_NE(host_p1.arr, nullptr);
        ASSERT_NE(host_p2.arr, nullptr);
    }
    ASSERT_EQ(*finished, true);
    thrust::device_delete(finished);
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.observers/get_deleter.pass.cpp
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

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.class/unique.ptr.observers/get_deleter.pass.cpp
TEST(UniquePtrGeneralTests, TestUniquePtrGetDeleterArray)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Deleter stored by value
    {
        thrust::unique_ptr<int[], GetDeleterTestDeleter> p;
        ASSERT_EQ(p.get_deleter().test(), 5);

        const thrust::unique_ptr<int[], GetDeleterTestDeleter> const_p;
        ASSERT_EQ(const_p.get_deleter().test(), 6);
    }

    // Deleter stored by const reference
    {
        const GetDeleterTestDeleter d;
        thrust::unique_ptr<int[], const GetDeleterTestDeleter&> p(nullptr, d);
        const thrust::unique_ptr<int[], const GetDeleterTestDeleter&>& const_p = p;

        ASSERT_EQ(p.get_deleter().test(), 6);
        ASSERT_EQ(const_p.get_deleter().test(), 6);
    }

    // Deleter stored by non-const reference
    {
        GetDeleterTestDeleter d;
        thrust::unique_ptr<int[], GetDeleterTestDeleter&> p(nullptr, d);
        const thrust::unique_ptr<int[], GetDeleterTestDeleter&>& const_p = p;

        ASSERT_EQ(p.get_deleter().test(), 5);
        ASSERT_EQ(const_p.get_deleter().test(), 5);
    }
}