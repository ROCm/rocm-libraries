// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// This executable owns its HIP and operator-new interposers. Do not link this
// source into hipblaslt-test: fault injection must remain in its own process.
// The library and HIP runtime are real; only the host handle allocation fails.
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cstdlib>
#include <new>

namespace
{
    enum class Fault
    {
        None,
        BadAlloc,
        Other
    };

    struct Allocation
    {
        void*  pointer = nullptr;
        size_t bytes   = 0;
    };

    struct State
    {
        bool                      active     = false;
        bool                      insideHip  = false;
        bool                      armed      = false;
        Fault                     fault      = Fault::None;
        int                       injections = 0;
        int                       mallocs    = 0;
        int                       frees      = 0;
        std::array<Allocation, 8> allocations{};
    };

    thread_local State state;

    struct InjectedException
    {
    };

    struct HipCall
    {
        bool nested = state.insideHip;
        HipCall()
        {
            state.insideHip = true;
        }
        ~HipCall()
        {
            state.insideHip = nested;
        }
    };

    template <class Function>
    Function nextSymbol(const char* name)
    {
        auto symbol = dlsym(RTLD_NEXT, name);
        if(!symbol)
            std::abort();
        return reinterpret_cast<Function>(symbol);
    }
} // namespace

void* operator new(std::size_t size)
{
    if(state.armed && !state.insideHip)
    {
        state.armed = false;
        ++state.injections;
        if(state.fault == Fault::BadAlloc)
            throw std::bad_alloc();
        throw InjectedException{};
    }
    if(auto memory = std::malloc(size ? size : 1))
        return memory;
    throw std::bad_alloc();
}

void operator delete(void* memory) noexcept
{
    std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept
{
    std::free(memory);
}

extern "C" hipError_t hipMalloc(void** pointer, size_t bytes)
{
    HipCall     call;
    static auto real   = nextSymbol<hipError_t (*)(void**, size_t)>("hipMalloc");
    auto        status = real(pointer, bytes);
    if(state.active && !call.nested && status == hipSuccess)
    {
        ++state.mallocs;
        for(auto& allocation : state.allocations)
        {
            if(!allocation.pointer)
            {
                allocation = {*pointer, bytes};
                return status;
            }
        }
        std::abort();
    }
    return status;
}

extern "C" hipError_t hipFree(void* pointer)
{
    HipCall     call;
    static auto real   = nextSymbol<decltype(&hipFree)>("hipFree");
    auto        status = real(pointer);
    if(status == hipSuccess)
        for(auto& allocation : state.allocations)
            if(allocation.pointer && allocation.pointer == pointer)
            {
                allocation = {};
                ++state.frees;
                break;
            }
    return status;
}

extern "C" hipError_t hipGetDevice(int* device)
{
    HipCall     call;
    static auto real   = nextSymbol<decltype(&hipGetDevice)>("hipGetDevice");
    auto        status = real(device);
    // hipblasLtCreate queries the device after allocating both flag buffers,
    // immediately before rocblaslt_create calls new _rocblaslt_handle().
    // Do not inject into HIP's own host allocations or before the buffers exist.
    if(state.active && !call.nested && state.mallocs == 2 && state.injections == 0
       && state.fault != Fault::None && status == hipSuccess)
        state.armed = true;
    return status;
}

namespace
{
    void checkCreate(Fault fault, bool nullOutput = false)
    {
        state = {};
        // The optional numeric checker owns an additional allocation, outside
        // this regression's two flag buffers.
        ASSERT_EQ(setenv("HIPBLASLT_CHECK_NUMERICS", "0", 1), 0);
        ASSERT_EQ(hipFree(nullptr), hipSuccess);
        state.fault              = fault;
        hipblasLtHandle_t handle = nullptr;
        state.active             = true;
        auto status              = hipblasLtCreate(nullOutput ? nullptr : &handle);
        state.active             = false;
        state.armed              = false;

        const auto expected = nullOutput                 ? HIPBLAS_STATUS_INVALID_VALUE
                              : fault == Fault::BadAlloc ? HIPBLAS_STATUS_ALLOC_FAILED
                              : fault == Fault::Other    ? HIPBLAS_STATUS_INTERNAL_ERROR
                                                         : HIPBLAS_STATUS_SUCCESS;
        EXPECT_EQ(status, expected);
        EXPECT_EQ(state.injections, fault == Fault::None ? 0 : 1);
        EXPECT_EQ(handle != nullptr, !nullOutput && fault == Fault::None);
        if(handle)
            EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);

        EXPECT_EQ(state.mallocs, nullOutput ? 0 : 2);
        EXPECT_EQ(state.frees, state.mallocs);
        for(auto& allocation : state.allocations)
        {
            EXPECT_EQ(allocation.pointer, nullptr) << "Leaked " << allocation.bytes << " bytes";
            // Record the failure before releasing leaks left by an unpatched
            // library, so one failing case cannot exhaust the next case's GPU.
            if(auto pointer = allocation.pointer)
                EXPECT_EQ(hipFree(pointer), hipSuccess);
        }
    }

    TEST(HandleCreateException_pre_checkin, Success)
    {
        checkCreate(Fault::None);
    }
    TEST(HandleCreateException_pre_checkin, NullOutput)
    {
        checkCreate(Fault::None, true);
    }
    TEST(HandleCreateException_pre_checkin, BadAlloc)
    {
        checkCreate(Fault::BadAlloc);
    }
    TEST(HandleCreateException_pre_checkin, OtherException)
    {
        checkCreate(Fault::Other);
    }
} // namespace
