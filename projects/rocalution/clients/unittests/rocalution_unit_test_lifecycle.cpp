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

// Tests of initialization and shutdown. Unlike rocalution-unit-test there is no
// global environment: every test starts and ends with an uninitialized platform.
// CTest runs each test in its own process. Run in one process, the tests also
// check that stop_rocalution() leaves a state from which the next init starts clean.

#include "unit_test_common.hpp"
#include "unit_test_local.hpp"

#ifdef __linux__
#include <unistd.h>
#endif

#ifdef ROCALUTION_UNIT_TEST_HIP_FAULTS
#include "hip_fault_injection.hpp"
#endif

using rocalution::_get_backend_descriptor;
using rocalution_unit::Backend;
using rocalution_unit::hip_device_present;
using rocalution_unit::on_accelerator;

namespace
{
    class lifecycle : public testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_FALSE(_get_backend_descriptor()->init);
            saved_ = *_get_backend_descriptor();
        }

        // Restores only what the tests set through the API. The state that
        // init_rocalution() derives is left to stop_rocalution().
        void TearDown() override
        {
            rocalution::stop_rocalution();
            EXPECT_FALSE(_get_backend_descriptor()->init);

            _get_backend_descriptor()->HIP_dev             = saved_.HIP_dev;
            _get_backend_descriptor()->rank                = saved_.rank;
            _get_backend_descriptor()->OpenMP_affinity     = saved_.OpenMP_affinity;
            _get_backend_descriptor()->disable_accelerator = saved_.disable_accelerator;

#ifdef SUPPORT_HIP
            // A failed device selection leaves an error that the next
            // init_rocalution() would read as its own
            (void)hipGetLastError();
#endif
        }

        std::string init_and_capture(int rank = -1, int dev_per_node = 1)
        {
            testing::internal::CaptureStdout();
            rocalution::init_rocalution(rank, dev_per_node);
            return testing::internal::GetCapturedStdout();
        }

    private:
        rocalution::Rocalution_Backend_Descriptor saved_{};
    };
}

TEST_F(lifecycle, stop_without_init_is_a_no_op)
{
    EXPECT_EQ(rocalution::stop_rocalution(), 0);
}

TEST_F(lifecycle, init_and_stop)
{
    EXPECT_EQ(rocalution::init_rocalution(), 0);
    EXPECT_TRUE(_get_backend_descriptor()->init);
    EXPECT_EQ(rocalution::_rocalution_available_accelerator(), hip_device_present());

    EXPECT_EQ(rocalution::stop_rocalution(), 0);
    EXPECT_FALSE(_get_backend_descriptor()->init);
    EXPECT_EQ(_get_backend_descriptor()->ROC_sparse_handle, nullptr);
}

TEST_F(lifecycle, init_twice_restarts)
{
    rocalution::init_rocalution();
    std::string out = init_and_capture();

    EXPECT_NE(out.find("has been initialized - restarting"), std::string::npos) << out;
    EXPECT_TRUE(_get_backend_descriptor()->init);
    EXPECT_EQ(rocalution::_rocalution_available_accelerator(), hip_device_present());
}

// The rank replaces the invalid device set before init with rank % dev_per_node
TEST_F(lifecycle, init_with_rank_selects_device)
{
    if(!hip_device_present())
    {
        GTEST_SKIP() << "no accelerator available";
    }

    rocalution::set_device_rocalution(99);
    init_and_capture(3, 1);

    EXPECT_EQ(_get_backend_descriptor()->rank, 3);
    EXPECT_EQ(_get_backend_descriptor()->HIP_dev, 0);
    EXPECT_TRUE(rocalution::_rocalution_available_accelerator());
}

// Without devices per node the rank does not select the device, so the invalid
// device set before init is kept. Only rank 0 logs, so the result is checked
// instead of the message.
TEST_F(lifecycle, init_with_rank_and_no_devices_per_node_keeps_device)
{
    if(!hip_device_present())
    {
        GTEST_SKIP() << "no accelerator available";
    }

    rocalution::set_device_rocalution(99);
    std::string out = init_and_capture(1, 0);

    EXPECT_EQ(_get_backend_descriptor()->rank, 1);
    EXPECT_EQ(_get_backend_descriptor()->HIP_dev, 99);
    EXPECT_EQ(out, "");
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

TEST_F(lifecycle, init_with_accelerator_disabled)
{
    rocalution::disable_accelerator_rocalution(true);
    std::string out = init_and_capture();

    EXPECT_NE(out.find("Warning: the accelerator is disabled"), std::string::npos) << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

// The same as init_with_accelerator_disabled, after a regular init and stop in
// the same process
TEST_F(lifecycle, restart_with_accelerator_disabled)
{
    rocalution::init_rocalution();
    rocalution::stop_rocalution();

    rocalution::disable_accelerator_rocalution(true);
    init_and_capture();

    if(hip_device_present())
    {
        KNOWN_BUG("neither stop_rocalution() nor the init resets the accelerator flag");
    }

    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

TEST_F(lifecycle, init_with_invalid_device_falls_back_to_host)
{
    if(!hip_device_present())
    {
        GTEST_SKIP() << "no accelerator available";
    }

    rocalution::set_device_rocalution(99);
    std::string out = init_and_capture();

    EXPECT_NE(out.find("HIP device 99 is invalid"), std::string::npos) << out;
    EXPECT_NE(out.find("Warning: the accelerator is disabled"), std::string::npos) << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

#ifdef SUPPORT_HIP
// A HIP error left by the application, or a HIP runtime without visible devices,
// makes the initialization fall back to the host
TEST_F(lifecycle, init_after_hip_error_falls_back_to_host)
{
    ASSERT_NE(hipSetDevice(99), hipSuccess);
    std::string out = init_and_capture();

    EXPECT_NE(out.find("falling back to host backend"), std::string::npos) << out;
    EXPECT_NE(out.find("Warning: the accelerator is disabled"), std::string::npos) << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}
#endif

TEST_F(lifecycle, omp_affinity_is_set_before_init)
{
    rocalution::set_omp_affinity_rocalution(false);
    init_and_capture();

    EXPECT_FALSE(_get_backend_descriptor()->OpenMP_affinity);
}

// Without the accelerator every move is ignored, so the objects keep their
// values on the host
TEST_F(lifecycle, without_an_accelerator_moves_keep_matrices_and_vectors_on_the_host)
{
    rocalution::disable_accelerator_rocalution(true);
    init_and_capture();

    // Only after an earlier init in the same process
    if(rocalution::_rocalution_available_accelerator())
    {
        KNOWN_BUG("neither stop_rocalution() nor the init resets the accelerator flag");
    }
    ASSERT_FALSE(rocalution::_rocalution_available_accelerator());

    const rocalution_unit::Tridiagonal<double> ref(5);
    const std::vector<double>                  x = {1, 2, 3, 4, 5};

    auto A = rocalution_unit::make_local_matrix(Backend::Host, ref);
    auto v = rocalution_unit::make_local_vector(Backend::Host, x);

    A->MoveToAccelerator();
    v->MoveToAccelerator();
    EXPECT_FALSE(on_accelerator(*A));
    EXPECT_FALSE(on_accelerator(*v));

    A->MoveToAcceleratorAsync();
    v->MoveToAcceleratorAsync();
    A->Sync();
    v->Sync();
    EXPECT_FALSE(on_accelerator(*A));
    EXPECT_FALSE(on_accelerator(*v));

    A->MoveToHostAsync();
    v->MoveToHostAsync();
    A->Sync();
    v->Sync();
    EXPECT_FALSE(on_accelerator(*A));
    EXPECT_FALSE(on_accelerator(*v));

    const rocalution_unit::Csr<double> C = rocalution_unit::csr_of(*A);
    EXPECT_EQ(C.row_offset, ref.row_offset);
    EXPECT_EQ(C.col, ref.col);
    EXPECT_EQ(C.val, ref.val);
    EXPECT_EQ(rocalution_unit::values_of(*v), x);

    auto y = rocalution_unit::make_local_vector(Backend::Host, std::vector<double>(5, 0.0));
    A->Apply(*v, y.get());
    EXPECT_EQ(rocalution_unit::values_of(*y), ref.apply(x));
}

#ifdef __linux__
namespace
{
    // Restores the CPU affinity of the calling thread and of every OpenMP pool
    // thread, as the threads created while pinned inherited the affinity
    class AffinityGuard
    {
    public:
        explicit AffinityGuard(const cpu_set_t& mask)
            : mask_(mask)
        {
        }

        ~AffinityGuard()
        {
#ifdef _OPENMP
#pragma omp parallel
#endif
            sched_setaffinity(0, sizeof(mask_), &mask_);
        }

        AffinityGuard(const AffinityGuard&)            = delete;
        AffinityGuard& operator=(const AffinityGuard&) = delete;

    private:
        cpu_set_t mask_;
    };
}

// While the OpenMP threads fit on every second CPU, the host thread is pinned to
// the even CPUs below twice the thread count. With more threads it is pinned to
// every CPU. The library takes CPUs 2k and 2k + 1 as hyperthreads of one core,
// which depends on the machine's numbering.
TEST_F(lifecycle, thread_affinity_uses_every_second_core_or_every_core)
{
    const int ncpu = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    if(ncpu < 2)
    {
        GTEST_SKIP() << "a single core";
    }

    cpu_set_t original;
    ASSERT_EQ(sched_getaffinity(0, sizeof(original), &original), 0);
    const AffinityGuard restore(original);

    rocalution::set_omp_affinity_rocalution(true);
    init_and_capture();

    const int half = ncpu / 2;
    cpu_set_t mask;
    rocalution::set_omp_threads_rocalution(half);
    ASSERT_EQ(sched_getaffinity(0, sizeof(mask), &mask), 0);
    for(int i = 0; i < ncpu; ++i)
    {
        if(i % 2 == 1)
        {
            EXPECT_FALSE(CPU_ISSET(i, &mask)) << "core " << i;
        }
        else if(i < 2 * half && CPU_ISSET(i, &original))
        {
            EXPECT_TRUE(CPU_ISSET(i, &mask)) << "core " << i;
        }
    }

    rocalution::set_omp_threads_rocalution(ncpu);
    EXPECT_EQ(_get_backend_descriptor()->OpenMP_threads, ncpu);
    ASSERT_EQ(sched_getaffinity(0, sizeof(mask), &mask), 0);
    for(int i = 0; i < ncpu; ++i)
    {
        if(CPU_ISSET(i, &original))
        {
            EXPECT_TRUE(CPU_ISSET(i, &mask)) << "core " << i;
        }
    }
}
#endif

#ifdef ROCALUTION_UNIT_TEST_HIP_FAULTS
using rocalution_unit::HipFault;
using rocalution_unit::ScopedHipFault;

namespace
{
    // The faults replace the results of a working device
    class lifecycle_hip_fault : public lifecycle
    {
    protected:
        void SetUp() override
        {
            lifecycle::SetUp();
            if(HasFatalFailure())
            {
                return;
            }

            if(!hip_device_present())
            {
                GTEST_SKIP() << "no HIP device available";
            }
        }

        int device_count()
        {
            int count = 0;
            EXPECT_EQ(hipGetDeviceCount(&count), hipSuccess);
            return count;
        }

        // Initializes with the fault armed and returns the output
        std::string init_with(HipFault fault, int expected_injections = 1)
        {
            ScopedHipFault armed(fault);
            std::string    out = init_and_capture();

            EXPECT_EQ(armed.injected(), expected_injections) << out;
            return out;
        }

        std::string stop_with(HipFault fault)
        {
            ScopedHipFault armed(fault);
            testing::internal::CaptureStdout();
            rocalution::stop_rocalution();
            std::string out = testing::internal::GetCapturedStdout();

            EXPECT_EQ(armed.injected(), 1) << out;
            return out;
        }
    };

    bool contains(const std::string& out, const std::string& text)
    {
        return out.find(text) != std::string::npos;
    }
}

TEST_F(lifecycle_hip_fault, device_count_query_fails)
{
    std::string out = init_with(HipFault::device_count_fails);

    EXPECT_TRUE(contains(out, "Querying for HIP devices failed - falling back to host backend"))
        << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

TEST_F(lifecycle_hip_fault, no_devices)
{
    std::string out = init_with(HipFault::no_devices);

    EXPECT_TRUE(contains(out, "No HIP device found")) << out;
    EXPECT_TRUE(contains(out, "HIP and rocBLAS/rocSPARSE have NOT been initialized!")) << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

TEST_F(lifecycle_hip_fault, device_context_in_use)
{
    std::string out = init_with(HipFault::context_in_use);

    EXPECT_TRUE(contains(out, "HIP context of device 0 is already in use")) << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

// Without a selected device, every device is tried
TEST_F(lifecycle_hip_fault, no_device_can_be_set)
{
    std::string out = init_with(HipFault::set_device_fails, device_count());

    EXPECT_TRUE(contains(out, "HIP and rocBLAS/rocSPARSE have NOT been initialized!")) << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

TEST_F(lifecycle_hip_fault, rocblas_handle_creation_fails)
{
    int         devices = device_count();
    std::string out     = init_with(HipFault::blas_handle_fails, devices);

    for(int dev = 0; dev < devices; ++dev)
    {
        EXPECT_TRUE(contains(out,
                             "HIP device " + std::to_string(dev)
                                 + " cannot create rocBLAS/rocSPARSE context and HIP streams"))
            << out;
    }
    EXPECT_TRUE(contains(out, "HIP and rocBLAS/rocSPARSE have NOT been initialized!")) << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

TEST_F(lifecycle_hip_fault, rocsparse_handle_creation_fails)
{
    int         devices = device_count();
    std::string out     = init_with(HipFault::sparse_handle_fails, devices);

    for(int dev = 0; dev < devices; ++dev)
    {
        EXPECT_TRUE(contains(out,
                             "HIP device " + std::to_string(dev)
                                 + " cannot create rocBLAS/rocSPARSE context and HIP streams"))
            << out;
    }
    EXPECT_TRUE(contains(out, "HIP and rocBLAS/rocSPARSE have NOT been initialized!")) << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

// A selected device is the only one tried
TEST_F(lifecycle_hip_fault, selected_device_cannot_create_handles)
{
    rocalution::set_device_rocalution(0);
    std::string out = init_with(HipFault::blas_handle_fails);

    EXPECT_TRUE(
        contains(out, "HIP device 0 cannot create rocBLAS/rocSPARSE context and HIP streams"))
        << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

TEST_F(lifecycle_hip_fault, low_compute_capability)
{
    std::string out = init_with(HipFault::low_compute_capability);

    EXPECT_TRUE(contains(out, "HIP device 0 has low compute capability (min 3.0 is needed)"))
        << out;
    EXPECT_FALSE(rocalution::_rocalution_available_accelerator());
}

TEST_F(lifecycle_hip_fault, unknown_registers_per_block)
{
    init_with(HipFault::no_registers_per_block);

    EXPECT_TRUE(rocalution::_rocalution_available_accelerator());
    EXPECT_EQ(_get_backend_descriptor()->HIP_max_threads, 65536);
}

TEST_F(lifecycle_hip_fault, rocblas_handle_destruction_fails)
{
    rocalution::init_rocalution();
    ASSERT_TRUE(rocalution::_rocalution_available_accelerator());

    std::string out = stop_with(HipFault::blas_destroy_fails);

    EXPECT_TRUE(contains(out, "Error in rocblas_destroy_handle")) << out;
    EXPECT_FALSE(contains(out, "Error in rocsparse_destroy_handle")) << out;
    EXPECT_FALSE(_get_backend_descriptor()->init);
}

TEST_F(lifecycle_hip_fault, rocsparse_handle_destruction_fails)
{
    rocalution::init_rocalution();
    ASSERT_TRUE(rocalution::_rocalution_available_accelerator());

    std::string out = stop_with(HipFault::sparse_destroy_fails);

    EXPECT_TRUE(contains(out, "Error in rocsparse_destroy_handle")) << out;
    EXPECT_FALSE(contains(out, "Error in rocblas_destroy_handle")) << out;
    EXPECT_FALSE(_get_backend_descriptor()->init);
}
#endif
