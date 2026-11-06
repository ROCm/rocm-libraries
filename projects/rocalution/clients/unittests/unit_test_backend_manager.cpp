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

// These tests run in the initialized process of rocalution-unit-test, which all
// tests share, so the fixture restores the backend descriptor after each test.
// Initialization and shutdown are tested in rocalution-unit-test-lifecycle.

#include "unit_test_common.hpp"

#ifdef __linux__
#include <unistd.h>
#endif

#include "base/local_vector.hpp"

using namespace rocalution_unit;
using rocalution::_get_backend_descriptor;

class backend_manager : public testing::Test
{
private:
    DescriptorGuard guard_;
};

TEST_F(backend_manager, set_omp_threads)
{
    rocalution::set_omp_threads_rocalution(2);

#ifdef _OPENMP
    EXPECT_EQ(_get_backend_descriptor()->OpenMP_threads, 2);
    EXPECT_EQ(omp_get_max_threads(), 2);
#else
    EXPECT_EQ(_get_backend_descriptor()->OpenMP_threads, 1);
#endif
}

#if defined(_OPENMP) && defined(__linux__)
// With the affinity enabled and at least twice as many CPUs as threads, the
// calling thread is pinned to the even CPUs
TEST_F(backend_manager, set_omp_threads_pins_to_every_second_core)
{
    cpu_set_t before;
    ASSERT_EQ(sched_getaffinity(0, sizeof(before), &before), 0);
    if(sysconf(_SC_NPROCESSORS_ONLN) < 4 || !CPU_ISSET(0, &before) || !CPU_ISSET(2, &before))
    {
        GTEST_SKIP() << "needs 4 cores, among them cores 0 and 2";
    }
    ASSERT_TRUE(descriptor().OpenMP_affinity);

    {
        DescriptorGuard guard;
        rocalution::set_omp_threads_rocalution(2);

        cpu_set_t pinned;
        ASSERT_EQ(sched_getaffinity(0, sizeof(pinned), &pinned), 0);
        EXPECT_EQ(CPU_COUNT(&pinned), 2);
        EXPECT_TRUE(CPU_ISSET(0, &pinned));
        EXPECT_TRUE(CPU_ISSET(2, &pinned));
    }

    cpu_set_t after;
    ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
    EXPECT_TRUE(CPU_EQUAL(&before, &after)) << "the guard did not restore the affinity";
}
#endif

TEST_F(backend_manager, omp_threshold_selects_single_thread_for_small_sizes)
{
    rocalution::set_omp_threshold_rocalution(100);
    EXPECT_EQ(_get_backend_descriptor()->OpenMP_threshold, 100);

#ifdef _OPENMP
    rocalution::_set_omp_backend_threads(descriptor(), 10);
    EXPECT_EQ(omp_get_max_threads(), 1);

    rocalution::_set_omp_backend_threads(descriptor(), 1000);
    EXPECT_EQ(omp_get_max_threads(), descriptor().OpenMP_threads);

    rocalution::_set_omp_backend_threads(descriptor(), -1);
    EXPECT_EQ(omp_get_max_threads(), descriptor().OpenMP_threads);
#endif
}

TEST_F(backend_manager, omp_threshold_zero_disables_the_threshold)
{
    rocalution::set_omp_threads_rocalution(2);
    rocalution::set_omp_threshold_rocalution(0);

    rocalution::_set_omp_backend_threads(descriptor(), 10);
#ifdef _OPENMP
    EXPECT_EQ(omp_get_max_threads(), 2);
#endif
}

TEST_F(backend_manager, info_of_initialized_platform)
{
    testing::internal::CaptureStdout();
    rocalution::info_rocalution();
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("rocALUTION ver"), std::string::npos) << out;
    EXPECT_NE(out.find("rocALUTION platform is initialized"), std::string::npos) << out;
#ifdef SUPPORT_HIP
    EXPECT_NE(out.find("Accelerator backend: HIP"), std::string::npos) << out;
#else
    EXPECT_NE(out.find("Accelerator backend: None"), std::string::npos) << out;
#endif
#ifdef _OPENMP
    EXPECT_NE(out.find("OpenMP threads:"), std::string::npos) << out;
#else
    EXPECT_NE(out.find("No OpenMP support"), std::string::npos) << out;
#endif
}

TEST_F(backend_manager, info_of_given_descriptor)
{
    rocalution::Rocalution_Backend_Descriptor d = descriptor();
    d.init                                      = false;
    d.disable_accelerator                       = true;
    d.accelerator                               = false;

    testing::internal::CaptureStdout();
    rocalution::info_rocalution(d);
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("rocALUTION platform is NOT initialized"), std::string::npos) << out;
    EXPECT_NE(out.find("The accelerator is disabled"), std::string::npos) << out;
#ifdef SUPPORT_HIP
    EXPECT_NE(out.find("HIP is not initialized"), std::string::npos) << out;
#endif
}

TEST_F(backend_manager, set_backend_descriptor)
{
    rocalution::Rocalution_Backend_Descriptor modified = descriptor();
    modified.OpenMP_threshold                          = 12345;

    rocalution::_set_backend_descriptor(modified);
    EXPECT_EQ(_get_backend_descriptor()->OpenMP_threshold, 12345);
}

// The interior and ghost streams only exist with multi-node support. Without it,
// every compute stream is the default stream.
TEST_F(backend_manager, compute_switches_streams)
{
    REQUIRE_BACKEND(Backend::Accelerator);

#ifdef SUPPORT_MULTINODE
    void* interior = descriptor().HIP_stream_interior;
    void* ghost    = descriptor().HIP_stream_ghost;
#else
    void* interior = descriptor().HIP_stream_default;
    void* ghost    = descriptor().HIP_stream_default;
#endif

    rocalution::_rocalution_compute_interior();
    EXPECT_EQ(descriptor().HIP_stream_current, interior);
    rocalution::_rocalution_compute_ghost();
    EXPECT_EQ(descriptor().HIP_stream_current, ghost);
    rocalution::_rocalution_compute_default();
    EXPECT_EQ(descriptor().HIP_stream_current, descriptor().HIP_stream_default);

    rocalution::_rocalution_sync();
    rocalution::_rocalution_sync_default();
    rocalution::_rocalution_sync_interior();
    rocalution::_rocalution_sync_ghost();
}

TEST_F(backend_manager, sync_and_compute_without_accelerator_do_nothing)
{
    int sentinel                                  = 0;
    _get_backend_descriptor()->accelerator        = false;
    _get_backend_descriptor()->HIP_stream_current = &sentinel;

    rocalution::_rocalution_compute_interior();
    rocalution::_rocalution_compute_ghost();
    rocalution::_rocalution_compute_default();
    rocalution::_rocalution_sync();
    rocalution::_rocalution_sync_default();
    rocalution::_rocalution_sync_interior();
    rocalution::_rocalution_sync_ghost();

    EXPECT_EQ(descriptor().HIP_stream_current, &sentinel);
}

TEST_F(backend_manager, get_arch)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    EXPECT_EQ(rocalution::get_arch_rocalution().rfind("gfx", 0), 0u);

    _get_backend_descriptor()->backend = rocalution::None;
    EXPECT_EQ(rocalution::get_arch_rocalution(), "cpu");
}

// Object tracking is a compile-time switch in utils/def.hpp, off by default
TEST_F(backend_manager, object_tracking)
{
    rocalution::LocalVector<double> v;

#ifdef OBJ_TRACKING_OFF
    EXPECT_TRUE(rocalution::_rocalution_check_if_any_obj());

    int dummy = 0;
    EXPECT_EQ(rocalution::_rocalution_add_obj(reinterpret_cast<rocalution::RocalutionObj*>(&dummy)),
              0u);
    EXPECT_TRUE(
        rocalution::_rocalution_del_obj(reinterpret_cast<rocalution::RocalutionObj*>(&dummy), 0));
#else
    EXPECT_FALSE(rocalution::_rocalution_check_if_any_obj());
#endif
}

// Death tests run their statement in a child process, so changes to the
// descriptor there do not reach the other tests

TEST(backend_manager_DeathTest, settings_that_require_an_initialized_platform)
{
    EXPECT_ASSERT_DEATH(
        {
            _get_backend_descriptor()->init = false;
            rocalution::set_omp_threads_rocalution(2);
        },
        "init == true");
    EXPECT_ASSERT_DEATH(
        {
            _get_backend_descriptor()->init = false;
            rocalution::set_omp_threshold_rocalution(100);
        },
        "init == true");
}

TEST(backend_manager_DeathTest, settings_that_require_an_uninitialized_platform)
{
    EXPECT_ASSERT_DEATH(rocalution::set_device_rocalution(0), "init == false");
    EXPECT_ASSERT_DEATH(rocalution::set_omp_affinity_rocalution(false), "init == false");
    EXPECT_ASSERT_DEATH(rocalution::disable_accelerator_rocalution(true), "init == false");
}

#ifdef _OPENMP
TEST(backend_manager_DeathTest, stop_requires_default_omp_threads)
{
    EXPECT_ASSERT_DEATH(
        {
            _get_backend_descriptor()->OpenMP_def_threads = 0;
            rocalution::stop_rocalution();
        },
        "OpenMP_def_threads > 0");
}
#endif

TEST(backend_manager_DeathTest, block_dimension_is_only_for_bcsr)
{
    EXPECT_ASSERT_DEATH(
        rocalution::_rocalution_init_base_host_matrix<double>(descriptor(), rocalution::CSR, 2),
        "blockdim == 1");
    EXPECT_ASSERT_DEATH(
        rocalution::_rocalution_init_base_backend_matrix<double>(descriptor(), rocalution::CSR, 2),
        "blockdim == 1");
}

TEST(backend_manager_DeathTest, backend_objects_without_backend_support)
{
    rocalution::Rocalution_Backend_Descriptor none = descriptor();
    none.backend                                   = rocalution::None;

    EXPECT_FATAL_ERROR(rocalution::_rocalution_init_base_backend_vector<double>(none),
                       "backend_manager\\.cpp");
    EXPECT_FATAL_ERROR(
        rocalution::_rocalution_init_base_backend_matrix<double>(none, rocalution::CSR, 1),
        "backend_manager\\.cpp");
}
