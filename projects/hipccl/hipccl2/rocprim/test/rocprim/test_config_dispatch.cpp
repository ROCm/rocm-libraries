#include "../common_test_header.hpp"

#include <rocprim/device/config_types.hpp>

#include <hip/hip_runtime.h>

using rocprim::detail::target_arch;

__global__ void write_target_arch(target_arch* dest_arch)
{
    static constexpr auto arch = rocprim::detail::device_target_arch();
    *dest_arch                 = arch;
}

// If this compile then
TEST(RocprimConfigDispatchTests, StrEqualN)
{
    using rocprim::detail::prefix_equals;

    static_assert(prefix_equals("", "", 0), "");
    ASSERT_TRUE(prefix_equals("", "", 0));
    static_assert(prefix_equals("a", "ab", 1), "");
    ASSERT_TRUE(prefix_equals("a", "ab", 1));
    static_assert(prefix_equals("prefix", "prefixaaa", 6), "");
    ASSERT_TRUE(prefix_equals("prefix", "prefixaaa", 6));

    static_assert(!prefix_equals("a", "b", 0), "");
    ASSERT_FALSE(prefix_equals("a", "b", 0));
    static_assert(!prefix_equals("a", "b", 1), "");
    ASSERT_FALSE(prefix_equals("a", "b", 1));
    static_assert(!prefix_equals("a", "", 1), "");
    ASSERT_FALSE(prefix_equals("a", "", 1));
    static_assert(!prefix_equals("", "b", 100), "");
    ASSERT_FALSE(prefix_equals("", "b", 100));
    static_assert(!prefix_equals("prefix", "prefixaaa", 7), "");
    ASSERT_FALSE(prefix_equals("prefix", "prefixaaa", 7));
    static_assert(!prefix_equals("hasprefix", "hasp", 4), "");
    ASSERT_FALSE(prefix_equals("hasprefix", "hasp", 4));
}

TEST(RocprimConfigDispatchTests, HostMatchesDevice)
{
    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    const hipStream_t stream = 0;

    target_arch host_arch;
    HIP_CHECK(rocprim::detail::host_target_arch(stream, host_arch));

    target_arch* device_arch_ptr;
    HIP_CHECK(hipMalloc(&device_arch_ptr, sizeof(*device_arch_ptr)));

    hipLaunchKernelGGL(write_target_arch, dim3(1), dim3(1), 0, stream, device_arch_ptr);
    HIP_CHECK(hipGetLastError());

    target_arch device_arch;
    HIP_CHECK(hipMemcpy(&device_arch, device_arch_ptr, sizeof(device_arch), hipMemcpyDeviceToHost));

    ASSERT_NE(host_arch, target_arch::invalid);
    ASSERT_EQ(host_arch, device_arch);
}

TEST(RocprimConfigDispatchTests, ParseCommonArches)
{
    using rocprim::detail::parse_gcn_arch;
    using rocprim::detail::target_arch;

    ASSERT_EQ(parse_gcn_arch(""), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch("not a gfx arch"), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch(":"), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch("g:"), target_arch::unknown);

    ASSERT_EQ(parse_gcn_arch("gfx803"), target_arch::gfx803);
    ASSERT_EQ(parse_gcn_arch("gfx900:sramecc+"), target_arch::gfx900);
    ASSERT_EQ(parse_gcn_arch("gfx906:::"), target_arch::gfx906);
    ASSERT_EQ(parse_gcn_arch("gfx908:"), target_arch::gfx908);
    ASSERT_EQ(parse_gcn_arch("gfx90a:sramecc+:xnack-"), target_arch::gfx90a);
}

#ifndef _WIN32
TEST(RocprimConfigDispatchTests, DeviceIdFromStream)
{
    using rocprim::detail::get_device_from_stream;

    const int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    int                          result;
    static constexpr hipStream_t default_stream = 0;
    HIP_CHECK(get_device_from_stream(default_stream, result));
    ASSERT_EQ(result, device_id);

    HIP_CHECK(get_device_from_stream(hipStreamPerThread, result));
    ASSERT_EQ(result, device_id);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(get_device_from_stream(stream, result));
    HIP_CHECK(hipStreamDestroy(stream));
    ASSERT_EQ(result, device_id);

    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK(get_device_from_stream(stream, result));
    HIP_CHECK(hipStreamDestroy(stream));
    ASSERT_EQ(result, device_id);
}
#endif
