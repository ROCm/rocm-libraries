// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Small tests of HIP observation and memory guards. GEMM coverage lives in YAML;
// numerical counterexamples belong to HostNumerics' verifier tests.
#include "device_validation.hpp"
#include "hipBuffer.hpp"
#include "hip_placement.hpp"
#include <gtest/gtest.h>
#include <hipblaslt/client/FreivaldsDevice.hpp>
#include <hipblaslt/hostnumerics/Types.hpp>

#include <array>
#include <cmath>
#include <cstring>

namespace
{
    namespace hn = roc::hostnumerics;
    constexpr std::array types{HIP_R_32F,
                               HIP_R_64F,
                               HIP_R_16F,
                               HIP_R_16BF,
                               HIP_R_32I,
                               HIP_R_8I,
                               HIP_R_8F_E4M3,
                               HIP_R_8F_E5M2,
                               HIP_R_8F_E4M3_FNUZ,
                               HIP_R_8F_E5M2_FNUZ};

    struct Stream
    {
        hipStream_t value{};
        Stream()
        {
            if(hipStreamCreateWithFlags(&value, hipStreamNonBlocking) != hipSuccess)
                throw std::runtime_error("could not create test stream");
        }
        ~Stream()
        {
            (void)hipStreamDestroy(value);
        }
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;
    };

    TEST(FreivaldsDevice_pre_checkin, TypedBatchesLocateFiniteCorruption)
    {
        const Stream     stream;
        constexpr size_t m = 7, n = 11, k = 3, ld = 10, stride = 120, batches = 2;
        for(auto apiType : types)
        {
            SCOPED_TRACE(int(apiType));
            const auto type = hipblaslt::hostnumerics::scalarType(apiType);
            hn::Tensor stored(type, hn::Layout(hn::Shape{batches, m, n}, {stride, 1, ld}));
            std::vector<hn::FreivaldsPlan> plans;
            hn::ScalarConversionOptions    conversion;
            conversion.integerOverflow = hn::IntegerOverflow::Saturate;
            for(size_t batch = 0; batch < batches; ++batch)
            {
                hn::Tensor a(hn::ScalarType::Float32, hn::Shape{m, k});
                hn::Tensor b(hn::ScalarType::Float32, hn::Shape{k, n});
                for(size_t i = 0; i < m; ++i)
                    for(size_t z = 0; z < k; ++z)
                        a.storeFrom({i, z}, int(i + z + batch) - 3);
                for(size_t z = 0; z < k; ++z)
                    for(size_t j = 0; j < n; ++j)
                        b.storeFrom({z, j}, int((z + j) % 3) - 1);
                // Independent scalar oracle; the observer never launches a GEMM.
                for(size_t i = 0; i < m; ++i)
                    for(size_t j = 0; j < n; ++j)
                    {
                        int sum = 0;
                        for(size_t z = 0; z < k; ++z)
                            sum += (int(i + z + batch) - 3) * (int((z + j) % 3) - 1);
                        stored.storeFrom({batch, i, j}, sum, conversion);
                    }
                auto prepared
                    = hn::prepareFreivalds(a, b, hn::MatmulOptions{}, hn::EpilogueOptions{}, type);
                ASSERT_TRUE(std::holds_alternative<hn::FreivaldsPlan>(prepared));
                plans.push_back(std::get<hn::FreivaldsPlan>(std::move(prepared)));
            }
            const auto      bytes = stored.rawEncodedBackingStorage();
            HipDeviceBuffer device(apiType, bytes.size() / device_element_size(apiType));
            ASSERT_EQ(
                hipMemcpyAsync(
                    device.buf(), bytes.data(), bytes.size(), hipMemcpyHostToDevice, stream.value),
                hipSuccess);
            const DeviceMatrix matrix{device.buf(), apiType, m, n, ld, stride};
            auto               report = verify_freivalds_device(matrix, plans, stream.value);
            ASSERT_TRUE(report.passed) << report.message;
            EXPECT_EQ(report.statistics.checked_elements, batches * m * n);
            ASSERT_EQ(corrupt_element_device(matrix, 1, 3, 7, stream.value), hipSuccess);
            report = verify_freivalds_device(matrix, plans, stream.value);
            EXPECT_FALSE(report.passed);
            EXPECT_NE(report.message.find("batch 1, row 3, col 7"), std::string::npos)
                << report.message;
            ASSERT_EQ(fill_sentinel_device(
                          device.as<char>() + (stride + 3 + 7 * ld) * device_element_size(apiType),
                          apiType,
                          1,
                          stream.value),
                      hipSuccess);
            EXPECT_FALSE(verify_freivalds_device(matrix, plans, stream.value).passed);
        }
    }

    TEST(FreivaldsDevice_pre_checkin, Encoded16BitValues_smoke)
    {
        const Stream     stream;
        constexpr size_t count = size_t{1} << 16;
        for(auto apiType : {HIP_R_16F, HIP_R_16BF})
        {
            SCOPED_TRACE(int(apiType));
            hn::Tensor stored(hipblaslt::hostnumerics::scalarType(apiType), hn::Shape{count, 1});
            const auto bytes = stored.rawEncodedBackingStorage();
            for(size_t i = 0; i < count; ++i)
            {
                const auto bits = uint16_t(i);
                std::memcpy(bytes.data() + i * sizeof(bits), &bits, sizeof(bits));
            }
            HipDeviceBuffer device(apiType, count);
            ASSERT_EQ(
                hipMemcpyAsync(
                    device.buf(), bytes.data(), bytes.size(), hipMemcpyHostToDevice, stream.value),
                hipSuccess);
            hn::FreivaldsRequest request;
            request.shape              = stored.shape();
            request.outputType         = stored.type();
            request.fastMagnitudeLimit = 2048;
            request.maximumExceptions  = count;
            request.rightProbe = hn::Tensor(uint64_t{1}).reshapeSharingStorage(hn::Shape{1});
            request.leftProbe  = hn::Tensor(hn::ScalarType::UInt64, hn::Shape{count});
            for(size_t i = 0; i < count; ++i)
                request.leftProbe.storeFrom({i}, uint64_t{1});
            const auto result = hipblaslt::client::observeFreivalds(
                request, device.buf(), stored.layout(), stream.value);
            const auto* observed = std::get_if<hn::FreivaldsObservation>(&result.output);
            ASSERT_NE(observed, nullptr);
            ASSERT_TRUE(observed->complete);
            EXPECT_EQ(observed->checkedElements, count);
            std::vector<bool> exceptional(count);
            for(const auto& element : observed->exceptions)
            {
                ASSERT_LT(element.row, count);
                EXPECT_EQ(element.column, 0u);
                EXPECT_FALSE(exceptional[element.row]);
                exceptional[element.row] = true;
                const double expected    = stored.loadAs<double>({element.row, 0});
                if(std::isnan(expected))
                    EXPECT_TRUE(std::isnan(element.storedValue));
                else
                    EXPECT_EQ(element.storedValue, expected) << "encoded bits " << element.row;
            }
            for(size_t i = 0; i < count; ++i)
            {
                const double value = stored.loadAs<double>({i, 0});
                const bool   exact
                    = std::fabs(value) < request.fastMagnitudeLimit && std::trunc(value) == value;
                EXPECT_EQ(exceptional[i], !exact) << "encoded bits " << i;
                EXPECT_EQ(observed->rowSums.loadAs<uint64_t>({i}),
                          exact ? hn::freivaldsResidue(int64_t(value)) : 0u)
                    << "encoded bits " << i;
            }
        }
    }

    TEST(FreivaldsDevice_pre_checkin, LayoutsFallbackAndExceptionBudget)
    {
        const Stream     stream;
        const hn::Shape  shape{7, 11};
        const hn::Tensor a(hn::ScalarType::Float32, hn::Shape{7, 0});
        const hn::Tensor b(hn::ScalarType::Float32, hn::Shape{0, 11});
        hn::Tensor       expected(hn::ScalarType::Float32, shape);
        for(size_t i = 0; i < 7; ++i)
            for(size_t j = 0; j < 11; ++j)
                expected.storeFrom({i, j}, int((i + j) % 5) - 2);
        hn::EpilogueOptions epilogue;
        epilogue.addend = expected;
        for(auto type : {hn::ScalarType::Float32, hn::ScalarType::Float8E4M3})
            for(const auto& layout : {hn::Layout(shape, {1, 10}, 3),
                                      hn::Layout(shape, {13, 1}, 3),
                                      hn::Layout(shape, {-1, 10}, 9),
                                      hn::Layout(shape, {3, 25}, 3)})
            {
                const auto prepared
                    = hn::prepareFreivalds(a, b, hn::MatmulOptions{}, epilogue, type);
                ASSERT_TRUE(std::holds_alternative<hn::FreivaldsPlan>(prepared));
                const auto&     plan   = std::get<hn::FreivaldsPlan>(prepared);
                auto            stored = expected.copyConvertedTo(type, layout);
                const auto      bytes  = stored.rawEncodedBackingStorage();
                HipDeviceBuffer device(HIP_R_8I, bytes.size());
                for(bool corrupt : {false, true})
                {
                    if(corrupt)
                        stored.storeFrom({3, 7}, expected.loadAs<float>({3, 7}) + 1);
                    ASSERT_EQ(hipMemcpyAsync(device.buf(),
                                             bytes.data(),
                                             bytes.size(),
                                             hipMemcpyHostToDevice,
                                             stream.value),
                              hipSuccess);
                    const auto observation = hipblaslt::client::observeFreivalds(
                        plan.copyRequest(), device.buf(), layout, stream.value);
                    const auto check = std::visit(
                        [&](const auto& output) { return plan.check(output); }, observation.output);
                    EXPECT_EQ(check.status,
                              corrupt ? hn::VerificationStatus::Mismatch
                                      : hn::VerificationStatus::Passed);
                    EXPECT_EQ(check.checkedElements, size_t(77));
                    EXPECT_EQ(std::holds_alternative<hn::Tensor>(observation.output),
                              type == hn::ScalarType::Float8E4M3);
                }
            }
        // A saturated integer output requires exact recomputation. Exhausting the
        // exception budget must fail, even if the stored value is correct.
        hn::Tensor largeA(256.0f), one(1.0f);
        const auto prepared = hn::prepareFreivalds(largeA.reshapeSharingStorage(hn::Shape{1, 1}),
                                                   one.reshapeSharingStorage(hn::Shape{1, 1}),
                                                   hn::MatmulOptions{},
                                                   hn::EpilogueOptions{},
                                                   hn::ScalarType::Int8);
        ASSERT_TRUE(std::holds_alternative<hn::FreivaldsPlan>(prepared));
        const std::array plans{std::get<hn::FreivaldsPlan>(prepared)};
        HipDeviceBuffer  device(HIP_R_8I, size_t(1));
        ASSERT_EQ(hipMemsetAsync(device.buf(), 127, 1, stream.value), hipSuccess);
        const DeviceMatrix matrix{device.buf(), HIP_R_8I, 1, 1, 1, 1};
        EXPECT_TRUE(verify_freivalds_device(matrix, plans, stream.value).passed);
        EXPECT_FALSE(verify_freivalds_device(matrix, plans, stream.value, 0).passed);
    }

    TEST(FreivaldsDevice_pre_checkin, AuxiliaryCopiesAndFailures)
    {
        const Stream               stream;
        const std::array<float, 4> e{-2, 0, 3, 7}, d{0, 0, 3, 7};
        HipDeviceBuffer            de(HIP_R_32F, e.size()), dd(HIP_R_32F, d.size());
        ASSERT_EQ(
            hipMemcpyAsync(de.buf(), e.data(), sizeof(e), hipMemcpyHostToDevice, stream.value),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(dd.buf(), d.data(), sizeof(d), hipMemcpyHostToDevice, stream.value),
            hipSuccess);
        const DeviceMatrix em{de.buf(), HIP_R_32F, 2, 2, 2, 4};
        const DeviceMatrix dm{dd.buf(), HIP_R_32F, 2, 2, 2, 4};
        double             maximum = 0;
        EXPECT_TRUE(
            verify_activation_device(dm, em, 1, 1, 1, hn::ReluActivation{}, stream.value, &maximum)
                .passed);
        EXPECT_EQ(maximum, 7);
        EXPECT_EQ(recover_amax_device(dm, 1, 1, stream.value), 7);
        ASSERT_EQ(corrupt_element_device(dm, 0, 1, 1, stream.value), hipSuccess);
        EXPECT_FALSE(
            verify_activation_device(dm, em, 1, 1, 1, hn::ReluActivation{}, stream.value, nullptr)
                .passed);
        auto unsupported = em;
        unsupported.type = HIP_C_32F;
        EXPECT_FALSE(verify_activation_device(
                         dm, unsupported, 1, 1, 1, hn::ReluActivation{}, stream.value, &maximum)
                         .passed);
        EXPECT_TRUE(std::isnan(maximum));
    }

    TEST(DeviceBounds_pre_checkin, TypedPaddingStoresCopiesAndFiniteInjection)
    {
        const Stream     stream;
        constexpr size_t rows = 5, cols = 3, ld = 8, stride = 30, batches = 2, total = 61;
        const auto       logical = [](size_t index) {
            return index / stride < batches && index % stride / ld < cols
                   && index % stride % ld < rows;
        };
        for(auto apiType : types)
        {
            SCOPED_TRACE(int(apiType));
            const auto   type  = hipblaslt::hostnumerics::scalarType(apiType);
            const size_t width = device_element_size(apiType);
            hn::Tensor   host(type, hn::Shape{total});
            for(size_t i = 0; i < total; ++i)
                host.storeFrom({i}, 1);
            const auto         bytes = host.rawEncodedBackingStorage();
            HipDeviceBuffer    device(apiType, total);
            const DeviceMatrix matrix{device.buf(), apiType, rows, cols, ld, stride};
            ASSERT_EQ(
                hipMemcpyAsync(
                    device.buf(), bytes.data(), bytes.size(), hipMemcpyHostToDevice, stream.value),
                hipSuccess);
            ASSERT_EQ(poison_padding_device(matrix, batches, total, stream.value), hipSuccess);
            EXPECT_TRUE(scan_padding_device(matrix, batches, total, true, stream.value).passed);
            ASSERT_EQ(hipMemcpy(bytes.data(), device.buf(), bytes.size(), hipMemcpyDeviceToHost),
                      hipSuccess);
            for(size_t i = 0; i < total; ++i)
                EXPECT_EQ(host.loadAs<double>({i}), logical(i) ? 1 : kPaddingPoisonValue);

            ASSERT_EQ(hipMemsetAsync(device.as<char>() + 36 * width, 0, width, stream.value),
                      hipSuccess);
            const auto badPadding = scan_padding_device(matrix, batches, total, true, stream.value);
            EXPECT_FALSE(badPadding.passed);
            EXPECT_NE(badPadding.message.find("batch 1, row 6, col 0"), std::string::npos);

            ASSERT_EQ(fill_sentinel_device(device.buf(), apiType, total, stream.value), hipSuccess);
            const auto unwritten = scan_padding_device(matrix, batches, total, false, stream.value);
            EXPECT_EQ(unwritten.passed, apiType == HIP_R_8I || apiType == HIP_R_32I);

            ASSERT_EQ(
                poison_padding_device({device.buf(), apiType, 0, 0, 1, 0}, 1, total, stream.value),
                hipSuccess);
            ASSERT_EQ(corrupt_element_device(matrix, 1, 3, 2, stream.value), hipSuccess);
            const auto changed = count_changed_device(device.buf(), apiType, total, stream.value);
            EXPECT_TRUE(changed.ok);
            EXPECT_EQ(changed.count, size_t(1));
            EXPECT_EQ(changed.first, size_t(49));
            hn::Tensor compact(type, hn::Shape{batches * rows * cols});
            ASSERT_EQ(copy_matrix_to_host(
                          compact.rawEncodedBackingStorage().data(), matrix, batches, stream.value),
                      hipSuccess);
            for(size_t i = 0; i < batches * rows * cols; ++i)
                EXPECT_EQ(compact.loadAs<double>({i}), i == 28 ? 0 : kPaddingPoisonValue);
        }
    }

    TEST(DeviceBounds_pre_checkin, EmptyAndUnsupportedBuffers)
    {
        const Stream stream;
        for(auto type : types)
        {
            const DeviceMatrix empty{nullptr, type, 0, 0, 1, 0};
            EXPECT_EQ(fill_sentinel_device(nullptr, type, 0, stream.value), hipSuccess);
            EXPECT_EQ(poison_padding_device(empty, 1, 0, stream.value), hipSuccess);
            EXPECT_TRUE(scan_padding_device(empty, 1, 0, false, stream.value).passed);
            EXPECT_TRUE(verify_freivalds_device(empty, {}, stream.value).passed);
        }
        const DeviceMatrix unsupported{nullptr, HIP_C_32F, 0, 0, 1, 0};
        EXPECT_EQ(fill_sentinel_device(nullptr, HIP_C_32F, 0, stream.value), hipErrorInvalidValue);
        EXPECT_EQ(poison_padding_device(unsupported, 1, 0, stream.value), hipErrorInvalidValue);
        EXPECT_FALSE(scan_padding_device(unsupported, 1, 0, false, stream.value).passed);
        EXPECT_FALSE(count_changed_device(nullptr, HIP_C_32F, 0, stream.value).ok);
    }

    TEST(DeviceBounds_pre_checkin, PlacementWindowsAndWorkspaceRanges)
    {
        // Leave a mapped tail even when the device uses 4 KiB allocation granules.
        const size_t bytes = (size_t(16) << 20) - 1;
        std::string  why;
        bool         unsupported = false;
        auto         region      = PlacedRegion::create(bytes, 0, HIP_R_8I, &why, &unsupported);
        if(!region && unsupported)
            GTEST_SKIP() << why;
        ASSERT_TRUE(region) << why;
        ASSERT_GT(region->span(), bytes);
        const auto   base = reinterpret_cast<uintptr_t>(region->ptr());
        const size_t room = bytes / 2;
        for(size_t size : {size_t(300), size_t(4096), size_t(1) << 20, room})
        {
            void* ptr = region->straddle(size, room);
            EXPECT_TRUE(region->crosses(ptr, size));
            const auto address = reinterpret_cast<uintptr_t>(ptr);
            EXPECT_EQ(address % 256, size_t(0));
            EXPECT_GE(address, base);
            EXPECT_LE(address + room, base + bytes);
        }
        EXPECT_TRUE(region->verify_poison("workspace", nullptr).passed);
        // Model an address with its carry dropped. The poison window is mapped
        // exactly 4 GiB below the intended access just past the boundary.
        const auto wrong = region->boundary() + 100 - PlacedRegion::k4GiB;
        int8_t     value = 0;
        ASSERT_EQ(hipMemcpy(&value, reinterpret_cast<void*>(wrong), 1, hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_EQ(value, kPaddingPoisonValue);
        ASSERT_EQ(hipMemset(reinterpret_cast<void*>(wrong), 0, 1), hipSuccess);
        EXPECT_FALSE(region->verify_poison("workspace", nullptr).passed);
        ASSERT_EQ(region->fill_poison(nullptr), hipSuccess);
        ASSERT_EQ(hipMemset(reinterpret_cast<void*>(base + bytes), 0, 1), hipSuccess);
        EXPECT_NE(region->verify_poison("workspace", nullptr).message.find("past the end"),
                  std::string::npos);
        EXPECT_FALSE(PlacedRegion::create(4096, 8192, HIP_R_32F, &why, &unsupported));
        EXPECT_FALSE(unsupported);
    }
} // namespace
