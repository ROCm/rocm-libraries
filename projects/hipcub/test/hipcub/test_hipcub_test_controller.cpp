// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <sstream>
#include <regex>
#include <vector>
#include <array>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

// Google Test
#include <gtest/gtest.h>

#include "common_test_header.hpp"
#include "test_utils_controller.hpp"

namespace test_controller
{

struct LineInfo
{
    std::string test;
    std::string arch;
    std::string size;
    std::string build_type;
    std::string skip_msg;
};
    
class HipcubTestControllerTests : public ::testing::Test
{
public:
    static std::string generate_control_text(const std::vector<LineInfo>& lines)
    {
        std::stringstream ss;
        for (const LineInfo& line : lines)
            ss << "/" << line.test << "/ : /" << line.arch << "/ : " << line.size << " : "
               << line.build_type << " : \"" << line.skip_msg << "\"" << std::endl;
        return ss.str();
    };

    template<class SizeType>
    static void test_filter(TestController&              controller,
                            const std::string&           text,
                            std::vector<SizeType>&       sizes,
                            const std::vector<SizeType>& expected_sizes)
    {
        SCOPED_TRACE(testing::Message() << "with text=" << std::endl << text);
        SCOPED_TRACE(testing::Message() << "with sizes=" << [&sizes](){
            std::stringstream ss;
            ss << "{ ";
            for (int i = 0; i < sizes.size(); i++)
            {
                ss << sizes[i];
                if (i < sizes.size() - 1)
                    ss << ", ";
            }
            ss << " }";
            return ss.str();
        }());
        
        controller.reset(std::make_optional(text));
        const bool expect_filtering = (sizes != expected_sizes);
        std::string msg;
        const std::vector<SizeType> result = controller.filter_sizes(sizes, msg);
        ASSERT_EQ(result, expected_sizes);
        ASSERT_EQ(expect_filtering, !msg.empty());
    }
};

TEST_F(HipcubTestControllerTests, GetArch)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));
    
    const std::string arch = TestController::get_arch();
    const std::regex arch_regex(R"(^(gfx[0-9a-f]+(?:\:xnack[+-])?)$|^nvidia$)");
    std::smatch match;
    ASSERT_TRUE(std::regex_match(arch, match, arch_regex));
}

TEST_F(HipcubTestControllerTests, CheckTestEnablement)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));
    
    const std::string arch = TestController::get_arch();
    const std::string alt_arch = (arch == "gfx1100" ? "gfx1200" : "gfx1100");
    const std::array<std::string, 2> os_vec = {"linux", "windows"};
    const std::string os = os_vec[env::is_windows()];
    const std::string alt_os = os_vec[!env::is_windows()];
    
    TestController& controller = TestController::get_or_create_instance(false);

    const auto test_enablement = [&controller](const std::string& text, const bool expected)
    {
        SCOPED_TRACE(testing::Message() << "with text=" << std::endl << text);
            
        controller.reset(std::make_optional(text));
        std::string msg;
        ASSERT_EQ(controller.check_test_enablement(msg), expected);
        ASSERT_EQ(msg.empty(), expected);
    };
    
    // When using size filter *, tests case should not be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\..*)", arch, "*", "*", "Skipping for unit test."}
        }),
        false
    );

    // When using individual sizes, test case should be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\..*)", arch, "1000", "*", "Skipping for unit test."}
        }),
        true
    );
    
    // When using alternate arch, test case should be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\..*)", alt_arch, "1000", "*", "Skipping for unit test."}
        }),
        true
    );

    // When using exact test name, test case should not be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\.CheckTestEnablement)", arch, "*", "*", "Skipping for unit test."}
        }),
        false
    );

    // When using alternate test name, test case should be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\..NonExistantTest)", arch, "*", "*", "Skipping for unit test."}
        }),
        true
    );
    
    // With multiple lines that cover the same test, one with * and others with individual sizes,
    // Test should not be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\..*)", arch, "1000", "*", "Skipping for unit test."},
                {R"(HipcubTestControllerTests\..*)", arch, "*", "*", "Skipping for unit test."},
                {R"(HipcubTestControllerTests\..*)", arch, "2000", "*", "Skipping for unit test."}
        }),
        false
    );

    // Use keyword in arch. Test should not be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\..*)", arch, "1000", "*", "Skipping for unit test."},
                {R"(HipcubTestControllerTests\..*)", "<all>", "*", "*", "Skipping for unit test."},
                {R"(HipcubTestControllerTests\..*)", arch, "2000", "*", "Skipping for unit test."}
        }),
        false
    );

    // Use alternate OS in build type. Test should be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\..*)", arch, "*", alt_os, "Skipping for unit test."},
        }),
        true
    );

    // Use current OS in build type. Test should not be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\..*)", arch, "*", os, "Skipping for unit test."},
        }),
        false
    );

    // Use both current and alternate OS in build type. Test should not be enabled.
    test_enablement(
        HipcubTestControllerTests::generate_control_text({
                {R"(HipcubTestControllerTests\..*)", arch, "*", os + ", " + alt_os, "Skipping for unit test."},
        }),
        false
    );
}

TEST_F(HipcubTestControllerTests, FilterSizes)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));
    
    const std::string arch = TestController::get_arch();
    const std::string alt_arch = (arch == "gfx1100" ? "gfx1200" : "gfx1100");
    
    TestController& controller = TestController::get_or_create_instance(false);
    
    // Create vector with sizes 0 - 9.
    std::vector<size_t> sizes(10);
    std::iota(sizes.begin(), sizes.end(), 0);

    // Filter out overlapping sizes across multiple lines.
    // Include a line for an alternate arch, which should not affect the result.
    std::vector<size_t> sizes_copy(sizes);
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "5", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "7", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "3", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", alt_arch, "1", "*", "Skipping for unit test."}
        }),
        sizes_copy,
        {0, 1, 2}
    );

    // Filter using operator that removes all sizes, with additional lines before and after.
    sizes_copy = sizes;
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "4", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "0", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "6", "*", "Skipping for unit test."}
        }),
        sizes_copy,
        {}
    );

    // Filter using * for size limit.
    sizes_copy = sizes;
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "*", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "4", "*", "Skipping for unit test."},
        }),
        sizes_copy,
        {}
    );

    // Use keywords in arch.
    sizes_copy = sizes;
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizes)", "<mi300-family>|" + arch, "9", "*", "Skipping for unit test."}
        }),
        sizes_copy,
        {0, 1, 2, 3, 4, 5, 6, 7, 8}
    );

    // Test with different OS build types.
    sizes_copy = sizes;
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "2", "windows", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "3", "linux", "Skipping for unit test."}
        }),
        sizes_copy,
        (env::is_windows() ? std::vector<size_t>({0, 1}) : std::vector<size_t>({0, 1, 2}))
    );

    // Test specifying multiple comma-separated build types at the same time.
    sizes_copy = sizes;
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "5", "windows, linux", "Skipping for unit test."}
        }),
        sizes_copy,
        {0, 1, 2, 3, 4}
    );

    // Test using * for build type.
    // Include extra line with alternate arch, which should not affect the result.
    sizes_copy = sizes;
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "5", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "4", "windows, *", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "3", "*, linux", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", alt_arch, "2", "*", "Skipping for unit test."}
        }),
        sizes_copy,
        {0, 1, 2}
    );
}

TEST_F(HipcubTestControllerTests, FilterSizesWithTransformer)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));
    
    const std::string arch = TestController::get_arch();
    const std::string alt_arch = (arch == "gfx1100" ? "gfx1200" : "gfx1100");
    
    TestController& controller = TestController::get_or_create_instance(false);

    // Test using a custom size transform functor.
    // Here, the test uses std::pair<size_t, size_t>.
    // The control file stores sizes as size_t values.
    // The PairTransformer functor converts the pair to a single size_t,
    // which is then compared against the control file values.
    struct PairTransformer
    {
        using size_type = std::pair<size_t, size_t>;
        size_t operator()(const size_type& size) const
        {
            // Use the largest value for the comparison.
            return std::max(size.first, size.second);
        }
    };

    controller.set_size_transformer(PairTransformer());
    
    std::vector<std::pair<size_t, size_t>> pairs = {
        {0, 1},
        {2, 3},
        {4, 5},
        {6, 7},
        {8, 9}
    };
    std::vector<std::pair<size_t, size_t>> pairs_copy(pairs);

    // Test removing based on the max value in each pair.
    // Use first value in pair as limit.
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizesWithTransformer)", arch, "4", "*", "Skipping for unit test."}
        }),
        pairs_copy,
        {
            {0, 1},
            {2, 3}
        }
    );

    // Use second value in pair as limit.
    pairs_copy = pairs;
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizesWithTransformer)", arch, "5", "*", "Skipping for unit test."}
        }),
        pairs_copy,
        {
            {0, 1},
            {2, 3}
        }
    );

    // Test after reseting the transformer. This should now use IdentityTransformer.
    controller.reset_size_transformer();
    std::vector<size_t> sizes(10);
    std::iota(sizes.begin(), sizes.end(), 0);
    
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizesWithTransformer)", arch, "2", "*", "Skipping for unit test."}
        }),
        sizes,
        {0, 1}
    );
}

TEST_F(HipcubTestControllerTests, FilterSizesWithUnits)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));
    
    const std::string arch = TestController::get_arch();
    const std::string alt_arch = (arch == "gfx1100" ? "gfx1200" : "gfx1100");
    
    TestController& controller = TestController::get_or_create_instance(false);
    
    // Test using units in size.
    std::vector<size_t> large_sizes = {2000, 2 * (1ull << 10), 3000000, 3 * (1ull << 20), 4000000000, 4 * (1ull << 30)};
    std::vector<size_t> large_sizes_copy(large_sizes);

    // Test with a size limit of 0 followed by unit.
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizesWithUnits)", arch, "5Mi", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizesWithUnits)", arch, "0K", "*", "Skipping for unit test."}
        }),
        large_sizes_copy,
        {}
    );

    // Test using a limit that is a multiple of a power of 2 unit.
    large_sizes_copy = large_sizes;
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizesWithUnits)", arch, "3Mi", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizesWithUnits)", arch, "3Gi", "*", "Skipping for unit test."}
        }),
        large_sizes_copy,
        std::vector(large_sizes.begin(), large_sizes.begin() + 3)
    );

    // Test using a limit that is a multiple of a power of 10 unit.
    large_sizes_copy = large_sizes;
    HipcubTestControllerTests::test_filter(
        controller,
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.FilterSizesWithUnits)", arch, "4G", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizesWithUnits)", arch, "4Gi", "*", "Skipping for unit test."}
        }),
        large_sizes_copy,
        std::vector(large_sizes.begin(), large_sizes.begin() + 4)
    );
}

// Note: Both TestController::check_size_enablement, amd TestController::filter_sizes
// call filter_sizes_inplace underneath. Since the majority of the filtering functionality
// is tested in FilterSizes, it's sufficient to just test a single case here.
TEST_F(HipcubTestControllerTests, CheckSizeEnablement)
{
    const std::string arch = TestController::get_arch();
    const std::string alt_arch = (arch == "gfx1100" ? "gfx1200" : "gfx1100");
    TestController& controller = TestController::get_or_create_instance(false);

    const auto test_size = [&controller](const std::string&         text,
                                        const std::vector<size_t>& sizes,
                                        const std::vector<bool>&   expected_results)
    {
        controller.reset(std::make_optional(text));
        for(size_t i = 0; i < sizes.size(); i++)
        {
            SCOPED_TRACE(testing::Message() << "with text=" << std::endl << text);
            SCOPED_TRACE(testing::Message() << "with size=" << sizes[i]);
            std::string msg;
            const bool result = controller.check_size_enablement(sizes[i], msg);
            ASSERT_EQ(result, expected_results[i]);
            ASSERT_EQ(expected_results[i], msg.empty());
        }
    };

    // Create vector with sizes 0 - 9.
    std::vector<size_t> sizes(10);
    std::iota(sizes.begin(), sizes.end(), 0);
    
    // Disable individual sizes across multiple lines.
    // Include extra lines for an alternate arch and an alternate test,
    // which should not affect the result.
    test_size(
        HipcubTestControllerTests::generate_control_text({
            {R"(HipcubTestControllerTests\.CheckSizeEnablement)", arch, "7", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.CheckSizeEnablement)", arch, "5", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.CheckSizeEnablement)", alt_arch, "3", "*", "Skipping for unit test."},
            {R"(HipcubTestControllerTests\.FilterSizes)", arch, "4", "*", "Skipping for unit test."}
        }),
        sizes,
        std::vector<bool>({true, true, true, true, true, false, false, false, false, false})
    );
}

}
