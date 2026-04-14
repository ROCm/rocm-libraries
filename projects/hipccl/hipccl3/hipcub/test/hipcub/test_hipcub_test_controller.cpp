#include <iostream>
#include <sstream>
#include <regex>
#include <vector>
#include <numeric>
#include <optional>

// Google Test
#include <gtest/gtest.h>

#if __has_include(<valgrind/valgrind.h>)
    #include <valgrind/valgrind.h>
    #define HAS_VALGRIND_H 1
#else
    #define HAS_VALGRIND_H 0
#endif

#if defined(__SANITIZE_ADDRESS__)
    #define IS_ASAN_BUILD 1
#elif defined(__has_feature)
    #if __has_feature(address_sanitizer)
        #define IS_ASAN_BUILD 1
    #endif
#endif
#ifndef IS_ASAN_BUILD
    #define IS_ASAN_BUILD 0
#endif

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
			ss << "/" << line.test << "/ : /" << line.arch << "/ : " << line.size << " : " << line.build_type << " : \"" << line.skip_msg << "\"" << std::endl;
		return ss.str();
	};

	static bool is_running_valgrind()
	{
		bool result = false;
#if HAS_VALGRIND_H
		result = RUNNING_ON_VALGRIND;
#endif
		return result;
	}

	static bool is_running_asan()
	{
		return static_cast<bool>(IS_ASAN_BUILD);
	}
};

TEST_F(HipcubTestControllerTests, GetArch)
{
	int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));
	
	const std::string arch = TestController::get_arch();
	const std::regex arch_regex("^gfx[0-9a-f]+$");
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
				{"HipcubTestControllerTests\\..*", arch, "*", "*", "Skipping for unit test."}
			}),
		false
	);

	// When using individual sizes, test case should be enabled.
	test_enablement(
		HipcubTestControllerTests::generate_control_text({
				{"HipcubTestControllerTests\\..*", arch, "1000", "*", "Skipping for unit test."}
		}),
		true
	);
	
	// When using alternate arch, test case should be enabled.
	test_enablement(
		HipcubTestControllerTests::generate_control_text({
				{"HipcubTestControllerTests\\..*", alt_arch, "1000", "*", "Skipping for unit test."}
		}),
		true
	);

	// When using exact test name, test case should not be enabled.
	test_enablement(
		HipcubTestControllerTests::generate_control_text({
				{"HipcubTestControllerTests\\.CheckTestEnablement", arch, "*", "*", "Skipping for unit test."}
		}),
		false
	);

	// When using alternate test name, test case should be enabled.
	test_enablement(
		HipcubTestControllerTests::generate_control_text({
				{"HipcubTestControllerTests\\..NonExistantTest", arch, "*", "*", "Skipping for unit test."}
		}),
		true
	);
	
	// With multiple lines that cover the same test, one with * and others with individual sizes,
	// Test should not be enabled.
	test_enablement(
		HipcubTestControllerTests::generate_control_text({
				{"HipcubTestControllerTests\\..*", arch, "1000", "*", "Skipping for unit test."},
				{"HipcubTestControllerTests\\..*", arch, "*", "*", "Skipping for unit test."},
				{"HipcubTestControllerTests\\..*", arch, "2000", "*", "Skipping for unit test."}
		}),
		false
	);

	// Use keyword in arch. Test should not be enabled.
	test_enablement(
		HipcubTestControllerTests::generate_control_text({
				{"HipcubTestControllerTests\\..*", arch, "1000", "*", "Skipping for unit test."},
				{"HipcubTestControllerTests\\..*", "<all>", "*", "*", "Skipping for unit test."},
				{"HipcubTestControllerTests\\..*", arch, "2000", "*", "Skipping for unit test."}
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
	
	const auto test_filter = [&controller](const std::string& text,
										   std::vector<size_t>& sizes,
										   const std::vector<size_t>& expected_sizes)
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
		const std::vector<size_t> result = controller.filter_sizes(sizes, msg);
		ASSERT_EQ(result, expected_sizes);
		ASSERT_EQ(expect_filtering, !msg.empty());
	};

	// Create vector with sizes 0 - 9.
	std::vector<size_t> sizes(10);
	std::iota(sizes.begin(), sizes.end(), 0);

	// Filter out individual sizes across multiple lines.
	// Include a line for an alternate arch, which should not affect the result.
	std::vector<size_t> sizes_copy(sizes);
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", arch, "0,5", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "7", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", alt_arch, "3", "*", "Skipping for unit test."}
		}),
		sizes_copy,
		{1, 2, 3, 4, 6, 8, 9}
	);

	// Filter using operators <, > across multiple lines.
	// Include a line for an alternate test, which should not affect the result.
	sizes_copy = sizes;
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", arch, "<1", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, ">6", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.CheckSizeEnablement", alt_arch, "3", "*", "Skipping for unit test."}
		}),
		sizes_copy,
		{1, 2, 3, 4, 5, 6}
	);

	// Filter using operators <=, >= across multiple lines.
	sizes_copy = sizes;
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", arch, "<=1", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, ">=6", "*", "Skipping for unit test."},
		}),
		sizes_copy,
		{2, 3, 4, 5}
	);

	// Filter using multiple operators on same line.
	sizes_copy = sizes;
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", arch, "<=1,5,>7", "*", "Skipping for unit test."}
		}),
		sizes_copy,
		{2, 3, 4, 6, 7}
	);

	// Filter using operator that removes all sizes, with additional lines before and after.
	sizes_copy = sizes;
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", arch, ">=4", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, ">=0", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "<6", "*", "Skipping for unit test."}
		}),
		sizes_copy,
		{}
	);

	// Filter using operators and *.
	sizes_copy = sizes;
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", arch, ">=4", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "*", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "<6", "*", "Skipping for unit test."}
		}),
		sizes_copy,
		{}
	);

	// Use keywords in arch.
	sizes_copy = sizes;
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", "<mi300-family>|" + arch, "<9", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, ">9", "*", "Skipping for unit test."}
		}),
		sizes_copy,
		{9}
	);

	// Use a single arithmetic expressions per line.
	sizes_copy = sizes;
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			//                                                Expression:               Parenthesized Equivalent:
			{"HipcubTestControllerTests\\.FilterSizes", arch, "2 << 1 + 1", "*", "Skipping for unit test."},            // 2 << (1 + 1)             = 8
			{"HipcubTestControllerTests\\.FilterSizes", arch, "(2 << 2) + 1", "*", "Skipping for unit test."},          //                          = 9
			{"HipcubTestControllerTests\\.FilterSizes", arch, "8 / 4 >> 2", "*", "Skipping for unit test."},            // (8 / 4) >> 2             = 0
			{"HipcubTestControllerTests\\.FilterSizes", arch, "7 - 3 * 2", "*", "Skipping for unit test."},             // 7 - (3 * 2)              = 1
			{"HipcubTestControllerTests\\.FilterSizes", arch, "2 * (1 + (1 << 2) / 2)", "*", "Skipping for unit test."} // 2 * (1 + ((1 << 2) / 2)) = 6
		}),
		sizes_copy,
		{2, 3, 4, 5, 7}
	);

	// Use multiple arithmetic expressions per line.
	sizes_copy = sizes;
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", arch, "1 << 1 + 2, 3 * 2", "*", "Skipping for unit test."},             // 8, 6
			{"HipcubTestControllerTests\\.FilterSizes", arch, "7 - 0 * 4, 6 >> 2", "*", "Skipping for unit test."},             // 7, 1
			{"HipcubTestControllerTests\\.FilterSizes", arch, "6 * 2 - 10, 81 / 9 - 9, 2 * 4", "*", "Skipping for unit test."}, // 2, 0, 8
		}),
		sizes_copy,
		{3, 4, 5, 9}
	);

	// Test intermediate results that drop below 0.
	// This should be fine, as long as the rest of the expression causes the final result to end up being >= 0.
	sizes_copy = sizes;
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", arch, "(1 - 2) + 3", "*", "Skipping for unit test."}, // 2
			{"HipcubTestControllerTests\\.FilterSizes", arch, "0 - (1 << 32) + (1 << 32)", "*", "Skipping for unit test."} // 0
		}),
		sizes_copy,
		{1, 3, 4, 5, 6, 7, 8, 9}
	);

	// Test with different build types.
	sizes_copy = sizes;
	const bool is_asan = HipcubTestControllerTests::is_running_asan();
	const bool is_valgrind = HipcubTestControllerTests::is_running_valgrind();
	std::vector<size_t> expected(sizes);
	
	expected.erase(std::find(expected.begin(), expected.end(), 3));
	expected.erase(std::find(expected.begin(), expected.end(), 4));
	expected.erase(std::find(expected.begin(), expected.end(), 5));
	
	if (is_asan)
	{
		expected.erase(std::find(expected.begin(), expected.end(), 0));
		expected.erase(std::find(expected.begin(), expected.end(), 2));
	}
	if (is_valgrind)
	{
		expected.erase(std::find(expected.begin(), expected.end(), 1));
		// The 2 may have already been removed by the if block above
		const auto it = std::find(expected.begin(), expected.end(), 2);
		if (it != expected.end())
			expected.erase(it);
	}
		
	test_filter(
		HipcubTestControllerTests::generate_control_text({
			{"HipcubTestControllerTests\\.FilterSizes", arch, "0", "asan", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "1", "valgrind", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "2", "asan, valgrind", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "3", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "4", "asan, *", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "5", "*, valgrind", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", alt_arch, "6", "*", "Skipping for unit test."}
		}),
		sizes_copy,
		expected
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
	
	const auto test_size = [&controller](const std::string& text,
										 const std::vector<size_t>& sizes,
										 const std::vector<bool>& expected_results)
	{
		controller.reset(std::make_optional(text));
		for (size_t i = 0; i < sizes.size(); i++)
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
			{"HipcubTestControllerTests\\.CheckSizeEnablement", arch, "0,5", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.CheckSizeEnablement", arch, "7", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.CheckSizeEnablement", alt_arch, "3", "*", "Skipping for unit test."},
			{"HipcubTestControllerTests\\.FilterSizes", arch, "4", "*", "Skipping for unit test."}
		}),
		sizes,
		std::vector<bool>({false, true, true, true, true, false, true, false, true, true})
	);
}

}
