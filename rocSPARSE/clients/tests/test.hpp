/*! \file */
/* ************************************************************************
* Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#pragma once

#include "rocsparse_data.hpp"
#include "rocsparse_reproducibility.hpp"
#include "rocsparse_reproducibility_test_label.hpp"
#include "rocsparse_test.hpp"
#include "rocsparse_test_template_traits.hpp"

#include "test_check.hpp"
void rocsparse_seedrand();

template <rocsparse_test_enum::value_type ROUTINE, typename F>
inline void testing_reproducibility(const Arguments& arg, F test_execute)
{
    using status_t = rocsparse_reproducibility_t::status_t;

    rocsparse_reproducibility_t::results_test_t results_test;
    results_test.set_name(
        std::string(arg.category) + std::string("/") + std::string(arg.function) + std::string(".")
        + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::string err_description = "<none>";

    if(false == rocsparse_reproducibility_t::instance().is_enabled() || arg.skip_reproducibility)
    {
        return;
    }

    auto& test = rocsparse_reproducibility_t::instance().test();
    test.reset();

    //
    // Reset random.
    //
    rocsparse_seedrand();

    auto status = status_t::reproducible;

    //
    // Run the first time.
    //
    try
    {
        test_execute(arg);
        if(::testing::Test::HasFailure())
        {
            status = status_t::test_failed;
        }
    }
    catch(...)
    {
        status = status_t::execution_failed;
    }

    if(status == status_t::reproducible)
    {
        auto* test_data_initial = test.get_initial_data();
        if(test_data_initial == nullptr)
        {
            // no reproducibility data
            status = status_t::non_tested;
        }
        else
        {
            // tell the test we will loop now.
            test.set_next();

            //
            results_test.set_hash(test_data_initial->compute_hash());
            const uint32_t rn = rocsparse_reproducibility_t::instance().get_num_iterations();
            auto           next_status = status_t::reproducible;
            for(uint32_t ri = 1; ri < rn; ++ri)
            {
                //
                // Reinitialize random numbers.
                //
                rocsparse_seedrand();

                //
                // Reset the next one.
                //
                test.reset_next();
                try
                {
                    test_execute(arg);
                    if(::testing::Test::HasFailure())
                    {
                        next_status = status_t::test_failed;
                        break;
                    }
                    else
                    {
                        auto*             current_data = test.get_current_data();
                        std::stringstream descr_binary_data_error;
                        const bool        are_same
                            = test_data_initial->is_same(current_data, descr_binary_data_error);
                        const uint64_t hash_current_data = current_data->compute_hash();
                        if(are_same == false)
                        {
                            next_status = status_t::non_reproducible;

                            results_test.set_description(descr_binary_data_error.str());
#if NDEBUG
                            if(hash_current_data == test_data_initial->compute_hash())
                            {
                                std::cout << "rocSPARSE.REPRODUCIBILITY collision hashing"
                                          << std::endl;
                            }
#endif
                            break;
                        }
                        else
                        {
                            if(hash_current_data != test_data_initial->compute_hash())
                            {
                                std::cerr << "rocSPARSE.REPRODUCIBILITY ERROR: Inconsistent "
                                             "reproducibility hashing, data are binary identical "
                                             "but hash values are different, "
                                          << hash_current_data << " vs "
                                          << test_data_initial->compute_hash()
                                          << ". rocSPARSE.REPRODUCIBILITY Abort." << std::endl;
                                throw(rocsparse_status_internal_error);
                            }
                        }
                    }
                }
                catch(...)
                {
                    next_status = status_t::execution_failed;
                }
            }
            if(next_status != status_t::reproducible)
            {
                status = next_status;
            }
        }
    }

    //
    // Set the status.
    //
    results_test.set_status(status);

    rocsparse_test_functors<ROUTINE>::set_reproducibility_test(arg, results_test);
    test.reset();
}

//
// INTERNAL MACRO TO SPECIALIZE TEST CALL NEEDED TO INSTANTIATE
//
#define SPECIALIZE_ROCSPARSE_TEST_CALL(ROUTINE)                                                   \
    template <>                                                                                   \
    struct rocsparse_test_call<rocsparse_test_enum::ROUTINE>                                      \
    {                                                                                             \
        template <typename... P>                                                                  \
        static void testing_bad_arg(const Arguments& arg)                                         \
        {                                                                                         \
            test_check::reset_auto_testing_bad_arg();                                             \
            const int state_debug_arguments_verbose = rocsparse_state_debug_arguments_verbose();  \
            if(state_debug_arguments_verbose == 1)                                                \
            {                                                                                     \
                rocsparse_disable_debug_arguments_verbose();                                      \
            }                                                                                     \
            testing_##ROUTINE##_bad_arg<P...>(arg);                                               \
            if(state_debug_arguments_verbose == 1)                                                \
            {                                                                                     \
                rocsparse_enable_debug_arguments_verbose();                                       \
            }                                                                                     \
            if(false && false == test_check::did_auto_testing_bad_arg())                          \
            {                                                                                     \
                std::cerr << "rocsparse_test warning testing bad arguments of "                   \
                          << rocsparse_test_enum::to_string(rocsparse_test_enum::ROUTINE)         \
                          << " must use auto_testing_bad_arg, or bad_arg_analysis." << std::endl; \
                CHECK_ROCSPARSE_ERROR(rocsparse_status_internal_error);                           \
            }                                                                                     \
        }                                                                                         \
        static void testing_extra(const Arguments& arg)                                           \
        {                                                                                         \
            try                                                                                   \
            {                                                                                     \
                testing_##ROUTINE##_extra(arg);                                                   \
            }                                                                                     \
            catch(rocsparse_status & status)                                                      \
            {                                                                                     \
                CHECK_ROCSPARSE_ERROR(status);                                                    \
            }                                                                                     \
            catch(hipError_t & error)                                                             \
            {                                                                                     \
                CHECK_HIP_ERROR(error);                                                           \
            }                                                                                     \
            catch(std::exception & error)                                                         \
            {                                                                                     \
                CHECK_ROCSPARSE_ERROR(rocsparse_status_thrown_exception);                         \
            }                                                                                     \
        }                                                                                         \
                                                                                                  \
        template <typename... P>                                                                  \
        static void testing(const Arguments& arg)                                                 \
        {                                                                                         \
            if(rocsparse_reproducibility_t::instance().is_enabled()                               \
               && arg.skip_reproducibility == false)                                              \
            {                                                                                     \
                testing_reproducibility<rocsparse_test_enum::ROUTINE>(arg,                        \
                                                                      testing_##ROUTINE<P...>);   \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                try                                                                               \
                {                                                                                 \
                    testing_##ROUTINE<P...>(arg);                                                 \
                }                                                                                 \
                catch(rocsparse_status & status)                                                  \
                {                                                                                 \
                    CHECK_ROCSPARSE_ERROR(status);                                                \
                }                                                                                 \
                catch(hipError_t & error)                                                         \
                {                                                                                 \
                    CHECK_HIP_ERROR(error);                                                       \
                }                                                                                 \
                catch(std::exception & error)                                                     \
                {                                                                                 \
                    CHECK_ROCSPARSE_ERROR(rocsparse_status_thrown_exception);                     \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
    };

/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// INTERNAL MACRO TO SPECIALIZE TEST FUNCTOR NEEDED TO INSTANTIATE
//
#define SPECIALIZE_ROCSPARSE_TEST_FUNCTORS(ROUTINE, ...)                                           \
    template <>                                                                                    \
    struct rocsparse_test_functors<rocsparse_test_enum::ROUTINE>                                   \
    {                                                                                              \
        static void set_reproducibility_test(                                                      \
            const Arguments& arg, const rocsparse_reproducibility_t::results_test_t& results_test) \
        {                                                                                          \
            rocsparse_reproducibility_t::results_test_input_t results_test_input;                  \
            rocsparse_reproducibility_utils::record(results_test_input, __VA_ARGS__);              \
            rocsparse_reproducibility_t::instance().results().add(                                 \
                arg.function, results_test_input, results_test);                                   \
        }                                                                                          \
        static std::string name_suffix(const Arguments& arg)                                       \
        {                                                                                          \
            std::ostringstream s;                                                                  \
            rocsparse_test_name_suffix_generator(s, __VA_ARGS__);                                  \
            return s.str();                                                                        \
        }                                                                                          \
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// INTERNAL MACRO TO SPECIALIZE TEST TRAITS NEEDED TO INSTANTIATE
//
#define SPECIALIZE_ROCSPARSE_TEST_TRAITS(ROUTINE, CONFIG)                                     \
    /**/ template <> /**/ struct rocsparse_test_traits<rocsparse_test_enum::ROUTINE> : CONFIG \
    /**/ {                                                                                    \
    /**/ }
/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// INSTANTIATE TESTS
//

template <rocsparse_test_enum::value_type ROUTINE>
using test_template_traits_t
    = rocsparse_test_template_traits<ROUTINE, rocsparse_test_traits<ROUTINE>::s_dispatch>;

template <rocsparse_test_enum::value_type ROUTINE>
using test_dispatch_t = rocsparse_test_dispatch<rocsparse_test_traits<ROUTINE>::s_dispatch>;

#define INSTANTIATE_ROCSPARSE_TEST(ROUTINE, CATEGORY)                                          \
    /**/ using ROUTINE = test_template_traits_t<rocsparse_test_enum::ROUTINE>::filter;         \
    /**/                                                                                       \
    /**/ template <typename... P>                                                              \
    /**/ using ROUTINE##_call                                                                  \
        = test_template_traits_t<rocsparse_test_enum::ROUTINE>::caller<P...>;                  \
    /**/                                                                                       \
    /**/ TEST_P(ROUTINE, CATEGORY)                                                             \
    /**/ {                                                                                     \
        /**/ test_dispatch_t<rocsparse_test_enum::ROUTINE>::template dispatch<ROUTINE##_call>( \
            GetParam());                                                                       \
    /**/ }                                                                                     \
    /**/                                                                                       \
    /**/ INSTANTIATE_TEST_CATEGORIES(ROUTINE)

/////////////////////////////////////////////////////////////////////////////////////////////////////

//
// DEFINE ALL REQUIRED INFORMATION FOR A TEST ROUTINE BUT WITH A PREDEFINED CONFIGURATION
// (i.e. [T (default) | <I,T> | <I,J,T>] + a selection of numeric types (all (default), real_only, complex_only, some other specific situations (?) ) )
//
#define TEST_ROUTINE_WITH_CONFIG(ROUTINE, CATEGORY, CONFIG, ...)   \
    /**/                                                           \
    /**/ SPECIALIZE_ROCSPARSE_TEST_TRAITS(ROUTINE, CONFIG);        \
    /**/ SPECIALIZE_ROCSPARSE_TEST_FUNCTORS(ROUTINE, __VA_ARGS__); \
    /**/ SPECIALIZE_ROCSPARSE_TEST_CALL(ROUTINE);                  \
    /**/ namespace                                                 \
    /**/ {                                                         \
        /**/ INSTANTIATE_ROCSPARSE_TEST(ROUTINE, CATEGORY);        \
    /**/ }

//
// DEFINE ALL REQUIRED INFORMATION FOR A TEST ROUTINE WITH A DEFAULT CONFIGURATION (i.e  T + all numeric types)
//
#define TEST_ROUTINE(ROUTINE, CATEGORY, ...) \
    TEST_ROUTINE_WITH_CONFIG(ROUTINE, CATEGORY, rocsparse_test_config, __VA_ARGS__)
/////////////////////////////////////////////////////////////////////////////////////////////////////
