
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/Utilities/Utils.hpp>

struct HipSuccessMatcher : Catch::Matchers::MatcherGenericBase
{
    HipSuccessMatcher() {}

    bool match(hipError_t result) const
    {
        if(result != hipSuccess)
        {
            m_message = hipGetErrorString(result);
        }

        return result == hipSuccess;
    }

    std::string describe() const override
    {
        if(m_message.empty())
            return "Hip success";
        else
            return "Hip error: " + m_message;
    }

private:
    mutable std::string m_message;
};

/**
 * Checks that the expression executes and returns hipSuccess (== 0).
 * Gets the Hip error on failure.
 */
inline auto HasHipSuccess(int val = 0)
{
    return HipSuccessMatcher();
}

template <typename Value>
struct DeviceScalarMatcher : Catch::Matchers::MatcherGenericBase
{
    DeviceScalarMatcher(Value const& value)
        : m_value(value)
    {
    }

    template <typename D_Value>
    bool match(D_Value const* d_ptr) const
    {
        D_Value h_value;

        HipSuccessMatcher hipMatcher;

        if(!hipMatcher.match(hipMemcpy(&h_value, d_ptr, sizeof(D_Value), hipMemcpyDefault)))
        {
            m_message += hipMatcher.describe();
        }

        if(h_value != m_value)
        {
            m_message += rocRoller::concatenate(
                " (Device value: ", h_value, ")\nExpected Value: ", m_value);
            return false;
        }

        return true;
    }

    template <typename D_Value>
    bool match(std::shared_ptr<D_Value> const& d_ptr) const
    {
        return match(d_ptr.get());
    }

    std::string describe() const override
    {
        if(m_message.empty())
            return "";
        return "Scalar: " + m_message;
    }

private:
    Value               m_value;
    mutable std::string m_message;
};

/**
 * Checks that the expression is a device pointer (or shared_ptr) which when copied to the host
 * equals 'value'.
 */
template <typename Value>
auto HasDeviceScalarEqualTo(Value value)
{
    return DeviceScalarMatcher<Value>{value};
}

template <typename SubMatcher>
struct CustomDeviceScalarMatcher : Catch::Matchers::MatcherGenericBase
{
    CustomDeviceScalarMatcher(SubMatcher matcher)
        : m_matcher(std::move(matcher))
    {
    }

    template <typename D_Value>
    bool match(D_Value const* d_ptr) const
    {
        D_Value h_value;

        HipSuccessMatcher hipMatcher;

        if(!hipMatcher.match(hipMemcpy(&h_value, d_ptr, sizeof(D_Value), hipMemcpyDefault)))
        {
            m_message += hipMatcher.describe();
        }

        m_message += rocRoller::concatenate("Device value: ", h_value);

        return m_matcher.match(h_value);
    }

    template <typename D_Value>
    bool match(std::shared_ptr<D_Value> const& d_ptr) const
    {
        return match(d_ptr.get());
    }

    std::string describe() const override
    {
        if(m_message.empty())
            return m_matcher.describe();
        return "(" + m_message + ")\n" + m_matcher.describe();
    }

private:
    SubMatcher          m_matcher;
    mutable std::string m_message;
};

/**
 * Allows checking a single on-device value against any matcher (such as the built-in Catch
 * ULP matcher).
 *
 * Checks that the expression is a device pointer (or shared_ptr), which when copied to the host
 * satisfies the provided matcher.
 */
template <typename Matcher>
auto HasDeviceScalar(Matcher matcher)
{
    return CustomDeviceScalarMatcher{std::move(matcher)};
}

template <typename Value>
struct DeviceVectorMatcher : Catch::Matchers::MatcherGenericBase
{
    DeviceVectorMatcher(std::vector<Value> const& values)
        : m_values(values)
    {
    }

    template <typename D_Value>
    bool match(D_Value const* d_ptr) const
    {
        std::vector<D_Value> h_values(m_values.size());

        HipSuccessMatcher hipMatcher;

        if(!hipMatcher.match(hipMemcpy(
               h_values.data(), d_ptr, sizeof(D_Value) * h_values.size(), hipMemcpyDefault)))
        {
            m_message += hipMatcher.describe();
        }

        if(h_values != m_values)
        {
            for(size_t i = 0; i < h_values.size(); i++)
            {
                if(h_values[i] not_eq m_values[i])
                {
                    m_message += rocRoller::concatenate("Index: ",
                                                        i,
                                                        " (Device value: ",
                                                        h_values[i],
                                                        ")\nExpected Value: ",
                                                        m_values[i]);
                }
            }
            return false;
        }

        return true;
    }

    template <typename D_Value>
    bool match(std::shared_ptr<D_Value> const& d_ptr) const
    {
        return match(d_ptr.get());
    }

    std::string describe() const override
    {
        if(m_message.empty())
            return "";
        return "Vector: " + m_message;
    }

private:
    std::vector<Value>  m_values;
    mutable std::string m_message;
};

/**
 * Checks that the values from a device pointer (or `shared_ptr`) which when copied to
 * the host equal `m_values` in DeviceVectorMatcher.
 */
template <typename Value>
auto HasDeviceVectorEqualTo(std::vector<Value> const& values)
{
    return DeviceVectorMatcher<Value>{values};
}
