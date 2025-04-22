
#include <miopen/miopen.h>

#include "get_handle.hpp"
#include <gtest/gtest.h>
#include "miopen/db_path.hpp"
#include "../lib_env_var.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_USER_DB_PATH)

class CPU_TuningPolicy_NONE : public ::testing::Test
{
};

TEST_F(CPU_TuningPolicy_NONE, TestTuningPolicyGetterAndSetter)
{
    auto&& handle = get_handle();
    // test initial value
    miopenTuningPolicy_t test_tuning_policy;
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyNone);

    // test setting
    EXPECT_EQ(miopenSetTuningPolicy(&handle, miopenTuningPolicy_t::miopenTuningPolicyDbUpdate),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyDbUpdate);

    // test by casting
    EXPECT_EQ(miopenSetTuningPolicy(&handle, static_cast<miopenTuningPolicy_t>(4)),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicySearchDbUpdate);

    // reset back to none
    EXPECT_EQ(miopenSetTuningPolicy(&handle, miopenTuningPolicy_t::miopenTuningPolicyNone),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyNone);
}
