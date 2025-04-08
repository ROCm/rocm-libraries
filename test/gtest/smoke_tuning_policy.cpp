
#include <miopen/miopen.h>

#include "get_handle.hpp"
#include <gtest/gtest.h>
#include "miopen/db_path.hpp"
#include "../lib_env_var.hpp"

MIOPEN_LIB_ENV_VAR(MIOPEN_USER_DB_PATH)

class TuningPolicy : public ::testing::Test
{
};

TEST_F(TuningPolicy, TestTuningPolicyGetterAndSetter)
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
    EXPECT_EQ(miopenSetTuningPolicy(&handle, static_cast<miopenTuningPolicy_t>(2)),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyDbUpdate);

    miopenStatus_t status = miopenSetTuningPolicy(&handle, static_cast<miopenTuningPolicy_t>(4));
    EXPECT_EQ(status, miopenStatusBadParm);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyDbUpdate);

    // reset back to none
    EXPECT_EQ(miopenSetTuningPolicy(&handle, miopenTuningPolicy_t::miopenTuningPolicyNone),
              miopenStatusSuccess);
    EXPECT_EQ(miopenGetTuningPolicy(&handle, &test_tuning_policy), miopenStatusSuccess);
    EXPECT_EQ(test_tuning_policy, miopenTuningPolicy_t::miopenTuningPolicyNone);
}
