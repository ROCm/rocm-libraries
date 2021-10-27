#pragma once

#include <unittest/testframework.h>
#include <thrust/system/hip/memory.h>
#include <thrust/system_error.h>
#include <vector>

class HIPTestDriver
  : public UnitTestDriver
{
  public:
    int current_device_architecture() const;

    bool supports_managed_memory() const;

  private:
    std::vector<int> target_devices(const ArgumentMap &kwargs);

    bool check_hip_error(bool concise);

    virtual bool post_test_smoke_check(const UnitTest &test, bool concise);

    virtual bool run_tests(const ArgumentSet &args, const ArgumentMap &kwargs);
};

UnitTestDriver &driver_instance(thrust::system::hip::tag);
