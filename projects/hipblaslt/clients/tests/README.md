# Gtest

hipblaslt-test is the main regression gtest for hipBLASLt. All test items should pass.

```shell
# Go to hipBLASLt build directory
cd hipBLASLt; cd build/release

# Before chip back, run gfx13 & matmul_medium test suite
# due to the long runtime on FFM.
# If want to run on ffm full -> export HIPBLASLT_TEST_TIMEOUT=6000
./clients/hipblaslt-test --gtest_filter=*gfx13*:*matmul_medium*

# Run full gtest tests
./clients/hipblaslt-test

# Run gtest tests with filter
./clients/hipblaslt-test --gtest_filter=<test pattern>

# Demo: gtest tests with filter
./clients/hipblaslt-test --gtest_filter=*quick*

# Demo: DRelu gradient matmul tests
./clients/hipblaslt-test --gtest_filter=*drelu*
```
