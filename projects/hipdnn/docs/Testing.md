# hipDNN Testing

This document provides an overview of hipDNN's testing approach and links to detailed testing documentation.

## Running Tests

Prior to running tests, follow the [Quick Start Guide](./Building.md#quick-start-guide) to clone hipDNN and prepare your environment.

Afterwards, proceed to run the tests:
```bash
# Configure the project (if not already configured).
cmake -GNinja ..

# Build and run all tests
ninja check

# Run all tests with additional details
ninja check-verbose

# Run specific test categories
ninja unit-check        # Unit tests only. Also `unit-check-verbose`.
ninja integration-check # Integration tests only. Also `integration-check-verbose`.

# Run with Address Sanitizer
cmake -GNinja -DBUILD_ADDRESS_SANITIZER=ON ..
ninja check
```

## Testing Documentation

### [Testing Strategy](./testing/TestingStrategy.md)
Comprehensive guide covering hipDNN's multi-layered testing approach:
- White box testing (unit tests) for internal implementations
- Black box testing (API tests) for public interfaces
- Integration testing for end-to-end functionality
- Performance testing roadmap

### [Test Plan](./testing/TestPlan.md)
Release testing checklist with prerequisites and test cases:
- CI pipeline verification
- Documentation currency checks
- Expected results from regular tests and ASAN

### [Test Run Template](./testing/TestRunTemplate.md)
Standardized template for recording and tracking test results:
- Test environment documentation
- Result recording format
- Example test runs
- Best practices for test reporting

## Quick Reference

### Test Organization

| Component | Test Location | Type |
|-----------|--------------|------|
| Backend | `backend/tests/` | Unit tests |
| Frontend | `frontend/tests/` | Unit tests |
| Data SDK | `data_sdk/tests/` | Unit tests |
| Plugin SDK | `plugin_sdk/tests/` | Unit tests |
| Test SDK | `test_sdk/tests/` | Unit tests |
| Plugins | `plugins/<name>/tests/` | Unit tests |
| Plugins | `plugins/<name>/integration_tests/` | Integration tests |
| API | `tests/backend/` | Black box API tests |
| Frontend Integration | `tests/frontend/` | Integration tests |

### Choosing Between TYPED_TEST and TEST_P

```
Need to test across multiple data types (float, half, bfloat16)?
│
├── NO → Use TEST() or TEST_F()
│
└── YES → Also need parameterized test cases?
          │
          ├── NO → Use TYPED_TEST
          │        (Compile-time type safety, simple)
          │
          └── YES → Use TEST_P with multi-declarations
                    (Handles both types AND parameters)
```

**Key Principle:** Prefer `TEST_P` over `TYPED_TEST` when both type variation and parameterized cases are needed. `TYPED_TEST` and `TEST_P` don't mix well together.

| Scenario | Approach |
|----------|----------|
| Single type, no params | `TEST()` / `TEST_F()` |
| Single type, with params | `TEST_P()` |
| Multi-type, no params | `TYPED_TEST` |
| Multi-type, with params | `TEST_P` + multi-declarations |

### Multi-Declaration Pattern (for types + parameters)

When you need both type variation AND parameterized tests, use explicit type aliases with `TEST_P`:

```cpp
template <typename DataType>
class ConvTest : public ::testing::TestWithParam<ConvTestCase> { };

// Explicit type aliases - one per type
using ConvTestFp32 = ConvTest<float>;
using ConvTestFp16 = ConvTest<half>;
using ConvTestBfp16 = ConvTest<hip_bfloat16>;

// Separate TEST_P for each type
TEST_P(ConvTestFp32, Correctness) { runTest(); }
TEST_P(ConvTestFp16, Correctness) { runTest(); }
TEST_P(ConvTestBfp16, Correctness) { runTest(); }

// Separate instantiation for each type
INSTANTIATE_TEST_SUITE_P(Smoke, ConvTestFp32, testing::ValuesIn(getCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, ConvTestFp16, testing::ValuesIn(getCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, ConvTestBfp16, testing::ValuesIn(getCases()));
```

**Why multi-declarations over macros?** Modern tooling makes boilerplate easy to handle, and avoiding macros makes tests easier to debug and understand.

### Type Combinations with TypePair

For testing type combinations (e.g., input type + compute type), define a struct to hold the types:

```cpp
template <typename T1, typename T2>
struct TypePair
{
    using InputType = T1;
    using ComputeType = T2;
};

using TypeCombinations = ::testing::Types<TypePair<float, float>,
                                          TypePair<half, float>,
                                          TypePair<hip_bfloat16, float>>;
TYPED_TEST_SUITE(MyTypedTest, TypeCombinations);

TYPED_TEST(MyTypedTest, Correctness)
{
    using InputType = typename TypeParam::InputType;
    using ComputeType = typename TypeParam::ComputeType;
    // Test implementation using InputType and ComputeType
}
```

### Testing Requirements

- **Coverage Target**: 80% overall, with each component maintaining >80% individually
- **GPU Tests**: Must be marked with `SKIP_IF_NO_DEVICE()` macro
- **Platform Support**: All tests must work on Windows and Linux
- **Performance**: Unit tests must execute quickly
- **CI**: All CI pipelines must pass on every PR
