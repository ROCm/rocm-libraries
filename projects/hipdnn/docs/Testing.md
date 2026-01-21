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

### Multi-Datatype Testing with TYPED_TEST

When testing functionality that should work across multiple data types (float, half, bfloat16), use Google Test's `TYPED_TEST` to avoid code duplication:

```cpp
#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

template <typename T>
class MyTypedTest : public ::testing::Test { };

using TestTypes = ::testing::Types<float, __half, hip_bfloat16>;
TYPED_TEST_SUITE(MyTypedTest, TestTypes);

TYPED_TEST(MyTypedTest, TestSomething)
{
    // TypeParam is the current type (float, __half, or hip_bfloat16)
    TypeParam value = static_cast<TypeParam>(1.0f);
    // Test implementation runs for each type
}
```

**When to use TYPED_TEST:**
- GPU kernel tests that should work for multiple precisions
- Validation utilities that are type-agnostic
- Any test logic that applies identically across float, half, and bfloat16

### Testing Requirements

- **Coverage Target**: 80% overall, with each component maintaining >80% individually
- **GPU Tests**: Must be marked with `SKIP_IF_NO_DEVICE()` macro
- **Platform Support**: All tests must work on Windows and Linux
- **Performance**: Unit tests must execute quickly
- **CI**: All CI pipelines must pass on every PR
