# hipDNN Testing

This document provides an overview of hipDNN's testing approach and links to detailed testing documentation.

## Running Tests

Prior to running tests, follow the [Quick Start Guide](./Building.md#quick-start-guide) to clone hipDNN and prepare your environment.

Afterwards, proceed to run the tests:
```bash
cmake -GNinja .. 

# Run all tests with CTest (recommended)
ninja check_ctest

# Alternatively, run with GTest
ninja check

# Run specific test categories
ninja unit-check        # Unit tests only
ninja integration-check # Integration tests only

# Run with Address Sanitizer
cmake -GNinja -DBUILD_ADDRESS_SANITIZER=ON ..
ninja check_ctest
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
| SDK | `sdk/tests/` | Unit tests |
| Plugins | `plugins/<name>/tests/` | Unit tests |
| Plugins | `plugins/<name>/integration_tests/` | Integration tests |
| API | `tests/backend/` | Black box API tests |
| Frontend Integration | `tests/frontend/` | Integration tests |

### Testing Requirements

- **Coverage Target**: 80% overall, with each component maintaining >80% individually
- **GPU Tests**: Must be marked with `SKIP_IF_NO_DEVICE()` macro
- **Platform Support**: All tests must work on Windows and Linux
- **Performance**: Unit tests must execute quickly
- **CI**: All CI pipelines must pass on every PR
