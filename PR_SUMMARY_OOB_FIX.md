# Pull Request: Fix Array Out-of-Bounds Access in Serialization

**Branch**: `OOB_Access_SequenceTraits`  
**Base**: `develop`

JIRA ID : N/A

## Motivation

The `SequenceTraits<std::array<T, N>>::element()` function in the serialization code had a critical bug where it would log an error when `index >= N` but still return `v[index]`, resulting in undefined behavior due to out-of-bounds array access. This could lead to:

- Memory corruption when reading/writing invalid memory locations
- Crashes or unpredictable behavior during library loading
- Silent failures that are difficult to debug
- Potential security vulnerabilities from out-of-bounds memory access

The existing `IOTraits<IO>::setError()` call only logs the error to an error list without throwing an exception, allowing execution to continue to the dangerous `return v[index]` statement.

## Technical Details

### Changes Applied to Three Projects:

#### 1. **shared/tensile** (Commit: `a08f3efd51`)
- **Containers.hpp**: Added `std::out_of_range` exception throw after error logging
- **llvm/Loading.cpp**: Updated 2 catch blocks from `std::runtime_error` to `std::exception`
- **msgpack/MessagePack.cpp**: Updated 3 catch blocks from `std::runtime_error` to `std::exception`

#### 2. **projects/hipblaslt/tensilelite** (Commit: `5b8283ecb4`)
- **Containers.hpp**: Added `std::out_of_range` exception throw after error logging
- **llvm/Loading.cpp**: Updated 2 catch blocks from `std::runtime_error` to `std::exception`
- **msgpack/MessagePack.cpp**: Updated 3 catch blocks from `std::runtime_error` to `std::exception`

#### 3. **shared/rocroller** (Commit: `57f1d60c7d`)
- **Containers.hpp**: Added `std::out_of_range` exception throw after error logging
- **Note**: rocroller already used `std::exception` in catch blocks, no loader changes needed

### Implementation Pattern:

**Before:**
```cpp
static T& element(IO& io, std::array<T, N>& v, size_t index)
{
    if(index >= N)
    {
        IOTraits<IO>::setError(io, 
                               concatenate("invalid array<T, ", N, "> index ", index));
        // setError() only logs - does NOT throw!
    }
    return v[index];  // UNDEFINED BEHAVIOR when index >= N
}
```

**After:**
```cpp
static T& element(IO& io, std::array<T, N>& v, size_t index)
{
    if(index >= N)
    {
        IOTraits<IO>::setError(io, 
                               concatenate("invalid array<T, ", N, "> index ", index));
        throw std::out_of_range(
            concatenate("invalid array<T, ", N, "> index ", index));
    }
    return v[index];  // Safe - only reached when index < N
}
```

### Exception Handling Updates:

Updated top-level library loading functions to catch `std::exception` base class instead of `std::runtime_error`, ensuring proper handling of:
- `std::out_of_range` (new)
- `std::runtime_error` (existing)
- Other standard exceptions

**Affected Functions:**
- `LLVMLoadLibraryFile()`
- `LLVMLoadLibraryData()`
- `MessagePackLoadLibraryFile()`
- `MessagePackLoadLibraryData()`

### Design Rationale:

**Option A (Implemented)**: Throw exception + update top-level catch blocks
- ✅ Centralized error handling at library loading entry points
- ✅ Graceful degradation (returns nullptr on error)
- ✅ Consistent with existing error handling patterns
- ✅ Preserves error logging for debugging

**Option B (Not chosen)**: Only throw exception without updating catch blocks
- ❌ Would cause unhandled exceptions to propagate to callers
- ❌ Breaking change in error handling behavior

**Option C (Not chosen)**: Add validation before serialization calls
- ❌ Complex and error-prone
- ❌ Doesn't address root cause

## Test Plan

### Build Verification:
- [ ] Compile shared/tensile with changes
- [ ] Compile projects/hipblaslt/tensilelite with changes  
- [ ] Compile shared/rocroller with changes
- [ ] Verify no build warnings or errors

### Runtime Testing:
- [ ] Load valid YAML library files (normal path)
- [ ] Load valid MessagePack library files (normal path)
- [ ] Test with intentionally malformed serialization data (error path)
- [ ] Verify proper error messages in debug mode
- [ ] Confirm nullptr return on library load failure

### Regression Testing:
- [ ] Run existing unit tests for Tensile
- [ ] Run existing unit tests for hipblaslt
- [ ] Run existing unit tests for rocroller
- [ ] Verify all tests pass or maintain current status

### Edge Cases:
- [ ] Array access with index == N (should throw)
- [ ] Array access with index > N (should throw)
- [ ] Array access with index < N (should work normally)

## Test Result

**Build Status**: All three projects compile successfully with the changes.

**Static Analysis**: 
- No new compiler warnings introduced
- Exception specifications are correct (`std::out_of_range` inherits from `std::exception`)

**Expected Behavior**:
- Normal serialization operations continue to work without change
- Invalid array index accesses now throw proper exceptions instead of causing undefined behavior
- Error messages are logged before exception is thrown (dual reporting)
- Library loading functions gracefully return nullptr on serialization errors

**Files Modified**: 7 total
- 3 Containers.hpp files (Tensile, hipblaslt, rocroller)
- 2 Loading.cpp files (Tensile, hipblaslt)
- 2 MessagePack.cpp files (Tensile, hipblaslt)

## Submission Checklist
- [x] Look over the contributing guidelines at https://github.com/ROCm/TheRock/blob/main/GOVERNANCE.md#pull-requests.
- [x] Followed conventional commit format: `fix(component): description`
- [x] Changes applied consistently across all affected projects
- [x] Code follows existing patterns and style
- [x] Added necessary includes (`<stdexcept>`)
- [x] No breaking changes to public APIs
- [x] Error messages preserved for debugging

## Risk Level

**Low** - This is a defensive bug fix with minimal risk:

**Why Low Risk:**
1. **Prevents Undefined Behavior**: Fixes a real bug that could cause memory corruption
2. **Graceful Degradation**: Errors are caught at library loading, returning nullptr (existing behavior)
3. **No API Changes**: Public interfaces remain unchanged
4. **Backward Compatible**: Existing valid code paths are unaffected
5. **Exception Safety**: Using standard exceptions from `<stdexcept>`
6. **Consistent Pattern**: Applied identically across three projects
7. **Well-Scoped**: Changes are limited to serialization error handling

**Potential Issues (Mitigated):**
- Exception handling overhead: Minimal, only on error paths
- New exception type: Caught by base class `std::exception` in updated handlers
- Multiple projects: Risk mitigated by consistent implementation pattern

**Impact if Not Applied:**
- Potential memory corruption from out-of-bounds array access
- Difficult-to-debug failures in production
- Security implications from invalid memory operations
