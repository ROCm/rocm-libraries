# To add new data-driven tests to the rocBLAS Google Test Framework:

**I**. Create a C++ header file with the name `testing_<function>.hpp` in the
`include` subdirectory, with templated functions for a specific rocBLAS
routine. Examples:
```
testing_gemm.hpp
testing_gemm_ex.hpp
```
In this `testing_*.hpp` file, create a templated function which returns `void`
and accepts a `const Arguments&` parameter. For example:
```
template<typename Ti, typename To, typename Tc>
void testing_gemm_ex(const Arguments& arg)
{
// ...
}
```
This function should be generalized with template parameters as much as possible,
to avoid copy-and-paste code.

In this function, use the following macros to check results:
```
HIP_CHECK_ERROR             Verifies that a HIP call returns success
ROCBLAS_CHECK_ERROR         Verifies that a rocBLAS call returns success
EXPECT_ROCBLAS_STATUS       Verifies that a rocBLAS call returns a certain status
UNIT_CHECK                  Check that two answers agree (see unit.hpp)
NEAR_CHECK                  Check that two answers are close (see near.hpp)
```
In addition, you can use Google Test Macros such as the below, as long as they are
guarded by `#ifdef GOOGLE_TEST`:
```
EXPECT_EQ
ASSERT_EQ
EXPECT_TRUE
ASSERT_TRUE
...
```
Note: The `device_vector` template allocates memory on the device. You must check whether
converting the `device_vector` to `bool` returns `false`, and if so, report a HIP memory
error and then exit the current function. For example:
```
// allocate memory on device
device_vector<T> dx(size_x);
device_vector<T> dy(size_y);
if(!dx || !dy)
{
    CHECK_HIP_ERROR(hipErrorOutOfMemory);
    return;
}
```

The general outline of the function should be:
1. Convert any scalar arguments (e.g., `alpha` and `beta`) to `double`.
2. If the problem size arguments are invalid, use a `safe_size` to allocate arrays,
call the rocBLAS routine with the original arguments, and verify that it returns
`rocblas_status_invalid_size`. Return.
3. Set up host and device arrays (see `rocblas_vector.hpp` and `rocblas_init.hpp`).
4. Call a CBLAS or other reference implementation on the host arrays.
5. Call rocBLAS using both device pointer mode and host pointer mode, verifying that
every rocBLAS call is successful by wrapping it in `ROCBLAS_CHECK_ERROR()`.
6. If `arg.unit_check` is enabled, use `UNIT_CHECK` to validate results.
7. (Deprecated) If `arg.norm_check` is enabled, calculate and print out norms.
8. If `arg.timing` is enabled, perform benchmarking (currently under refactoring).

**II**. Create a C++ file with the name `<function>_gtest.cpp` in the `gtest`
subdirectory, where `<function>` is a non-type-specific shorthand for the
function(s) being tested. Examples:
```
gemm_gtest.cpp
trsm_gtest.cpp
blas1_gtest.cpp
```
In the C++ file, perform these steps:

A. Include the header files related to the tests, as well as `type_dispatch.hpp`.
For example:
```c++
#include "testing_syr.hpp"
#include "type_dispatch.hpp"
```
B. Wrap the body with an anonymous namespace, to minimize namespace collisions:
```c++
namespace {
```
C. Create a templated class which accepts any number of type parameters followed by one anonymous trailing type parameter defaulted to `void` (to be used with `enable_if`).

Choose the number of type parameters based on how likely in the future that
the function will support a mixture of that many different types, e.g. Input
type (`Ti`), Output type (`To`), Compute type (`Tc`). If the function will
never support more than 1-2 type parameters, then that many can be used. But
if the function may be expanded later to support mixed types, then those
should be planned for ahead of time and placed in the template parameters.

Unless the number of type parameters is greater than one and is always
fixed, then later type parameters should default to earlier ones, so that
a subset of type arguments can used, and so that code which works for
functions which take one type parameter may be used for functions which
take one or more type parameters. For example:
```c++
template< typename Ti, typename To = Ti, typename Tc = To, typename = void>
```
Make the primary definition of this class template derive from the `rocblas_test_invalid` class. For example:
```c++
 template <typename T, typename = void>
 struct syr_testing : rocblas_test_invalid
 {
 };
```
D. Create one or more partial specializations of the class template conditionally enabled by the type parameters matching legal combinations of types.

If the first type argument is `void`, then these partial specializations must not apply, so that the default based on `rocblas_test_invalid` can perform the correct behavior when `void` is passed to indicate failure.

In the partial specialization(s), derive from the `rocblas_test_valid` class.

In the partial specialization(s), create a functional `operator()` which takes a `const Arguments&` parameter and calls templated test functions (usually in `include/testing_*.hpp`) with the specialization's template arguments when the `arg.function` string matches the function name. If `arg.function` does not match any function related to this test, mark it as a test failure. For example:
```c++
 template <typename T>
 struct syr_testing<T,
                    std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>
                   > : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        if(!strcmp(arg.function, "syr"))
            testing_syr<T>(arg);
        else
            FAIL() << "Internal error: Test called with unknown function: "
                   << arg.function;
    }
};
```
E. If necessary, create a type dispatch function for this function (or group of functions it belongs to) in `include/type_dispatch.hpp`. If possible, use one of the existing dispatch functions, even if it covers a superset of allowable types. The purpose of `type_dispatch.hpp` is to perform runtime type dispatch in a single place, rather than copying it across several test files.

The type dispatch function takes a `template` template parameter of `template<typename...> class` and a function parameter of type `const Arguments&`. It looks at the runtime type values in `Arguments`, and instantiates the template with one or more static type arguments, corresponding to the dynamic runtime type arguments.

It treats the passed template as a functor, passing the Arguments argument to a particular instantiation of it.

The combinations of types handled by this "runtime type to template type instantiation mapping" function can be general, because the type combinations which do not apply to a particular test case will have the template argument set to derive from `rocblas_test_invalid`, which will not create any unresolved instantiations. If unresolved instantiation compile or link errors occur, then the `enable_if<>` condition in step D needs to be refined to be `false` for type combinations which do not apply.

The return type of this function needs to be `auto`, picking up the return type of the functor.

If the runtime type combinations do not apply, then this function should return `TEST<void>{}(arg)`, where `TEST` is the template parameter. However, this is less important than step D above in excluding invalid type
combinations with `enable_if`, since this only excludes them at run-time, and they need to be excluded by step D at compile-time in order to avoid unresolved references or invalid instantiations. For example:
```c++
template <template <typename...> class TEST>
auto rocblas_simple_dispatch(const Arguments& arg)
{
    switch(arg.a_type)
    {
      case rocblas_datatype_f16_r: return TEST<rocblas_half>{}(arg);
      case rocblas_datatype_f32_r: return TEST<float>{}(arg);
      case rocblas_datatype_f64_r: return TEST<double>{}(arg);
      case rocblas_datatype_bf16_r: return TEST<rocblas_bfloat16>{}(arg);
      case rocblas_datatype_f16_c: return TEST<rocblas_half_complex>{}(arg);
      case rocblas_datatype_f32_c: return TEST<rocblas_float_complex>{}(arg);
      case rocblas_datatype_f64_c: return TEST<rocblas_double_complex>{}(arg);
      default: return TEST<void>{}(arg);
    }
}
```
F. Create a (possibly-templated) test implementation class which derives from the `RocBLAS_Test` template class, passing itself to `RocBLAS_Test` (the CRTP pattern) as well as the template class defined above. Example:
```c++
struct syr : RocBLAS_Test<syr, syr_testing>
{
    // ...
};
```
In this class, implement three static functions:

 `static bool type_filter(const Arguments& arg)` returns `true` if the types described by `*_type` in the `Arguments` structure, match a valid type combination.

This is usually implemented simply by calling the dispatch function in step E, passing it the helper `type_filter_functor` template class defined in `RocBLAS_Test`. This functor uses the same runtime type checks as are used to instantiate test functions with particular type arguments, but instead, this returns `true` or `false` depending on whether a function would have been called. It is used to filter out tests whose runtime parameters do not match a valid test.

Since `RocBLAS_Test` is a dependent base class if this test implementation class is templated, you may need to use a fully-qualified name (`A::B`) to resolve `type_filter_functor`, and in the last part of this name, the keyword `template` needs to precede `type_filter_functor`. The first half of the fully-qualified name can be this class itself, or the full instantiation of `RocBLAS_Test<...>`. For example:
```c++
static bool type_filter(const Arguments& arg)
{
    return rocblas_blas1_dispatch<
        blas1_test_template::template type_filter_functor>(arg);
}
```
`static bool function_filter(const Arguments& arg)` returns `true` if the function name in `Arguments` matches one of the functions handled by this test. For example:
```c++
// Filter for which functions apply to this suite
static bool function_filter(const Arguments& arg)
{
  return !strcmp(arg.function, "ger") || !strcmp(arg.function, "ger_bad_arg");
}
```
 `static std::string name_suffix(const Arguments& arg)` returns a string which will be used as the Google Test name's suffix. It will provide an alphanumeric representation of the test's arguments.

The `RocBLAS_TestName` helper class template should be used to create the name. It accepts ostream output, and can be automatically converted to `std::string` after all of the text of the name has been streamed to it.

The `RocBLAS_TestName` helper class constructor accepts a string argument which will be included in the test name. It is generally passed the `Arguments` structure's `name` member.

The `RocBLAS_TestName` helper class template should be passed the name of this test implementation class (including any implicit template arguments) as a template argument, so that every instantiation of this test implementation class creates a unique instantiation of `RocBLAS_TestName`. `RocBLAS_TestName` has some static data which needs to be kept local to each test.

 `RocBLAS_TestName` converts non-alphanumeric characters into suitable replacements, and disambiguates test names when the same arguments appear more than once.

 Since the conversion of the stream into a `std::string` is a destructive one-time operation, the `RocBLAS_TestName` value converted to `std::string` needs to be an rvalue. For example:
```c++
static std::string name_suffix(const Arguments& arg)
{
    // Okay: rvalue RocBLAS_TestName object streamed to and returned
    return RocBLAS_TestName<syr>(arg.name) << rocblas_datatype2string(arg.a_type)
        << '_' << (char) std::toupper(arg.uplo) << '_' << arg.N
        << '_' << arg.alpha << '_' << arg.incx << '_' << arg.lda;
}

static std::string name_suffix(const Arguments& arg)
{
    RocBLAS_TestName<gemm_test_template> name(arg.name);
    name << rocblas_datatype2string(arg.a_type);
    if(GEMM_TYPE == GEMM_EX || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX)
        name << rocblas_datatype2string(arg.b_type)
             << rocblas_datatype2string(arg.c_type)
             << rocblas_datatype2string(arg.d_type)
             << rocblas_datatype2string(arg.compute_type);
    name << '_' << (char) std::toupper(arg.transA)
                << (char) std::toupper(arg.transB) << '_' << arg.M
                << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_'
                << arg.lda << '_' << arg.ldb << '_' << arg.beta << '_'
                << arg.ldc;
    // name is an lvalue: Must use std::move to convert it to rvalue.
    // name cannot be used after it's converted to a string, which is
    // why it must be "moved" to a string.
    return std::move(name);
}
```
G. Choose a non-type-specific shorthand name for the test, which will be displayed as part of the test name in the Google Tests output (and hence will be stringified). Create a type alias for this name, unless the name is already the name of the class defined in step F, and it is not templated. For example, for a templated class defined in step F, create an alias for one of its instantiations:
```c++
using gemm = gemm_test_template<gemm_testing, GEMM>;
```
H. Pass the name created in step G to the `TEST_P` macro, along with a broad test category name that this test belongs to (so that Google Test filtering can be used to select all tests in a category). The broad test category suffix should be _tensile if it requires Tensile.

In the body following this `TEST_P` macro, call the dispatch function from step E, passing it the class from step C as a template template argument, passing the result of `GetParam()` as an `Arguments` structure, and wrapping the call in the `CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES()` macro. For example:
```c++
TEST_P(gemm, blas3_tensile) { CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_gemm_dispatch<gemm_testing>(GetParam())); }
```
The `CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES()` macro detects signals such as `SIGSEGV` and uncaught C++ exceptions returned from rocBLAS C APIs as failures, without terminating the test program.
I. Call the `INSTANTIATE_TEST_CATEGORIES` macro which instantiates the Google Tests across all test categories (`quick`, `pre_checkin`, `nightly`, `known_bug`), passing it the same test name as in steps G and H. For example:
```c++
INSTANTIATE_TEST_CATEGORIES(gemm);
```
J. Don't forget to close the anonymous namespace:
```c++
} // namespace
```
**III.** Create a `<function>.yaml` file with the same name as the C++ file, just with
   a `.yaml` extension.

   In the YAML file, define tests with combinations of parameters.

   The YAML files are organized as files which `include:` each other (an extension to YAML), define anchors for data types and data structures, list of test parameters or subsets thereof, and `Tests` which describe a combination of parameters including `category` and `function`.

   `category` must be one of `quick`, `pre_checkin`, `nightly`, or `known_bug`. The category is automatically changed to `known_bug` if the test matches a test in `known_bugs.yaml`.

   `function` must be one of the functions tested for and recognized in steps D-F.

   The syntax and idioms of the YAML files is best described by looking at the
   existing `*_gtest.yaml` files as examples.

**IV.** Add the YAML file to `rocblas_gtest.yaml`, to be included. For examnple:
```yaml
include: blas1_gtest.yaml
```
**V.** Add the YAML file to the list of dependencies for `rocblas_gtest.data` in `CMakeLists.txt`.  For example:
```cmake
add_custom_command( OUTPUT "${ROCBLAS_TEST_DATA}"
                    COMMAND ../common/rocblas_gentest.py -I ../include rocblas_gtest.yaml -o "${ROCBLAS_TEST_DATA}"
                    DEPENDS ../common/rocblas_gentest.py rocblas_gtest.yaml ../include/rocblas_common.yaml known_bugs.yaml blas1_gtest.yaml gemm_gtest.yaml gemm_batched_gtest.yaml gemm_strided_batched_gtest.yaml gemv_gtest.yaml symv_gtest.yaml syr_gtest.yaml ger_gtest.yaml trsm_gtest.yaml trtri_gtest.yaml geam_gtest.yaml dgmm_gtest.yaml set_get_vector_gtest.yaml set_get_matrix_gtest.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
```
**VI.** Add the `.cpp` file to the list of sources for `rocblas-test` in `CMakeLists.txt`. For example:
```c++
set(rocblas_test_source
    rocblas_gtest_main.cpp
    ${Tensile_TEST_SRC}
    set_get_pointer_mode_gtest.cpp
    logging_mode_gtest.cpp
    set_get_vector_gtest.cpp
    set_get_matrix_gtest.cpp
    blas1_gtest.cpp
    gemv_gtest.cpp
    ger_gtest.cpp
    syr_gtest.cpp
    symv_gtest.cpp
    geam_gtest.cpp
    dgmm_gtest.cpp
    trtri_gtest.cpp
   )
```
### Many examples are available in `gtest/*_gtest.{cpp,yaml}`
