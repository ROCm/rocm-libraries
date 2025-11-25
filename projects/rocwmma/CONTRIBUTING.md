<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to rocWMMA">
  <meta name="keywords" content="ROCm, contributing, rocWMMA">
</head>

# Contributing to rocWMMA #

External contributions and feedback are welcome for rocWMMA. Please see the following details to help maximize the likelihood your contributions will be accepted.

## Issue Discussion ##

Please use the [GitHub Issues](https://github.com/ROCm/rocWMMA/issues) tab to notify us of issues.

* Use your best judgement for issue creation. If your issue is already listed, upvote the existing issue and
  comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar/related issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
    * When filing an issue, be sure to provide as much information as possible, including script output so
      we can collect information about your configuration. This helps reduce the time required to
      reproduce your issue.
    * Check on updates to your issue regularly, as we may require additional information to successfully reproduce the
      issue.
* You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

New issues should use the following templates:

* **Bugs**
    + Any unintended functionality within the rocWMMA library should be thoroughly documented with the template below.
        1. Description: ***Please be clear and concise***
        2. Steps for Reproduction:
            - Hardware Information:
            - Docker Environment or Software Versions:
            - Expected Behavior:
            - Actual Behavior:
        3. Any additional information:
* **Enhancement Requests**
    + Any proposed enhancements to rocWMMA should be thoroughly documented with the template below.
        1. Description: ***Please be clear and concise***
        2. Value and Motivation
            - Feature and/or Functionalities Enabled:
            - Any Alternatives
        3. Any additional information:

## Acceptance Criteria ##

The goal of rocWMMA is to provide a C++ API for facilitating block-wise decomposition of matrix multiply accumulate (MMA) workflows while
leveraging specialized AMD GPU hardware. rocWMMA also facilitates migration of nvcuda::wmma users to AMD's HIP environment on AMD GPUs and
provides a level of equivalent functionality. Contributors that wish to help optimize and expand the capabilities of rocWMMA in pursuit
of this goal should adhere to the following guidelines for all features and fixes. Detailed coding style and pull request guidelines are
covered later sections.

Contributors wishing to submit new features for rocWMMA should follow the guidelines outlined below:

- Performance Improvements
    * Features targeting performance improvements for any aspect of rocWMMA are generally permitted. Any optimizations must be both notable
      and repeatable in order to avoid unnecessary code maintenance. Documentation regarding performance improvements such as benchmark test
      comparisons must also be provided.
- Bug Fixes
    * Any observed unintended behavior within the rocWMMA library should first be documented by filing a bug report issue using the above
      template. Non-critical issues may be deferred until future releases.
- WMMA Porting
    * Developers wishing to implement gap closures with nvcuda::wmma may suggest additional features to do so

All new features and fixes should also tie into the rocWMMA [GitHub Issues](https://github.com/ROCm/rocWMMA/issues) tab:

- **Enhancements**
    + Any implementations of pre-filed enhancement requests should clearly link to the original issue.
    + Any new enhancements should be documented as if filing a new enhancement request using the template above.
- **Bug Fixes**
    + Any fixes for pre-filed issues should clearly link to the original issue.
    + Any newly found issues should be documented as if filing a new issue using the template above.

### Exceptions ###

Exceptions to these criteria will be handled on a case-by-case basis, and should be discussed via the Issues tab.

## Code Structure ##

The organization of the rocWMMA library is explained in detail in the [Programmers Guide](https://github.com/ROCm/rocWMMA/blob/develop/docs/programmers-guide.rst)

## Coding Style ##

This project follows the [CPP Core Guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md)
with few modifications or additions noted below. All pull-requests should in good faith attempt to follow the guidelines stated therein,
but we recognize that the content is lengthy. Below we list our primary concerns when reviewing pull-requests.

### Interface ###

- Library code should use C++17.
- Our minimum supported compiler is hipcc 4.4.
- Avoid CamelCase.
    * This rule applies specifically to publicly visible APIs, but is also encouraged (not mandated) for internal code.
    * If you are unsure, inspect surrounding code for consistency and don't be afraid to pose questions for clarification on the PRs.

### Philosophy ###

-  [P.2](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-Cplusplus)
   Write in ISO Standard C++17 (especially to support Windows, Linux and
   MacOS platforms)
-  [P.5](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-compile-time)
   Prefer compile-time checking to run-time checking

### Implementation ###

-  [SF.1](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-file-suffix)
   Use a ``.cpp`` suffix for code files and an ``.hpp`` suffix for
   interface files if your project doesn't already follow another
   convention
-  [SF.5](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-consistency)
   A ``.cpp`` file must include the ``.hpp`` file(s) that defines its
   interface
-  [SF.7](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive)
   Don't put a global ``using``-directive in a header file
-  [SF.8](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-guards)
   Use ``#include`` guards for all ``.hpp`` files
-  [SF.21](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-unnamed)
   Don't use an unnamed (anonymous) ``namespace`` in a header
-  [SL.10](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rsl-arrays)*
   Prefer using ``std::array`` or ``std::vector`` instead of a C array.
-  [C.9](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-private)
   Minimize the exposure of class members
-  [F.3](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-single)
   Keep functions short and simple
-  [F.21](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out-multi)*
   To return multiple 'out' values, prefer returning a ``std::tuple``
-  [R.1](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rr-raii)*
   Manage resources automatically using RAII (this includes
   ``std::unique_ptr`` & ``std::shared_ptr``)
-  [ES.11](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-auto)
   Use ``auto`` to avoid redundant repetition of type names
-  [ES.20](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-always)
   Always initialize an object
-  [ES.23](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-list)
   Prefer the ``{}`` initializer syntax
-  [CP.1](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#S-concurrency)
   Assume that your code will run as part of a multi-threaded program
-  [I.2](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global)
   Avoid global variables

| *std functions are not recommended for any ``rocwmma/internal`` code to maintain hipRTC compatibility. |
|--------------------------------------------------------------------------------------------------------|

## Pull Request Guidelines ##
Our code contribution guidelines closely follows the model of [GitHub Pull-Requests](https://help.github.com/articles/using-pull-requests).
The rocWMMA repository follows a workflow which dictates a ``/master`` branch where releases are cut, and a ``/develop`` branch which serves as
an integration branch for new code.

No changes are allowed to be directly committed to the develop branch of the rocWMMA repository. All authors are required to develop their change
sets on a separate branch (preferably in your own fork), and then create a pull request targeting the develop branch of the upstream repository.
When you create a pull request, you should target the **develop** branch for integration.

The typical workflow for creating a rocWMMA pull request is as follows:

1. Create and track a rocWMMA fork, if you haven't already done so.
2. Clone your fork:

    ```bash
    git clone -b develop https://github.com/<your_fork>/rocWMMA.git .
    .githooks/install
    git checkout -b <new_branch>
    ...
    git add <new_work>
    git commit -m "What was changed"
    git push origin <new_branch>
    ...
    ```

3. Create a pull request to the ROCmSoftwarePlatform/rocWMMA develop branch.
4. Await CI and approval feedback.
5. Once approved, merge.

| You must install GitHooks every time you clone a rocWMMA repository to ensure automated triggers for Clang formatting are executed upon making commits. Instructions for formatting rocWMMA are included in [Formatting](#formatting). |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Deliverables ###

rocWMMA has a set of required deliverables for every pull request that are as follows.

1. **Test Integration**:
    - All new functionality introduced to rocWMMA must be accompanied by unit tests. Unit tests should integrate within the existing
    googletest framework and must have good code coverage. Existing unit tests should be used as a guide and are found in ``test/unit``. Be sure to consider
    rocWMMA's support matrix for datatypes, block sizes and architectures.

    - New features that aim to optimize rocWMMA must have benchmark and validation tests, and performance must approach the compute bound limit or
    memory bound limit. These tests should follow the same googletest framework laid out in the rocWMMA GEMM tests found in ``test/gemm``.
    Features that impact the performance of existing rocWMMA kernels must be accompanied with a performance analysis against the pre-existing
    kernels.

2. **API Documentation**:
    - Any new outward facing rocWMMA API functions must be properly documented and included in the [API Reference Guide](https://github.com/ROCm/rocWMMA/blob/develop/docs/api-reference-guide.rst).

3. **Type Support**:
    - All features introduced to rocWMMA must maintain support for the following types:
        - **Supported Datatypes (gfx9)**
            - Native Data Types: int8, f16, f32, f64*
            - Non-Native Data Types: h16 (__half), bf16, f8**, bf8**

        - **Supported Datatypes (gfx11)**
            - Native Data Types: int8, f16
            - Non-Native Data Types: h16, bf16

        - **Supported Datatypes (gfx12)**
            - Native Data Types: int8, f16
            - Non-Native Data Types: h16, bf16, f8**, bf8**

		| *Only on gfx90a, gfx942 & gfx950. |
		|-----------------------------------|

		| **Only on gfx942, gfx950 and gfx12. |
		|-------------------------------------|

    - Support for the other rocWMMA fragment parameters as described in ``library/include/rocwmma/rocwmma.hpp`` must also be maintained.

4. **Licensing**:
    - All code submitted to rocWMMA must be original, no AI generated code is currently being accepted.
    - The code you are contributing is your own, and you have the right to license it.
    - No code found under other licenses is permitted.
    - Any submitted code will subsequently be covered under the MIT License.
    - For each new file introduced in your pull request, please include the licensing header:

    ```
    /*******************************************************************************
    *
    * MIT License
    *
    * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
    *
    * Permission is hereby granted, free of charge, to any person obtaining a copy
    * of this software and associated documentation files (the "Software"), to deal
    * in the Software without restriction, including without limitation the rights
    * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    * copies of the Software, and to permit persons to whom the Software is
    * furnished to do so, subject to the following conditions:
    *
    * The above copyright notice and this permission notice shall be included in
    * all copies or substantial portions of the Software.
    *
    * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    * SOFTWARE.
    *
    *******************************************************************************/
    ```

### Process ###

Reviewers for rocWMMA pull requests are listed under ``.github/CODEOWNERS``. Pull requests should be properly documented with comments
and linked to their corresponding issues. Pull request reviews should include insightful comments where changes are requested.

#### Formatting ####

rocWMMA C++ code is formatted using ``clang-format``.
- To manually format using clang-format use the version in the
  ``/opt/rocm/llvm/bin`` directory. Please do not use your system's built-in ``clang-format``, as this may be an older
  version that will result in different results.

  To format a file, use:

  ```
  /opt/rocm/llvm/bin/clang-format -style=file -i <path-to-source-file>
  ```


  To format all files, run the following script in rocWMMA directory:

  ```
  #!/bin/bash
  git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/llvm/bin/clang-format -style=file -i
  ```

- Alternatively, githooks can be installed to format the code per-commit:

  ```
  <path_to_rocWMMA>/.githooks/install
  ```
