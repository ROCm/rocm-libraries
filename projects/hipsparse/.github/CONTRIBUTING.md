<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to hipSPARSE">
  <meta name="keywords" content="ROCm, contributing, hipSPARSE">
</head>

# Contributing to hipSPARSE #

AMD welcomes contributions to hipSPARSE from the community. Whether those contributions are bug reports, bug fixes, documentation additions, performance notes, or other improvements, we value collaboration with our users. We can build better solutions together. Please follow these details to help ensure your contributions will be successfully accepted.

Our code contriubtion guidelines closely follow the model of [GitHub pull-requests](https://help.github.com/articles/using-pull-requests/).  This repository follows the [git flow](http://nvie.com/posts/a-successful-git-branching-model/) workflow, which dictates a /master branch where releases are cut, and a /develop branch which serves as an integration branch for new code.

## Issue Discussion ##

Please use the GitHub Issues tab to notify us of issues.

* Use your best judgement for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When filing an issue, be sure to provide as much information as possible, including script output so
    we can collect information about your configuration. This helps reduce the time required to
    reproduce your issue.
  * Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.
* You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

## Acceptance Criteria ##

hipSPARSE exposes a common interface that provides basic linear algebra subroutines for sparse computation implemented on top of the AMD ROCm runtime and toolchains. hipSPARSE is a SPARSE marshalling library supporting both rocSPARSE and cuSPARSE as backends. It sits between the application and a worker SPARSE library, marshalling inputs into the backend library and marshalling results back to the application.

Because hipSPARSE is simply a wrapper library around rocSPARSE (on AMD GPU systems), any routine added to hipSPARSE must have a corresponding routine in rocSPARSE. Additionally, all routines found in cuSPARSE should have an equivalent on in hipSPARSE.

With this in mind, contributions that accompilish the following are greatly appreciated:
* Fixing compilation issues when using different cuSPARSE or rocSPARSE versions.
* Adding missing hipSPARSE routines that match routines found in cuSPARSE.

## Code Structure ##

The following is the structure of the hipSPARSE library in the GitHub repository.

The `library/include/` directory contains the hipsparse.h header which declares the public API of hipSPARSE

The `library/src/` directory contains the implementations of all the hipSPARSE routines. For hipSPARSE code wrapping rocSPARSE routines (i.e. when running hipSPARSE on an AMD GPU system), the code can be found in `library/src/amd_detail`. Similarily, for hipSPARSE code wrapping cuSPARSE routines (i.e. when running hipSPARSE on a NVIDIA GPU system), the code can be found in `library/src/nvidia_detail`.

The `clients/` directory contains the testing and benchmarking code as well as all the samples demonstrating hipSPARSE usage.

The `docs/` directory contains all of the documentation files.

## Coding Style ##

In general, follow the style of the surrounding code. C and C++ code is formatted using `clang-format`. Use the clang-format version installed with ROCm (found in the `/opt/rocm/llvm/bin` directory). Please do not use your system's built-in `clang-format`, as this is a different version that may result in incorrect results.

To format a file, use:

```
/opt/rocm/llvm/bin/clang-format -style=file -i <path-to-source-file>
```

To format all files, run the following script in rocSPARSE directory:

```
#!/bin/bash
git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/llvm/bin/clang-format  -style=file -i
```

Also, githooks can be installed to format the code per-commit:
```
./.githooks/install
```

## Pull Request Guidelines ##

When you create a pull request, you should target the default branch. Our current default branch is the **develop** branch, which serves as our integration branch.

### Deliverables ###

When raising a PR in hipSPARSE here are some important things to include:

1. For each new file in the repository, Please include the licensing header
```
/* ************************************************************************
* Copyright (C) 20xx Advanced Micro Devices, Inc. All rights Reserved.
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
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */
```
and adjust the date to the current year. When simply modifying a file, the date should automatically be updated pre-commit as long as the githook has been installed (./.githooks/install).

2. When adding a new routine, please make sure you are also adding appropriate testing code. These new unit tests should integrate within the existing [googletest framework](https://github.com/google/googletest/blob/master/googletest/docs/primer.md). This typically involves adding the following files:

* testing_<routine_name>.hpp file in the directory `clients/include/`
* test_<routine_name>.cpp file in directory `clients/tests/`

See existing tests for guidance when adding your own.

3. When modifiying an existing routine, add appropriate testing to test_<routine_name>.cpp file in directory `clients/tests/`.

4. Tests must have good code coverage.

7. Ensure code builds successfully. This includes making sure that the code can compile, that the code is properly formatted, and that all tests pass.

Because hipSPARSE is just a wrapper around both rocSPARSE (on AMD GPU systems) and cuSPARSE (on NVIDIA GPU systems), please ensure that your code compiles and runs on both an AMD GPU system using the rocSPARSE backend and a NVIDIA GPU system using the cuSPARSE backend. Please be mindful that routines often become deprecated in rocSPARSE/cuSPARSE and that this must be properly accounted for in hipSPARSE to ensure successful compilation depending on which version of rocSPARSE/cuSPARSE you are using. In hipSPARSE this is currently accomplished using the pre-processor for condition compilation. See existing code for guidance.

8. Do not break existing test cases

### Process ###

When a PR is raised targetting the develop branch in hipSPARSE, CI will be automatically triggered. This will:

* Test that the PR passes static analysis (i.e ensure clang formatting rules have been followed).
* Test that the documentation can be properly built.
* Ensure that the PR compiles on different OS and GPU device architecture combinations (including systems using both rocSPARSE and cuSPARSE).
* Ensure that all tests pass on different OS and GPU device architecture combinations (including systems using both rocSPARSE and cuSPARSE).

Feel free to ask questions on your PR regarding any CI failures you encounter.

* Reviewers are listed in the CODEOWNERS file
