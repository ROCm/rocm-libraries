<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to rocPRIM">
  <meta name="keywords" content="ROCm, contributing, rocPRIM">
</head>

# Contributing to rocPRIM #

We welcome contributions to rocPRIM.  Please follow these details to help ensure your contributions will be successfully accepted.

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

rocPRIM provides a number of foundational parallel algorithms that have optimized for AMD ROCm platforms.
The purpose of the library is provide a reliable, performant foundation upon which other libraries can build.
Algorithms are written in HIP to maximize portability.

Correctness and performance are both important goals in rocPRIM. Because of this, new changes should include
both test and benchmark coverage. Tests and benchmarks should be broad enough to ensure that code runs correctly
and performs well across a variety of input types and sizes.

We also employ automated testing and benchmarking via checks that are run when a pull request is created.
These checks:
- test all algorithms for correctness across a variety of input configurations (eg. types, sizes, etc.)
- run benchmarks to check for performance degredation
- test the change on various OS platforms (Ubuntu, RHEL, etc.)
- build and run the code on different GPU architectures (MI-series, Radeon series cards, etc.)

## Code Structure ##

rocPRIM is a header-only library. Library code lives inside of /rocprim/include.
Algorithms are organized by the level-of-scope at which they operate. For example,
the following subdirectories (inside /rocprim/include/rocprim/) are organized by 
hardware-level scope:
* device/ contains headers for device-level algorithms
* block/ contains headers for block-level algorithms
* warp/ contains headers for warp/wavefront-level algorithms.

The following subdirectories contain supporting code:
* detail/ - utility functions and structs for the internal state of algorithms
* intrinsics/ - specialized intrinsic functions (eg. atomics, warp-shuffling, bit manipulation, etc.)
* iterator/ - contains the iterators that are used to interact with most algorithms in the library
* thread/ - primitive single-threaded algorithms (search, scan, etc.), low-level thread load and store operations
* types/ - contains a number of convenience types used in the library (eg. for storing future values, compile-time integer sequences, etc.)

Algorithms often perform better if their launch parameters (number of blocks, block size, items per thread, etc.) are tailored for the particular architecture they are run on.
rocPRIM achieves this by passing structs called configs to algorithms when they are run. A config struct encapsulates all the information that's needed to run a particular algorithm
in the most performant way for a specific device. You can find configs for a number of algorithms in the device/detail/config/ directory.

## Coding Style ##

C and C++ code should be formatted using `clang-format`. Use the clang-format version for Clang 9, which is available in the `/opt/rocm` directory. Please do not use your system's built-in `clang-format`, as this is an older version that will have different results.

To format a file, use:

```
/opt/rocm/hcc/bin/clang-format -style=file -i <path-to-source-file>
```

To format all files, run the following script in rocPRIM directory:

```
#!/bin/bash
git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/hcc/bin/clang-format  -style=file -i
```

Also, githooks can be installed to format the code per-commit:

```
./.githooks/install
```

## Pull Request Guidelines ##

Our code contribution guidelines closely follows the model of [GitHub pull-requests](https://help.github.com/articles/using-pull-requests/).

When you create a pull request, you should target the default branch. Our current default branch is the **develop** branch, which serves as our integration branch.
Releases are cut to release/rocm-rel-x.y, where x and y refer to the release major and minor numbers.

### Deliverables ###

Code that introduces new features should have test coverage and benchmark coverage. 
rocPRIM tests are located in the /test/rocprim/ directory, while benchmarks can be found in the /benchmark/ directory.

rocPRIM is an open source library. Because of this, we include the following license description at the top of every source file.
If you create new source files in the repository, please include this text in them as well (replacing "xx" with the digits for the current year):
```
// Copyright (c) 20xx Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
```

### Process ###

After you create a PR, you can take a look at a diff of the changes you made using the PR's "Files" tab.

PRs must pass through the checks and the code review described in the [Acceptance Criteria](#acceptance-criteria) section before they can be merged.

Checks may take some time to complete. You can view their progress in the table near the bottom of the pull request page. You may also be able to use the links in the table
to view logs associated with a check if it fails.

During code reviews, another developer will take a look through your proposed change. If any modifications are requested (or further discussion about anything is
needed), they may leave a comment. You can follow up and respond to the comment, and/or create comments of your own if you have questions or ideas.
When a modification request has been completed, the conversation thread about it will be marked as resolved.

To update the code in your PR (eg. in response to a code review discussion), you can simply push another commit to the branch used in your pull request.