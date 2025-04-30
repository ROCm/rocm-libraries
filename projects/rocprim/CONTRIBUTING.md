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

rocPRIM provides a number of foundational parallel algorithms that are optimized for AMD ROCm platforms.
The purpose of the library is to provide a reliable, performant foundation upon which other libraries can be built.
The library is written in HIP, targeting AMD's ROCm platform.

Correctness and performance are both important goals in rocPRIM. Because of this, new changes should include
both **test** and **benchmark** coverage. Tests and benchmarks should be broad enough to ensure that code runs correctly
and performs well across a variety of input types and sizes. More specifically:
- Tests must cover all the functionality added to the public API.
- Tests must cover the whole range of supported sizes, not by testing every single possible size but rather using representative sizes that ensure that the algorithms run succesfully with any size from the range.
  - On this note, it also needs to be taken into account that some algorithms have support for large indices (indices that cannot be stored in a 32-bit integer), so input sizes should also cover that case.
- Tests and benchmarks must be instantiated with all supported data types.
  - If the algorithm uses multiple data types (for instance, if it uses different types for input and output), a selected and representative few combinations should be tested instead of the full combination matrix.

Any utility needed by the tests **and** benchmarks must be added to the appropriate header within the `common` folder. Non-common utilities may be hosted in the corresponding headers from the `test` or `benchmark` folders. For a more detailed description of the cases to be considered for adding new utilities, please check [common](/common/README.md).

We also employ automated testing and benchmarking via checks that are run when a pull request is created.
These checks:
- test all algorithms for correctness across a variety of input configurations (eg. types, sizes, etc.)
- run benchmarks to check for performance degradation
- test the change on various OS platforms (Ubuntu, RHEL, etc.)
- build and run the code on different GPU architectures (MI-series, Radeon series cards, etc.)

## Code Structure ##

rocPRIM is a header-only library. Library code is located inside of `rocprim/include/rocprim/`, and within the `rocprim` namespace. Note that all the symbols inside the `rocprim::detail` namespace are not part of the public API.

Algorithms are grouped by the level-of-scope at which they operate. The following subdirectories organize them by hardware-level scope:
* `device/`: contains headers for device-level algorithms, which are to be called from host code.
* `block/`: contains headers for block-level algorithms, only callable from device code.
* `warp/`: contains headers for warp/wavefront-level algorithms, only callable from device code.
* `thread/`: contains headers for thread-level algorithms, only callable from device code.

Supporting code is distributed into several subdirectories depending on its scope:
* `detail/`: utility functions and structs for the internal state of algorithms.
    * `detail/config/`: configs for tuned algorithms (see [tuning](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/concepts/tuning.html)).
* `intrinsics/`: specialized intrinsic functions (eg. atomics, warp-shuffling, bit manipulation, etc.). Some of them are just wrappers around HIP's intrinsics, atomic/warp-shuffle functions or compiler's intrinsics.
* `iterator/`: iterators that are used to interact with most algorithms in the library (like `constant_iterator` for iterating over a homogeneous range of values or `transform_iterator` for applying a transformation to a given range of values).
* `types/`: a number of convenient types used in the library (eg. for storing future values, compile-time integer sequences, etc.).

Correctness code (tests) is located inside the `test` folder. Several test suites exist depending on what they assess:

* `extra`: test suite that should be run after rocPRIM is installed from package or from source. It is a short smoke test to verify the correctness of the installation or packaging process. 
* `hip`: test suite that checks HIP functionality that is of particular interest to rocPRIM.
* `hipgraph`: test suite for verifying that rocPRIM's algorithms work with `hipGraph`.
* `rocprim`: test suite for checking the correctness of rocPRIM's algorithms.

Finally, performance code (benchmarks) is located inside the `benchmark` folder. Tuned algorithms use three files:
* `benchmark/benchmark_<algorithm>.cpp`
* `benchmark/benchmark_<algorithm>.parallel.cpp.in`
* `benchmark/benchmark_<algorithm>.parallel.hpp`

while non-tuned algorithms have only one `benchmark/benchmark_<algorithm>.cpp` file.

## Coding Style ##

C and C++ code should be formatted using `clang-format`. Use the clang-format version shipped with ROCm, which is available in the `/opt/rocm` directory. Please do not use your system's built-in `clang-format`, as this is an older version that will have different results.

The check_format script (`scripts/code-format/check-format.sh`) allows to check for formatting violations. These can be easily fixed as described below.

To format a file, use:

```bash
/opt/rocm/llvm/bin/clang-format -style=file -i <path-to-source-file>
```

To format all modified (staged) files, use the following command inside the root directory of rocPRIM:
```bash
/opt/rocm/llvm/bin/git-clang-format --style=file --binary /opt/rocm/llvm/bin/clang-format
```
Format modifications will stay unstaged, so that they can be reviewed before commiting.

The formatting can also be done on a per-commit basis, by running:

```bash
/opt/rocm/llvm/bin/git-clang-format --style=file --binary /opt/rocm/llvm/bin/clang-format <git-hash>
```

or installing githooks:

```bash
./.githooks/install
```

The githooks installed will both format the code and update the copyright dates (see [deliverables](#deliverables)).

Additionally, some code editors (such as Visual Studio Code, CLion, XCode, Eclipse, Vim, etc.) have clang-format plugins available, so that formatting can be done from the editor instead of from command-line. This is especially useful for formatting while coding.

### Namespaces ###
As mentioned in [Code Structure](#code-structure), rocPRIM's symbols are exposed within the `rocprim` namespace, with the exception of the ones intended for internal use which are inside `rocprim::detail`. This is done so that users can place rocPRIM in a different namespace (keeping `rocprim` as the innermost namespace) to prevent a namespace collision when two independent rocPRIM libraries end up in the same compute unit through, for instance, indirect inclusion.

Therefore, files from `rocprim/include/rocprim` containing any rocPRIM symbol should start with `BEGIN_ROCPRIM_NAMESPACE` and end with `END_ROCPRIM_NAMESPACE`. These are macros that wrap the namespace opening and closing, respectively.

Implementation details are put into the `rocprim::detail` namespace. No wrapping macros are defined for this one, so just the ususal 
```c++
namespace detail
{ 
  ...
}
```
should be used when needed.

## Documenting Style ##

Apart from the usual comments to ease understanding of the code, Sphinx and Doxygen are used to document the functionality available from rocPRIM.

The Sphinx docs for the API are organized mostly following the code structure. The folders `<hardware-level>_ops` (block_ops, device_ops, etc.) contain the documentation files for methods operating in the correspondent hardware levels. The documentation for supporting code is placed in separate files, located inside `docs/reference`.

To connect Sphinx with Doxygen, Breathe is used. There is a Doxygen group defined for each folder under `rocprim/include/rocprim/` which has documented functionality named as `<folder_name>module` (for example, `threadmodule` for members of `rocprim/include/rocprim/thread` or `intrinsicsmodule` for members of `rocprim/include/rocprim/intrinsics`). Placing the contents of a file inside the correspondent Doxygen group guarantees that Sphinx will get access to the documentation inside that file.

Only members of the public API need to be documented with these two tools, as in the ones outside the `rocprim::detail` namespace, as all symbols inside said namespace are excluded from the documentation.

If some member does not need documentation (such as a specialization of a class that doesn't need any extra clarifications) it can be left out of Doxygen docs by encapsulating the code as shown below:

```c++
/// \cond <section-label>

// code without doxygen documentation here

/// \endcond
```

This isn't always possible (for instance, when base classes need to be excluded), so a pre-processor approach is also available:

```c++
#ifndef DOXYGEN_DOCUMENTATION_BUILD

// code without doxygen documentation here

#endif // DOXYGEN_DOCUMENTATION_BUILD

```
Some files also use the following structure:
```c++
#ifndef DOXYGEN_SHOULD_SKIP_THIS

// code without doxygen documentation here

#endif // DOXYGEN_SHOULD_SKIP_THIS

```
New code should prefer `DOXYGEN_DOCUMENTATION_BUILD` over `DOXYGEN_SHOULD_SKIP_THIS`, as its easier to understand. `DOXYGEN_SHOULD_SKIP_THIS` is defined to be 1 when Doxygen is parsing, logically making its correct usage a double-negation. 

In general terms, a file properly documented should look like something along the lines of:

```c++
/// \addtogroup <group_name>
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
  // here just add comments if needed
  ...
} // end namespace detail

/// \brief Some public class.
///
/// Here some more info can be added to the brief description.
/// \tparam A Template type used by the class.
/// \tparam ...
template<class A, ...>
class some_class
{
  /// \brief A type used within the class.
  using class_type = some_other_type;

  /// \brief A method member of the class.
  ///
  /// \tparam B Another template parameter.
  /// \param [in] param_in_first Input parameter description.
  /// \param [in] param_in_second [optional] Optional input parameter description.
  /// \param [out] param_out Output parameter description.
  /// \param [in,out] param_in_out Input/Output parameter description.
  /// \return Returned object description.
  template<class B>
  return_type some_class_method(A param_in_first, B param_in_second = {})
  {
      ...
  }
}

...

END_ROCPRIM_NAMESPACE

/// @}
// end of group <group_name>

```

## Pull Request Guidelines ##

Our code contribution guidelines closely follows the model of [GitHub pull-requests](https://help.github.com/articles/using-pull-requests/). We also mostly abide [GitHub's best practices for pull-requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/best-practices-for-pull-requests), namely:
1. **Write small PRs**. PRs should be feature-focused and respect the scope of the issue(s) that it refers to. This makes reviews easier and faster, and yields less chances of overlooking bugs in the new/modified code.
2. **Review your own PR**. Before opening/undrafting your PR, take your time to review all the changes as if you were one of the reviewers. This helps catching typos or small errors in advance.
3. **Provide context and guidance**. PRs should generally have a descriptive title and an explanatory body that includes:
    - **scope** (purpose) of the PR: explanation of the scope of the PR (for instance, what feature/bug the PR adds/fixes). This helps identifying new issues to be spawned from the comments received in the PR: if any comment suggests any addition/fix that falls out of this scope, a new issue should be created so that the comment is tackled in another (feature-focused) PR.
    - some **notes** explaining the changes/additions made so that reviewers know which decisions were taken and why. Here you can also explicitly request feedback on specific matters that you think may need to be discussed.
    - if necessary, **how to verify** that the issue(s) at hand are indeed tackled with this PR (something like "the newly added test covering the fixed bug's case is passing").

When you create a pull request, you should target the default branch. Our current default branch is the **develop** branch, which serves as our integration branch.
Releases are cut to `release/rocm-rel-x.y`, where x and y refer to the release major and minor numbers.

### Deliverables ###

#### Correctness, performance and documentation ####

Code that introduces new features should have **test coverage** and **benchmark coverage**. **Documentation** must also be added following the guidelines described in [Documentation Style](#documentation-style). If modifying existing functionality, tests, benchmarks and documentation must be updated to fit the new behavior and/or parameters.

If the autotuning is run, benchmarks should be re-run to check whether performance indeed improves. If so, the new configuration files generated should be added to the corresponding PR.


#### License ####

rocPRIM is an open source library. Because of this, we include the **license description** shown below at the top of every source file.
If you create new source files in the repository, please include this text in them as well (replacing "xx" with the digits for the current year):
```c++
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
If you modify existing files licensed in a previous year, add a dash followed by the modification year to indicate that the license also covers the most recent changes (like so: `Copyright (c) 20xx-20yy`). It may also happen that such an interval is already specified in the license, but the last year indicated is previous to the current modification date, in this case just change it accordingly.

Under the `scripts/copyright-date` folder there is `check-copyright` script that we use to check if the copyright date updates are done. It can also be used to automatize those updates. Run

```bash
scripts/copyright-date/check-copyright.sh -u
```

inside rocPRIM's root directory to update the copyright statements of modified files
or set

```bash
git config --local hooks.updateCopyright true
```
to automatically update copyrights when committing.

#### Changes Record ####

All noticeable changes are recorded in the `CHANGELOG.md` file. For every release, we annotate the additions, fixes, changes, deprecations and/or optimizations introduced within that release.

When opening a PR, make sure to add to the correspondent sections under the latest unreleased release all the meaningful changes introduced.

### Process ###

After you create a PR, you can take a look at a diff of the changes you made using the PR's "Files" tab.

PRs must pass through the checks and the code review described in the [Acceptance Criteria](#acceptance-criteria) section before they can be merged.

Checks may take some time to complete. You can view their progress in the table near the bottom of the pull request page. You may also be able to use the links in the table
to view logs associated with a check if it fails.

During code reviews, another developer(s) will take a look through your proposed change. If any modifications are requested (or further discussion about anything is
needed), they may leave a comment. You can follow up and respond to the comment, and/or create comments of your own if you have questions or ideas.
When a modification request has been completed, the conversation thread about it will be marked as resolved.

To update the code in your PR (eg. in response to a code review discussion), you can simply push another commit to the branch used in your pull request.
