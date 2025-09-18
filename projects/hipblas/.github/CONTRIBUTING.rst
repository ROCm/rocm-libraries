=======================
Contributing to hipBLAS
=======================

We welcome contributions to hipBLAS. Please follow these details to help ensure your contributions will be successfully accepted.

Issue Discussion
================

Please use the GitHub Issues tab to notify us of issues.

- Use your best judgment for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
- If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
- If your issue doesn't exist, use the issue template to file a new issue.

  - When filing an issue, be sure to provide as much information as possible, including script output so
    we can collect information about your configuration. This helps reduce the time required to
    reproduce your issue.
  - Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.
- You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

Acceptance Criteria
===================

Contributors wanting to submit improvements, or bug fixes
should follow the below mentioned guidelines.

Pull requests will be reviewed by members of
`CODEOWNERS <https://github.com/ROCm/hipBLAS/blob/develop/.github/CODEOWNERS>`__
Continuous Integration tests will be run on the pull request. Once the pull request
is approved and tests pass it will be merged by a member of
`CODEOWNERS <https://github.com/ROCm/hipBLAS/blob/develop/.github/CODEOWNERS>`__
Attribution for your commit will be preserved when it is merged.

Pull Request Guidelines
=======================

By creating a pull request, you agree to the statements made in the
`Code License`_
section. Your pull request should target the default branch. Our current
default branch is the develop branch, which serves as our integration branch.

Pull requests should:

- ensure code builds successfully.
- do not break existing test cases.
- new functionality will only be merged with new unit tests.
- new unit tests should integrate within the existing googletest framework.
- tests must have good code coverage.
- code must also have benchmark tests, and performance must approach the compute bound limit or memory bound limit.

Deliverables
============

For each new file, please include the licensing header

.. code:: cpp

    /*******************************************************************************
     * Copyright (c) 20xx Advanced Micro Devices, Inc.
     *
     * Permission is hereby granted, free of charge, to any person obtaining a copy
     * of this software and associated documentation files (the "Software"), to deal
     * in the Software without restriction, including without limitation the rights
     * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
     * copies of the Software, and to permit persons to whom the Software is
     * furnished to do so, subject to the following conditions:
     *
     * The above copyright notice and this permission notice shall be included in all
     * copies or substantial portions of the Software.
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


Process
=======

hipBLAS uses the ``clang-format`` tool for formatting code in source files. To format a file, use:

::

    clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in hipBLAS directory:

::

    #!/bin/bash
    git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 clang-format -style=file -i

Also, githooks can be installed to format the code per-commit:

::

    ./.githooks/install

Code License
============

All code contributed to this project will be licensed under the license identified in the
`LICENSE.md <https://github.com/ROCm/hipBLAS/blob/develop/LICENSE.md>`__.
Your contribution will be accepted under the same license.

References
==========

`hipBLAS documentation <https://rocm.docs.amd.com/projects/hipBLAS/en/latest/index.html>`__
