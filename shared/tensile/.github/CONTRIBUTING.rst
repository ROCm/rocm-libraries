********************************************************************
Contribution guidelines
********************************************************************

This document provides the guidelines for contributing to the Tensile source code.

.. seealso::

   For information about environment setup and development processes, see :ref:`programmers-guide`.

Tensile's development practice is based on the `Gitflow workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. The **develop** branch is the default branch for development, where all new features and bug fixes are merged. After a PR is merged into **develop**, it undergoes extended testing and profiling. If these checks pass, the PR might be merged into **staging** to be included in the next release. A PR is available in the upcoming release only if it is merged before the release branch is cut.

================
Raising issues
================

To notify us of any existing issue, use the GitHub *Issues* tab.

- Use your best judgment for issue creation. If your issue is already listed, upvote the issue and comment or post to provide additional details, such as how you reproduced this issue.
- If you are not sure of the listed issue being the same as yours, err on the side of caution and file your issue. You can link your issue with the existing issue by providing your issue link and details in the comment section. If your issue is evaluated to be a duplicate, it will be closed.
- If your issue doesn't exist, use the issue template to file a new issue.
  - When filing an issue, provide as much information as possible including the script output, which is required to collect information about your configuration. This helps to reproduce the issue effectively.
  - Check your issue regularly, as we might require additional information to successfully reproduce the issue.
- You can also open an issue to ask the maintainers if a proposed change meets the acceptance criteria, or to discuss an idea pertaining to the library.

===================
Acceptance criteria
===================

Pull Requests (PR) are reviewed by the members of `CODEOWNERS.md <https://github.com/ROCm/Tensile/blob/develop/.github/CODEOWNERS>`_.
Depending on the PR, the reviewers might post comments or request changes. This might require several iterations.
The PR is approved only when all the changes requested by the reviewers are marked complete.
When a Pull Request is submitted, it undergoes a standard suite of continuous integration tests.

Once the pull request is approved and tests pass, it is merged by a member of the codeowner's community.
Attribution for your commit will be preserved when it is merged.

==========================
Submitting a Pull Request
==========================

By creating a PR, you agree to the statements made in the `Code License`_ section. Your PR must target the default *develop* branch, which also serves as our integration branch.

a. **Forking the repository and making changes:**

   1. Create a fork of Tensile. Don't create feature branches directly in https://github.com/ROCm/Tensile.
   2. Clone your fork locally and set up your :ref:`development-environment`.
   3. Create your feature branch from **develop** and make changes to the code.
   4. Issue ``tox run -m precommit`` and ensure that all checks pass.
   5. Commit your changes using the convention for :ref:`commit-messages`.
   6. If you are updating documentation, issue ``tox run -e docs`` and verify the styling and formatting.
   7. Push the changes to your fork.

.. tip::

   Keeping the scope of new PRs as narrow as possible improves the chances of it getting accepted. If you are making multiple changes, consider breaking them into separate PRs. Keeping PRs small supports timely code reviews, traceability, and straightforward reversions.

b. **Creating the PR:**

   1. Ensure that **your develop** branch is up-to-date with the **upstream develop** branch. This might require a rebase or a merge.
   2. Verify that your changes pass static analysis checks and all pre-checkin, host library, and unit tests by running ``tox run -m prepr``.
   3. Create the PR against the https://github.com/ROCm/Tensile **develop** branch.
   4. Fill in as many details as possible. Include description, outcomes, notable changes, and environment information. The availability of information makes the PR review process easier, increasing the likelihood of the PR getting merged in a timely manner.
   5. Title the PR in present imperative tense. For example, "*Update* kernel parameters", not "Updates" or "Updated".

.. tip::

   To merge **develop** into your feature branch after a PR is opened, use a merge instead of a rebase.

   In general, refrain from force pushing once a feature branch is in PR as it is prone to gotchas in our CI system. Ideally, the git history is linear and clean *before* a PR is created. Hence, we encourage contributors to conduct any rebases or amends prior to opening a PR.

c. **Merging the PR:**

   1. Ensure the title of the PR properly describes the changes.
   2. Squash and merge the PR. If you are not the maintainer, a maintainer does this for you. When merging multiple changes, use bullet points in the commit message to break down the changes.

------
Labels
------

.. table:: GitHub PR labels

   ============= =======
   Label         Effect
   ============= =======
   ci:profiling  Adds the *profiling* job to the CI pipeline. Profiling artifacts are saved for 10 days.
   ci:docs-only  Only runs the *docs/readthedocs* job; omits all other pipeline jobs.
   ============= =======

============================
Coding style and conventions
============================

-------------------
General conventions
-------------------

Always use space indentation (four spaces). Never commit a tab (``\t``).

------------------
Python doc-strings
------------------

Tensile uses `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ to pull in documentation from doc-strings and integrate them into this site. Use the following guidelines when writing Python functions and modules to maintain quality and consistency.

1. Identify the parameters and returned values with type-hints.
2. For all functions, specify doc-string describing the parameters, return value, and any exception. However, if the function is small and the implementation is straightforward, a one-line doc-string is sufficient.
3. Don't include types directly in the doc-string. Add them as type-hints in the function definition.
4. For doc-string styling, use the `Google Python style guide <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

.. _commit-messages:

---------------
Commit messages
---------------

1. Use `conventional commits <https://www.conventionalcommits.org/>`_.
2. Use the present imperative tense. For example, "add" not "adds" or "added".
3. Don't end the message with a period (.).

============
Code license
============

All code contributed to this project will be licensed under the given `LICENSE <https://github.com/ROCm/Tensile/blob/develop/LICENSE.md>`_. Your contribution will be accepted under the same license.

For each new file, include the following licensing header:

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

===============
Release cadence
===============

Official Tensile releases are subject to the general ROCm release cadence, which typically follows a quarterly cycle. Latest stable versions of Tensile are available in the **staging** branch.
