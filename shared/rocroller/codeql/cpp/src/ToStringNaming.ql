// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @name Enforce 'toString' naming
 * @description Finds functions named 'ToString' as they should be renamed to 'toString'.
 * @kind problem
 * @problem.severity error
 * @precision high
 * @id rr/tostring-naming
 * @tags maintainability
 *       readability
 */

import cpp
import NamingUtils

from Function fc
where
  isInCheckedDirs(fc) and
  fc.toString().toLowerCase().regexpMatch("^tostring$") and
  not fc.toString() = "toString"
select fc, "function should be renamed to `toString`"
