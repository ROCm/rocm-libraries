# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Provider test-category labeling. Included after all test registrations from
# the parent tests/CMakeLists.txt to preserve its CMake directory scope.

# Apply YAML-driven quick/standard/comprehensive/full tier labels to the tests
# above. Must run in THIS directory scope: CMake (pre-3.28, which this project's
# cmake_minimum_required predates) only allows set_property(TEST ...)/
# set_tests_properties() on a test from the directory that add_test()'d it, so
# calling this from a parent scope (e.g. hip-kernel-provider/src/CMakeLists.txt)
# would silently do nothing for them. Guarded because rocke/platform also builds
# standalone (outside hip-kernel-provider), where neither the shared parser
# (apply_ctest_category_labels) nor ROCKE_ENGINE_CTEST_CATEGORIES_YAML/
# HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED exist. Without this, these tests
# carry only the hardcoded "ckc;unit;host" label and silently drop out of
# `ctest -L <tier>` selection -- the mechanism TheRock CI's per-tier
# test_hipkernelprovider.py relies on.
if(COMMAND apply_ctest_category_labels
   AND HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED
   AND ROCKE_ENGINE_CTEST_CATEGORIES_YAML)
  apply_ctest_category_labels("${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}")
endif()
