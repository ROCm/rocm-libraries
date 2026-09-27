cmake_minimum_required(VERSION 3.16)
include("${CMAKE_CURRENT_LIST_DIR}/../grouped_conv_arch_plan.cmake")

ck_dispatcher_conv_variants_for_arch("gfx1100" _v)
if(NOT _v STREQUAL "fwd")
  message(FATAL_ERROR "gfx1100 variants: '${_v}'")
endif()

ck_dispatcher_conv_variants_for_arch("gfx942" _v)
if(NOT _v STREQUAL "fwd;bwd_weight;bwd_data")
  message(FATAL_ERROR "gfx942 variants: '${_v}'")
endif()

ck_dispatcher_conv_rule_set_for_arch("gfx1200" "tests" _rs)
if(NOT _rs STREQUAL "rdna")
  message(FATAL_ERROR "gfx1200 rule set: '${_rs}'")
endif()

ck_dispatcher_conv_rule_set_for_arch("gfx942" "tests" _rs)
if(NOT _rs STREQUAL "tests")
  message(FATAL_ERROR "gfx942 rule set: '${_rs}'")
endif()

ck_dispatcher_has_gfx9_or_gfx1250("gfx942;gfx1100" _has)
if(NOT _has)
  message(FATAL_ERROR "mixed gfx942+gfx1100 must keep gfx9")
endif()

ck_dispatcher_has_gfx9_or_gfx1250("gfx1100;gfx1200" _has)
if(_has)
  message(FATAL_ERROR "rdna-only must not look like gfx9")
endif()

ck_dispatcher_has_gfx9_or_gfx1250("gfx1250" _has)
if(NOT _has)
  message(FATAL_ERROR "gfx1250 must count as the CDNA-side catalog")
endif()

message(STATUS "dispatcher arch plan tests passed")
