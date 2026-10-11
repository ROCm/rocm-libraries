# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Shared arch plan for CK Tile grouped-convolution instance libraries:
# which GPU_TARGETS get a catalog, which variants, and which rule set.
#
# Included from sibling directories, so it deliberately has no include_guard():
# every including scope needs its own copy of the functions below.

# Set <out_var> to the first supported RDNA target present in <targets>, or "".
# Iteration is over the candidate list, so the result does not depend on how the
# caller ordered GPU_TARGETS. The candidate list is only a superset filter: the
# generator re-derives datatype support from arch_specs and aborts configuration
# with a fatal error if a listed target is not actually supported there.
function(ck_dispatcher_rdna_target targets out_var)
    set(_candidates gfx1100 gfx1200 gfx1201)
    string(REPLACE " " ";" _requested "${targets}")
    list(FILTER _requested EXCLUDE REGEX "^$")
    foreach(_candidate IN LISTS _candidates)
        list(FIND _requested "${_candidate}" _index)
        if(NOT _index EQUAL -1)
            set(${out_var} "${_candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(ck_dispatcher_conv_is_rdna_arch arch out_var)
  ck_dispatcher_rdna_target("${arch}" _rdna)
  if(_rdna)
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(ck_dispatcher_conv_variants_for_arch arch out_var)
  ck_dispatcher_conv_is_rdna_arch("${arch}" _is_rdna)
  if(_is_rdna)
    set(${out_var} "fwd" PARENT_SCOPE)
  else()
    set(${out_var} "fwd;bwd_weight;bwd_data" PARENT_SCOPE)
  endif()
endfunction()

function(ck_dispatcher_conv_rule_set_for_arch arch default_rule_set out_var)
  ck_dispatcher_conv_is_rdna_arch("${arch}" _is_rdna)
  if(_is_rdna)
    set(${out_var} "rdna" PARENT_SCOPE)
  else()
    set(${out_var} "${default_rule_set}" PARENT_SCOPE)
  endif()
endfunction()

function(ck_dispatcher_has_gfx9_or_gfx1250 targets out_var)
  string(REPLACE " " ";" _requested "${targets}")
  list(FILTER _requested EXCLUDE REGEX "^$")
  foreach(_t IN LISTS _requested)
    if(_t MATCHES "^gfx9" OR _t STREQUAL "gfx1250")
      set(${out_var} TRUE PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_var} FALSE PARENT_SCOPE)
endfunction()
