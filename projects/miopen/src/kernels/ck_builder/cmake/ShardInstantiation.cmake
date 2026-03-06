# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Generates sharded instantiation files for a given instance template.
# This mirrors CK's ShardInstantiation.cmake, adapted for MIOpen namespaces.
#
# Arguments:
#   INSTANCES_NAME  - base name of the instance (e.g. device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_instances)
#   TEMPLATE_FILE   - path to the .in template file
#   NUM_SHARDS      - number of shards to generate
#   SRC_LIST        - variable to append generated source files to
#   OUTPUT_DIR      - build directory for generated files
function(generate_sharded_instantiations)
    set(options)
    set(oneValueArgs INSTANCES_NAME TEMPLATE_FILE NUM_SHARDS SRC_LIST OUTPUT_DIR)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "${options}" "${oneValueArgs}" "${multiValueArgs}")

    set(INSTANCES ${ARG_INSTANCES_NAME})

    # 1. Configure the .in template to an .inc file
    configure_file(
        ${ARG_TEMPLATE_FILE}
        ${ARG_OUTPUT_DIR}/${INSTANCES}.inc
        @ONLY
    )

    set(SHARD_SOURCES)

    # 2. For each shard, generate a _shard_N.cpp from instantiate_shard.in
    set(SHARD_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/instantiate_shard.in)
    math(EXPR LAST_SHARD "${ARG_NUM_SHARDS} - 1")
    set(NUM_SHARDS ${ARG_NUM_SHARDS})

    set(EXTERN_TEMPLATE_STATEMENTS "")
    set(CALL_STATEMENTS "")

    foreach(SHARD_ID RANGE 0 ${LAST_SHARD})
        configure_file(
            ${SHARD_TEMPLATE}
            ${ARG_OUTPUT_DIR}/${INSTANCES}_shard_${SHARD_ID}.cpp
            @ONLY
        )
        list(APPEND SHARD_SOURCES ${ARG_OUTPUT_DIR}/${INSTANCES}_shard_${SHARD_ID}.cpp)

        # Build extern declaration for this shard
        string(APPEND EXTERN_TEMPLATE_STATEMENTS
            "extern template void add_${INSTANCES}_shard<${ARG_NUM_SHARDS}, ${SHARD_ID}>(\n"
            "    ${INSTANCES}& instances);\n"
        )

        # Build call statement for this shard
        string(APPEND CALL_STATEMENTS
            "    add_${INSTANCES}_shard<${ARG_NUM_SHARDS}, ${SHARD_ID}>(instances);\n"
        )
    endforeach()

    # 3. Configure the caller .cpp from call_shard.in
    set(CALLER_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/call_shard.in)
    configure_file(
        ${CALLER_TEMPLATE}
        ${ARG_OUTPUT_DIR}/${INSTANCES}.cpp
        @ONLY
    )
    list(APPEND SHARD_SOURCES ${ARG_OUTPUT_DIR}/${INSTANCES}.cpp)

    # 4. Append all generated files to the caller's source list
    set(${ARG_SRC_LIST} ${${ARG_SRC_LIST}} ${SHARD_SOURCES} PARENT_SCOPE)
endfunction()
