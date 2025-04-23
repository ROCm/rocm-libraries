#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

include(ExportHeader)

include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(RegisterOp)
include(CheckCXXLinkerFlag)
include(CheckCXXSourceCompiles)

add_library(migraphx
    adjust_allocation.cpp
    analyze_streams.cpp
    apply_alpha_beta.cpp
    argument.cpp
    autocast_fp8.cpp
    auto_contiguous.cpp
    base64.cpp
    common.cpp
    common_dims.cpp
    compile_src.cpp
    convert_to_json.cpp
    cpp_generator.cpp
    dead_code_elimination.cpp
    dom_info.cpp
    dynamic_loader.cpp
    eliminate_allocation.cpp
    eliminate_common_subexpression.cpp
    eliminate_concat.cpp
    eliminate_contiguous.cpp
    eliminate_convert.cpp
    eliminate_data_type.cpp
    eliminate_identity.cpp
    eliminate_pad.cpp
    env.cpp
    file_buffer.cpp
    fileutils.cpp
    fp_to_double.cpp
    fp8_ocp_to_fnuz.cpp
    fuse_concat.cpp
    fuse_pointwise.cpp
    fuse_pointwise_reduce.cpp
    fuse_reduce.cpp
    generate.cpp
    inline_module.cpp
    insert_pad.cpp
    instruction.cpp
    json.cpp
    layout_convolution.cpp
    lexing.cpp
    load_save.cpp
    make_op.cpp
    memory_coloring.cpp
    module.cpp
    msgpack.cpp
    netron_output.cpp
    normalize_attributes.cpp
    normalize_ops.cpp
    op_enums.cpp
    operation.cpp
    optimize_module.cpp
    pad_calc.cpp
    param_utils.cpp
    pass.cpp
    pass_manager.cpp
    permutation.cpp
    preallocate_param.cpp
    process.cpp
    program.cpp
    propagate_constant.cpp
    promote_literals.cpp
    quantization.cpp
    quantize_int4.cpp
    quantize_8bits.cpp
    reduce_dims.cpp
    register_op.cpp
    register_target.cpp
    replace_allocate.cpp
    rewrite_reduce.cpp
    rewrite_dot.cpp
    simplify_qdq.cpp
    split_reduce.cpp
    sqlite.cpp
    rewrite_gelu.cpp
    rewrite_low_precision.cpp
    rewrite_pooling.cpp
    rewrite_quantization.cpp
    rewrite_rnn.cpp
    rewrite_topk.cpp
    schedule.cpp
    serialize.cpp
    shape.cpp
    shape_transform_descriptor.cpp
    simplify_algebra.cpp
    simplify_dyn_ops.cpp
    simplify_reshapes.cpp
    split_single_dyn_dim.cpp
    target.cpp
    tmp_dir.cpp
    truncate_float.cpp
    value.cpp
    verify_args.cpp
)

if(WIN32)
    # Due to compilation crashing, we need to use type-erased matchers on Windows.
    target_compile_definitions(migraphx PUBLIC MIGRAPHX_USE_TYPE_ERASED_MATCHERS=1)
    target_compile_options(migraphx PUBLIC "-mno-ms-bitfields")
endif()

configure_file(version.h.in include/migraphx/version.h)
add_library(migraphx_version INTERFACE)
rocm_install_targets(
  TARGETS migraphx_version
  INCLUDE
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

rocm_set_soversion(migraphx ${MIGRAPHX_SO_VERSION})
function(register_migraphx_ops)
    foreach(OP ${ARGN})
        register_op(migraphx HEADER migraphx/op/${OP}.hpp OPERATORS op::${OP})
    endforeach()
endfunction()
register_migraphx_ops(
    abs
    acosh
    acos
    add
    allocate
    argmax
    argmin
    asinh
    asin
    as_shape
    atanh
    atan
    bit_cast
    bitwise_and
    broadcast
    broadcast_for_dot
    broadcast_with_dims
    capture
    ceil
    clip
    concat
    contiguous
    convert
    convolution
    convolution_backwards
    cosh
    cos
    dequantizelinear
    dimensions_of
    div
    dot
    elu
    equal
    erf
    exp
    fill
    flatten
    floor
    fmod
    gather
    gathernd
    get_tuple_elem
    greater
    group_query_attention
    gru
    identity
    if_op
    im2col
    isinf
    isnan
    layout
    leaky_relu
    less
    load
    log
    log2
    logical_and
    logical_or
    logical_xor
    logsoftmax
    loop
    lrn
    lstm
    max
    min
    mod
    mul
    multibroadcast
    multinomial
    nearbyint
    neg
    nonmaxsuppression
    nonzero
    onehot
    outline
    pack_int4
    pad
    pointwise
    pooling
    pow
    prefix_scan_sum
    prelu
    quant_convolution
    quant_dot
    quantizelinear
    random_uniform
    random_seed
    recip
    reduce_all
    reduce_any
    reduce_max
    reduce_mean
    reduce_min
    reduce_prod
    reduce_sum
    relu
    reshape
    reshape_lazy
    resize
    reverse
    rnn
    rnn_last_cell_output
    rnn_last_hs_output
    rnn_var_sl_last_output
    roialign
    rsqrt
    run_on_target
    scalar
    scan_slice
    scatter_none
    scatter_add
    scatter_mul
    scatter_min
    scatter_max
    scatternd_add
    scatternd_mul
    scatternd_none
    scatternd_max
    scatternd_min
    select_module
    sigmoid
    sign
    sinh
    sin
    slice
    softmax
    sqdiff
    sqrt
    squeeze
    step
    sub
    tanh
    tan
    topk
    transpose
    unary_not
    undefined
    unique
    unknown
    unpack_int4
    unsqueeze
    where
)
register_op(migraphx HEADER migraphx/op/rnn_variable_seq_lens.hpp OPERATORS op::rnn_var_sl_shift_output op::rnn_var_sl_shift_sequence)
register_op(migraphx HEADER migraphx/builtin.hpp OPERATORS builtin::literal builtin::param builtin::returns)
rocm_clang_tidy_check(migraphx)
migraphx_generate_export_header(migraphx)
rocm_install_targets(
  PRIVATE
  TARGETS migraphx
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(migraphx PUBLIC migraphx_version)

if(NOT WIN32)
    check_cxx_linker_flag(-lstdc++fs HAS_LIB_STD_FILESYSTEM)
    if(HAS_LIB_STD_FILESYSTEM)
        target_link_libraries(migraphx PRIVATE -lstdc++fs)
    endif()
    target_link_libraries(migraphx PRIVATE -ldl)
endif()

target_include_directories(migraphx SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)
target_link_libraries(migraphx PUBLIC Threads::Threads)

if(MIGRAPHX_HAS_EXECUTORS AND ParallelSTL_USES_TBB)
    list(APPEND MIGRAPHX_CONFIG_DEPENDS PACKAGE TBB)
endif()

if(MIGRAPHX_HAS_EXECUTORS)
    message(STATUS "Parallel STL enabled")
    target_compile_definitions(migraphx PUBLIC MIGRAPHX_HAS_EXECUTORS=1)
    target_link_libraries(migraphx PUBLIC ${ParallelSTL_LIBRARIES})
else()
    message(STATUS "Parallel STL disabled")
    target_compile_definitions(migraphx PUBLIC MIGRAPHX_HAS_EXECUTORS=0)
endif()

find_package(nlohmann_json 3.8.0 REQUIRED)
target_link_libraries(migraphx PRIVATE nlohmann_json::nlohmann_json)

find_package(SQLite3 REQUIRED)
target_link_libraries(migraphx PRIVATE SQLite::SQLite3)

find_package(msgpackc-cxx QUIET)
if(NOT msgpackc-cxx_FOUND)
    find_package(msgpack REQUIRED)
endif()
target_link_libraries(migraphx PRIVATE msgpackc-cxx)
# Make this available to the tests
target_link_libraries(migraphx INTERFACE $<BUILD_INTERFACE:msgpackc-cxx>)

add_library(migraphx_all_targets INTERFACE)

add_subdirectory(api)
add_subdirectory(driver)
add_subdirectory(onnx)
add_subdirectory(tf)

if(MIGRAPHX_ENABLE_PYTHON)
add_subdirectory(py)
endif()
add_subdirectory(targets/ref)
target_link_libraries(migraphx_all_targets INTERFACE migraphx_ref)
if(MIGRAPHX_ENABLE_CPU)
add_subdirectory(targets/cpu)
target_link_libraries(migraphx_all_targets INTERFACE migraphx_cpu)
target_compile_definitions(migraphx_all_targets INTERFACE -DHAVE_CPU)
endif()
if(MIGRAPHX_ENABLE_GPU)
    if(MIGRAPHX_USE_MIOPEN)
        list(APPEND MIGRAPHX_CONFIG_DEPENDS PACKAGE MIOpen)
    endif()
    if(MIGRAPHX_USE_ROCBLAS)
        list(APPEND MIGRAPHX_CONFIG_DEPENDS PACKAGE rocblas)
    endif()
    if(MIGRAPHX_USE_HIPBLASLT)
        list(APPEND MIGRAPHX_CONFIG_DEPENDS PACKAGE hipblaslt)
    endif()
add_subdirectory(targets/gpu)
target_link_libraries(migraphx_all_targets INTERFACE migraphx_gpu)
target_compile_definitions(migraphx_all_targets INTERFACE -DHAVE_GPU)
endif()
if(MIGRAPHX_ENABLE_FPGA)
add_subdirectory(targets/fpga)
target_link_libraries(migraphx_all_targets INTERFACE migraphx_fpga)
target_compile_definitions(migraphx_all_targets INTERFACE -DHAVE_FPGA)
endif()

if(MIGRAPHX_USE_MIOPEN)
    target_compile_definitions(migraphx_all_targets INTERFACE MIGRAPHX_USE_MIOPEN=1)
else()
    target_compile_definitions(migraphx_all_targets INTERFACE MIGRAPHX_USE_MIOPEN=0)
endif()

if(HAVE_HALF_EXPR)
    target_compile_definitions(migraphx PUBLIC -DHAS_HALF_V1)
endif()

if(BUILD_DEV)
    target_compile_definitions(migraphx PUBLIC -DBUILD_DEV)
endif()

target_compile_definitions(migraphx PUBLIC MIGRAPHX_CXX_COMPILER="${CMAKE_CXX_COMPILER}")

rocm_export_targets(
  TARGETS migraphx::migraphx_c
  NAMESPACE migraphx::
  DEPENDS
    Threads
    ${MIGRAPHX_CONFIG_DEPENDS}
)


