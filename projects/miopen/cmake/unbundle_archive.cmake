# unbundle_archive.cmake
# Extracts single-arch objects from a fat static archive.
#
# Handles the new-style HIP compilation model where device code is embedded
# in .hip_fatbin ELF sections (CCOB format) within regular host objects,
# rather than the old clang-offload-bundler fat binary format.
#
# Required -D arguments:
#   FAT_ARCHIVE   - Path to the fat .a file
#   ARCH          - GPU architecture (e.g., gfx90a)
#   OUTPUT        - Path for the output per-arch .a file
#   BUNDLER       - Path to clang-offload-bundler
#   AR            - Path to ar tool
#   LLVM_OBJCOPY  - Path to llvm-objcopy (NOT GNU objcopy, which rewrites ELFs)

cmake_minimum_required(VERSION 3.16)

get_filename_component(work_dir "${OUTPUT}" DIRECTORY)
set(work_dir "${work_dir}/_unbundle_${ARCH}")

# Clean and create work directory
file(REMOVE_RECURSE "${work_dir}")
file(MAKE_DIRECTORY "${work_dir}")

# Strip feature suffixes (e.g., gfx942:sramecc+:xnack- → gfx942)
string(FIND "${ARCH}" ":" _colon_pos)
if(_colon_pos GREATER -1)
    string(SUBSTRING "${ARCH}" 0 ${_colon_pos} BASE_ARCH)
else()
    set(BASE_ARCH "${ARCH}")
endif()

# Extract .o files from fat archive
execute_process(
    COMMAND ${AR} x "${FAT_ARCHIVE}"
    WORKING_DIRECTORY "${work_dir}"
    OUTPUT_VARIABLE ar_output
    ERROR_VARIABLE  ar_error
    RESULT_VARIABLE ar_result)
if(NOT ar_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to extract ${FAT_ARCHIVE}\n"
        "  exit code: ${ar_result}\n"
        "  stderr: ${ar_error}\n"
        "  stdout: ${ar_output}")
endif()

file(GLOB obj_files "${work_dir}/*.o" "${work_dir}/*.obj")

set(thin_objs)
foreach(obj IN LISTS obj_files)
    get_filename_component(obj_name "${obj}" NAME)
    set(fatbin_file "${work_dir}/${obj_name}.fatbin")

    # Extract .hip_fatbin section using llvm-objcopy (GNU objcopy rewrites the
    # ELF in-place even for --dump-section, inflating it by ~500KB per object)
    execute_process(
        COMMAND ${LLVM_OBJCOPY} --dump-section .hip_fatbin=${fatbin_file} "${obj}"
        RESULT_VARIABLE extract_result
        ERROR_QUIET)

    if(NOT extract_result EQUAL 0)
        # No .hip_fatbin section — pure host code (e.g. sharded instantiation
        # callers).  Include as-is since it contains no device code to unbundle.
        set(thin_obj "${work_dir}/thin_${obj_name}")
        file(COPY_FILE "${obj}" "${thin_obj}")
        list(APPEND thin_objs "${thin_obj}")
        continue()
    endif()

    # List targets in the extracted fatbin bundle
    execute_process(
        COMMAND ${BUNDLER} --type=o "--input=${fatbin_file}" -list
        OUTPUT_VARIABLE targets_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE list_result
        ERROR_QUIET)

    if(NOT list_result EQUAL 0)
        continue()
    endif()

    # Find the target triple for our architecture and collect all targets
    string(REPLACE "\n" ";" target_lines "${targets_output}")
    set(matched_target "")
    set(host_target "")
    set(all_targets "")
    foreach(line IN LISTS target_lines)
        string(STRIP "${line}" line)
        if(line STREQUAL "")
            continue()
        endif()
        list(APPEND all_targets "${line}")
        if(line MATCHES "^host-")
            set(host_target "${line}")
        else()
            # Try full ARCH first (handles archives built with feature qualifiers),
            # then fall back to base arch without feature suffixes.
            # Use literal substring match — MATCHES uses regex, and arch
            # strings like gfx942:sramecc+:xnack- contain regex metacharacters.
            string(FIND "${line}" "${ARCH}" _arch_pos)
            if(NOT _arch_pos EQUAL -1)
                set(matched_target "${line}")
            elseif(NOT BASE_ARCH STREQUAL ARCH)
                string(FIND "${line}" "${BASE_ARCH}" _arch_pos)
                if(NOT _arch_pos EQUAL -1)
                    set(matched_target "${line}")
                endif()
            endif()
        endif()
    endforeach()

    if(NOT matched_target)
        continue()
    endif()

    # Unbundle: extract all targets into separate files so we can pick just
    # the ones we need (bundler requires all targets to be listed)
    set(unbundle_targets "")
    set(unbundle_outputs "")
    set(device_output "")
    set(host_output "")
    foreach(target IN LISTS all_targets)
        string(REPLACE "/" "_" safe_target "${target}")
        set(out_file "${work_dir}/${obj_name}.${safe_target}")
        if(unbundle_targets)
            set(unbundle_targets "${unbundle_targets},${target}")
        else()
            set(unbundle_targets "${target}")
        endif()
        list(APPEND unbundle_outputs "--output=${out_file}")
        if(target STREQUAL matched_target)
            set(device_output "${out_file}")
        elseif(target STREQUAL host_target)
            set(host_output "${out_file}")
        endif()
    endforeach()

    execute_process(
        COMMAND ${BUNDLER} --type=o
            "--targets=${unbundle_targets}"
            "--input=${fatbin_file}"
            ${unbundle_outputs}
            --unbundle
        OUTPUT_VARIABLE unbundle_output
        ERROR_VARIABLE  unbundle_error
        RESULT_VARIABLE unbundle_result)

    if(NOT unbundle_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to unbundle ${obj_name} for ${ARCH}\n"
            "  exit code: ${unbundle_result}\n"
            "  stderr: ${unbundle_error}\n"
            "  stdout: ${unbundle_output}")
    endif()

    # Re-bundle with only host + target arch, compressed
    set(rebundle_targets "${host_target},${matched_target}")
    set(thin_fatbin "${work_dir}/${obj_name}.thin_fatbin")
    execute_process(
        COMMAND ${BUNDLER} --type=o
            "--targets=${rebundle_targets}"
            "--input=${host_output}" "--input=${device_output}"
            "--output=${thin_fatbin}"
            --compress
        OUTPUT_VARIABLE rebundle_output
        ERROR_VARIABLE  rebundle_error
        RESULT_VARIABLE rebundle_result)

    if(NOT rebundle_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to re-bundle ${obj_name} for ${ARCH}\n"
            "  exit code: ${rebundle_result}\n"
            "  stderr: ${rebundle_error}\n"
            "  stdout: ${rebundle_output}")
    endif()

    # Replace the .hip_fatbin section with the single-arch fatbin.
    # --update-section preserves relocations (.hipFatBinSegment references
    # .hip_fatbin) and works on both ELF and COFF, but requires the new
    # content to be no larger than the original section.
    # For multi-arch bundles this holds: we removed device code blobs.
    # For single-arch bundles, re-encoding adds overhead that can make the
    # thin fatbin larger than the original. In that case the object already
    # contains only our target arch, so use it unchanged.
    file(SIZE "${fatbin_file}" _orig_fatbin_size)
    file(SIZE "${thin_fatbin}" _thin_fatbin_size)

    if(_thin_fatbin_size GREATER _orig_fatbin_size)
        # Count device targets (those containing "gfx"). If more than one,
        # the thin fatbin should have been smaller — something is wrong.
        set(_num_device_targets 0)
        foreach(_t IN LISTS all_targets)
            if(_t MATCHES "gfx")
                math(EXPR _num_device_targets "${_num_device_targets} + 1")
            endif()
        endforeach()
        if(NOT _num_device_targets EQUAL 1)
            message(FATAL_ERROR
                "Thin fatbin for ${obj_name} is larger than original "
                "(${_thin_fatbin_size} > ${_orig_fatbin_size}) but bundle "
                "has ${_num_device_targets} device targets")
        endif()
        list(APPEND thin_objs "${obj}")
        continue()
    endif()

    set(thin_obj "${work_dir}/thin_${obj_name}")
    execute_process(
        COMMAND ${LLVM_OBJCOPY}
            --update-section=.hip_fatbin=${thin_fatbin}
            "${obj}" "${thin_obj}"
        OUTPUT_VARIABLE patch_output
        ERROR_VARIABLE  patch_error
        RESULT_VARIABLE patch_result)

    if(patch_result EQUAL 0)
        list(APPEND thin_objs "${thin_obj}")
    else()
        message(FATAL_ERROR
            "Failed to patch ${obj_name} for ${ARCH}\n"
            "  exit code: ${patch_result}\n"
            "  stderr: ${patch_error}\n"
            "  stdout: ${patch_output}")
    endif()
endforeach()

# Create per-arch archive from patched objects
if(thin_objs)
    list(LENGTH thin_objs count)

    # Write object list to a response file to avoid command-line length
    # limits on Windows (~1400 objects can exceed the 32767 char cap).
    set(rsp_file "${work_dir}/thin_objs.rsp")
    file(WRITE "${rsp_file}" "")
    foreach(_obj IN LISTS thin_objs)
        file(APPEND "${rsp_file}" "${_obj}\n")
    endforeach()

    execute_process(
        COMMAND ${AR} rcs "${OUTPUT}" "@${rsp_file}"
        OUTPUT_VARIABLE ar_output
        ERROR_VARIABLE  ar_error
        RESULT_VARIABLE ar_result)
    if(NOT ar_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to create ${OUTPUT}\n"
            "  exit code: ${ar_result}\n"
            "  stderr: ${ar_error}\n"
            "  stdout: ${ar_output}\n"
            "  object count: ${count}")
    endif()
    message(STATUS "Created ${OUTPUT} with ${count} objects")
else()
    message(WARNING "${ARCH} is included in MIOpen's GPU_TARGETS, but no device code objects were found for ${ARCH} in ${FAT_ARCHIVE}. Did you build CK to include ${ARCH}? Creating empty archive")
    # Create an empty archive so the build does not fail with a missing output.
    execute_process(
        COMMAND ${AR} rcs "${OUTPUT}"
        OUTPUT_VARIABLE ar_output
        ERROR_VARIABLE  ar_error
        RESULT_VARIABLE ar_empty_result)
    if(NOT ar_empty_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to create empty archive ${OUTPUT}\n"
            "  exit code: ${ar_empty_result}\n"
            "  stderr: ${ar_error}")
    endif()
endif()

# Cleanup
file(REMOVE_RECURSE "${work_dir}")
