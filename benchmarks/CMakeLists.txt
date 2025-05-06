# ########################################################################
# Copyright 2024 Advanced Micro Devices, Inc.
# ########################################################################

include(Benchmarks)

# Benchmarks directory in project's root
set(BENCHMARKS_ROOT "${CMAKE_CURRENT_LIST_DIR}")

# Subdirectory of BENCHMARKS_ROOT containing all the .cu files grouped in
# subdirectories accordingly to the functionality benchmarked
set(BENCHMARKS_DIR "bench")

# ****************************************************************************
# Functions
# ****************************************************************************

# Gets all subdirs of benchmarks recursively
function(get_recursive_subdirs subdirs dir_prefix dir_name)
  set(dirs)
  file(GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${dir_prefix}/${dir_name}/*"
  )

  foreach(dir IN LISTS contents)
    if(IS_DIRECTORY "${dir}")
      list(APPEND dirs "${dir}")
    endif()
  endforeach()

  set(${subdirs} "${dirs}" PARENT_SCOPE)
endfunction()

function(add_bench_dir bench_dir)
  # Get algorithm name
  get_filename_component(algo_name "${bench_dir}" NAME_WLE)

  # For scan, we also append the parent diretory name, as the algo_name
  # will be exclusive/inclusive
  get_filename_component(PARENT_DIR "${bench_dir}" DIRECTORY)
  get_filename_component(PARENT_DIR_NAME "${PARENT_DIR}" NAME_WLE)
  if(PARENT_DIR_NAME STREQUAL "scan")
    set(algo_name "${PARENT_DIR_NAME}_${algo_name}")
  endif()

  # Take all .cu (tests) inside the dir
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  # Set benchmark prefix as its relative path to benchmarks separated
  # by . instead of /
  file(RELATIVE_PATH bench_prefix "${BENCHMARKS_ROOT}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  # Add each benchmark as thrust benchmark
  foreach(bench_src IN LISTS bench_srcs)
      set(real_bench_src "${bench_src}")
      # Get file name without directory nor last extension
      get_filename_component(bench_name "${bench_src}" NAME_WLE)
      add_thrust_benchmark("${algo_name}_${bench_name}" ${bench_src} ON)
  endforeach()
endfunction()

# ****************************************************************************
# Benchmarks
# ****************************************************************************
message (STATUS "Configuring benchmarks")

# Get all the subdirectories inside the bench directory
get_recursive_subdirs(subdirs ${BENCHMARKS_ROOT} ${BENCHMARKS_DIR})

# Add benchmarks from each subdirectory present in bench
foreach(subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
