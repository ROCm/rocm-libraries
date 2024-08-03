# ########################################################################
# Copyright 2020-2024 Advanced Micro Devices, Inc.
# ########################################################################

include(Benchmarks)

# ****************************************************************************
# Benchmarks
# ****************************************************************************
message (STATUS "Configuring internal benchmarks")

add_thrust_benchmark("internal_bench" "bench.cu" OFF)
