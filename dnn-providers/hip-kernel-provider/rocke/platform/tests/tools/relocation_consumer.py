# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Relocate a selected SDK configuration and execute a CMake consumer.

Shared by source SDK tests and installed minimal-SDK regression tests. Neither
caller needs to invoke another test or depend on its environment discovery.
"""

import subprocess
from pathlib import Path


def run_relocated_consumer(root: Path, build: Path, config: str, consumer: str) -> None:
    """Install, relocate, compile and run one consumer of the selected build."""
    if consumer not in ("cpp_lowerer", "c_mma"):
        raise ValueError(f"unknown relocation consumer {consumer!r}")
    prefix = root / "install"
    subprocess.run(
        ["cmake", "--install", str(build), "--config", config, "--prefix", str(prefix)],
        check=True,
    )
    relocated = root / "relocated"
    prefix.rename(relocated)
    source = root / "consumer"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(consumer LANGUAGES C CXX)\n"
        "find_package(ckc CONFIG REQUIRED)\n"
        "if(NOT TARGET Threads::Threads)\n"
        '  message(FATAL_ERROR "missing transitive thread dependency")\n'
        "endif()\n"
        f"add_executable(consumer main.{'c' if consumer == 'c_mma' else 'cpp'})\n"
        "set_target_properties(consumer PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF LINKER_LANGUAGE CXX)\n"
        "target_link_libraries(consumer PRIVATE ckc::rocke_core)\n"
        'file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/consumer-path-$<CONFIG>.txt" '
        'CONTENT "$<TARGET_FILE:consumer>")\n'
    )
    if consumer == "cpp_lowerer":
        # Taking the API address forces archive extraction without a GPU or a
        # valid kernel; all detector dependencies must resolve transitively.
        (source / "main.cpp").write_text(
            '#include "rocke/lower_llvm.h"\n'
            "auto volatile lower = &rocke_lower_kernel_to_llvm_ex;\n"
            "int main() { return lower == nullptr; }\n"
        )
    else:
        (source / "main.c").write_text(
            '#include "rocke/arch_target.h"\n'
            '#include "rocke/abi.h"\n'
            '#include "rocke/instance_conv_direct_fwd_wmma.h"\n'
            "#include <stddef.h>\n#include <string.h>\n"
            "int main(void) {\n"
            "  if (rocke_abi_version() != ROCKE_ABI_VERSION) return 6;\n"
            "  if (rocke_recipe_abi_level() != ROCKE_RECIPE_ABI) return 7;\n"
            "  rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();\n"
            "  p.groups = p.cpg = p.kpg = 1 << 20;\n"
            "  if (rocke_direct_conv_problem_total_c(&p) != (1LL << 40)) return 9;\n"
            "  if (rocke_direct_conv_problem_total_k(&p) != (1LL << 40)) return 10;\n"
            '  const char* id = "wmma_gfx1250_bf16f32_16x16x32_bf16";\n'
            "  const char* c = rocke_arch_mma_op_id_c_dtype(id);\n"
            "  const char* d = rocke_arch_mma_op_id_dst_dtype(id);\n"
            '  if (!c || !d || strcmp(c, "fp32") || strcmp(d, "bf16")) return 1;\n'
            "  if (rocke_arch_mma_op_id_c_dtype(NULL) != NULL) return 2;\n"
            '  if (rocke_arch_mma_op_id_c_dtype("unknown") != NULL) return 3;\n'
            "  if (rocke_arch_mma_c_frag_len(id) != 8 || rocke_arch_mma_dst_frag_len(id) != 8) return 4;\n"
            '  const rocke_arch_target_t* t = rocke_arch_target_from_gfx("gfx1250");\n'
            '  const rocke_mma_op_t* op = rocke_mma_catalog_op_for_shape(&t->mma, "wmma", "bf16", "bf16", c, 16, 16, 32, NULL, d);\n'
            "  if (!op || strcmp(op->op_id, id)) return 5;\n"
            "  return 0;\n}\n"
        )
    output = root / "consumer-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(output),
            f"-DCMAKE_PREFIX_PATH={relocated}",
            f"-DCMAKE_BUILD_TYPE={config}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(output), "--config", config], check=True)
    executable = (output / f"consumer-path-{config}.txt").read_text()
    subprocess.run([executable], check=True)
