# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os

root_dir = os.getcwd()
ck_tile_include = root_dir + "/projects/composablekernel/include/ck_tile"

# Regenerate the ck_tile aggregation headers. example/ck_tile has no headers to
# generate; its formatting is covered by the clang-format and crlf-checker hooks.
os.chdir(ck_tile_include)
_ = os.system("python remod.py")
