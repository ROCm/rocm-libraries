#!/usr/bin/env python

################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################


import argparse
import re

import utils

macro_definition = re.compile(r"^\s*\.macro\s+(\w+)\s+.*$")
macro_end = re.compile(r".endm(acro)?.*\n+")
set_definition = re.compile(r"^\s*\.set\s+(\w+),.*$")
set_end = re.compile(r"\n")


"""
This script removes unused macros from a kernel file.
"""


def is_macro_used(my_lines, macro_name):
    macro_usage = re.compile(r"^\s*" + macro_name + r"\s+.*$")
    counter = 0
    for my_line in my_lines:
        macro = macro_usage.match(my_line)
        if macro:
            counter += 1
    return counter > 0


def is_id_used(my_lines, id_name):
    id_usage = re.compile(r"^.*" + id_name + r".*$")
    counter = 0
    for my_line in my_lines:
        id_match = id_usage.match(my_line)
        if id_match:
            counter += 1
    return counter > 1


def removeUnusedMacros(full_text):
    result = full_text

    # Repeat until there is no change.
    # This allows us to remove macors that are only used in macros that are removed.
    while True:
        prev_result = result
        cleaned_lines = utils.clean_lines(result.split("\n"), False)
        for line in cleaned_lines:
            macro = macro_definition.match(line)
            if macro and not is_macro_used(cleaned_lines, macro.group(1)):
                result = utils.remove_all_between_regex(
                    result, re.compile(re.escape(line)), macro_end
                )
            var_set = set_definition.match(line)
            if var_set and not is_id_used(cleaned_lines, var_set.group(1)):
                result = utils.remove_all_between_regex(
                    result, re.compile(re.escape(line)), set_end
                )
        if result == prev_result:
            break

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Removes unused macros from a kernel file."
    )
    parser.add_argument("input_file", type=str, help="File with AMD GPU machine code")
    args = parser.parse_args()

    with open(args.input_file) as f:
        full_text = f.read()

    print(removeUnusedMacros(full_text))
