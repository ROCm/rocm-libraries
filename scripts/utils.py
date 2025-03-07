
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

import re


# Remove the text between the first occurance of start and the first occurrence of stop,
# after start, where stop and start are compiled regex patterns.
def remove_between_regex(text, start_regex, stop_regex):
    start_match = start_regex.search(text)
    if start_match:
        stop_match = stop_regex.search(text, start_match.end())
        if stop_match:
            return text[: start_match.start()] + text[stop_match.end() :]
    return text


# Repeatedly remove text between the first occurance of start and the first occurrence
# of stop, after start, until there are no more starts and stops, where stop and start
# are compiled regex patterns.
def remove_all_between_regex(text, start_regex, stop_regex):
    while True:
        start_match = start_regex.search(text)
        if start_match:
            stop_match = stop_regex.search(text, start_match.end())
        if start_match and stop_match:
            text = text[: start_match.start()] + text[stop_match.end() :]
        else:
            break
    return text


# Remove the text between the first occurance of start and the first occurrence of stop,
# after start.
def remove_between(text, start, stop):
    return remove_between_regex(
        text,
        re.compile(re.escape(start)),
        re.compile(re.escape(stop)),
    )


# Repeatedly remove the text between the first occurance of start and the first
# occurrence of stop, after start, until there are no more starts and stops.
def remove_all_between(text, start, stop):
    return remove_all_between_regex(
        text,
        re.compile(re.escape(start)),
        re.compile(re.escape(stop)),
    )


# Get the text between the first occurance of start and the first occurrence of stop,
# after start, where stop and start are compiled regex patterns.
def get_between_regex(text, start_regex, stop_regex):
    start_match = start_regex.search(text)
    if start_match:
        stop_match = stop_regex.search(text, start_match.end())
    if start_match and stop_match:
        return text[start_match.end() : stop_match.start()]
    return ""


# Get the text between the first occurance of start and the first occurrence of stop,
# after start.
def get_between(text, start, stop):
    return get_between_regex(
        text,
        re.compile(re.escape(start)),
        re.compile(re.escape(stop)),
    )


# Clean up all of the lines by removing comments and whitespace
def clean_lines(source_lines, leave_comments):
    comment_start = re.compile(r"/\*")
    comment_end = re.compile(r"\*/")
    meta_start = re.compile(r"\.amdgpu_metadata")
    meta_end = re.compile(r"\.end_amdgpu_metadata")
    result = []
    full_text = "\n".join(source_lines)
    if not leave_comments:
        full_text = remove_all_between_regex(full_text, comment_start, comment_end)
    full_text = remove_all_between_regex(full_text, meta_start, meta_end)
    source_lines = full_text.split("\n")
    for my_line in source_lines:
        if not leave_comments and "//" in my_line:
            my_line = my_line[0 : my_line.find("//")]
        my_line = my_line.strip()
        my_line = my_line.replace("\\", "\\\\")
        if my_line:
            result += [my_line]
    return result
