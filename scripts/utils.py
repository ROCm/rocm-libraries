import argparse
import itertools
import re

# Remove the text between the first occurance of start and the first occurrence of stop, after start, where stop and start are compiled regex patterns.
def remove_between_regex(text, start_regex, stop_regex):
    start_match = start_regex.search(text)
    if start_match:
        stop_match = stop_regex.search(text, start_match.end())
        if stop_match:
            return text[:start_match.start()] + text[stop_match.end():]
    return text

# Repeatedly remove the text between the first occurance of start and the first occurrence of stop, after start, until there are no more starts and stops, where stop and start are compiled regex patterns.
def remove_all_between_regex(text, start_regex, stop_regex):
    while True:
        start_match = start_regex.search(text)
        if start_match:
            stop_match = stop_regex.search(text, start_match.end())
        if start_match and stop_match:
            text = text[:start_match.start()] + text[stop_match.end():]
        else:
            break
    return text

# Remove the text between the first occurance of start and the first occurrence of stop, after start.
def remove_between(text, start, stop):
    return remove_between_regex(text, re.compile(re.escape(start)), re.compile(re.escape(stop)))

# Repeatedly remove the text between the first occurance of start and the first occurrence of stop, after start, until there are no more starts and stops.
def remove_all_between(text, start, stop):
    return remove_all_between_regex(text, re.compile(re.escape(start)), re.compile(re.escape(stop)))

# Get the text between the first occurance of start and the first occurrence of stop, after start, where stop and start are compiled regex patterns.
def get_between_regex(text, start_regex, stop_regex):
    start_match = start_regex.search(text)
    if start_match:
        stop_match = stop_regex.search(text, start_match.end())
    if start_match and stop_match:
        return text[start_match.end() : stop_match.start()]
    return ""

# Get the text between the first occurance of start and the first occurrence of stop, after start.
def get_between(text, start, stop):
    return get_between_regex(text, re.compile(re.escape(start)), re.compile(re.escape(stop)))

# Clean up all of the lines by removing comments and whitespace
def clean_lines(source_lines, leave_comments):
    comment_start = re.compile("/\*")
    comment_end = re.compile("\*/")
    meta_start = re.compile("\.amdgpu_metadata")
    meta_end = re.compile("\.end_amdgpu_metadata")
    result = []
    full_text = "\n".join(source_lines)
    if not leave_comments:
        full_text = remove_all_between_regex(full_text, comment_start, comment_end)
    full_text = remove_all_between_regex(full_text, meta_start, meta_end)
    source_lines = full_text.split("\n")
    for my_line in source_lines:
        if not leave_comments and "//" in my_line:
            my_line = my_line[0:my_line.find("//")]
        my_line = my_line.strip()
        my_line = my_line.replace("\\", "\\\\")
        if my_line:
            result += [my_line]
    return result
