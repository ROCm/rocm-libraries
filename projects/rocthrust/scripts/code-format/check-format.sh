#!/usr/bin/env bash

# ########################################################################
# Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# ########################################################################

SOURCE_COMMIT="$1"
if [ "$#" -gt 0 ]; then
    shift
fi

# If no source commit is given target the default branch
if [ "x$SOURCE_COMMIT" = "x" ]; then
    # If remote is not set use the remote of the current branch or fallback to "origin"
    if [ "x$REMOTE" = "x" ]; then
        BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        REMOTE="$(git config --local --get "branch.$BRANCH.remote" || echo 'origin')"
    fi
    SOURCE_COMMIT="remotes/$REMOTE/HEAD"
fi

# Force colored diff output
DIFF_COLOR_SAVED="$(git config --local --get color.diff)"
if [ "x$DIFF_COLOR_SAVED" != "x" ]; then
    git config --local --replace-all "color.diff" "always"
else
    git config --local --add "color.diff" "always"
fi

scratch="$(mktemp -t check-format.XXXXXXXXXX)"
finish () {
    # Remove temporary file
    rm -rf "$scratch"
    # Restore setting
    if [ "x$DIFF_COLOR_SAVED" != "x" ]; then
        git config --local --replace-all "color.diff" "$DIFF_COLOR_SAVED"
    else
        git config --local --unset "color.diff"
    fi
}
# The trap will be invoked whenever the script exits, even due to a signal, this is a bash only
# feature
trap finish EXIT

GIT_CLANG_FORMAT="${GIT_CLANG_FORMAT:-git-clang-format}"
"$GIT_CLANG_FORMAT" --style=file --diff "$@" "$SOURCE_COMMIT" > "$scratch"

# Check for no-ops
grep '^no modified files to format$\|^clang-format did not modify any files$' \
    "$scratch" > /dev/null && exit 0

# Dump formatting diff and signal failure
printf \
"\033[31m==== FORMATTING VIOLATIONS DETECTED ====\033[0m
run '\033[33m%s --style=file %s %s\033[0m' to apply these formatting changes\n\n" \
"$GIT_CLANG_FORMAT" "$*" "$SOURCE_COMMIT"

cat "$scratch"
exit 1
