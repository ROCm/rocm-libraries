################################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

## Should not import any module

import sys

__TI_DEBUG_LEVEL__ = 0

########################################
# Text format
########################################

def slash(comment):
    """
    This comment is a single line // MYCOMMENT
    """
    return "// %s\n"%comment

def slash50(comment):
    """
    This comment is a single line // MYCOMMENT
    """
    return "%-50s // %s\n"%("", comment)

def block(comment):
    """
    This comment is a single line /* MYCOMMENT  */
    """
    return "/* %s */\n"%comment

def blockNewLine(comment):
    """
    This comment is a blank line followed by /* MYCOMMENT  */
    """
    return "\n/* %s */\n"%comment

def block3Line(comment):
    kStr = "\n/******************************************/\n"
    for line in comment.split("\n"):
        kStr += "/*"
        kStr += " %-38s " % line
        kStr += "*/\n"
    kStr += "/******************************************/\n"
    return kStr

########################################
# format string with comment
########################################

def formatStr(outputInlineAsm, instStr, comment) -> str:
    instStr = "\"%s"%instStr if outputInlineAsm else instStr
    if outputInlineAsm:
        instStr += "\\n\\t\""
    if comment:
        return "%-50s // %s\n" % (instStr, comment)
    else:
        return "%s\n" % (instStr)

########################################
# Terminal output
########################################

def print1(message):
    if __TI_DEBUG_LEVEL__ >= 1:
        print(message)
        sys.stdout.flush()

def print2(message):
    if __TI_DEBUG_LEVEL__ >= 2:
        print(message)
        sys.stdout.flush()

def printWarning(message):
    print("TensileInstructions::WARNING: %s" % message)
    sys.stdout.flush()

def printExit(message):
    print("TensileInstructions::FATAL: %s" % message)
    sys.stdout.flush()
    sys.exit(-1)

def printAssert(message):
    print("TensileInstructions::FATAL: %s" % message)
    assert 0
