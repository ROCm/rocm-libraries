#!/usr/bin/env python3

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


import pandas as pd
from collections import Counter

try:
    df = pd.read_csv("codeql/build/codeql.csv")

    types_count = pd.DataFrame([Counter(df.iloc[:, 0])])
except pd.errors.EmptyDataError:
    # No CodeQL warnings or errors!
    df = pd.DataFrame()
    types_count = pd.DataFrame()

html = (
    "<h1>CodeQL Report</h1>\n"
    + "<text>"
    + types_count.to_html(index=False, justify="center")
    + "</text>\n<br></br>\n"
    + df.to_html(index=False, justify="center")
)

markdown = """## Results Summary

{}

<details><summary>Full table of results</summary>

{}
</details>

""".format(
    types_count.to_markdown(index=False), df.to_markdown(index=False)
)

with open("codeql/build/types_count.md", "w") as file:
    print(file.name)
    file.write(markdown)

with open("codeql/build/codeql.html", "w") as file:
    print(file.name)
    file.write(html)
