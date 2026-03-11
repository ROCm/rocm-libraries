#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from collections import Counter

import pandas as pd

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

""".format(types_count.to_markdown(index=False), df.to_markdown(index=False))

with open("codeql/build/types_count.md", "w") as file:
    print(file.name)
    file.write(markdown)

with open("codeql/build/codeql.html", "w") as file:
    print(file.name)
    file.write(html)
