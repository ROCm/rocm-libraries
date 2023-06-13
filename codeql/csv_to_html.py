#!/usr/bin/env python3

import pandas as pd
from collections import Counter

df = pd.read_csv("codeql/build/codeql.csv")
df = df.iloc[:, 2:]

titles = [
    "Type",
    "Description",
    "File Path",
    "Start Line",
    "Start Column",
    "End Line",
    "End Column",
]
df = pd.DataFrame([df.columns.values.tolist()] + df.values.tolist(), columns=titles)

types_count = str(dict(Counter(df.iloc[:, 0].tolist())))[1:-1]

html = (
    "<h1>CodeQL Report</h1>\n"
    + "<text>"
    + types_count
    + "</text>\n<br></br>\n"
    + df.to_html(justify="center")
)

with open("codeql/build/types_count.md", "w") as file:
    file.write(types_count)

with open("codeql/build/codeql.html", "w") as file:
    file.write(html)
