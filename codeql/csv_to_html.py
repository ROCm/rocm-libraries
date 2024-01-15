#!/usr/bin/env python3

import pandas as pd
from collections import Counter

try:
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
