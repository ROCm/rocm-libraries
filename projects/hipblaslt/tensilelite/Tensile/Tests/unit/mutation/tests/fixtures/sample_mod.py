"""Fixture source for mutmut-results-adapter selftest AST resolution.

Line numbers matter: the survivors fixture references specific lines here.
Includes similar top-level names (parse / parse_all) and a class method named
`parse` to prove {module,function} grouping and collision-free test_file naming.
"""


def parse(text):
    cleaned = text.strip()  # line 11
    if not cleaned:  # line 12
        return None  # line 13
    return cleaned.split(",")  # line 14


def parse_all(items):
    out = []  # line 18
    for it in items:  # line 19
        out.append(parse(it))  # line 20
    return out  # line 21


class Reader:
    def parse(self, blob):
        n = len(blob)  # line 26
        return n * 2  # line 27

    def read(self, path):
        with open(path) as fh:  # line 30
            return fh.read()  # line 31


TOP_LEVEL_CONST = 42  # line 34 (module-level, no enclosing function)


import functools  # line 37


@functools.lru_cache(maxsize=8)  # line 40 (decorator line -> belongs to cached_double)
def cached_double(x):  # line 41
    return x * 2  # line 42
