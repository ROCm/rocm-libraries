# Minimal `pathlib.Path` shim for MicroPython.
#
# Subclasses str (so `open(path)` works), but WITHOUT overriding __new__ —
# MicroPython's str has no __new__ and only supports single-arg construction.
# ck_dsl only uses single-arg Path(x) + `/` + .parent/.name/read_*/write_*/exists,
# so every derived path is built as a single joined string.


class Path(str):
    def __truediv__(self, other):
        base = self.rstrip("/")
        return Path(base + "/" + str(other))

    def __rtruediv__(self, other):
        return Path(str(other) + "/" + self)

    @property
    def parent(self):
        s = self.rstrip("/")
        i = s.rfind("/")
        if i < 0:
            return Path(".")
        if i == 0:
            return Path("/")
        return Path(s[:i])

    @property
    def name(self):
        s = self.rstrip("/")
        i = s.rfind("/")
        return s[i + 1 :] if i >= 0 else s

    @property
    def stem(self):
        n = self.name
        j = n.rfind(".")
        return n[:j] if j > 0 else n

    @property
    def suffix(self):
        n = self.name
        j = n.rfind(".")
        return n[j:] if j > 0 else ""

    def read_text(self, encoding=None):
        with open("" + self) as f:
            return f.read()

    def read_bytes(self):
        with open("" + self, "rb") as f:
            return f.read()

    def write_text(self, data, encoding=None):
        with open("" + self, "w") as f:
            return f.write(data)

    def write_bytes(self, data):
        with open("" + self, "wb") as f:
            return f.write(data)

    def exists(self):
        try:
            f = open("" + self)
            f.close()
            return True
        except OSError:
            return False
