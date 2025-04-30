################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from .Common import CHeader 

import itertools
import os

class Namespace:
    def __init__(self, parent, name=None):
        self.parent = parent
        self.name = name

    def __enter__(self):
        if self.name is None:
            self.parent.write("namespace {")
        else:
            self.parent.write("namespace {} {{".format(self.name))
        self.parent.indent()
        return self

    def __exit__(self, *args, **kwargs):
        self.parent.dedent()
        if self.name is None:
            self.parent.write("} // anonymous namespace")
        else:
            self.parent.write("}} // namespace {}".format(self.name))

class Indent:
    def __init__(self, parent):
        self.parent = parent

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.parent.dedent()

class EmbeddedDataFile:
    def __init__(self, filename, file=None, indent_spaces=4):
        self.filename = filename

        self._indent_spaces = indent_spaces
        self._indent_levels = [0]
        self._open_blocks = []

        self.file = file
        if self.file is None:
            self.file = open(filename, 'w')

        self.write_header()

    def __enter__(self):
        self.file.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        while len(self._open_blocks) > 0:
            b = self._open_blocks.pop()
            b.__exit__(*args, **kwargs)

        self.write_footer()

        self.file.__exit__(*args, **kwargs)

    def namespace(self, name=None):
        ns = Namespace(self, name).__enter__()
        self._open_blocks.append(ns)

    def end_namespace(self, name=None):
        ns = self._open_blocks.pop()
        if not isinstance(ns, Namespace):
            raise RuntimeError("Mismatched block types: expected Namespace, found {}".format(ns.__class__))
        if name != ns.name:
            raise RuntimeError("Mismatched namespace open/close: expected {}, found {}".format(name, ns.name))

        ns.__exit__(None, None, None)

    def write_header(self):
        self.write(CHeader)

        self.write(self.includes)

        self.namespace("Tensile")

    @property
    def include_guard(self):
        return "#pragma once"

    @property
    def includes(self):
        return """
        #include <Tensile/EmbeddedData.hpp>

        #include <Tensile/Contractions.hpp>
        #include <Tensile/Tensile.hpp>
    """

    def get_lines(self, item):
        if isinstance(item, str):
            return item.split('\n')

        if hasattr(item, '__iter__'):
            return item

        return str(item).split('\n')

    def format(self, item):
        out_lines = []
        for line in self.get_lines(item):
            out_lines.append(self.apply_indent(line))

        return '\n'.join(out_lines) + '\n'

    @property
    def indent_level(self):
        if len(self._indent_levels) == 0:
            return 0

        return self._indent_levels[-1]

    def apply_indent(self, line=None):
        if line is None:
            return ' ' * self.indent_level

        line = line.strip()

        if line.startswith('#'):
            return line

        return (' ' * self.indent_level) + line

    def indent(self, spaces=None):
        if spaces is None:
            spaces = self._indent_spaces
        self._indent_levels.append(spaces + self.indent_level)
        return Indent(self)

    def dedent(self):
        self._indent_levels.pop()

    def write(self, *items):
        for item in items:
            self.file.write(self.format(item))

    def comment(self, text):
        self.write(['// ' + line for line in text.split('\n')])

    def write_footer(self):
        self.write('')

    def embed_data(self, assocType, data, nullTerminated=False, comment=None, key=None):
        if nullTerminated:
            empty = False
            data = itertools.chain(data, [0])
        else:
            empty = len(data) == 0
            data = iter(data)

        with Namespace(self):
            if comment is not None:
                self.comment(comment)
            if empty:
                if key is None:
                    self.write("EmbedData<{0}> TENSILE_EMBED_SYMBOL_NAME{{}};".format(assocType))
                else:
                    self.write('EmbedData<{0}> TENSILE_EMBED_SYMBOL_NAME("{1}", {{}});'.format(assocType, key))
                return

            hex_format = '{:#04x}'

            if key is None:
                self.write("EmbedData<{0}> TENSILE_EMBED_SYMBOL_NAME({{".format(assocType))
            else:
                self.write('EmbedData<{0}> TENSILE_EMBED_SYMBOL_NAME("{1}", {{'.format(assocType, key))
            with self.indent():
                line = hex_format.format(next(data))
                for byteIdx, byte in enumerate(data):
                    if byteIdx % 16 == 15:
                        self.write(line + ",")
                        line = hex_format.format(byte)
                    else:
                        line += ', ' + hex_format.format(byte)

                self.write(line + '});')

    def embed_file(self, assocType, filename, nullTerminated=False, key=None):
        with open(filename, 'rb') as f:
          byteArray = bytearray(f.read())
        self.embed_data(assocType, byteArray, nullTerminated, os.path.basename(filename), key)

