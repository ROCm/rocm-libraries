################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import globalParameters

import os

class ReplacementKernels:
    def __init__(self, dirpath, codeObjectVersion):
        self.dirpath = dirpath
        self.codeObjectVersion = codeObjectVersion
        self._cache = None

    @property
    def marker(self):
        return '.amdhsa_kernel'

    def getKernelName(self, filename):
        marker = self.marker

        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith(marker):
                        return line[len(marker):].strip()
        except Exception:
            print(filename)
            raise
        raise RuntimeError("Could not parse kernel name from {}".format(filename))

    @property
    def cache(self):
        if not self._cache:
            self._cache = self.generateCache()
        return self._cache

    def populateCache(self):
        _ = self.cache

    def generateCache(self):
        cache = {}

        for filename in os.listdir(self.dirpath):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.dirpath, filename)
                kernelName = self.getKernelName(filepath)
                if kernelName in cache:
                    raise RuntimeError("Duplicate replacement kernels.  Kernel name: {}, file names: {}, {}".format(kernelName, cache[kernelName], filepath))
                cache[kernelName] = filepath

        return cache

    def get(self, kernelName):
        if kernelName in self.cache:
            return self.cache[kernelName]

        return None

    @classmethod
    def Directory(cls):
        scriptDir = os.path.dirname(os.path.realpath(__file__))
        dirName = 'ReplacementKernels-cov3'

        return os.path.join(scriptDir, dirName)

    _instance = None
    @classmethod
    def Instance(cls):
        if cls._instance:
            return cls._instance
        return cls(cls.Directory(), globalParameters["CodeObjectVersion"])

    @classmethod
    def Get(cls, kernelName):
        return cls.Instance().get(kernelName)
