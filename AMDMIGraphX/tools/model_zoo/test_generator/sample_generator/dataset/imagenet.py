#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#####################################################################################
from .base import BaseDataset
from datasets import load_dataset


class ImageNet2012Val(BaseDataset):
    @property
    def url(self):
        return "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

    @property
    def split(self):
        return "val"

    @staticmethod
    def name():
        return "imagenet-2012-val"

    def __iter__(self):
        print(f"Load dataset from {self.url}")
        self.dataset = iter(
            load_dataset("webdataset",
                         data_files={self.split: self.url},
                         split=self.split,
                         streaming=True))
        return self.dataset

    def __next__(self):
        return next(self.dataset)

    def transform(self, inputs, data, prepocess_fn):
        result = prepocess_fn(data["jpeg"])
        inputs, keys = sorted(inputs), sorted(list(result.keys()))
        assert inputs == keys, f"{inputs = } == {keys = }"
        # The result should be a simple dict, the preproc returns a wrapped class, dict() will remove it
        return dict(result)
