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
from .base import BaseDataset, ValidationDatasetHFIteratorMixin
from datasets import load_dataset


class SQuADTransformMixin(object):
    def transform(self, inputs, data, prepocess_fn):
        result = prepocess_fn(data["question"],
                              data["context"],
                              max_length=384)
        inputs, keys = sorted(inputs), sorted(list(result.keys()))
        assert inputs == keys, f"{inputs = } == {keys = }"
        # The result should be a simple dict, the preproc returns a wrapped class, dict() will remove it
        return dict(result)


class SQuADv1_1(SQuADTransformMixin, BaseDataset):
    @property
    def url(self):
        return "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"

    @property
    def split(self):
        return "validation"

    @staticmethod
    def name():
        return "squad-v1.1"

    def __iter__(self):
        print(f"Load dataset from {self.url}")
        self.dataset = ({
            "context": paragraph["context"],
            "question": qas["question"]
        } for data in load_dataset("json",
                                   data_files={"val": self.url},
                                   split="val",
                                   field="data",
                                   streaming=True)
                        for paragraph in data["paragraphs"]
                        for qas in paragraph["qas"])
        return self.dataset

    def __next__(self):
        return next(self.dataset)


class SQuAD_HF(ValidationDatasetHFIteratorMixin, SQuADTransformMixin,
               BaseDataset):
    @property
    def url(self):
        return "squad"

    @property
    def split(self):
        return "validation"

    @staticmethod
    def name():
        return "squad-hf"
