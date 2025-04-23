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
from .base import BaseModel, OptimumHFModelDownloadMixin
from ..utils import get_imagenet_classes
from transformers import AutoProcessor


class AutoProcessorHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._processor

    def preprocess(self, images, text, max_length=384):
        return self.processor(images=images,
                              text=text,
                              max_length=max_length,
                              padding='max_length',
                              return_tensors="np")


class ClipVitLargePatch14(OptimumHFModelDownloadMixin, AutoProcessorHFMixin,
                          BaseModel):
    def __init__(self):
        import random
        random.seed(42)
        # cache a few labels, the full 1000 label is overkill
        self.imagenet_labels = random.sample(get_imagenet_classes(), 10)

    @property
    def model_id(self):
        return "openai/clip-vit-large-patch14"

    @staticmethod
    def name():
        return "clip-vit-large-patch14"

    def preprocess(self, *args, **kwargs):
        # extend image with imagenet labels
        new_args, new_kwargs = list(args), kwargs
        new_args.append(self.imagenet_labels)
        new_kwargs["max_length"] = 77
        result = super().preprocess(*new_args, **new_kwargs)
        return result
