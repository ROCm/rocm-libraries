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
from .base import BaseModel, SingleModelDownloadMixin, OptimumHFModelDownloadMixin
from .preprocess import process_image

from transformers import AutoImageProcessor
import timm


class AutoImageProcessorHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        return self._processor

    def preprocess(self, image_data):
        return self.processor(image_data, return_tensors="np")


class ResNet50_v1(SingleModelDownloadMixin, BaseModel):
    @property
    def model_id(self):
        return "https://zenodo.org/record/2592612/files/resnet50_v1.onnx"

    @staticmethod
    def name():
        return "resnet50_v1"

    def preprocess(self, image_data):
        IMAGENET_MEANS = [123.68, 116.78, 103.94]  # RGB
        return {
            "input_tensor:0": process_image(image_data, means=IMAGENET_MEANS)
        }


class ResNet50_v1_5(OptimumHFModelDownloadMixin, AutoImageProcessorHFMixin,
                    BaseModel):
    @property
    def model_id(self):
        return "microsoft/resnet-50"

    @staticmethod
    def name():
        return "resnet50_v1.5"


class VitBasePatch16_224(OptimumHFModelDownloadMixin,
                         AutoImageProcessorHFMixin, BaseModel):
    @property
    def model_id(self):
        return "google/vit-base-patch16-224"

    @staticmethod
    def name():
        return "vit-base-patch16-224"


class TIMM_MobileNetv3_large(OptimumHFModelDownloadMixin, BaseModel):
    def __init__(self):
        data_config = timm.data.resolve_model_data_config(
            timm.create_model(self.model_id, pretrained=True))
        self.processor = timm.data.create_transform(**data_config,
                                                    is_training=False)

    @property
    def model_id(self):
        return "timm/mobilenetv3_large_100.ra_in1k"

    @staticmethod
    def name():
        return "timm-mobilenetv3-large"

    def preprocess(self, image_data):
        return {
            "pixel_values":
            self.processor(image_data).unsqueeze(0).cpu().detach().numpy()
        }


# TODO enable it when BiT is supported by optimum
class Bit50(OptimumHFModelDownloadMixin, AutoImageProcessorHFMixin, BaseModel):
    @property
    def model_id(self):
        return "google/bit-50"

    @staticmethod
    def name():
        return "bit-50"
