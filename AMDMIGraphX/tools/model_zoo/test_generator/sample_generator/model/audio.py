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
from .base import BaseModel, DecoderModel, OptimumHFModelDownloadMixin
import numpy as np
from transformers import AutoFeatureExtractor


class AutoFeatureExtractorHFMixin(object):

    _processor = None

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoFeatureExtractor.from_pretrained(
                self.model_id)
        return self._processor

    def preprocess(self, audio_data, sampling_rate):
        return self.processor(audio_data,
                              sampling_rate=sampling_rate,
                              return_tensors="np")


class Wav2Vec2_base_960h(OptimumHFModelDownloadMixin,
                         AutoFeatureExtractorHFMixin, BaseModel):
    @property
    def model_id(self):
        return "facebook/wav2vec2-base-960h"

    @staticmethod
    def name():
        return "wav2vec2-base-960h"


class WhisperSmallEn(OptimumHFModelDownloadMixin, AutoFeatureExtractorHFMixin,
                     DecoderModel):
    def __init__(self):
        # Whisper specific part
        # TODO get these from config
        self.eos_token_id = 50256  # "<|endoftext|>"
        decoder_start_token_id = 50257  # <|startoftranscript|>
        notimestamps = 50362  # <|notimestamps|>
        sot = [decoder_start_token_id, notimestamps]
        max_length = 448
        self.initial_input_ids = np.array(
            [sot + [self.eos_token_id] * (max_length - len(sot))])

    @property
    def model_id(self):
        return "openai/whisper-small.en"

    @staticmethod
    def name():
        return "whisper-small-en"

    def preprocess(self, *args, **kwargs):
        # result only contains encoder data, extend it with decoder
        result = super().preprocess(*args, **kwargs)
        result["decoder_input_ids"] = np.copy(self.initial_input_ids)
        return result

    def decode_step(self, input_map, output_map):
        timestep = np.argmax(
            input_map["decoder_input_ids"][0] == self.eos_token_id)
        new_token = np.argmax(output_map["logits"][0][timestep - 1])
        input_map["decoder_input_ids"][0][timestep] = new_token
        return new_token == self.eos_token_id
