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
#####################################################################################

from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import WhisperOnnxConfig
from transformers import AutoConfig

from optimum.exporters.onnx.base import ConfigBehavior
from typing import Dict


class CustomWhisperOnnxConfig(WhisperOnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}

        if self._behavior is ConfigBehavior.ENCODER:
            common_inputs["input_features"] = {
                0: "batch_size",
                1: "feature_size",
                2: "encoder_sequence_length"
            }

        if self._behavior is ConfigBehavior.DECODER:
            common_inputs["decoder_input_ids"] = {
                0: "batch_size",
                1: "decoder_sequence_length"
            }
            common_inputs["decoder_attention_mask"] = {
                0: "batch_size",
                1: "decoder_sequence_length"
            }
            common_inputs["encoder_outputs"] = {
                0: "batch_size",
                1: "encoder_sequence_length"
            }

        return common_inputs

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {
                "decoder_input_ids": "input_ids",
                "decoder_attention_mask": "attention_mask",
                "encoder_outputs": "encoder_hidden_states",
            }
        return {}


def export():
    model_id = "openai/whisper-tiny.en"
    config = AutoConfig.from_pretrained(model_id)

    custom_whisper_onnx_config = CustomWhisperOnnxConfig(
        config=config,
        task="automatic-speech-recognition",
    )

    encoder_config = custom_whisper_onnx_config.with_behavior("encoder")
    decoder_config = custom_whisper_onnx_config.with_behavior("decoder",
                                                              use_past=False)

    custom_onnx_configs = {
        "encoder_model": encoder_config,
        "decoder_model": decoder_config,
    }

    output = "models/whisper-tiny.en_modified"
    main_export(model_id,
                output=output,
                no_post_process=True,
                do_validation=False,
                custom_onnx_configs=custom_onnx_configs)

    print(f"Done. Check {output}")


if __name__ == "__main__":
    export()
