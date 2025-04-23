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

from argparse import ArgumentParser
from transformers import WhisperProcessor
from datasets import load_dataset
from pydub import AudioSegment

import migraphx as mgx
import os
import numpy as np
import sys
import time
from functools import wraps


# measurement helper
def measure(fn):
    @wraps(fn)
    def measure_ms(*args, **kwargs):
        start_time = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        end_time = time.perf_counter_ns()
        print(
            f"Elapsed time for {fn.__name__}: {(end_time - start_time) * 1e-6:.4f} ms\n"
        )
        return result

    return measure_ms


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-a",
        "--audio",
        type=str,
        help="Path to audio file. Default: HF test dataset",
    )

    parser.add_argument(
        "-l",
        "--log-process",
        action="store_true",
        help="Print the current state of transcribing.",
    )

    return parser.parse_args()


class WhisperMGX():
    def __init__(self):
        model_id = "openai/whisper-tiny.en"
        print(f"Using {model_id}")

        print("Creating Whisper processor")
        self.processor = WhisperProcessor.from_pretrained(model_id)

        self.decoder_start_token_id = 50257  # <|startoftranscript|>
        self.eos_token_id = 50256  # "<|endoftext|>"
        self.notimestamps = 50362  # <|notimestamps|>
        self.max_length = 448
        self.sot = [self.decoder_start_token_id, self.notimestamps]

        print("Load models...")
        self.encoder_model = WhisperMGX.load_mgx_model(
            "encoder", {"input_features": [1, 80, 3000]})
        self.decoder_model = WhisperMGX.load_mgx_model(
            "decoder", {
                "input_ids": [1, self.max_length],
                "attention_mask": [1, self.max_length],
                "encoder_hidden_states": [1, 1500, 384]
            })

    @staticmethod
    @measure
    def load_audio_from_file(filepath):
        audio = AudioSegment.from_file(filepath)
        # Only 16k is supported
        audio = audio.set_frame_rate(16000)
        data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        data /= np.max(np.abs(data))
        return data, audio.frame_rate

    @staticmethod
    @measure
    def load_mgx_model(name, shapes):
        file = f"models/whisper-tiny.en_modified/{name}_model"
        print(f"Loading {name} model from {file}")
        if os.path.isfile(f"{file}.mxr"):
            print("Found mxr, loading it...")
            model = mgx.load(f"{file}.mxr", format="msgpack")
        elif os.path.isfile(f"{file}.onnx"):
            print("Parsing from onnx file...")
            model = mgx.parse_onnx(f"{file}.onnx", map_input_dims=shapes)
            model.compile(mgx.get_target("gpu"))
            print(f"Saving {name} model to mxr file...")
            mgx.save(model, f"{file}.mxr", format="msgpack")
        else:
            print(f"No {name} model found. Please download it and re-try.")
            sys.exit(1)
        return model

    @property
    def initial_decoder_inputs(self):
        input_ids = np.array([
            self.sot + [self.eos_token_id] * (self.max_length - len(self.sot))
        ])
        # 0 masked | 1 un-masked
        attention_mask = np.array([[1] * len(self.sot) + [0] *
                                   (self.max_length - len(self.sot))])
        return (input_ids, attention_mask)

    @measure
    def get_input_features_from_sample(self, sample_data, sampling_rate):
        return self.processor(sample_data,
                              sampling_rate=sampling_rate,
                              return_tensors="np").input_features

    @measure
    def encode_features(self, input_features):
        return np.array(
            self.encoder_model.run(
                {"input_features": input_features.astype(np.float32)})[0])

    def decode_step(self, input_ids, attention_mask, hidden_states):
        return np.array(
            self.decoder_model.run({
                "input_ids":
                input_ids.astype(np.int64),
                "attention_mask":
                attention_mask.astype(np.int64),
                "encoder_hidden_states":
                hidden_states.astype(np.float32)
            })[0])

    @measure
    def generate(self, input_features, log_process=False):
        hidden_states = self.encode_features(input_features)
        input_ids, attention_mask = self.initial_decoder_inputs
        for timestep in range(len(self.sot) - 1, self.max_length):
            # get logits for the current timestep
            logits = self.decode_step(input_ids, attention_mask, hidden_states)
            # greedily get the highest probable token
            new_token = np.argmax(logits[0][timestep])

            # add it to the tokens and unmask it
            input_ids[0][timestep + 1] = new_token
            attention_mask[0][timestep + 1] = 1

            if log_process:
                print("Transcribing: " + ''.join(
                    self.processor.decode(input_ids[0][:timestep + 1],
                                          skip_special_tokens=True)),
                      end='\r')

            if new_token == self.eos_token_id:
                break

        if log_process:
            print(flush=True)

        return ''.join(
            self.processor.decode(input_ids[0][:timestep + 1],
                                  skip_special_tokens=True))


if __name__ == "__main__":
    args = get_args()

    if args.audio:
        data, fr = WhisperMGX.load_audio_from_file(args.audio)
        ds = [{"audio": {"array": data, "sampling_rate": fr}}]
    else:
        # load dummy dataset and read audio files
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                          "clean",
                          split="validation")

    w = WhisperMGX()

    for idx, data in enumerate(ds):
        print(f"#{idx+1}/{len(ds)} Sample...")
        sample = data["audio"]
        input_features = w.get_input_features_from_sample(
            sample["array"], sample["sampling_rate"])
        result = w.generate(input_features, log_process=args.log_process)
        print(f"Result: {result}")
