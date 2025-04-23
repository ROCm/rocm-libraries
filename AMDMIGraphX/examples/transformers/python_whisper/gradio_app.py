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

from asr import WhisperMGX
import gradio as gr
import os


def main():
    # Note: This will load the models, which can take several minutes
    w = WhisperMGX()

    def gr_wrapper(audio):
        data, fr = WhisperMGX.load_audio_from_file(audio)
        input_features = w.get_input_features_from_sample(data, fr)
        return w.generate(input_features)

    examples = [
        os.path.join(os.path.dirname(__file__), "audio/sample1.flac"),
        os.path.join(os.path.dirname(__file__), "audio/sample2.flac"),
    ]
    # skip if there is no file
    examples = [e for e in examples if os.path.isfile(e)]

    demo = gr.Interface(
        gr_wrapper,
        gr.Audio(sources=["upload", "microphone"], type="filepath"),
        "text",
        examples=examples,
    )
    demo.launch()


if __name__ == "__main__":
    main()
