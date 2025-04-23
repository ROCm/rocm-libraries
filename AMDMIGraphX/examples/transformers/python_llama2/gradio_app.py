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

import gradio as gr
from txtgen import Llama2MGX


def main():
    # Note: This will load the models, which can take several minutes
    llama = Llama2MGX(1024)

    def gr_wrapper(prompt):
        if prompt == "":
            return "Please provide a prompt."

        input_ids = llama.tokenize(prompt)
        result = llama.generate(input_ids)

        # trim input prompt from result
        result = result[len(prompt) + 2:]

        return result

    with gr.Blocks() as demo:
        gr.Markdown(
            "Start typing below and then click **Run** to see the output.")
        inp = gr.Textbox(placeholder="Type something here...",
                         label="Input prompt")
        btn = gr.Button("Run!")
        out = gr.Textbox(placeholder="The result will be displayed here",
                         label="Response")

        btn.click(fn=gr_wrapper, inputs=inp, outputs=out)

    demo.launch()


if __name__ == "__main__":
    main()
