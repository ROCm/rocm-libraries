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

from txt2img import StableDiffusionMGX, get_args
import gradio as gr
import sys


class PrintWrapper(object):
    def __init__(self, org_handle):
        self.org_handle = org_handle
        self.log = ""

        def wrapper_write(x):
            self.log += x
            return org_handle.write(x)

        self.wrapper_write = wrapper_write

    def __getattr__(self, attr):
        return self.wrapper_write if attr == 'write' else getattr(
            self.org_handle, attr)

    def get_log(self):
        return self.log


def main():
    args = get_args()
    # Note: This will load the models, which can take several minutes
    sd = StableDiffusionMGX(args.pipeline_type, args.onnx_model_path,
                            args.compiled_model_path, args.use_refiner,
                            args.refiner_onnx_model_path,
                            args.refiner_compiled_model_path, args.fp16,
                            args.force_compile, args.exhaustive_tune)
    sd.warmup(5)

    def gr_wrapper(prompt, negative_prompt, steps, seed, scale, refiner_steps,
                   aesthetic_score, negative_aesthetic_score):
        img = None
        try:
            oldStdout, oldStderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = PrintWrapper(sys.stdout), PrintWrapper(
                sys.stderr)
            result = sd.run(
                str(prompt),
                str(negative_prompt),
                int(steps),
                int(seed),
                float(scale),
                int(refiner_steps),
                float(aesthetic_score),
                float(negative_aesthetic_score),
                args.verbose,
            )
            img = StableDiffusionMGX.convert_to_rgb_image(result)
            sd.print_summary(steps)
        finally:
            log = ''.join([sys.stdout.get_log(), sys.stderr.get_log()])
            sys.stdout, sys.stderr = oldStdout, oldStderr
        return img, log

    demo = gr.Interface(gr_wrapper, [
        gr.Textbox(value=args.prompt, label="Prompt"),
        gr.Textbox(value=args.negative_prompt,
                   label="Negative prompt (Optional)"),
        gr.Slider(1, 100, step=1, value=args.steps, label="Number of steps"),
        gr.Textbox(value=args.seed, label="Random seed"),
        gr.Slider(1, 20, step=0.1, value=args.scale, label="Guidance scale"),
        gr.Slider(0,
                  100,
                  step=1,
                  value=args.refiner_steps,
                  label="Number of refiner steps. (Use 0 to skip it)",
                  visible=args.use_refiner),
        gr.Slider(1,
                  20,
                  step=0.1,
                  value=args.refiner_aesthetic_score,
                  label="Aesthetic score (Refiner)",
                  visible=args.use_refiner),
        gr.Slider(1,
                  20,
                  step=0.1,
                  value=args.refiner_negative_aesthetic_score,
                  label="Negative Aesthetic score (Refiner)",
                  visible=args.use_refiner),
    ], [
        "image",
        gr.Textbox(placeholder="Output log of the run", label="Output log")
    ])
    demo.launch()
    sd.cleanup()


if __name__ == "__main__":
    main()
