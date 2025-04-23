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

from txt2img import StableDiffusionMGX, get_args, image_to_tensor
import gradio as gr


def main():
    args = get_args()
    # Note: This will load the models, which can take several minutes
    sd = StableDiffusionMGX(args.onnx_model_path, args.compiled_model_path,
                            args.fp16, args.force_compile,
                            args.exhaustive_tune)
    sd.warmup(5)

    def gr_wrapper(prompt, negative_prompt, image, steps, seed, scale,
                   conditioning_scale):
        result = sd.run(str(prompt), str(negative_prompt),
                        image_to_tensor(image), int(steps), int(seed),
                        float(scale), float(conditioning_scale))
        return StableDiffusionMGX.convert_to_rgb_image(result)

    demo = gr.Interface(
        gr_wrapper,
        [
            gr.Textbox(value=args.prompt, label="Prompt"),
            gr.Textbox(value=args.negative_prompt,
                       label="Negative prompt (Optional)"),
            gr.Image(label="Canny Control Image",
                     value=args.control_image,
                     type='filepath'),
            gr.Slider(
                1, 100, step=1, value=args.steps, label="Number of steps"),
            gr.Textbox(value=args.seed, label="Random seed"),
            gr.Slider(
                1, 20, step=0.1, value=args.scale, label="Guidance scale"),
            gr.Slider(0,
                      1,
                      step=0.1,
                      value=args.conditioning_scale,
                      label="Conditioning scale"),
        ],
        "image",
    )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
