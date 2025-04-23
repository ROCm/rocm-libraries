#  The MIT License (MIT)
#
#  Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the 'Software'), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

from argparse import ArgumentParser
from diffusers import AutoencoderKL
import os
import torch


def argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="models/sdxl-1.0-base/vae_decoder_fp16_fix/model.onnx",
        help=
        "Path to save the onnx model. Use it to override the default models/<sdxl*> path."
    )
    return parser.parse_args()


class VAEDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent_sample):
        return self.vae.decode(latent_sample)


def export_vae_fp16(output_path):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    vae.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(VAEDecoder(vae),
                      torch.randn(1, 4, 128, 128),
                      output_path,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['latent_sample'])


if __name__ == "__main__":
    args = argparser()
    export_vae_fp16(**vars(args))
