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
from .base import BaseDataset
from collections import namedtuple

Prompt = namedtuple('Prompt', ['name', 'prompt', 'negative_prompt'])

# source: https://huggingface.co/spaces/google/sdxl/blob/main/app.py#L19-L70
style_list = [
    Prompt(
        name="Cinematic",
        prompt=
        "cinematic still, emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        negative_prompt=
        "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    ),
    Prompt(
        name="Photographic",
        prompt=
        "cinematic photo, 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        negative_prompt=
        "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    ),
    Prompt(
        name="Anime",
        prompt=
        "anime artwork, anime style, key visual, vibrant, studio anime,  highly detailed",
        negative_prompt=
        "photo, deformed, black and white, realism, disfigured, low contrast",
    ),
    Prompt(
        name="Manga",
        prompt=
        "manga style, vibrant, high-energy, detailed, iconic, Japanese comic style",
        negative_prompt=
        "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    ),
    Prompt(
        name="Digital Art",
        prompt=
        "concept art, digital artwork, illustrative, painterly, matte painting, highly detailed",
        negative_prompt="photo, photorealistic, realism, ugly",
    ),
    Prompt(
        name="Pixel art",
        prompt="pixel-art, low-res, blocky, pixel art style, 8-bit graphics",
        negative_prompt=
        "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    ),
    Prompt(
        name="Fantasy art",
        prompt=
        "ethereal fantasy concept art, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        negative_prompt=
        "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    ),
    Prompt(
        name="Neonpunk",
        prompt=
        "neonpunk style, cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        negative_prompt=
        "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    ),
    Prompt(
        name="3D Model",
        prompt=
        "professional 3d model, octane render, highly detailed, volumetric, dramatic lighting",
        negative_prompt="ugly, deformed, noisy, low poly, blurry, painting",
    ),
]


class StylePrompts(BaseDataset):
    @property
    def url(self):
        return ""

    @property
    def split(self):
        return ""

    @staticmethod
    def name():
        return "style-prompts"

    def __iter__(self):
        print(f"Load dataset for {self.name()}")
        self.dataset = iter(style_list)
        return self.dataset

    def __next__(self):
        return next(self.dataset)

    def transform(self, inputs, data, prepocess_fn):
        result = prepocess_fn(data.prompt, data.negative_prompt)
        inputs, keys = sorted(inputs), sorted(list(result.keys()))
        assert inputs == keys, f"{inputs = } == {keys = }"
        # The result should be a simple dict, the preproc returns a wrapped class, dict() will remove it
        return dict(result)
