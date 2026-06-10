/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Converts RGB color values to HSV colorspace
__device__ void rgb_to_hsv_hip(float *pixelR, float *pixelG, float *pixelB, float &hue, float &sat, float &val, float &add)
{
    // Find maximum and minimum values among RGB components
    float cmax = fmaxf(fmaxf(*pixelR, *pixelG), *pixelB);
    float cmin = fminf(fminf(*pixelR, *pixelG), *pixelB);
    float delta = cmax - cmin;

    // Initialize HSV values
    hue = 0.0f;
    sat = 0.0f;
    val = cmax;

    // Calculate saturation and hue if delta is not zero and max value is not zero
    if ((delta != 0) && (cmax != 0))
    {
        sat = delta / cmax;
        // Calculate hue based on which RGB component is maximum
        if (cmax == *pixelR)
        {
            hue = (*pixelG - *pixelB) / delta;
            add = 0.0f;
        }
        else if (cmax == *pixelG)
        {
            hue = (*pixelB - *pixelR) / delta;
            add = 2.0f;
        }
        else
        {
            hue = (*pixelR - *pixelG) / delta;
            add = 4.0f;
        }
    }
}

// Converts HSV color values back to RGB colorspace
__device__ void hsv_to_rgb_hip(float &hue, float &sat, float &val, float *pixelR, float *pixelG, float *pixelB)
{
    // Calculate intermediate values for RGB conversion
    float hueFraction = hue - floor(hue);
    float p = val * (1.0f - sat);
    float q = val * (1.0f - (sat * hueFraction));
    float t = val * (1.0f - (sat * (1.0f - hueFraction)));

    // Assign RGB values based on hue section (0-5)
    switch ((int)hue)
    {
        case 0: *pixelR = val; *pixelG = t;   *pixelB = p;   break;
        case 1: *pixelR = q;   *pixelG = val; *pixelB = p;   break;
        case 2: *pixelR = p;   *pixelG = val; *pixelB = t;   break;
        case 3: *pixelR = p;   *pixelG = q;   *pixelB = val; break;
        case 4: *pixelR = t;   *pixelG = p;   *pixelB = val; break;
        case 5: *pixelR = val; *pixelG = p;   *pixelB = q;   break;
    }
}
