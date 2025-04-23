#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
import numpy as np

params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (6, 5),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'x-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large'
}
pylab.rcParams.update(params)


#-----------------------------------------------------------
def show_n_images(imgs, titles=None, enlarge=20, cmap='jet'):

    plt.set_cmap(cmap)
    n = len(imgs)
    gs1 = gridspec.GridSpec(1, n)

    fig1 = plt.figure()
    # create a figure with the default size
    fig1.set_size_inches(enlarge, 2 * enlarge)

    for i in range(n):

        ax1 = fig1.add_subplot(gs1[i])

        ax1.imshow(imgs[i], interpolation='none')
        if (titles is not None):
            ax1.set_title(titles[i])
        ax1.set_ylim(ax1.get_ylim()[::-1])

    plt.show()


#--------------------------------------------------------------
from skimage import color, img_as_float
from skimage.exposure import adjust_gamma


# Creates an image of original brain with segmentation overlay
def show_label_on_image(test_img, test_lbl):

    label_im = test_lbl

    ones = np.argwhere(label_im == 1)
    twos = np.argwhere(label_im == 2)
    threes = np.argwhere(label_im == 3)
    fours = np.argwhere(label_im == 4)

    gray_img = img_as_float(test_img / test_img.max())

    # adjust gamma of image
    # print(color.gray2rgb(gray_img))
    image = adjust_gamma(np.abs(color.gray2rgb(gray_img)), 0.45)
    #sliced_image = image.copy()

    green_multiplier = [0.35, 0.75, 0.25]
    blue_multiplier = [0, 0.5, 1.]  #[0,0.25,0.9]
    yellow_multiplier = [1, 1, 0.25]
    brown_miltiplier = [40. / 255, 26. / 255, 13. / 255]

    # change colors of segmented classes
    for i in range(len(ones)):
        image[ones[i][0]][ones[i][1]] = blue_multiplier
    for i in range(len(twos)):
        image[twos[i][0]][twos[i][1]] = yellow_multiplier
    for i in range(len(threes)):
        image[threes[i][0]][threes[i][1]] = brown_miltiplier  #blue_multiplier
    for i in range(len(fours)):
        image[fours[i][0]][fours[i][1]] = green_multiplier  #yellow_multiplier

    return image


#-------------------------------------------------------------------------------------
def show_label_on_image4(test_img, label_im):

    alpha = 0.8

    img = img_as_float(test_img / test_img.max())
    rows, cols = img.shape

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    green_multiplier = [0.35, 0.75, 0.25]
    blue_multiplier = [0, 0.25, 0.9]
    yellow_multiplier = [1, 1, 0.25]
    brown_miltiplier = [40. / 255, 26. / 255, 13. / 255]

    color_mask[label_im == 1] = blue_multiplier  #[1, 0, 0]  # Red block
    color_mask[label_im == 2] = yellow_multiplier  #[0, 1, 0] # Green block
    color_mask[label_im == 3] = brown_miltiplier  #[0, 0, 1] # Blue block
    color_mask[label_im == 4] = green_multiplier  #[0, 1, 1] # Blue block

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    return img_masked


#------------------------------------------------------------------------------
