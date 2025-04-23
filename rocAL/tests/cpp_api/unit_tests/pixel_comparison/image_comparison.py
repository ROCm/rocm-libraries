# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

from PIL import Image
import os
import sys
import datetime
import logging


def compare_pixels(img1, img2, aug_name, width, height, image_offset=0):
    pixel_difference = [0, 0, 0, 0, 0, 0]
    if "rgb" in aug_name:
        pixels1 = img1.load()
        pixels2 = img2.load()
        channel = 3
    else:
        pixels1 = img1.convert("L").load()
        pixels2 = img2.convert("L").load()
        channel = 1
    total_valid_pixel_count = width * height * channel
    for wt in range(width):
        for ht in range(height):
            ht = ht + image_offset
            if pixels1[wt, ht] != pixels2[wt, ht]:
                if channel == 1:
                    diff_val = abs(pixels1[wt, ht] - pixels2[wt, ht])
                    diff_val = min(diff_val, 5)
                    pixel_difference[diff_val] += 1
                else:
                    for ch in range(channel):
                        diff_val = abs(
                            pixels1[wt, ht][ch] - pixels2[wt, ht][ch])
                        diff_val = min(diff_val, 5)
                        pixel_difference[diff_val] += 1
            else:
                pixel_difference[0] += channel
    return pixel_difference, total_valid_pixel_count


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
    handlers = [
        logging.FileHandler("./rocal_unittest_log_file_" + timestamp + ".log"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    if len(sys.argv) < 3:
        print("Please pass ref_output_folder_path rocal_ouput_folder_path")
        logging.error(
            "Please pass ref_output_folder_path rocal_ouput_folder_path")
        exit(0)

    # Open the two images
    ref_output_path = sys.argv[1]
    rocal_output_path = sys.argv[2]

    if not (os.path.exists(ref_output_path) and os.path.exists(rocal_output_path)):
        logging.error("Path does not Exists")
        exit()

    total_case_count = 0
    passed_case_count = 0
    failed_case_count = 0
    failed_case_list = []
    golden_output_dir_list = os.listdir(ref_output_path)
    rocal_output_dir_list = os.listdir(rocal_output_path)
    randomized_augmentation = ["Snow", "Rain", "Jitter", "SNPNoise"]
    golden_file_path = ""
    for aug_name in rocal_output_dir_list:
        temp = aug_name.split(".")
        file_name_split = temp[0].split("_")
        if len(file_name_split) > 3:
            file_name_split.pop()
            golden_file_path = "_".join(file_name_split) + ".png"
        else:
            golden_file_path = aug_name

        # For randomized augmentation
        if file_name_split[0] in randomized_augmentation:
            total_case_count = total_case_count + 1
            augmentation_name = aug_name.split(".")[0]
            logging.info("Running %s", augmentation_name)
            passed_case_count = passed_case_count + 1
            logging.info("PASSED ")
        elif golden_file_path in golden_output_dir_list:
            total_case_count = total_case_count + 1
            ref_file_path = ref_output_path + golden_file_path
            rocal_file_path = rocal_output_path + aug_name
            if os.path.exists(rocal_file_path) and os.path.exists(ref_file_path):
                logging.info("Running %s ", aug_name.split(".")[0])
                img1 = Image.open(ref_file_path)
                img2 = Image.open(rocal_file_path)

                # Check if the images have the same dimensions
                if img1.size != img2.size:
                    logging.info(
                        "Golden output and augmentation outputs are having different sizes. Exiting!")
                    exit()

                # Compare the pixel values for each image
                pixel_diff = None
                total_count = 0
                if "larger" in aug_name:
                    resize_width = 400
                    resize_height = 400
                    image_offset = 400
                    pixel_diff, total_count = compare_pixels(
                        img1, img2, aug_name, resize_width, resize_height)
                    pixel_diff2, total_count2 = compare_pixels(
                        img1, img2, aug_name, resize_width, resize_height, image_offset)
                    pixel_diff = [x + y for x,
                                  y in zip(pixel_diff, pixel_diff2)]
                    total_count = total_count + total_count2
                elif "smaller" in aug_name:
                    resize_width = 400
                    resize_height = 400
                    image_offset = 2400
                    pixel_diff, total_count = compare_pixels(
                        img1, img2, aug_name, resize_width, resize_height)
                    pixel_diff2, total_count2 = compare_pixels(
                        img1, img2, aug_name, resize_width, resize_height, image_offset)
                    pixel_diff = [x + y for x,
                                  y in zip(pixel_diff, pixel_diff2)]
                    total_count = total_count + total_count2
                else:
                    pixel_diff, total_count = compare_pixels(
                        img1, img2, aug_name, img1.size[0], img1.size[1])
                total_pixel_diff = 0
                for pix_diff in range(1, 6):
                    total_pixel_diff += pixel_diff[pix_diff]
                mismatch_percentage = round(
                    (total_pixel_diff / total_count) * 100, 2)
                if ((total_pixel_diff == 0) or (mismatch_percentage < 5.0 and pixel_diff[1] == total_pixel_diff) or       # Ignore test cases with single pixel differences less than 5% of total pixel count
                        (mismatch_percentage < 0.5 and ("Blend" in aug_name or "Rotate" in aug_name) and "hip" in aug_name)):  # Ignore mismatch in rotate augmentation less than 0.5% of total pixel count
                    passed_case_count = passed_case_count + 1
                    logging.info("PASSED")
                else:
                    failed_case_list.append(golden_file_path)
                    failed_case_count = failed_case_count + 1
                    logging.info("FAILED")
                    logging.info("Printing pixel mismatch %s", pixel_diff)
                    logging.info("Mismatach percentage %0.2f",
                                 mismatch_percentage)
                    for pix_diff in range(1, 6):
                        logging.info("Percentage of %d pixel mismatch %0.2f", pix_diff, round(
                            (pixel_diff[pix_diff] / total_pixel_diff) * 100, 2))
            else:
                logging.info(
                    "Skipping the testcase as file not found %s", rocal_file_path)
        else:
            logging.info("File not found in ref_output_folder %s",
                         golden_file_path)
    if len(failed_case_list) != 0:
        logging.info("Failing cases: {}".format(", ".join(failed_case_list)))
    logging.info(
        "Total case passed --> {} / {} ".format(passed_case_count, total_case_count))
    logging.info(
        "Total case failed --> {} / {} ".format(failed_case_count, total_case_count))


if __name__ == "__main__":
    main()
