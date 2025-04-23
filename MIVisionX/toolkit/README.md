[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Toolkit

AMD MIVisionX Toolkit is a comprehensive set of help tools for neural net creation, development, training, and deployment. The Toolkit provides you with help tools to design, develop, quantize, prune, retrain, and infer your neural network work in any framework. The Toolkit is designed to help you deploy your work to any AMD or 3rd party hardware, from embedded to servers.

MIVisionX provides you with tools for accomplishing your tasks throughout the whole neural net life-cycle, from creating a model to deploying them for your target platforms.

## AMD Data Analysis Toolkit - ADAT

[AMD Data Analysis Toolkit](amd_data_analysis_toolkit) - **ADAT** is currently available for image classification and object detection. Features such as label summary, hierarchy, image summary, an advanced method of scoring, and many other insightful features can be viewed using this toolkit.

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/classification_summary.png" /></p>

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/bounding_box_summary.png" /></p>

<p align="center"><img width="90%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/classification_graph.png" /></p>

## AMD Data Generation Toolkit

[AMD Data Generation Toolkit](amd_data_generation_toolkit) creates an image resize and validation list for a given image database and label set. The images in the dataset are assumed to have the correct labels in their metadata. The tools help to rename the dataset, resize the images, pad the images with a value(0-255) if they are resized to a square image to keep the aspect ratio, extract the labels from the metadata, and generate logs to indicate errors and mismatches in the dataset.
