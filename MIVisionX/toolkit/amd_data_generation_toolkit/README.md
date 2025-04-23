# AMD Data Generation Toolkit

## Classification Label Generator Toolkit

[CLG-Toolset](classification) creates a classification label validation list for a given image database and labels set. The images in the dataset are assumed to have the correct labels in their metadata. The tools help to rename the dataset, resize the images, pad the images with a value(0-255) if they are resized to a square image to keep the aspect ratio, extract the labels from the metadata, and generate logs to indicate errors and mismatches in the dataset.