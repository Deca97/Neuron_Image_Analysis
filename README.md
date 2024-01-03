Neuron Image Analysis Tool

Overview

The Neuron Image Analysis Tool is a Python program designed for the analysis of neuronal images in LIF format. It provides a user-friendly interface to perform various image processing and statistical analysis tasks, enabling researchers to analyze images with ease. The tool supports images with 2 channels and allows users to select, crop, apply background removal, threshold, count dots, and perform statistical tests between control and intervention groups.

Features

File Selection: Open and select LIF files containing neuronal images with 2 channels.

Image Cropping: Choose specific regions of interest (ROIs) by cropping the images.

Image Processing:

Background Removal: Apply background removal techniques to enhance the signal-to-noise ratio.
Thresholding: Set intensity thresholds to segment the images.
Dot Counting: Count dots with a specific area relative to the expression of a specific protein.

Grouping:

Split into Groups: Automatically categorize images into control and intervention groups based on naming conventions.
Outlier Detection and Removal: Detect and remove outliers to ensure robust statistical analysis.

Statistical Analysis:

t-Tests: Perform t-tests between control and intervention groups for dot count, mean area of each channel, and overlapping dots of the two channels.
Results Visualization:

Plotting: Generate comprehensive plots to visualize the results of the statistical analyses.

Requirements

Python 3.x
