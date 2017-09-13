# Crater Detection

Several approaches have been developed for crater detection, in reverse order of efficacy those are:

**DNN**

The Convolutional Neural Network has provided the greatest result, hinging on the labelled data samples collected.
Precision accuracy for Polar regions, trained on equatorial and polar data reach approx. 98%.

**ConvolutionFilter**

The Single-layer adaptive convolution filter performs slowly and with low accuracy, but without the need for human intervention.
The mathematical model for the filter describing a semi-illuminated crater was written by Casey Handmer.

**Polygon**

This script takes a bit too long to run and didn't end up being too effective, written by Timothy Seabrook

Description:
  1. Detect edges using a canny filter (This in itself isn't reliable enough)
  2. Group edges into 'shapes', permitting that some gaps may exist
  3. For each shape, use a line-of-fit split-and-merge strategy to form straight lines from pixels
  4. Convert shapes into graphs, lines to nodes and edges
  5. Find cycles in graphs to identify convex shapes
  6. Threshold convex shapes to identify craters

**Hough**

Simplistic use of the Hough Circle detector over a Canny filter for preliminary crater detection results.
