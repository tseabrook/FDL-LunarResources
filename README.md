# FDL2017: Lunar Water and Volatiles
Contributions by Dietmar Backes, Timothy Seabrook, Eleni Bohacek, Anthony Dobrovolskis, Casey Handmer, Yarin Gal.

This repository represents the work of the Frontier Development Labs 2017: Lunar Water and Volatiles team.

NASA's LCROSS mission indicated that water is present in the permanently shadowed regions of the Lunar poles. Water is a key resource for human spaceflight, not least for astronaut life-support but also as an ingredient for rocket fuels.

It is important that the presence of Lunar water is further quantified through rover missions such as NASA's Resource Prospector (RP).
RP will be required to traverse the lunar polar regions and evaluate water distribution by periodically drilling to suitable depths to retrieve samples for analysis.

In order to maximise the value of RP and of future missions, it is important for robust and effective plans to be constructed.
This year's LWAV team began by replicating traverse planning algorithms currently in use by NASA JPL. 
However, when beginning an automated search for maximally lengthed traverses an opportunity became apparent.

Current maps of the Lunar surface are in large composed from optical images captured by the Lunar Reconnaissance Orbiter (LRO) mission.
For our study we were largely interested in optical images from the LRO Narrow Angled Camera (NAC), and elevation measures from the Lunar Orbiter Laser Altimeter Digital Elevation Model (LOLA DEM).

During the production of the digital elevation model, synthetic artefacts were introduced into the images. 
The artefacts represent several different forms of noise, the most troublesome of which appears as thin but large walls and gorges across the surface of the image that are not present in reality.
These artefacts greatly reduced the ability for an automated traverse search. In practice, a team will take a region of interest and spend a lot of time cleaning the artefacts by hand.

A conventional approach to artefact removal might be based on the correlation of multiple images of the same static target. If an element of one image doesn't appear in the others then that element is likely an artefact.
The higher 0.5m resolution NAC images provide an opportunity to identify artefacts in the 20m resolution DEM. However, the NAC images are not properly registered to lunar coordinates.

The LOLA DEM acts as a baseline registration of lunar features to a coordinate system. Whilst NAC optical images do contain latitude and longitude information, they are only accurate to degree minutes (i.e. circumference/(360deg*60mins) <= approx. 500metres).
In order to correlate optical NAC images with the DEM and remove artefacts, coregistration between the two images needs to take place first.

A common approach to coregistration involves feature matching, where features common to both images are identified and the highest likelihood pairing of those features gives a transformation matrix to overlay the images.
Lunar craters are irregular in distribution and shape and very plentiful. As such, craters make great features with which to coregister the LRO NAC and LOLA DEM images.

In order to coregister using craters, the crater features must first be extracted.
State-of-the-art approaches to feature extraction involve the use of Deep Neural Networks, which represent a complex hierarchical set of many convolution filters which, using examples, learn to approximate the optical representation of 'craters'.
Once trained, the Deep Neural Network is able to identify unlabelled examples of craters in new images.

# Future Work
Following feature identification, a segmentation algorithm should be used to extract the pixels that constitute the crater to accurately pinpoint the location.

Feature matching can be achieved using SIFT or other more modern techniques.

Correlation of images might be performed by illumating the digital elevation model to the same conditions as present in the optical image before using a subtractive or divisive approach.

Thresholding represents the simplest approach to determining artefact removal, however the removal of elements falsely identified as artefacts poses a great risk to missions and should be avoided at all costs.
