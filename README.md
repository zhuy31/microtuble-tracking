**Microtubule Tracking**
(warning: experimental)

Some python code to track (potentially very noisy and low-quality) microtubule data.
In the future, tools to analyze single-microtubule motion will be added.

How to use:

1. Use preprocess.py to preprocess the images into another directory
2. Use trackimages.py to view a video of the tracked microtubule. The output is in output_coordinates.txt.

Rough flowchart of the algorithm:
1) Denoising
2) CLAHE 
3) Connected component analysis
4) Skeletonizing and pruning
5) Contouring
6) Tracking points using contour

To-do:
1) Zoom algorithim to improve accuracy
2) Potentially use CNN to upscale noisy images / denoise
3) Clean up code, comment, and organize
4) (?) Create neural network to do all of this faster (?)
5) Create phonon spectrum calculator.