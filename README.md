WIP

Code for the tracking of microtubles

Two methods:
1) Manual tracking (WIP)
2) Hybrid CNN and RNN neural network (nn.py)

**Rudimentary Documentation/Example commands:**

**Preprocessing**
select ROI, press **enter** to confirm once, press **enter** to confirm for the second time (**c** to cancel at any time)
Allows for selection of region of interest (ROI), and converts all images to 512x512 grayscale. Scales coordinate training file correspondingly.

```
c:/Users/Jackson/Documents/GitHub/microtuble-tracking/nn/preprocessingnn.py --input_dir C:\Users\Jackson\Downloads\MT7_30min_100x_443_453pm_1500 --output_dir C:\Users\Jackson\Documents\mt_data\preprocessed\imageset1 --coords_file C:\Users\Jackson\Downloads\MT7snake.txt
```


**Postprocessing**

Creates many folders each with transformations of the imageset along with transformed coordinates
```
c:/Users/Jackson/Documents/GitHub/microtuble-tracking/nn/postprocessingnn.py --input_dir C:\Users\Jackson\Documents\mt_data\preprocessed --output_dir C:\Users\Jackson\Documents\mt_data\postprocessed
```