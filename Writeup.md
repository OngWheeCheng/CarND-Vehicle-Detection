# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

All image references can be found in output_images folder.
* Example of a car and non-car image: /test_images/carNotCars/car_not_car.jpg
* Example of a car and non-car HOG features extraction image: /test_images/hog/HOG.jpg
* Sliding windows applied to test images: /test_images/slidingWindows/*.jpg
* Bounded boxes and heat maps in test images: /test_images/final/*.jpg
* Number of cars found in test images: /test_images/final /*.jpg
* Bounding boxes for vehicles detected in test images: /test_images/final/*.jpg
* Six consecutive video frames: /video_frames/*.jpg

[//]: # (Image References)

[image1]: ./output_images/test_images/carNotCars/car_not_car.jpg "Car Not Car"
[image2]: ./output_images/test_images/hog/HOG.jpg "HOG"
[image3]: ./md_images/HOG_parameters.png "HOG Parameters"
[image4]: ./output_images/test_images/slidingWindows/test1.jpg "Example Sliding Window"
[image5]: ./md_images/example_search_result.png "Example Search Result"
[image6]: ./md_images/heatmap_28s.png "Heatmap @ 28s"
[image7]: ./md_images/heatmap_2804s.png "Heatmap @ 28.04s"
[image8]: ./md_images/heatmap_2808s.png "Heatmap @ 28.08s"
[image9]: ./md_images/heatmap_2812s.png "Heatmap @ 28.12s"
[image10]: ./md_images/heatmap_2816s.png "Heatmap @ 28.16s"
[image11]: ./md_images/heatmap_2820s.png "Heatmap @ 28.20s"
[image12]: ./md_images/integrated_heatmap_28s.png "Integrated Heatmap @ 28s"
[image13]: ./md_images/integrated_heatmap_2804s.png "Integrated Heatmap @ 28.04s"
[image14]: ./md_images/integrated_heatmap_2808s.png "Integrated Heatmap @ 28.08s"
[image15]: ./md_images/integrated_heatmap_2812s.png "Integrated Heatmap @ 28.12s"
[image16]: ./md_images/integrated_heatmap_2816s.png "Integrated Heatmap @ 28.16s"
[image17]: ./md_images/integrated_heatmap_2820s.png "Integrated Heatmap @ 28.20s"
[image18]: ./md_images/bounding_box_28s.png "Bounding Box @ 28s"
[image19]: ./md_images/bounding_box_2804s.png "Bounding Box @ 28.04s"
[image20]: ./md_images/bounding_box_2808s.png "Bounding Box @ 28.08s"
[image21]: ./md_images/bounding_box_2812s.png "Bounding Box @ 28.12s"
[image22]: ./md_images/bounding_box_2816s.png "Bounding Box @ 28.16s"
[image23]: ./md_images/bounding_box_2820s.png "Bounding Box @ 28.20s"

### Histogram of Oriented Gradients (HOG)
#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the Hog Feature Extraction cell of the IPython notebook,
_Vehicle_Extraction.ipynb_.

The first step is to read in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the 'vehicle' and 'non-vehicle' classes:

![alt text][image1]

Different color spaces and different `skimage.hog()` parameters ('orientations', 'pixels_per_cell', and 'cells_per_block') were explored. Random images from each of the two classes were selected and displayed to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the 'YCrCb' color space and HOG parameters of 'orientations=9', 'pixels_per_cell=(8, 8)' and 'cells_per_block=(2, 2)':

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

Various combinations of parameters are attempted and the best accuracy that is supported by my laptop is chosen.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in IPython notebook, _Train Model.ipynb_.

The ‘vehicle’ and ‘non-vehicle’ images are randomly selected and split, where the percentages of training data and test data are 80% and 20% respectively.

The linear SVM was trained using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. The classifier accuracy is determined using the test set. This avoids overfitting / improves the generalization of the classifier.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The multi-scale sliding window search is contained in the Sliding Window (Multi-Scale) cell of the IPython notebook, _Vehicle_Extraction.ipynb_. It is implemented using Udacity-provided functions: `slide_windows()` and `hot_windows()`. The sliding window search is limited to the area below the horizon (y-axis); to limit the search to the ‘ground’ area.

The sliding window search area is subdivided into 3 sections, with the smallest scale window search taking place nearest to the horizon (y-axis) and the remaining 2 sections with progressively larger scale search window and further away from the horizon.

The results of the sliding window search is drawn using Udacity-provided function, `draw_boxes()`. The parameters of the sliding window search are shown below:

| Window Scale | Start Coordinates (x,y) | End Coordinates (x, y) | Color |
|:-:|:-:|:-:|:-:|
| 96 x 64 | 250, 380 | 1280, 520 | Green |
| 192 x 128 | 225, 380 | 1280, 600 | Cyan |
| 288 x 192 | 200, 400 | 1280, 690 | Yellow |

An example of the sliding window search is shown below:

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

A 3 scales search was implemented using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. Each search scale has a different search area (i.e. start and end coordinates along the x- and y-axes) to minimize search time and obtain good search results. An example image of the search result is shown below:

![alt text][image5]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link](https://www.dropbox.com/s/ftmg4mc6k98yo5g/CarND-Vehicle-Detection.mp4?dl=0) to the video result.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The positions of positive detections were recorded in each frame of the video. From the positive detections, a heatmap was created and then thresholded to identify vehicle positions. `scipy.ndimage.measurements.label()` was used to identify individual blobs in the heatmap. Each blob was assumed to correspond to a vehicle and bounding boxes were constructed to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six video frames and their corresponding heatmaps, at 40msec interval starting from 28secs:

| Frame @ t=28s |
|:-:|
| ![alt text][image6] |

| Frame @ t=28.04s |
|:-:|
| ![alt text][image7] |

| Frame @ t=28.08s |
|:-:|
| ![alt text][image8] |

| Frame @ t=28.12s |
|:-:|
| ![alt text][image9] |

| Frame @ t=28.16s |
|:-:|
| ![alt text][image10] |

| Frame @ t=28.20s |
|:-:|
| ![alt text][image11] |

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

| Frame @ t=28s | Frame @ t=28.04s |
|:-:|:-:|
| ![alt text][image12] | ![alt text][image13] |

| Frame @ t=28.08s | Frame @ t=28.12s |
|:-:|:-:|
| ![alt text][image14] | ![alt text][image15] |

| Frame @ t=28.16s | Frame @ t=28.20s |
|:-:|:-:|
| ![alt text][image16] | ![alt text][image17] |

Here the resulting bounding boxes are drawn onto the last frame in the series:

| Frame @ t=28s | Frame @ t=28.04s |
|:-:|:-:|
| ![alt text][image18] | ![alt text][image19] |

| Frame @ t=28.08s | Frame @ t=28.12s |
|:-:|:-:|
| ![alt text][image20] | ![alt text][image21] |

| Frame @ t=28.16s | Frame @ t=28.20s |
|:-:|:-:|
| ![alt text][image22] | ![alt text][image23] |

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Issues faced:
* the training of classifier, processing of images and video takes a long time due to lack of processing power on my laptop
* A lot of time is spent testing different hyper parameters to obtain reasonably good classifier accuracy and vehicle search results.

Likely pipeline failure:
* The start coordinate along the x-axis has various offsets (200 – 250 pixels). This assumes that the self-driving car is driving along the leftmost lane. If the self-driving car is driving along lanes other than the leftmost, it may not be able to detect vehicles on its left hand side.

Improvements:
* Modify the start coordinate along the x-axis to 0. This ensures detection of vehicles on both sides of the self-driving car, amid longer processing time.
* Train the classifier using hyper parameters of higher values, e.g. spatial size of 64x64
