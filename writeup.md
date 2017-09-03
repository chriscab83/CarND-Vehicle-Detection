## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_images.jpg
[image2]: ./output_images/hog_example.jpg
[image3]: ./output_images/size1_boxes.jpg
[image4]: ./output_images/size2_boxes.jpg
[image5]: ./output_images/size3_boxes.jpg
[image6]: ./output_images/size4_boxes.jpg
[image7]: ./output_images/found_rects.jpg
[image8]: ./output_images/heatmap.jpg
[image9]: ./output_images/heatmap_threshold.jpg
[image10]: ./output_images/labels.jpg
[image11]: ./output_images/pipline_foundcars.jpg
[image12]: ./output_images/pipeline_testimg_0.jpg
[image13]: ./output_images/pipeline_testimg_1.jpg
[image14]: ./output_images/pipeline_testimg_2.jpg
[image15]: ./output_images/pipeline_testimg_3.jpg
[image16]: ./output_images/pipeline_testimg_4.jpg
[image17]: ./output_images/pipeline_testimg_5.jpg
[video1]: ./output_images/project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook in the method, `get_hog_features`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and used them to train my SVM. The above parameters gave the highest accuracy ratings in that training so that is what I went with.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the `svm.fit()` method of SKLearns LinearSVM classifier ran on the feature sets of both car nad not car images. I created a feature set of both car and not car images using the method in the first code cell, `extract_features()`.  This method returned a flattened feature vector of HOG features, spatial binning features, and color histogram features.  Those feature sets were then stacked to create a single vector of car / not car feature sets.  That vector was then normalized using SKLearn's `StandardScaler().fit()` method. The vector was then split into training and testing vectors.  The training vector set were the feature sets used in with the `svm.fit()` to train my SVM. Afterwords, the SVM was tested on the testing feature sets to an accuracy rating of 99.47%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First, I implemented a method `find_cars()` that took in various parameter details to search a specified region of a given image at specified intervals (windows) with specifics on the classification details to use on those windows to return car bounding boxes.  This method can be found in the 8th code cell in the jupyter notebook. I then tested various region threshold and overlaps at those regions to find which values returned promising bounding box results on my test image.  The examples of those tests can be seen in the following images:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

I then combined of the bounding boxes returned from those methods into a single bounding box vector to be used through the car detection, code can be seen in the 13th code cell of the jupyter notebook.

![alt_text][image7]

I then implemented a heatmap method over the resulting bounding boxes in code cells 14 and 15.

![alt_text][image8]

I thresholded the heatmap to help narrow down on actual car and eliminate noise using a threshold factor of 4 in code cell 16.

![alt_text][image9]

Finally, I used `scipy.ndimage.measurements.label()` to label the bounding boxes and find individual cars in the image.

![alt_text][image10]

And then drew those labeled bounding boxes over the image to show the found vehicles.

![alt_text][image11]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt_text][image12]
![alt_text][image13]
![alt_text][image14]
![alt_text][image15]
![alt_text][image16]
![alt_text][image17]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I reimplemented my pipeline to work slightly different on time series data such as video frames. I created a `VehicleDetection` class that would hold the last n bound box results from frame to frame. Then, rather than using the raw bounding boxes returned from my `find_cars()` method, I added those bounding boxes to the list of bounding boxes held in the vehicle detection object. I then used those rectangle sets from the previous frames along with the current frame to create the heatmap.  I then used a multiple of the length of frames stored in the detector and the single frame threshold as the new threshold of the heatmap.  By doing this, I was able to smooth the transitions of the bounding boxes between frames as well as reduce noise over the course of the video.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline had issues in that over a course of a few frames, there would be some false trailing bounding boxes where the vehicle was rather than is.  Also, the bounding boxes were often tighter than the actual cars which in a true self driving vehicle could cause problems in the calculation of the distance away from us the found vehicle is.  This could be fixed by testing further scaling factors in the window sizes and posibly using smaller windows to get a tighter read on a vehicles location.  Also, because my high threshold used to try and get a tighter bounding box, I found the pipeline had a hard time finding a car on the edges coming into view as there were not enough overlapping positive results until the vehicle was further in frame.   
