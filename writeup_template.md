## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1]: ./examples/corners_found_0.jpg "detected corners on chessboard image"
[image2]: ./examples/calibration1_distorted.jpg "distorted"
[image3]: ./examples/calibration1_undistorted.jpg "undistorted"
[image4]: ./examples/straight_lines1_distorted.jpg 
[image5]: ./examples/straight_lines1_undistorted.jpg
[image6]: ./examples/warped_straight_lines2.jpg

[image7]: ./examples/thresholdedBinaryImage73.png "pipeline for creating the thresholded binary image"

[video1]: ./video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file named `calibrate_camera.py` (lines #13 - #62 --> function calibrate_camera).

basically i followed the 'default' procedure to calibrate a camera using opencv. that means, take several images of a predefined chesboard pattern from different angels and distances. then go through all those images and extract the chessboard corners (in this case there are 9 corners on the X-axis and 6 on the Y-axis) using cv2.findChessboardCorners().

to map those corners to real world coordinates, i prepared some "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. here the chessboard pattern helps to ease things up a lot, since i know that they all have the same distance. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

Finally the output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I added some images that describe the process.

![image1]
![image2]
![image3]
![image4]
![image5]
![image6]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![image2]
![image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for creating a binary thresholded image can be found in `lanefinder.py -> function: create_thresholded_binary_image()` (line #154 - #236).
I used the colored image as well as a grayscale version of it to generate several gradient thresholds. on the grayscale image i applied an CLAHE upfront, since i achived good results with it on the traffic sign project.

| function applied         		|     parameters	        					| 
|:---------------------:|:---------------------------------------------:| 
| thresholded absolute sobel in x direction		|  sobel_kernel = 13, thresh = (23, 100) | 
| thresholded absolute sobel in y direction		|  sobel_kernel = 13, thresh = (23, 100) | 
| thresholded magnitude of gradient		|  sobel_kernel = 13, thresh = (70, 130) | 
| thresholded direction of gradient		|  sobel_kernel = 15, thresh = (0.7, 1.2) | 

I combined the first three using a binary AND between x- and y- gradient combined through a binary OR on magnitude gradient (line #173). the direction gradient did not help me very much in the end.

then i applied some gradient combinations on different colorspaces HLS and HSV.

| function applied         		|     parameters	        					| 
|:---------------------:|:---------------------------------------------:| 
| thresholded absolute sobel in x-direction on L-channel combined with an thresholded saturation channel (S-channel) in HLS color space |  sobel_kernel = 3, color_thresh=(170, 255), gradient_thresh=(15, 70)  | 
| thresholded absolute sobel in x-direction on V-channel combined with an thresholded saturation channel (S-channel) in HSV color space |  sobel_kernel = 3, color_thresh=(170, 255), gradient_thresh=(9, 42)  | 

additionally i also tried to filter stronger for the white and yellow parts of the image. therefor i applied a filter on the HLS colorspace. (line #183)

Finally i combined the gray-binary from the first step and the HLS binary images using a logical OR and combined that with the white-yellow-filter using and AND. (code line #184). The HSV binary image i didn't use in the end.

here you can see the intermediate images of a sample frame:

![image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

for the code to calulcate the perspectice transformation matrix, please check the function `calc_perspactive_transform_matrix` (lines #64 - #77 in `calibrate_camera.py`)

i manually extracted the source ann destination points from undistorted version of the test_image straight_lines2.jpg which i hardcoded

```python
    src = np.float32([[585, 460], [702, 460], [310, 660], [1000, 660]])
    dst = np.float32([[400, 200], [900, 200], [400, 660], [900, 660]])
    M = cv2.getPerspectiveTransform(src, dst)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 400, 200      | 
| 702, 460      | 900, 200      |
| 310, 660      | 400, 660      |
| 1000, 660     | 900, 660      |

I verified that my perspective transform was working as expected by comparing an normal image and its warped counter part. i choose an image of a straight road, to make visual comparisson easy. expectation is that the lines of the road become as parallel as possible.

![image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
