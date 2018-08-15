import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

#TODO introduce a commandline option
#DEBUG = True
DEBUG = False

def calibrateCamera():

    ## find chessboard corners

    # prepare object points
    nx = 9# the number of inside corners in x
    ny = 6# the number of inside corners in y

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    for index, filename in enumerate(images):
        image = mpimg.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if DEBUG:
            print("find chessboard corners in: {} [which has the shape: {}]".format(filename, image.shape))
        #NOTE some of the images have the shape (721, 1281, 3)! i guess it's small enough to NOT be a problem,
        #     but we may have to rescale them later to get all images into the same shape, since cv2.calibrateCamera()
        #     will get all points at once, while taking one image size as parameter

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            write_name = 'corners_found_' + str(index) + '.jpg'
            cv2.imwrite('output_images/' + write_name, image)

    # use the last image to get the shape/size of the images
    #NOTE for some reason openCV puts y before x as output of image.shape, so x has index 1 instead of 0
    imageSize = (image.shape[1], image.shape[0])
    if DEBUG:
        print(imageSize)

    ## Do camera calibration given object points and image points
    return cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)


def undistortImage(image, cameraMatrix, distortionCoeffs):
    image = cv2.undistort(image, cameraMatrix, distortionCoeffs, None, cameraMatrix)
    #TODO uncomment to save undistorted image
    #mpimg.imwrite('output_images/test_undist.jpg', image)
    return image


# gray must be a 1 channel grayscale image
def absSobelThresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Apply threshold
    # Create a copy and apply the threshold
    grad_binary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

# gray must be a 1 channel grayscale image
def magThresh(gray, sobel_kernel=3, magThresh=(0, 255)):
    # Calculate gradient magnitude

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # Apply threshold
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= magThresh[0]) & (gradmag <= magThresh[1])] = 1

    return mag_binary

# gray must be a 1 channel grayscale image
def dirThreshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Apply threshold
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary

def hlsThreshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    s_sx_binary = np.zeros_like(s_channel)
    #s_sx_binary[(s_binary | sxbinary)] = 1
    s_sx_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return s_sx_binary

def hsvThreshold(img, s_thresh=(170, 255), vsx_thresh=(9, 42)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= vsx_thresh[0]) & (scaled_sobel <= vsx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    s_sx_binary = np.zeros_like(s_channel)
    #s_sx_binary[(s_binary | sxbinary)] = 1
    s_sx_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return s_sx_binary

def createThresholdedBinaryImage(image):
    # gradients on grayscale image
    ksize = 13
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Apply each of the thresholding functions
    gradx = absSobelThresh(gray, orient='x', sobel_kernel=ksize, thresh=(23, 100))
    grady = absSobelThresh(gray, orient='y', sobel_kernel=ksize, thresh=(23, 100))
    mag_binary = magThresh(gray, sobel_kernel=ksize, magThresh=(70, 130))
    #dir_binary = dirThreshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
    dir_binary = dirThreshold(gray, sobel_kernel=15, thresh=(0.7, 1.2))

    gray_combined = np.zeros_like(dir_binary)
    gray_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # gradients on different color spaces
    hls_binary = hlsThreshold(image, s_thresh=(170, 255), sx_thresh=(15, 70))
    hsv_binary = hsvThreshold(image, s_thresh=(170, 255), vsx_thresh=(9, 42))

    combined_binary = np.zeros_like(gray_combined)
    combined_binary[(gray_combined == 1) | ((hls_binary == 1) & (hsv_binary == 1))] = 1

    return combined_binary

#TODO clean up! most stuff is not really needed here. only for debugging...
def calcPerspactiveTransformMatrix(cameraMatrix, distortionCoeffs):
    #                  top_left    top_right  bot_left    bot_right
    # from undistorted straight_lines2
    src = np.float32([[585, 460], [702, 460], [310, 660], [1000, 660]])
    # from orig straight_lines1
    #src = np.float32([[580, 460], [700, 460], [300, 660], [1000, 660]])
    dst = np.float32([[400, 200], [900, 200], [400, 660], [900, 660]])
    #dst = np.float32([[400, 460], [900, 460], [400, 660], [900, 660]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    if DEBUG:
        image = mpimg.imread('test_images/straight_lines1.jpg')
        image = undistortImage(image, cameraMatrix, distortionCoeffs)

        print(image.shape)
        y_max, x_max = image.shape[0], image.shape[1]
        warped = cv2.warpPerspective(image, M, (x_max, y_max), flags=cv2.INTER_LINEAR)

        image2 = mpimg.imread('test_images/straight_lines2.jpg')
        image2 = undistortImage(image2, cameraMatrix, distortionCoeffs)
        warped2 = cv2.warpPerspective(image2, M, (x_max, y_max), flags=cv2.INTER_LINEAR)

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(image)
        ax1.set_title('image', fontsize=15)

        ax2.imshow(image2)
        ax2.set_title('image2', fontsize=15)

        ax3.imshow(warped)
        ax3.set_title('warped', fontsize=15)

        ax4.imshow(warped2)
        ax4.set_title('warped2', fontsize=15)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    #TODO check if return of (x_max, y_max) makes sense here!
    #return M, (x_max, y_max)
    return M


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    if DEBUG:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 70
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        if DEBUG:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0, 1, 0), 3)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0, 1, 0), 3)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        #pass # Remove this when you add your function
        if len(good_left_inds) > minpix:
            mean_left = np.mean(nonzerox[good_left_inds])
            leftx_current = np.int(mean_left)
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if DEBUG:
        return leftx, lefty, rightx, righty, out_img

    #TODO check if returning 'None' makes sense here
    return leftx, lefty, rightx, righty, None


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    global left_fit
    global right_fit

    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty**2  + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    global left_fit
    global right_fit

    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 32

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

left_fit = None
right_fit = None

def pipeline(image, cameraMatrix, distortionCoeffs, transformationMatrix, image_size):
    global left_fit
    global right_fit

    ## Apply a distortion correction to raw images.
    # first undistort all images to minimize camera effects on the image
    image = undistortImage(image, cameraMatrix, distortionCoeffs)

    ## Use color transforms, gradients, etc., to create a thresholded binary image.
    #TODO crop image first, to save computation power
    binary_image = createThresholdedBinaryImage(image)

    ## Apply a perspective transform to rectify binary image ("birds-eye view").
    binary_warped = cv2.warpPerspective(binary_image, transformationMatrix, image_size, flags=cv2.INTER_LINEAR)

    ## Detect lane pixels and fit to find the lane boundary.
    out_image = None
    if (left_fit == None) or (right_fit == None):
        leftx, lefty, rightx, righty, out_image = find_lane_pixels(binary_warped)
    else:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped)

    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    #FIXME fulfill this missing requirement
    ## Determine the curvature of the lane and vehicle position with respect to center.

    if DEBUG:
        #==========
        #FIXME remove this suff
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(image)
        ax1.set_title('image', fontsize=15)

        ax2.imshow(binary_image)
        ax2.set_title('binary_image', fontsize=15)

        ax3.imshow(binary_warped)
        ax3.set_title('binary_warped', fontsize=15)

        ax4.plot(left_fitx, ploty, color='yellow')
        ax4.plot(right_fitx, ploty, color='yellow')
        ax4.imshow(out_image)
        ax4.set_title('out_image', fontsize=15)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        #plt.imshow(binary_warped)
        #plt.imshow(out_img)

        plt.show()
        #==========

    # ATTENTION: out_image is 'None' if DEBUG == False
    return out_image, left_fitx, right_fitx, ploty


def drawing(undist_image, left_fitx, right_fitx, ploty, transformationMatrix):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undist_image).astype(np.uint8)
    #TODO find out how and when to use this approach
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp = warp_zero

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, transformationMatrix, (undist_image.shape[1], undist_image.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    # Combine the result with the original image
    result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)

    return result

def process_image(image):
    # use global variable for simplicity reasons here
    global cameraMatrix
    global distortionCoeffs
    global transformationMatrix

    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #TODO do i really need image_size here?
    image_size = (image.shape[1], image.shape[0])

    if DEBUG:
        print("run pipeline on {}".format(filename))

    out_image, left_fitx, right_fitx, ploty = pipeline(image, cameraMatrix, distortionCoeffs, transformationMatrix, image_size)

    undist_image = undistortImage(image, cameraMatrix, distortionCoeffs)
    ## Warp the detected lane boundaries back onto the original image.
    ## Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    #FIXME add numerical estimations
    result = drawing(undist_image, left_fitx, right_fitx, ploty, transformationMatrix)

    if DEBUG:
        plt.imshow(result)
        plt.show()

    return result
    

cameraMatrix = None
distortionCoeffs = None
transformationMatrix = None

def prepare_globals():
    # use global variable for simplicity reasons here
    global cameraMatrix
    global distortionCoeffs
    global transformationMatrix

    ## Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    print("calibrate camera...")
    ret, cameraMatrix, distortionCoeffs, rvecs, tvecs = calibrateCamera()
    print("done.")
    
    if DEBUG:
        image = mpimg.imread('test_images/straight_lines1.jpg')
        image = undistortImage(image, cameraMatrix, distortionCoeffs)
        mpimg.imsave('output_images/straight_lines1_undistorted.jpg', image)

        image = mpimg.imread('test_images/straight_lines2.jpg')
        image = undistortImage(image, cameraMatrix, distortionCoeffs)
        mpimg.imsave('output_images/straight_lines2_undistorted.jpg', image)

    # calculate transformation matrix
    #transformationMatrix, image_size = calcPerspactiveTransformMatrix(cameraMatrix, distortionCoeffs)
    transformationMatrix = calcPerspactiveTransformMatrix(cameraMatrix, distortionCoeffs)


def single_image_main():
    global left_fit
    global right_fit

    # Make a list of calibration images
    images = glob.glob('test_images/test*.jpg')

    for index, filename in enumerate(images):
        image = mpimg.imread(filename)

        # set left_fit and right_fit to None in single_image_mode to force sliding window on each image without
        # 'remembering' anything from the previous image, since those are not related
        left_fit = None
        right_fit = None

        result = process_image(image)
        plt.imshow(result)
        plt.show()

def main():
    video_out = 'video_out.mp4'
    #video_in = VideoFileClip('project_video.mp4')
    video_in = VideoFileClip('project_video_cut.mp4')

    print("processing video...")

    video_clip = video_in.fl_image(process_image)
    video_clip.write_videofile(video_out, audio=False)

if __name__ == "__main__":
    prepare_globals()

    #FIXME in single_image_mode we have to deactivate the reuse of prior left_fit and right_fit polynoms!
    #single_image_main()
    main()

