import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import pickle

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

#TODO introduce a commandline option
#DEBUG = True
DEBUG = False

# pickle file used for camera specifica
PICKLE_FILE_NAME = 'camera_calibration_data.p'

def undistortImage(image, camera_matrix, distortion_coeffs):
    image = cv2.undistort(image, camera_matrix, distortion_coeffs, None, camera_matrix)
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

# special thanks to gpavlov2016
# https://github.com/gpavlov2016/CarND-Advanced-Lane-Lines/blob/master/utils.py
# Define a function that thresholds the S-channel of HLS
def select_white_and_yellow(img):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #Filter out all colors except yellow and white:
    lower_yellow = np.array([0, 0, 40])
    upper_yellow = np.array([100, 255, 255])
    lower_white = np.array([0, 160, 0])
    upper_white = np.array([255, 255, 255])
    ymask = cv2.inRange(hls, lower_yellow, upper_yellow)
    wmask = cv2.inRange(hls, lower_white, upper_white)
    mask = np.logical_or(ymask, wmask)
    return mask

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
    #gray_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    gray_combined[((gradx == 1) & (grady == 1)) | (mag_binary == 1)] = 1

    # gradients on different color spaces
    hls_binary = hlsThreshold(image, s_thresh=(170, 255), sx_thresh=(15, 70))
    hsv_binary = hsvThreshold(image, s_thresh=(170, 255), vsx_thresh=(9, 42))

    combined_binary = np.zeros_like(gray_combined)
    #combined_binary[(gray_combined == 1) | ((hls_binary == 1) & (hsv_binary == 1))] = 1
    #combined_binary[(gray_combined == 1) | (hls_binary == 1)] = 1

    white_yellow = select_white_and_yellow(image)
    combined_binary[((gray_combined == 1) | (hls_binary == 1)) & (white_yellow == 1)] = 1

    if DEBUG:
        #==========
        #FIXME remove this suff
        f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(gradx)
        ax1.set_title('gradx', fontsize=15)

        ax2.imshow(grady)
        ax2.set_title('grady', fontsize=15)

        ax3.imshow(mag_binary)
        ax3.set_title('mag_binary', fontsize=15)

        ax4.imshow(dir_binary)
        ax4.set_title('dir_binary', fontsize=15)

        ax5.imshow(gray_combined)
        ax5.set_title('gray_combined', fontsize=15)

        ax6.imshow(hls_binary)
        ax6.set_title('hls_binry', fontsize=15)

        ax7.imshow(hsv_binary)
        ax7.set_title('hsv_binary', fontsize=15)

        ax8.imshow(combined_binary)
        ax8.set_title('combined_binary', fontsize=15)

        ax9.imshow(white_yellow)
        ax9.set_title('white_yellow', fontsize=15)

        #ax9.imshow(scaled_sobel)
        #ax9.set_title('scaled_sobel', fontsize=15)

        #ax10.imshow(scaled_sobel_combined)
        #ax10.set_title('scaled_sobel_combined', fontsize=15)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        #plt.imshow(binary_warped)
        #plt.imshow(out_img)

        filename = 'output_images/debug/thresholdedBinaryImage' + str(frame_count) + '.png'
        plt.savefig(filename)
        plt.close()
        #==========


    return combined_binary


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    if DEBUG:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        plt.imshow(histogram)
        filename = 'output_images/debug/histogram' + str(frame_count) + '.png'
        plt.savefig(filename)
        plt.close()


    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 60
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
    margin = 42

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

    out_img = None
    if DEBUG:
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 1, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 1, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##


    return leftx, lefty, rightx, righty, out_img

left_fit = None
right_fit = None

def pipeline(image, camera_matrix, distortion_coeffs, transformation_matrix, image_size):
    global left_fit
    global right_fit

    if DEBUG:
        global frame_count

    ## Apply a distortion correction to raw images.
    # first undistort all images to minimize camera effects on the image
    image = undistortImage(image, camera_matrix, distortion_coeffs)

    ## Use color transforms, gradients, etc., to create a thresholded binary image.
    #TODO crop image first, to save computation power
    binary_image = createThresholdedBinaryImage(image)

    ## Apply a perspective transform to rectify binary image ("birds-eye view").
    binary_warped = cv2.warpPerspective(binary_image, transformation_matrix, image_size, flags=cv2.INTER_LINEAR)

    ## Detect lane pixels and fit to find the lane boundary.
    out_image = None
    if (left_fit == None) or (right_fit == None):
        leftx, lefty, rightx, righty, out_image = find_lane_pixels(binary_warped)
    else:
        leftx, lefty, rightx, righty, out_image = search_around_poly(binary_warped)

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

        filename = 'output_images/debug/fig' + str(frame_count) + '.png'
        plt.savefig(filename)
        plt.close()
        #==========

    # ATTENTION: out_image is 'None' if DEBUG == False
    return out_image, left_fitx, right_fitx, ploty

frame_count = 0

def drawing(undist_image, left_fitx, right_fitx, ploty, transformation_matrix):
    global frame_count

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
    newwarp = cv2.warpPerspective(color_warp, transformation_matrix, (undist_image.shape[1], undist_image.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    # Combine the result with the original image
    result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)

    ## add some text to the images
    left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fitx, right_fitx)

    text_color = (255, 255, 0)
    cv2.putText(result, "frame: {}".format(frame_count), (100, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=text_color)
    cv2.putText(result, "left_curverad: {} m".format(left_curverad), (100, 65), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=text_color)
    cv2.putText(result, "right_curverad: {} m".format(right_curverad), (100, 80), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=text_color)
    frame_count += 1

    return result

def process_image(image):
    # use global variable for simplicity reasons here
    global camera_matrix
    global distortion_coeffs
    global transformation_matrix
    global image_size

    out_image, left_fitx, right_fitx, ploty = pipeline(image, camera_matrix, distortion_coeffs, transformation_matrix, image_size)

    undist_image = undistortImage(image, camera_matrix, distortion_coeffs)
    ## Warp the detected lane boundaries back onto the original image.
    ## Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    #FIXME add numerical estimations
    result = drawing(undist_image, left_fitx, right_fitx, ploty, transformation_matrix)

    if DEBUG:
        global frame_count

        plt.imshow(result)
        filename = 'output_images/debug/result' + str(frame_count) + '.png'
        plt.savefig(filename)
        plt.close()

    return result


def measure_curvature_pixels(ploty, left_fitx, right_fitx):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+(2*left_fit[0]*y_eval*ym_per_pix+left_fit[1])**2)**(3/2) ) / np.absolute(2*left_fit[0])  ## Implement the calculation of the left line here
    right_curverad = ((1+(2*right_fit[0]*y_eval*xm_per_pix+right_fit[1])**2)**(3/2) ) / np.absolute(2*right_fit[0])  ## Implement the calculation of the right line here

    return left_curverad, right_curverad


camera_matrix = None
distortion_coeffs = None
transformation_matrix = None
image_size = None

def prepare_globals():
    # use global variable for simplicity reasons here
    global camera_matrix
    global distortion_coeffs
    global transformation_matrix
    global image_size

    with open(PICKLE_FILE_NAME, 'rb') as f:
        calibration_data = pickle.load(f)
    camera_matrix = calibration_data["camera_matrix"]
    distortion_coeffs = calibration_data["distortion_coeffs"]
    transformation_matrix = calibration_data["transformation_matrix"]
    image_size = calibration_data["image_size"]


def single_image_main():
    global left_fit
    global right_fit

    # Make a list of calibration images
    #images = glob.glob('test_images/test*.jpg')
    images = glob.glob('test_images/test1*.jpg')
    #images = glob.glob('test_images/test4*.jpg')

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
    video_in = VideoFileClip('project_video.mp4')
    #video_in = VideoFileClip('project_video_cut.mp4')
    #video_in = video_in.subclip(1.3, 3.3)

    print("processing video...")

    video_clip = video_in.fl_image(process_image)
    video_clip.write_videofile(video_out, audio=False)

if __name__ == "__main__":
    prepare_globals()

    #FIXME in single_image_mode we have to deactivate the reuse of prior left_fit and right_fit polynoms!
    #single_image_main()
    main()

