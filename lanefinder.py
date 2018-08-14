import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

#TODO introduce a commandline option
DEBUG = True

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


def absSobelThresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

def magThresh(image, sobel_kernel=3, magThresh=(0, 255)):
    # Calculate gradient magnitude

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

def dirThreshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction

    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
    #TODO convert image to grayscale only once before calling thresholding functions
    # Apply each of the thresholding functions
    gradx = absSobelThresh(image, orient='x', sobel_kernel=ksize, thresh=(23, 100))
    grady = absSobelThresh(image, orient='y', sobel_kernel=ksize, thresh=(23, 100))
    mag_binary = magThresh(image, sobel_kernel=ksize, magThresh=(70, 130))
    #dir_binary = dirThreshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
    dir_binary = dirThreshold(image, sobel_kernel=15, thresh=(0.7, 1.2))

    gray_combined = np.zeros_like(dir_binary)
    gray_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # gradients on different color spaces
    hls_binary = hlsThreshold(image, s_thresh=(170, 255), sx_thresh=(15, 70))
    hsv_binary = hsvThreshold(image, s_thresh=(170, 255), vsx_thresh=(9, 42))

    combined_binary = np.zeros_like(gray_combined)
    combined_binary[(gray_combined == 1) | ((hls_binary == 1) & (hsv_binary == 1))] = 1

    return combined_binary


def calcPerspactiveTransformMatrix():
    image = mpimg.imread('test_images/straight_lines1.jpg')

    print(image.shape)
    y_max, x_max = image.shape[0], image.shape[1]
    #                  top_left    top_right  bot_left    bot_right
    src = np.float32([[580, 460], [700, 460], [300, 660], [1000, 660]])
    dst = np.float32([[400, 460], [900, 460], [400, 660], [900, 660]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    if DEBUG:
        warped = cv2.warpPerspective(image, M, (x_max, y_max), flags=cv2.INTER_LINEAR)

        image2 = mpimg.imread('test_images/straight_lines2.jpg')
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
        #plt.show()

    return M, (x_max, y_max)


def pipeline(image, cameraMatrix, distortionCoeffs, transformationMatrix, image_size):

    ## Apply a distortion correction to raw images.
    # first undistort all images to minimize camera effects on the image
    image = undistortImage(image, cameraMatrix, distortionCoeffs)

    ## Use color transforms, gradients, etc., to create a thresholded binary image.
    #TODO crop image first, to save computation power
    binary_image = createThresholdedBinaryImage(image)

    #==========
    #FIXME remove this suff
    #f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 9))
    #f.tight_layout()

    #ax1.imshow(gray_combined)
    #ax1.set_title('gray_combined', fontsize=15)

    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #plt.show()
    #==========

    ## Apply a perspective transform to rectify binary image ("birds-eye view").
    warped = cv2.warpPerspective(binary_image, transformationMatrix, image_size, flags=cv2.INTER_LINEAR)


def main():
    ## Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    ret, cameraMatrix, distortionCoeffs, rvecs, tvecs = calibrateCamera()

    # calculate transformation matrix
    transformationMatrix, image_size = calcPerspactiveTransformMatrix()

    # Make a list of calibration images
    images = glob.glob('test_images/test*.jpg')

    for index, filename in enumerate(images):
        image = mpimg.imread(filename)
        #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if DEBUG:
            print("run pipeline on {}".format(filename))

        pipeline(image, cameraMatrix, distortionCoeffs, transformationMatrix, image_size)

if __name__ == "__main__":
    main()

