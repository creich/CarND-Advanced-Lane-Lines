import cv2
import numpy as np
import glob
import matplotlib.image as mpimg

import pickle

DEBUG = False
#TODO make filename a parameter
PICKLE_FILE_NAME = 'camera_calibration_data.p'

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

    # load one image even in non-DEBUG-mode to determine image.shape of the camera
    image = mpimg.imread('test_images/straight_lines1.jpg')
    if DEBUG:
        print("using image shape: {}".format(image.shape))
    y_max, x_max = image.shape[0], image.shape[1]

    if DEBUG:
        image = undistortImage(image, cameraMatrix, distortionCoeffs)

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

    return M, (x_max, y_max)



## Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
print("calibrate camera...")
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = calibrateCamera()
print("done.")

if DEBUG:
    image = mpimg.imread('test_images/straight_lines1.jpg')
    image = undistortImage(image, cameraMatrix, distortionCoeffs)
    mpimg.imsave('output_images/straight_lines1_undistorted.jpg', image)

    image = mpimg.imread('test_images/straight_lines2.jpg')
    image = undistortImage(image, cameraMatrix, distortionCoeffs)
    mpimg.imsave('output_images/straight_lines2_undistorted.jpg', image)

# calculate transformation matrix
transformation_matrix, image_size = calcPerspactiveTransformMatrix(camera_matrix, distortion_coeffs)

# thx to ajsmilutin for the snippet of 'how to use pickle'
# https://github.com/ajsmilutin/CarND-Advanced-Lane-Lines/blob/master/calibrate.py

# pickle the data and save it
calibration_data = {'camera_matrix': camera_matrix,
                    'distortion_coeffs': distortion_coeffs,
                    'transformation_matrix': transformation_matrix,
                    'image_size': image_size}
with open(PICKLE_FILE_NAME, 'wb') as f:
    pickle.dump(calibration_data, f)



