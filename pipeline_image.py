import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def undistort(img, dist_pickle):
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    img_size = (img.shape[1], img.shape[0])
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def binary(img, s_thresh=(0, 255), mag_thresh=(0, 255), g_thresh=(0, np.pi/2), l_thresh=(0, 255), sobel_kernel=11):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Convert to grayscale
    gray = hls[:,:,1]
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)


    cond = ( \
                    ((gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1]))  |\
                    ((s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])) \
                ) &\
              ((grad_direction >= g_thresh[0]) & (grad_direction <= g_thresh[1])) &\
              ((gray >= l_thresh[0]) & (gray <= l_thresh[1]))

    binary_output = np.zeros_like(s_channel)
    binary_output[cond] = 1

    #320, 960
    mask = np.zeros_like(s_channel)
    margin = 200
    vertices = np.array([[(320-margin,img.shape[0]),(320-margin, 0), (320+margin, 0), (320+margin,img.shape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 1)

    vertices = np.array([[(960-margin,img.shape[0]),(960-margin, 0), (960+margin, 0), (960+margin,img.shape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 1)

    binary_output = cv2.bitwise_and(binary_output, mask)
    color_filter_img  = color_filter(img)
    #return binary_output
    return color_filter_img

def perspective(undistort_img, src, dst):
    img_size = (undistort_img.shape[1], undistort_img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undistort_img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def find_lane_pixels(binary_warped, convolve_window_width=50):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    convolve_window = np.ones(convolve_window_width)
    leftx_base = np.argmax(np.convolve(convolve_window, histogram[:midpoint])) - convolve_window_width/2
    rightx_base = np.argmax(np.convolve(convolve_window, histogram[midpoint:])) + midpoint - convolve_window_width/2

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
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
    w_left = []
    w_right = []

    continue_bad_left, continue_bad_right = 0, 0
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        #print "window", window, leftx_current, rightx_current
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
                (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
                (win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]


        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix and continue_bad_left < 3:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            w_current = np.zeros(len(nonzerox[good_left_inds]))
            w_current[:] = 1000.0 / len(w_current)
            w_left = np.append(w_left, w_current)
            left_lane_inds.append(good_left_inds)
            continue_bad_left = 0
        else:
            continue_bad_left += 1

        if len(good_right_inds) > minpix and continue_bad_right < 3:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            w_current = np.zeros(len(nonzerox[good_right_inds]))
            w_current[:] = 1000.0 / len(w_current)
            w_right = np.append(w_right, w_current)
            right_lane_inds.append(good_right_inds)
            continue_bad_right = 0
        else:
            continue_bad_right += 1
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        if len(left_lane_inds) > 0:
            left_lane_inds = np.concatenate(left_lane_inds)
        if len(right_lane_inds) > 0:
            right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print "processerr"
        print "left", left_lane_inds
        print "right", right_lane_inds
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each using `np.polyfit`
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fit, right_fit = (0, 0, 0), (0, 0, 0)
    left_fitx, right_fitx = [], []
    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2, w=w_left)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        #left_fit, residuals, rank, singular_values, rcond = np.polyfit(lefty, leftx, 1, w=w_left, full=True)
        #print left_fit, residuals
    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2, w=w_right)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fitx, ploty, right_fitx, ploty


def cal_curvature_and_offset(left_fitx, right_fitx, ploty):

    ym_per_pix = 30.0/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = np.max(ploty)
    # Calculation of R_curve (radius of curvature)
    left_curverad, right_curverad = 0, 0
    if len(left_fitx) != 0:
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    if len(right_fitx) != 0:
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


    lane_center = (right_fitx[-1] + left_fitx[-1])/2
    center_offset_pixels = abs(len(right_fitx) - lane_center)
    offset = xm_per_pix*center_offset_pixels

    return (left_curverad + right_curverad) * 1.0 / 2, offset


def map_lane(img, src, dst, left_fitx, right_fitx, ploty, width=10, alpha=0.7):

    output = np.zeros_like(img)
    i_ploty = ploty.astype(int)

    if len(left_fitx) > 0:
        i_leftx = left_fitx.astype(int)
    if len(right_fitx) > 0:
        i_rightx = right_fitx.astype(int)


    for i in range(0-width, width, 1):
        if len(left_fitx) > 0:
            y_arr, x_arr = [], []
            for idx in range(len(left_fitx)):
                v = i_leftx[idx]
                if v + i < img.shape[1] and v+ i >= 0:
                    #print "xxx", i_ploty[i], v, v + i
                    y_arr.append(i_ploty[idx])
                    x_arr.append(v + i)
            output[y_arr, x_arr] = [255, 0, 0]
        if len(right_fitx) > 0:
            y_arr, x_arr = [], []
            for idx in range(len(right_fitx)):
                v = i_rightx[idx]
                if v + i < img.shape[1] and v + i >= 0:
                    y_arr.append(i_ploty[idx])
                    x_arr.append(v + i)
            output[y_arr, x_arr] = [255, 0, 0]

    output = perspective(output, dst, src)

    output = cv2.addWeighted(img, alpha, output, 2.0, 0.0);

    curvature, offset = cal_curvature_and_offset(left_fitx, right_fitx, ploty)
    curvature_string = "curvature: " + str(curvature) + " m"
    offset_string = "offset: " + str(offset) + " m"
    cv2.putText(output, curvature_string , (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
    cv2.putText(output, offset_string, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)

    return output

def color_filter(image):
    def bin_it(image, threshold):
        output_bin = np.zeros_like(image)
        output_bin[(image >= threshold[0]) & (image <= threshold[1])]=1
        return output_bin

    # convert image to hls colour space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

    # binary threshold values
    bin_thresh = [20, 255]

    # rgb thresholding for yellow
    lower = np.array([225,180,0],dtype = "uint8")
    upper = np.array([255, 255, 170],dtype = "uint8")
    mask = cv2.inRange(image, lower, upper)
    rgb_y = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
    rgb_y = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
    rgb_y = bin_it(rgb_y, bin_thresh)


    # rgb thresholding for white (best)
    lower = np.array([100,100,200],dtype = "uint8")
    upper = np.array([255, 255, 255],dtype = "uint8")
    #mask = cv2.inRange(lighter_rgb, lower, upper)
    #rgb_w = cv2.bitwise_and(lighter_rgb, lighter_rgb, mask = mask).astype(np.uint8)
    mask = cv2.inRange(image, lower, upper)
    rgb_w = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
    rgb_w = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2GRAY)
    rgb_w = bin_it(rgb_w, bin_thresh)


    # hls thresholding for yellow
    lower = np.array([20,120,80],dtype = "uint8")
    upper = np.array([45, 200, 255],dtype = "uint8")
    mask = cv2.inRange(hls, lower, upper)
    hls_y = cv2.bitwise_and(image, image, mask = mask).astype(np.uint8)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_HLS2RGB)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_RGB2GRAY)
    hls_y = bin_it(hls_y, bin_thresh)

    im_bin = np.zeros_like(hls_y)
    #im_bin [(hls_y == 1)|(rgb_y==1)|(rgb_w==1)]= 1
    im_bin [(hls_y==1)|(rgb_w==1)]= 1
    return im_bin


if __name__ == "__main__":

    dist_pickle = pickle.load(open("./calibration.pickle", "rb"))
    
    #img = cv2.imread('./test_images/straight_lines1.jpg')
    img = mpimg.imread('./test_images/test5.jpg')
    #img = mpimg.imread('./mpv-shot0003.jpg')
    #img = mpimg.imread('/Users/luolei/Desktop/mpv-shot0025.jpg')
    undistort_img = undistort(img, dist_pickle)
    #binary_img = binary(undistort_img, s_thresh=(80, 255), g_thresh=(0.7, 1.3), mag_thresh=(50, 100))
    
    img_size = (undistort_img.shape[1], undistort_img.shape[0])

    src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
                [((img_size[0] / 6) - 10), img_size[1]],
                [(img_size[0] * 5 / 6) + 60, img_size[1]],
                [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
            [[(img_size[0] / 4), 0],
                [(img_size[0] / 4), img_size[1]],
                [(img_size[0] * 3 / 4), img_size[1]],
                [(img_size[0] * 3 / 4), 0]])
    warped_img = perspective(undistort_img, src, dst)
    
    #warped_binary_img = binary(warped_img, s_thresh=(50, 255), mag_thresh=(10, 255), g_thresh=(0.7, 1.3), l_thresh=(256, 0))
    warped_binary_img = binary(warped_img, s_thresh=(50, 255), mag_thresh=(10, 255), g_thresh=(0.7, 1.3), l_thresh=(50, 255))
    
    lane_img, left_fitx, ploty, right_fitx, ploty = find_lane_pixels(warped_binary_img)

    output = map_lane(img, src, dst, left_fitx, right_fitx, ploty)
    
    plt.imshow(output)
    """
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,7))
    ax1.imshow(img)
    ax2.imshow(warped_img)
    ax3.imshow(warped_binary_img, cmap="gray")
    # Plots the left and right polynomials on the lane lines
    ax4.imshow(lane_img)
    #ax4.plot(left_fitx, ploty, color='yellow')
    #ax4.plot(right_fitx, ploty, color='yellow')
    ax5.imshow(output)
    ax2.set_title('Processed Image', fontsize=30)
    """
    plt.show()

