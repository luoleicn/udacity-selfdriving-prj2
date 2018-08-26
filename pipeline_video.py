from pipeline_image import *
from moviepy.editor import VideoFileClip
import sys

def get_avg_fit(fit, stack_fits, num_frame, margin, mag_thresh=(0.7, 1.3)):
    global lane_margin

    if lane_margin < 0:
        lane_margin = margin

    bad = False
    if margin < 0:
        bad = True
    if len(fit) == 0:
        bad = True
    if lane_margin > 0 and (margin < mag_thresh[0] * lane_margin or margin > mag_thresh[1] * lane_margin):
        bad = True

    if bad:
        out_fit = np.zeros_like(stack_fits[0])
        for x in stack_fits:
            out_fit += x
        out_fit /= len(stack_fits)
        return out_fit

    lane_margin = 0.95 * lane_margin + 0.05 * margin

    if len(stack_fits) < num_frame:
        stack_fits.append(fit)
        return fit
    else:
        stack_fits[0:num_frame-1] = stack_fits[1:num_frame]
        stack_fits[num_frame-1] = fit
        out_fit = np.zeros_like(fit)
        for x in stack_fits:
            out_fit += x
        out_fit /= num_frame
        return out_fit


def process_image(img):
    
    #img = cv2.imread('./test_images/straight_lines1.jpg')
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
    
    warped_binary_img = binary(warped_img, s_thresh=(80, 255), mag_thresh=(50, 100))
    
    lane_img, left_fitx, ploty, right_fitx, ploty = find_lane_pixels(warped_binary_img)
    

    if len(left_fitx) > 0 and len(right_fitx) > 0:
        margin = np.mean(right_fitx - left_fitx)
    else:
        margin = -1

    avg_left_fitx = get_avg_fit(left_fitx, stack_left_fits, num_frame, margin)
    avg_right_fitx = get_avg_fit(right_fitx, stack_right_fits, num_frame, margin)

    """
    print "lane_img", lane_img.shape
    plt.imshow(lane_img)
    plt.plot(left_fitx, ploty[:len(left_fitx)], color='yellow')
    plt.plot(right_fitx, ploty[:len(right_fitx)], color='yellow')
    plt.show()
    """
    
    left_curverad, right_curverad = cal_curvature(avg_left_fitx, avg_right_fitx, ploty)
    
    output = map_lane(img, src, dst, avg_left_fitx, avg_right_fitx, ploty)

    return output
    

script, input_video_fn, output_video_fn = sys.argv

dist_pickle = pickle.load(open("./calibration.pickle", "rb"))

num_frame  = 5
lane_margin = -1
stack_left_fits = []
stack_right_fits = []

output = output_video_fn
clip = VideoFileClip(input_video_fn)
white_clip = clip.fl_image(process_image) 
white_clip.write_videofile(output, audio=False)

