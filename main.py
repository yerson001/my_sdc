import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import calib, undistort
from threshold import get_combined_gradients, get_combined_hls, combine_grad_hls
from line import Line, get_perspective_transform, get_lane_lines_img, illustrate_driving_lane, illustrate_info_panel, illustrate_driving_lane_with_topdownview
from moviepy.editor import VideoFileClip


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#       Select desired input name/type          #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
input_type = 'image'

input_name = 'test_images/test3.jpg'
#input_name = 'test_images/line.png'
#input_name = 'test_images/calibration1.jpg'


# If input_type is `image`, select whether you'd like to save intermediate images or not. 
save_img = False

left_line = Line()
right_line = Line()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   Tune Parameters for different inputs        #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)

th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()


if __name__ == '__main__':

    # If working with images, don't use moviepy
    if input_type == 'image':
        img = cv2.imread(input_name)
        a,l,h = img.shape
        #print(a," -->",l," -->",h)
        #imge = cv2.resize(img,(400,300), interpolation=cv2.INTER_AREA)
        #cv2.imshow('img',imge)

        undist_img = undistort(img, mtx, dist)
        

        undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
        rows, cols = undist_img.shape[:2]
        #cv2.imshow('sin_distorsion_img',undist_img)
    
        combined_gradient = get_combined_gradients(undist_img, th_sobelx, th_sobely, th_mag, th_dir)
        #cv2.imshow('combined_gradient',combined_gradient)

        combined_hls = get_combined_hls(undist_img, th_h, th_l, th_s)
        #cv2.imshow('combined_hls',combined_hls)

 
        combined_result = combine_grad_hls(combined_gradient, combined_hls)
        cv2.imshow("combinet_result",combined_result)
       


        c_rows, c_cols = combined_result.shape[:2]
        s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
        s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

        src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
        dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

        warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (720, 720))


        searching_img = get_lane_lines_img(warp_img, left_line, right_line)
        cv2.imshow("searching",searching_img)
       

        w_comb_result, w_color_result = illustrate_driving_lane(searching_img, left_line, right_line)
        cv2.imshow("w_comb_result",w_comb_result)
        cv2.imshow("warp_img",w_color_result)


        # Drawing the lines back down onto the road
        color_result = cv2.warpPerspective(w_color_result, Minv, (c_cols, c_rows))
        cv2.imshow("color_result",color_result)

        comb_result = np.zeros_like(undist_img)
        comb_result[220:rows - 12, 0:cols] = color_result
        

        # Combine the result with the original image
        result = cv2.addWeighted(undist_img, 1, comb_result, 0.3, 0)

        
        cv2.imshow('result',result)
        cv2.waitKey(0)
   


