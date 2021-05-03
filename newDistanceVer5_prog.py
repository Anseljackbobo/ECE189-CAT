import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
from mpl_toolkits.mplot3d import Axes3D
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time
from numba import cuda
import imutils
from newDistanceVer5_func import *
from piUart import *
import sys
import multiprocessing
print("Environment Ready")
ted = init_UART()
print("uart ready")



# displayText configration
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
org1 = (50,100)
org2 = (50,150)
org3 = (50, 200)
fontScale = 1
color = (255, 255, 0)
thickness = 2


out_bound_pad_layer_parameter = 20
in_bound_pad_layer_parameter = 6

pipe = rs.pipeline()
cfg = rs.config()
# cfg.enable_device_from_file("../object_detection.bag")
# profile = pipe.start(cfg)

width = 640
height = 480
fps = 30


cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

profile = pipe.start(cfg)
# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# get something to check if tracking program is finished.
trcf=multiprocessing.Value('i',1)


# Skip 15 first frames to give the Auto-Exposure time to adjust
for x in range(15):
    pipe.wait_for_frames()

while True:
    try:
        start = time.time()
        frames = pipe.wait_for_frames()
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        aligned_depth_frame = frames.get_depth_frame()


        color_image = np.asanyarray(frames.get_color_frame().get_data())
        color_image=cv2.resize(color_image,(320,240),interpolation = cv2.INTER_AREA)
        
        maskImage, mask_center_x, mask_center_y = getMask(color_image)

        # post process depth image
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)

        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 1)
        spatial.set_option(rs.option.filter_smooth_delta, 50)

        temporal = rs.temporal_filter()

        hole_filling = rs.hole_filling_filter()

        new_depth_frame = aligned_depth_frame
        new_depth_frame = decimation.process(new_depth_frame)
        new_depth_frame = depth_to_disparity.process(new_depth_frame)
        new_depth_frame = spatial.process(new_depth_frame)
        new_depth_frame = temporal.process(new_depth_frame)
        new_depth_frame = disparity_to_depth.process(new_depth_frame)
        new_depth_frame = hole_filling.process(new_depth_frame)
        depth_image_new = np.asanyarray(new_depth_frame.get_data())

        depth = depth_image_new.astype(np.float32)
        distance = depth * depth_scale

        # post process maskImage, save in temp_filted_mask
        mask_2 = gpu_boolmask_filter(maskImage)
        temp_filted_mask = gpu_mask_filter(mask_2)

        # get outer bound, save in out_bound_line_result_img
        pad_8 = temp_filted_mask
        pad_8 = gpu_padding_device(pad_8, out_bound_pad_layer_parameter)
        pad_9 = pad_8
        pad_9 = gpu_padding_device(pad_9, 1)

        out_bound_line_result = gpu_minus_device(pad_9, pad_8)
        out_bound_line_result_img = out_bound_line_result.copy_to_host()

        # get inner bound
        temp_inverse_pad_img1 = gpu_padding_inverse_device(temp_filted_mask, in_bound_pad_layer_parameter)
        temp_inverse_pad_img2 = gpu_padding_inverse_device(temp_inverse_pad_img1, 1)
        in_bound_line_result_img = gpu_minus_device(temp_inverse_pad_img1, temp_inverse_pad_img2).copy_to_host()


        # get position of element in bounds
        in_bound_non_zero_tuple = np.nonzero(in_bound_line_result_img)
        out_bound_non_zero_tuple = np.nonzero(out_bound_line_result_img)

        # check how many points does in and out bound have
        num_of_in = len(in_bound_non_zero_tuple[0])
        num_of_out = len(out_bound_non_zero_tuple[0])

        image_to_display = gpu_stack(color_image, out_bound_line_result_img+in_bound_line_result_img).astype(np.uint8)


        # calculate distance
        if(num_of_in > 0 and num_of_out > 0):
            depth_matrix = gpu_get_depth_matrix_ver2(in_bound_non_zero_tuple, out_bound_non_zero_tuple, distance)
            indices_min_val = np.where(depth_matrix <= 50 + np.min(depth_matrix))
            # this is real minimal
            minimal = np.where(depth_matrix == np.min(depth_matrix))
            minimal = depth_matrix[minimal[0][0], minimal[1][0]]
            if(minimal < 1500 and len(indices_min_val[0])>5):
                for i in range(len(indices_min_val[0])):
                    in_val_y = in_bound_non_zero_tuple[0][indices_min_val[0][i]]
                    in_val_x = in_bound_non_zero_tuple[1][indices_min_val[0][i]]

                    out_val_y = out_bound_non_zero_tuple[0][indices_min_val[1][i]]
                    out_val_x = out_bound_non_zero_tuple[1][indices_min_val[1][i]]

                    temp_in_y = in_val_y
                    temp_in_x = in_val_x
                    temp_out_y = out_val_y
                    temp_out_x = out_val_x

                    image_to_display = cv2.putText(image_to_display, "is blocked", org1, font, fontScale, color, thickness, cv2.LINE_AA)
                    image_to_display = cv2.circle(image_to_display, (temp_in_x,temp_in_y), radius=3, color=(255, 0, 0), thickness=3)
                    image_to_display = cv2.circle(image_to_display, (temp_out_x,temp_out_y), radius=3, color=(0, 255, 0), thickness=3)
                    if(i>=10):
                        break
        else:
            minimal = float('inf')

        # display text of min val and fps
        image_to_display = cv2.putText(image_to_display, str(minimal), org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        image_to_display = cv2.putText(image_to_display, "fps = "+str(1/(time.time()-start)), org3, font, 
                        fontScale, color, thickness, cv2.LINE_AA)

        if(mask_center_x > 0 and mask_center_y > 0):
            cv2.circle(image_to_display, (int(mask_center_x), int(mask_center_y)), int(5),
                    (0, 255, 255), 2)
        
            # tracking algo
            if trcf.value==1:
                trc=multiprocessing.Process(target=tracking,args=(color_image, mask_center_x, mask_center_y, ted, trcf))
                trc.start()
        
        
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', image_to_display)
        # if the 'q' key is pressed, stop the loop
        
    except:
        print("warning: skipped a frame")
