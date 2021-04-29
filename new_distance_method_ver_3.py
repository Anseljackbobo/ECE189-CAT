import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
from mpl_toolkits.mplot3d import Axes3D
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time
from numba import cuda
import imutils
from new_distance_function_ver_3 import *
from piUart import *
import multiprocessing
print("Environment Ready")

# import time
import serial 
print("UART Program")

ted = init_UART()
# Wait a second to let the port initialize
time.sleep(1)


# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
org1 = (50,100)
org2 = (50,150)
org3 = (50, 200)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 255, 0)
  
# Line thickness of 2 px
thickness = 2


# config camera

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

last_min_val = 0
trcf=multiprocessing.Value('i',1)


while True:
    start = time.time()
    frames = pipe.wait_for_frames()
    align = rs.align(rs.stream.color)
    frames = align.process(frames)
    aligned_depth_frame = frames.get_depth_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(frames.get_color_frame().get_data())

    # resize 3 frames
    color_image=cv2.resize(color_image,(320,240),interpolation = cv2.INTER_AREA)
    color_image=cv2.blur(color_image,(2,2))
    depth_image=cv2.resize(depth_image,(320,240),interpolation = cv2.INTER_AREA)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    maskImage, mask_center_x, mask_center_y = getMask(color_image)

    # show 3 images
    # fig, ax = plt.subplots(2,2)
    # ax[0,0].imshow(color_image)
    # ax[0,1].imshow(depth_colormap)
    # ax[1,0].imshow(maskImage)

    # get distance in meter
    depth = depth_image.astype(np.float32)
    distance = depth * depth_scale
    depth_image=cv2.blur(depth_image,(3,3))

    distance = gpu_set_noise_negative(distance)

    # save 
    # cv2.imwrite("filename.jpg", color_image)

    # draw inner bound
    img1 = gpu_padding_inverse_device(maskImage, 12)
    img2 = gpu_padding_inverse_device(img1, 1)
    internal_pad_result_img = gpu_minus_device(img1, img2)
    internal_pad_result_img = internal_pad_result_img.copy_to_host()

    # draw out bound
    pad_8 = maskImage
    pad_8 = gpu_padding_device(pad_8, 15)

    pad_9 = pad_8
    pad_9 = gpu_padding_device(pad_9, 1)

    warning_line_result = gpu_minus_device(pad_9, pad_8)
    warning_line_result_img = warning_line_result.copy_to_host()

    # for feed in worker
    in_bound_non_zero_tuple = np.nonzero(internal_pad_result_img)
    out_bound_non_zero_tuple = np.nonzero(warning_line_result_img)

    # for multi processing
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()

    # in_position_y_return = multiprocessing.Manager().dict()
    # in_position_x_return = multiprocessing.Manager().dict()
    # out_position_y_return = multiprocessing.Manager().dict()
    # out_position_x_return = multiprocessing.Manager().dict()



    # calculate distance
    num_of_in = len(in_bound_non_zero_tuple[0])
    num_of_out = len(out_bound_non_zero_tuple[0])



    image_to_display = gpu_stack(depth_colormap, (internal_pad_result_img+warning_line_result_img)).astype(np.uint8)
    if(num_of_in > 0 and num_of_out > 0):
        
        depth_matrix = gpu_get_depth_matrix_ver1(in_bound_non_zero_tuple, out_bound_non_zero_tuple, distance)

        indices_min_val = np.where(depth_matrix == np.min(depth_matrix))
        
        min_value = depth_matrix[indices_min_val[0][0], indices_min_val[1][0]]
        in_val_y = in_bound_non_zero_tuple[0][indices_min_val[0][0]]
        in_val_x = in_bound_non_zero_tuple[1][indices_min_val[0][0]]

        out_val_y = out_bound_non_zero_tuple[0][indices_min_val[1][0]]
        out_val_x = out_bound_non_zero_tuple[1][indices_min_val[1][0]]

        temp = min_value
        temp_in_y = in_val_y
        temp_in_x = in_val_x
        temp_out_y = out_val_y
        temp_out_x = out_val_x
        if(min_value < 9998 and abs(last_min_val-min_value)<50):
            image_to_display = cv2.circle(image_to_display, (temp_in_x,temp_in_y), radius=3, color=(255, 0, 0), thickness=3)
            image_to_display = cv2.circle(image_to_display, (temp_out_x,temp_out_y), radius=3, color=(0, 255, 0), thickness=3)
            
            in_display_distance = distance[temp_in_y, temp_in_x]
            out_display_distance = distance[temp_out_y, temp_out_x]
            image_to_display = cv2.putText(image_to_display, str(in_display_distance), org1, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
            image_to_display = cv2.putText(image_to_display, str(out_display_distance), org2, font, 
                        fontScale, color, thickness, cv2.LINE_AA)


    else:
        
        min_value = float('inf')

    last_min_val = min_value
    image_to_display = cv2.putText(image_to_display, str(min_value), org, font, 
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
    cv2.waitKey(1)