import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
from mpl_toolkits.mplot3d import Axes3D
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time
from numba import cuda
import imutils
from new_distance_function_ver_2 import *
import multiprocessing
print("Environment Ready")

# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
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



while True:
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
    depth_image=cv2.blur(depth_image,(2,2))
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    maskImage= getMask(color_image)

    # show 3 images
    # fig, ax = plt.subplots(2,2)
    # ax[0,0].imshow(color_image)
    # ax[0,1].imshow(depth_colormap)
    # ax[1,0].imshow(maskImage)

    # get distance in meter
    depth = depth_image.astype(np.float32)
    distance = depth * depth_scale

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
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    in_position_y_return = multiprocessing.Manager().dict()
    in_position_x_return = multiprocessing.Manager().dict()
    out_position_y_return = multiprocessing.Manager().dict()
    out_position_x_return = multiprocessing.Manager().dict()



    # calculate distance
    num_of_works = len(in_bound_non_zero_tuple[0])
    each_worker = num_of_works//16

    p0 = multiprocessing.Process(target=worker, args=(0,each_worker,in_bound_non_zero_tuple,out_bound_non_zero_tuple, distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p1 = multiprocessing.Process(target=worker, args=(each_worker,each_worker*2,in_bound_non_zero_tuple,out_bound_non_zero_tuple, distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p2 = multiprocessing.Process(target=worker, args=(each_worker*2,each_worker*3,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p3 = multiprocessing.Process(target=worker, args=(each_worker*3,each_worker*4,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p4 = multiprocessing.Process(target=worker, args=(each_worker*4,each_worker*5,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p5 = multiprocessing.Process(target=worker, args=(each_worker*5,each_worker*6,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p6 = multiprocessing.Process(target=worker, args=(each_worker*6,each_worker*7,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p7 = multiprocessing.Process(target=worker, args=(each_worker*7,each_worker*8,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p8 = multiprocessing.Process(target=worker, args=(each_worker*8,each_worker*9,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p9 = multiprocessing.Process(target=worker, args=(each_worker*9,each_worker*10,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p10 = multiprocessing.Process(target=worker, args=(each_worker*10,each_worker*11,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p11 = multiprocessing.Process(target=worker, args=(each_worker*11,each_worker*12,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p12 = multiprocessing.Process(target=worker, args=(each_worker*12,each_worker*13,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p13 = multiprocessing.Process(target=worker, args=(each_worker*13,each_worker*14,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p14 = multiprocessing.Process(target=worker, args=(each_worker*14,each_worker*15,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))
    p15 = multiprocessing.Process(target=worker, args=(each_worker*15,num_of_works,in_bound_non_zero_tuple,out_bound_non_zero_tuple,distance,return_dict,in_position_y_return,in_position_x_return,out_position_y_return,out_position_x_return,))

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()
    p15.start()

    p0.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    p15.join()

    temp = np.asarray(return_dict.values())
    temp_in_y = np.asarray(in_position_y_return.values())
    temp_in_x = np.asarray(in_position_x_return.values())
    temp_out_y = np.asarray(out_position_y_return.values())
    temp_out_x = np.asarray(out_position_x_return.values())

    image_to_display = gpu_stack(color_image, (internal_pad_result_img+warning_line_result_img)).astype(np.uint8)
    try:
        # min_value = np.min(temp)
        min_value_index =  np.argmin(temp)
        min_value = temp[min_value_index]
        if(True):
            temp_in_y = temp_in_y[min_value_index]
            temp_in_x = temp_in_x[min_value_index]
            temp_out_y = temp_out_y[min_value_index]
            temp_out_x = temp_out_x[min_value_index]
            image_to_display = cv2.circle(image_to_display, (temp_in_x,temp_in_y), radius=3, color=(255, 0, 0), thickness=3)
            image_to_display = cv2.circle(image_to_display, (temp_out_x,temp_out_y), radius=3, color=(0, 255, 0), thickness=3)

    except:
        min_value = float('inf')

    
    image_to_display = cv2.putText(image_to_display, str(min_value), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    
    
    cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
    cv2.imshow('RealSense', image_to_display)
    cv2.waitKey(1)