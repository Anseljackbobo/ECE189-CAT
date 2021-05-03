import cv2
import numpy as np
from numba import cuda
from piUart import *
import imutils
def getMask(color_image):
    greenLower = (29, 60, 20)
    greenUpper = (90, 255, 255)
    frame = color_image
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    return_x = -1
    return_y = -1

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
	# it to compute the minimum enclosing circle and
	# centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	# only proceed if the radius meets a minimum size
        if radius > 10:
            return_x, return_y = center

    return mask, return_x, return_y

@cuda.jit
def gpu_boolconverter_helper(input_image, result_image):
    y_val = (cuda.blockIdx.x)//320*12+cuda.threadIdx.x
    x_val = (cuda.blockIdx.x)%320
    if(input_image[y_val,x_val]):
        result_image[y_val,x_val] = 1
    else:
        result_image[y_val,x_val] = 0
        
def gpu_boolmask_filter(maskImage):
    maskImage_device = cuda.to_device(maskImage)
    gpu_result = np.zeros((240, 320), dtype = np.bool)
    gpu_result_device = cuda.to_device(gpu_result)
    blocks_per_grid = 6400
    threads_per_block = 12
    gpu_boolconverter_helper[blocks_per_grid, threads_per_block](maskImage_device, gpu_result_device)
    cuda.synchronize()
    gpu_result = gpu_result_device.copy_to_host()
    return gpu_result

@cuda.jit
def gpu_mask_filter_helper(input_image, result_image):
    y_val = (cuda.blockIdx.x)//320*12+cuda.threadIdx.x
    x_val = (cuda.blockIdx.x)%320
    
    temp_counter = 0
    if(y_val >=5 and x_val>=5 and y_val<=234 and x_val!=314):
        temp_counter = (input_image[y_val-5, x_val-5]+input_image[y_val-5, x_val-4]+input_image[y_val-5, x_val-3]+input_image[y_val-5, x_val-2]+input_image[y_val-5, x_val-1]+input_image[y_val-5, x_val]
                       +input_image[y_val-5, x_val+5]+input_image[y_val-5, x_val+4]+input_image[y_val-5, x_val+3]+input_image[y_val-5, x_val+2]+input_image[y_val-5, x_val+1]
                       +input_image[y_val-4, x_val+5]+input_image[y_val-3, x_val+5]+input_image[y_val-2, x_val+5]+input_image[y_val-1, x_val+5]+input_image[y_val, x_val+5]+input_image[y_val+1, x_val+5]
                       +input_image[y_val+2, x_val+5]+input_image[y_val+3, x_val+5]+input_image[y_val+4, x_val+5]+input_image[y_val+5, x_val+5]+input_image[y_val+5, x_val+4]+input_image[y_val+5, x_val+3]
                    +input_image[y_val+5, x_val+2]+input_image[y_val+5, x_val+1]+input_image[y_val+5, x_val]+input_image[y_val+5, x_val-1]+input_image[y_val+5, x_val-2]+input_image[y_val+5, x_val-3]+input_image[y_val+5, x_val-4]+input_image[y_val+5, x_val-5]+input_image[y_val+4, x_val-5]
                    +input_image[y_val+3, x_val-5]+input_image[y_val+2, x_val-5]+input_image[y_val+1, x_val-5]+input_image[y_val, x_val-5]+input_image[y_val-1, x_val-5]+input_image[y_val-2, x_val-5]+input_image[y_val-3, x_val-5]+input_image[y_val-4, x_val-5])
        if(temp_counter<=21):
            result_image[y_val,x_val] = 0
        else:
            result_image[y_val,x_val] = input_image[y_val,x_val]
#     debug_mat[y_val,x_val]= temp_counter
            
       
        

def gpu_mask_filter(maskImage):
    maskImage_device = cuda.to_device(maskImage)
    gpu_result = np.zeros((240, 320), dtype = np.float32)
    gpu_result_device = cuda.to_device(gpu_result)
    
    
#     debug_mat = np.zeros((240, 320), dtype = np.float32)
#     debug_mat_device = cuda.to_device(debug_mat)
    blocks_per_grid = 6400
    threads_per_block = 12
    gpu_mask_filter_helper[blocks_per_grid, threads_per_block](maskImage_device, gpu_result_device)
    cuda.synchronize()
    gpu_result = gpu_result_device.copy_to_host()
    return gpu_result

@cuda.jit
def gpu_set_noise_negative_helper(depth_map, result):
    y_val = (cuda.blockIdx.x)//320*12+cuda.threadIdx.x
    x_val = (cuda.blockIdx.x)%320
    if(depth_map[y_val, x_val] < 0.1):
        result[y_val, x_val] = -999.0
    else:
        result[y_val, x_val] = depth_map[y_val, x_val]
def gpu_set_noise_negative(depth_map):
    depth_map_device = cuda.to_device(depth_map)
    gpu_result = np.zeros((240, 320), dtype = np.float32)
    gpu_result_device = cuda.to_device(gpu_result)
    blocks_per_grid = 6400
    threads_per_block = 12
    gpu_set_noise_negative_helper[blocks_per_grid, threads_per_block](depth_map_device, gpu_result_device)
    cuda.synchronize()
    gpu_result = gpu_result_device.copy_to_host()
    return gpu_result



@cuda.jit
def gpu_worker_ver_2(in_bound_non_zero_y, in_bound_non_zero_x,out_bound_non_zero_y, out_bound_non_zero_x, distance_map, result):
    in_index = cuda.blockIdx.x
    out_index = cuda.threadIdx.x
    
    in_y = in_bound_non_zero_y[in_index]
    in_x = in_bound_non_zero_x[in_index]
    out_y = out_bound_non_zero_y[out_index]
    out_x = out_bound_non_zero_x[out_index]
    
    in_depth = distance_map[in_y, in_x]
    out_depth = distance_map[out_y, out_x]
    
    if(in_depth > 0.05):
        if(out_depth > 0.05):
            if(abs(in_depth - out_depth) < 0.05):
                result[in_index, out_index] = (out_y-in_y)**2 + (out_x-in_x)**2
            else:
                result[in_index, out_index] = 9999.0
        else:
            result[in_index, out_index] = 9999.0
    else:
        result[in_index, out_index] = 9999.0
        
    
    
    
    
def gpu_get_depth_matrix_ver2(in_bound_non_zero_tuple, out_bound_non_zero_tuple, distance_map):

    
    in_bound_non_zero_y = np.ascontiguousarray(in_bound_non_zero_tuple[0])
    in_bound_non_zero_x = np.ascontiguousarray(in_bound_non_zero_tuple[1])
    out_bound_non_zero_y = np.ascontiguousarray(out_bound_non_zero_tuple[0])
    out_bound_non_zero_x = np.ascontiguousarray(out_bound_non_zero_tuple[1])

    in_bound_non_zero_y_device = cuda.to_device(in_bound_non_zero_y)
    in_bound_non_zero_x_device = cuda.to_device(in_bound_non_zero_x)
    out_bound_non_zero_y_device = cuda.to_device(out_bound_non_zero_y)
    out_bound_non_zero_x_device = cuda.to_device(out_bound_non_zero_x)

    distance_map_device = cuda.to_device(distance_map)
    
    num_of_in = len(in_bound_non_zero_tuple[0])
    num_of_out = len(out_bound_non_zero_tuple[0])

    
    gpu_result = np.zeros((num_of_in, num_of_out), dtype = np.float32)
    gpu_result_device = cuda.to_device(gpu_result)
    
    blocks_per_grid = num_of_in
    threads_per_block = num_of_out
    try:
        gpu_worker_ver_2[blocks_per_grid, threads_per_block](in_bound_non_zero_y_device, in_bound_non_zero_x_device, out_bound_non_zero_y_device, out_bound_non_zero_x_device, distance_map_device, gpu_result_device)
        cuda.synchronize()
        gpu_result = gpu_result_device.copy_to_host()
    except:
        np.save('in_bound_non_zero_tuple', in_bound_non_zero_tuple)
        np.save('out_bound_non_zero_tuple', out_bound_non_zero_tuple)
        np.save('distance_map', distance_map)
        print("warning: skipped a frame in gpu_get_depth_matrix_ver2!")
    return gpu_result



@cuda.jit
def gpu_pad_helper(input_image, result):
    y_val = (cuda.blockIdx.x)//320*12+cuda.threadIdx.x
    x_val = (cuda.blockIdx.x)%320
#     idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if(y_val!=0 and x_val!=0 and y_val!=239 and x_val!=319):
        if(result[y_val, x_val] == 0):
            if(input_image[y_val-1, x_val-1]):
                result[y_val, x_val] = 1
            elif(input_image[y_val-1, x_val]):
                result[y_val, x_val] = 1
            elif(input_image[y_val-1, x_val+1]):
                result[y_val, x_val] = 1
            elif(input_image[y_val, x_val-1]):
                result[y_val, x_val] = 1
            elif(input_image[y_val, x_val]):
                result[y_val, x_val] = 1
            elif(input_image[y_val, x_val+1]):
                result[y_val, x_val] = 1
            elif(input_image[y_val+1, x_val-1]):
                result[y_val, x_val] = 1
            elif(input_image[y_val+1, x_val]):
                result[y_val, x_val] = 1
            elif(input_image[y_val+1, x_val+1]):
                result[y_val, x_val] = 1

def gpu_padding(maskImage):
    maskImage_device = cuda.to_device(maskImage)
    gpu_result = np.zeros((240, 320), dtype = np.float32)
    gpu_result_device = cuda.to_device(gpu_result)
    blocks_per_grid = 6400
    threads_per_block = 12
    gpu_pad_helper[blocks_per_grid, threads_per_block](maskImage_device, gpu_result_device)
    cuda.synchronize()
    gpu_result = gpu_result_device.copy_to_host()
    return gpu_result

def gpu_padding_device(maskImage, iterator_number):
    maskImage_device = cuda.to_device(maskImage)
    
    blocks_per_grid = 6400
    threads_per_block = 12
    
    for i in range(iterator_number):
        gpu_result = np.zeros((240, 320), dtype = np.float32)
        gpu_result_device = cuda.to_device(gpu_result)
        gpu_pad_helper[blocks_per_grid, threads_per_block](maskImage_device, gpu_result_device)
        maskImage_device = gpu_result_device
    cuda.synchronize()
    return gpu_result_device





@cuda.jit
def gpu_pad_inverse_helper(input_image, result):
    y_val = (cuda.blockIdx.x)//320*12+cuda.threadIdx.x
    x_val = (cuda.blockIdx.x)%320
#     idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if(y_val!=0 and x_val!=0 and y_val!=239 and x_val!=319):
        if(result[y_val, x_val] == 1):
            if(input_image[y_val-1, x_val-1]==0):
                result[y_val, x_val] = 0
            elif(input_image[y_val-1, x_val]==0):
                result[y_val, x_val] = 0
            elif(input_image[y_val-1, x_val+1]==0):
                result[y_val, x_val] = 0
            elif(input_image[y_val, x_val-1]==0):
                result[y_val, x_val] = 0
            elif(input_image[y_val, x_val]==0):
                result[y_val, x_val] = 0
            elif(input_image[y_val, x_val+1]==0):
                result[y_val, x_val] = 0
            elif(input_image[y_val+1, x_val-1]==0):
                result[y_val, x_val] = 0
            elif(input_image[y_val+1, x_val]==0):
                result[y_val, x_val] = 0
            elif(input_image[y_val+1, x_val+1]==0):
                result[y_val, x_val] = 0

def gpu_padding_inverse(maskImage):
    maskImage_device = cuda.to_device(maskImage)
    gpu_result = np.ones((240, 320), dtype = np.float32)
    gpu_result_device = cuda.to_device(gpu_result)
    blocks_per_grid = 6400
    threads_per_block = 12
    gpu_pad_inverse_helper[blocks_per_grid, threads_per_block](maskImage_device, gpu_result_device)
    cuda.synchronize()
    gpu_result = gpu_result_device.copy_to_host()
    return gpu_result

def gpu_padding_inverse_device(maskImage, iterator_number):
    maskImage_device = cuda.to_device(maskImage)
    
    blocks_per_grid = 6400
    threads_per_block = 12
    
    for i in range(iterator_number):
        gpu_result = np.ones((240, 320), dtype = np.float32)
        gpu_result_device = cuda.to_device(gpu_result)
        gpu_pad_inverse_helper[blocks_per_grid, threads_per_block](maskImage_device, gpu_result_device)
        maskImage_device = gpu_result_device
    cuda.synchronize()
    return gpu_result_device


@cuda.jit
def gpu_minus_helper(input_image_minuend, input_image_subtracter, result):
    y_val = (cuda.blockIdx.x)//320*12+cuda.threadIdx.x
    x_val = (cuda.blockIdx.x)%320
    result[y_val, x_val] = input_image_minuend[y_val, x_val] - input_image_subtracter[y_val, x_val]
    

def gpu_minus(input_image_minuend, input_image_subtracter):
    input_image_minuend_device = cuda.to_device(input_image_minuend)
    input_image_subtracter_device = cuda.to_device(input_image_subtracter)
    
    gpu_result = np.zeros((240, 320), dtype = np.float32)
    gpu_result_device = cuda.to_device(gpu_result)
    
    blocks_per_grid = 6400
    threads_per_block = 12
    gpu_minus_helper[blocks_per_grid, threads_per_block](input_image_minuend_device, input_image_subtracter_device, gpu_result_device)
    cuda.synchronize()
    gpu_result = gpu_result_device.copy_to_host()
    return gpu_result

def gpu_minus_device(input_image_minuend_device, input_image_subtracter_device):
    gpu_result = np.zeros((240, 320), dtype = np.float32)
    gpu_result_device = cuda.to_device(gpu_result)
    
    blocks_per_grid = 6400
    threads_per_block = 12
    gpu_minus_helper[blocks_per_grid, threads_per_block](input_image_minuend_device, input_image_subtracter_device, gpu_result_device)
    cuda.synchronize()
    return gpu_result_device


@cuda.jit
def gpu_stack_helper(input_image_one, input_image_two, result):
    y_val = (cuda.blockIdx.x)//320*12+cuda.threadIdx.x
    x_val = (cuda.blockIdx.x)%320
    
    if(input_image_two[y_val, x_val] > 0):
        result[y_val, x_val, 0] = 255
        result[y_val, x_val, 1] = 255
        result[y_val, x_val, 2] = 255
    else:
        result[y_val, x_val, 0] = input_image_one[y_val, x_val,0]
        result[y_val, x_val, 1] = input_image_one[y_val, x_val,1]
        result[y_val, x_val, 2] = input_image_one[y_val, x_val,2]
def gpu_stack(input_image_one, input_image_two):
    input_image_one_device = cuda.to_device(input_image_one)
    input_image_two_device = cuda.to_device(input_image_two)
    
    gpu_result = np.zeros((240, 320, 3), dtype = np.float32)
    gpu_result_device = cuda.to_device(gpu_result)
    
    blocks_per_grid = 6400
    threads_per_block = 12
    gpu_stack_helper[blocks_per_grid, threads_per_block](input_image_one_device, input_image_two_device, gpu_result_device)
    cuda.synchronize()
    gpu_result = gpu_result_device.copy_to_host()
    return gpu_result


def tracking(color_image, x, y, ted, trcf):
    trcf.value=0
    if True:

        cx = int(x)
        cy = int(y)
        if (cx < 220/2):
            send_to_UART(ted, "L\n")
            print("L")
            #frame = cv2.putText(frame, 'left', org, font, fontScale, color, thickness, cv2.LINE_AA) 
            received = read_byte_UART(ted)
            print(received)
            received = read_byte_UART(ted)
            print(received)

        elif cx > 420/2:
            send_to_UART(ted, "R\n")
            print("R")
            #frame = cv2.putText(frame, 'right', org, font, fontScale, color, thickness, cv2.LINE_AA) 
            received = read_byte_UART(ted)
            print(received)
            received = read_byte_UART(ted)
            print(received)
        
        if cy < 140/2:
            send_to_UART(ted, "U\n")
            print("U")
            # frame = cv2.putText(frame, 'up', org2, font, fontScale, color, thickness, cv2.LINE_AA) 
            received = read_byte_UART(ted)
            print(received)
            received = read_byte_UART(ted)
            print(received)

        elif cy > 240/2:
            send_to_UART(ted, "D\n")
            print("D")
            # frame = cv2.putText(frame, 'down', org2, font, fontScale, color, thickness, cv2.LINE_AA) 
            received = read_byte_UART(ted)
            print(received)
            
            received = read_byte_UART(ted)
            print(received)
             
    trcf.value=1