import cv2
import numpy as np
from numba import cuda

def worker(lowerBound, upperBound, in_bound_non_zero_tuple, out_bound_non_zero_tuple, distance_map, pixel_diff_in_bound, in_position_y_return, in_position_x_return, out_position_y_return, out_position_x_return):
    # pixel_diff_in_bound is return_dict
    for in_index in range(lowerBound, upperBound):
        if(in_index%8 == 0):
            in_y = in_bound_non_zero_tuple[0][in_index]
            in_x = in_bound_non_zero_tuple[1][in_index]
            pixel_diff_in_bound[in_index] = float('inf')

            if(distance_map[in_y, in_x]>0):
                for out_index in range(len(out_bound_non_zero_tuple[0])):
                    if(out_index%8 == 0):
                        out_y = out_bound_non_zero_tuple[0][out_index]
                        out_x = out_bound_non_zero_tuple[1][out_index]

                        if(distance_map[out_y, out_x]>0):
                            if(np.linalg.norm(distance_map[out_y, out_x]-distance_map[in_y, in_x]) < 0.05):
                                dist = np.linalg.norm(np.asarray([out_y, out_x]) - np.asarray([in_y, in_x]))
                                if(dist<pixel_diff_in_bound[in_index]):
                                    pixel_diff_in_bound[in_index] = dist
                                    in_position_y_return[in_index] = in_y
                                    in_position_x_return[in_index] = in_x
                                    out_position_y_return[in_index] = out_y
                                    out_position_x_return[in_index] = out_x


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
    return mask



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