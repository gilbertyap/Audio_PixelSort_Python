import argparse
import cv2
import numpy as np
import os
import re
import scipy.signal
import sys

# Some variables to assist with display, randomness, etc.
window_name = 'Output Image'
fps = 30
length = 30 # in seconds
loop_len = fps*length 
pseudo_normal = np.convolve(np.random.normal(loc=0.0, scale=0.5, size=(loop_len,1)).flatten(), np.ones(4)/4,'same')
pseudo_random = np.convolve(np.random.random((loop_len,1)).flatten(), np.ones(4)/4,'same')
pass_percent = 0.55

'''

Function name: quantizeAs6Bit

Description: Requiantizes the input color image to 6-bit per color

Input:
 - input_mat: The N x M x 3 image. Assumes BGR.

Output:
 - The 6-bit per color version of the input image.

def quantizeAs6Bit(input_mat):
    print(np.min(input_mat.flatten()))
    print(np.max(input_mat.flatten()))


    bins = np.round(np.linspace(start=0,stop=255,num=63))
    new_vals = np.zeros(shape=input_mat.shape)

    new_vals[:,:,0] = np.ndarray.astype(((np.digitize(input_mat[:,:,0], bins)+1)*4)-1,dtype='uint8')
    new_vals[:,:,1] = np.ndarray.astype(((np.digitize(input_mat[:,:,1], bins)+1)*4)-1,dtype='uint8')
    new_vals[:,:,2] = np.ndarray.astype(((np.digitize(input_mat[:,:,2], bins)+1)*4)-1,dtype='uint8')

    print(np.min(new_vals.flatten()))
    print(np.max(new_vals.flatten()))

    return np.ndarray.astype(new_vals,dtype='uint8')
'''

'''
Function name: check2ndOrdMatch

Description: Checks to see if the pixel at (x_pos, y_pos) within mat matches any of its 2nd order neighbors.

Input:
 - mat: An N x M  matrix
 - x_pos: The column position
 - y_pos: The row position

Output:
 - A boolean that says if the current pixel has a matching 2nd order neighborhood pixel.
'''
def check2ndOrdMatch(mat, x_pos, y_pos):
    current_val = mat[x_pos, y_pos]
    if mat[x_pos-1, y_pos-1] == current_val or \
    mat[x_pos, y_pos-1] == current_val or \
    mat[x_pos+1, y_pos-1] == current_val or \
    mat[x_pos-1, y_pos] == current_val or \
    mat[x_pos+1, y_pos] == current_val or \
    mat[x_pos-1, y_pos+1] == current_val or \
    mat[x_pos, y_pos+1] == current_val or \
    mat[x_pos+1, y_pos+1] == current_val:
        return True
    else:
        return False

'''
Function name: doPixelSort

Description: Does pixel sort on the input matrix mat and returns the sorted pixel mask

Input:
 - mat: An N x M x 3 matrix (3 for color, 1 for grayscale)
 - edge_mat: An N x M map describing the edges in the image
 - sort_len: The length of pixels that should be used for each column sort
 - pass_percent: The value needed to determine if that pixel should be sorted

Output:
 - A matrix that matches the dimensions of "mat" with the sorted pixels in their respective locations based on length and zero elsewhere
'''
def doPixelSort(input_mat, edge_mat, sort_len, pass_percent):
    input_size = input_mat.shape

    if((sort_len % 2) != 0):
        sort_len = sort_len - 1

    if (sort_len <= 1):
        # print("Sort length needs to at least be >1!")
        return

    sorted_pixels_mask = np.ndarray.astype((np.zeros(shape=(input_size[0], input_size[1], input_size[2]))),dtype='uint8')
    
    # Start the rand_counter in a random position
    rand_counter = int(np.random.random(1)*(loop_len-1))

    # Perform sorting on input
    for x in range(0, input_size[1]-1):
        loop_sort_len = sort_len 

        # Skip columns that don't have any threshold pixels
        if (np.sum(edge_mat[:,x]) > 0):
            # Randomize the sort length for every column. Make all lengths even to make it easier to scale

            prev_sort_start = -9999
            for y in range(0, input_size[0]-1):
                # Don't sort more than pass_percent of the threshold pixels for any given frame
                if ((edge_mat[y,x] == 1) and (pseudo_random[rand_counter] <= pass_percent)):
                    # Check that you are not going to collide into the edge of the image, shorten sort_len if needed.
                    if((y + loop_sort_len) > (input_size[0] - 1)):
                        loop_sort_len = (input_size[0] - y) - 1

                    # Skip this edge pixel if the length will end up being 0 or 1. Also prevent the resorting of pixels.
                    if (loop_sort_len < 2) and (y < (prev_sort_start + sort_len)):
                        continue

                    # Store and sort 
                    sorted_pixels = np.zeros(shape=(loop_sort_len, 1, input_size[2]))

                    sorted_pixels[:,0,0] = input_mat[y:(y + loop_sort_len), x, 0]
                    sorted_pixels[:,0,1] = input_mat[y:(y + loop_sort_len), x, 1]
                    sorted_pixels[:,0,2] = input_mat[y:(y + loop_sort_len), x, 2]

                    # Sort the pixels along their x axis
                    sorted_pixels = np.sort(sorted_pixels, axis=0)

                    # Map the pixels to their postion in the reduced image size
                    sorted_pixels_mask[y:(y + loop_sort_len), x, 0] = sorted_pixels[:,0,0]
                    sorted_pixels_mask[y:(y + loop_sort_len), x, 1] = sorted_pixels[:,0,1]
                    sorted_pixels_mask[y:(y + loop_sort_len), x, 2] = sorted_pixels[:,0,2]
                    
                    prev_sort_start = y

                # Progress the random counter
                rand_counter = (rand_counter + 1) % (loop_len-1)

    return sorted_pixels_mask

'''
The program accepts 1 required argument and 1 optional argument. The "file" argument is the file path to the image.
'''
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", default="",
                    help="Image path. The file should be placed in the same directory as this script or in a subdirectory.")

    ap.add_argument("-t", "--threshold", default="",
                    help="Optional argument. Determines how many edges are represented. An integer value from 0 to 255.")

    args = vars(ap.parse_args())

    # Verify the file path
    # TODO - add support for OS with backslashes
    if os.path.isfile(os.getcwd()+'//'+args['file']):
        img_name = os.getcwd()+'//'+args['file']
    else:
        print('File name {} is invalid!'.format(args['file']))
        sys.exit(-1)

    # Check the input of the threshold value
    if not(args['threshold'] == ''):
        m = re.match('[a-z]*', args['threshold'],re.IGNORECASE)
        if not (m.group(0) ==''):
            print('The argument for \'-t\' must be an integer value.')
            sys.exit(-2)
        
        if (float(args['threshold']) < 0 or float(args['threshold']) > 255):
            print(' The argument for \'t\' must be between 0 and 255. Smaller numbers may cause slow downs.')
            sys.exit(-2)

    # Read in the image
    base_img = cv2.imread(args['file'], flags=(cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION ))
    try:
       org_img_size = base_img.shape
    except AttributeError:
        print('Could not open file {}! Closing...'.format(img_name))
        sys.exit(-3)

    # TODO - Might be good to first increase image contrast to accentuate the edges

    # Reduce the image by the scale_factor in both directions
    scale_factor = int((base_img.shape[0] if (base_img.shape[0] < base_img.shape[1]) else base_img.shape[1])/125)

    reduced_img = cv2.resize(np.array(base_img,copy=True), dsize=(int(org_img_size[1]/scale_factor), int(org_img_size[0]/scale_factor)), interpolation=cv2.INTER_NEAREST)
    reduced_img_size = reduced_img.shape

    # Generate the edges of the image based on green channel in reduced_img. Uses Sobel method.
    edge_img_x = scipy.signal.convolve2d(np.ndarray.astype(reduced_img[:,:,1], 'float'), np.ndarray.astype(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), 'float'), 'same')
    edge_img_y = scipy.signal.convolve2d(np.ndarray.astype(reduced_img[:,:,1], 'float'), -1*np.transpose(np.ndarray.astype(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), 'float')), 'same')
    edge_img = np.abs(edge_img_x) + np.abs(edge_img_y)

    if (args['threshold'] == ''):
        edge_thresh = 35.0
    else:
        edge_thresh = float(args['threshold'])

    # Scale the edge image range and threshold it.
    edge_img = np.ndarray.astype(np.round((edge_img/np.max(edge_img.flatten())*255) > edge_thresh), dtype='uint8')

    # Set the outer border to have no edge pixel
    edge_img[0, :] = 0
    edge_img[reduced_img_size[0]-1, :] = 0
    edge_img[:, 0] = 0
    edge_img[:, reduced_img_size[1]-1] = 0

    # Remove outlier edge pixels using 2nd order neighborhoods
    for x in range(1, reduced_img_size[0]-1):
        for y in range(1, reduced_img_size[1]-1):
            if not check2ndOrdMatch(edge_img, x, y):
                edge_img[x,y] = 0

    # Scale the maximum sort length based on the long edge of e image
    sort_scale = (base_img.shape[0] if (base_img.shape[0] < base_img.shape[1]) else base_img.shape[1])*0.05

    norm_counter = int(np.random.random(1)*(loop_len-1))

    # Loop until the user presses a key while the window is active
    while True:
        disp_img = np.array(base_img,copy=True)

        sort_len = int(np.round(sort_scale*np.abs(pseudo_normal[norm_counter])))

        # Make the sort length divisible by two.
        if((sort_len % 2) != 0):
            sort_len = sort_len - 1

        # Don't need to sort anything less than 2.
        if not (sort_len <= 1):
            reduced_sort_mask = doPixelSort(reduced_img, edge_img, sort_len, pass_percent)

            # Upscale the pixel map
            upscaled_sorted_pixel_mask = cv2.resize(reduced_sort_mask, dsize=(org_img_size[1], org_img_size[0]), interpolation=cv2.INTER_NEAREST)

            # Replace the pixels
            mask_bool_table = (upscaled_sorted_pixel_mask[:,:,0] > 0) | (upscaled_sorted_pixel_mask[:,:,1] > 0) | (upscaled_sorted_pixel_mask[:,:,2] > 0)
        
            np.copyto(dst=disp_img[:,:,0], src=upscaled_sorted_pixel_mask[:,:,0], where=(mask_bool_table == True))
            np.copyto(dst=disp_img[:,:,1], src=upscaled_sorted_pixel_mask[:,:,1], where=(mask_bool_table == True))
            np.copyto(dst=disp_img[:,:,2], src=upscaled_sorted_pixel_mask[:,:,2], where=(mask_bool_table == True))

            # Display the image
            cv2.imshow(window_name, disp_img)

            # Allow person to exit by pressing a key. Display at the same rate as the frame rate.
            if not (cv2.waitKey(int(np.round(1/fps*1000))) == -1):
                break

        norm_counter = (norm_counter + 1) % (loop_len-1)

    cv2.destroyWindow(window_name)