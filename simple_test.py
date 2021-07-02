import cv2
import numpy as np
import scipy.signal
import sys

# Function for checking 2nd order neighborhood of pixel to see if it matches the current value
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

# TODO - make file name entry a command line arguement
img_name = "images/RedHat_master.tif"
window_name = 'Output Image'

# Read in the image
# base_img = cv2.resize(cv2.imread(img_name, flags=(cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION )), dsize=(500,500))
base_img = cv2.imread(img_name, flags=(cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION ))


try:
   org_img_size = base_img.shape
except AttributeError:
    print('Could not open file {}! Closing...'.format(img_name))
    sys.exit(1)

# TODO - make the scale factor a command line arguement or try to find the largest factor that results in a whole number?
scale_factor = 8
reduced_img = cv2.resize(np.array(base_img,copy=True), dsize=(int(org_img_size[1]/scale_factor), int(org_img_size[0]/scale_factor)), interpolation=cv2.INTER_NEAREST)
reduced_img_size = reduced_img.shape

# The edges will be based on the green channel
# Generate the edge image based on the Sobel kernels
edge_img_x = scipy.signal.convolve2d(np.ndarray.astype(reduced_img[:,:,1], 'float'), np.ndarray.astype(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), 'float'), 'same')
edge_img_y = scipy.signal.convolve2d(np.ndarray.astype(reduced_img[:,:,1], 'float'), -1*np.transpose(np.ndarray.astype(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), 'float')), 'same')
edge_img = np.abs(edge_img_x) + np.abs(edge_img_y)

# Scale the edge image range and threshold it
edge_thresh = 50.0
edge_img = np.ndarray.astype(np.round(edge_img/np.max(edge_img.flatten())*255) > edge_thresh, dtype='uint8')

# Set the outer border to have no edge
edge_img[0, :] = 0
edge_img[reduced_img_size[0]-1, :] = 0
edge_img[:, 0] = 0
edge_img[:, reduced_img_size[1]-1] = 0

# Thin all of the edges down to one pixel and remove edges that do not have 2nd order neighbors
# Do not consider the 1-pixel border of the image
for x in range(1, reduced_img_size[0]-2):
    for y in range(1, reduced_img_size[1]-2):
        if ((edge_img[x,y] == 1) and (edge_img[x,y] == edge_img[x,y+1])) or (not check2ndOrdMatch(edge_img, x, y)):
            edge_img[x,y] = 0

# Rather than generate the random values each time, make a long loop of random values and then smooth them by running a mean filter of length 4.
# TODO - For real time, this should be moved to before frame analysis
fps = 30
length = 30 # in seconds
loop_len = fps*length
rand_counter = 0 
norm_counter = 0

pseudo_random = np.convolve(np.random.random([loop_len,1]).flatten(), np.ones(4)/4,'same')
pseudo_normal = np.convolve(np.random.normal(loc=0.0, scale=0.5, size=[loop_len,1]).flatten(), np.ones(4)/4,'same')
pass_percent = 0.45
sort_scale = reduced_img_size[1]*0.2;

# Loop until the user presses a key while the window is active
while True:
    disp_img = np.array(base_img,copy=True)
    sorted_pixels_mask = np.ndarray.astype((np.zeros(shape=(reduced_img_size[1], reduced_img_size[0], reduced_img_size[2]))),dtype='uint8')

    # Perform sorting on reduced_img. Image range does not include the outer 1 pixel border.
    for y in range(1, reduced_img_size[1]-2):
        # Skip columns that don't have any threshold pixels
        if (np.sum(edge_img[:,y]) > 0):
            # Randomize the sort length for every column. Make all lengths even to make it easier to scale
            sort_len = int(np.round(sort_scale*np.abs(pseudo_normal[norm_counter])))
            if((sort_len % 2) != 0):
                sort_len = sort_len - 1

            for x in range(1, reduced_img_size[0]-2):
                # Don't sort more than pass_percent of the threshold pixels for any given frame
                if ((edge_img[x,y] == 1) and (pseudo_random[rand_counter] <= pass_percent)):
                    # Check that you are not going to collide into the edge of the image, shorten sort_len if needed.
                    if((x + sort_len) > (reduced_img_size[0] - 1)):
                        sort_len = (reduced_img_size[0] - x) - 1

                    # Skip this edge pixel if the length will end up being 0 or 1
                    if (sort_len < 2):
                        continue

                    # Store and sort 
                    sorted_pixels = np.zeros(shape=(sort_len, 1, reduced_img_size[2]))
                    
                    # Add one to the range since n:m is actually n to (m-1)
                    sorted_pixels[:,0,0] = reduced_img[x:(x +sort_len), y, 0]
                    sorted_pixels[:,0,1] = reduced_img[x:(x +sort_len), y, 1]
                    sorted_pixels[:,0,2] = reduced_img[x:(x +sort_len), y, 2]
                    
                    # Sort the pixels along their y axis
                    sorted_pixels = np.sort(sorted_pixels, axis=0)

                    # Map the pixels to their postion in the reduced image size
                    sorted_pixels_mask[x:(x+sort_len), y, 0] = sorted_pixels[:,0,0]
                    sorted_pixels_mask[x:(x+sort_len), y, 1] = sorted_pixels[:,0,1]
                    sorted_pixels_mask[x:(x+sort_len), y, 2] = sorted_pixels[:,0,2]

                # Progress the random counter
                rand_counter = (rand_counter + 1) % loop_len

    # Upscale the sorted pixel map
    # upscaled_sorted_pixel_mask = cv2.resize(sorted_pixels_mask, dsize=(org_img_size[1], org_img_size[0]), interpolation=cv2.INTER_AREA)
    upscaled_sorted_pixel_mask = cv2.resize(sorted_pixels_mask, dsize=(org_img_size[1], org_img_size[0]), interpolation=cv2.INTER_NEAREST)


    # Replace the pixels
    np.copyto(dst=disp_img[:,:,0], src=upscaled_sorted_pixel_mask[:,:,0], where=(upscaled_sorted_pixel_mask[:,:,0] > 0))
    np.copyto(dst=disp_img[:,:,1], src=upscaled_sorted_pixel_mask[:,:,1], where=(upscaled_sorted_pixel_mask[:,:,1] > 0))
    np.copyto(dst=disp_img[:,:,2], src=upscaled_sorted_pixel_mask[:,:,2], where=(upscaled_sorted_pixel_mask[:,:,2] > 0))

    norm_counter = (norm_counter + 1) % loop_len
    
    # Display the image
    cv2.imshow(window_name, disp_img)
    
    # Allow person to exit by pressing a key. 3ms is approximately 30fps
    if not (cv2.waitKey(3) == -1):
        break

cv2.destroyWindow(window_name)