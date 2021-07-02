# Audio_PixelSort_Python
Enhanced Python-implementation of my repository "Audio_PixelSort". Sort locations have been moved from brightness thresholds to image edges using a Sobel edge detector. 

## Requirements 
* Tested on **Windows 10** (version 20H2)
* **Python** (3.8.5 used)
* **numpy** (1.19.3 used)
* **scipy** (1.4.1 used)
* **imutils** (0.5.4 used)

## Image sample
![Frame Example](https://raw.githubusercontent.com/gilbertyap/Audio_PixelSort_Python/main/samples/img_sample_1.jpg)
Example frame from running `img_pixel_sort.py` with `-t 65`

## Video Sample
Coming soon...

## Summary
Pixel sorting, which was popularized by Kim Asendorf, is an artistic rendition of an image based on portions of its image "sorted". This repository contains a Python interpretation of pixel sorting for both images and video.

The goal of this project was to port my Processing code that performed "audio-based" pixel sorting to Python which is a much more accesible coding language. The original code would sort pixels within a given image based semi-random thresholds based on the FFT of a piece of audio. This version of the code is split into two: an image-only mode and a video mode.

Both run on the same principle: 
1. Acquire an image
1. Scale down the image dimensions
1. Perform Sobel edge detection and thresholding
1. Generate a low-res image of pixels that have been sorted vertically along edges
1. Upscale the sorted image and replace the original image pixels with the sorted pixels

In image-only mode, the selection of edge pixels and the length of the sort are determined somewhat randomly. The selection of edge pixels is determined through thresholding a random uniform distribution, while the sort lengths are based on a random Gaussian distribution. 

Details on video mode coming soon...

## How to run
First set up the virtual environment
```
python -m venv virt_env
./virt_env/Scripts/activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

###Image-only:
```
python img_pixel_sort.py -f subdir/path/to/file.ext
```

Custom thresholding (any value from 0 to 255):
```
python img_pixel_sort.py -f subdir/path/to/file.ext -t 65
```