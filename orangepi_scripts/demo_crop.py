from osgeo import gdal 
import numpy as np
from PIL import Image as im 

import cv2
import os

# Main function for processing images and annotations
def process_image_with_annotations(image_path, output_image, crop_size=800):
    # Load image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    # Calculate the grid and process each crop
    x_steps = (img_width + crop_size - 1) // crop_size  # ceil division
    y_steps = (img_height + crop_size - 1) // crop_size
    
    for i in range(x_steps):
        for j in range(y_steps):
            # Determine the crop region
            crop_x = i * crop_size if i < x_steps - 1 else img_width - crop_size
            crop_y = j * crop_size if j < y_steps - 1 else img_height - crop_size
            crop_width = min(crop_size, img_width - crop_x)
            crop_height = min(crop_size, img_height - crop_y)

            # Crop the image
            cropped_img = img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

            # Save cropped image and corresponding annotations
            cropped_image_path = os.path.join(output_image, f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}_{j}.jpg")
            
            cv2.imwrite(cropped_image_path, cropped_img)

def tif_to_jpg(image_path, output_image):
    ds = gdal.Open(image_path)

    band1 = ds.GetRasterBand(1) # Red channel 
    band2 = ds.GetRasterBand(2) # Green channel 
    band3 = ds.GetRasterBand(3) # Blue channel
    b1 = band1.ReadAsArray() 
    b2 = band2.ReadAsArray() 
    b3 = band3.ReadAsArray() 

    img = np.dstack((b1, b2, b3))
    width, height = img.shape[:2]
    maxi = np.max(img)

    img_jpg = (img/maxi) * 255
    im.fromarray(img_jpg.astype(np.uint8)).save(output_image)
    with open('./img_size.txt', 'w') as f:
        f.write(f"{width} {height}")

def tif_to_cropped(image_path='./demo/demo.tif', output_image='./demo/demo.jpg', cropped_image_path= "./demo/images"):
    tif_to_jpg(image_path, output_image)
    os.makedirs(cropped_image_path, exist_ok=True)
    process_image_with_annotations(output_image, cropped_image_path)
