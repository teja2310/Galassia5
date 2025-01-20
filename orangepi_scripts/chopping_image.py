from osgeo import gdal 
import numpy as np
from PIL import Image as im 
import cv2
import os
# Load YOLO annotations (example function - depends on your specific format)
def load_yolo_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            annotations.append((class_id, x_center, y_center, width, height))
    return annotations

# Save YOLO annotations for cropped images
def save_yolo_annotations(annotations, save_path):
    with open(save_path, 'w') as file:
        for ann in annotations:
            file.write(f"{ann[0]} {ann[1]} {ann[2]} {ann[3]} {ann[4]}\n")

# Translate and crop annotations
def translate_annotations(annotations, crop_x, crop_y, crop_width, crop_height, img_width, img_height):
    new_annotations = []
    for class_id, x_center, y_center, width, height in annotations:
        # Convert normalized center coordinates to absolute pixel values
        abs_x_center = x_center * img_width
        abs_y_center = y_center * img_height
        abs_width = width * img_width
        abs_height = height * img_height

        # Check if the annotation is within the cropped region
        if (crop_x <= abs_x_center <= crop_x + crop_width and
            crop_y <= abs_y_center <= crop_y + crop_height):
            # Adjust coordinates relative to the new cropped image
            new_x_center = (abs_x_center - crop_x) / crop_width
            new_y_center = (abs_y_center - crop_y) / crop_height
            new_width = abs_width / crop_width
            new_height = abs_height / crop_height
            new_annotations.append((class_id, new_x_center, new_y_center, new_width, new_height))
    return new_annotations

# Main function for processing images and annotations
def process_image_with_annotations(image_path, annotation_path, output_image, output_label, crop_size=800):
    # Load image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    # Load annotations
    annotations = load_yolo_annotations(annotation_path)
    
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

            # Translate annotations to the cropped image
            cropped_annotations = translate_annotations(annotations, crop_x, crop_y, crop_width, crop_height, img_width, img_height)

            # Save cropped image and corresponding annotations
            cropped_image_path = os.path.join(output_image, f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}_{j}.jpg")
            cropped_annotation_path = os.path.join(output_label, f"{os.path.splitext(os.path.basename(annotation_path))[0]}_{i}_{j}.txt")
            
            cv2.imwrite(cropped_image_path, cropped_img)
            save_yolo_annotations(cropped_annotations, cropped_annotation_path)


# Load YOLO annotations
def load_yolo_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            annotations.append((class_id, x_center, y_center, width, height))
    return annotations

# Function to save image with YOLO annotations
def save_image_with_annotations(image_path, annotation_path, output_folder):
    # Load image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    # Load annotations
    annotations = load_yolo_annotations(annotation_path)
    
    # Draw bounding boxes on the image
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        
        # Convert normalized coordinates to pixel values
        box_x = int((x_center - width / 2) * img_width)
        box_y = int((y_center - height / 2) * img_height)
        box_width = int(width * img_width)
        box_height = int(height * img_height)

        # Draw rectangle for bounding box
        cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 255), 2)

        # Put class ID label on the top-left corner of the bounding box
        label = f"Class: {int(class_id)}"
        cv2.putText(img, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Save annotated image to output folder
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Saved annotated image to {output_path}")


# Load YOLO annotations
def load_yolo_annotations(annotation_file):
    annotations = []
    with open(annotation_file, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            annotations.append((class_id, x_center, y_center, width, height))
    return annotations

# Function to save image with YOLO annotations
def save_image_with_annotations(image_path, annotation_path, output_folder):
    # Load image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    # Load annotations
    annotations = load_yolo_annotations(annotation_path)
    
    # Draw bounding boxes on the image
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        
        # Convert normalized coordinates to pixel values
        box_x = int((x_center - width / 2) * img_width)
        box_y = int((y_center - height / 2) * img_height)
        box_width = int(width * img_width)
        box_height = int(height * img_height)

        # Draw rectangle for bounding box
        cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 255), 2)

        # Put class ID label on the top-left corner of the bounding box
        label = f"Class: {int(class_id)}"
        cv2.putText(img, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Save annotated image to output folder
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Saved annotated image to {output_path}")



def main() -> None:

    ds = gdal.Open(r'../blindtest-america/EO_20230503_185913_0046201001_001_L1G_RGB_5m.tif')
    band1 = ds.GetRasterBand(1) # Red channel 
    band2 = ds.GetRasterBand(2) # Green channel 
    band3 = ds.GetRasterBand(3) # Blue channel
    b1 = band1.ReadAsArray() 
    b2 = band2.ReadAsArray() 
    b3 = band3.ReadAsArray() 
    img_size = b1.shape
    # save the image size in text file
    with open('../blindtest-america/pipeline_test/labeled_jpg/img_size.txt', 'w') as f:
        f.write(f"{img_size[0]} {img_size[1]}")
    img = np.dstack((b1, b2, b3))
    maxi = np.max(img)
    img_jpg = (img/maxi) * 255
    im.fromarray(img_jpg.astype(np.uint8)).save('../blindtest-america/pipeline_test/labeled_jpg/blindtest.jpg')

    output_image = "../blindtest-america/pipeline_test/images"
    output_label = "../blindtest-america/pipeline_test/labels"
    os.makedirs(output_image, exist_ok=True)
    os.makedirs(output_label, exist_ok=True)


    image_path = "../blindtest-america/pipeline_test/labeled_jpg/blindtest.jpg"
    label_path = "../blindtest-america/blindtest.txt"
    process_image_with_annotations(image_path, label_path, output_image, output_label)

    output_folder = '../blindtest-america/pipeline_test/visualise'
    for image in os.listdir("../blindtest-america/pipeline_test/images"):
        if not image.endswith(".jpg"):
            continue
        image_path = os.path.join("../blindtest-america/pipeline_test/images", image)
        annotation_path = os.path.join("../blindtest-america/pipeline_test/labels", os.path.splitext(image)[0] + ".txt")
        save_image_with_annotations(image_path, annotation_path, output_folder)
if __name__ == "__main__":
    main()
