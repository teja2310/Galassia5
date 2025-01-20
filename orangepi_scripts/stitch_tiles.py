import os
import cv2
import numpy as np


def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1_inter = max(box1[1], box2[1])
    y1_inter = max(box1[2], box2[2])
    x2_inter = min(box1[3], box2[3])
    y2_inter = min(box1[4], box2[4])

    # Compute the area of intersection
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    box1_area = (box1[3] - box1[1]) * (box1[4] - box1[2])
    box2_area = (box2[3] - box2[1]) * (box2[4] - box2[2])

    # Compute IoU
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def filter_labels(labels, confidence_threshold=0.9):
    # Filter boxes with IoU > 90% and retain the one with higher confidence
    filtered_labels = []
    labels = sorted(labels, key=lambda x: x[5], reverse=True)  # Sort by confidence score

    for i, box1 in enumerate(labels):
        keep = True
        for j, box2 in enumerate(filtered_labels):
            if calculate_iou(box1, box2) > confidence_threshold:
                # Keep the box with the higher confidence score
                if box1[5] > box2[5]:
                    filtered_labels[j] = box1  # Replace box2 with box1
                keep = False
                break
        if keep:
            filtered_labels.append(box1)
    return filtered_labels

def reconstruct_image(image_folder, label_folder, label_filename, orig_width, orig_height, tile_size=800, overlap=0):
    # Determine the number of rows and columns in the original image
    num_cols = int(np.ceil(orig_width / tile_size))
    num_rows = int(np.ceil(orig_height / tile_size))

    # Create an empty array to hold the reconstructed image
    reconstructed_image = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)

    for i in range(num_cols):
        for j in range(num_rows):
            # Calculate the top-left corner of where this tile should go
            x_start = i * tile_size
            y_start = j * tile_size

            # Read the image tile
            tile_path = os.path.join(image_folder, label_filename +  f"_{i}_{j}.jpg")
            print(tile_path)
            tile = cv2.imread(tile_path)

            # Handle overlaps in last row/column
            x_end = min(x_start + tile_size, orig_width)
            y_end = min(y_start + tile_size, orig_height)
            # Handle the overlap region if this is in the last row or column
            if i == num_cols-1:
                tile = tile[:, -(x_end - x_start):]
            if j == num_rows-1:
                tile = tile[-(y_end - y_start):, :]  # Crop overlap vertically
                
            reconstructed_image[y_start:y_end, x_start:x_end] = tile

    # Translate labels to the reconstructed image
    labels = []

    for i in range(num_cols):
        for j in range(num_rows):
            label_path = os.path.join(label_folder, label_filename +  f"_{i}_{j}.txt")
            
            # Calculate the offset for each label based on the tile's position
            x_offset = i * tile_size
            y_offset = j * tile_size

            x_end = min(x_start + tile_size, orig_width)
            y_end = min(y_start + tile_size, orig_height)
            
            with open(label_path, 'r') as f:
                for line in f:
                    label_data = line.strip().split()
                    class_id = int(label_data[0])
                    x_min, y_min, x_max, y_max, confidence = map(float, label_data[1:])

                    # Translate bounding box coordinates
                    x_min += x_offset
                    x_max += x_offset
                    y_min += y_offset
                    y_max += y_offset
                    if (i == num_cols-1):
                        if x_min >= x_end - (x_end - x_start):
                            continue
                    if (j == num_rows-1):
                        if y_min >= y_end - (y_end - y_start):
                            continue

                    # Append the translated label including the confidence score
                    labels.append([class_id, x_min, y_min, x_max, y_max, confidence])

    # Filter labels to keep only those with IoU < 90% or higher confidence
    filtered_labels = filter_labels(labels)

    return reconstructed_image, filtered_labels

def main() -> None:
    # Usage example:
    image_folder = '../blindtest-america/pipeline_test/images'
    label_folder = '../gi_training/data800/valid/better'
    # read file to get size
    with open('../blindtest-america/pipeline_test/labeled_jpg/img_size.txt', 'r') as f:
        orig_height, orig_width = map(int, f.readline().strip().split())
    print(orig_width, orig_height)
    reconstructed_image, translated_labels = reconstruct_image(image_folder, label_folder, orig_width, orig_height)

    # Saving the reconstructed image and translated labels
    cv2.imwrite('../blindtest-america/pipeline_test/reconstructed_image.jpg', reconstructed_image)
    with open('../blindtest-america/pipeline_test/translated_labels.txt', 'w') as f:
        for label in translated_labels:
            f.write(f"{label[0]} {label[1]:.2f} {label[2]:.2f} {label[3]:.2f} {label[4]:.2f} {label[5]:.2f}\n")

if __name__ == "__main__":
    main()
