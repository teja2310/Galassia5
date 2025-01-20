import cv2
# import matplotlib.pyplot as plt

def plot_bounding_boxes(image_path, boxes, color=(0, 255, 0), thickness=2):
    """
    Plots bounding boxes on a JPG image and displays it.

    Parameters:
    - image_path (str): Path to the JPG image.
    - boxes (list of lists): A list of bounding boxes, where each box is represented as [x_min, y_min, x_max, y_max].
    - labels (list of str, optional): A list of labels or class names for each box. Defaults to None.
    - color (tuple): Color of the bounding box in BGR format. Default is green.
    - thickness (int): Thickness of the bounding box lines. Default is 2.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Plot each bounding box
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box[:4])
        conf = box[4]
        
        # Draw the rectangle on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # Add label if provided
        label = f"Confidence: {conf}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x_min, y_min - label_height - baseline), (x_min + label_width, y_min), color, cv2.FILLED)
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite('./checking.jpg', image)

    # Convert the image to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # # Display the image with matplotlib
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_rgb)
    # plt.axis("off")
    # plt.show()

# Example usage
image_path = './reconstructed_image.jpg'
# read file to get label, xyxy, and confidence
with open('./translated_labels.txt', 'r') as f:
    boxes = []
    for line in f:
        label, x_min, y_min, x_max, y_max, confidence = map(float, line.strip().split())
        boxes.append([x_min, y_min, x_max, y_max, confidence])

plot_bounding_boxes(image_path, boxes)
