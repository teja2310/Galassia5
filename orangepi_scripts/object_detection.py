#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
import queue
import threading
from PIL import Image
from typing import List
from object_detection_utils import ObjectDetectionUtils
from stitch_tiles import reconstruct_image
import cv2

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_input_images, validate_images, divide_list_to_batches


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detection Example")
    parser.add_argument(
        "-n", "--net", 
        help="Path for the network in HEF format.",
        default="yolov7.hef"
    )
    parser.add_argument(
        "-i", "--input", 
        default="zidane.jpg",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size", 
        default=1,
        type=int,
        required=False,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-l", "--labels", 
        default="coco.txt",
        help="Path to a text file containing labels. If no labels file is provided, coco2017 will be used."
    )

    parser.add_argument(
        "-o", "--output_path", 
        help="Path generated image outputs.",
        default="./output_images"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path not found: {args.input}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    return args


def enqueue_images(
    images: List[Image.Image],
    batch_size: int,
    input_queue: queue.Queue,
    width: int,
    height: int,
    utils: ObjectDetectionUtils
) -> None:
    """
    Preprocess and enqueue images into the input queue as they are ready.

    Args:
        images (List[Image.Image]): List of PIL.Image.Image objects.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
        utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
    """
    for batch in divide_list_to_batches(images, batch_size):
        processed_batch = []
        batch_array = []

        for image in batch:
            #processed_image = utils.preprocess(image, width, height)
            processed_batch.append(image)
            batch_array.append(np.array(image))
        input_queue.put(processed_batch)

    input_queue.put(None)  # Add sentinel value to signal end of input


def process_output(
    output_queue: queue.Queue,
    output_path: Path,
    width: int,
    height: int,
    utils: ObjectDetectionUtils,
    names: List[str]
) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        output_path (Path): Path to save the output images.
        width (int): Image width.
        height (int): Image height.
        utils (ObjectDetectionUtils): Utility class for object detection visualization.
    """
    image_id = 0
    for name in names:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received

        processed_image, infer_results = result
        
        detections = utils.extract_detections(infer_results)

        # Deals with the expanded results from hailort versions < 4.19.0
        #if len(infer_results) == 1:
            #infer_results = infer_results[0]

        #detections = utils.extract_detections(infer_results)
        dets = utils.visualize(
            detections, processed_image, image_id,
            output_path, width, height
        )
        image_id += 1
        print(dets)
        print('./' + str(output_path) + '/output_labels/' + name + '.txt')
        with open('./' + str(output_path) + '/output_labels/' + name + '.txt', 'w+') as f:
            for det in dets:
                f.write(f"0 {det[0]:.2f} {det[1]:.2f} {det[2]:.2f} {det[3]:.2f} {det[4]:.2f}\n")

    output_queue.task_done()  # Indicate that processing is complete


def infer(
    images: List[Image.Image],
    net_path: str,
    labels_path: str,
    batch_size: int,
    output_path: Path,
    names: List[str]
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.

    Args:
        images (List[Image.Image]): List of images to process.
        net_path (str): Path to the HEF model file.
        labels_path (str): Path to a text file containing labels.
        batch_size (int): Number of images per batch.
        output_path (Path): Path to save the output images.
    """
    utils = ObjectDetectionUtils(labels_path)

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size
    )
    height, width, _ = hailo_inference.get_input_shape()
    with open('./img_size.txt', 'w') as f:
        f.write(f"{width} {height}")

    enqueue_thread = threading.Thread(
        target=enqueue_images, 
        args=(images, batch_size, input_queue, width, height, utils)
    )
    process_thread = threading.Thread(
        target=process_output, 
        args=(output_queue, output_path, width, height, utils, names)
    )

    enqueue_thread.start()
    process_thread.start()

    hailo_inference.run()

    enqueue_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    process_thread.join()

    logger.info(
        f'Inference was successful! Results have been saved in {output_path}'
    )


def main() -> None:
    """
    Main function to run the script.
    """
    # Parse command line arguments
    args = parse_args()

    # Load input images
    images, names = load_input_images(args.input)

    # Validate images
    try:
        validate_images(images, args.batch_size)
    except ValueError as e:
        logger.error(e)
        return

    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    input_path = args.input.split('/')[0]
    # Start the inference
    infer(images, args.net, args.labels, args.batch_size, output_path, names)

    with open('./' + input_path + '/img_size.txt', 'r') as f:
        orig_height, orig_width = map(int, f.readline().strip().split())
    print(orig_width, orig_height)
    label_filename = names[0].split('_')[0]
    reconstructed_image, translated_labels = reconstruct_image(args.input, './' + str(output_path) + '/output_labels', label_filename, orig_width, orig_height)

    # Saving the reconstructed image and translated labels
    cv2.imwrite('./reconstructed_image.jpg', reconstructed_image)
    with open('./translated_labels.txt', 'w') as f:
        for label in translated_labels:
            f.write(f"{label[0]} {label[1]:.2f} {label[2]:.2f} {label[3]:.2f} {label[4]:.2f} {label[5]:.2f}\n")
    
    for i, (c, xmin, ymin, xmax, ymax, conf) in enumerate(translated_labels):
        cropped_image = reconstructed_image[int(ymin):int(ymax), int(xmin):int(xmax)]
        cv2.imwrite(f"./cropped/crop_{i+1}_{int(conf*100)}.jpg", cropped_image)

if __name__ == "__main__":
    main()
